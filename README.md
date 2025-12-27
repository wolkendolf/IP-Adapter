# IP-Adapter implementation

**Казачков Даниил Иванович** _В описании все ссылки привязаны к файлам_

## Постановка задачи

Существует множество применений генеративных моделей: text2img, img2img -
основанных на GAN, VAE или diffusion models
((StyleGAN)[https://arxiv.org/abs/1812.04948],
(LatentVAE)[https://arxiv.org/abs/2510.15301],
(StableDiffusion)[https://arxiv.org/abs/2112.10752],
(DALLE-2)[https://openai.com/ru-RU/index/dall-e-2/]). Данный проект
рассматривается как учебный, чтобы попробовать завернуть модель в MLOps
пайплайн. Работа опирается на статью
(IP-Adapter)[https://github.com/tencent-ailab/IP-Adapter/tree/main], где авторы
предложили легковесный вариант (22M параметров) для совершенствования
text-to-image диффузионной модели за счет раздвоения cross-attention механизма
путем разделения слоев перекрестного внимания для текстовых элементов и
элементов изображений. Целью работы является создание MLOps обвязки вокруг
модели для обучения, логгирования и упаковки модели как сервиса. Сравнение будет
проводиться с базовым StableDiffusion без адаптера.

### Формат входных и выходных данных

Поскольку изображения в CLIP проходят через center crop, IP-Adapter лучше
работает с квадратными изображениями. Для иных форматов часть информации за
пределами центральной области может быть потеряна. Однако вы можете сделать
resize до 512x512. Вход:

- ImagePrompt в формате RGB, приводимое к размеру 512x512 (тензор 3x512x512
  после нормализации). TextPrompt на английском языке, описывающий стиль
  изображения, дополнительные артефакты (portait photo, studio lighting и т.п.)
  Выход:
- Сгенерированное изображение в формате RGB (тензор 3x512x512), которое
  соответствует одновременно и текстовому описанию, и сохраняет визуальную
  информацию из референсного изображения

### Метрики

- Косинусное сходство между CLIP-эмбеддингами референсного и сгенерированного
  изображения. Позволяет измерить, насколько хорошо сохраняется общий визуальный
  стиль и содержание (чем выше, тем лучше).
- FID (Frechet Inception Distance) между множеством сгенерированных портретов и
  реальными изображениями из тестовой выборки. Сравнение распределений признаков
  реальных и сгенерированных изображений (чем ниже, тем лучше). Сама по себе
  метрика далеко не идеально, но де-факто стандарт индустрии.

### Валидация и тест

В качестве валидации используем COCO2017 для того, чтобы избежать утечки данных
в train. Для валидации используем CLIP-I, CLIP-T.

- CLIP-I: the similarity in CLIP image embedding of generated images with the
  image prompt.
- CLIP-T: the CLIPScore of the generated images with captions of the image
  prompts. Проблемой может стать расчет метрик, так как при следовании стратегии
  авторов IP-Adapter придется для каждого изображения валидационной выборки
  генерировать по 4 изображения на каждый image-промпт (то есть суммарно 20
  тысяч картинок на метод). Далее по сгенерированным сэмплам считаем две метрики
  с CLIP ViT-L/14.

### Датасеты

Для обучения использовался датасет
(COYO-700M)[https://github.com/kakaobrain/coyo-dataset]. Это довольно старый
датасет, поэтому для обхода неработающих ссылок была добавлена предобработка.
Предоставляют файл метадаты в Apache Parquet формате.

Для валидации использовался датасет
(COCO2017)[https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset] имеет
около 200 000 изображений с аннотациями. Нас будет интересовать часть val2017: 5
000 изображений для валидации. Имеется 80 категорий объектов.

## Моделирование

### Бейзлайн

Предобученный StableDiffusion
((SD1.5)[https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5])
без IP-Adapter. Два режима text2img, img2img.

### Основная модель

IP-Adapter поверх Stable Diffusion. IP-Adapter реализует раздельные слои
кросс-внимания для текстовых и визуальных признаков: текст кодируется текстовым
энкодером (CLIP text encoder), референсное изображение с помощью CLIP image
encoder, а затем в UNet добавляется отдельная ветка кросс-внимания, которая
принимает визуальные признаки. Базовая диффузионная модель при этом заморожена,
обучаются только параметры адаптера (22M параметров), что делает обучение
гораздо дешевле.

## Setup

При обучении и инференсе использовалась одна машина с Nvidia H100 с cuda 12.8.
Для управления виртуальными окружениями используется
(conda)[https://www.anaconda.com/docs/getting-started/miniconda/install], для
управления зависимостями (uv)[https://docs.astral.sh/uv/].

```bash
conda create --name ip_adapter python=3.12
conda activate ip_adapter
```

Клонируйте репозиторий

```bash
git clone <URL_ВАШЕГО_РЕПОЗИТОРИЯ>.git
cd <ИМЯ_ПАПКИ_РЕПОЗИТОРИЯ>
```

Настроим conda и uv:

```bash
echo "$CONDA_PREFIX"
export VIRTUAL_ENV="$CONDA_PREFIX"
```

Опционально можно сменить расположение кэша uv в свою папку.

```bash
uv cache dir
mkdir -p /home/.../.uv-cache
export UV_CACHE_DIR=/home/.../.uv-cache
```

Чтобы не было ошибок при установке, а именно чтобы не использовать PyTorch index
для всех библиотек, добавьте следующие строки в `pyproject.toml`.

```markdown
[[tool.uv.index]] name = "pytorch-cu128" url =
"https://download.pytorch.org/whl/cu128" explicit = true

[tool.uv.sources] torch = { index = "pytorch-cu128" } torchvision = { index =
"pytorch-cu128" }
```

Ставим библиотеки

```bash
uv sync --active --frozen
```

## Train

<!-- Необходимо скачать данные с репозитория
```bash
to_do
```
Далее переведем датасет в подходящий для обучения IP-Adapter, вызвав функцию
```bash
python tools/coyo_wds_to_original.py \
  --wds_dir ../data/processed/coyo-700m-webdataset \
  --out_images ../data/processed/coyo_original/images \
  --out_jsonl ../data/processed/coyo_original/data.jsonl
```
и приведя `jsonl` в `json`:
```
python tools/jsonl_to_json_array.py \
  --in_jsonl ../data/processed/coyo_original/data.jsonl \
  --out_json ../data/processed/coyo_original/data.json
```
Дополнительно отфильтруем битые данные
```bash
python tools/check_data.py \
  --in_json ../data/processed/coyo_original/data.json \
  --image_root ../data/processed/coyo_original/images \
  --out_json ../data/processed/coyo_original/data_filtered.json \
  --bad_report ../data/processed/coyo_original/bad_samples.jsonl \
  --min_size 16 \
  --max_aspect 20
``` -->

Поскольку залить данные на удаленный remote не получилось, я воспользовался
локальным remote. Создать dvc:

```bash
cd path_to_proj
dvc init
dvc remote add -d data /home/jovyan/shares/SR008.fs2/nkiselev/sandbox/kazachkovdi/ipad_mlops_dvc_remote/data

# проверка
dvc remote list
cat .dvc/config
```

Трекаем датасет папкой и пушим в локальный remote

```bash
dvc add data/processed/coyo_original
dvc push -r data
```

Чтобы загрузить данные, выполните команду:

```bash
dvc pull -r data
```

В случае если доступ к локальному dvc remote не будет найден, то при запуске
обучения tutorial_train:

- не найдёт coyo_original
- попробует dvc pull (упадёт/не найдёт remote)
- вызовет download_data() и соберёт датасет из HF (что и требует задание для
  локального хранилища)

Запуск обучения

```bash
python tutorial_train.py
```

## Production preparation

После обучения мы экспортируем:

- `ip_adapter.bin` — компактные веса IP-Adapter (`image_proj` +
  `adapter_modules`) совместимые с официальным репозиторием IP-Adapter.
- `ip_adapter_unet.onnx` — ONNX модель одного диффузионного шага с обусловленным
  IP-Adapter:
  `noise_pred = UNet(noisy_latents, t, concat(text_hidden_states, ip_tokens(image_embeds)))`.
- `ip_adapter_unet.meta.json` — форма и экспорт метаданных.

Экспорт весов IP-Adapter (`ip_adapter.bin`)

```bash
python -m tools.production.export_ip_adapter_bin \
  sd-ip_adapter/lightning_checkpoints/last-v1.ckpt \
  sd-ip_adapter/production
```

ONNX export (важно: train_py — файл с классом IPAdapterLitModule)

```bash
python -m tools.production.export_onnx_ip_adapter_unet \
  sd-ip_adapter/lightning_checkpoints/last-v1.ckpt \
  sd-ip_adapter/production/ip_adapter_unet.onnx \
  --opset=18 \
  --device=cpu \
  --resolution=512 \
  --batch=1
```

Validate ONNX

```bash
python -m tools.production.validate_onnx_ip_adapter_unet \
  sd-ip_adapter/lightning_checkpoints/last-v1.ckpt \
  sd-ip_adapter/production/ip_adapter_unet.onnx \
  --device=cpu \
  --resolution=512 \
  --batch=1
```

## Infer

В этом разделе описывается, как запустить вывод результатов после обучения на
новых данных (подсказка в виде изображения + необязательная текстовая
подсказка). Код вывода результатов реализован в файле
`tools/infer_ip_adapter_generate.py`. Необходимые артефакты:

- натренированные веса `ip_adapter.bin` (обработано после обучения)
- энкодер изображений такой же, что и во время тренировки
- базовая Stable diffusion модель (например, `runwayml/stable-diffusion-v1-5`)
  Формат входных данных: jsonl.

```
{"id":"woman","image":"data/infer_examples/images/woman.png","prompt":"portrait photo","seed":42,"num_samples":4}
```

Запуск:

```bash
python infer_ip_adapter_generate.py \
  --ip_ckpt sd-ip_adapter/production/ip_adapter.bin \
  --image_encoder_path models/image_encoder \
  --requests_jsonl data/infer_examples/requests.jsonl \
  --out_dir outputs/infer \
  --device cuda \
  --num_tokens 4
```

Сгенрированные изображения будут лежать по адресу `outputs/infer/`.
