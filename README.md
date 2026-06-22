# CycleGAN для Streamlit Community Cloud

Проект работает без Hugging Face и без внешнего хранилища моделей.
Два checkpoint лежат непосредственно в GitHub-репозитории и клонируются
Streamlit Community Cloud вместе с приложением.

## Итоговая структура

```text
.
├── .streamlit/
│   └── config.toml
├── models/
│   ├── cyclegan_export.pt
│   └── cyclegan_export_monet.pt
├── .gitignore
├── app.py
├── model.py
├── requirements.txt
└── README.md
```

## Какие модели положить в папку models

```text
models/cyclegan_export.pt
models/cyclegan_export_monet.pt
```

Для приложения не нужны:

```text
apple2orange_full.pt
monet2photo_full.pt
apple2orange.pt
monet2photo.pt
```

Файлы `*_full.pt` предназначены для продолжения обучения. Файлы
`apple2orange.pt` и `monet2photo.pt` дублируют соответствующие экспортные
checkpoint.

## Сборка папки проекта из ноутбука

Поместите `prepare_repository.py` в каталог с файлами:

```text
cyclegan_export.pt
cyclegan_export_monet.pt
```

Затем выполните:

```bash
python prepare_repository.py
```

Скрипт скопирует модели в `models/` текущего проекта.

## Проверка локально

Рекомендуемая версия Python — 3.11.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Linux/macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Отправка в GitHub

Checkpoint весят больше лимита браузерной загрузки GitHub, поэтому
загружайте репозиторий через Git из терминала.

```bash
git init
git add .
git commit -m "Deploy CycleGAN Streamlit app"
git branch -M main
git remote add origin https://github.com/De4u/cyclegan-streamlit.git
git push -u origin main
```

Git может показать предупреждение о файлах больше 50 МиБ. Это допустимо:
каждый экспортный checkpoint меньше жёсткого лимита GitHub в 100 МиБ.

## Деплой

1. Откройте Streamlit Community Cloud.
2. Нажмите **Create app**.
3. Выберите репозиторий `De4u/cyclegan-streamlit`.
4. Выберите ветку `main`.
5. Укажите main file: `app.py`.
6. В Advanced settings выберите Python 3.11.
7. Нажмите **Deploy**.

Secrets не нужны.
