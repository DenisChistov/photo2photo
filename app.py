from __future__ import annotations

import io
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps

from model import CycleGANInference, load_cyclegan_checkpoint, translate_image


APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"

MODEL_CATALOG: dict[str, dict[str, str]] = {
    "Яблоко ↔ апельсин": {
        "path": "cyclegan_export.pt",
        "domain_a": "Яблоко",
        "domain_b": "Апельсин",
        "description": (
            "CycleGAN, обученный переводить изображения яблок "
            "в апельсины и обратно."
        ),
    },
    "Моне ↔ фотография": {
        "path": "cyclegan_export_monet.pt",
        "domain_a": "Картина Моне",
        "domain_b": "Фотография",
        "description": (
            "CycleGAN для перевода картин Моне в фотографии "
            "и фотографий в стиль Моне."
        ),
    },
}


@st.cache_resource(show_spinner="Загружаю модель в память…")
def get_model(checkpoint_path: str) -> CycleGANInference:
    path = Path(checkpoint_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Файл модели не найден: {path}. "
            "Проверьте, что checkpoint добавлен в папку models "
            "и отправлен в GitHub."
        )

    return load_cyclegan_checkpoint(path)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


# На CPU избыточное количество потоков иногда замедляет небольшую модель.
torch.set_num_threads(2)

st.set_page_config(
    page_title="CycleGAN Image Translator",
    page_icon="🎨",
    layout="wide",
)

st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {max-width: 1100px; padding-top: 2rem;}
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: .3rem;
    }
    .app-subtitle {
        color: #6b7280;
        margin-bottom: 1.4rem;
        line-height: 1.55;
    }
    div[data-testid="stImage"] img {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-title">CycleGAN: перевод изображений</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-subtitle">'
    "Загрузите изображение, выберите модель и направление преобразования. "
    "Модель работает непосредственно внутри Streamlit Community Cloud."
    "</div>",
    unsafe_allow_html=True,
)

selected_name = st.selectbox(
    "Модель",
    options=list(MODEL_CATALOG),
)
selected = MODEL_CATALOG[selected_name]

domain_a = selected["domain_a"]
domain_b = selected["domain_b"]
checkpoint_path = MODELS_DIR / selected["path"]

st.caption(selected["description"])

direction_label = st.radio(
    "Направление",
    options=[
        f"{domain_a} → {domain_b}",
        f"{domain_b} → {domain_a}",
    ],
    horizontal=True,
)
direction = (
    "a_to_b"
    if direction_label == f"{domain_a} → {domain_b}"
    else "b_to_a"
)

uploaded_file = st.file_uploader(
    "Исходное изображение",
    type=["jpg", "jpeg", "png", "webp"],
    help="Поддерживаются JPG, PNG и WebP.",
)

if uploaded_file is None:
    st.info("Загрузите изображение, чтобы запустить преобразование.")
    st.stop()

try:
    original = Image.open(uploaded_file)
    original = ImageOps.exif_transpose(original).convert("RGB")
except Exception as error:
    st.error(f"Не удалось прочитать изображение: {error}")
    st.stop()

left, right = st.columns(2)

with left:
    st.subheader("Исходное изображение")
    st.image(original, use_container_width=True)

run_inference = st.button(
    "Преобразовать изображение",
    type="primary",
    use_container_width=True,
)

if run_inference:
    try:
        model = get_model(str(checkpoint_path))

        with st.spinner("Выполняю инференс…"):
            result = translate_image(
                image=original,
                model=model,
                direction=direction,
            )

        with right:
            st.subheader("Результат")
            st.image(result, use_container_width=True)
            st.download_button(
                label="Скачать результат в PNG",
                data=image_to_png_bytes(result),
                file_name=f"cyclegan_{direction}.png",
                mime="image/png",
                use_container_width=True,
            )

        with st.expander("Техническая информация"):
            st.write(
                {
                    "checkpoint": str(checkpoint_path.relative_to(APP_DIR)),
                    "direction": direction,
                    "model_image_size": model.image_size,
                    "input_size": original.size,
                    "output_size": result.size,
                    "dataset_name": model.dataset_name,
                }
            )

    except Exception as error:
        st.error("Инференс завершился ошибкой.")
        st.exception(error)
