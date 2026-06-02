import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dropout,
    Dense,
    BatchNormalization
)

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Klasifikasi Kematangan Buah",
    page_icon="🍌",
    layout="wide"
)


# ==========================================
# LOAD CSS
# ==========================================
def load_css(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


load_css("style.css")


# ==========================================
# BUILD MODEL PISANG
# ==========================================
def build_banana_model():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(4, activation="softmax")
    ])

    return model


# ==========================================
# BUILD MODEL TOMAT
# ==========================================
def build_tomato_model():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(3, activation="softmax")
    ])

    return model


# ==========================================
# LOAD MODEL WITH CACHE
# ==========================================
@st.cache_resource
def load_banana_model():
    model = build_banana_model()
    model.load_weights("model_manual_augmentation.weights.h5")
    return model


@st.cache_resource
def load_tomato_model():
    model = build_tomato_model()
    model.load_weights("tomato4.weights.h5")
    return model


banana_model = load_banana_model()
tomato_model = load_tomato_model()


banana_classes = [
    "overripe",
    "ripe",
    "rotten",
    "unripe"
]

tomato_classes = [
    "reject",
    "ripe",
    "unripe"
]


# ==========================================
# HELPER FUNCTION
# ==========================================
def translate_label(label):
    label_map = {
        "overripe": "Terlalu Matang",
        "ripe": "Matang",
        "rotten": "Busuk",
        "unripe": "Belum Matang",
        "reject": "Tidak Layak / Reject"
    }

    return label_map.get(label, label)


def get_result_emoji(label):
    emoji_map = {
        "overripe": "🟤",
        "ripe": "✅",
        "rotten": "❌",
        "unripe": "🟢",
        "reject": "⚠️"
    }

    return emoji_map.get(label, "🍎")


def predict_image(img, selected_model):
    img = img.resize((224, 224))
    img_array = np.array(img)

    if selected_model == "Pisang":
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = banana_model.predict(img_array)
        class_names = banana_classes

    else:
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = tomato_model.predict(img_array)
        class_names = tomato_classes

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    label = class_names[predicted_class]

    return label, confidence


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">🍌 Fruit Classifier</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-caption">Aplikasi klasifikasi kematangan buah berbasis MobileNetV2</div>',
        unsafe_allow_html=True
    )

    st.divider()

    selected_model = st.selectbox(
        "Pilih Jenis Buah",
        ["Pisang", "Tomat"]
    )

    threshold = st.slider(
        "Confidence Minimum",
        min_value=0.50,
        max_value=0.99,
        value=0.80,
        step=0.01
    )

    st.divider()

    st.markdown(
        """
        <div class="info-box">
        <b>Tips hasil lebih akurat:</b><br>
        Gunakan gambar yang terang, objek terlihat jelas, dan latar belakang tidak terlalu ramai.
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# HEADER
# ==========================================
st.markdown(
    '<div class="main-title">🍅 Klasifikasi Kematangan Buah 🍌</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Upload gambar atau ambil foto langsung dari kamera untuk mendeteksi tingkat kematangan buah.</div>',
    unsafe_allow_html=True
)


# ==========================================
# MAIN LAYOUT
# ==========================================
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)

    st.subheader("📷 Input Gambar")

    input_method = st.radio(
        "Pilih sumber gambar",
        ["Upload Gambar", "Ambil dari Kamera"],
        horizontal=True
    )

    uploaded_file = None
    camera_file = None

    if input_method == "Upload Gambar":
        uploaded_file = st.file_uploader(
            "Upload gambar buah",
            type=["jpg", "jpeg", "png"]
        )
    else:
        camera_file = st.camera_input(
            "Ambil gambar dari kamera"
        )

    image_source = uploaded_file if uploaded_file is not None else camera_file

    st.markdown("</div>", unsafe_allow_html=True)


with right_col:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)

    st.subheader("📌 Informasi Model")

    if selected_model == "Pisang":
        st.write("Model aktif: **Klasifikasi Kematangan Pisang**")
        st.write("Kelas prediksi:")
        st.write("🟤 Terlalu Matang | ✅ Matang | ❌ Busuk | 🟢 Belum Matang")
    else:
        st.write("Model aktif: **Klasifikasi Kematangan Tomat**")
        st.write("Kelas prediksi:")
        st.write("⚠️ Reject | ✅ Matang | 🟢 Belum Matang")

    st.write(f"Confidence minimum: **{threshold*100:.0f}%**")

    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# IMAGE PREVIEW AND PREDICTION
# ==========================================
if image_source is not None:
    st.markdown("### 🖼️ Preview dan Hasil Prediksi")

    preview_col, result_col = st.columns([1, 1], gap="large")

    with preview_col:
        img = Image.open(image_source).convert("RGB")

        st.image(
            img,
            caption="Gambar yang Dipilih",
            use_container_width=True
        )

    with result_col:
        with st.spinner("Sedang memproses gambar..."):
            label, confidence = predict_image(img, selected_model)

        if confidence < threshold:
            st.markdown(
                """
                <div class="warning-card">
                    Gambar tidak dapat dideteksi dengan yakin.<br>
                    Coba gunakan gambar yang lebih jelas.
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write(f"Confidence: **{confidence*100:.2f}%**")
            st.progress(float(confidence))

        else:
            emoji = get_result_emoji(label)
            translated_label = translate_label(label)

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">{emoji} {translated_label}</div>
                    <div class="confidence-text">Confidence: {confidence*100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(float(confidence))

            st.success(
                f"Hasil prediksi {selected_model}: {translated_label}"
            )

else:
    st.info("Silakan upload gambar atau ambil gambar dari kamera terlebih dahulu.")


# ==========================================
# FOOTER
# ==========================================
st.markdown(
    '<div class="footer">Dibuat dengan Streamlit dan TensorFlow</div>',
    unsafe_allow_html=True
)
