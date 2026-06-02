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
    page_icon="🍌🍅",
    layout="wide"
)

# ==========================================
# BUILD MODEL PISANG
# ==========================================
def build_banana_model():

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)
    )

    base_model.trainable = False

    model = Sequential([

        base_model,

        GlobalAveragePooling2D(),

        BatchNormalization(),

        Dense(128, activation='relu'),

        Dropout(0.4),

        Dense(4, activation='softmax')

    ])

    return model


# ==========================================
# BUILD MODEL TOMAT
# ==========================================
def build_tomato_model():

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)
    )

    base_model.trainable = False

    model = Sequential([

        base_model,

        GlobalAveragePooling2D(),

        BatchNormalization(),

        Dropout(0.3),

        Dense(128, activation='relu'),

        Dropout(0.2),

        Dense(3, activation='softmax')

    ])

    return model


# ==========================================
# LOAD MODEL PISANG
# ==========================================
@st.cache_resource
def load_banana_model():
    model = build_banana_model()
    model.load_weights(
        "model_manual_augmentation.weights.h5"
    )
    return model


banana_model = load_banana_model()

banana_classes = [
    'Terlalu Matang',
    'Matang',
    'Busuk',
    'Belum Matang'
]


# ==========================================
# LOAD MODEL TOMAT
# ==========================================
@st.cache_resource
def load_tomato_model():
    model = build_tomato_model()
    model.load_weights(
        "tomato4.weights.h5"
    )
    return model


tomato_model = load_tomato_model()

tomato_classes = [
    'Tidak Layak',
    'Matang',
    'Belum Matang'
]


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("🍌🍅Pilih Model")

selected_model = st.sidebar.selectbox(
    "Jenis Buah",
    ["🍌Pisang", "🍅Tomat"]
)


# ==========================================
# TITLE
# ==========================================
st.title("🍌🍅Klasifikasi Kematangan Buah")


# ==========================================
# INPUT GAMBAR
# ==========================================
input_method = st.radio(
    "Pilih Sumber Gambar",
    ["Upload Gambar", "Ambil dari Kamera"]
)

uploaded_file = None
camera_file = None

if input_method == "Upload Gambar":
    uploaded_file = st.file_uploader(
        "Upload Gambar",
        type=["jpg", "jpeg", "png"]
    )

else:
    camera_file = st.camera_input(
        "Ambil Gambar dari Kamera"
    )


# ==========================================
# PREDICTION
# ==========================================
image_source = uploaded_file if uploaded_file is not None else camera_file

if image_source is not None:

    # ==========================
    # LOAD IMAGE
    # ==========================
    img = Image.open(image_source).convert("RGB")

    st.image(
        img,
        caption="Gambar Upload",
        width="stretch"
    )

    # resize
    img = img.resize((224,224))

    # convert ke array
    img_array = np.array(img)

    # ======================================
    # MODEL PISANG
    # ======================================
    if selected_model == "🍌Pisang":

        # preprocessing sesuai training
        img_array = img_array / 255.0

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # predict
        prediction = banana_model.predict(
            img_array
        )

        predicted_class = np.argmax(
            prediction
        )

        confidence = np.max(
            prediction
        )

        st.subheader(
            "Hasil Prediksi Pisang 🍌"
        )

        # confidence threshold
        if confidence < 0.80:

            st.error(
                "Gambar tidak dapat dideteksi"
            )

        else:

            st.success(
                banana_classes[predicted_class]
            )

            st.write(
                f"Confidence: {confidence*100:.2f}%"
            )

    # ======================================
    # MODEL TOMAT
    # ======================================
    else:

        # preprocessing sesuai training
        img_array = preprocess_input(
            img_array
        )

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # predict
        prediction = tomato_model.predict(
            img_array
        )

        predicted_class = np.argmax(
            prediction
        )

        confidence = np.max(
            prediction
        )

        st.subheader(
            "Hasil Prediksi Tomat 🍅"
        )

        # confidence threshold
        if confidence < 0.80:

            st.error(
                "Gambar tidak dapat dideteksi"
            )

        else:

            st.success(
                tomato_classes[predicted_class]
            )

            st.write(
                f"Confidence: {confidence*100:.2f}%"
            )
