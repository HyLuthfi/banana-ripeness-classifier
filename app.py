import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import os
import h5py

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Banana Quality Classifier",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS TEMA GELAP (GLASSMORPHISM) ---
st.markdown("""
<style>
    /* Global Styling */
    .stApp {
        background: linear-gradient(135deg, #12141D 0%, #1A2035 100%);
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Typography */
    h1, h2, h3 { 
        text-align: center; 
        font-weight: 800; 
        color: #FFFFFF !important;
        text-shadow: 0 0 20px rgba(255,255,255,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Result Card (Glassmorphism) */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    /* Status Colors */
    .status-mentah { 
        color: #10B981; 
        font-size: 36px; 
        font-weight: 800; 
        text-shadow: 0 0 15px rgba(16, 185, 129, 0.4); 
    } 
    .status-matang { 
        color: #F59E0B; 
        font-size: 36px; 
        font-weight: 800; 
        text-shadow: 0 0 15px rgba(245, 158, 11, 0.4); 
    } 
    .status-terlalu { 
        color: #EF4444; 
        font-size: 36px; 
        font-weight: 800; 
        text-shadow: 0 0 15px rgba(239, 68, 68, 0.4); 
    } 
    
    .conf-label {
        font-size: 14px;
        color: #94A3B8;
        margin-top: 10px;
    }
    
    .conf-value {
        font-size: 24px;
        color: #FFFFFF;
        font-weight: bold;
    }
    
    /* Button Override */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        border: none;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.5);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI PEMBUATAN ARSITEKTUR MODEL ---
def inject_dense_weights(model, h5_path, expected_in):
    kernel_1 = None
    bias_1 = None
    kernel_2 = None
    bias_2 = None
    
    with h5py.File(h5_path, 'r') as f:
        g = f['model_weights'] if 'model_weights' in f else f
        
        def search_group(group):
            nonlocal kernel_1, bias_1, kernel_2, bias_2
            for k in group.keys():
                item = group[k]
                if isinstance(item, h5py.Group):
                    search_group(item)
                elif isinstance(item, h5py.Dataset):
                    if item.shape == (expected_in, 128):
                        kernel_1 = item[:]
                        for sub_k in group.keys():
                            if group[sub_k].shape == (128,):
                                bias_1 = group[sub_k][:]
                                break
                    elif item.shape == (128, 3):
                        kernel_2 = item[:]
                        for sub_k in group.keys():
                            if group[sub_k].shape == (3,):
                                bias_2 = group[sub_k][:]
                                break
        
        search_group(g)
        
    if kernel_1 is not None and bias_1 is not None:
        model.layers[-3].set_weights([kernel_1, bias_1])
    if kernel_2 is not None and bias_2 is not None:
        model.layers[-1].set_weights([kernel_2, bias_2])

def get_augmentation_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.1)
    ], name="Augmentasi_On_The_Fly")

@st.cache_resource
def load_mobilenet_v3():
    try:
        data_augmentation = get_augmentation_layer()
        base_model = tf.keras.applications.MobileNetV3Large(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = data_augmentation(inputs)
        x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d_1')(x)
        x = tf.keras.layers.Dense(128, activation='relu', name='dense_2')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs, name="MobileNetV3_Large")
        inject_dense_weights(model, "models/best_mobilenet_pisang.h5", 960)
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_convnext_tiny():
    try:
        data_augmentation = get_augmentation_layer()
        base_model = tf.keras.applications.ConvNeXtTiny(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = data_augmentation(inputs)
        x = tf.keras.applications.convnext.preprocess_input(x)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d')(x)
        x = tf.keras.layers.Dense(128, activation='relu', name='dense')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs, name="ConvNeXt_Tiny")
        inject_dense_weights(model, "models/best_convnext_pisang.h5", 768)
        return model, None
    except Exception as e:
        return None, str(e)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3014/3014491.png", width=100)
    st.title("⚙️ Tentang Proyek")
    st.markdown("Sistem berbasis *Deep Learning* ini mengkomparasikan dua arsitektur mutakhir untuk mengklasifikasikan pisang ke dalam tiga fase kematangan.")
    st.markdown("---")
    st.markdown("**1. ConvNeXt-Tiny**")
    st.markdown("Arsitektur pure-convolutional modern (Akurasi Uji: 98%).")
    st.markdown("**2. MobileNetV3-Large**")
    st.markdown("Arsitektur super ringan & efisien (Akurasi Uji: 100%).")

# Definisi Kelas
CLASS_NAMES = ['matang', 'mentah', 'terlalu_matang']

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)
    return img_array

def get_color_class(label):
    if label == 'mentah': return 'status-mentah'
    elif label == 'matang': return 'status-matang'
    else: return 'status-terlalu'

def format_label(label):
    return label.replace('_', ' ').title()

# --- HEADER UTAMA ---
st.markdown("<h1>🍌 Banana Quality Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Sistem Pintar Komparasi Model Kematangan Pisang Berbasis AI</p>", unsafe_allow_html=True)
st.write("")

# --- UPLOAD & PREVIEW ---
uploaded_file = st.file_uploader("Upload Foto Pisang Anda (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col_img, col_btn = st.columns([1, 2])
    
    with col_img:
        st.image(image, use_column_width=True, caption="Citra Input (Real-Time)")
        
    with col_btn:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("🔍 Mulai Prediksi Head-to-Head", use_container_width=True):
            with st.spinner("Memproses melalui Neural Networks..."):
                # Load both models
                model_mobilenet, err_mob = load_mobilenet_v3()
                model_convnext, err_conv = load_convnext_tiny()
                
                if model_mobilenet is None or model_convnext is None:
                    st.error(f"Sistem Gagal Memuat Model.")
                else:
                    img_tensor = preprocess_image(image)
                    
                    # ConvNeXt Prediction
                    start_time = time.time()
                    pred_c = model_convnext.predict(img_tensor, verbose=0)
                    calc_time_c = time.time() - start_time
                    class_c = CLASS_NAMES[np.argmax(pred_c[0])]
                    conf_c = np.max(pred_c[0]) * 100
                    color_c = get_color_class(class_c)
                    
                    # MobileNet Prediction
                    start_time = time.time()
                    pred_m = model_mobilenet.predict(img_tensor, verbose=0)
                    calc_time_m = time.time() - start_time
                    class_m = CLASS_NAMES[np.argmax(pred_m[0])]
                    conf_m = np.max(pred_m[0]) * 100
                    color_m = get_color_class(class_m)
                    
                    # Result Display
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        st.markdown(f"""
<div class="result-card">
<div style="font-size: 14px; color: #94A3B8; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 5px;">
Model: ConvNeXt-Tiny
</div>
<div style="font-size: 16px; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;">
Status Kematangan
</div>
<div class="{color_c}">{format_label(class_c)}</div>
<div style="margin-top: 30px; display: flex; justify-content: space-around;">
<div>
<div class="conf-label">Tingkat Kepercayaan</div>
<div class="conf-value">{conf_c:.2f}%</div>
</div>
<div>
<div class="conf-label">Waktu Komputasi</div>
<div class="conf-value">{calc_time_c:.3f} s</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

                    with col_res2:
                        st.markdown(f"""
<div class="result-card">
<div style="font-size: 14px; color: #94A3B8; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 5px;">
Model: MobileNetV3-Large
</div>
<div style="font-size: 16px; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;">
Status Kematangan
</div>
<div class="{color_m}">{format_label(class_m)}</div>
<div style="margin-top: 30px; display: flex; justify-content: space-around;">
<div>
<div class="conf-label">Tingkat Kepercayaan</div>
<div class="conf-value">{conf_m:.2f}%</div>
</div>
<div>
<div class="conf-label">Waktu Komputasi</div>
<div class="conf-value">{calc_time_m:.3f} s</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)
