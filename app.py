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

# --- CSS TEMA PUTIH & SIMETRIS (TANPA EMOJI) ---
st.markdown("""
<style>
    /* Global Styling */
    .stApp {
        background-color: #FFFFFF;
        color: #111827;
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
    }
    
    /* Header Alignment */
    h1, h2, h3 { 
        text-align: center; 
        font-weight: 800; 
        color: #111827 !important;
    }
    .subtitle {
        text-align: center;
        color: #6B7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Centered Tabs */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #4B5563;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #111827 !important;
        border-bottom: 3px solid #111827 !important;
        background-color: #F9FAFB;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 16px;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .model-name {
        font-size: 14px;
        color: #6B7280;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .result-value {
        font-size: 28px;
        font-weight: 800;
        margin: 10px 0;
    }
    .confidence-val {
        font-size: 14px;
        color: #9CA3AF;
        font-weight: 500;
    }
    
    /* Status Colors */
    .status-mentah { color: #16A34A; } 
    .status-matang { color: #D97706; } 
    .status-terlalu { color: #DC2626; } 
    
    /* Image Display */
    .uploaded-image {
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin: 20px auto;
        display: block;
        border: 4px solid #F3F4F6;
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
        model.layers[-2].set_weights([kernel_1, bias_1])
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

# Inisialisasi Model
model_mobilenet, err_mob = load_mobilenet_v3()
model_convnext, err_conv = load_convnext_tiny()

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
st.markdown("<h1>Banana Quality Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analisis Kematangan Pisang Otomatis Berbasis Deep Learning</p>", unsafe_allow_html=True)

# --- NAVIGASI TABS ---
tab_predict, tab_method = st.tabs(["Prediksi Head-to-Head", "Metodologi Riset"])

with tab_predict:
    st.write("") # Spacer
    uploaded_file = st.file_uploader("Upload Foto Pisang (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col_img, col_res = st.columns([1, 2])
        
        with col_img:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_column_width=True, caption="Citra Input")
            
        with col_res:
            st.markdown("<br>", unsafe_allow_html=True)
            if model_mobilenet is None or model_convnext is None:
                st.error("Sistem Gagal Memuat Model.")
                if err_mob: st.code(f"Error MobileNet: {err_mob}")
                if err_conv: st.code(f"Error ConvNeXt: {err_conv}")
                st.info("Penyebab Umum: File .h5 rusak (LFS pointer issue di Streamlit Cloud) atau versi TensorFlow tidak cocok.")
            else:
                with st.spinner("Memproses melalui Neural Networks..."):
                    img_tensor = preprocess_image(image)
                    
                    # Prediksi ConvNeXt
                    start_conv = time.time()
                    pred_conv = model_convnext.predict(img_tensor, verbose=0)
                    time_conv = time.time() - start_conv
                    class_conv = CLASS_NAMES[np.argmax(pred_conv[0])]
                    conf_conv = np.max(pred_conv[0]) * 100
                    
                    # Prediksi MobileNet
                    start_mob = time.time()
                    pred_mob = model_mobilenet.predict(img_tensor, verbose=0)
                    time_mob = time.time() - start_mob
                    class_mob = CLASS_NAMES[np.argmax(pred_mob[0])]
                    conf_mob = np.max(pred_mob[0]) * 100
                    
                    # Layout Head-to-Head Simetris
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        color_conv = get_color_class(class_conv)
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="model-name">ConvNeXt-Tiny (98% Acc)</div>
                            <div class="result-value {color_conv}">{format_label(class_conv)}</div>
                            <div class="confidence-val">Tingkat Kepercayaan: {conf_conv:.1f}%</div>
                            <div class="confidence-val" style="font-size:11px; margin-top:5px;">Waktu Komputasi: {time_conv:.3f} dtk</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col2:
                        color_mob = get_color_class(class_mob)
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="model-name">MobileNetV3-Large (100% Acc)</div>
                            <div class="result-value {color_mob}">{format_label(class_mob)}</div>
                            <div class="confidence-val">Tingkat Kepercayaan: {conf_mob:.1f}%</div>
                            <div class="confidence-val" style="font-size:11px; margin-top:5px;">Waktu Komputasi: {time_mob:.3f} dtk</div>
                        </div>
                        """, unsafe_allow_html=True)

with tab_method:
    st.markdown("""
    ### Arsitektur Neural Network
    Proyek ini mengkomparasikan dua arsitektur mutakhir dalam Computer Vision:
    
    1. **ConvNeXt-Tiny**: Arsitektur pure-convolutional modern yang dioptimalkan dengan teknik training ala Vision Transformers (ViT). Memiliki parameter yang cukup ringan namun performa sangat kompetitif.
    2. **MobileNetV3-Large**: Arsitektur ringan yang didesain menggunakan Hardware-Aware Network Architecture Search (NAS). Sangat efisien untuk deployment di perangkat berspesifikasi rendah dengan latensi minimal.
    
    ### Skema Dataset & Augmentasi
    - Dataset terdiri dari tiga kelas: **Mentah (Green)**, **Matang (Yellow)**, dan **Terlalu Matang (Spotted/Brown)**.
    - Menggunakan teknik **On-The-Fly Data Augmentation** (Random Flip, Rotation, Zoom, Contrast) sebagai lapisan layer pertama dalam model untuk memastikan AI kebal terhadap variasi background dan posisi objek.
    
    ### Evaluasi (Berdasarkan Uji Validasi)
    - **ConvNeXt-Tiny**: Akurasi **98%**
    - **MobileNetV3-Large**: Akurasi **100%**
    - Kedua model berhasil mendeteksi fitur visual kematangan pisang dengan recall dan precision di atas 0.95 pada setiap kelas.
    """)
