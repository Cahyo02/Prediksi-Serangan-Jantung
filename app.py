import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Serangan Jantung",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #F0F8FF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4A90E2;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF8DC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FFB347;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #F0FFF0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #90EE90;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #FFE4E1;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B6B;
        margin: 1rem 0;
    }
    .result-container {
        background-color: #F8F9FA;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
        border: 2px solid #E9ECEF;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    """Memuat model yang telah disimpan"""
    # Cari file model di direktori saat ini
    current_dir = os.getcwd()
    
    # Daftar kemungkinan pola nama file model
    model_patterns = [
        "*.joblib",
        "*.pkl",
        "*.pickle",
        "*model*.joblib",
        "*model*.pkl"
    ]
    
    model_files = []
    for pattern in model_patterns:
        model_files.extend(glob.glob(os.path.join(current_dir, pattern)))
    
    # Jika tidak ada di direktori utama, coba cari di subdirektori
    if not model_files:
        subdirs = ["saved_models", "saved_models_new_4", "models", "model"]
        for subdir in subdirs:
            subdir_path = os.path.join(current_dir, subdir)
            if os.path.exists(subdir_path):
                for pattern in model_patterns:
                    model_files.extend(glob.glob(os.path.join(subdir_path, pattern)))
    
    if not model_files:
        st.error("❌ Tidak ada file model yang ditemukan!")
        st.info("📁 File model yang dicari: .joblib, .pkl, .pickle")
        st.info(f"📂 Direktori saat ini: {current_dir}")
        return None, None
    
    # Pilih model terbaru berdasarkan waktu modifikasi
    latest_model = max(model_files, key=os.path.getmtime)
    
    try:
        model_package = joblib.load(latest_model)
        model_name = os.path.basename(latest_model)
        st.success(f"✅ Model berhasil dimuat: {model_name}")
        return model_package, model_name
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None, None

# Fungsi untuk melakukan prediksi
def predict_heart_attack(model_package, input_data, debug=False):
    """Melakukan prediksi serangan jantung"""
    try:
        # Cek apakah model_package adalah dictionary dengan komponen terpisah
        if isinstance(model_package, dict):
            # Ekstrak komponen model jika tersedia
            model = model_package.get('model')
            selected_indices = model_package.get('selected_indices')
            scaler = model_package.get('scaler')
            selected_features = model_package.get('selected_features')
            
            if model is None:
                st.error("❌ Model tidak ditemukan dalam package")
                return None, None
            
            # Debug info (opsional)
            if debug:
                st.write(f"🔍 Debug - Input data shape: {input_data.shape}")
                st.write(f"🔍 Debug - Selected indices: {selected_indices}")
                if selected_indices is not None:
                    st.write(f"🔍 Debug - Selected indices type: {type(selected_indices)}")
                    st.write(f"🔍 Debug - Max index in selected_indices: {max(selected_indices) if len(selected_indices) > 0 else 'Empty'}")
            
            # Jika ada selected_indices, pilih fitur yang sesuai dengan validasi
            if selected_indices is not None:
                # Pastikan selected_indices adalah array/list yang valid
                if isinstance(selected_indices, (list, np.ndarray)) and len(selected_indices) > 0:
                    # Validasi bahwa semua indeks dalam rentang yang valid
                    max_index = max(selected_indices)
                    if max_index >= len(input_data):
                        if debug:
                            st.warning(f"⚠️ Index {max_index} melebihi jumlah fitur ({len(input_data)}). Menggunakan semua fitur.")
                        X_selected = input_data
                    else:
                        X_selected = input_data[selected_indices]
                        if debug:
                            st.success(f"✅ Menggunakan {len(selected_indices)} fitur terpilih")
                else:
                    if debug:
                        st.warning("⚠️ Selected indices tidak valid. Menggunakan semua fitur.")
                    X_selected = input_data
            else:
                if debug:
                    st.info("ℹ️ Tidak ada selected indices. Menggunakan semua fitur.")
                X_selected = input_data
            
            # Jika ada scaler, terapkan scaling
            if scaler is not None:
                X_scaled = scaler.transform(X_selected.reshape(1, -1))
                if debug:
                    st.success("✅ Scaling diterapkan")
            else:
                X_scaled = X_selected.reshape(1, -1)
                if debug:
                    st.info("ℹ️ Tidak ada scaler. Menggunakan data asli.")
            
            if debug:
                st.write(f"🔍 Debug - Final data shape untuk prediksi: {X_scaled.shape}")
            
            # Prediksi
            prediction = model.predict(X_scaled)[0]
            
            # Cek apakah model memiliki predict_proba
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(X_scaled)[0]
            else:
                # Jika tidak ada predict_proba, buat probabilitas sederhana
                probability = np.array([1-prediction, prediction]) if prediction in [0, 1] else np.array([0.5, 0.5])
            
        else:
            # Jika model_package adalah model langsung
            model = model_package
            X_scaled = input_data.reshape(1, -1)
            
            prediction = model.predict(X_scaled)[0]
            
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(X_scaled)[0]
            else:
                probability = np.array([1-prediction, prediction]) if prediction in [0, 1] else np.array([0.5, 0.5])
        
        return prediction, probability
        
    except Exception as e:
        st.error(f"❌ Gagal melakukan prediksi: {e}")
        st.error(f"Debug info: Model type: {type(model_package)}")
        if isinstance(model_package, dict):
            st.error(f"Debug info: Keys in model_package: {list(model_package.keys())}")
            if 'selected_indices' in model_package:
                st.error(f"Debug info: Selected indices: {model_package['selected_indices']}")
        return None, None

# Fungsi untuk membuat visualisasi risiko
def create_risk_visualization(probability, prediction):
    """Membuat visualisasi tingkat risiko"""
    risk_prob = probability[1] * 100  # Probabilitas kelas positif (serangan jantung)
    
    # Gauge chart untuk tingkat risiko
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_prob,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Tingkat Risiko Serangan Jantung (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

# Fungsi untuk interpretasi hasil
def interpret_result(prediction, probability):
    """Memberikan interpretasi hasil prediksi"""
    risk_prob = probability[1] * 100
    
    if prediction == 0:
        if risk_prob < 30:
            return "🟢 **RISIKO RENDAH**", "Berdasarkan data yang dimasukkan, risiko serangan jantung Anda tergolong rendah. Tetap jaga pola hidup sehat!"
        else:
            return "🟡 **RISIKO SEDANG**", "Meskipun prediksi menunjukkan tidak ada serangan jantung, tingkat risiko Anda cukup tinggi. Konsultasikan dengan dokter."
    else:
        if risk_prob > 70:
            return "🔴 **RISIKO TINGGI**", "Prediksi menunjukkan risiko serangan jantung yang tinggi. Segera konsultasikan dengan dokter spesialis jantung!"
        else:
            return "🟠 **RISIKO SEDANG-TINGGI**", "Ada indikasi risiko serangan jantung. Disarankan untuk melakukan pemeriksaan lebih lanjut dengan dokter."

# Header utama
st.markdown('<h1 class="main-header"> Sistem Prediksi Serangan Jantung</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Menggunakan AI untuk deteksi dini risiko serangan jantung</p>', unsafe_allow_html=True)

# Sidebar untuk informasi model
with st.sidebar:
    st.header("📊 Informasi Model")
    
    # Muat model
    model_package, model_name = load_model()
    
    if model_package:
        # Tampilkan informasi model
        st.success("✅ Model Siap Digunakan")
        
        with st.expander("Detail Model"):
            st.write(f"**Nama Model:** {model_name}")
            
            # Cek apakah model_package adalah dictionary
                
            if 'performance_metrics' in model_package:
                metrics = model_package['performance_metrics']
                st.write("**Performa Model:**")
                st.write(f"- Akurasi: {metrics.get('test_accuracy', 0):.3f}")
                st.write(f"- F1-Score: {metrics.get('test_f1_score', 0):.3f}")
                st.write(f"- AUC-ROC: {metrics.get('test_auc_roc', 0):.3f}")
            else:
                st.write("**Tipe:** Model Sederhana")
                st.write("**Status:** Siap digunakan")
    
    st.markdown("---")
    st.markdown("### 🏥 Disclaimer")
    st.warning("⚠️ Aplikasi ini hanya untuk tujuan skrining awal.")

# Main content - Cek model terlebih dahulu
if model_package is None:
    st.error("❌ Model tidak dapat dimuat. Pastikan file model tersedia di direktori yang sama dengan app.py")
    st.info("💡 Tips: Letakkan file model (.joblib, .pkl, atau .pickle) di folder yang sama dengan app.py")
    st.stop()

# Input Form
st.header("📝 Masukkan Data Pasien")

# Buat form input
with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Data Demografis")
        age = st.slider("Umur (tahun)", 20, 100, 50)
        
        st.subheader("Riwayat Penyakit")
        hypertension = st.selectbox("Hipertensi", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
        diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
        obesity = st.selectbox("Obesitas", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
        family_history = st.selectbox("Riwayat Keluarga Penyakit Jantung", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
        
    with col2:
        st.subheader("Gaya Hidup")
        smoking_status = st.selectbox("Status Merokok", [0, 1, 2], format_func=lambda x: {0: "Tidak Pernah", 1: "Mantan Perokok", 2: "Perokok Aktif"}[x])
        physical_activity = st.slider("Aktivitas Fisik (jam/minggu)", 0, 20, 5)
        sleep_hours = st.slider("Jam Tidur per Hari", 3, 12, 7)
        
        st.subheader("Pemeriksaan Fisik")
        blood_pressure_diastolic = st.slider("Tekanan Darah Diastolik (mmHg)", 60, 120, 80)
        
    with col3:
        st.subheader("Hasil Lab")
        cholesterol_level = st.slider("Kolesterol Total (mg/dL)", 100, 400, 200)
        fasting_blood_sugar = st.slider("Gula Darah Puasa (mg/dL)", 70, 200, 100)
        cholesterol_hdl = st.slider("Kolesterol HDL (mg/dL)", 20, 100, 50)
        
        st.subheader("Pemeriksaan Lain")
        ekg_results = st.selectbox("Hasil EKG", [0, 1], format_func=lambda x: "Normal" if x == 0 else "Abnormal")
        previous_heart_disease = st.selectbox("Riwayat Penyakit Jantung", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
    
    submit_button = st.form_submit_button("🔍 Prediksi Risiko Serangan Jantung", use_container_width=True)

# Hasil Prediksi - Tampil langsung setelah submit
if submit_button:
    # Siapkan data input
    input_array = np.array([
        age, hypertension, diabetes, cholesterol_level, obesity,
        family_history, smoking_status, physical_activity, sleep_hours,
        blood_pressure_diastolic, fasting_blood_sugar, cholesterol_hdl,
        ekg_results, previous_heart_disease
    ])
    
    patient_info = {
        'age': age,
        'hypertension': hypertension,
        'diabetes': diabetes,
        'cholesterol_level': cholesterol_level,
        'obesity': obesity,
        'family_history': family_history,
        'smoking_status': smoking_status,
        'physical_activity': physical_activity,
        'sleep_hours': sleep_hours,
        'blood_pressure_diastolic': blood_pressure_diastolic,
        'fasting_blood_sugar': fasting_blood_sugar,
        'cholesterol_hdl': cholesterol_hdl,
        'ekg_results': ekg_results,
        'previous_heart_disease': previous_heart_disease
    }
    
    # Lakukan prediksi
    with st.spinner("🔄 Memproses prediksi..."):
        prediction, probability = predict_heart_attack(model_package, input_array)
    
    if prediction is not None:
        # Container untuk hasil
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        
        st.header("📊 Hasil Prediksi")
        
        # Interpretasi hasil
        risk_level, risk_description = interpret_result(prediction, probability)
        
        # Tampilkan hasil utama
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### {risk_level}")
            st.markdown(f"**Deskripsi:** {risk_description}")
            
            # Probabilitas detail
            st.markdown("#### Probabilitas Detail:")
            st.write(f"- **Tidak Berisiko:** {probability[0]*100:.1f}%")
            st.write(f"- **Berisiko Serangan Jantung:** {probability[1]*100:.1f}%")
        
        with col2:
            # Gauge chart
            fig_gauge = create_risk_visualization(probability, prediction)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
# Footer

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>Dikembangkan Oleh: Kelompok 10 Kecerdasan Buatan</p>
        <p>⚠️ Aplikasi ini tidak menggantikan konsultasi medis profesional</p>
    </div>
    """, 
    unsafe_allow_html=True
)