import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
from skimage.feature import graycomatrix, graycoprops

# =============================================================
# 1. KONFIGURASI HALAMAN
# =============================================================
st.set_page_config(
    page_title="Pneumonia Detection - GLCM & Ensemble ML",
    page_icon="🫁",
    layout="wide"
)

# Custom CSS ringkas untuk mempercantik tampilan
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-title {
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        text-align: center;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    scaler = joblib.load('glcm_scaler.pkl')
    model_svm = joblib.load('model_svm.pkl')
    model_rf = joblib.load('model_rf.pkl')
    model_knn = joblib.load('model_knn.pkl')
    model_voting = joblib.load('model_voting.pkl')
    return scaler, model_svm, model_rf, model_knn, model_voting

scaler, model_svm, model_rf, model_knn, model_voting = load_artifacts()

# =============================================================
# 2. PREPROCESSING & EKSTRAKSI FITUR GLCM
# =============================================================
def extract_glcm_from_bytes(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        return None, None, None, None
        
    # Preprocessing: Grayscale & Resize 128x128
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    
    # Ekstraksi GLCM
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(resized, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    features = {
        'contrast': graycoprops(glcm, 'contrast').mean(),
        'dissimilarity': graycoprops(glcm, 'dissimilarity').mean(),
        'homogeneity': graycoprops(glcm, 'homogeneity').mean(),
        'energy': graycoprops(glcm, 'energy').mean(),
        'correlation': graycoprops(glcm, 'correlation').mean(),
        'asm': graycoprops(glcm, 'ASM').mean()
    }
    
    return img_bgr, resized, pd.DataFrame([features]), features

# =============================================================
# 3. SIDEBAR & HEADER
# =============================================================
st.markdown("<h1 class='main-title'>🫁 Deteksi Pneumonia Citra Rontgen</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Klasifikasi berbasis <b>GLCM (Gray-Level Co-occurrence Matrix)</b> & <b>Ensemble Learning</b></p>", unsafe_allow_html=True)

st.sidebar.header("⚙️ Pengaturan Model")
selected_model_name = st.sidebar.selectbox(
    "Pilih Algoritma Model:",
    ["Voting Ensemble (SVM + RF + KNN)", "Random Forest", "Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)"]
)

model_dict = {
    "Voting Ensemble (SVM + RF + KNN)": model_voting,
    "Random Forest": model_rf,
    "Support Vector Machine (SVM)": model_svm,
    "K-Nearest Neighbors (KNN)": model_knn
}
chosen_model = model_dict[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.info(
    "**Petunjuk Usage:**\n"
    "1. Unggah gambar Rontgen dada (.jpg / .png).\n"
    "2. Sistem mengekstrak fitur GLCM secara otomatis.\n"
    "3. Hasil klasifikasi dan tingkat keyakinan akan langsung muncul."
)

# =============================================================
# 4. LAYOUT UTAMA (2 KOLOM SIMETRIS)
# =============================================================
col1, col2 = st.columns([1, 1], gap="medium")

# --- KOLOM KIRI: INPUT & IMAGE PREVIEW ---
with col1:
    with st.container(border=True):
        st.subheader("📁 Input Citra Rontgen")
        uploaded_file = st.file_uploader("Unggah file gambar X-Ray (JPG / PNG):", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            img_bgr, resized_gray, df_features, feat_dict = extract_glcm_from_bytes(uploaded_file)
            
            if img_bgr is not None:
                # Tab tampilan gambar
                tab_orig, tab_proc = st.tabs(["🖼️ Citra Asli", "⚙️ Preprocessed (128x128)"])
                with tab_orig:
                    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                with tab_proc:
                    st.image(resized_gray, caption="Output Grayscale & Resized untuk GLCM", use_container_width=True)
            else:
                st.error("Gagal membaca file gambar. Pastikan format file adalah citra yang valid.")
        else:
            # Placeholder ketika gambar belum diunggah
            st.info("Gunakan tombol di atas untuk memilih dan mengunggah gambar rontgen dada.")

# --- KOLOM KANAN: HASIL ANALISIS & DIAGNOSIS ---
with col2:
    if uploaded_file is not None and img_bgr is not None:
        # Standardisasi & Prediksi
        feature_cols = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'asm']
        scaled_features = scaler.transform(df_features[feature_cols])
        
        prediction = chosen_model.predict(scaled_features)[0]
        probabilities = chosen_model.predict_proba(scaled_features)[0]
        
        # 1. Card Hasil Utama
        with st.container(border=True):
            st.subheader("🔍 Hasil Diagnosis")
            if prediction == 1:
                st.error("### ⚠️ Diagnosa: PNEUMONIA")
                confidence = probabilities[1] * 100
            else:
                st.success("### ✅ Diagnosa: NORMAL")
                confidence = probabilities[0] * 100
                
            st.write(f"Tingkat Keyakinan Model: **{confidence:.2f}%**")
            st.progress(float(confidence / 100))

        # 2. Card Ekstraksi Fitur GLCM (Grid 3x2 Metric)
        with st.container(border=True):
            st.subheader("📊 Ekstraksi Fitur Tekstur GLCM")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Contrast", f"{feat_dict['contrast']:.4f}")
            m2.metric("Dissimilarity", f"{feat_dict['dissimilarity']:.4f}")
            m3.metric("Homogeneity", f"{feat_dict['homogeneity']:.4f}")
            
            m4, m5, m6 = st.columns(3)
            m4.metric("Energy", f"{feat_dict['energy']:.4f}")
            m5.metric("Correlation", f"{feat_dict['correlation']:.4f}")
            m6.metric("ASM", f"{feat_dict['asm']:.4f}")

        # 3. Card Breakdown Model Individual (Jika memilih Voting Ensemble)
        if selected_model_name == "Voting Ensemble (SVM + RF + KNN)":
            with st.container(border=True):
                st.subheader("🤖 Keputusan Model Individual")
                
                p_svm = "PNEUMONIA" if model_svm.predict(scaled_features)[0] == 1 else "NORMAL"
                p_rf  = "PNEUMONIA" if model_rf.predict(scaled_features)[0] == 1 else "NORMAL"
                p_knn = "PNEUMONIA" if model_knn.predict(scaled_features)[0] == 1 else "NORMAL"
                
                b1, b2, b3 = st.columns(3)
                
                with b1:
                    st.caption("**SVM**")
                    if p_svm == "PNEUMONIA":
                        st.error("PNEUMONIA")
                    else:
                        st.success("NORMAL")
                        
                with b2:
                    st.caption("**Random Forest**")
                    if p_rf == "PNEUMONIA":
                        st.error("PNEUMONIA")
                    else:
                        st.success("NORMAL")
                        
                with b3:
                    st.caption("**KNN (k=5)**")
                    if p_knn == "PNEUMONIA":
                        st.error("PNEUMONIA")
                    else:
                        st.success("NORMAL")
    else:
        # Placeholder simetris saat belum ada gambar yang diunggah
        with st.container(border=True):
            st.subheader("🔍 Hasil Analisis & Prediksi")
            st.info("👈 Silakan unggah citra rontgen dada di panel sebelah kiri untuk melihat hasil prediksi dan ekstraksi fitur GLCM.")