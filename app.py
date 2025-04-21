import streamlit as st
import cv2
import numpy as np
import os
import pickle
import sys
import tempfile
import dlib
import io
import matplotlib.pyplot as plt
from imutils import face_utils

# --- Face Preprocessing imports ---
# Add parent directory to path to import our preprocessing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.alignment import FacePreprocessor
from src.normalization import ImageNormalizer

# --- Face Similarity & Classification imports ---
from models.ekstraksi_wajah import extract_face_embedding
from models.perbandingan_wajah import euclidean_distance, compute_similarity_score
from src.klasifikasi_suku import KlasifikasiSuku
from src.visualisasi_hasil import VisualisasiHasil

# --- Caching functions ---
@st.cache_data
def load_embeddings(file_path='embeddings.pkl'):
    with open(file_path, 'rb') as f:
        return pickle.load(f, fix_imports=True)

@st.cache_resource
def load_klasifikasi_model():
    return KlasifikasiSuku(
        model_path='models/model_klasifikasi_suku.h5',
        label_path='models/label_encoder_classes.npy'
    )

@st.cache_resource
def load_face_detector():
    return dlib.get_frontal_face_detector()

@st.cache_resource
def load_landmark_predictor(path):
    if os.path.exists(path):
        return dlib.shape_predictor(path)
    st.error(f"Landmark predictor model not found at {path}")
    return None

# --- Visualization helper ---
visualisasi = VisualisasiHasil()

# --- Main app ---
def main():
    st.set_page_config(page_title="Unified Face App", page_icon="🧩", layout="wide")
    st.title("Aplikasi Deteksi, Klasifikasi & Preprocessing Wajah")
    st.sidebar.title("Navigasi")
    pilihan = st.sidebar.radio("Pilih Halaman", [
        "Face Similarity", "Klasifikasi Suku/Etnis", "Face Preprocessing", "Tentang Aplikasi", "Dataset"
    ])

    if pilihan == "Face Similarity":
        tampilkan_face_similarity()
    elif pilihan == "Klasifikasi Suku/Etnis":
        tampilkan_halaman_utama()
    elif pilihan == "Face Preprocessing":
        tampilkan_face_preprocessing()
    elif pilihan == "Tentang Aplikasi":
        tampilkan_tentang_aplikasi()
    else:
        tampilkan_dataset()

# --- Face Similarity (uses FacePreprocessor) ---
def tampilkan_face_similarity():
    st.header("Face Similarity (Kemiripan Wajah)")
    st.write("Unggah gambar untuk membandingkan dengan embeddings dataset.")
    data = load_embeddings()

    # Initialize FacePreprocessor
    detector = load_face_detector()
    predictor = load_landmark_predictor("models/shape_predictor_68_face_landmarks.dat")
    pre = FacePreprocessor(
        target_size=(224,224),
        face_detector=detector,
        landmark_predictor_path="models/shape_predictor_68_face_landmarks.dat"
    )

    uploaded = st.file_uploader("Pilih Gambar", type=["jpg","jpeg","png"], key="facesim")
    if not uploaded:
        return
    arr = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Gagal membaca gambar!")
        return
    st.image(img, caption="Gambar Asli", use_container_width=True)

    with st.spinner("Memproses wajah, mohon tunggu..."):
        faces = pre.detect_face(img)
        st.write(f"Jumlah wajah terdeteksi: {len(faces)}")
        st.write("")  # blank line
        for i, face_rect in enumerate(faces):
            top, right, bottom, left = face_rect.top(), face_rect.right(), face_rect.bottom(), face_rect.left()
            emb = extract_face_embedding(img, (top, right, bottom, left))
            if not emb:
                continue
            inp = emb[0]
            dists = [euclidean_distance(inp, e) for e in data['embeddings']]
            md, mi = min(dists), np.argmin(dists)
            match = md < 0.6
            score = compute_similarity_score(md, 0.6)
            name = data['names'][mi] if match else "Unknown"
            eth = data['ethnicities'][mi] if match else "-"
            # Print formatted result
            st.write(f"Wajah ke-{i+1}:")
            st.write(f"Nama={eth}")
            st.write(f"Etnis={name}")
            st.write(f"Distance={md:.2f}")
            st.write(f"Score={score:.2f}")
            # Draw box and label on image
            cv2.rectangle(img, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(img, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        st.image(img, caption="Hasil Face Similarity", use_container_width=True)

# --- Ethnicity Classification ---
def tampilkan_halaman_utama():
    st.header("Klasifikasi Suku/Etnis dari Gambar Wajah")
    
    uploaded_file = st.file_uploader("Unggah gambar wajah:", type=["jpg", "jpeg", "png"], key="klasifikasi")
    st.write("Atau pilih gambar contoh:")
    
    contoh_gambar = []
    for nama in os.listdir("dataset"):
        nama_path = os.path.join("dataset", nama)
        if os.path.isdir(nama_path):
            for suku in os.listdir(nama_path):
                suku_path = os.path.join(nama_path, suku)
                if os.path.isdir(suku_path):
                    for file in os.listdir(suku_path):
                        if file.endswith((".jpg", ".jpeg", ".png")):
                            contoh_gambar.append(os.path.join(suku_path, file))
    
    if contoh_gambar:
        num_cols = 3
        cols = st.columns(num_cols)
        for i, gambar in enumerate(contoh_gambar[:6]):  # Batasi 6 contoh
            with cols[i % num_cols]:
                st.image(gambar, width=150)
                if st.button(f"Pilih Gambar {i+1}", key=f"btn_{i}"):
                    uploaded_file = gambar
    
    if uploaded_file is not None:
        try:
            if isinstance(uploaded_file, str):
                image = cv2.imread(uploaded_file)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image_rgb, caption="Gambar Wajah", use_container_width=True)
            else:
                image_bytes = uploaded_file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image_rgb, caption="Gambar Wajah", use_container_width=True)
            
            with st.spinner("Sedang mengklasifikasikan..."):
                klasifikasi = load_klasifikasi_model()
                predicted_class, class_probabilities, confidence = klasifikasi.predict_from_image(image)
                if isinstance(class_probabilities, np.ndarray):
                    class_probabilities = dict(zip(klasifikasi.label_classes, class_probabilities))
            st.success(f"Hasil Klasifikasi: {predicted_class}")
            st.info(f"Tingkat Keyakinan: {confidence:.2%}")
            
            st.subheader("Distribusi Probabilitas untuk Setiap Suku/Etnis")
            prob_plot = visualisasi.plot_probability_bar(class_probabilities)
            st.image(prob_plot, use_container_width=True)
            
            result_image = visualisasi.create_result_image(image_rgb, predicted_class, confidence)
            st.subheader("Hasil Akhir")
            st.image(result_image, use_container_width=True)
            
        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")

# --- Face Preprocessing ---
def tampilkan_face_preprocessing():
    st.header("Face Preprocessing Demo")
    st.write("Demo pipeline preprocessing wajah: alignment & normalization.")

    # Sidebar config
    st.sidebar.title("Konfigurasi Preprocessing")
    landmark_path = st.sidebar.text_input("Path Landmark Predictor", "models/shape_predictor_68_face_landmarks.dat")
    target_size = st.sidebar.selectbox("Ukuran Target", [(224,224),(160,160),(96,96),(299,299)], format_func=lambda x:f"{x[0]}x{x[1]}")
    st.sidebar.subheader("Opsi Preprocessing")
    do_alignment       = st.sidebar.checkbox("Face Alignment", True)
    do_white_balance   = st.sidebar.checkbox("White Balance", True)
    do_contrast_norm   = st.sidebar.checkbox("Contrast Normalization", True)
    do_color_norm      = st.sidebar.checkbox("Color Normalization", True)
    do_brightness_norm = st.sidebar.checkbox("Brightness Normalization", True)

    # Init tools
    try:
        fd   = load_face_detector()
        lp   = load_landmark_predictor(landmark_path)
        pre  = FacePreprocessor(target_size=target_size,
                                face_detector=fd,
                                landmark_predictor_path=(landmark_path if os.path.exists(landmark_path) else None))
        norm = ImageNormalizer()
    except Exception as e:
        st.error(f"Error init tools: {e}")
        return

    # Upload and read
    uploaded = st.file_uploader("Pilih Gambar...", type=["jpg","jpeg","png"], key="preproc")
    if not uploaded:
        return
    tf = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    tf.write(uploaded.read()); tf.flush(); tf.close()
    img = cv2.imread(tf.name)
    if img is None:
        st.error("Gagal membaca gambar."); os.unlink(tf.name); return

    # 1) Original
    st.subheader("Original Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=200)

    # 2) Detection + landmarks
    faces = pre.detect_face(img)
    if not faces:
        st.error("No faces detected."); os.unlink(tf.name); return
    det = img.copy()
    for f in faces:
        x1,y1,x2,y2 = f.left(),f.top(),f.right(),f.bottom()
        cv2.rectangle(det,(x1,y1),(x2,y2),(0,255,0),2)
        lm = pre.get_landmarks(img,f)
        if lm is not None:
            for (x,y) in lm:
                cv2.circle(det,(x,y),2,(0,0,255),-1)
    st.subheader("Face Detection")
    st.image(cv2.cvtColor(det,cv2.COLOR_BGR2RGB), width=200)

    # 3) Align/Crop
    idx=0
    if len(faces)>1:
        idx = st.selectbox("Pilih Face:", list(range(len(faces))), format_func=lambda i:f"Face {i+1}")
    f=faces[idx]
    x,y,w,h = f.left(),f.top(),f.right()-f.left(),f.bottom()-f.top()
    if do_alignment:
        lm = pre.get_landmarks(img,f)
        if lm is not None and len(lm) > 0:
            try:
                aligned = pre.align_face(img, lm)
                st.subheader("Face Alignment")
                st.image(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB), width=200)
            except Exception as e:
                st.warning(f"Align gagal: {e}; fallback ke crop.")
                aligned = cv2.resize(img[y:y + h, x:x + w], target_size)
        else:
            st.warning("No landmarks; pakai crop biasa.")
            aligned = cv2.resize(img[y:y + h, x:x + w], target_size)
            st.subheader("Cropped Face")
            st.image(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB), width=200)
    else:
        aligned = cv2.resize(img[y:y + h, x:x + w], target_size)
        st.subheader("Cropped Face")
        st.image(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB), width=200)

    # 4) White Balance
    nn = aligned.copy()
    if do_white_balance:
        nn = norm.normalize_white_balance(nn); st.subheader("After White Balance"); st.image(cv2.cvtColor(nn,cv2.COLOR_BGR2RGB), width=200)

    # 5) Contrast
    if do_contrast_norm:
        nn=norm.normalize_contrast(nn); st.subheader("After Contrast Normalization"); st.image(cv2.cvtColor(nn,cv2.COLOR_BGR2RGB), width=200)

    # 6) Color
    if do_color_norm:
        nn=norm.normalize_color(nn); st.subheader("After Color Normalization"); st.image(cv2.cvtColor(nn,cv2.COLOR_BGR2RGB), width=200)

    # 7) Brightness
    if do_brightness_norm:
        g=st.sidebar.slider("Gamma",0.5,2.0,1.2,0.1); nn=norm.gamma_correction(nn,g); st.subheader(f"After Gamma Correction (γ={g})"); st.image(cv2.cvtColor(nn,cv2.COLOR_BGR2RGB), width=200)

    # 8) Final & comparison
    st.header("Final Preprocessed Result")
    st.image(cv2.cvtColor(nn, cv2.COLOR_BGR2RGB), width=300)

    # === ⬇⬇⬇ Tambahkan blok DOWNLOAD di sini ⬇⬇⬇ ===
    # Encode ke PNG → BytesIO agar bisa di‑download
    success, png = cv2.imencode('.png', nn)
    if success:
        st.download_button(
            label="Download Preprocessed Image",
            data=png.tobytes(),
            file_name="preprocessed_face.png",
            mime="image/png",
            help="Klik untuk mengunduh hasil preprocessing"
        )
    # === ⬆⬆⬆ End blok DOWNLOAD ⬆⬆⬆ ===

    st.header("Before and After Comparison")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=200)
    with c2:
        st.subheader("Preprocessed Face")
        st.image(cv2.cvtColor(nn, cv2.COLOR_BGR2RGB), width=200)

    os.unlink(tf.name)

# --- About ---
def tampilkan_tentang_aplikasi():
    st.header("Tentang Aplikasi")
    st.write("Aplikasi gabungan: Face Similarity, Klasifikasi Suku/Etnis & Face Preprocessing.")

# --- Dataset ---
def tampilkan_dataset():
    st.header("Dataset")
    st.write("Stats dan contoh dataset.")

if __name__ == "__main__":
    main()
