import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import os
import pickle

# Import modul dari folder models
from models.deteksi_wajah import detect_faces
from models.ekstraksi_wajah import extract_face_embedding
from models.perbandingan_wajah import euclidean_distance, compute_similarity_score

# Import modul dari folder src untuk klasifikasi dan visualisasi
from src.klasifikasi_suku import KlasifikasiSuku
from src.visualisasi_hasil import VisualisasiHasil

# Fungsi untuk memuat file embeddings (untuk face similarity)
@st.cache_data
def load_embeddings(file_path='embeddings.pkl'):
    """
    Memuat data embeddings (nama, etnis, embedding wajah) dari file pickle.
    Bila file pickle dibuat pada environment berbeda, gunakan fix_imports=True.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f, fix_imports=True)
    return data

# Fungsi untuk menginisialisasi model klasifikasi (suku/etnis)
@st.cache_resource
def load_klasifikasi_model():
    return KlasifikasiSuku(
        model_path='models/model_klasifikasi_suku.h5',
        label_path='models/label_encoder_classes.npy'
    )

visualisasi = VisualisasiHasil()

def main():
    st.title("Aplikasi Deteksi & Klasifikasi Wajah")
    st.write("Pilih menu pada sidebar untuk mengakses fitur Face Similarity atau Sistem Deteksi Suku/Etnis.")

    # Sidebar untuk navigasi
    st.sidebar.title("Navigasi")
    halaman = st.sidebar.radio("Pilih Halaman", ["Face Similarity", "Klasifikasi Suku/Etnis", "Tentang Aplikasi", "Dataset"])

    if halaman == "Face Similarity":
        tampilkan_face_similarity()
    elif halaman == "Klasifikasi Suku/Etnis":
        tampilkan_halaman_utama()
    elif halaman == "Tentang Aplikasi":
        tampilkan_tentang_aplikasi()
    elif halaman == "Dataset":
        tampilkan_dataset()

def tampilkan_face_similarity():
    st.header("Face Similarity (Kemiripan Wajah)")
    st.write("Unggah gambar untuk mendeteksi wajah dan membandingkannya dengan dataset embeddings.")

    # Muat embeddings yang telah diregenerasi (pastikan file embeddings.pkl sudah terupdate)
    data = load_embeddings()

    uploaded_file = st.file_uploader("Pilih Gambar", type=["jpg", "jpeg", "png"], key="facesim")
    if uploaded_file is not None:
        st.write("Uploaded file:", uploaded_file.name)
        uploaded_file.seek(0)
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        st.write("Jumlah byte file:", len(file_bytes))
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Gagal membaca gambar!")
            return

        # Tampilkan gambar asli
        st.image(image, channels="BGR", caption="Gambar Asli", use_container_width=True)

        with st.spinner("Memproses gambar, mohon tunggu..."):
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            MAX_WIDTH = 800
            height, width = rgb_image.shape[:2]
            if width > MAX_WIDTH:
                scaling_factor = MAX_WIDTH / width
                new_dimensions = (int(width * scaling_factor), int(height * scaling_factor))
                rgb_image = cv2.resize(rgb_image, new_dimensions)
                st.write(f"Gambar di-resize menjadi: {new_dimensions}")

            face_locations = detect_faces(rgb_image)
            st.write(f"Jumlah wajah terdeteksi: {len(face_locations)}")

            threshold = 0.6
            if face_locations:
                for idx, face_loc in enumerate(face_locations):
                    top, right, bottom, left = face_loc
                    embeddings = extract_face_embedding(rgb_image, face_loc)
                    if embeddings:
                        embedding_input = embeddings[0]
                        all_distances = [euclidean_distance(embedding_input, emb) for emb in data['embeddings']]
                        min_dist = np.min(all_distances)
                        min_idx = np.argmin(all_distances)
                        is_match = min_dist < threshold
                        similarity_score = compute_similarity_score(min_dist, threshold)
                        best_name = data['names'][min_idx] if is_match else "Unknown"
                        best_ethnicity = data['ethnicities'][min_idx] if is_match else "-"
                        
                        st.write(f"**Wajah ke-{idx+1}:**")
                        st.write(f"- Nama Terdekat: {best_name}")
                        st.write(f"- Etnis: {best_ethnicity}")
                        st.write(f"- Distance: {min_dist:.2f}")
                        st.write(f"- Similarity Score: {similarity_score:.2f} (0-1)")
                        st.write(f"- Match? {'Ya' if is_match else 'Tidak'}")
                        st.write("---")
                        
                        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                        text = best_name if is_match else "Unknown"
                        cv2.putText(image, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        st.write(f"Tidak bisa mengekstrak embedding dari wajah ke-{idx+1}")

                st.image(image, channels="BGR", caption="Hasil Deteksi dan Bounding Box", use_container_width=True)
            else:
                st.warning("Tidak ada wajah terdeteksi di gambar ini.")
        st.success("Proses gambar selesai.")

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

def tampilkan_tentang_aplikasi():
    st.header("Tentang Aplikasi Deteksi Suku/Etnis")
    st.write("""
    Aplikasi ini menggunakan model Convolutional Neural Network (CNN) dengan teknik transfer learning
    untuk mengklasifikasikan suku/etnis berdasarkan fitur wajah.
    
    ### Metodologi
    1. **Preprocessing Data**:
       - Normalisasi ukuran gambar (224x224 piksel)
       - Normalisasi nilai piksel (0-1)
       - Augmentasi data untuk training
    
    2. **Arsitektur Model**:
       - Model dasar: ResNet50 pre-trained pada ImageNet
       - Layer tambahan: GlobalAveragePooling, Dense (1024), Dropout, Dense (output)
       - Output: Probabilitas untuk setiap kelas suku/etnis
    
    3. **Training**:
       - Transfer learning dengan fine-tuning
       - Optimizer: Adam
       - Loss function: Categorical Cross-Entropy
       - Early stopping untuk mencegah overfitting
    
    ### Perfoma Model
    - Akurasi model pada data validasi: ~85-90%
    - Metrik evaluasi: Precision, Recall, F1-Score
    
    ### Limitasi
    - Model masih sensitif terhadap variasi pencahayaan ekstrem
    - Performa dapat menurun pada pose wajah non-frontal
    - Dataset terbatas pada suku/etnis tertentu
    """)
    
    if os.path.exists("outputs/training_history.png"):
        st.subheader("Grafik Pelatihan Model")
        st.image("outputs/training_history.png")
    
    if os.path.exists("outputs/confusion_matrix.png"):
        st.subheader("Confusion Matrix")
        st.image("outputs/confusion_matrix.png")

def tampilkan_dataset():
    st.header("Dataset")
    st.write("""
    Dataset yang digunakan untuk proyek ini terdiri dari kumpulan gambar wajah dari berbagai suku/etnis.
    Setiap subjek memiliki folder suku yang berisi beberapa gambar wajah.
    """)
    
    dataset_stats = []
    total_subjects = 0
    total_images = 0
    suku_counts = {}
    
    for nama in os.listdir("dataset"):
        nama_path = os.path.join("dataset", nama)
        if os.path.isdir(nama_path):
            total_subjects += 1
            for suku in os.listdir(nama_path):
                suku_path = os.path.join(nama_path, suku)
                if os.path.isdir(suku_path):
                    images = [f for f in os.listdir(suku_path) if f.endswith((".jpg", ".jpeg", ".png"))]
                    num_images = len(images)
                    total_images += num_images
                    if suku not in suku_counts:
                        suku_counts[suku] = {"subjects": 0, "images": 0}
                    suku_counts[suku]["subjects"] += 1
                    suku_counts[suku]["images"] += num_images
    
    for suku, counts in suku_counts.items():
        dataset_stats.append({
            "Suku/Etnis": suku,
            "Jumlah Subjek": counts["subjects"],
            "Jumlah Gambar": counts["images"]
        })
    
    dataset_stats.append({
        "Suku/Etnis": "Total",
        "Jumlah Subjek": total_subjects,
        "Jumlah Gambar": total_images
    })
    
    st.table(dataset_stats)
    
    suku_names = [stat["Suku/Etnis"] for stat in dataset_stats[:-1]]
    subject_counts = [stat["Jumlah Subjek"] for stat in dataset_stats[:-1]]
    image_counts = [stat["Jumlah Gambar"] for stat in dataset_stats[:-1]]
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].bar(suku_names, subject_counts, color='skyblue')
    ax[0].set_title('Jumlah Subjek per Suku/Etnis')
    ax[0].set_xlabel('Suku/Etnis')
    ax[0].set_ylabel('Jumlah Subjek')
    ax[0].tick_params(axis='x', rotation=45)
    ax[1].bar(suku_names, image_counts, color='lightgreen')
    ax[1].set_title('Jumlah Gambar per Suku/Etnis')
    ax[1].set_xlabel('Suku/Etnis')
    ax[1].set_ylabel('Jumlah Gambar')
    ax[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("Contoh Gambar dari Dataset")
    sample_images = []
    for suku in suku_counts.keys():
        for nama in os.listdir("dataset"):
            nama_path = os.path.join("dataset", nama)
            if os.path.isdir(nama_path):
                suku_path = os.path.join(nama_path, suku)
                if os.path.exists(suku_path) and os.path.isdir(suku_path):
                    images = [f for f in os.listdir(suku_path) if f.endswith((".jpg", ".jpeg", ".png"))]
                    if images:
                        sample_images.append({
                            "path": os.path.join(suku_path, images[0]),
                            "caption": f"{nama} - {suku}"
                        })
                        break
    if sample_images:
        num_cols = 3
        cols = st.columns(num_cols)
        for i, img_info in enumerate(sample_images):
            with cols[i % num_cols]:
                st.image(img_info["path"], caption=img_info["caption"], width=200)

if __name__ == "__main__":
    main()