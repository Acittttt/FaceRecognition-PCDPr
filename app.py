import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import os
from src.klasifikasi_suku import KlasifikasiSuku
from src.visualisasi_hasil import VisualisasiHasil

# Inisialisasi model dan visualisasi
@st.cache_resource
def load_klasifikasi_model():
    return KlasifikasiSuku(model_path='models/model_klasifikasi_suku.h5', 
                          label_path='models/label_encoder_classes.npy')

visualisasi = VisualisasiHasil()

def main():
    st.title("Sistem Deteksi Suku/Etnis")
    st.write("Aplikasi ini menggunakan model CNN untuk mengklasifikasikan suku/etnis berdasarkan wajah")
    
    # Sidebar
    st.sidebar.title("Navigasi")
    halaman = st.sidebar.radio("Pilih Halaman", ["Halaman Utama", "Tentang Aplikasi", "Dataset"])
    
    if halaman == "Halaman Utama":
        tampilkan_halaman_utama()
    elif halaman == "Tentang Aplikasi":
        tampilkan_tentang_aplikasi()
    else:
        tampilkan_dataset()

def tampilkan_halaman_utama():
    st.header("Klasifikasi Suku/Etnis dari Gambar Wajah")
    
    # Upload gambar
    uploaded_file = st.file_uploader("Unggah gambar wajah:", type=["jpg", "jpeg", "png"])
    
    # Atau gunakan gambar contoh
    st.write("Atau pilih gambar contoh:")
    
    # Temukan semua gambar contoh
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
    
    # Buat grid untuk gambar contoh
    if contoh_gambar:
        num_cols = 3
        cols = st.columns(num_cols)
        for i, gambar in enumerate(contoh_gambar[:6]):  # Batasi hanya 6 gambar contoh
            with cols[i % num_cols]:
                st.image(gambar, width=150)
                if st.button(f"Pilih Gambar {i+1}", key=f"btn_{i}"):
                    uploaded_file = gambar
    
    # Proses gambar jika diunggah
    if uploaded_file is not None:
        try:
            # Tampilkan gambar asli
            if isinstance(uploaded_file, str):  # Jika path file
                image = cv2.imread(uploaded_file)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image_rgb, caption="Gambar Wajah", use_column_width=True)
            else:  # Jika file yang diunggah
                image_bytes = uploaded_file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image_rgb, caption="Gambar Wajah", use_column_width=True)
            
            # Lakukan prediksi
            with st.spinner("Sedang mengklasifikasikan..."):
                klasifikasi = load_klasifikasi_model()
                predicted_class, class_probabilities, confidence = klasifikasi.predict_from_image(image)
                if isinstance(class_probabilities, np.ndarray):
                    class_probabilities = dict(zip(klasifikasi.label_classes, class_probabilities))
            # Tampilkan hasil
            st.success(f"Hasil Klasifikasi: {predicted_class}")
            st.info(f"Tingkat Keyakinan: {confidence:.2%}")
            
            # Visualisasi probabilitas
            st.subheader("Distribusi Probabilitas untuk Setiap Suku/Etnis")
            prob_plot = visualisasi.plot_probability_bar(class_probabilities)
            st.image(prob_plot, use_column_width=True)
            
            # Tambahkan hasil pada gambar
            result_image = visualisasi.create_result_image(image_rgb, predicted_class, confidence)
            st.subheader("Hasil Akhir")
            st.image(result_image, use_column_width=True)
            
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
    
    # Tampilkan plot training history jika ada
    if os.path.exists("outputs/training_history.png"):
        st.subheader("Grafik Pelatihan Model")
        st.image("outputs/training_history.png")
    
    # Tampilkan confusion matrix jika ada
    if os.path.exists("outputs/confusion_matrix.png"):
        st.subheader("Confusion Matrix")
        st.image("outputs/confusion_matrix.png")

def tampilkan_dataset():
    st.header("Dataset")
    
    st.write("""
    Dataset yang digunakan untuk proyek ini terdiri dari kumpulan gambar wajah dari berbagai suku/etnis.
    Setiap subjek memiliki folder suku yang berisi beberapa gambar wajah.
    """)
    
    # Hitung statistik dataset
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
                    # Hitung gambar dalam folder suku
                    images = [f for f in os.listdir(suku_path) if f.endswith((".jpg", ".jpeg", ".png"))]
                    num_images = len(images)
                    total_images += num_images
                    
                    # Update suku counts
                    if suku not in suku_counts:
                        suku_counts[suku] = {"subjects": 0, "images": 0}
                    suku_counts[suku]["subjects"] += 1
                    suku_counts[suku]["images"] += num_images
    
    # Buat tabel statistik
    for suku, counts in suku_counts.items():
        dataset_stats.append({
            "Suku/Etnis": suku,
            "Jumlah Subjek": counts["subjects"],
            "Jumlah Gambar": counts["images"]
        })
    
    # Tambahkan total
    dataset_stats.append({
        "Suku/Etnis": "Total",
        "Jumlah Subjek": total_subjects,
        "Jumlah Gambar": total_images
    })
    
    st.table(dataset_stats)
    
    # Visualisasi statistik dataset
    suku_names = [stat["Suku/Etnis"] for stat in dataset_stats[:-1]]  # Exclude total
    subject_counts = [stat["Jumlah Subjek"] for stat in dataset_stats[:-1]]
    image_counts = [stat["Jumlah Gambar"] for stat in dataset_stats[:-1]]
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot jumlah subjek
    ax[0].bar(suku_names, subject_counts, color='skyblue')
    ax[0].set_title('Jumlah Subjek per Suku/Etnis')
    ax[0].set_xlabel('Suku/Etnis')
    ax[0].set_ylabel('Jumlah Subjek')
    ax[0].tick_params(axis='x', rotation=45)
    
    # Plot jumlah gambar
    ax[1].bar(suku_names, image_counts, color='lightgreen')
    ax[1].set_title('Jumlah Gambar per Suku/Etnis')
    ax[1].set_xlabel('Suku/Etnis')
    ax[1].set_ylabel('Jumlah Gambar')
    ax[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Tampilkan beberapa contoh gambar dari dataset
    st.subheader("Contoh Gambar dari Dataset")
    
    # Ambil satu contoh gambar dari setiap suku
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
    
    # Tampilkan gambar contoh dalam grid
    if sample_images:
        num_cols = 3
        cols = st.columns(num_cols)
        for i, img_info in enumerate(sample_images):
            with cols[i % num_cols]:
                st.image(img_info["path"], caption=img_info["caption"], width=200)

if __name__ == "__main__":
    main()