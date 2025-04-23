# Sistem Pengenalan Wajah dan Deteksi Suku Menggunakan Computer Vision

## Deskripsi
Sistem ini mengimplementasikan teknologi computer vision untuk pengenalan wajah (face similarity) dan deteksi suku/etnis berdasarkan fitur wajah. Dibangun menggunakan pendekatan deep learning dengan CNN dan transfer learning, sistem dapat mendeteksi wajah, melakukan perbandingan kecocokan wajah, serta mengklasifikasikan suku/etnis dari gambar wajah yang diberikan.

## Fitur Utama
1. Preprocessing Wajah
   - Deteksi wajah dengan HOG dan CNN detector
   - Ekstraksi landmark wajah (68 titik)
   - Face alignment berdasarkan posisi mata
   - Normalisasi brightness, kontras, dan white balance

2. Sistem Face Similarity
   - Ekstraksi face embedding 128-dimensi
   - Komputasi similarity dengan metrik jarak Euclidean
   - Pencarian kecocokan wajah berdasarkan threshold

3. Sistem Deteksi Suku/Etnis
   - Klasifikasi suku/etnis menggunakan CNN dengan transfer learning
   - Model berbasis ResNet50 dengan layer kustom
   - Hasil klasifikasi dengan probabilitas per kategori suku

4. Antarmuka Aplikasi
   - Interface berbasis Streamlit dengan tampilan interaktif
   - Visualisasi hasil deteksi wajah dan klasifikasi suku
   - Upload gambar dan pemrosesan real-time

## Prasyarat
- Python 3.8+
- Pustaka yang tercantum dalam `requirements.txt`
- GPU (opsional, untuk performa lebih baik)

## Instalasi

### 1. Clone Repository
- git clone https://github.com/Acittttt/FaceRecognition-PCDPr.git
- cd face-ethnicity-detection


### 2. Install Dependensi
pip install -r requirements.txt

Requirements utama meliputi:
- tensorflow
- opencv-python
- dlib
- face_recognition
- streamlit
- numpy
- scikit-learn
- scikit-image
- matplotlib

### 3. Download Model Landmark Predictor
Download `shape_predictor_68_face_landmarks.dat` dari [dlib models](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), ekstrak dan tempatkan di folder `models/`.

## Langkah Penggunaan Sistem

### Sistem Face Similarity
1. Generate face embeddings dari dataset yang sudah di-preprocessing:
   python regenerate_embeddings.py

   Proses ini akan menghasilkan file `embeddings.pkl` yang berisi face embedding dari seluruh gambar di dataset.

2. Setelah embeddings terbentuk, jalankan aplikasi:
   python app.py

3. Pada antarmuka aplikasi, pilih tab "Face Similarity" untuk mengunggah gambar dan melakukan perbandingan wajah.

### Sistem Deteksi Suku/Etnis
1. Latih model deteksi suku/etnis (pastikan sudah melakukan preprocessing sebelumnya):
   python train_model.py

   Proses ini akan menghasilkan dua file model:
   - `models/model_klasifikasi_suku.h5` (model tahap feature extraction)
   - `models/model_klasifikasi_suku_finetuned.h5` (model dengan fine-tuning)

2. Setelah model terlatih, jalankan aplikasi:
   python app.py

3. Pada antarmuka aplikasi, pilih tab "Deteksi Suku/Etnis" untuk mengunggah gambar dan melakukan klasifikasi suku.

### Preprocessing Data
1. Letakkan gambar dataset di folder `dataset/` dengan struktur:
   ```
   dataset/
   ├── [nama_subjek]/
   │   ├── [suku]/
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   │   └── ...
   ```
   
2. Setelah menjalankan apk jalankan preprocessing data kemudia simpan gambar dataset di folder dataset/ dengan struktur yang sama

## Cara Kerja Sistem

### Preprocessing
Saat mengunggah gambar ke sistem, gambar akan melalui tahapan preprocessing:
1. Deteksi wajah menggunakan HOG/CNN detector
2. Ekstraksi 68 landmark wajah
3. Face alignment berdasarkan posisi mata
4. Normalisasi brightness menggunakan CLAHE di ruang warna LAB
5. Normalisasi kontras di ruang warna YCrCb
6. White balance menggunakan algoritma Gray World
7. Resizing ke 224x224 piksel

### Face Similarity
Saat melakukan perbandingan wajah:
1. Wajah yang dideteksi diekstrak feature embedding-nya (vektor 128-dimensi)
2. Sistem menghitung jarak Euclidean antara embedding wajah input dengan semua embedding di database
3. Sistem menampilkan hasil kecocokan dengan skor similarity tertinggi (jarak terkecil)

### Deteksi Suku/Etnis
Saat melakukan klasifikasi suku:
1. Wajah yang telah di-preprocessing diproses oleh model CNN (ResNet50 dengan layer kustom)
2. Model mengeluarkan probabilitas untuk setiap kategori suku/etnis (Jawa, Sunda, Melayu)
3. Sistem menampilkan prediksi suku dengan probabilitas tertinggi beserta visualisasi distribusi probabilitas

## Tips Penggunaan
- Untuk hasil terbaik, gunakan gambar wajah frontal dengan pencahayaan yang baik
- Pastikan wajah terlihat jelas dalam gambar (tidak terlalu kecil)
- Untuk meningkatkan akurasi klasifikasi suku, gunakan fitur preprocessing sebelum melakukan training ulang model
- Jika terjadi kegagalan deteksi, coba gunakan gambar dengan resolusi lebih tinggi atau posisi wajah yang lebih jelas

## Troubleshooting
- Error "No face detected": Pastikan gambar memiliki wajah yang jelas dan cukup besar
- Error saat loading model: Pastikan file model (.h5) dan embeddings (.pkl) sudah terbentuk
- Akurasi rendah: Coba lakukan re-training model dengan dataset yang lebih besar atau gunakan fitur preprocessing
