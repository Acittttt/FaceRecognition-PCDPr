import streamlit as st
import pickle
import cv2
import numpy as np

# Import modul dari folder models
from models.deteksi_wajah import detect_faces
from models.ekstraksi_wajah import extract_face_embedding
from models.perbandingan_wajah import euclidean_distance, compute_similarity_score

@st.cache_data
def load_embeddings(file_path='embeddings.pkl'):
    """
    Memuat data embeddings (nama, etnis, embedding wajah) dari file pkl.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def main():
    st.title("Aplikasi Face Similarity (Kemiripan Wajah)")
    st.write("Unggah gambar untuk mendeteksi dan membandingkan wajah dengan dataset.")

    # Load data embedding
    data = load_embeddings()

    # File uploader (tanpa parameter key acak)
    uploaded_file = st.file_uploader("Pilih Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.write("Uploaded file:", uploaded_file.name)
        
        # Reset pointer file ke awal agar file bisa dibaca dengan benar
        uploaded_file.seek(0)
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        st.write("Jumlah byte file:", len(file_bytes))
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Gagal membaca gambar!")
            return

        # Tampilkan gambar asli
        st.image(image, channels="BGR", caption="Gambar Asli", use_container_width=True)

        # Konversi ke RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize gambar jika resolusi terlalu besar
        MAX_WIDTH = 800
        height, width = rgb_image.shape[:2]
        if width > MAX_WIDTH:
            scaling_factor = MAX_WIDTH / width
            new_dimensions = (int(width * scaling_factor), int(height * scaling_factor))
            rgb_image = cv2.resize(rgb_image, new_dimensions)
            st.write(f"Gambar di-resize menjadi: {new_dimensions}")
        
        # Deteksi wajah
        face_locations = detect_faces(rgb_image)
        st.write(f"Jumlah wajah terdeteksi: {len(face_locations)}")

        threshold = 0.6  # Batas untuk menyatakan match

        if len(face_locations) > 0:
            for idx, face_loc in enumerate(face_locations):
                top, right, bottom, left = face_loc
                # Ekstraksi embedding
                embeddings = extract_face_embedding(rgb_image, face_loc)
                
                if len(embeddings) > 0:
                    embedding_input = embeddings[0]
                    # Lakukan perbandingan dengan dataset
                    all_distances = [euclidean_distance(embedding_input, emb) for emb in data['embeddings']]
                    # Cari jarak minimum
                    min_dist = np.min(all_distances)
                    min_idx = np.argmin(all_distances)
                    # Tentukan apakah match
                    is_match = min_dist < threshold
                    similarity_score = compute_similarity_score(min_dist, threshold)
                    # Dapatkan nama & etnis yang sesuai
                    best_name = data['names'][min_idx] if is_match else "Unknown"
                    best_ethnicity = data['ethnicities'][min_idx] if is_match else "-"
                    
                    # Info output
                    st.write(f"**Wajah ke-{idx+1}:**")
                    st.write(f"- Nama Terdekat: {best_name}")
                    st.write(f"- Etnis: {best_ethnicity}")
                    st.write(f"- Distance: {min_dist:.2f}")
                    st.write(f"- Similarity Score: {similarity_score:.2f} (0-1)")
                    st.write(f"- Match? {'Ya' if is_match else 'Tidak'}")
                    st.write("---")
                    
                    # Gambarkan bounding box di gambar
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                    text = best_name if is_match else "Unknown"
                    cv2.putText(image, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    st.write(f"Tidak bisa mengekstrak embedding dari wajah ke-{idx+1}")

            # Tampilkan hasil dengan bounding box
            st.image(image, channels="BGR", caption="Hasil Deteksi dan Bounding Box", use_container_width=True)
        else:
            st.warning("Tidak ada wajah terdeteksi di gambar ini.")

if __name__ == '__main__':
    main()