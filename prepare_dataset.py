import os
import cv2
import pickle
import numpy as np

# Import modul dari folder models
from models.deteksi_wajah import detect_faces
from models.ekstraksi_wajah import extract_face_embedding

def prepare_dataset(dataset_dir='dataset', output_file='embeddings.pkl'):
    """
    Membaca dataset di folder dataset_dir,
    mendeteksi wajah, mengekstrak embedding,
    lalu menyimpan (nama, etnis, embeddings) ke output_file (pkl).
    """
    
    data = {
        'names': [],
        'ethnicities': [],
        'embeddings': []
    }
    
    # Loop etnis
    for ethnicity in os.listdir(dataset_dir):
        ethnicity_path = os.path.join(dataset_dir, ethnicity)
        if not os.path.isdir(ethnicity_path):
            continue
        
        # Loop nama orang
        for person_name in os.listdir(ethnicity_path):
            person_path = os.path.join(ethnicity_path, person_name)
            if not os.path.isdir(person_path):
                continue
            
            # Loop gambar
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                image = cv2.imread(img_path)
                
                if image is None:
                    print(f"Gagal membaca {img_path}")
                    continue
                
                # Konversi ke RGB untuk face_recognition
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Deteksi wajah
                face_locations = detect_faces(rgb_image)
                if len(face_locations) == 0:
                    print(f"Tidak ada wajah terdeteksi di {img_path}")
                    continue
                
                # Ambil embedding dari wajah pertama (asumsi satu wajah per foto)
                embeddings = extract_face_embedding(rgb_image, face_locations[0])
                if len(embeddings) == 0:
                    print(f"Gagal ekstrak embedding dari {img_path}")
                    continue
                
                # Simpan hasil
                data['names'].append(person_name)
                data['ethnicities'].append(ethnicity)
                data['embeddings'].append(embeddings[0])
    
    # Simpan data ke pkl
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Berhasil menyimpan {len(data['embeddings'])} embedding ke {output_file}")

if __name__ == '__main__':
    prepare_dataset()