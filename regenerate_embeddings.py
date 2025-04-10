import os
import cv2
import face_recognition
import pickle
import numpy as np

def regenerate_embeddings(dataset_dir='dataset', output_file='embeddings.pkl'):
    """
    Membaca seluruh gambar dari dataset, mendeteksi wajah, mengekstrak embedding,
    dan menyimpan data (nama, etnis, embedding) ke file pickle.
    
    Struktur folder dataset diharapkan seperti:
      dataset/
          Ethnic1/
              Person1/
                  image1.jpg, image2.jpg, ...
              Person2/
          Ethnic2/
              Person3/
          ...
    """
    data = {
        'names': [],
        'ethnicities': [],
        'embeddings': []
    }
    
    # Iterasi untuk setiap folder etnis
    for ethnicity in os.listdir(dataset_dir):
        ethnicity_path = os.path.join(dataset_dir, ethnicity)
        if not os.path.isdir(ethnicity_path):
            continue
        
        # Iterasi untuk setiap folder nama (subjek)
        for person in os.listdir(ethnicity_path):
            person_path = os.path.join(ethnicity_path, person)
            if not os.path.isdir(person_path):
                continue
            
            # Proses setiap gambar dalam folder subjek
            for file in os.listdir(person_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_path, file)
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Cannot read image: {image_path}")
                        continue

                    # Konversi ke RGB (karena face_recognition memerlukan RGB)
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Deteksi wajah; jika terdapat lebih dari satu wajah, kita ambil wajah pertama
                    face_locations = face_recognition.face_locations(rgb_image)
                    if len(face_locations) == 0:
                        print(f"No face detected in {image_path}")
                        continue
                    
                    # Ekstraksi embedding wajah, gunakan wajah pertama
                    face_encodings = face_recognition.face_encodings(rgb_image, known_face_locations=face_locations)
                    if len(face_encodings) == 0:
                        print(f"Failed to extract embedding from {image_path}")
                        continue
                    embedding = face_encodings[0]
                    
                    # Simpan nama (subjek), etnis, dan embedding ke dalam dictionary data
                    data['names'].append(person)
                    data['ethnicities'].append(ethnicity)
                    data['embeddings'].append(embedding)
                    
                    print(f"Processed {image_path}")
    
    # Simpan data ke file pickle
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Successfully saved {len(data['embeddings'])} embeddings to {output_file}")

if __name__ == '__main__':
    regenerate_embeddings()