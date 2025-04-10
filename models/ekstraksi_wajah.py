import face_recognition

def extract_face_embedding(rgb_image, face_location):
    """
    Mengekstrak embedding (vektor fitur) dari citra RGB,
    berdasarkan lokasi wajah (top, right, bottom, left).
    Mengembalikan list embedding (bisa berisi satu atau lebih).
    """
    # face_encodings mengembalikan list embedding dengan ukuran 128 dimensi (default face_recognition).
    encodings = face_recognition.face_encodings(rgb_image, known_face_locations=[face_location])
    return encodings
