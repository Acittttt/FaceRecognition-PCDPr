import face_recognition

def detect_faces(rgb_image):
    """
    Mendeteksi wajah pada citra RGB dan 
    mengembalikan list bounding box (top, right, bottom, left).
    """
    # face_locations akan berisi list tuple (top, right, bottom, left)
    face_locations = face_recognition.face_locations(rgb_image, number_of_times_to_upsample=1, model="cnn")
    return face_locations
