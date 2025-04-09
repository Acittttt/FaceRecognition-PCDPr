import numpy as np

def euclidean_distance(embedding1, embedding2):
    """
    Menghitung jarak Euclidean antara dua vektor embedding.
    """
    return np.linalg.norm(embedding1 - embedding2)

def compute_similarity_score(distance, threshold=0.6):
    """
    Mengonversi jarak (distance) menjadi skor kemiripan (0-1).
    Semakin kecil jarak, semakin tinggi skor.
    Jika jarak melebihi threshold, skor = 0 (no match).
    """
    if distance > threshold:
        return 0.0
    # Misal pendekatan linear: score 1.0 saat distance=0, score 0.0 saat distance=threshold
    return 1.0 - (distance / threshold)
