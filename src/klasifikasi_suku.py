import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

class KlasifikasiSuku:
    def __init__(self, model_path='models/model_klasifikasi_suku.h5', label_path='models/label_encoder_classes.npy'):
        self.model = load_model(model_path)
        self.label_classes = np.load(label_path, allow_pickle=True)
        self.target_size = (224, 224)

    def preprocess_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.target_size)
        img_array = np.expand_dims(img_resized, axis=0)
        return preprocess_input(img_array)

    def predict_from_file(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Tidak dapat membaca gambar dari {image_path}")
        return self.predict_from_image(img)

    def predict_from_image(self, img):
        img_processed = self.preprocess_image(img)
        probabilities = self.model.predict(img_processed)[0]
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.label_classes[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        # KONVERSI ke dictionary
        class_probabilities = dict(zip(self.label_classes, probabilities))

        return predicted_class, class_probabilities, confidence
