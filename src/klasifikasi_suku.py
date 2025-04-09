import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

class KlasifikasiSuku:
    def __init__(self, model_path='models/model_klasifikasi_suku_finetuned.h5', label_path='models/label_encoder_classes.npy'):
        # Load model
        self.model = load_model(model_path)
        # Load label classes
        self.label_classes = np.load(label_path, allow_pickle=True)
        self.target_size = (224, 224)
    
    def preprocess_image(self, img):
        """
        Pra-pemrosesan gambar untuk prediksi
        
        Args:
            img: Gambar dalam format OpenCV (BGR)
            
        Returns:
            Gambar yang telah di-pra-proses
        """
        # Konversi BGR ke RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize
        img_resized = cv2.resize(img_rgb, self.target_size)
        # Preprocess untuk ResNet50
        img_preprocessed = np.expand_dims(img_resized, axis=0)
        img_preprocessed = preprocess_input(img_preprocessed)
        return img_preprocessed
    
    def predict_from_file(self, image_path):
        """
        Prediksi suku dari file gambar
        
        Args:
            image_path: Path ke file gambar
            
        Returns:
            predicted_class: Kelas suku yang diprediksi
            probabilities: Nilai probabilitas untuk setiap kelas
            confidence: Tingkat keyakinan prediksi
        """
        # Load dan preprocess gambar
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Tidak dapat membaca gambar dari {image_path}")
        
        return self.predict_from_image(img)
    
    def predict_from_image(self, img):
        """
        Prediksi suku dari objek gambar
        
        Args:
            img: Gambar dalam format OpenCV (BGR)
            
        Returns:
            predicted_class: Kelas suku yang diprediksi
            probabilities: Dictionary dengan kelas sebagai key dan probabilitas sebagai value
            confidence: Tingkat keyakinan prediksi
        """
        # Preprocess gambar
        preprocessed_img = self.preprocess_image(img)
        
        # Lakukan prediksi
        predictions = self.model.predict(preprocessed_img)[0]
        
        # Dapatkan kelas dengan probabilitas tertinggi
        predicted_index = np.argmax(predictions)
        predicted_class = self.label_classes[predicted_index]
        confidence = predictions[predicted_index]
        
        # Buat dictionary untuk semua probabilitas kelas
        class_probabilities = {}
        for i, prob in enumerate(predictions):
            class_probabilities[self.label_classes[i]] = float(prob)
        
        return predicted_class, class_probabilities, confidence
    
    def get_top_predictions(self, img, top_n=3):
        """
        Mendapatkan top N prediksi
        
        Args:
            img: Gambar dalam format OpenCV (BGR)
            top_n: Jumlah prediksi teratas yang diinginkan
            
        Returns:
            top_classes: List tuple (nama_kelas, probabilitas) untuk N kelas teratas
        """
        _, class_probabilities, _ = self.predict_from_image(img)
        
        # Urutkan berdasarkan probabilitas
        sorted_classes = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # Ambil top N
        top_classes = sorted_classes[:top_n]
        
        return top_classes
    
    def evaluate_batch(self, image_paths, true_labels):
        """
        Evaluasi model pada batch gambar
        
        Args:
            image_paths: List path gambar
            true_labels: List label sebenarnya
            
        Returns:
            accuracy: Akurasi model pada batch
            results: Dictionary hasil evaluasi
        """
        correct = 0
        results = []
        
        for img_path, true_label in zip(image_paths, true_labels):
            try:
                predicted_class, class_probabilities, confidence = self.predict_from_file(img_path)
                is_correct = predicted_class == true_label
                
                if is_correct:
                    correct += 1
                
                results.append({
                    'image_path': img_path,
                    'true_label': true_label,
                    'predicted_label': predicted_class,
                    'confidence': confidence,
                    'is_correct': is_correct
                })
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
        
        accuracy = correct / len(image_paths) if image_paths else 0
        
        return accuracy, results