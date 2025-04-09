import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

class VisualisasiHasil:
    def __init__(self):
        pass
    
    def plot_probability_bar(self, class_probabilities, title="Distribusi Probabilitas Suku/Etnis"):
        """
        Membuat plot batang untuk probabilitas setiap kelas
        
        Args:
            class_probabilities: Dictionary dengan kelas sebagai key dan probabilitas sebagai value
            title: Judul plot
            
        Returns:
            Gambar plot dalam format BytesIO
        """
        plt.figure(figsize=(10, 6))
        
        # Sortir probabilitas dari tertinggi ke terendah
        sorted_probabilities = {k: v for k, v in sorted(class_probabilities.items(), key=lambda item: item[1], reverse=True)}
        
        # Plot bar chart
        bars = plt.bar(sorted_probabilities.keys(), sorted_probabilities.values(), color='skyblue')
        
        # Tambahkan nilai di atas bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.2%}', ha='center', va='bottom', rotation=0)
        
        plt.title(title)
        plt.xlabel('Suku/Etnis')
        plt.ylabel('Probabilitas')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Konversi plot ke bytes untuk ditampilkan di Streamlit
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return buf
    
    def create_result_image(self, image, predicted_class, confidence):
        """
        Membuat gambar hasil dengan label prediksi
        
        Args:
            image: Gambar asli (numpy array atau PIL Image)
            predicted_class: Kelas yang diprediksi
            confidence: Nilai confidence prediksi
            
        Returns:
            Gambar dengan label dalam format PIL Image
        """
        # Konversi ke PIL Image jika perlu
        if isinstance(image, np.ndarray):
            # Pastikan gambar dalam format RGB
            if image.shape[2] == 3:
                if cv2.imwrite('temp.jpg', image):  # Check if BGR
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Buat copy gambar untuk ditambahkan teks
        result_image = pil_image.copy()
        draw = ImageDraw.Draw(result_image)
        
        # Coba load font, gunakan default jika gagal
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        
        # Tambahkan text
        text = f"Prediksi: {predicted_class} ({confidence:.2%})"
        
        # Dapatkan ukuran text
        try:
            text_width, text_height = draw.textsize(text, font=font)
        except AttributeError:
            # Untuk PIL versi terbaru
            text_width, text_height = font.getsize(text)
        
        # Posisi text (di bawah gambar)
        position = (10, pil_image.height - text_height - 10)
        
        # Tambah background gelap untuk text
        draw.rectangle(
            [position[0], position[1], position[0] + text_width, position[1] + text_height],
            fill=(0, 0, 0, 180)
        )
        
        # Tambahkan text
        draw.text(position, text, (255, 255, 255), font=font)
        
        return result_image