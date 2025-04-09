from PIL import Image
import os

# Ganti path ini ke folder utama dataset kamu
folder_path = r'C:\Users\rindi\Downloads\TUBES-SUKU-PCD-PR\TUBES-SUKU-PCD-PR\face_ethnicity_recognition\dataset'
allowed_extensions = ['.jpg', '.jpeg', '.png']

def cek_gambar(folder):
    for root, _, files in os.walk(folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in allowed_extensions):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        if width < 224 or height < 224:
                            print(f"[!] Resolusi kecil: {file_path} ({width}x{height})")
                        else:
                            print(f"[OK] {file_path} ({width}x{height})")
                except Exception as e:
                    print(f"[ERROR] Gagal membaca {file_path} -> {e}")

if __name__ == "__main__":
    cek_gambar(folder_path)
