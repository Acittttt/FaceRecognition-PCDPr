import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

def load_image(image_path, target_size=(224, 224)):
    """
    Memuat gambar dan melakukan resize
    
    Args:
        image_path: Path ke file gambar
        target_size: Ukuran target untuk resize
        
    Returns:
        Gambar yang sudah dimuat dan di-resize
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"File tidak dapat dibaca: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

def preprocess_for_resnet(img_array):
    """
    Pra-pemrosesan gambar sesuai kebutuhan ResNet50
    
    Args:
        img_array: Array gambar RGB
        
    Returns:
        Array gambar yang telah di pra-proses
    """
    img = np.expand_dims(img_array, axis=0)
    img = preprocess_input(img)
    return img

def scan_dataset(dataset_dir):
    """
    Memindai folder dataset dan menghasilkan statistik
    
    Args:
        dataset_dir: Direktori dataset
        
    Returns:
        Dictionary berisi statistik dataset
    """
    stats = {
        'total_subjects': 0,
        'total_images': 0,
        'suku_stats': {}
    }
    
    for nama in os.listdir(dataset_dir):
        nama_path = os.path.join(dataset_dir, nama)
        if os.path.isdir(nama_path):
            stats['total_subjects'] += 1
            
            for suku in os.listdir(nama_path):
                suku_path = os.path.join(nama_path, suku)
                if os.path.isdir(suku_path):
                    # Hitung gambar
                    images = [f for f in os.listdir(suku_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                    num_images = len(images)
                    stats['total_images'] += num_images
                    
                    # Update statistik suku
                    if suku not in stats['suku_stats']:
                        stats['suku_stats'][suku] = {
                            'subjects': 0,
                            'images': 0
                        }
                    
                    stats['suku_stats'][suku]['subjects'] += 1
                    stats['suku_stats'][suku]['images'] += num_images
    
    return stats

def create_metadata_csv(dataset_dir, output_file='dataset_metadata.csv'):
    """
    Membuat file CSV berisi metadata dari dataset
    
    Args:
        dataset_dir: Direktori dataset
        output_file: Nama file output
        
    Returns:
        DataFrame metadata
    """
    metadata = []
    
    for nama in os.listdir(dataset_dir):
        nama_path = os.path.join(dataset_dir, nama)
        if os.path.isdir(nama_path):
            for suku in os.listdir(nama_path):
                suku_path = os.path.join(nama_path, suku)
                if os.path.isdir(suku_path):
                    for image_file in os.listdir(suku_path):
                        if image_file.endswith(('.jpg', '.jpeg', '.png')):
                            image_path = os.path.join(suku_path, image_file)
                            
                            # Get image dimensions and size
                            try:
                                img = cv2.imread(image_path)
                                height, width, _ = img.shape
                                file_size = os.path.getsize(image_path) / 1024  # KB
                                
                                metadata.append({
                                    'image_path': image_path,
                                    'nama': nama,
                                    'suku': suku,
                                    'filename': image_file,
                                    'width': width,
                                    'height': height,
                                    'size_kb': file_size
                                })
                            except Exception as e:
                                print(f"Error processing {image_path}: {str(e)}")
    
    # Convert to DataFrame and save
    df = pd.DataFrame(metadata)
    df.to_csv(output_file, index=False)
    
    return df

def ensure_directories():
    """
    Memastikan direktori yang diperlukan tersedia
    """
    required_dirs = ['models', 'outputs', 'src']
    for dir_name in required_dirs:
        os.makedirs(dir_name, exist_ok=True)