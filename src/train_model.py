import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2

def buat_dataset(folder_dataset):
    """
    Membuat dataframe dari struktur folder dataset
    Format: dataset/[nama]/[suku]/images
    """
    data = []
    for nama in os.listdir(folder_dataset):
        nama_path = os.path.join(folder_dataset, nama)
        if os.path.isdir(nama_path):
            for suku in os.listdir(nama_path):
                suku_path = os.path.join(nama_path, suku)
                if os.path.isdir(suku_path):
                    for image_file in os.listdir(suku_path):
                        if image_file.endswith(('.jpg', '.jpeg', '.png')):
                            image_path = os.path.join(suku_path, image_file)
                            data.append({
                                'path_gambar': image_path,
                                'nama': nama,
                                'suku': suku
                            })
    return pd.DataFrame(data)

def buat_model_klasifikasi_suku(dataset_df, output_dir='models'):
    """
    Membuat dan melatih model klasifikasi suku/etnis
    """
    os.makedirs(output_dir, exist_ok=True)
    output_model_path = os.path.join(output_dir, 'model_klasifikasi_suku.h5')
    
    # Encode label suku
    label_encoder = LabelEncoder()
    dataset_df['suku_encoded'] = label_encoder.fit_transform(dataset_df['suku'])
    
    # Simpan label encoder untuk penggunaan nanti
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    np.save(os.path.join(output_dir, 'label_encoder_classes.npy'), label_encoder.classes_)
    
    # Split data menjadi training dan validation
    train_df, val_df = train_test_split(dataset_df, test_size=0.2, stratify=dataset_df['suku'], random_state=42)
    
    # Data generator untuk augmentasi
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator()
    
    # Parameter untuk generator
    batch_size = 16
    target_size = (224, 224)
    num_classes = len(dataset_df['suku'].unique())
    
    # Generator untuk training
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='path_gambar',
        y_col='suku_encoded',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Generator untuk validasi
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='path_gambar',
        y_col='suku_encoded',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Buat model dengan ResNet50 sebagai base model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Bekukan layer pada base model
    for layer in base_model.layers:
        layer.trainable = False
    
    # Tambahkan layer klasifikasi
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Model final
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Kompilasi model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        output_model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Latih model
    epochs = 50
    steps_per_epoch = len(train_df) // batch_size
    validation_steps = len(val_df) // batch_size
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Plot history training
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Akurasi Model')
    plt.ylabel('Akurasi')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss Model')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    
    # Simpan plot
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/training_history.png')
    
    return model, label_encoder.classes_

def fine_tune_model(model, dataset_df, label_classes, output_dir='models'):
    """
    Fine-tuning model yang sudah dilatih
    """
    output_model_path = os.path.join(output_dir, 'model_klasifikasi_suku_finetuned.h5')
    
    # Buka kembali beberapa layer teratas dari ResNet50 untuk fine-tuning
    for layer in model.layers[-20:]:  # Buka 20 layer terakhir
        layer.trainable = True
    
    # Rekompilasi model dengan learning rate yang lebih kecil
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Split data
    train_df, val_df = train_test_split(dataset_df, test_size=0.2, stratify=dataset_df['suku'], random_state=42)
    
    # Data generator yang sama seperti sebelumnya
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator()
    
    # Parameter untuk generator
    batch_size = 16
    target_size = (224, 224)
    
    # Generator untuk training
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='path_gambar',
        y_col='suku_encoded',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Generator untuk validasi
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='path_gambar',
        y_col='suku_encoded',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        output_model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Fine-tune dengan epoch lebih sedikit
    epochs = 20
    steps_per_epoch = len(train_df) // batch_size
    validation_steps = len(val_df) // batch_size
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Plot history fine-tuning
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Akurasi Model (Fine-tuning)')
    plt.ylabel('Akurasi')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss Model (Fine-tuning)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('outputs/finetuning_history.png')
    
    return model

if __name__ == "__main__":
    # Buat dataset dari folder
    dataset_folder = 'dataset'
    df = buat_dataset(dataset_folder)
    
    print(f"Total data: {len(df)}")
    print(f"Distribusi suku: {df['suku'].value_counts()}")
    
    # Latih model fase pertama
    model, label_classes = buat_model_klasifikasi_suku(df)
    
    # Fine-tuning model
    fine_tuned_model = fine_tune_model(model, df, label_classes)
    
    print("Pelatihan model selesai!")