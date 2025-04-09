import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def buat_dataset(folder_dataset):
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

def buat_generator(train_df, val_df, batch_size=16, target_size=(224, 224)):
    train_datagen = ImageDataGenerator(
        rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
        shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_dataframe(
        train_df, x_col='path_gambar', y_col='suku_encoded',
        target_size=target_size, batch_size=batch_size, class_mode='categorical'
    )
    val_generator = val_datagen.flow_from_dataframe(
        val_df, x_col='path_gambar', y_col='suku_encoded',
        target_size=target_size, batch_size=batch_size, class_mode='categorical'
    )
    return train_generator, val_generator

def buat_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history, filename):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Akurasi Model')
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi')
    plt.legend(['Train', 'Validation'])

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/{filename}')

def buat_model_klasifikasi_suku(dataset_df, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)
    label_encoder = LabelEncoder()
    dataset_df['suku_encoded'] = label_encoder.fit_transform(dataset_df['suku']).astype(str)
    np.save(os.path.join(output_dir, 'label_encoder_classes.npy'), label_encoder.classes_)

    train_df, val_df = train_test_split(dataset_df, test_size=0.2, stratify=dataset_df['suku'], random_state=42)
    train_gen, val_gen = buat_generator(train_df, val_df)
    num_classes = len(label_encoder.classes_)
    model = buat_model(num_classes)

    checkpoint = ModelCheckpoint(os.path.join(output_dir, 'model_klasifikasi_suku.h5'),
                                 monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    history = model.fit(train_gen, epochs=50, validation_data=val_gen,
                        callbacks=[checkpoint, early_stopping])

    plot_history(history, 'training_history.png')
    return model, label_encoder.classes_, dataset_df

def fine_tune_model(model, dataset_df, output_dir='models'):
    for layer in model.layers[-20:]:
        layer.trainable = True
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    train_df, val_df = train_test_split(dataset_df, test_size=0.2, stratify=dataset_df['suku'], random_state=42)
    train_gen, val_gen = buat_generator(train_df, val_df)

    checkpoint = ModelCheckpoint( os.path.join(output_dir, 'model_klasifikasi_suku_finetuned.h5').replace("\\", "/"),
    monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    history = model.fit(train_gen, epochs=20, validation_data=val_gen, callbacks=[checkpoint, early_stopping])
    plot_history(history, 'finetuning_history.png')
    return model

if __name__ == "__main__":
    df = buat_dataset('dataset')
    print(f"Total data: {len(df)}\nDistribusi suku:\n{df['suku'].value_counts()}")
    model, label_classes, df_encoded = buat_model_klasifikasi_suku(df)
    fine_tuned_model = fine_tune_model(model, df_encoded)
    print("Pelatihan model selesai!")
