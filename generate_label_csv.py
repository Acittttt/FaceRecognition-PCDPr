import os
import csv

dataset_path = r'C:\Users\rindi\Downloads\TUBES-SUKU-PCD-PR\TUBES-SUKU-PCD-PR\face_ethnicity_recognition\dataset'
output_csv = 'labels.csv'
allowed_extensions = ['.jpg', '.jpeg', '.png']

# List referensi kata kunci
ekspresi_list = ['tersenyum', 'serius', 'terkejut', 'netral']
sudut_list = ['frontal', 'miring', 'profil']
pencahayaan_list = ['terang', 'redup', 'indoor', 'outdoor']
jarak_list = ['dekat', 'sedang', 'jauh']

def parse_filename(filename):
    # Hapus ekstensi dan pisah berdasarkan "-"
    name_parts = os.path.splitext(filename)[0].lower().split('-')

    ekspresi = None
    sudut = None
    pencahayaan = None
    jarak = None

    for part in name_parts:
        part = part.strip().lower()
        if part in ekspresi_list:
            ekspresi = part
        elif part in sudut_list:
            sudut = part
        elif part in pencahayaan_list:
            pencahayaan = part
        elif part in jarak_list:
            jarak = part

    return ekspresi, sudut, pencahayaan, jarak

def generate_label_csv():
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['path_gambar', 'nama', 'suku', 'ekspresi', 'sudut', 'pencahayaan', 'jarak'])

        for root, _, files in os.walk(dataset_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in allowed_extensions):
                    file_path = os.path.join(root, file).replace("\\", "/")
                    parts = file_path.split('/')
                    if len(parts) >= 4:
                        nama = parts[-3].strip().lower()
                        suku = parts[-2].strip().lower()
                        ekspresi, sudut, pencahayaan, jarak = parse_filename(file)

                        writer.writerow([
                            file_path,
                            nama,
                            suku,
                            ekspresi,
                            sudut,
                            pencahayaan,
                            jarak
                        ])
    print(f"[✓] CSV label berhasil dibuat di: {output_csv}")

if __name__ == "__main__":
    generate_label_csv()