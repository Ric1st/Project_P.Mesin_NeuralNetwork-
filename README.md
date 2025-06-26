# 🧠 Learning Style Classification using Neural Network

Proyek ini bertujuan untuk mengklasifikasikan gaya belajar mahasiswa (Visual, Auditory, Kinesthetic) berdasarkan jawaban kuesioner 10 soal. Model dibangun menggunakan Neural Network dengan TensorFlow dan disediakan antarmuka web menggunakan Flask.

---

## 📁 Struktur Proyek

Neural Network/
├── app.py # Aplikasi Flask
├── model_gaya_belajar.h5 # Model hasil pelatihan Neural Network
├── learning_style_dataset_patterned-10000.csv # Dataset yang digunakan
├── requirements.txt # Daftar dependensi
├── README.md # Dokumentasi proyek
├── student_performance_encoded.csv # Dataset Kaggle yang sudah diencoding dan dibersihkan
├── model_nn.py # kode untuk train model dari dataset kaggle sekaligus cek akurasinya
├── model_nn2.py # kode untuk train model dari dataset sintetik sekaligus cek akurasinya
├── data_sintetik.py # kode untuk membuat data sintetik
├── templates/
│ ├── index.html # Halaman input pengguna
│ └── result.html # Halaman hasil prediksi

---

## 📊 Dataset

- Dataset terdiri dari 1000 data sintetik berpola (`learning_style_dataset_patterned.csv`)
- Setiap entri berisi 10 jawaban soal pilihan (A/B/C) dan label gaya belajar (`Visual`, `Auditory`, `Kinesthetic`)
- Dataset dibuat secara otomatis berdasarkan pola dominasi jawaban

---

## 🧠 Arsitektur Model

- Model Neural Network dengan 2 hidden layer
  - Input layer: 13 neuron
  - Hidden layer 1: 1024 neuron, ReLU
  - Hidden layer 2: 1024 neuron, ReLU
  - Output layer: 3 neuron (Softmax)
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Akurasi Model: **92%**

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Clone repository

```bash
git clone https://github.com/Ric1st/Project_P.Mesin_NeuralNetwork-.git
cd Project_P.Mesin_NeuralNetwork-

python -m venv env
env\Scripts\activate  # Untuk Windows

pip install -r requirements.txt

python app.py


Buka browser di http://127.0.0.1:5000/

```

---

🔗 Teknologi yang Digunakan
Python

TensorFlow + Keras

Scikit-learn

Flask

HTML + Bootstrap (untuk antarmuka)

---

📌 Catatan
File model_gaya_belajar.h5 telah disimpan dan digunakan saat prediksi

Untuk menjalankan kembali pelatihan, jalankan script Python training model (model_nn2.py jika tersedia)

---

🧑‍💻 Kontributor
Nama:
1.Muhammad Rifki Hendarto
2.Richard Christoper Subianto
3.Firnanda Rahmawati
4.Najwa Kaila Nuraisyah
5.Virgina Staraina

NIM:

1. A11.2023.14921
2. A11.2023.14922
3. A11.2023.15373
4. A11.2023.15379
5. A11.2023.15381

Prodi: S1 - Teknik Informatika - Pembelajaran Mesin - A11.4407
