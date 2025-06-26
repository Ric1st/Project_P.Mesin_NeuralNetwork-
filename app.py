from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('model_gaya_belajar.h5')

labels = ['Auditory', 'Kinesthetic', 'Visual']

pertanyaan = [
    'Saat belajar sesuatu yang baru, kamu lebih suka:',
    'Kamu paling mudah mengingat informasi ketika:',
    'Saat presentasi, kamu lebih suka membuat:',
    'Ketika membaca buku teks, kamu biasanya:',
    'Jika harus menghafal, kamu lebih suka:',
    'Saat belajar di kelas, kamu merasa paling fokus saat:',
    'Kalau kamu membeli barang, kamu cenderung:',
    'Ketika membuat catatan:',
    'Kegiatan favorit saat belajar:',
    'Jika sedang bingung, kamu cenderung:',
]

pernyataan = [
    {"A": "Melihat diagram atau ilustrasi", "B": "Mendengar penjelasan dari orang lain", "C": "Mencoba langsung atau praktik"},
    {"A": "Melihat tulisan atau gambar", "B": "Mendengarkan rekaman atau penjelasan", "C": "Melakukan kegiatan atau eksperimen"},
    {"A": "Slide penuh gambar dan poin visual", "B": "Narasi dan penjelasan verbal", "C": "Demonstrasi atau simulasi langsung"},
    {"A": "Membuat catatan berwarna atau mindmap", "B": "Membaca keras atau mendengarkan rekaman", "C": "Menggerakkan tangan atau mencoba contoh"},
    {"A": "Melihat rangkuman atau flashcard", "B": "Mengulang-ulang dengan suara", "C": "Menulis sambil berjalan atau bergerak"},
    {"A": "Melihat slide presentasi atau video", "B": "Mendengar dosen menjelaskan", "C": "Melakukan praktikum atau diskusi langsung"},
    {"A": "Membaca detail dan melihat foto produk", "B": "Bertanya ke penjual atau teman", "C": "Mencoba sendiri barangnya"},
    {"A": "Pakai warna dan bentuk diagram", "B": "Menulis dalam bentuk poin-poin suara", "C": "Sering sambil bergerak atau menulis cepat"},
    {"A": "Membuat poster/mindmap", "B": "Mendengarkan podcast/pemutaran suara", "C": "Mengerjakan tugas langsung/praktikum"},
    {"A": "Mencari penjelasan visual", "B": "Bertanya dan mendengarkan", "C": "Mencoba sampai menemukan sendiri"}
]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", pertanyaan=pertanyaan, pernyataan=pernyataan)


@app.route('/predict', methods=['POST'])
def predict():
    jawaban = []
    mapping = {'A': 0, 'B': 1, 'C': 2}

    for i in range(1, 11):
        val = request.form.get(f'Q{i}')
        jawaban.append(mapping.get(val, 0))  # default 0 kalau kosong

    score_a = jawaban.count(0)
    score_b = jawaban.count(1)
    score_c = jawaban.count(2)

    X_input = np.array([jawaban + [score_a, score_b, score_c]])

    pred = model.predict(X_input)
    result = labels[np.argmax(pred)]

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
