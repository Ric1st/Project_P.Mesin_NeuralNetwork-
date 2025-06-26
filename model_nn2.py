import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score

# Load updated dataset dengan pola terarah
df = pd.read_csv("learning_style_dataset_patterned-10000.csv") #92%

# df = pd.read_csv("learning_style_dataset_preprocessed.csv") #78%

# Konversi A/B/C ke numerik untuk pertanyaan Q1–Q10
mapping = {'A': 0, 'B': 1, 'C': 2}
for col in [f"Q{i+1}" for i in range(10)]:
    df[col] = df[col].map(mapping)
    #Kalau pakai yang 92% harus aktifkan ini

# Label encoding untuk kolom Label
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

# Pisahkan fitur dan label
X = df.drop("Label", axis=1).values
y = df["Label"].values

# One-hot encode label
y_encoded = to_categorical(y)

# Split data 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Bangun model Neural Network dengan input 13 fitur
model = Sequential([
    Dense(1024, input_shape=(13,), activation='relu'), #kalau 78% pakai in_shape 10 sesuai fitur di dataset, kalau 92% pakai 13
    Dense(1024, activation='relu'),
    Dense(3, activation='softmax')  # 3 kelas: Visual, Auditory, Kinesthetic
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Latih model
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1) #hasilnya sama dengan epochs 300

# Evaluasi model
y_pred_probs = model.predict(X_test)
y_pred = y_pred_probs.argmax(axis=1)
y_true = y_test.argmax(axis=1)

# Hitung akurasi
acc = accuracy_score(y_true, y_pred)
print(f"\nAkurasi Model: {acc * 100:.2f}%")

# Laporan klasifikasi
print("\nLaporan Klasifikasi:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# Simpan model ke file .h5
# model.save("model_gaya_belajar.h5") # ini dimatikan karena model sudah ada agar model tidak terupdate lagi
print(f"✅ Model Neural Network berhasil disimpan.")
# Asumsi setelah preprocessing
