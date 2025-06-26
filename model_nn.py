import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# --- 1. Load Dataset ---
df = pd.read_csv("student_performance_encoded.csv")

# --- 2. Pisahkan Fitur dan Label ---
X = df.drop("Preferred_Learning_Style", axis=1).values
y = df["Preferred_Learning_Style"].values  # Sudah numerik

# --- 3. Normalisasi Fitur ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. One-Hot Encode Label ---
y_encoded = to_categorical(y)

# --- 5. Split Dataset 80:20 ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# --- 6. Bangun Model Neural Network ---
model = Sequential([
    Dense(64, input_shape=(X.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(y_encoded.shape[1], activation='softmax')  # Jumlah kelas
])

# --- 7. Compile Model ---
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- 8. Train Model ---
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# --- 9. Evaluasi Model ---
y_pred_probs = model.predict(X_test)
y_pred = y_pred_probs.argmax(axis=1)
y_true = y_test.argmax(axis=1)

# --- 10. Akurasi & Tabel Laporan ---
acc = accuracy_score(y_true, y_pred)
print(f"\nAkurasi Model: {acc * 100:.2f}%\n")

# Tampilkan Laporan Klasifikasi sebagai Tabel
report = classification_report(y_true, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

print("Laporan Klasifikasi:")
print(df_report.round(2))  # Buat tampil rapi 2 angka desimal

# --- 11. Simpan Model ---
# model.save("model_gaya_belajar.h5")
print("\n✅ Model Neural Network berhasil disimpan.")
