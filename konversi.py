import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np

# Load dataset
df = pd.read_csv("learning_style_dataset_1000.csv")

# Konversi A/B/C menjadi angka
map_values = {'A': 0, 'B': 1, 'C': 2}
for col in df.columns[:-1]:
    df[col] = df[col].map(map_values)

# Encode label
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])  # Visual=2, Auditory=0, Kinesthetic=1

# Cek distribusi label
print("Distribusi Label:")
print(df['Label'].value_counts())

# Cek korelasi sederhana antar fitur
print("\nKorelasi antar fitur:")
print(df.corr())

# Simpan dataset hasil konversi
# df.to_csv("learning_style_dataset_preprocessed.csv", index=False)
print("\n✅ Dataset telah dikonversi dan disimpan sebagai 'learning_style_dataset_preprocessed.csv'")
