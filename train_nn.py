import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# --- 1. Load Data ---
df = pd.read_csv("student_performance_encoded.csv")
X = df.drop("Preferred_Learning_Style", axis=1).values
y = df["Preferred_Learning_Style"].values

# --- 2. Preprocessing ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_encoded = to_categorical(y)

# --- 3. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# --- 4. Build Model ---
model = Sequential([
    Dense(64, input_shape=(X.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(y_encoded.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

# --- 5. Predict & Evaluate ---
y_pred_probs = model.predict(X_test)
y_pred = y_pred_probs.argmax(axis=1)
y_true = y_test.argmax(axis=1)

acc = accuracy_score(y_true, y_pred)
print(f"Akurasi Model: {acc * 100:.2f}%")

# --- 6. Confusion Matrix ---
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --- 7. Plot Precision, Recall, F1-score ---
report = classification_report(y_true, y_pred, output_dict=True)
metrics_df = pd.DataFrame(report).transpose().drop(['accuracy', 'macro avg', 'weighted avg'])

plt.figure(figsize=(8, 5))
metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar', legend=True)
plt.title("Precision, Recall, and F1-Score per Class")
plt.xlabel("Kelas")
plt.ylabel("Nilai")
plt.xticks(rotation=0)
plt.ylim(0, 1.1)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
