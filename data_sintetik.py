import pandas as pd
import random

# Pilihan jawaban
choices = ['A', 'B', 'C']

# Fungsi untuk menentukan label dari jawaban
def determine_label(answers):
    counts = {'A': answers.count('A'), 'B': answers.count('B'), 'C': answers.count('C')}
    max_score = max(counts.values())
    dominant = [k for k, v in counts.items() if v == max_score]
    return random.choice(dominant)  # Jika seri, pilih acak dari yang tertinggi

# Generate dataset
data = []
for _ in range(10000):
    answers = [random.choice(choices) for _ in range(10)]
    label = determine_label(answers)
    score_A = answers.count('A')
    score_B = answers.count('B')
    score_C = answers.count('C')
    data.append(answers + [score_A, score_B, score_C, label])

# Kolom dataset
columns = [f'Q{i+1}' for i in range(10)] + ['Score_A', 'Score_B', 'Score_C', 'Label']

# Simpan ke DataFrame
df = pd.DataFrame(data, columns=columns)
df.to_csv("learning_style_dataset_patterned-10000.csv", index=False)
df.head()
