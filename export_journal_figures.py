import os
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Buat direktori output gambar
OUTPUT_DIR = "figures_jurnal"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style publikasi ilmiah
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# ==============================================================================
# 1. GENERATE GAMBAR 1: PERBANDINGAN SAMPEL CITRA X-RAY (NORMAL VS PNEUMONIA)
# ==============================================================================
print("[1/4] Mengenerate Gambar 1: Sampel Citra X-Ray (Normal vs Pneumonia)...")

normal_path = "chest_xray/test/NORMAL/IM-0031-0001.jpeg"
pneumonia_path = "chest_xray/test/PNEUMONIA/person100_bacteria_482.jpeg"

# Fallback jika path tidak ditemukan
if not os.path.exists(normal_path):
    normal_candidates = [f for f in os.listdir("chest_xray/train/NORMAL") if f.endswith(('.jpg', '.jpeg', '.png'))]
    normal_path = os.path.join("chest_xray/train/NORMAL", normal_candidates[0])
if not os.path.exists(pneumonia_path):
    pneu_candidates = [f for f in os.listdir("chest_xray/train/PNEUMONIA") if f.endswith(('.jpg', '.jpeg', '.png'))]
    pneumonia_path = os.path.join("chest_xray/train/PNEUMONIA", pneu_candidates[0])

img_normal_raw = cv2.imread(normal_path)
img_normal_rgb = cv2.cvtColor(img_normal_raw, cv2.COLOR_BGR2RGB)
img_normal_gray = cv2.cvtColor(img_normal_raw, cv2.COLOR_BGR2GRAY)
img_normal_resized = cv2.resize(img_normal_gray, (128, 128), interpolation=cv2.INTER_AREA)

img_pneu_raw = cv2.imread(pneumonia_path)
img_pneu_rgb = cv2.cvtColor(img_pneu_raw, cv2.COLOR_BGR2RGB)
img_pneu_gray = cv2.cvtColor(img_pneu_raw, cv2.COLOR_BGR2GRAY)
img_pneu_resized = cv2.resize(img_pneu_gray, (128, 128), interpolation=cv2.INTER_AREA)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))

axes[0, 0].imshow(img_normal_rgb)
axes[0, 0].set_title("(a) Normal - Citra Asli", pad=8)
axes[0, 0].axis('off')

axes[0, 1].imshow(img_normal_resized, cmap='gray')
axes[0, 1].set_title("(b) Normal - Preprocessed (128x128)", pad=8)
axes[0, 1].axis('off')

axes[1, 0].imshow(img_pneu_rgb)
axes[1, 0].set_title("(c) Pneumonia - Citra Asli", pad=8)
axes[1, 0].axis('off')

axes[1, 1].imshow(img_pneu_resized, cmap='gray')
axes[1, 1].set_title("(d) Pneumonia - Preprocessed (128x128)", pad=8)
axes[1, 1].axis('off')

plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, "gambar1_sampel_xray.png")
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   -> Tersimpan di: {fig1_path}")

# ==============================================================================
# 2. LOAD DATASET DAN EVALUASI 4 MODEL
# ==============================================================================
print("\n[2/4] Menghitung prediksi dan confusion matrix untuk 4 model...")

df = pd.read_csv("glcm_features_pneumonia.csv")
feature_cols = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'asm']
X = df[feature_cols]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = joblib.load('glcm_scaler.pkl')
model_svm = joblib.load('model_svm.pkl')
model_rf = joblib.load('model_rf.pkl')
model_knn = joblib.load('model_knn.pkl')
model_voting = joblib.load('model_voting.pkl')

X_test_scaled = scaler.transform(X_test)

models = {
    'SVM': model_svm,
    'Random Forest': model_rf,
    'KNN (k=5)': model_knn,
    'Ensemble Voting': model_voting
}

predictions = {}
metrics_list = []
cms = {}

for name, model in models.items():
    preds = model.predict(X_test_scaled)
    predictions[name] = preds
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    cms[name] = cm
    
    tn, fp, fn, tp = cm.ravel()
    metrics_list.append({
        'Model': name,
        'Accuracy (%)': round(acc * 100, 2),
        'Precision (%)': round(prec * 100, 2),
        'Recall (%)': round(rec * 100, 2),
        'F1-Score (%)': round(f1 * 100, 2),
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp
    })

df_metrics = pd.DataFrame(metrics_list)
print("\n=== TABEL HASIL EVALUASI ===")
print(df_metrics.to_string(index=False))

# Simpan tabel ke CSV untuk jurnal
df_metrics.to_csv(os.path.join(OUTPUT_DIR, "tabel_evaluasi_model.csv"), index=False)

# ==============================================================================
# 3. GENERATE GAMBAR 2: CONFUSION MATRIX 4 MODEL (2x2 GRID)
# ==============================================================================
print("\n[3/4] Mengenerate Gambar 2: Confusion Matrix (Grid 2x2)...")

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
axes = axes.flatten()

letters = ['(a)', '(b)', '(c)', '(d)']
for idx, (name, cm) in enumerate(cms.items()):
    ax = axes[idx]
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
        annot_kws={'size': 14, 'weight': 'bold'},
        xticklabels=['Normal', 'Pneumonia'],
        yticklabels=['Normal', 'Pneumonia']
    )
    ax.set_title(f"{letters[idx]} {name}", fontsize=13, pad=10, weight='semibold')
    ax.set_xlabel("Kelas Prediksi", fontsize=11, labelpad=8)
    ax.set_ylabel("Kelas Aktual", fontsize=11, labelpad=8)

plt.tight_layout(pad=2.5)
fig2_path = os.path.join(OUTPUT_DIR, "gambar2_confusion_matrix.png")
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   -> Tersimpan di: {fig2_path}")

# ==============================================================================
# 4. GENERATE GAMBAR 3: FEATURE IMPORTANCE RANDOM FOREST
# ==============================================================================
print("\n[4/4] Mengenerate Gambar 3 & 4: Analisis Fitur & Komparasi Metrik...")

importances = model_rf.feature_importances_
df_imp = pd.DataFrame({
    'Feature': [f.capitalize() for f in feature_cols],
    'Importance': importances
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.barh(df_imp['Feature'], df_imp['Importance'], color='#1f77b4', edgecolor='black', alpha=0.85)

for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, f"{width:.4f}",
            va='center', ha='left', fontsize=10, weight='semibold')

ax.set_xlim(0, max(importances) * 1.15)
ax.set_title("Tingkat Kepentingan Fitur GLCM (Random Forest Feature Importance)", pad=12, weight='semibold')
ax.set_xlabel("Skor Kepentingan (Importance Score)")
ax.set_ylabel("Fitur Tekstur GLCM")
ax.grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, "gambar3_feature_importance.png")
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   -> Tersimpan di: {fig3_path}")

# ==============================================================================
# 5. GENERATE GAMBAR 4: GROUPED BAR CHART KOMPARASI METRIK
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(df_metrics))
width = 0.18

rects1 = ax.bar(x - 1.5*width, df_metrics['Accuracy (%)'], width, label='Accuracy', color='#2b5c8f')
rects2 = ax.bar(x - 0.5*width, df_metrics['Precision (%)'], width, label='Precision', color='#41924b')
rects3 = ax.bar(x + 0.5*width, df_metrics['Recall (%)'], width, label='Recall', color='#e26d2d')
rects4 = ax.bar(x + 1.5*width, df_metrics['F1-Score (%)'], width, label='F1-Score', color='#8c564b')

ax.set_ylabel('Nilai Persentase (%)', fontsize=11)
ax.set_title('Perbandingan Kinerja Model Berdasarkan Metrik Evaluasi', fontsize=13, weight='semibold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(df_metrics['Model'], fontsize=11, weight='semibold')
ax.set_ylim(70, 100)
ax.legend(loc='lower right', frameon=True)
ax.grid(axis='y', linestyle='--', alpha=0.4)

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=0)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

plt.tight_layout()
fig4_path = os.path.join(OUTPUT_DIR, "gambar4_komparasi_metrik.png")
plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   -> Tersimpan di: {fig4_path}")

print("\n Seluruh gambar dan tabel berhasil dibuat dengan resolusi 300 DPI di folder 'figures_jurnal/'!")
