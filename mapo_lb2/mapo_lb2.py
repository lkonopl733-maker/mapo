
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Вихідні дані для Варіанта 3
data_v3 = [
    {"No": 3,  "Class": "A", "Area": 1100, "Perimeter": 128, "Brightness": 90},
    {"No": 4,  "Class": "B", "Area": 1640, "Perimeter": 174, "Brightness": 188},
    {"No": 5,  "Class": "A", "Area": 1210, "Perimeter": 136, "Brightness": 82},
    {"No": 6,  "Class": "B", "Area": 1730, "Perimeter": 181, "Brightness": 169},
    {"No": 7,  "Class": "A", "Area": 1030, "Perimeter": 125, "Brightness": 96},
    {"No": 8,  "Class": "B", "Area": 1580, "Perimeter": 170, "Brightness": 194},
    {"No": 9,  "Class": "A", "Area": 1150, "Perimeter": 133, "Brightness": 77},
    {"No": 10, "Class": "B", "Area": 1820, "Perimeter": 190, "Brightness": 181},
]

df = pd.DataFrame(data_v3)

# Назви ознак
features = ["Area", "Perimeter", "Brightness"]

#Обчислення статистичних параметрів (min, max, mu, sigma)
stats = {}
for feat in features:
    values = df[feat].values
    val_min = np.min(values)
    val_max = np.max(values)
    val_mean = np.mean(values)
    # З методички: ddof=0, тобто ділення на n (популяційне ст. відхилення)
    val_std = np.std(values, ddof=0)
    stats[feat] = {"min": val_min, "max": val_max, "mu": val_mean, "sigma": val_std}

stats_df = pd.DataFrame(stats).T
print("Статистичні параметри (Параметри перетворення)")
print(stats_df.round(3))
print("\n" + "="*60 + "\n")

#Нормалізація Min-Max та Z-score
df_minmax = df.copy()
df_zscore = df.copy()

for feat in features:
    f_min = stats[feat]["min"]
    f_max = stats[feat]["max"]
    f_mu = stats[feat]["mu"]
    f_sigma = stats[feat]["sigma"]
    
    # Min-Max
    df_minmax[feat] = (df[feat] - f_min) / (f_max - f_min)
    
    # Z-score
    df_zscore[feat] = (df[feat] - f_mu) / f_sigma

print("Початкові дані")
print(df[["No", "Class", "Area", "Perimeter", "Brightness"]])

print("\n Min-Max Normalization (0 до 1) ")
print(df_minmax[["No", "Class", "Area", "Perimeter", "Brightness"]].round(3))

print("\n Z-score Standardization ")
print(df_zscore[["No", "Class", "Area", "Perimeter", "Brightness"]].round(3))
print("\n" + "="*60 + "\n")

#Побудова точкових графіків (Площа vs Периметр)
plt.figure(figsize=(12, 5))

# Словник кольорів для класів
colors = {'A': 'blue', 'B': 'red'}

# Графік 1: До масштабування
plt.subplot(1, 2, 1)
for cls in ['A', 'B']:
    sub = df[df['Class'] == cls]
    plt.scatter(sub['Area'], sub['Perimeter'], c=colors[cls], label=f'Клас {cls}', s=70, alpha=0.8)

plt.title('До масштабування (Початкові дані)')
plt.xlabel('Площа $X_1$ (пікселів²)')
plt.ylabel('Периметр $X_2$ (пікселів)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Графік 2: Після Min-Max нормалізації
plt.subplot(1, 2, 2)
for cls in ['A', 'B']:
    sub = df_minmax[df_minmax['Class'] == cls]
    plt.scatter(sub['Area'], sub['Perimeter'], c=colors[cls], label=f'Клас {cls}', s=70, alpha=0.8)

plt.title('Після Min-Max нормалізації')
plt.xlabel('Нормалізована Площа $X_1^\prime$ [0, 1]')
plt.ylabel('Нормалізований Периметр $X_2^\prime$ [0, 1]')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()