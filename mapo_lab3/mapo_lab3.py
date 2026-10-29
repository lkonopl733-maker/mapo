import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import Counter

# варіант 3
samples = [
    # №, Клас, Площа, Периметр, Яскравість
    (3,  'A', 1100, 128, 90),
    (4,  'B', 1640, 174, 188),
    (5,  'A', 1210, 136, 82),
    (6,  'B', 1730, 181, 169),
    (7,  'A', 1030, 125, 96),
    (8,  'B', 1580, 170, 194),
    (9,  'A', 1150, 133, 77),
    (10, 'B', 1820, 190, 181),
]

q = np.array([1500, 161, 145])   # новий об'єкт

print("=" * 70)
print("ВАРІАНТ 3")
print("=" * 70)

# вихідні дані
print("\n1. Навчальні зразки та новий об'єкт q:")
print(f"{'№':<4} {'Клас':<6} {'Площа':<8} {'Периметр':<10} {'Яскравість'}")
print("-" * 45)
for s in samples:
    print(f"{s[0]:<4} {s[1]:<6} {s[2]:<8} {s[3]:<10} {s[4]}")
print(f"\nq = ({q[0]}; {q[1]}; {q[2]})")

# MIN-MAX маштабування
X = np.array([[s[2], s[3], s[4]] for s in samples])
labels = [s[1] for s in samples]
numbers = [s[0] for s in samples]

mins = X.min(axis=0)
maxs = X.max(axis=0)

print("\n2. Мінімуми та максимуми (лише серед навчальних зразків):")
print(f"  Площа:      min = {mins[0]}, max = {maxs[0]}")
print(f"  Периметр:   min = {mins[1]}, max = {maxs[1]}")
print(f"  Яскравість: min = {mins[2]}, max = {maxs[2]}")

def scale(x, mins, maxs):
    result = np.zeros_like(x, dtype=float)
    for j in range(3):
        if maxs[j] == mins[j]:
            result[j] = 0.0
        else:
            result[j] = (x[j] - mins[j]) / (maxs[j] - mins[j])
    return result

X_scaled = np.array([scale(x, mins, maxs) for x in X])
q_scaled = scale(q, mins, maxs)

print("\nМасштабований q:")
print(f"  q' = ({q_scaled[0]:.4f}; {q_scaled[1]:.4f}; {q_scaled[2]:.4f})")

#Приклад масштабування q з підстановкою 
print("\nПриклад масштабування q за формулою:")
print("x' = (x - min) / (max - min)")
for j, name in enumerate(["Площа", "Периметр", "Яскравість"]):
    print(
        f"  {name}: ({q[j]} - {mins[j]}) / "
        f"({maxs[j]} - {mins[j]}) = {q_scaled[j]:.4f}"
    )

# ручні обчисення до 1 зразка
print("\n3. Ручні обчислення відстаней до першого зразка (№3):")
first = X_scaled[0]
diff = np.abs(q_scaled - first)
print(f"  Різниці: |{q_scaled[0]:.4f}-{first[0]:.4f}| = {diff[0]:.4f}; "
      f"|{q_scaled[1]:.4f}-{first[1]:.4f}| = {diff[1]:.4f}; "
      f"|{q_scaled[2]:.4f}-{first[2]:.4f}| = {diff[2]:.4f}")

d2_manual = np.sqrt(np.sum(diff**2))
d1_manual = np.sum(diff)
dinf_manual = np.max(diff)

print(f"  Евклідова:   d₂ = √({diff[0]:.4f}² + {diff[1]:.4f}² + {diff[2]:.4f}²) = {d2_manual:.4f}")
print(f"  Мангеттенська: d₁ = {diff[0]:.4f} + {diff[1]:.4f} + {diff[2]:.4f} = {d1_manual:.4f}")
print(f"  Чебишова:    d∞ = max({diff[0]:.4f}, {diff[1]:.4f}, {diff[2]:.4f}) = {dinf_manual:.4f}")

# відстані до зразків
def euclidean(a, b):
    return np.sqrt(np.sum((a - b)**2))

def manhattan(a, b):
    return np.sum(np.abs(a - b))

def chebyshev(a, b):
    return np.max(np.abs(a - b))

distances = []
for i in range(len(samples)):
    d2 = euclidean(q_scaled, X_scaled[i])
    d1 = manhattan(q_scaled, X_scaled[i])
    dinf = chebyshev(q_scaled, X_scaled[i])
    distances.append((numbers[i], labels[i], d2, d1, dinf))

print("\n4. Таблиця відстаней (масштабовані ознаки):")
print(f"{'№':<4} {'Клас':<6} {'d₂':<10} {'d₁':<10} {'d∞':<10}")
print("-" * 45)
for num, cls, d2, d1, dinf in distances:
    print(f"{num:<4} {cls:<6} {d2:<10.3f} {d1:<10.3f} {dinf:<10.3f}")

# рейтенги ТА k-NN 
print("\n5. Рейтинги найближчих сусідів (за рівності — менший номер):")

def rank_by_metric(metric_idx, name):
    sorted_dist = sorted(distances, key=lambda x: (x[metric_idx], x[0]))
    print(f"\n  {name}:")
    for i, (num, cls, d2, d1, dinf) in enumerate(sorted_dist, 1):
        dist = [d2, d1, dinf][metric_idx-2]
        print(f"    {i}. №{num} (клас {cls}) — {dist:.4f}")
    return sorted_dist

rank_d2 = rank_by_metric(2, "Евклідова відстань")
rank_d1 = rank_by_metric(3, "Мангеттенська відстань")
rank_dinf = rank_by_metric(4, "Відстань Чебишова")

# k-NN тільки для евклідової
print("\n  Прогноз за евклідовою відстанню:")
nearest3 = rank_d2[:3]
print(f"    Три найближчі: {[f'№{n}({c})' for n,c,_,_,_ in nearest3]}")

# k=1
pred_k1 = nearest3[0][1]
print(f"    k = 1 → клас {pred_k1} (найближчий сусід №{nearest3[0][0]})")

# k=3
votes = [c for _,c,_,_,_ in nearest3]
pred_k3 = Counter(votes).most_common(1)[0][0]
print(f"    k = 3 → класи {votes} → більшість: {pred_k3}")

# порівняння без маштабування
print("\n6. Найближчий сусід за евклідовою відстанню БЕЗ масштабування:")
dist_raw = []
for i in range(len(samples)):
    d = euclidean(q, X[i])
    dist_raw.append((numbers[i], labels[i], d))

sorted_raw = sorted(dist_raw, key=lambda x: (x[2], x[0]))
print(f"  Найближчий: №{sorted_raw[0][0]} (клас {sorted_raw[0][1]}), d₂ = {sorted_raw[0][2]:.2f}")

print("\n  Порівняння:")
print(f"    Після масштабування: №{rank_d2[0][0]} (клас {rank_d2[0][1]})")
print(f"    Без масштабування:   №{sorted_raw[0][0]} (клас {sorted_raw[0][1]})")

if rank_d2[0][0] != sorted_raw[0][0]:
    print("    → Результат ЗМІНИВСЯ! Причина — різні діапазони ознак "
          "(площа ~1000, яскравість ~100). Без масштабування площа домінує.")
else:
    print("    → Результат не змінився.")

# графік
print("\n7. Графік за двома масштабованими ознаками (площа, периметр)...")

plt.figure(figsize=(9, 7))
for i in range(len(samples)):
    color = 'blue' if labels[i] == 'A' else 'red'
    marker = 'o' if labels[i] == 'A' else 's'
    plt.scatter(X_scaled[i, 0], X_scaled[i, 1], c=color, marker=marker, s=100,
                edgecolors='black', zorder=3)
    plt.text(X_scaled[i, 0]+0.02, X_scaled[i, 1]+0.02, f"№{numbers[i]}", fontsize=9)

plt.scatter(q_scaled[0], q_scaled[1], c='green', marker='*', s=250,
            edgecolors='black', zorder=5, label='q')
plt.text(q_scaled[0]+0.02, q_scaled[1]+0.02, 'q', fontsize=11, fontweight='bold')

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
           markersize=10, label='Клас A'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
           markersize=10, label='Клас B'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='green',
           markersize=15, label='Новий об\'єкт q'),
]
plt.legend(handles=legend_elements, loc='best')

plt.xlabel('Площа (масштабована)', fontsize=12)
plt.ylabel('Периметр (масштабований)', fontsize=12)
plt.title('Варіант 3: розташування об\'єктів у просторі двох ознак', fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph_variant3.png', dpi=150)
plt.show()

print("\nГрафік збережено у файл graph_variant3.png")
print("\nПояснення: на графіку показано лише дві ознаки (площа і периметр).")
print("Третя ознака (яскравість) не відображена, тому порядок сусідів")
print("за двовимірною відстанню може відрізнятися від тривимірної.")

# висновок
print("\n" + "=" * 70)
print("ВИСНОВОК")
print("=" * 70)
print(f"За k = 1 об'єкт q належить до класу {pred_k1}.")
print(f"За k = 3 об'єкт q належить до класу {pred_k3}.")
print("Масштабування дає змогу порівнювати ознаки з різними діапазонами значень.")
print("Графік відображає лише дві ознаки, тоді як класифікація враховує три.")
print("=" * 70)