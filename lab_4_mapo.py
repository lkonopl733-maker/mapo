import numpy as np
import matplotlib.pyplot as plt

# 1. вихідні дані (Варіант 3: зразки 3-10, q=(1500; 145))
data = [
    [3, 'A', 1100, 90],
    [4, 'B', 1640, 188],
    [5, 'A', 1210, 82],
    [6, 'B', 1730, 169],
    [7, 'A', 1030, 96],
    [8, 'B', 1580, 194],
    [9, 'A', 1150, 77],
    [10, 'B', 1820, 181]
]

q_raw = np.array([1500.0, 145.0])
sample_ids = [row[0] for row in data]
y = np.array([row[1] for row in data])
X_raw = np.array([row[2:4] for row in data], dtype=float)

# 2. масштабування ознак (Min-Max)
min_val = X_raw.min(axis=0)
max_val = X_raw.max(axis=0)

X_scaled = (X_raw - min_val) / (max_val - min_val)
q_scaled = (q_raw - min_val) / (max_val - min_val)

print("--- 2. МАСШТАБУВАННЯ ОЗНАК ---")
print(f"Min (площа x1, яскравість x2): {min_val}")
print(f"Max (площа x1, яскравість x2): {max_val}")
print(f"Масштабований вектор q': [{q_scaled[0]:.4f}, {q_scaled[1]:.4f}]\n")

# 3 знаходження центрів класів
mask_A = (y == 'A')
mask_B = (y == 'B')

X_A = X_scaled[mask_A]
X_B = X_scaled[mask_B]
ids_A = [sample_ids[i] for i in range(len(data)) if y[i] == 'A']
ids_B = [sample_ids[i] for i in range(len(data)) if y[i] == 'B']

c_A = X_A.mean(axis=0)
c_B = X_B.mean(axis=0)

print("--- 3. ЦЕНТРИ КЛАСІВ ---")
print(f"Центр класу A (cA): [{c_A[0]:.4f}, {c_A[1]:.4f}]")
print(f"Центр класу B (cB): [{c_B[0]:.4f}, {c_B[1]:.4f}]\n")

# 4 відстані до центру і покахники класів
dist_A = np.sqrt(np.sum((X_A - c_A)**2, axis=1))
dist_B = np.sqrt(np.sum((X_B - c_B)**2, axis=1))

print("--- 4.1 ТАБЛИЦЯ ВІДСТАНЕЙ НАВЧАЛЬНИХ ТОЧОК ДО СВОГО ЦЕНТРУ ---")
print("Клас A:")
for s_id, d in zip(ids_A, dist_A):
    print(f"  Зразок №{s_id}: d2(x, cA) = {d:.4f}")

print("Клас B:")
for s_id, d in zip(ids_B, dist_B):
    print(f"  Зразок №{s_id}: d2(x, cB) = {d:.4f}")

S_A = dist_A.mean()
S_B = dist_B.mean()
R_A = dist_A.max()
R_B = dist_B.max()
D = np.sqrt(np.sum((c_A - c_B)**2))

print("\n--- 4.2 ПОКАЗНИКИ РОЗКИДУ ТА ВІДСТАНІ МІЖ ЦЕНТРАМИ ---")
print(f"Середній розкид SA = {S_A:.4f}, Радіус RA = {R_A:.4f}")
print(f"Середній розкид SB = {S_B:.4f}, Радіус RB = {R_B:.4f}")
print(f"Відстань між центрами D = {D:.4f}")
print(f"RA + RB = {R_A + R_B:.4f}")

if R_A + R_B < D:
    print("Інтерпретація: RA + RB < D — гіперкола класів НЕ перетинаються (ідеальне геометричне розділення).")
else:
    print("Інтерпретація: RA + RB >= D — гіперкола перетинаються.")


# 5 вирішувальні функції і класифіфкації q'
g_A_q = -np.sum((q_scaled - c_A)**2)
g_B_q = -np.sum((q_scaled - c_B)**2)

print("\n--- 5. ОЦІНКИ ТА РІШЕННЯ НАЙБЛИЖЧОГО ЦЕНТРА ---")
print(f"gA(q') = {g_A_q:.4f}")
print(f"gB(q') = {g_B_q:.4f}")

class_nc = 'A' if g_A_q > g_B_q else 'B'
print(f"Рішення методом найближчого центра: об'єкт q належить до класу **{class_nc}**\n")

# 6 рівняння межі рішення
w = 2 * (c_A - c_B)
b = np.sum(c_B**2) - np.sum(c_A**2)

print("--- 6. РІВНЯННЯ МЕЖІ РІШЕННЯ ---")
print(f"Вектор ваг w: [{w[0]:.4f}, {w[1]:.4f}]")
print(f"Вільний член b: {b:.4f}")
print(f"Рівняння межі: {w[0]:.4f} * x1' + ({w[1]:.4f}) * x2' + ({b:.4f}) = 0\n")

# 7 порівняння з методом  1-NN (k=1)
dists_to_q = np.sqrt(np.sum((X_scaled - q_scaled)**2, axis=1))
min_idx = np.argmin(dists_to_q)
nearest_sample_id = sample_ids[min_idx]
class_1nn = y[min_idx]
min_dist = dists_to_q[min_idx]

print("--- 7. ПОРІВНЯННЯ З МЕТОДОМ 1-NN (k=1) ---")
print(f"Найближчий навчальний зразок до q': №{nearest_sample_id} (Клас {class_1nn}) з відстанню {min_dist:.4f}")
print(f"Рішення методом 1-NN: **Клас {class_1nn}**")
if class_nc == class_1nn:
    print("Висновок порівняння: Обидва методи дають ОДНАКОВИЙ результат класифікації.")
else:
    print("Висновок порівняння: Методи дають РІЗНІ результати.")

# 8 пояснення компактоності і висновок
print("\n--- 8. ПОЯСНЕННЯ КОМПАКТНОСТІ ТА ВИСНОВОК ---")
print("1. Оцінка компактності: Внутрішньокласові розкиди SA та SB є малими порівняно з відстанню між центрами D (D >> SA, SB).")
print("   Умова RA + RB < D виконується, що підтверджує гіпотезу компактності: класи утворюють щільні відокремлені групи.")
print(f"2. Підсумок: За вирішувальними функціями gA(q') і gB(q') новий об'єкт q віднесено до **класу {class_nc}**, що повністю узгоджується з результатами методу 1-NN та взаємним розташуванням точок на графіку.")

# 9 графік
plt.figure(figsize=(8, 6))
plt.scatter(X_A[:, 0], X_A[:, 1], color='blue', label='Клас A', s=80, marker='o')
plt.scatter(X_B[:, 0], X_B[:, 1], color='red', label='Клас B', s=80, marker='s')
plt.scatter(c_A[0], c_A[1], color='darkblue', marker='x', s=150, linewidth=3, label='Центр cA')
plt.scatter(c_B[0], c_B[1], color='darkred', marker='x', s=150, linewidth=3, label='Центр cB')
plt.scatter(q_scaled[0], q_scaled[1], color='green', marker='*', s=200, label="Новий об'єкт q'")

x1_vals = np.linspace(-0.2, 1.2, 100)
if abs(w[1]) > 1e-6:
    x2_vals = (-w[0] * x1_vals - b) / w[1]
    plt.plot(x1_vals, x2_vals, 'k--', label='Межа gA(x) - gB(x) = 0')

plt.title('Класифікація методом найближчого центра (Варіант 3)')
plt.xlabel('Масштабована площа (x1\')')
plt.ylabel('Масштабована яскравість (x2\')')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()