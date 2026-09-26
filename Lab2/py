# ==============================================================================
# ЛАБОРАТОРНА РОБОТА №2 | Дискретне перетворення Фур'є (ДПФ) та ОДПФ
# Студентка: Бойко О-Д.Т., група ПМ-43
# Варіант №3 (n = 3, непарний)
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# Встановлюємо фіксоване зерно для відтворюваності випадкових даних
np.random.seed(43)

# ------------------------------------------------------------------------------
# РОЗРАХУНОК ПАРАМЕТРІВ ДЛЯ ВАРІАНТУ №3 (n = 3)
# ------------------------------------------------------------------------------
n_variant = 3
N1 = 10 + n_variant  # Частина I: N = 10 + 3 = 13 відліків

print(f"✅ ВАРІАНТ №{n_variant} УСПІШНО ІНІЦІАЛІЗОВАНО:")
print(f"   • Розмірність сигналу для Частини I (N1) = {N1}\n")

# ==============================================================================
# ЧАСТИНА І: ПРЯМЕ ДИСКРЕТНЕ ПЕРЕТВОРЕННЯ ФУР'Є (ДПФ) ТА ОЦІНКА СКЛАДНОСТІ
# ==============================================================================
print("="*65)
print("ЧАСТИНА І: Пряме дискретне перетворення Фур'є (ДПФ)")
print("="*65)

# 1. Генерація випадкового вхідного сигналу f із N1=13 елементів
f_signal = np.round(np.random.uniform(-10, 10, N1), 3)

# 2. Алгоритм обчислення коефіцієнтів ДПФ C_k
start_time = time.perf_counter()

num_mult = 0  # Лічильник дійсних операцій множення
num_add = 0   # Лічильник дійсних операцій додавання

C_k = np.zeros(N1, dtype=complex)

for k in range(N1):
    sum_real = 0.0
    sum_imag = 0.0
    for n in range(N1):
        angle = -2 * np.pi * k * n / N1
        cos_v = np.cos(angle)
        sin_v = np.sin(angle)
        
        # Дійсні множення: f[n] * cos_v та f[n] * sin_v
        prod_real = f_signal[n] * cos_v
        prod_imag = f_signal[n] * sin_v
        num_mult += 2

        # Накопичення суми (дійсні додавання)
        sum_real += prod_real
        sum_imag += prod_imag
        if n > 0:
            num_add += 2
            
    C_k[k] = complex(sum_real, sum_imag)

elapsed_time = (time.perf_counter() - start_time) * 1000  # у мс

amplitudes1 = np.abs(C_k)
phases1 = np.angle(C_k)

# Формуємо таблицю результатів Частини I
df_part1 = pd.DataFrame({
    'k': range(N1),
    'f[k] (вхідний)': f_signal,
    'C_k (Комплексне)': [f"{c.real:.3f} + {c.imag:.3f}j" for c in C_k],
    '|C_k| (Амплітуда)': np.round(amplitudes1, 3),
    'arg(C_k) (Фаза, рад)': np.round(phases1, 3)
})

display(df_part1)

print(f"\n📊 ОЦІНКА ЕФЕКТИВНОСТІ АЛГОРИТМУ (ДПФ N={N1}):")
print(f"   • Час обчислення: {elapsed_time:.4f} мс")
print(f"   • Кількість дійсних операцій множення: {num_mult} (формула: 2 * N² = 2 * {N1}² = {2 * N1**2})")
print(f"   • Кількість дійсних операцій додавання: {num_add} (формула: 2 * N * (N - 1) = 2 * {N1} * {N1-1} = {2 * N1 * (N1 - 1)})")

# Графіки Частини І
fig, axs = plt.subplots(1, 2, figsize=(14, 4))

axs[0].stem(range(N1), amplitudes1, linefmt='b-', markerfmt='bo', basefmt=" ")
axs[0].set_title("Спектр амплітуд |C_k| (N=13)", fontsize=12)
axs[0].set_xlabel("Індекс k")
axs[0].set_ylabel("Амплітуда")
axs[0].grid(True, linestyle='--', alpha=0.6)

axs[1].stem(range(N1), phases1, linefmt='g-', markerfmt='go', basefmt=" ")
axs[1].set_title("Спектр фаз arg(C_k) (N=13)", fontsize=12)
axs[1].set_xlabel("Індекс k")
axs[1].set_ylabel("Фаза (радіани)")
axs[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# ==============================================================================
# ЧАСТИНА ІІ: ВІДТВОРЕННЯ АНАЛОГОВОГО СИГНАЛУ НА ОСНОВІ ЧИСЛА N
# ==============================================================================
print("\n" + "="*65)
print("ЧАСТИНА ІІ: Двійкові відліки та аналоговий сигнал s(t)")
print("="*65)

# 1. Формування 8-розрядного масиву відліків на основі числа N = 96 + n (для n=3)
N2_val = 96 + n_variant  # N2 = 99

# Переводимо 99 у 8-бітовий двійковий рядок: '01100011'
binary_str = format(N2_val, '08b')
s_binary_list = [int(bit) for bit in binary_str]

# Оскільки n=3 (непарний варіант), у 8-й розряд (індекс 0 у старшому біті) вставляємо 1
s_binary_list[0] = 1
s_binary = np.array(s_binary_list)
N8 = len(s_binary)  # N = 8 відліків

print(f"Число N = 96 + {n_variant} = {N2_val}")
print(f"Отриманий 8-розрядний вектор відліків s(nT_δ): {s_binary}\n")

# 2. Обчислення комплексних коефіцієнтів C_n для 8 відліків
C_n2 = np.fft.fft(s_binary)
mod_C_n2 = np.abs(C_n2)
arg_C_n2 = np.angle(C_n2)

df_part2 = pd.DataFrame({
    'n': range(N8),
    's(nT_δ)': s_binary,
    'C_n (Комплексне)': [f"{c.real:.3f} + {c.imag:.3f}j" for c in C_n2],
    '|C_n| (Модуль)': np.round(mod_C_n2, 3),
    'arg(C_n) (Аргумент, рад)': np.round(arg_C_n2, 3)
})
print("Комплексні коефіцієнти C_n (Частина ІІ):")
display(df_part2)

# 3. Відтворення аналогового сигналу s(t)
t_cont = np.linspace(0, 1, 1000)
s_reconstructed = np.zeros_like(t_cont)

for k in range(N8):
    s_reconstructed += (1 / N8) * (C_n2[k].real * np.cos(2 * np.pi * k * t_cont) - 
                                   C_n2[k].imag * np.sin(2 * np.pi * k * t_cont))

# Графік Частини ІІ
plt.figure(figsize=(12, 4))
plt.plot(t_cont, s_reconstructed, color='#1f77b4', linewidth=1.5, label="Відтворений аналоговий сигнал s(t)")
plt.stem(np.linspace(0, 1, N8, endpoint=False), s_binary, linefmt='r-', markerfmt='ro', basefmt=" ", label="Дискретні відліки s(nT_δ)")
plt.title("Відтворений аналоговий сигнал s(t) (N=8 відліків)", fontsize=12)
plt.xlabel("Час t (відносний період)")
plt.ylabel("s(t)")
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ==============================================================================
# ЧАСТИНА ІІІ: ОБЕРНЕНЕ ДИСКРЕТНЕ ПЕРЕТВОРЕННЯ ФУР'Є (ОДПФ)
# ==============================================================================
print("\n" + "="*65)
print("ЧАСТИНА ІІІ: Обернене дискретне перетворення Фур'є (ОДПФ)")
print("="*65)

# 1. Обчислення значення відліків s(nT_δ) при n = 0..7
s_recovered = np.zeros(N8)

for n in range(N8):
    val = 0 + 0j
    for k in range(N8):
        angle = 2 * np.pi * k * n / N8
        val += C_n2[k] * complex(np.cos(angle), np.sin(angle))
    s_recovered[n] = (val / N8).real

df_part3 = pd.DataFrame({
    'n': range(N8),
    'Відновлене s(n T_δ)': np.round(s_recovered, 4),
    'Початкове s(nT_δ)': s_binary,
    'Абсолютна похибка': np.abs(np.round(s_recovered, 4) - s_binary)
})

display(df_part3)

# 2. Вивід аналітичних виразів відліків s(nT_δ) для n = 0 та n = 1
print("\n📝 АНАЛІТИЧНІ ВИРАЗИ ВІДЛІКІВ s(n T_δ) ПРИ n = 0, 1:")
print(f"   • n = 0:  s(0)     = (1 / {N8}) * Sum_(k=0)^{{{N8-1}}} [ C_k ]")
print(f"   • n = 1:  s(1 T_δ) = (1 / {N8}) * Sum_(k=0)^{{{N8-1}}} [ C_k * exp( j * 2π * k / {N8} ) ]")
