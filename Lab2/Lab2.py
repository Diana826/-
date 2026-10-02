# ==============================================================================
# ЛАБОРАТОРНА РОБОТА №2 | Дискретне перетворення Фур'є (ДПФ) та ОДПФ
# Студентка: Бойко О-Д.Т., група ПМ-43
# Варіант №3 (n = 3, непарний)
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# Фіксуємо зерно генератора випадкових чисел для гарантії відтворюваності результатів
np.random.seed(43)

# ------------------------------------------------------------------------------
# ФУНКЦІЇ ДПФ ТА ОДПФ
# ------------------------------------------------------------------------------

def dft_coefficient(signal, k):
    """
    Обчислення одного (k-го) комплексного коефіцієнта ДПФ з підрахунком операцій.
    """
    N = len(signal)
    sum_real = 0.0
    sum_imag = 0.0
    num_m = 0  # Лічильник дійсних множень
    num_a = 0  # Лічильник дійсних додавань
    
    for n in range(N):
        angle = -2 * np.pi * k * n / N
        cos_v = np.cos(angle)
        sin_v = np.sin(angle)
        
        prod_real = signal[n] * cos_v
        prod_imag = signal[n] * sin_v
        num_m += 2  # Дві операції множення
        
        sum_real += prod_real
        sum_imag += prod_imag
        if n > 0:
            num_a += 2  # Дві операції додавання
            
    A_k = sum_real / N  # Косинусний коефіцієнт A_k
    B_k = sum_imag / N  # Синусний коефіцієнт B_k
    return complex(A_k, B_k), num_m, num_a


def dft(signal):
    """
    Повне пряме ДПФ для всіх k = 0 ... N-1 з підрахунком складності.
    """
    N = len(signal)
    coefficients = np.zeros(N, dtype=complex)
    total_num_m = 0
    total_num_a = 0
    
    start_time = time.perf_counter()
    
    for k in range(N):
        c_k, num_m, num_a = dft_coefficient(signal, k)
        coefficients[k] = c_k
        total_num_m += num_m
        total_num_a += num_a
        
    elapsed_time = time.perf_counter() - start_time
    return coefficients, elapsed_time, total_num_m, total_num_a


def idft(coefficients):
    """
    Обернене дискретне перетворення Фур'є (ОДПФ) з лічильниками num_m та num_a.
    """
    N = len(coefficients)
    recovered = np.zeros(N, dtype=complex)
    num_m = 0  # Лічильник дійсних операцій множення
    num_a = 0  # Лічильник дійсних операцій додавання
    
    for n in range(N):
        value = 0.0 + 0.0j
        for k in range(N):
            angle = 2 * np.pi * k * n / N
            value += coefficients[k] * (np.cos(angle) + 1j * np.sin(angle))
            num_m += 2
            if n > 0:
                num_a += 2
                
        recovered[n] = value
        
    return recovered, num_m, num_a


def print_coefficients(coefficients):
    """
    Форматоване виведення спектральних коефіцієнтів (включаючи A_k та B_k) у консоль.
    """
    print(f"{'k':>3} {'A_k (Re)':>12} {'B_k (Im)':>12} {'|Ck|':>12} {'arg(Ck)':>12}")
    print("-" * 57)
    for k, c in enumerate(coefficients):
        print(f"{k:>3} {c.real:>12.4f} {c.imag:>12.4f} {np.abs(c):>12.4f} {np.angle(c):>12.4f}")


# ------------------------------------------------------------------------------
# РОЗРАХУНОК ПАРАМЕТРІВ ДЛЯ ВАРІАНТУ №3 (n = 3)
# ------------------------------------------------------------------------------
n_variant = 3
N1 = 10 + n_variant  # N1 = 13

print(f"✅ ВАРІАНТ №{n_variant} УСПІШНО ІНІЦІАЛІЗОВАНО:")
print(f"   • Розмірність сигналу для Частини I (N1) = {N1}\n")

# ==============================================================================
# ЧАСТИНА І: ПРЯМЕ ДИСКРЕТНЕ ПЕРЕТВОРЕННЯ ФУР'Є (ДПФ)
# ==============================================================================
print("="*65)
print("ЧАСТИНА І: Пряме дискретне перетворення Фур'є (ДПФ)")
print("="*65)

f_signal = np.round(np.random.uniform(-10, 10, N1), 3)

# Обчислення коефіцієнтів ДПФ
C_k, elapsed_time, num_m1, num_a1 = dft(f_signal)
elapsed_time_ms = elapsed_time * 1000

# Виведення коефіцієнтів A_k, B_k, модуля та фази
print_coefficients(C_k)

# Додаткова наочна таблиця через pandas для Частини I
df_part1 = pd.DataFrame({
    'k': range(N1),
    'f[k] (вхідний)': f_signal,
    'A_k (Re)': np.round(C_k.real, 4),
    'B_k (Im)': np.round(C_k.imag, 4),
    '|C_k| (Амплітуда)': np.round(np.abs(C_k), 4),
    'arg(C_k) (Фаза, рад)': np.round(np.angle(C_k), 4)
})

print("\nДетальна таблиця коефіцієнтів A_k та B_k:")
display(df_part1)

print(f"\n📊 ОЦІНКА ЕФЕКТИВНОСТІ АЛГОРИТМУ (ДПФ N={N1}):")
print(f"   • Час обчислення: {elapsed_time_ms:.4f} мс")
print(f"   • Кількість дійсних операцій множення (num_m): {num_m1} (формула: 2 * N² = 2 * {N1}² = {2 * N1**2})")
print(f"   • Кількість дійсних операцій додавання (num_a): {num_a1} (формула: 2 * N * (N - 1) = 2 * {N1} * {N1-1} = {2 * N1 * (N1 - 1)})")

# Графіки Частини І
amplitudes1 = np.abs(C_k)
phases1 = np.angle(C_k)

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
# ЧАСТИНА ІІ: ВІДТВОРЕННЯ АНАЛОГОВОГО СИГНАЛУ s(t)
# ==============================================================================
print("\n" + "="*65)
print("ЧАСТИНА ІІ: Двійкові відліки та аналоговий сигнал s(t)")
print("="*65)

N2_val = 96 + n_variant
binary_str = format(N2_val, '08b')
s_binary_list = [int(bit) for bit in binary_str]
s_binary_list[0] = 1
signal2 = np.array(s_binary_list, dtype=float)

print("8 відліків сигналу:")
print(signal2.astype(int))

C2, elapsed2, num_m2, num_a2 = dft(signal2)

print("\nКоефіцієнти ДПФ для 8 відліків:")
print_coefficients(C2)

Tc = 1.0
N2 = len(signal2)
t = np.linspace(0, Tc, 2000, endpoint=False)

s_t_complex = np.zeros_like(t, dtype=complex)
for k in range(N2):
    s_t_complex += C2[k] * np.exp(1j * 2 * np.pi * k * t / Tc)

s_t = s_t_complex.real

print(f"\nC0 (постійна складова) = {C2[0].real:.6f}")
for k in range(1, N2 // 2):
    amplitude = 2 * abs(C2[k])
    phase = np.angle(C2[k])
    print(f"k={k}: A={amplitude:.6f}, phi={phase:.6f}")

k_mid = N2 // 2
print(f"k={k_mid} (Найвища гармоніка): A={abs(C2[k_mid]):.6f}, phi={np.angle(C2[k_mid]):.6f}")

plt.figure(figsize=(12, 4))
plt.plot(t, s_t, color='#1f77b4', linewidth=1.5, label="Відтворений аналоговий сигнал s(t)")
plt.stem(np.linspace(0, 1, N2, endpoint=False), signal2, linefmt='r-', markerfmt='ro', basefmt=" ", label="Дискретні відліки s(nT_δ)")
plt.title("Відтворений аналоговий сигнал s(t) (N=8 відліків)", fontsize=12)
plt.xlabel("Час t (відносний період)")
plt.ylabel("s(t)")
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ==============================================================================
# ЧАСТИНА ІІІ: ОБЕРНЕНЕ ДИСКРЕТНЕ ПЕРЕТВОРЕННЯ ФУР'Є (ОДПФ)
# ==============================================================================
print("\n" + "="*70)
print("ЧАСТИНА ІІІ: Обернене дискретне перетворення Фур'є (ОДПФ)")
print("="*70)

# Відновлення відліків та отримання значень num_m і num_a
recovered, num_m_idft, num_a_idft = idft(C2)

# Виведення порівняльної таблиці
print(f"{'n':>3} {'початкове':>12} {'відновлене':>15} {'похибка':>12}")
print("-" * 47)

for n in range(N2):
    rec_val = recovered[n].real
    orig_val = signal2[n]
    err = abs(rec_val - orig_val)
    print(f"{n:>3} {orig_val:>12.6f} {rec_val:>15.6f} {err:>12.2e}")

# Виведення порахованих значення num_m та num_a для ОДПФ
print(f"\n📊 ОЦІНКА СКЛАДНОСТІ ОДПФ (N={N2}):")
print(f"   • Кількість дійсних операцій множення (num_m): {num_m_idft} (формула: 2 * N² = 2 * {N2}² = {2 * N2**2})")
print(f"   • Кількість дійсних операцій додавання (num_a): {num_a_idft} (формула: 2 * N * (N - 1) = 2 * {N2} * {N2-1} = {2 * N2 * (N2 - 1)})")

print("\n📝 АНАЛІТИЧНІ ВИРАЗИ ДЛЯ n = 0 та n = 1:")
print(f"   • n = 0:  s(0)     = Sum_(k=0)^{{{N2-1}}} [ C_k ]")
print(f"   • n = 1:  s(1 T_δ) = Sum_(k=0)^{{{N2-1}}} [ C_k * exp( j * 2π * k / {N2} ) ]")
