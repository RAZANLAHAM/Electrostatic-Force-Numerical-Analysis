# Electrostatic-Force-Numerical-Analysis
import math
import numpy as np
import matplotlib.pyplot as plt

# --- القيم الفيزيائية (مثال على مجموعة 1) ---
q = 2e-5
Q = 2e-5
a = 0.9
F_target = 1.0
eps0 = 8.85e-12

# --- دالة القوة الكهروستاتيكية ---
def F(x):
    return (1 / (4 * math.pi * eps0)) * (q * Q * x) / ((a**2 + x**2)**(3/2))

# --- f(x) = الفرق بين القوة المطلوبة والمحسوبة ---
def f(x):
    return F(x) - F_target

# --- مشتقة f(x) لطريقة نيوتن رافسون ---
def df(x):
    numerator = q * Q * ((a**2 + x**2)**(3/2)) - q * Q * x * 3 * x * (a**2 + x**2)**(1/2)
    denominator = (a**2 + x**2)**3
    return (1 / (4 * math.pi * eps0)) * (numerator / denominator)

# --- Bisection Method ---
def bisection_method(x_low, x_high, tol=1e-6, max_iter=100):
    for i in range(max_iter):
        x_mid = (x_low + x_high) / 2
        if abs(f(x_mid)) < tol:
            return x_mid, i + 1
        if f(x_low) * f(x_mid) < 0:
            x_high = x_mid
        else:
            x_low = x_mid
    return x_mid, max_iter

# --- Newton-Raphson Method ---
def newton_raphson(x0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ZeroDivisionError("Zero derivative encountered.")
        x_new = x - fx / dfx
        if abs(f(x_new)) < tol:
            return x_new, i + 1
        x = x_new
    return x, max_iter

# --- تشغيل الطريقتين ---
x_bisect, it_bisect = bisection_method(0.01, 2)
x_newton, it_newton = newton_raphson(1.0)

# --- طباعة النتائج ---
print(f"[Bisection] x = {x_bisect:.6f} m, iterations = {it_bisect}")
print(f"[Newton-Raphson] x = {x_newton:.6f} m, iterations = {it_newton}")

# --- رسم القوة F(x) ---
x_vals = np.linspace(0.01, 2, 200)
f_vals = [F(x) for x in x_vals]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, f_vals, label="F(x)", linewidth=2)
plt.axhline(y=F_target, color='r', linestyle='--', label="Target Force = 1 N")
plt.axvline(x=x_bisect, color='green', linestyle='--', label="Bisection Root")
plt.axvline(x=x_newton, color='orange', linestyle='--', label="Newton Root")
plt.xlabel("x (m)")
plt.ylabel("F(x) (N)")
plt.title("Electrostatic Force vs Distance")
plt.grid(True)
plt.legend()
plt.show()
