import numpy as np
import json
import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# === 1. Load curvature values ===
with open("results/curvature.json", "r") as f:
    data = json.load(f)

pattern = re.compile(r'\((\d+)\)')
curvatures = []

for fname, vals in data.items():
    if match := pattern.search(fname):
        curvatures.append(vals["curvature"])

curvatures = np.array(sorted(curvatures))

# === 2. Define pseudo contact ratio (normalized curvature) ===
min_k, max_k = curvatures.min(), curvatures.max()
contact_ratio = (curvatures - min_k) / (max_k - min_k)

# === 3. Fit candidate models ===
def exp_model(k, a): return 1 - np.exp(-a * k)
def logistic_model(k, a, b): return 1 / (1 + np.exp(-a * (k - b)))
def poly_model(k, a, b, c): return a * k**2 + b * k + c

popt_exp, _ = curve_fit(exp_model, curvatures, contact_ratio, bounds=(0, [1000]))
popt_logi, _ = curve_fit(logistic_model, curvatures, contact_ratio, bounds=([0,0], [1000,1]))
popt_poly, _ = curve_fit(poly_model, curvatures, contact_ratio)

# === 4. Plot and save ===
k_test = np.linspace(min_k, max_k, 200)
plt.figure(figsize=(10, 5))
plt.scatter(curvatures, contact_ratio, label="Data", color="black", alpha=0.5)
plt.plot(k_test, exp_model(k_test, *popt_exp), label=f"Exp Fit (a={popt_exp[0]:.1f})", color="red")
plt.plot(k_test, logistic_model(k_test, *popt_logi), label=f"Logistic Fit (a={popt_logi[0]:.1f}, b={popt_logi[1]:.4f})", color="green")
plt.plot(k_test, poly_model(k_test, *popt_poly), label="Poly Fit", color="blue")
plt.xlabel("Curvature (1/mm)")
plt.ylabel("Contact Ratio (Normalized)")
plt.title("Contact Ratio Model Fitting")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("results/contact_ratio_model_fit.png")

# === 5. Save model parameters ===
model_result = {
    "exp": {"a": popt_exp[0]},
    "logistic": {"a": popt_logi[0], "b": popt_logi[1]},
    "poly": {"a": popt_poly[0], "b": popt_poly[1], "c": popt_poly[2]},
    "curvature_range": [float(min_k), float(max_k)]
}

with open("results/contact_ratio_model.json", "w") as f:
    json.dump(model_result, f, indent=2)

print("[✓] Contact ratio model fitting 완료")
