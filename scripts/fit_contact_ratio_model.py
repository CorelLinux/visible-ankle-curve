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

# === 2. Define contact ratio using freq-based normalization ===
scale_min = 0.005  # most frequent curvature range
scale_max = 0.010
MAX_CONTACT_RATIO = 0.9 # 현실적인 최대 밀착률 (논문 기반)
contact_ratio = (curvatures - scale_min) / (scale_max - scale_min)
contact_ratio = np.clip(contact_ratio, 0, 1) * MAX_CONTACT_RATIO # 0~0.9 사이로 스케일링

# === 3. Fit candidate models ===
def exp_model(k, a, c): return c * (1 - np.exp(-a * k))
def logistic_model(k, a, b, c): return c / (1 + np.exp(-a * (k - b)))
def poly_model(k, a, b, c): return a * k**2 + b * k + c

popt_exp, _ = curve_fit(exp_model, curvatures, contact_ratio, bounds=([0, 0], [1000, 1]))
popt_logi, _ = curve_fit(logistic_model, curvatures, contact_ratio, bounds=([0, 0, 0], [1000, 1, 1]))
popt_poly, _ = curve_fit(poly_model, curvatures, contact_ratio)

# === 4. Plot and save ===
k_test = np.linspace(curvatures.min(), curvatures.max(), 200)
plt.figure(figsize=(10, 5))
plt.scatter(curvatures, contact_ratio, label="Data", color="black", alpha=0.5)
plt.plot(k_test, exp_model(k_test, *popt_exp), label=f"Exp Fit (a={popt_exp[0]:.1f}, c={popt_exp[1]:.2f})", color="red")
plt.plot(k_test, logistic_model(k_test, *popt_logi), label=f"Logistic Fit (a={popt_logi[0]:.1f}, b={popt_logi[1]:.4f}, c={popt_logi[2]:.2f})", color="green")
plt.plot(k_test, poly_model(k_test, *popt_poly), label="Poly Fit", color="blue")
plt.xlabel("Curvature (1/mm)")
plt.ylabel("Contact Ratio (0.9 = 100% Adhesion)")
plt.title("Contact Ratio Model Fitting (Max Adhesion = 0.9)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("results/contact_ratio_model_fit_realistic.png")

# === 5. Save model parameters ===
model_result = {
    "scaling": {
        "type": "freq-based-realistic",
        "range": [scale_min, scale_max],
        "max_contact_ratio": MAX_CONTACT_RATIO
    },
    "exp": {"a": popt_exp[0], "c": popt_exp[1]},
    "logistic": {"a": popt_logi[0], "b": popt_logi[1], "c": popt_logi[2]},
    "poly": {"a": popt_poly[0], "b": popt_poly[1], "c": popt_poly[2]},
    "curvature_range": [float(curvatures.min()), float(curvatures.max())]
}

with open("results/contact_ratio_model_realistic.json", "w") as f:
    json.dump(model_result, f, indent=2)

print("[✓] Contact ratio model fitting 완료 (현실적 최대 밀착률 0.9 적용)")
