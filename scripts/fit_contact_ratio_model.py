import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. 샘플 곡률 값 (정렬된 상태로 예시)
curvatures = np.array([...])
# 정규화된 가상 밀착률
contact_ratio = (curvatures - min(curvatures)) / (max(curvatures) - min(curvatures))

# 2. 피팅 함수 정의
def exp_model(k, a):
    return 1 - np.exp(-a * k)

def logistic_model(k, a, b):
    return 1 / (1 + np.exp(-a * (k - b)))

def poly_model(k, a, b, c):
    return a * k**2 + b * k + c

# 3. 모델 피팅
popt_exp, _ = curve_fit(exp_model, curvatures, contact_ratio, bounds=(0, [1000]))
popt_logi, _ = curve_fit(logistic_model, curvatures, contact_ratio, bounds=([0,0], [1000, 1]))
popt_poly, _ = curve_fit(poly_model, curvatures, contact_ratio)

# 4. 시각화
k_test = np.linspace(min(curvatures), max(curvatures), 100)
plt.scatter(curvatures, contact_ratio, label='Data', color='black')
plt.plot(k_test, exp_model(k_test, *popt_exp), label='Exp Fit', color='red')
plt.plot(k_test, logistic_model(k_test, *popt_logi), label='Logistic Fit', color='green')
plt.plot(k_test, poly_model(k_test, *popt_poly), label='Poly Fit', color='blue')
plt.legend()
plt.xlabel('Curvature (1/mm)')
plt.ylabel('Contact Ratio (Normalized)')
plt.title('Contact Ratio Model Fitting')
plt.grid()
plt.savefig('results/contact_ratio_model_fit.png')
