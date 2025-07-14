import json
import re
import numpy as np
import matplotlib.pyplot as plt

RESULT_JSON = 'results/curvature.json'

def analyze_and_save_results(json_path=RESULT_JSON):
    with open(json_path, 'r') as f:
        data = json.load(f)

    pattern = re.compile(r'\((\d+)\)')
    slices, radius_vals, curvature_vals = [], [], []

    for filename, vals in data.items():
        match = pattern.search(filename)
        if match:
            slices.append(int(match.group(1)))
            radius_vals.append(vals['radius'])
            curvature_vals.append(vals['curvature'])

    slices, radius_vals, curvature_vals = zip(*sorted(zip(slices, radius_vals, curvature_vals)))

    # 통계 계산
    stats = {
        'radius_mean': float(np.mean(radius_vals)),
        'radius_median': float(np.median(radius_vals)),
        'radius_std': float(np.std(radius_vals)),
        'curvature_mean': float(np.mean(curvature_vals)),
        'curvature_median': float(np.median(curvature_vals)),
        'curvature_std': float(np.std(curvature_vals)),
        'num_samples': len(slices)
    }

    # 시각화 - 선 그래프
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Slice Number')
    ax1.set_ylabel('Radius (mm)', color='tab:blue')
    ax1.plot(slices, radius_vals, 'o-', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Curvature (1/mm)', color='tab:red')
    ax2.plot(slices, curvature_vals, 'x--', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Curvature and Radius vs Slice Number')
    fig.tight_layout()
    plt.savefig('results/curvature_vs_slice.png')
    plt.close()

    # 히스토그램
    plt.hist(radius_vals, bins=15, color='blue', alpha=0.7)
    plt.title('Radius Distribution')
    plt.xlabel('Radius (mm)')
    plt.ylabel('Frequency')
    plt.savefig('results/radius_histogram.png')
    plt.close()

    plt.hist(curvature_vals, bins=15, color='red', alpha=0.7)
    plt.title('Curvature Distribution')
    plt.xlabel('Curvature (1/mm)')
    plt.ylabel('Frequency')
    plt.savefig('results/curvature_histogram.png')
    plt.close()

    # 통계 json 저장
    with open('results/statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    analyze_and_save_results()
