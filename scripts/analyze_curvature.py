import json
import re
import numpy as np
import matplotlib.pyplot as plt
import os

# process_canny.py가 생성하는 JSON 파일 경로
INPUT_JSON_FROM_CANNY = 'results/curvature.json'
# 이 스크립트가 생성할 통계 및 그래프 파일 경로
OUTPUT_DIR = 'results'
STATISTICS_JSON = os.path.join(OUTPUT_DIR, 'statistics.json')
CURVATURE_VS_SLICE_GRAPH = os.path.join(OUTPUT_DIR, 'curvature_vs_slice.png')
RADIUS_HISTOGRAM_GRAPH = os.path.join(OUTPUT_DIR, 'radius_histogram.png')
CURVATURE_HISTOGRAM_GRAPH = os.path.join(OUTPUT_DIR, 'curvature_histogram.png')

os.makedirs(OUTPUT_DIR, exist_ok=True) # results 폴더가 없을 경우 생성

def analyze_and_save_results(json_path=INPUT_JSON_FROM_CANNY):
    print(f"[*] Loading data from '{json_path}'...") # 영어로 변경
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"[✓] Successfully loaded '{json_path}'.") # 영어로 변경
    except FileNotFoundError:
        print(f"Error: File '{json_path}' not found. Please run process_canny.py first.") # 영어로 변경
        return
    except json.JSONDecodeError:
        print(f"Error: '{json_path}' is not a valid JSON file.") # 영어로 변경
        return

    def extract_slice_number(filename):
        nums = re.findall(r'\d+', filename)
        if nums:
            return int(nums[-1])
        return 0

    processed_data = []
    for filename, vals in data.items():
        if vals['radius'] is not None and vals['curvature'] is not None:
            slice_num = extract_slice_number(filename)
            processed_data.append((slice_num, vals['radius'], vals['curvature'], filename))

    processed_data.sort(key=lambda x: x[0])
    
    slices, radius_vals, curvature_vals, fnames_in_order = zip(*processed_data)
    
    slices = list(slices)
    radius_vals = list(radius_vals)
    curvature_vals = list(curvature_vals)
    fnames_in_order = list(fnames_in_order)


    if not slices:
        print("Warning: No valid slice data for analysis. Skipping graph and statistics generation.") # 영어로 변경
        return

    print("[*] Calculating statistics and saving 'statistics.json'...") # 영어로 변경
    stats = {
        'slices': slices,
        'fnames': fnames_in_order,
        'radius_vals': radius_vals,
        'curvature_vals': curvature_vals,
        'radius_mean': float(np.mean(radius_vals)),
        'radius_median': float(np.median(radius_vals)),
        'radius_std': float(np.std(radius_vals)),
        'curvature_mean': float(np.mean(curvature_vals)),
        'curvature_median': float(np.median(curvature_vals)),
        'curvature_std': float(np.std(curvature_vals)),
        'num_samples': len(slices)
    }
    with open(STATISTICS_JSON, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[✓] 'statistics.json' saved successfully.") # 영어로 변경

    # --- 시각화 - 선 그래프 (기존 analyze_curvature.py의 역할) ---
    print("[*] Generating basic visualization graphs...") # 영어로 변경
    fig, ax1 = plt.subplots(figsize=(10, 6))
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
    plt.savefig(CURVATURE_VS_SLICE_GRAPH)
    plt.close()
    print(f"[✓] '{os.path.basename(CURVATURE_VS_SLICE_GRAPH)}' saved.") # 영어로 변경

    # --- 히스토그램 (Radius) ---
    plt.figure(figsize=(8, 6))
    plt.hist(radius_vals, bins=15, color='blue', alpha=0.7)
    plt.title('Radius Distribution')
    plt.xlabel('Radius (mm)')
    plt.ylabel('Frequency')
    plt.savefig(RADIUS_HISTOGRAM_GRAPH)
    plt.close()
    print(f"[✓] '{os.path.basename(RADIUS_HISTOGRAM_GRAPH)}' saved.") # 영어로 변경

    # --- 히스토그램 (Curvature) ---
    plt.figure(figsize=(8, 6))
    plt.hist(curvature_vals, bins=15, color='red', alpha=0.7)
    plt.title('Curvature Distribution')
    plt.xlabel('Curvature (1/mm)')
    plt.ylabel('Frequency')
    plt.savefig(CURVATURE_HISTOGRAM_GRAPH)
    plt.close()
    print(f"[✓] '{os.path.basename(CURVATURE_HISTOGRAM_GRAPH)}' saved.") # 영어로 변경

    print(f"[✓] Basic analysis and graph generation completed for {len(slices)} slices.") # 영어로 변경


if __name__ == "__main__":
    analyze_and_save_results()
