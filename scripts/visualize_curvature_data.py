#!/usr/bin/env python3
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import cv2
from scipy.stats import mode

# --- 설정값 ---
input_stats_file = "results/statistics.json"
input_dcm_dir = "data/dcm"
output_graph_dir = "results/final_graphs"
smoothing_window_size = 5

os.makedirs(output_graph_dir, exist_ok=True)

print("[*] Starting data and statistics loading for graph generation.") # 영어로 변경

# 1. statistics.json 파일에서 통계 데이터 불러오기
try:
    with open(input_stats_file, "r") as f:
        stats_data = json.load(f)
    print(f"[✓] Successfully loaded '{input_stats_file}'.") # 영어로 변경
except FileNotFoundError:
    print(f"Error: File '{input_stats_file}' not found. Please run analyze_curvature.py first.") # 영어로 변경
    exit()
except json.JSONDecodeError:
    print(f"Error: '{input_stats_file}' is not a valid JSON file.") # 영어로 변경
    exit()

# 통계 데이터에서 필요한 값 추출
try:
    slices = np.array(stats_data['slices'])
    fnames_in_order = stats_data['fnames']
    radius_vals = np.array(stats_data['radius_vals'])
    curvature_vals = np.array(stats_data['curvature_vals'])
    
except KeyError as e:
    print(f"Error: Required key '{e}' missing in '{input_stats_file}'. Please check analyze_curvature.py script.") # 영어로 변경
    exit()

if not slices.size or not radius_vals.size or not curvature_vals.size:
    print("Error: No valid data for analysis. Exiting script.") # 영어로 변경
    exit()

print(f"[✓] Loaded {len(slices)} slices of data.") # 영어로 변경

# --- 2차 가공 및 그래프 생성 ---

# --- 1. 슬라이스 번호에 따른 반경/곡률 추세선 그래프 (스무딩 적용) ---
print("[*] Generating smoothed radius/curvature trend graph per slice...") # 영어로 변경

# 이동 평균 (Moving Average) 계산
smoothed_radii = np.convolve(radius_vals, np.ones(smoothing_window_size)/smoothing_window_size, mode='valid')
smoothed_curvatures = np.convolve(curvature_vals, np.ones(smoothing_window_size)/smoothing_window_size, mode='valid')

# 스무딩된 데이터의 슬라이스 번호 범위 조정
smoothed_slice_numbers = slices[smoothing_window_size-1:]

plt.figure(figsize=(14, 7))

# Radius 플롯
ax1 = plt.gca()
ax1.plot(smoothed_slice_numbers, smoothed_radii, 'b-', label='Smoothed Radius (mm)') # 영어로 변경
ax1.set_xlabel('Slice Number')
ax1.set_ylabel('Radius (mm)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_title('Smoothed Average Radius and Curvature Trend vs Slice Number') # 영어로 변경
ax1.grid(True, linestyle='--', alpha=0.7)

# Curvature 플롯 (두 번째 y축 사용)
ax2 = ax1.twinx()
ax2.plot(smoothed_slice_numbers, smoothed_curvatures, 'r--', label='Smoothed Curvature (1/mm)') # 영어로 변경
ax2.set_ylabel('Curvature (1/mm)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# 범례 추가
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(output_graph_dir, "smoothed_radius_curvature_trend.png"))
print(f"[✓] 'smoothed_radius_curvature_trend.png' saved.") # 영어로 변경
plt.close()

# --- 2. 반경/곡률 분포 히스토그램 (대표값 표시) ---
print("[*] Generating radius/curvature distribution histograms...") # 영어로 변경

# 반경 분포 히스토그램
plt.figure(figsize=(10, 6))
plt.hist(radius_vals, bins=15, color='royalblue', edgecolor='black', alpha=0.7)
plt.title('Radius Distribution') # 영어로 변경
plt.xlabel('Radius (mm)') # 영어로 변경
plt.ylabel('Frequency') # 영어로 변경

# 대표값 계산 및 표시
mean_radius = np.mean(radius_vals)
median_radius = np.median(radius_vals)
mode_result_radius = mode(radius_vals, keepdims=True)
mode_radius = mode_result_radius.mode[0] if mode_result_radius.mode.size > 0 else np.nan

plt.axvline(mean_radius, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_radius:.2f} mm') # 영어로 변경
plt.axvline(median_radius, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_radius:.2f} mm') # 영어로 변경
if not np.isnan(mode_radius):
    plt.axvline(mode_radius, color='purple', linestyle='dashed', linewidth=2, label=f'Mode: {mode_radius:.2f} mm') # 영어로 변경
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_graph_dir, "radius_distribution_with_stats.png"))
print(f"[✓] 'radius_distribution_with_stats.png' saved.") # 영어로 변경
plt.close()

# 곡률 분포 히스토그램
plt.figure(figsize=(10, 6))
plt.hist(curvature_vals, bins=15, color='lightcoral', edgecolor='black', alpha=0.7)
plt.title('Curvature Distribution') # 영어로 변경
plt.xlabel('Curvature (1/mm)') # 영어로 변경
plt.ylabel('Frequency') # 영어로 변경

# 대표값 계산 및 표시
mean_curvature = np.mean(curvature_vals)
median_curvature = np.median(curvature_vals)
mode_result_curvature = mode(curvature_vals, keepdims=True)
mode_curvature = mode_result_curvature.mode[0] if mode_result_curvature.mode.size > 0 else np.nan

plt.axvline(mean_curvature, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_curvature:.5f} 1/mm') # 영어로 변경
plt.axvline(median_curvature, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_curvature:.5f} 1/mm') # 영어로 변경
if not np.isnan(mode_curvature):
    plt.axvline(mode_curvature, color='purple', linestyle='dashed', linewidth=2, label=f'Mode: {mode_curvature:.5f} 1/mm') # 영어로 변경
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_graph_dir, "curvature_distribution_with_stats.png"))
print(f"[✓] 'curvature_distribution_with_stats.png' saved.") # 영어로 변경
plt.close()

# --- 3. 가장 흔한 곡률/반경을 가진 슬라이스의 이미지 예시 ---
print("[*] Generating example slice images...") # 영어로 변경

def analyze_and_draw_example_image(fname, output_path):
    path = os.path.join(input_dcm_dir, fname)
    try:
        ds = pydicom.dcmread(path)
        img_original = ds.pixel_array.astype(np.uint8)
        img_processed = cv2.equalizeHist(img_original)
        edges = cv2.Canny(img_processed, 50, 150) 

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            print(f"Warning: No contours found for {fname}. Skipping image generation.") # 영어로 변경
            return

        cnt = max(cnts, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(r)

        plt.figure(figsize=(8, 8))
        plt.imshow(img_original, cmap='gray')
        plt.imshow(edges, cmap='Reds', alpha=0.5)

        circle_patch = plt.Circle(center, radius, color='blue', fill=False, linewidth=2, linestyle='--')
        plt.gca().add_patch(circle_patch)

        plt.title(f'Slice {fname} (Radius: {r:.2f}mm, Curvature: {1/r:.5f} 1/mm)') # 영어로 변경
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"[✓] {os.path.basename(output_path)} saved.") # 영어로 변경

    except Exception as e:
        print(f"⚠️ Failed to generate example slice image for {fname} ({e})") # 영어로 변경

# 최빈 반경에 해당하는 슬라이스 찾기
if not np.isnan(mode_radius):
    example_radius_fname = None
    min_radius_diff = float('inf')
    
    for i, r_val in enumerate(radius_vals):
        diff = abs(r_val - mode_radius)
        if diff < min_radius_diff:
            min_radius_diff = diff
            example_radius_fname = fnames_in_order[i]
    
    if example_radius_fname:
        analyze_and_draw_example_image(example_radius_fname, os.path.join(output_graph_dir, f"example_radius_{mode_radius:.2f}.png"))
    else:
        print("Warning: Could not find an example slice for mode radius.") # 영어로 변경
else:
    print("Warning: Mode radius value could not be calculated, skipping example slice generation.") # 영어로 변경

# 최빈 곡률에 해당하는 슬라이스 찾기
if not np.isnan(mode_curvature):
    example_curvature_fname = None
    min_curvature_diff = float('inf')
    
    for i, c_val in enumerate(curvature_vals):
        diff = abs(c_val - mode_curvature)
        if diff < min_curvature_diff:
            min_curvature_diff = diff
            example_curvature_fname = fnames_in_order[i]
    
    if example_curvature_fname:
        analyze_and_draw_example_image(example_curvature_fname, os.path.join(output_graph_dir, f"example_curvature_{mode_curvature:.5f}.png"))
    else:
        print("Warning: Could not find an example slice for mode curvature.") # 영어로 변경
else:
    print("Warning: Mode curvature value could not be calculated, skipping example slice generation.") # 영어로 변경

print("[*] All graphs generated successfully.") # 영어로 변경

print(f"\n[INFO] All generated graphs are saved in '{output_graph_dir}' folder.") # 영어로 변경
print("You can view these by uploading the folder as a GitHub Actions artifact.") # 영어로 변경
