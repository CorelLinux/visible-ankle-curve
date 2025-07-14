#!/usr/bin/env python3
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import cv2
from scipy.stats import mode
from matplotlib import font_manager, rc

# --- 한글 폰트 설정 시작 (수정된 부분) ---
# 직접 다운로드한 폰트 파일 경로 지정
FONT_PATH = 'scripts/fonts/NotoSansKR-Regular.otf'

# 폰트 매니저에 폰트 추가 및 설정
if os.path.exists(FONT_PATH):
    font_name = font_manager.FontProperties(fname=FONT_PATH).get_name()
    rc('font', family=font_name)
    print(f"'{FONT_PATH}' 폰트를 사용하여 Matplotlib 설정 완료.")
else:
    # 폰트 파일이 없을 경우 경고 메시지 출력 및 기본 폰트 사용
    print(f"경고: 폰트 파일 '{FONT_PATH}'을(를) 찾을 수 없습니다. 기본 'sans-serif' 폰트를 사용합니다. 한글이 깨질 수 있습니다.")
    rc('font', family='sans-serif')

plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
# --- 한글 폰트 설정 끝 ---

# --- 설정값 ---
input_stats_file = "results/statistics.json"
input_dcm_dir = "data/dcm"
output_graph_dir = "results/final_graphs"
smoothing_window_size = 5

os.makedirs(output_graph_dir, exist_ok=True)

# ... (나머지 visualize_curvature_data.py 코드는 동일)

print("[*] 그래프 생성을 위한 데이터 및 통계 불러오기 시작")

# 1. statistics.json 파일에서 통계 데이터 불러오기
try:
    with open(input_stats_file, "r") as f:
        stats_data = json.load(f)
    print(f"[✓] '{input_stats_file}' 불러오기 완료.")
except FileNotFoundError:
    print(f"오류: '{input_stats_file}' 파일을 찾을 수 없습니다. analyze_curvature.py 스크립트를 먼저 실행해주세요.")
    exit()
except json.JSONDecodeError:
    print(f"오류: '{input_stats_file}' 파일이 올바른 JSON 형식이 아닙니다.")
    exit()

# 통계 데이터에서 필요한 값 추출
try:
    slices = np.array(stats_data['slices'])
    radius_vals = np.array(stats_data['radius_vals'])
    curvature_vals = np.array(stats_data['curvature_vals'])
    
    # 대표값은 statistics.json에 이미 계산되어 있을 수 있지만, 
    # visualize_curvature_data.py 에서 다시 계산하여 일관성을 유지
    # 또는 statistics.json에 포함시켜서 불러오는 것이 더 안정적.
    # 여기서는 다시 계산하는 방식으로 진행.
    
except KeyError as e:
    print(f"오류: '{input_stats_file}' 파일에 필요한 키({e})가 없습니다. analyze_curvature.py 스크립트를 확인해주세요.")
    exit()

if not slices.size or not radius_vals.size or not curvature_vals.size:
    print("오류: 분석할 유효한 데이터가 없습니다. 스크립트를 종료합니다.")
    exit()

print(f"[✓] 총 {len(slices)}개 슬라이스 데이터 불러오기 완료.")

# --- 2차 가공 및 그래프 생성 ---

# --- 1. 슬라이스 번호에 따른 반경/곡률 추세선 그래프 (스무딩 적용) ---
print("[*] 슬라이스별 반경/곡률 추세선 그래프 생성 중...")

# 이동 평균 (Moving Average) 계산
smoothed_radii = np.convolve(radius_vals, np.ones(smoothing_window_size)/smoothing_window_size, mode='valid')
smoothed_curvatures = np.convolve(curvature_vals, np.ones(smoothing_window_size)/smoothing_window_size, mode='valid')

# 스무딩된 데이터의 슬라이스 번호 범위 조정
smoothed_slice_numbers = slices[smoothing_window_size-1:]

plt.figure(figsize=(14, 7))

# Radius 플롯
ax1 = plt.gca()
ax1.plot(smoothed_slice_numbers, smoothed_radii, 'b-', label='스무딩된 반경 (mm)')
ax1.set_xlabel('슬라이스 번호')
ax1.set_ylabel('반경 (mm)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_title('슬라이스 번호에 따른 평균 반경 및 곡률 변화 추이 (Smoothed)')
ax1.grid(True, linestyle='--', alpha=0.7)

# Curvature 플롯 (두 번째 y축 사용)
ax2 = ax1.twinx()
ax2.plot(smoothed_slice_numbers, smoothed_curvatures, 'r--', label='스무딩된 곡률 (1/mm)')
ax2.set_ylabel('곡률 (1/mm)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# 범례 추가
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(output_graph_dir, "smoothed_radius_curvature_trend.png"))
print("[✓] 'smoothed_radius_curvature_trend.png' 저장 완료.")
plt.close() # 메모리 관리를 위해 그래프를 닫음

# --- 2. 반경/곡률 분포 히스토그램 (대표값 표시) ---
print("[*] 반경/곡률 분포 히스토그램 생성 중...")

# 반경 분포 히스토그램
plt.figure(figsize=(10, 6))
plt.hist(radius_vals, bins=15, color='royalblue', edgecolor='black', alpha=0.7) # analyze_curvature.py와 동일한 bin 수 (15)
plt.title('반경 분포 (Radius Distribution)')
plt.xlabel('반경 (mm)')
plt.ylabel('빈도수')

# 대표값 계산 및 표시
mean_radius = np.mean(radius_vals)
median_radius = np.median(radius_vals)
mode_result_radius = mode(radius_vals, keepdims=True) # keepdims=True 추가하여 미래 버전 호환성 확보
mode_radius = mode_result_radius.mode[0] if mode_result_radius.mode.size > 0 else np.nan

plt.axvline(mean_radius, color='red', linestyle='dashed', linewidth=2, label=f'평균: {mean_radius:.2f} mm')
plt.axvline(median_radius, color='green', linestyle='dashed', linewidth=2, label=f'중앙값: {median_radius:.2f} mm')
if not np.isnan(mode_radius): # 최빈값이 있을 경우에만 표시
    plt.axvline(mode_radius, color='purple', linestyle='dashed', linewidth=2, label=f'최빈값: {mode_radius:.2f} mm')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_graph_dir, "radius_distribution_with_stats.png"))
print("[✓] 'radius_distribution_with_stats.png' 저장 완료.")
plt.close()

# 곡률 분포 히스토그램
plt.figure(figsize=(10, 6))
plt.hist(curvature_vals, bins=15, color='lightcoral', edgecolor='black', alpha=0.7) # analyze_curvature.py와 동일한 bin 수 (15)
plt.title('곡률 분포 (Curvature Distribution)')
plt.xlabel('곡률 (1/mm)')
plt.ylabel('빈도수')

# 대표값 계산 및 표시
mean_curvature = np.mean(curvature_vals)
median_curvature = np.median(curvature_vals)
mode_result_curvature = mode(curvature_vals, keepdims=True) # keepdims=True 추가
mode_curvature = mode_result_curvature.mode[0] if mode_result_curvature.mode.size > 0 else np.nan

plt.axvline(mean_curvature, color='red', linestyle='dashed', linewidth=2, label=f'평균: {mean_curvature:.5f} 1/mm')
plt.axvline(median_curvature, color='green', linestyle='dashed', linewidth=2, label=f'중앙값: {median_curvature:.5f} 1/mm')
if not np.isnan(mode_curvature): # 최빈값이 있을 경우에만 표시
    plt.axvline(mode_curvature, color='purple', linestyle='dashed', linewidth=2, label=f'최빈값: {mode_curvature:.5f} 1/mm')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_graph_dir, "curvature_distribution_with_stats.png"))
print("[✓] 'curvature_distribution_with_stats.png' 저장 완료.")
plt.close()

# --- 3. 가장 흔한 곡률/반경을 가진 슬라이스의 이미지 예시 ---
print("[*] 대표 슬라이스 이미지 예시 생성 중...")

def analyze_and_draw(fname, output_path):
    path = os.path.join(input_dcm_dir, fname)
    try:
        ds = pydicom.dcmread(path)
        img_original = ds.pixel_array.astype(np.uint8)
        img_processed = cv2.equalizeHist(img_original)
        # analyze_curvature.py 에서 사용된 Canny 파라미터와 동일하게 유지
        edges = cv2.Canny(img_processed, 50, 150) 

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            print(f"경고: {fname} 에서 윤곽선을 찾을 수 없습니다. 이미지 생략.")
            return

        cnt = max(cnts, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(r)

        plt.figure(figsize=(8, 8))
        plt.imshow(img_original, cmap='gray') # 원본 이미지 표시
        plt.imshow(edges, cmap='Reds', alpha=0.5) # Canny 엣지 (투명하게 오버레이)

        # Matplotlib에 원 그리기
        circle_patch = plt.Circle(center, radius, color='blue', fill=False, linewidth=2, linestyle='--')
        plt.gca().add_patch(circle_patch)

        plt.title(f'슬라이스 {fname} (반경: {r:.2f}mm, 곡률: {1/r:.5f} 1/mm)')
        plt.axis('off') # 축 제거
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"[✓] {os.path.basename(output_path)} 저장 완료.")

    except Exception as e:
        print(f"⚠️ 대표 슬라이스 이미지 생성 실패: {fname} ({e})")

# 최빈 반경에 해당하는 슬라이스 찾기
if not np.isnan(mode_radius):
    # 최빈 반경 값에 가장 가까운 슬라이스 파일명 찾기
    # analyze_curvature.py에서 사용된 파일명 순서를 유지하기 위해
    # stats_data['files'] (또는 이에 상응하는) 정보가 있다면 그것을 활용.
    # 현재 stats_data에는 파일명 순서 정보가 없으므로,
    # radius_vals 배열의 인덱스를 사용하여 원본 파일명을 역추적해야 함.
    # 이를 위해 analyze_curvature.py에서 파일명 리스트도 함께 저장하도록 수정이 필요함.
    # 일단은 stats_data에서 슬라이스별 파일명을 가져오는 것을 가정하고,
    # 없으면 sorted_fnames를 직접 만들어서 사용.
    
    # analyze_curvature.py의 statistics.json에 'fnames' 리스트를 추가했다는 가정하에 진행
    if 'fnames' in stats_data:
        all_fnames = stats_data['fnames']
    else:
        # 'fnames'가 없다면, 기존 results.json에서 사용했던 방식으로 정렬하여 사용
        # (이 경우 visualize_curvature_data.py 스크립트 상단에 results.json 로드 로직 필요)
        # 여기서는 간단하게 input_dcm_dir에서 바로 정렬하여 사용
        all_fnames = sorted(os.listdir(input_dcm_dir), key=lambda x: int(''.join(filter(str.isdigit, x))) if any(char.isdigit() for char in x) else x)
        all_fnames = [f for f in all_fnames if f.endswith(".dcm")]


    example_radius_fname = None
    min_radius_diff = float('inf')
    
    # 모든 파일명에 대해 순회하며 가장 가까운 반경 값을 가진 파일 찾기
    for i, r_val in enumerate(radius_vals):
        if abs(r_val - mode_radius) < min_radius_diff:
            min_radius_diff = abs(r_val - mode_radius)
            example_radius_fname = all_fnames[i] # 해당 인덱스의 파일명

    if example_radius_fname:
        analyze_and_draw(example_radius_fname, os.path.join(output_graph_dir, f"example_radius_{mode_radius:.2f}.png"))
    else:
        print("경고: 최빈 반경에 해당하는 예시 슬라이스를 찾을 수 없습니다.")
else:
    print("경고: 최빈 반경 값을 계산할 수 없어 예시 슬라이스를 생성하지 않습니다.")

# 최빈 곡률에 해당하는 슬라이스 찾기
if not np.isnan(mode_curvature):
    example_curvature_fname = None
    min_curvature_diff = float('inf')
    
    for i, c_val in enumerate(curvature_vals):
        if abs(c_val - mode_curvature) < min_curvature_diff:
            min_curvature_diff = abs(c_val - mode_curvature)
            example_curvature_fname = all_fnames[i] # 해당 인덱스의 파일명
    
    if example_curvature_fname:
        analyze_and_draw(example_curvature_fname, os.path.join(output_graph_dir, f"example_curvature_{mode_curvature:.5f}.png"))
    else:
        print("경고: 최빈 곡률에 해당하는 예시 슬라이스를 찾을 수 없습니다.")
else:
    print("경고: 최빈 곡률 값을 계산할 수 없어 예시 슬라이스를 생성하지 않습니다.")

print("[*] 모든 그래프 생성 완료.")

# GitHub Actions의 artifact로 업로드하기 위한 메시지 (선택 사항)
# 실제 업로드는 .yml 파일에서 actions/upload-artifact@v4를 통해 이루어짐.
print(f"\n[INFO] 생성된 모든 그래프는 '{output_graph_dir}' 폴더에 저장되었습니다.")
print("이 폴더의 내용을 GitHub Actions artifact로 업로드하여 확인할 수 있습니다.")

