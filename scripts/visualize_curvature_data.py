#!/usr/bin/env python3
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import cv2
from scipy.stats import mode # 최빈값 계산을 위해 추가

# --- 설정값 ---
results_file = "results/curvature.json"
input_dcm_dir = "data/dcm" # 원본 dcm 파일 경로 (이미지 예시를 위해 필요)
output_graph_dir = "results/graphs" # 그래프 저장 폴더
smoothing_window_size = 5 # 스무딩(이동 평균) 윈도우 크기. 숫자가 클수록 더 부드러워짐.

os.makedirs(output_graph_dir, exist_ok=True)

print("[*] 그래프 생성을 위한 데이터 불러오기 시작")

# 1. results.json 파일에서 데이터 불러오기
try:
    with open(results_file, "r") as f:
        results = json.load(f)
except FileNotFoundError:
    print(f"오류: '{results_file}' 파일을 찾을 수 없습니다. 분석 스크립트를 먼저 실행해주세요.")
    exit()
except json.JSONDecodeError:
    print(f"오류: '{results_file}' 파일이 올바른 JSON 형식이 아닙니다.")
    exit()

# 데이터 추출 및 정렬
slice_numbers = []
radii = []
curvatures = []
# 슬라이스 파일 이름을 숫자로 정렬하기 위해, 파일명에서 숫자 부분을 추출하여 정렬 기준으로 사용
sorted_fnames = sorted(results.keys(), key=lambda x: int(''.join(filter(str.isdigit, x))) if any(char.isdigit() for char in x) else x)

for i, fname in enumerate(sorted_fnames):
    if results[fname]['radius'] is not None and results[fname]['curvature'] is not None:
        slice_numbers.append(i) # 정렬된 순서대로 슬라이스 번호 부여
        radii.append(results[fname]['radius'])
        curvatures.append(results[fname]['curvature'])
    else:
        print(f"경고: {fname} 파일의 데이터가 유효하지 않아 그래프에서 제외합니다.")

if not slice_numbers:
    print("오류: 분석할 유효한 데이터가 없습니다. 스크립트를 종료합니다.")
    exit()

radii = np.array(radii)
curvatures = np.array(curvatures)

print(f"[✓] 총 {len(slice_numbers)}개 슬라이스 데이터 불러오기 완료.")

# --- 2차 가공 및 그래프 생성 ---

# --- 1. 슬라이스 번호에 따른 반경/곡률 추세선 그래프 (스무딩 적용) ---
print("[*] 슬라이스별 반경/곡률 추세선 그래프 생성 중...")

# 이동 평균 (Moving Average) 계산
smoothed_radii = np.convolve(radii, np.ones(smoothing_window_size)/smoothing_window_size, mode='valid')
smoothed_curvatures = np.convolve(curvatures, np.ones(smoothing_window_size)/smoothing_window_size, mode='valid')

# 스무딩된 데이터의 슬라이스 번호 범위 조정
smoothed_slice_numbers = slice_numbers[smoothing_window_size-1:]

plt.figure(figsize=(14, 7))

# Radius 플롯
ax1 = plt.gca()
ax1.plot(smoothed_slice_numbers, smoothed_radii, 'b-', label='스무딩된 반경 (mm)')
ax1.set_xlabel('슬라이스 번호')
ax1.set_ylabel('반경 (mm)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_title('슬라이스 번호에 따른 평균 반경 및 곡률 변화 추이')
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
# plt.show() # 스크립트 실행 시 바로 그래프를 보고 싶다면 주석 해제

# --- 2. 반경/곡률 분포 히스토그램 (대표값 표시) ---
print("[*] 반경/곡률 분포 히스토그램 생성 중...")

# 반경 분포 히스토그램
plt.figure(figsize=(10, 6))
plt.hist(radii, bins=20, color='royalblue', edgecolor='black', alpha=0.7)
plt.title('반경 분포 (Radius Distribution)')
plt.xlabel('반경 (mm)')
plt.ylabel('빈도수')

# 대표값 계산 및 표시
mean_radius = np.mean(radii)
median_radius = np.median(radii)
mode_result_radius = mode(radii) # scipy.stats.mode는 튜플을 반환 (mode, count)
mode_radius = mode_result_radius.mode[0] if mode_result_radius.mode.size > 0 else np.nan # 최빈값이 여러개일 수 있으므로 첫번째 값 사용

plt.axvline(mean_radius, color='red', linestyle='dashed', linewidth=2, label=f'평균: {mean_radius:.2f} mm')
plt.axvline(median_radius, color='green', linestyle='dashed', linewidth=2, label=f'중앙값: {median_radius:.2f} mm')
plt.axvline(mode_radius, color='purple', linestyle='dashed', linewidth=2, label=f'최빈값: {mode_radius:.2f} mm') # 최빈값 추가
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_graph_dir, "radius_distribution_with_stats.png"))
print("[✓] 'radius_distribution_with_stats.png' 저장 완료.")
# plt.show()

# 곡률 분포 히스토그램
plt.figure(figsize=(10, 6))
plt.hist(curvatures, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
plt.title('곡률 분포 (Curvature Distribution)')
plt.xlabel('곡률 (1/mm)')
plt.ylabel('빈도수')

# 대표값 계산 및 표시
mean_curvature = np.mean(curvatures)
median_curvature = np.median(curvatures)
mode_result_curvature = mode(curvatures)
mode_curvature = mode_result_curvature.mode[0] if mode_result_curvature.mode.size > 0 else np.nan

plt.axvline(mean_curvature, color='red', linestyle='dashed', linewidth=2, label=f'평균: {mean_curvature:.5f} 1/mm')
plt.axvline(median_curvature, color='green', linestyle='dashed', linewidth=2, label=f'중앙값: {median_curvature:.5f} 1/mm')
plt.axvline(mode_curvature, color='purple', linestyle='dashed', linewidth=2, label=f'최빈값: {mode_curvature:.5f} 1/mm') # 최빈값 추가
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_graph_dir, "curvature_distribution_with_stats.png"))
print("[✓] 'curvature_distribution_with_stats.png' 저장 완료.")
# plt.show()

# --- 3. 가장 흔한 곡률/반경을 가진 슬라이스의 이미지 예시 ---
print("[*] 대표 슬라이스 이미지 예시 생성 중...")

def analyze_and_draw(fname, output_path):
    path = os.path.join(input_dcm_dir, fname)
    try:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.uint8)
        img = cv2.equalizeHist(img) # 원본 분석 코드와 동일한 전처리
        edges = cv2.Canny(img, 50, 150) # 원본 분석 코드와 동일한 Canny 파라미터

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            print(f"경고: {fname} 에서 윤곽선을 찾을 수 없습니다. 이미지 생략.")
            return

        cnt = max(cnts, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(r)

        # Matplotlib으로 이미지 그리기 (OpenCV BGR이 아닌 RGB로 변환하여 출력)
        plt.figure(figsize=(8, 8))
        plt.imshow(img, cmap='gray') # 원본 이미지
        plt.imshow(edges, cmap='Reds', alpha=0.5) # Canny 엣지 (투명하게 오버레이)
        cv2.circle(img, center, radius, (255, 0, 0), 2) # 원본 이미지에 빨간색 원 그리기 (OpenCV는 BGR이 기본이므로 Matplotlib에서는 빨간색으로 나올 것임)

        # Matplotlib에 그릴 때는 BGR이 아니라 RGB로 이미지를 불러와야 제대로 된 색상으로 표시됨.
        # cv2.circle을 직접 plt.imshow 위에 그릴 방법이 마땅치 않으므로,
        # 원본 이미지에 그리지 않고, matplotlib의 circle 함수를 사용하여 그리는 것이 더 깔끔함.
        # 아래는 Matplotlib 함수를 사용하는 방식:
        circle_patch = plt.Circle(center, radius, color='blue', fill=False, linewidth=2, linestyle='--')
        plt.gca().add_patch(circle_patch)

        plt.title(f'슬라이스 {fname} (반경: {r:.2f}mm, 곡률: {1/r:.5f} 1/mm)')
        plt.axis('off') # 축 제거
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close() # 메모리 관리를 위해 plt.show() 대신 plt.close() 사용
        print(f"[✓] {os.path.basename(output_path)} 저장 완료.")

    except Exception as e:
        print(f"⚠️ 대표 슬라이스 이미지 생성 실패: {fname} ({e})")

# 최빈 반경에 해당하는 슬라이스 찾기
if not np.isnan(mode_radius):
    most_common_radius_fnames = []
    for fname, data in results.items():
        if data['radius'] is not None and abs(data['radius'] - mode_radius) < 0.1: # 소수점 오차 감안
            most_common_radius_fnames.append(fname)
    if most_common_radius_fnames:
        # 가장 흔한 반경을 가진 슬라이스 중 하나만 선택 (예: 첫 번째로 발견된 슬라이스)
        example_fname = sorted_fnames[slice_numbers.index(most_common_radius_fnames[0])] if most_common_radius_fnames[0] in sorted_fnames else most_common_radius_fnames[0]
        analyze_and_draw(example_fname, os.path.join(output_graph_dir, f"example_radius_{mode_radius:.2f}.png"))
    else:
        print("경고: 최빈 반경에 해당하는 예시 슬라이스를 찾을 수 없습니다.")
else:
    print("경고: 최빈 반경 값을 계산할 수 없어 예시 슬라이스를 생성하지 않습니다.")

# 최빈 곡률에 해당하는 슬라이스 찾기
if not np.isnan(mode_curvature):
    most_common_curvature_fnames = []
    for fname, data in results.items():
        if data['curvature'] is not None and abs(data['curvature'] - mode_curvature) < 0.0001: # 소수점 오차 감안
            most_common_curvature_fnames.append(fname)
    if most_common_curvature_fnames:
        example_fname = sorted_fnames[slice_numbers.index(most_common_curvature_fnames[0])] if most_common_curvature_fnames[0] in sorted_fnames else most_common_curvature_fnames[0]
        analyze_and_draw(example_fname, os.path.join(output_graph_dir, f"example_curvature_{mode_curvature:.5f}.png"))
    else:
        print("경고: 최빈 곡률에 해당하는 예시 슬라이스를 찾을 수 없습니다.")
else:
    print("경고: 최빈 곡률 값을 계산할 수 없어 예시 슬라이스를 생성하지 않습니다.")

print("[*] 모든 그래프 생성 완료.")
