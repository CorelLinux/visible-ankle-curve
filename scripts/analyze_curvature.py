import json
import re
import numpy as np
import matplotlib.pyplot as plt
import os # os 모듈 추가

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
    print(f"[*] '{json_path}' 에서 데이터 불러오기 시작")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"[✓] '{json_path}' 불러오기 완료.")
    except FileNotFoundError:
        print(f"오류: '{json_path}' 파일을 찾을 수 없습니다. process_canny.py를 먼저 실행해주세요.")
        return
    except json.JSONDecodeError:
        print(f"오류: '{json_path}' 파일이 올바른 JSON 형식이 아닙니다.")
        return

    # 파일명에서 슬라이스 번호 추출 및 데이터 정렬
    # 'image_N.dcm' 또는 'CT_slice(N).dcm' 형태를 예상
    # 슬라이스 번호 추출 패턴을 좀 더 유연하게 (숫자만 추출)
    # 정규식 패턴 수정: 파일명에서 숫자만 추출하여 정렬 기준으로 사용
    def extract_slice_number(filename):
        nums = re.findall(r'\d+', filename)
        if nums:
            # 여러 숫자가 있을 경우 마지막 숫자를 슬라이스 번호로 가정
            return int(nums[-1])
        return 0 # 숫자가 없으면 0 반환 (정렬의 최하위로 밀림)

    processed_data = []
    for filename, vals in data.items():
        if vals['radius'] is not None and vals['curvature'] is not None:
            slice_num = extract_slice_number(filename)
            processed_data.append((slice_num, vals['radius'], vals['curvature'], filename)) # filename도 같이 저장

    # 슬라이스 번호를 기준으로 데이터 정렬
    processed_data.sort(key=lambda x: x[0])
    
    # 정렬된 데이터 언팩
    slices, radius_vals, curvature_vals, fnames_in_order = zip(*processed_data)
    
    # 튜플을 리스트로 변환
    slices = list(slices)
    radius_vals = list(radius_vals)
    curvature_vals = list(curvature_vals)
    fnames_in_order = list(fnames_in_order)


    if not slices:
        print("경고: 분석할 유효한 슬라이스 데이터가 없습니다. 그래프 및 통계 생성을 건너뜜니다.")
        return

    # 통계 계산 및 statistics.json 저장
    print("[*] 통계 계산 및 'statistics.json' 저장 중...")
    stats = {
        'slices': slices, # 정렬된 슬라이스 번호 리스트
        'fnames': fnames_in_order, # 정렬된 파일명 리스트 (visualize_curvature_data.py에서 활용)
        'radius_vals': radius_vals, # 정렬된 반경 값 리스트
        'curvature_vals': curvature_vals, # 정렬된 곡률 값 리스트
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
    print(f"[✓] 'statistics.json' 저장 완료.")

    # --- 시각화 - 선 그래프 (기존 analyze_curvature.py의 역할) ---
    print("[*] 기본 시각화 그래프 생성 중...")
    fig, ax1 = plt.subplots(figsize=(10, 6)) # figsize 추가
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
    print(f"[✓] '{os.path.basename(CURVATURE_VS_SLICE_GRAPH)}' 저장 완료.")

    # --- 히스토그램 (Radius) ---
    plt.figure(figsize=(8, 6)) # figsize 추가
    plt.hist(radius_vals, bins=15, color='blue', alpha=0.7)
    plt.title('Radius Distribution')
    plt.xlabel('Radius (mm)')
    plt.ylabel('Frequency')
    plt.savefig(RADIUS_HISTOGRAM_GRAPH)
    plt.close()
    print(f"[✓] '{os.path.basename(RADIUS_HISTOGRAM_GRAPH)}' 저장 완료.")

    # --- 히스토그램 (Curvature) ---
    plt.figure(figsize=(8, 6)) # figsize 추가
    plt.hist(curvature_vals, bins=15, color='red', alpha=0.7)
    plt.title('Curvature Distribution')
    plt.xlabel('Curvature (1/mm)')
    plt.ylabel('Frequency')
    plt.savefig(CURVATURE_HISTOGRAM_GRAPH)
    plt.close()
    print(f"[✓] '{os.path.basename(CURVATURE_HISTOGRAM_GRAPH)}' 저장 완료.")

    print(f"[✓] 총 {len(slices)}개 슬라이스 기본 분석 및 그래프 생성 완료.")


if __name__ == "__main__":
    analyze_and_save_results()

