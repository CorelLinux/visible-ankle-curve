import json
import re
import matplotlib.pyplot as plt

# 결과 파일 경로
RESULT_JSON = 'results/curvature.json'

def plot_curvature_and_radius(json_path=RESULT_JSON):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 파일명에서 슬라이스 번호 추출해서 정렬
    pattern = re.compile(r'\((\d+)\)')  # 괄호 안 숫자 추출용

    slices = []
    radius_vals = []
    curvature_vals = []

    for filename, vals in data.items():
        match = pattern.search(filename)
        if match:
            slice_num = int(match.group(1))
            slices.append(slice_num)
            radius_vals.append(vals['radius'])
            curvature_vals.append(vals['curvature'])
        else:
            # 번호 못 뽑으면 무시
            continue

    # 슬라이스 번호 기준으로 정렬
    sorted_data = sorted(zip(slices, radius_vals, curvature_vals), key=lambda x: x[0])
    slices, radius_vals, curvature_vals = zip(*sorted_data)

    # 그래프 그리기
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Slice Number')
    ax1.set_ylabel('Radius (mm)', color='tab:blue')
    ax1.plot(slices, radius_vals, marker='o', linestyle='-', color='tab:blue', label='Radius')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Curvature (1/mm)', color='tab:red')
    ax2.plot(slices, curvature_vals, marker='x', linestyle='--', color='tab:red', label='Curvature')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Curvature and Radius vs Slice Number')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_curvature_and_radius()
