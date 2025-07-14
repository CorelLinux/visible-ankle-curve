#!/usr/bin/env python3
import os, json
import cv2, pydicom
import numpy as np

input_dir = "data/dcm"
output_file = "results/curvature.json"
os.makedirs("results", exist_ok=True)
results = {}

def analyze(fname):
    path = os.path.join(input_dir, fname)
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.uint8)
    img = cv2.equalizeHist(img)
    edges = cv2.Canny(img, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return None
    cnt = max(cnts, key=cv2.contourArea)
    (_, _), r = cv2.minEnclosingCircle(cnt)
    return round(float(r), 2), round(1/float(r), 5)

print("[*] 곡률 분석 시작")
for fname in os.listdir(input_dir):
    if not fname.endswith(".dcm"): continue
    try:
        print(f" → {fname}")
        radius, curvature = analyze(fname)
        results[fname] = {
            "radius": radius,
            "curvature": curvature
        }
    except Exception as e:
        print(f"  ⚠️ 분석 실패: {fname} ({e})")

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"[✓] 총 {len(results)}개 슬라이스 분석 완료 → {output_file}")
