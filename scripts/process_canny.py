import os, json
import cv2, pydicom
import numpy as np

INPUT_DIR = os.path.abspath("data/dcm")
OUTPUT_JSON = os.path.abspath("results/curvature.json")
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

def extract_radius(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.uint8)
    img = cv2.equalizeHist(img)
    edges = cv2.Canny(img, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(cnt)
    return round(float(r), 2)

results = {}
for fname in sorted(os.listdir(INPUT_DIR)):
    if fname.lower().endswith(".dcm"):
        path = os.path.join(INPUT_DIR, fname)
        r = extract_radius(path)
        if r:
            results[fname] = {"radius": r, "curvature": round(1/r, 5)}

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(f"[+] 처리 완료! {len(results)}개 파일에서 곡률 결과 생성됨.")
