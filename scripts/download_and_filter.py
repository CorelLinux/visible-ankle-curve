#!/usr/bin/env python3
import os, tarfile, requests
import pydicom

# 1. 설정
DATA_DIR = "data/dcm"
URL = "https://dataverse.harvard.edu/api/access/datafile/3086866"
ARCHIVE_PATH = os.path.join(DATA_DIR, "VH_M_CT.tar.bz2")
os.makedirs(DATA_DIR, exist_ok=True)

# 2. 다운로드
print("[*] Downloading CT archive...")
r = requests.get(URL, stream=True)
with open(ARCHIVE_PATH, "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
print("[✓] Download complete.")

# 3. 압축 해제
print("[*] Extracting archive...")
with tarfile.open(ARCHIVE_PATH, "r:bz2") as tar:
    tar.extractall(DATA_DIR)

# 4. 필터링 - ankle만 유지
print("[*] Filtering for ankle slices...")
count = 0
for fname in os.listdir(DATA_DIR):
    fpath = os.path.join(DATA_DIR, fname)
    if not fname.lower().endswith(".dcm"):
        continue
    try:
        ds = pydicom.dcmread(fpath, stop_before_pixels=True)
        desc = str(getattr(ds, "SeriesDescription", "")).lower()
        if "ankle" not in fname.lower() and "ankle" not in desc:
            os.remove(fpath)
        else:
            count += 1
    except Exception as e:
        os.remove(fpath)

print(f"[✓] Filtering done. {count} ankle-related slices kept.")

# 5. 압축파일 삭제
os.remove(ARCHIVE_PATH)
