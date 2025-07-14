#!/usr/bin/env bash

set -e

echo "[*] DICOM 디렉터리 확인 중..."
ls data/dcm/*.dcm || { echo "❌ DICOM 파일 없음. data/dcm/에 업로드 필요"; exit 1; }

echo "[*] 곡률 분석 시작..."
python scripts/process_canny.py

echo "[+] 완료! results/curvature.json 생성됨"
