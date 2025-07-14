#!/data/data/com.termux/files/usr/bin/bash

set -e
cd "$(dirname "$0")/../data/dcm"

echo "[*] Visible Human ankle CT 데이터 다운로드 시작..."
wget --user-agent="Mozilla/5.0" -c https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Male/CT/Ankle/ankle.tar.gz -O ankle.tar.gz

echo "[*] 압축 해제 중..."
tar -xvzf ankle.tar.gz
rm ankle.tar.gz

echo "[*] 완료: $(ls *.dcm | wc -l)개의 파일이 준비되었습니다."
