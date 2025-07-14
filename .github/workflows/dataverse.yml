name: Download from Dataverse & Process

on:
  workflow_dispatch:
  push:
    paths:
      - 'scripts/download_dataverse.py'

jobs:
  download-and-process:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with: python-version: '3.10'
      - name: Install libs
        run: pip install requests pydicom opencv-python-headless numpy
      - name: Download DICOMs
        run: python scripts/download_dataverse.py
      - name: Process Canny Curvature
        run: python scripts/process_canny.py
      - name: Commit results
        run: |
          git config user.name "ankle-bot"
          git config user.email "bot@example.com"
          git add data/dcm/*.dcm results/curvature.json
          git commit -m "📥 Dataverse download + 곡률 분석" || echo "No changes"
          git push
