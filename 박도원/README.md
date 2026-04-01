## 📂 파일 구조 (Directory Structure)

```text
박도원/
├── data_preprocessing/      # 데이터 전처리 스크립트
│   ├── 01_dataset_eda.py           # 데이터셋 EDA
│   ├── 02_convert_json_to_yolo.py  # 라벨 형식 변환 (JSON -> txt)
│   ├── 03_extract_pill_crops.py    # 소수클래스 알약 크롭
│   ├── 04_generate_synthetic.py    # 합성 데이터 생성(개별 알약 데이터 + 백그라운드 합성)
│   ├── 05_bbox_check.py            # 합성데이터 바운딩 박스 검토
│   └── extract_class_ids.py        # 클래스 ID 추출
├── eda_visualization/       # 데이터 분석 시각화 결과
│   ├── check_bbox_0~9.png          # BBox 시각화 검증 이미지
│   ├── pill_bottom20.png           # 클래스 분포 하위 20개 알약 분포
│   ├── pill_top20.png              # 클래스 분포 상위 20개 알약 분포
│   └── pill_full_distribution.png  # 전체 데이터 클래스 분포도
├── models/                  # 모델 학습코드
│   ├── rt-detr-l.ipynb             # RT-DETR-L 실험 코드
│   ├── YOLO11m.ipynb               # YOLO11m 실험 코드 + 합성 데이터 추가
│   ├── YOLOv12.ipynb               # YOLO12n 실험 코드
│   └── WBF_csv.ipynb               # WBF-RT-DETRv4-x + YOLO11m
├── .gitignore             
└── README.md                # v
