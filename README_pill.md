# RT-DETRv4-X 알약 Object Detection

경구약제(알약) 이미지에서 약의 종류를 탐지하는 RT-DETRv4-X 모델입니다.

## 실험 결과

| 항목 | 값 |
|------|-----|
| 모델 | RT-DETRv4-X (HGNetv2-X backbone) |
| 파라미터 수 | 62,961,838 |
| 학습 해상도 | 640 × 640 |
| Epochs | 58 |
| Best val AP@50:95 | 0.9774 (epoch 45) |
| Best val AP@50 | 1.0000 |
| Best val AP@75 | 0.9962 |
| Kaggle 제출 점수 | **0.96833** (mAP@0.75:0.95, conf=0.1) |

## 환경 설정

```bash
# Python 3.9+, PyTorch 2.x, CUDA 11.8+
pip install -r requirements.txt
```

> **Windows 주의사항**
> - 분산 학습 백엔드: `gloo` (nccl 미지원)
> - `USE_LIBUV=0` 환경변수 필요 (libuv 미포함 PyTorch 빌드)
> - 학습 전 `set PYTHONUTF8=1` 설정 필요 (COCO JSON UTF-8 파싱)

## 데이터셋 준비

```
RT-DETRv4/
└── dataset/
    ├── images/
    │   ├── train/   # 학습 이미지
    │   └── val/     # 검증 이미지
    └── annotations/
        ├── instances_train.json
        └── instances_val.json
```

COCO 포맷 JSON이며 `remap_mscoco_category: False`로 원본 category_id를 그대로 사용합니다.

## 학습

```bash
# Windows (Anaconda Prompt)
set PYTHONUTF8=1
set USE_LIBUV=0
set RANK=0
set LOCAL_RANK=0
set WORLD_SIZE=1
set MASTER_ADDR=localhost
set MASTER_PORT=29500

python train.py -c configs/rtv4/rtv4_hgnetv2_x_pill_640.yml
```

### 주요 학습 설정

| 항목 | 값 |
|------|-----|
| Optimizer | AdamW |
| LR | 0.0005 |
| Backbone LR | 0.000005 |
| Weight decay | 0.000125 |
| Batch size | 2 |
| Epochs | 58 (flat 29 + cosine 21 + no_aug 8) |
| Mosaic | output_size=320, p=1.0 |
| MixUp | epochs 4~29 |

### Pretrained Weights

| 파일 | 위치 |
|------|------|
| RT-DETRv4-X backbone | `pretrain/` |
| DINOv3 ViT-B/16 (teacher) | `pretrain/dinov3_vitb16_pretrain_lvd1689m.pth` |

## 추론 및 제출 CSV 생성

```bash
# 기본 실행 (conf=0.1, checkpoint0047 사용)
python predict.py --test_dir path/to/test_images

# 옵션 지정
python predict.py \
  --config configs/rtv4/rtv4_hgnetv2_x_pill_640.yml \
  --checkpoint outputs/rtv4_pill/checkpoint0047.pth \
  --test_dir path/to/test_images \
  --output submission.csv \
  --conf_threshold 0.1 \
  --visualize
```

### predict.py 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--config` | `configs/rtv4/rtv4_hgnetv2_x_pill_640.yml` | YAMLConfig 경로 |
| `--checkpoint` | `outputs/rtv4_pill/checkpoint0047.pth` | 체크포인트 경로 |
| `--ann_file` | `dataset/annotations/instances_train.json` | 카테고리 매핑용 COCO JSON |
| `--test_dir` | (필수) | 테스트 이미지 폴더 |
| `--output` | `submission.csv` | 출력 CSV 파일명 |
| `--conf_threshold` | `0.1` | 신뢰도 임계값 |
| `--visualize` | False | 샘플 4장 시각화 여부 |
| `--n_samples` | 4 | 시각화할 샘플 수 |

## 출력 CSV 포맷

```
annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score
```

- `bbox`: COCO 포맷 (x_min, y_min, width, height), 픽셀 단위
- `category_id`: 원본 데이터셋 pill ID (0-based index 아님)

## 디렉토리 구조

```
RT-DETRv4/
├── configs/
│   └── rtv4/
│       ├── rtv4_hgnetv2_x_pill_640.yml   # 640px 학습/추론 설정
│       └── rtv4_hgnetv2_x_pill.yml        # 800px 학습 설정 (실험용)
├── engine/                                # 모델 소스코드
├── outputs/
│   └── rtv4_pill/
│       ├── checkpoint0047.pth             # 추론에 사용한 체크포인트
│       └── best_stg1.pth
├── pretrain/                              # Pretrained weights
├── predict.py                             # 추론 스크립트
├── train.py                               # 학습 스크립트
└── README_pill.md                         # 이 파일
```
