"""
RT-DETRv4-X 추론 스크립트
- 학습된 체크포인트로 테스트 이미지 추론
- 제출용 CSV 생성
- 샘플 예측 시각화

사용법:
    python predict.py --checkpoint outputs/rtv4_pill/checkpoint0047.pth \
                      --test_dir path/to/test_images \
                      --output submission.csv \
                      --conf_threshold 0.1 \
                      --visualize
"""

import argparse
import csv
import glob
import json
import os
import random
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as T
from pathlib import Path
from PIL import Image

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
RTDETRV4_DIR = os.path.dirname(os.path.abspath(__file__))
if RTDETRV4_DIR not in sys.path:
    sys.path.insert(0, RTDETRV4_DIR)

from engine.core import YAMLConfig


# ── 모델 래퍼 ──────────────────────────────────────────────────────────────────
class RTv4Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        return self.postprocessor(self.model(images), orig_target_sizes)


def init_dist():
    """노트북/단일 GPU 환경에서 분산 학습 없이 실행하기 위한 process group 초기화."""
    if not dist.is_initialized():
        os.environ["USE_LIBUV"] = "0"
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(
            backend="gloo", init_method="env://", world_size=1, rank=0
        )


def load_model(config_path, checkpoint_path, device):
    """YAMLConfig로 모델 빌드 후 체크포인트 로드."""
    cfg = YAMLConfig(config_path, resume=checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "ema" in ckpt:
        state = ckpt["ema"]["module"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    cfg.model.load_state_dict(state)
    model = RTv4Model(cfg).to(device).eval()
    return model


def build_idx2cat(ann_file):
    """COCO JSON에서 0-based label index → 원본 category_id 역매핑 생성."""
    with open(ann_file, encoding="utf-8") as f:
        coco = json.load(f)
    cat2idx = {cat["id"]: i for i, cat in enumerate(coco["categories"])}
    idx2cat = {v: k for k, v in cat2idx.items()}
    id2name = {cat["id"]: cat["name"] for cat in coco["categories"]}
    return idx2cat, id2name


def run_inference(model, img_path, transforms, device, conf_threshold):
    """이미지 1장 추론 → (labels, boxes, scores) 반환 (conf 필터 적용)."""
    img_pil = Image.open(img_path).convert("RGB")
    w, h = img_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)
    im_data = transforms(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        labels, boxes, scores = model(im_data, orig_size)

    labels = labels[0]
    boxes = boxes[0]
    scores = scores[0]

    mask = scores > conf_threshold
    return labels[mask], boxes[mask], scores[mask]


# ── 제출 CSV 생성 ──────────────────────────────────────────────────────────────
def generate_submission(model, test_dir, transforms, device,
                        conf_threshold, idx2cat, output_csv):
    test_img_paths = sorted(
        glob.glob(os.path.join(test_dir, "*.*")),
        key=lambda p: int(Path(p).stem)
    )
    print(f"테스트 이미지 수: {len(test_img_paths)}장")

    rows = []
    annotation_id = 1

    for img_path in test_img_paths:
        image_id = int(Path(img_path).stem)
        labels, boxes, scores = run_inference(
            model, img_path, transforms, device, conf_threshold
        )

        for label, box, score in zip(labels.tolist(), boxes.tolist(), scores.tolist()):
            cat_id = idx2cat.get(int(label), int(label))
            x1, y1, x2, y2 = box
            rows.append([
                annotation_id, image_id, cat_id,
                round(x1), round(y1),
                round(x2 - x1), round(y2 - y1),
                round(score, 4)
            ])
            annotation_id += 1

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "annotation_id", "image_id", "category_id",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"
        ])
        writer.writerows(rows)

    print(f"저장 완료: {output_csv}  (총 탐지 수: {len(rows)}개)")


# ── 샘플 시각화 ────────────────────────────────────────────────────────────────
def visualize_samples(model, test_dir, transforms, device,
                      conf_threshold, idx2cat, id2name, n_samples=4):
    img_paths = sorted(
        glob.glob(os.path.join(test_dir, "*.*")),
        key=lambda p: int(Path(p).stem)
    )
    sample_paths = random.sample(img_paths, min(n_samples, len(img_paths)))

    fig, axes = plt.subplots(1, len(sample_paths), figsize=(5 * len(sample_paths), 5))
    if len(sample_paths) == 1:
        axes = [axes]

    for ax, img_path in zip(axes, sample_paths):
        img_pil = Image.open(img_path).convert("RGB")
        labels, boxes, scores = run_inference(
            model, img_path, transforms, device, conf_threshold
        )

        ax.imshow(img_pil)
        for label, box, score in zip(labels.tolist(), boxes.tolist(), scores.tolist()):
            x1, y1, x2, y2 = box
            cat_id = idx2cat.get(int(label), int(label))
            name = id2name.get(cat_id, str(cat_id))
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 4, f"{name} {score:.2f}",
                    fontsize=7, color="red",
                    bbox=dict(facecolor="white", alpha=0.6, pad=1, edgecolor="none"))

        ax.set_title(Path(img_path).name, fontsize=9)
        ax.axis("off")

    plt.suptitle(f"RT-DETRv4 샘플 예측 (conf >= {conf_threshold})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("시각화 저장: sample_predictions.png")


# ── 메인 ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="RT-DETRv4-X 추론 스크립트")
    parser.add_argument(
        "--config",
        default="configs/rtv4/rtv4_hgnetv2_x_pill_640.yml",
        help="YAMLConfig 경로"
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/rtv4_pill/checkpoint0047.pth",
        help="체크포인트 경로"
    )
    parser.add_argument(
        "--ann_file",
        default="dataset/annotations/instances_train.json",
        help="카테고리 매핑용 COCO JSON 경로"
    )
    parser.add_argument(
        "--test_dir",
        required=True,
        help="테스트 이미지 폴더 경로"
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="출력 CSV 파일명"
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.1,
        help="신뢰도 임계값 (기본: 0.1)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="샘플 예측 시각화 (4장)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="시각화할 샘플 수 (기본: 4)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 분산 환경 초기화
    init_dist()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 모델 로드
    config_path = os.path.join(RTDETRV4_DIR, args.config)
    checkpoint_path = os.path.join(RTDETRV4_DIR, args.checkpoint)
    print(f"config: {config_path}")
    print(f"checkpoint: {checkpoint_path}")

    model = load_model(config_path, checkpoint_path, device)
    print("모델 로드 완료")

    # 카테고리 매핑
    ann_file = os.path.join(RTDETRV4_DIR, args.ann_file)
    idx2cat, id2name = build_idx2cat(ann_file)

    # 전처리
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    # 제출 CSV 생성
    generate_submission(
        model, args.test_dir, transforms, device,
        args.conf_threshold, idx2cat, args.output
    )

    # 시각화 (선택)
    if args.visualize:
        visualize_samples(
            model, args.test_dir, transforms, device,
            args.conf_threshold, idx2cat, id2name, args.n_samples
        )


if __name__ == "__main__":
    main()
