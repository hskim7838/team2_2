"""
배경 교체 데이터 증강 스크립트

알약 이미지의 회색 배경을 다양한 색상으로 교체하여 배경 다양성을 늘립니다.
배경 감지: 이미지 가장자리에서 flood fill로 배경 마스크 생성.
교체 색상: 다양한 밝기의 회색 계열 + 미세한 색상 틴트 + 가우시안 노이즈.

사용법:
    python bg_replace_augment.py

출력:
    - dataset/images/train_aug/   : bg 교체 이미지 추가
    - dataset/annotations/instances_train_aug.json : 업데이트된 COCO JSON
"""

import json
import os
import random
from pathlib import Path
from collections import deque

import numpy as np
from PIL import Image

# ── 설정 ──────────────────────────────────────────────────────────────────────
RTDETRV4_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_JSON     = os.path.join(RTDETRV4_DIR, "dataset/annotations/instances_train_aug.json")
SRC_IMG_DIR  = os.path.join(RTDETRV4_DIR, "dataset/images/train_aug")
AUG_JSON     = SRC_JSON
AUG_IMG_DIR  = SRC_IMG_DIR

# 원본 이미지당 생성할 배경 교체 버전 수
N_BG_PER_IMAGE = 2

# 배경 감지: flood fill seed를 이미지 가장자리에서 시작
# 이웃 픽셀과의 색상 차이가 이 값 이하면 같은 배경으로 판단
FLOOD_TOLERANCE = 30

# 교체할 배경 색상 범위
BG_BASE_MIN = 100   # 최소 밝기
BG_BASE_MAX = 220   # 최대 밝기
BG_TINT_MAX = 15    # R/G/B 틴트 최대 편차
BG_NOISE_STD = 8    # 가우시안 노이즈 표준편차

random.seed(42)
np.random.seed(42)

# ── 배경 마스크 생성 (flood fill from edges) ──────────────────────────────────
def get_bg_mask(img_np: np.ndarray) -> np.ndarray:
    """이미지 가장자리에서 flood fill로 배경 마스크(True=배경) 생성"""
    H, W = img_np.shape[:2]
    visited = np.zeros((H, W), dtype=bool)
    mask    = np.zeros((H, W), dtype=bool)

    # seed: 이미지 4개 가장자리 픽셀
    seeds = []
    for x in range(W):
        seeds.append((0, x))
        seeds.append((H - 1, x))
    for y in range(H):
        seeds.append((y, 0))
        seeds.append((y, W - 1))

    queue = deque()
    for (y, x) in seeds:
        if not visited[y, x]:
            visited[y, x] = True
            queue.append((y, x))

    seed_color = img_np[0, 0].astype(float)  # 기준 배경 색상 (top-left)

    while queue:
        y, x = queue.popleft()
        pixel = img_np[y, x].astype(float)
        diff = np.abs(pixel - seed_color).max()
        if diff > FLOOD_TOLERANCE:
            continue
        mask[y, x] = True
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))

    return mask


def make_bg_color(H: int, W: int) -> np.ndarray:
    """랜덤 배경 색상 배열 생성 (H x W x 3, uint8)"""
    base  = random.randint(BG_BASE_MIN, BG_BASE_MAX)
    tint  = np.array([
        random.randint(-BG_TINT_MAX, BG_TINT_MAX),
        random.randint(-BG_TINT_MAX, BG_TINT_MAX),
        random.randint(-BG_TINT_MAX, BG_TINT_MAX),
    ], dtype=float)
    flat_color = np.clip(base + tint, 0, 255)
    bg = np.full((H, W, 3), flat_color, dtype=float)
    noise = np.random.normal(0, BG_NOISE_STD, (H, W, 3))
    bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
    return bg


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    with open(SRC_JSON, encoding='utf-8') as f:
        coco = json.load(f)

    os.makedirs(AUG_IMG_DIR, exist_ok=True)

    # 원본(aug 포함) 이미지 목록 — prefix 'aug_'가 없는 원본만 처리
    orig_images = [img for img in coco['images']
                   if not img['file_name'].startswith('bg_')]

    existing_ids  = {img['id'] for img in coco['images']}
    new_img_id    = max(existing_ids) + 1
    new_ann_id    = max(a['id'] for a in coco['annotations']) + 1

    img2anns = {}
    for a in coco['annotations']:
        img2anns.setdefault(a['image_id'], []).append(a)

    new_images      = []
    new_annotations = []
    count           = 0

    for img_info in orig_images:
        src_path = os.path.join(SRC_IMG_DIR, img_info['file_name'])
        if not os.path.exists(src_path):
            continue

        img_np = np.array(Image.open(src_path).convert('RGB'))
        H, W   = img_np.shape[:2]

        try:
            bg_mask = get_bg_mask(img_np)
        except Exception:
            continue

        src_anns = img2anns.get(img_info['id'], [])

        for k in range(N_BG_PER_IMAGE):
            new_bg  = make_bg_color(H, W)
            result  = img_np.copy()
            result[bg_mask] = new_bg[bg_mask]

            stem    = Path(img_info['file_name']).stem
            new_fname = f"bg_{stem}_{k}.png"
            save_path = os.path.join(AUG_IMG_DIR, new_fname)
            Image.fromarray(result).save(save_path)

            new_images.append({
                'id':        new_img_id,
                'file_name': new_fname,
                'width':     W,
                'height':    H,
            })

            for ann in src_anns:
                new_annotations.append({
                    **ann,
                    'id':       new_ann_id,
                    'image_id': new_img_id,
                })
                new_ann_id += 1

            new_img_id += 1
            count += 1

    coco['images']      += new_images
    coco['annotations'] += new_annotations

    with open(AUG_JSON, 'w', encoding='utf-8') as f:
        json.dump(coco, f)

    print(f"완료!")
    print(f"  배경 교체 이미지 생성: {count}장 (+{N_BG_PER_IMAGE}장/원본)")
    print(f"  전체 이미지 수: {len(coco['images'])}장")
    print(f"  저장 위치: {AUG_IMG_DIR}")


if __name__ == '__main__':
    main()
