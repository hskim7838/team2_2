"""
Copy-Paste 데이터 증강 스크립트

희귀 클래스(annotation 수가 적은) 이미지를 다른 이미지에 복사-붙여넣기하여
클래스 불균형을 완화합니다.

사용법:
    python copy_paste_augment.py

출력:
    - dataset/images/train_aug/   : 증강된 이미지
    - dataset/annotations/instances_train_aug.json : 증강된 COCO JSON
"""

import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# ── 설정 ──────────────────────────────────────────────────────────────────────
RTDETRV4_DIR  = os.path.dirname(os.path.abspath(__file__))
TRAIN_JSON    = os.path.join(RTDETRV4_DIR, "dataset/annotations/instances_train.json")
TRAIN_IMG_DIR = os.path.join(RTDETRV4_DIR, "dataset/images/train")
AUG_IMG_DIR   = os.path.join(RTDETRV4_DIR, "dataset/images/train_aug")
AUG_JSON      = os.path.join(RTDETRV4_DIR, "dataset/annotations/instances_train_aug.json")

# 클래스별 반복 횟수 (annotation 수 기준)
REPEAT_SCHEDULE = {
    1:  12,   # 1개짜리 → 12회 붙여넣기
    2:   7,   # 2개짜리 → 7회
    3:   4,   # 3개짜리 → 4회
    4:   3,   # 4개짜리 → 3회
    5:   2,   # 5개짜리 → 2회
}

IOU_OVERLAP_THR = 0.05   # 붙여넣기 시 기존 bbox와 허용 overlap
MAX_PASTE_TRIES = 50     # 위치 탐색 최대 시도 횟수
SCALE_RANGE     = (0.8, 1.2)  # 붙여넣기 시 크기 조정 범위
random.seed(42)
np.random.seed(42)


# ── 유틸 함수 ─────────────────────────────────────────────────────────────────
def iou(boxA, boxB):
    """xywh → IoU 계산"""
    ax1, ay1, aw, ah = boxA
    bx1, by1, bw, bh = boxB
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def coverage(paste_box, exist_box):
    """paste_box가 exist_box 면적의 몇 %를 가리는지 계산"""
    px1, py1, pw, ph = paste_box
    px2, py2 = px1 + pw, py1 + ph
    bx1, by1, bw, bh = exist_box
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(px1, bx1), max(py1, by1)
    ix2, iy2 = min(px2, bx2), min(py2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    exist_area = bw * bh
    return inter / exist_area if exist_area > 0 else 0.0


def find_paste_position(W, H, pw, ph, existing_boxes):
    """겹치지 않는 붙여넣기 위치 탐색 (IoU + coverage 체크)"""
    for _ in range(MAX_PASTE_TRIES):
        x = random.randint(0, max(0, W - pw))
        y = random.randint(0, max(0, H - ph))
        candidate = [x, y, pw, ph]
        # IoU 체크 + 기존 약을 20% 이상 가리면 스킵
        if (all(iou(candidate, b) < IOU_OVERLAP_THR for b in existing_boxes) and
                all(coverage(candidate, b) < 0.05 for b in existing_boxes)):
            return x, y
    return None


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    with open(TRAIN_JSON, encoding="utf-8") as f:
        coco = json.load(f)

    id2img  = {img["id"]: img for img in coco["images"]}
    id2name = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # 이미지별 annotation 목록
    img2anns = defaultdict(list)
    for ann in coco["annotations"]:
        img2anns[ann["image_id"]].append(ann)

    # 클래스별 annotation 수 & 이미지 목록
    class_cnt  = Counter(ann["category_id"] for ann in coco["annotations"])
    class2imgs = defaultdict(list)
    for ann in coco["annotations"]:
        class2imgs[ann["category_id"]].append(ann["image_id"])

    print("=== 클래스 불균형 현황 ===")
    for cat_id, cnt in sorted(class_cnt.items(), key=lambda x: x[1]):
        repeat = REPEAT_SCHEDULE.get(cnt, 0)
        mark   = f"← {repeat}회 붙여넣기 예정" if repeat > 0 else ""
        print(f"  id={cat_id:2d}  {cnt:3d}개  {id2name[cat_id]}  {mark}")

    # 출력 디렉토리 준비
    os.makedirs(AUG_IMG_DIR, exist_ok=True)

    # 기존 이미지 복사
    print("\n기존 이미지 복사 중...")
    for img_info in coco["images"]:
        src = os.path.join(TRAIN_IMG_DIR, img_info["file_name"])
        dst = os.path.join(AUG_IMG_DIR, img_info["file_name"])
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # 증강 JSON 초기화
    new_images      = list(coco["images"])
    new_annotations = list(coco["annotations"])
    new_img_id      = max(img["id"] for img in coco["images"]) + 1
    new_ann_id      = max(ann["id"] for ann in coco["annotations"]) + 1

    # Copy-Paste 수행
    print("\nCopy-Paste 증강 시작...")
    paste_count = 0

    # 타겟 이미지 목록 (전체 이미지)
    all_img_ids = [img["id"] for img in coco["images"]]

    for cat_id, cnt in class_cnt.items():
        repeat = REPEAT_SCHEDULE.get(cnt, 0)
        if repeat == 0:
            continue

        src_img_ids = list(set(class2imgs[cat_id]))

        for _ in range(repeat):
            # 소스: 해당 클래스가 있는 이미지
            src_id  = random.choice(src_img_ids)
            src_info = id2img[src_id]
            src_path = os.path.join(TRAIN_IMG_DIR, src_info["file_name"])
            if not os.path.exists(src_path):
                continue

            # 해당 클래스 annotation 선택
            src_anns = [a for a in img2anns[src_id] if a["category_id"] == cat_id]
            if not src_anns:
                continue
            src_ann = random.choice(src_anns)
            x, y, w, h = [int(v) for v in src_ann["bbox"]]

            src_img = Image.open(src_path).convert("RGB")
            crop = src_img.crop((x, y, x + w, y + h))

            # 랜덤 스케일 조정
            scale = random.uniform(*SCALE_RANGE)
            pw, ph = max(1, int(w * scale)), max(1, int(h * scale))
            crop   = crop.resize((pw, ph), Image.BILINEAR)


            # 타겟: 다른 이미지
            tgt_id = random.choice(all_img_ids)
            tgt_info = id2img[tgt_id]
            tgt_path = os.path.join(TRAIN_IMG_DIR, tgt_info["file_name"])
            if not os.path.exists(tgt_path):
                continue

            tgt_img   = Image.open(tgt_path).convert("RGB").copy()
            tgt_W, tgt_H = tgt_img.size
            existing  = [a["bbox"] for a in img2anns[tgt_id]]

            pos = find_paste_position(tgt_W, tgt_H, pw, ph, existing)
            if pos is None:
                continue
            px, py = pos

            # 직사각형 붙여넣기
            tgt_img.paste(crop, (px, py))

            # 새 파일 저장
            new_fname = f"aug_{new_img_id}_{cat_id}.png"
            tgt_img.save(os.path.join(AUG_IMG_DIR, new_fname))

            # 새 이미지 메타데이터
            new_images.append({
                "id": new_img_id,
                "file_name": new_fname,
                "width": tgt_W,
                "height": tgt_H,
            })

            # 기존 타겟 annotation 복사 (새 image_id로)
            for ann in img2anns[tgt_id]:
                new_annotations.append({
                    **ann,
                    "id": new_ann_id,
                    "image_id": new_img_id,
                })
                new_ann_id += 1

            # 붙여넣기한 객체 annotation 추가
            new_annotations.append({
                "id": new_ann_id,
                "image_id": new_img_id,
                "category_id": cat_id,
                "bbox": [px, py, pw, ph],
                "area": pw * ph,
                "iscrowd": 0,
                "ignore": 0,
                "segmentation": [],
            })
            new_ann_id += 1

            # 현재 이미지의 annotation 캐시 업데이트
            img2anns[new_img_id] = [
                a for a in new_annotations if a["image_id"] == new_img_id
            ]

            new_img_id  += 1
            paste_count += 1

    # 새 COCO JSON 저장
    aug_coco = {
        "images":      new_images,
        "categories":  coco["categories"],
        "annotations": new_annotations,
    }
    with open(AUG_JSON, "w", encoding="utf-8") as f:
        json.dump(aug_coco, f, ensure_ascii=True)

    print(f"\n완료!")
    print(f"  원본 이미지 수:    {len(coco['images'])}장")
    print(f"  증강 후 이미지 수: {len(new_images)}장 (+{len(new_images)-len(coco['images'])}장)")
    print(f"  붙여넣기 횟수:     {paste_count}회")
    print(f"  저장 위치: {AUG_JSON}")
    print(f"  이미지 위치: {AUG_IMG_DIR}")
    print("\n학습 시 config에서 아래 경로로 변경해주세요:")
    print("  img_folder: ./dataset/images/train_aug/")
    print("  ann_file:   ./dataset/annotations/instances_train_aug.json")


if __name__ == "__main__":
    main()
