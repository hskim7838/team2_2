import cv2
import os
import numpy as np
import random  

def draw_yolo_bbox(lbl_dir, img_dir, output_dir, sample_count=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 1. 파일 목록 중 'syn_'으로 시작하는 파일만
    all_lbl_files = [f for f in os.listdir(lbl_dir) if f.lower().endswith('.txt')]
    syn_lbl_files = [f for f in all_lbl_files if f.startswith('syn_')]

    if not syn_lbl_files:
        print(f"'{lbl_dir}' 폴더에 'syn_'으로 시작하는 합성 데이터 라벨이 없습니다.")
        return

    actual_sample_count = min(len(syn_lbl_files), sample_count)
    lbl_samples = random.sample(syn_lbl_files, actual_sample_count)

    print(f"총 {len(syn_lbl_files)}개의 합성 데이터 중 {actual_sample_count}개를 검수합니다.")

    for lbl_name in lbl_samples:
        lbl_path = os.path.join(lbl_dir, lbl_name)
        file_name_only = os.path.splitext(lbl_name)[0]

        # 이미지 읽기
        img_path = os.path.join(img_dir, file_name_only + ".jpg")
        
        if not os.path.exists(img_path):
            print(f"이미지를 찾을 수 없음: {img_path}")
            continue

        # 한글 경로 읽기용 numpy 우회
        nparr = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: continue
        h, w, _ = img.shape

        # YOLO TXT 읽기 및 박스 그리기
        with open(lbl_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5: continue
            
            cls_id = parts[0]
            # YOLO는 [center_x, center_y, width, height]
            cx, cy, nw, nh = map(float, parts[1:])

            # 절대좌표(Pixel)로 변환
            x1 = int((cx - nw/2) * w)
            y1 = int((cy - nh/2) * h)
            x2 = int((cx + nw/2) * w)
            y2 = int((cy + nh/2) * h)

            # 빨간색 박스 그리기
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"ID:{cls_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 결과 저장 
        save_path = os.path.join(output_dir, f"check_{file_name_only}.jpg")
        res, nparr_enc = cv2.imencode('.jpg', img)
        if res:
            with open(save_path, mode='wb') as f:
                nparr_enc.tofile(f)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.abspath(os.path.join(current_dir, ".."))

    base_synthetic_dir = os.path.join(project_root_dir, "yolo_dataset", "train")
    
    synthetic_labels_dir = os.path.join(base_synthetic_dir, "labels")
    synthetic_images_dir = os.path.join(base_synthetic_dir, "images")
    bbox_check_results_dir = os.path.join(project_root_dir, "bbox_check")

    draw_yolo_bbox(synthetic_labels_dir, synthetic_images_dir, bbox_check_results_dir, sample_count=10)