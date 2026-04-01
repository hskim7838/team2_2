import cv2
import os
import numpy as np
import random
import shutil
import unicodedata
from tqdm import tqdm

TEAM_CLASS_ORDER = [
    "보령부스파정 5mg", "가바토파정 100mg", "스토가정 10mg", "레일라정", "신바로정",
    "라비에트정 20mg", "울트라셋이알서방정", "놀텍정 10mg", "비모보정 500/20mg", "동아가바펜틴정 800mg",
    "에스원엠프정 20mg", "뮤테란캡슐 100mg", "알드린정", "다보타민큐정 10mg/병", "써스펜8시간이알서방정 650mg",
    "리렉스펜정 300mg/PTP", "큐시드정 31.5mg/PTP", "트루비타정 60mg/병", "맥시부펜이알정 300mg", "일양하이트린정 2mg",
    "뉴로메드정(옥시라세탐)", "리피토정 20mg", "크레스토정 20mg", "오마코연질캡슐(오메가-3-산에틸에스테르90)", "플라빅스정 75mg",
    "아토르바정 10mg", "리피로우정 20mg", "리바로정 4mg", "아토젯정 10/40mg", "로수젯정10/5밀리그램",
    "로수바미브정 10/20mg", "에빅사정(메만틴염산염)(비매품)", "종근당글리아티린연질캡슐(콜린알포세레이트)", "콜리네이트연질캡슐 400mg", "마도파정",
    "아질렉트정(라사길린메실산염)", "글리아타민연질캡슐", "글리틴정(콜린알포세레이트)", "카발린캡슐 25mg", "리리카캡슐 150mg",
    "기넥신에프정(은행엽엑스)(수출용)", "엑스포지정 5/160mg", "트라젠타듀오정 2.5/850mg", "제미메트서방정 50/1000mg", "자누비아정 50mg",
    "아모잘탄정 5/100mg", "노바스크정 5mg", "트라젠타정(리나글립틴)", "트윈스타정 40/5mg", "자누메트정 50/850mg",
    "자누메트엑스알서방정 100/1000mg", "카나브정 60mg", "무코스타정(레바미피드)(비매품)", "에어탈정(아세클로페낙)", "아빌리파이정 10mg",
    "졸로푸트정 100mg"
]

#유니코드 정규화 및 공백 정리
def clean_text(text):
    if not text: return ""
    text = unicodedata.normalize('NFC', text)
    return " ".join(text.split()).strip().replace("/", "_")

name_to_id = {clean_text(name): i for i, name in enumerate(TEAM_CLASS_ORDER)}

def imread_korean(path, flags=cv2.IMREAD_COLOR):
    try:
        nparr = np.fromfile(path, np.uint8)
        img = cv2.imdecode(nparr, flags)
        return img
    except Exception as e:
        print(f"읽기 실패: {path} | {e}")
        return None

def make_dir(path, rebuild=False):
    if rebuild and os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

#클래스별로 분류 후 라이브러리 생성
def build_pills_library(library_root):
    library = {}
    if not os.path.exists(library_root): return library
    
    pill_folders = [f for f in os.listdir(library_root) if os.path.isdir(os.path.join(library_root, f))]
    
    for p_folder in pill_folders:
        cleaned_folder_name = clean_text(p_folder)
        
        if cleaned_folder_name in name_to_id:
            cls_path = os.path.join(library_root, p_folder)
            pill_files = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.lower().endswith('.png')]
            
            if pill_files:
                library[cleaned_folder_name] = pill_files
        else:
            print(f"매칭 실패: '{p_folder}'는 리스트에 없는 이름입니다.")
            
    return library

#배경 이미지에 알약 합성
def add_pill_to_bg_safe(bg, pill, x_start, y_start):
    bg_h, bg_w = bg.shape[:2]
    pill_h, pill_w = pill.shape[:2]
    
    if x_start + pill_w > bg_w or y_start + pill_h > bg_h or x_start < 0 or y_start < 0:
        return bg, False
        
    overlay_img = pill[:, :, :3]
    overlay_mask = pill[:, :, 3:] / 255.0
    
    y1, y2 = y_start, y_start + pill_h
    x1, x2 = x_start, x_start + pill_w
    
    bg[y1:y2, x1:x2] = (1.0 - overlay_mask) * bg[y1:y2, x1:x2] + overlay_mask * overlay_img
    return bg, True

#알약 무작위 배치, txt 파일 생성
def run_synthesis_no_overlap(bg_dir, lib_dir, save_dir, total_count=50):
    # 이미지는 train/images에, 라벨은 train/labels에 저장되도록 경로 분리
    img_save_path = os.path.join(save_dir, 'images')
    lbl_save_path = os.path.join(save_dir, 'labels')
    
    # 기존 데이터가 있을 수 있으므로 rebuild=False로 설정하여 누적 저장 가능하게 함
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(lbl_save_path, exist_ok=True)
    
    # 배경 이미지 경로들만 수집
    bg_files = [os.path.join(bg_dir, f) for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.png'))]
    pills_lib = build_pills_library(lib_dir)
    
    if not bg_files or not pills_lib:
        print("배경 이미지 또는 알약 라이브러리를 확인해주세요.")
        return

    available_pill_names = list(pills_lib.keys())

    for i in tqdm(range(total_count), desc="합성 데이터 생성 중"):
        bg_p = random.choice(bg_files)
        bg = imread_korean(bg_p)
        if bg is None: continue
        bg = bg.copy()
        bg_h, bg_w = bg.shape[:2]
        
        sections = [[0, 0, bg_w//2, bg_h//2], [bg_w//2, 0, bg_w, bg_h//2], 
                    [0, bg_h//2, bg_w//2, bg_h], [bg_w//2, bg_h//2, bg_w, bg_h]]
        
        num_pills = random.randint(3, 4)
        random.shuffle(sections)
        labels = []

        for s in range(num_pills):
            sx1, sy1, sx2, sy2 = sections[s]
            p_name = random.choice(available_pill_names)
            pill_path = random.choice(pills_lib[p_name])
            yolo_cls_id = name_to_id[p_name]
            
            # 알약 로드 (한글 경로 + 투명도 대응)
            pill = imread_korean(pill_path, cv2.IMREAD_UNCHANGED)
            if pill is None: continue
            
            ph, pw = pill.shape[:2]
            margin = 40
            max_x = sx2 - pw - margin
            max_y = sy2 - ph - margin
            
            x = random.randint(sx1 + margin, max(sx1 + margin, max_x))
            y = random.randint(sy1 + margin, max(sy1 + margin, max_y))
            
            bg, success = add_pill_to_bg_safe(bg, pill, x, y)
            
            if success:
                labels.append(f"{yolo_cls_id} {(x+pw/2)/bg_w:.6f} {(y+ph/2)/bg_h:.6f} {pw/bg_w:.6f} {ph/bg_h:.6f}")

        # 파일명 중복 방지를 위해 timestamp나 고유 ID 사용
        fn = f"syn_{i:04d}"
        save_img_path = os.path.join(img_save_path, f"{fn}.jpg")
        
        result, nparr = cv2.imencode('.jpg', bg)
        if result:
            with open(save_img_path, mode='wb') as f:
                nparr.tofile(f)
        
        with open(os.path.join(lbl_save_path, f"{fn}.txt"), 'w') as f:
            f.write("\n".join(labels))

    print(f"총 {total_count}장의 데이터가 {save_dir}에 준비되었습니다.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.abspath(os.path.join(current_dir, ".."))
    bg_images_dir = os.path.join(project_root_dir, "backgrounds_images") 
    pill_lib_dir = os.path.join(project_root_dir, "pill_crops_transparent")
    final_train_dir = os.path.join(project_root_dir, "yolo_dataset", "train")

    run_synthesis_no_overlap(
        bg_dir=bg_images_dir,
        lib_dir=pill_lib_dir,
        save_dir=final_train_dir,
        total_count=100
    )