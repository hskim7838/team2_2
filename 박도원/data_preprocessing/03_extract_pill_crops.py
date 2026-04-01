import cv2
import os
import json
import numpy as np
import unicodedata
from collections import Counter
from rembg import remove

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
    return " ".join(text.split()).strip()

def find_files(root_path, extension):
    found_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(extension):
                found_files.append(os.path.join(root, file))
    return found_files

#소수 클래스 선별 함수
def get_classes_json(label_files, threshold=10):
    all_class_ids = []
    for json_path in label_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for obj in data.get('annotations', []):
                cid = obj.get('category_id')
                if cid is not None:
                    all_class_ids.append(cid)
    counts = Counter(all_class_ids)
    # 모든 알약을 뽑고 싶다면 threshold를 1000 이상으로 설정하세요.
    minority_classes = [cls for cls, count in counts.items() if count <= threshold]
    print(f"클래스 추출 완료: {len(minority_classes)}개")
    return minority_classes

# 알약 crop, rembg 사용
def run_recursive_crop_with_rembg(ann_dir, img_dir, save_dir, threshold=10):
    all_json_files = find_files(ann_dir, '.json')
    if not all_json_files:
        print(f"JSON 파일을 찾지 못했습니다.")
        return

    target_classes = get_classes_json(all_json_files, threshold)

    for json_path in all_json_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
            
        id_to_name = {cat['id']: clean_text(cat['name']) for cat in label_data.get('categories', [])}
        file_name_only = os.path.splitext(os.path.basename(json_path))[0]
        
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            temp_path = os.path.join(img_dir, file_name_only + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
        
        if not img_path: continue
        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape
        
        for i, obj in enumerate(label_data.get('annotations', [])):
            cls_id = obj.get('category_id')
            
            if cls_id in target_classes:
                pill_name = id_to_name.get(cls_id, "unknown")
                
                # 특수문자(/) 및 경로 방해 문자 제거
                safe_pill_name = pill_name.replace("/", "_").replace("\\", "_").replace(":", "_")
                
                class_path = os.path.join(save_dir, safe_pill_name)
                os.makedirs(class_path, exist_ok=True)
                
                # 크롭
                bx = obj.get('bbox') 
                if not bx: continue
                x, y, bw, bh = map(int, bx)
                crop_img = img[max(0, y):min(h, y+bh), max(0, x):min(w, x+bw)]
                if crop_img.size == 0: continue
                
                # 배경 제거
                try:
                    transparent_crop = remove(crop_img)
                except:
                    transparent_crop = crop_img 

                # [수정 2] 한글 경로 저장을 위한 numpy 우회 방식
                save_file_path = os.path.join(class_path, f"c{cls_id}_{file_name_only}_{i}.png")
                
                try:
                    # 메모리에서 PNG로 인코딩
                    result, nparr = cv2.imencode('.png', transparent_crop)
                    if result:
                        # 한글 경로를 포함한 파일 쓰기
                        with open(save_file_path, mode='wb') as f:
                            nparr.tofile(f)
                    else:
                        print(f"인코딩 실패: {pill_name}")
                except Exception as e:
                    print(f"저장 실패 ({pill_name}): {e}")

if __name__ == "__main__":
    # --- 경로 설정 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.abspath(os.path.join(current_dir, ".."))

    base_data_dir = os.path.join(
        project_root_dir, 
        "ai09-level1-project", 
        "sprint_ai_project1_data"
    )

    train_ann_dir = os.path.join(base_data_dir, "train_annotations")
    train_img_dir = os.path.join(base_data_dir, "train_images")

    save_crops_dir = os.path.join(project_root_dir, "pill_crops_transparent")

    run_recursive_crop_with_rembg(train_ann_dir, train_img_dir, save_crops_dir, threshold=10)