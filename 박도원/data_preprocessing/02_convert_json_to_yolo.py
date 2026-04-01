import os
import json
import shutil
import yaml
import glob
import unicodedata
from tqdm import tqdm

# 경로설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_dir, ".."))

base_data_dir = os.path.join(
    project_root_dir, 
    "ai09-level1-project", 
    "sprint_ai_project1_data"
)

train_img_dir = os.path.join(base_data_dir, "train_images")
train_ann_dir = os.path.join(base_data_dir, "train_annotations")
test_img_dir = os.path.join(base_data_dir, "test_images")

output_dataset_dir = os.path.join(project_root_dir, "yolo_dataset")

classes = [
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

def clean_text(text):
    if not text: return ""
    text = unicodedata.normalize('NFC', text)
    return " ".join(text.split()).strip()

name_to_id = {clean_text(name): i for i, name in enumerate(TEAM_CLASS_ORDER)}

def save_to_yolo(pairs, split_type):
    img_dir = os.path.join(output_dataset_dir, split_type, "images")
    lbl_dir = os.path.join(output_dataset_dir, split_type, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for img_path, json_list, _ in tqdm(pairs, desc=f"{split_type} 저장 중"):
        parent_name = os.path.basename(os.path.dirname(img_path))
        unique_name = f"{parent_name}_{os.path.basename(img_path)}"
        shutil.copy(img_path, os.path.join(img_dir, unique_name))
            
        txt_path = os.path.join(lbl_dir, os.path.splitext(unique_name)[0] + ".txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            for j_p in json_list:
                with open(j_p, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                    img_w, img_h = data['images'][0]['width'], data['images'][0]['height']
                    for ann in data['annotations']:
                        raw_name = next(c['name'] for c in data['categories'] if c['id'] == ann['category_id'])
                        c_name = clean_text(raw_name)
                        if c_name in name_to_id:
                            cls_id = name_to_id[c_name]
                            x, y, w, h = ann['bbox']
                            f.write(f"{cls_id} {(x+w/2)/img_w:.6f} {(y+h/2)/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}\n")

def copy_test_images():
    test_out_dir = os.path.join(output_dataset_dir, "test", "images")
    os.makedirs(test_out_dir, exist_ok=True)
    
    test_files = glob.glob(os.path.join(test_img_dir, "**", "*.png"), recursive=True)
    if not test_files:
        print("Test 이미지가 해당 경로에 없습니다.")
        return

    for img_p in tqdm(test_files, desc="Test 이미지 복사 중"):
        parent_name = os.path.basename(os.path.dirname(img_p))
        unique_name = f"{parent_name}_{os.path.basename(img_p)}"
        shutil.copy(img_p, os.path.join(test_out_dir, unique_name))

def build_dataset_recursive():
    if os.path.exists(output_dataset_dir):
        shutil.rmtree(output_dataset_dir)
    
    img_files = glob.glob(os.path.join(train_img_dir, "**", "*.png"), recursive=True)
    actual_images = {os.path.basename(f).lower(): f for f in img_files}
    ann_files = glob.glob(os.path.join(train_ann_dir, "**", "*.json"), recursive=True)

    image_to_data = {}
    for json_p in tqdm(ann_files, desc="데이터 분석 중"):
        try:
            with open(json_p, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
                img_name = os.path.basename(data['images'][0]['file_name']).lower()
                if img_name in actual_images:
                    if img_name not in image_to_data:
                        image_to_data[img_name] = [actual_images[img_name], [], set()]
                    image_to_data[img_name][1].append(json_p)
                    for ann in data.get('annotations', []):
                        cat_name = next(c['name'] for c in data['categories'] if c['id'] == ann['category_id'])
                        image_to_data[img_name][2].add(clean_text(cat_name))
        except: continue

    val_images_keys = set()
    covered_in_val = set()
    class_to_images = {cls: [] for cls in name_to_id.keys()}
    for k, v in image_to_data.items():
        for cls_name in v[2]:
            if cls_name in class_to_images:
                class_to_images[cls_name].append(k)

    for cls in name_to_id.keys():
        if cls in covered_in_val: continue
        candidates = class_to_images[cls]
        if candidates:
            chosen_key = candidates[0]
            val_images_keys.add(chosen_key)
            covered_in_val.update(image_to_data[chosen_key][2])

    train_pairs, val_pairs = [], []
    for img_name, d_info in image_to_data.items():
        if img_name in val_images_keys:
            val_pairs.append(d_info)
        else:
            train_pairs.append(d_info)

    save_to_yolo(train_pairs, "train")
    save_to_yolo(val_pairs, "val")
    
    copy_test_images()

    with open(os.path.join(output_dataset_dir, "data.yaml"), 'w', encoding='utf-8') as y:
        yaml.dump({
            'train': '../train/images', 
            'val': '../val/images',
            'test': '../test/images',
            'nc': len(classes), 
            'names': classes
        }, y, allow_unicode=True)

    print(f"완료")

if __name__ == "__main__":
    build_dataset_recursive()