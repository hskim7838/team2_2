import os
import json
from collections import Counter
import platform
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 데이터 개수와 무결성 검사 함수
def check_data_integrity(img_path, ann_path, test_img_path):
    img_files = [f for f in os.listdir(img_path) if f.lower().endswith('.png')]
    ann_files = []
    test_files = [f for f in os.listdir(test_img_path) if f.lower().endswith('.png')]

    for _, _, files in os.walk(ann_path):
        for f in files:
            if f.lower().endswith('.json'):
                ann_files.append(f)

    # 데이터 개수 확인
    print(f"이미지 총{len(img_files)}개")
    print(f"라벨 총{len(ann_files)}개")
    print(f"테스트 이미지 총{len(test_files)}개")

    # 무결성 검사
    img_names = set([os.path.splitext(f)[0] for f in img_files])
    ann_names = set([os.path.splitext(f)[0] for f in ann_files])
    
    only_img = img_names - ann_names
    only_ann = ann_names - img_names

    if not only_img and not only_ann:
        print("무결성 검사 완료")
    else:
        if only_img:
            print(f"라벨 없는 이미지({len(only_img)}): {list(only_img)[:3]}")
        if only_ann:
            print(f"이미지 없는 라벨({len(only_ann)}): {list(only_ann)[:3]}")


#JSON 파싱 함수
def load_json(ann_path):
    all_data = []
    for root, _, files in os.walk(ann_path):
        for f in files:
            if f.lower().endswith('.json'):
                full_path = os.path.join(root, f)
                try:
                    with open(full_path, 'r', encoding='utf-8') as jf:
                        all_data.append(json.load(jf))
                except:
                    print("파일 읽기 오류")
    return all_data


#클래스 분포도 확인 함수
def show_class(parsed_data):
    all_category_names = []
    for data in parsed_data:
        id_to_name = {cat['id']: cat['name'] for cat in data.get('categories', [])}
        for ann in data.get('annotations', []):
            cat_id = ann.get('category_id')
            all_category_names.append(id_to_name.get(cat_id, f"Unknown{cat_id}"))


    counts = Counter(all_category_names)

    if platform.system() == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    plt.rc('axes', unicode_minus=False)

    # 데이터 준비
    common_counts = counts.most_common(30)
    names = [item[0] for item in common_counts]
    values = [item[1] for item in common_counts]

    plt.figure(figsize=(12, 10))
    bars = plt.barh(names, values, color='skyblue', edgecolor='black')
    
    # 막대 끝에 숫자 표시
    for bar in bars:
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                 f'{int(bar.get_width())}개', va='center')

    plt.xlabel('데이터 개수')
    plt.ylabel('알약 이름')
    plt.title('상위 30개 알약 데이터 분포 (EDA)')
    plt.gca().invert_yaxis() 
    plt.tight_layout()
    plt.show()

    # 사진 저장
    # plt.savefig('pill_distribution.png')

# 해상도 크기 확인 함수
def check_resolutions(parsed_data):
    resolutions = []
    for data in parsed_data:
        for img_info in data.get('images', []):
            resolutions.append((img_info['width'], img_info['height']))

    res_counts = Counter(resolutions)

    for res, count in res_counts.items():
        res_str = f"{res[0]} x {res[1]}"
        print(f"해상도({res_str}) {count}개")

#바운딩 박스 확인 함수
def visual_sanity_check(img_path, parsed_data, num_sample=10):
    count = 0
    for data in parsed_data:
        if count >= num_sample: break

        for img_info in data.get('images', []):
            img_name = img_info['file_name']
            actual_img_path = os.path.join(img_path, img_name)

            if os.path.exists(actual_img_path):
                img = cv2.imread(actual_img_path)
                img_id = img_info['id']

                # BBox 그리기
                for ann in data.get('annotations', []):
                    if ann['image_id'] == img_id:
                        x, y, w, h = map(int, ann['bbox'])
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

                save_name = f"check_{count}_{img_name}"
                cv2.imwrite(save_name, img)
                print(f"BBox 샘플 저장완료: {save_name}")
                count += 1
                if count >= num_sample: break


if __name__ == "__main__":
    path = "./data"
    train_img_path = os.path.join(path, "train_images") # train 이미지 경로
    train_ann_path = os.path.join(path, "train_annotations") # annotations 경로
    test_img_path = os.path.join(path, "test_images")

    check_data_integrity(train_img_path, train_ann_path, test_img_path)

    parsed_dataset = load_json(train_ann_path)

    show_class(parsed_dataset)
    check_resolutions(parsed_dataset)
    visual_sanity_check(train_img_path, parsed_dataset)