import json
import glob
import os

my_yaml_names = [
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

# JSON 루트
base_path = "./data"
root_path = os.path.join(base_path, "test_images")
json_files = glob.glob(os.path.join(root_path, "**", "*.json"), recursive=True)

# 약 이름 : ID 매칭 정보 수집
full_id_map = {}
for file_path in json_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'categories' in data:
                for cat in data['categories']:
                    # 이름에서 공백을 제거하여 매칭 확률을 높입니다.
                    name = cat['name'].strip()
                    full_id_map[name] = cat['id']
    except Exception as e:
        continue

# 최종 ID 리스트 생성
true_class_ids = []
missing_names = []

for name in my_yaml_names:
    target_name = name.strip()
    if target_name in full_id_map:
        true_class_ids.append(full_id_map[target_name])
    else:
        true_class_ids.append(None)
        missing_names.append(target_name)

print("--- 추출 결과 ---")
print(f"찾은 약 개수: {len(true_class_ids) - len(missing_names)} / 56")
if missing_names:
    print(f"못 찾은 이름: {missing_names}")
print("\n최종 true_class_ids 리스트 (이걸 복사하세요):")
print(true_class_ids)