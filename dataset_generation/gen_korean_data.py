import argparse
import json
import os
import random
from typing import Dict, List

# 예시 한국어 개체 및 설명 데이터
KOREAN_ENTITIES = [
    {"name": "서울", "description_type": "수도", "description": "대한민국의 수도"},
    {"name": "부산", "description_type": "도시", "description": "대한민국의 제2의 도시이자 항구도시"},
    {"name": "한라산", "description_type": "산", "description": "제주도에 위치한 대한민국에서 가장 높은 산"},
    {"name": "독도", "description_type": "섬", "description": "대한민국 동해에 위치한 섬"},
    {"name": "세종대왕", "description_type": "인물", "description": "훈민정음을 창제한 조선의 제4대 국왕"},
    {"name": "이순신", "description_type": "인물", "description": "임진왜란 당시 조선의 해군 장수"},
    {"name": "백두산", "description_type": "산", "description": "한반도와 만주 사이에 위치한 산"},
    {"name": "김연아", "description_type": "인물", "description": "2010년 밴쿠버 올림픽 피겨스케이팅 금메달리스트"},
    {"name": "광화문", "description_type": "건물", "description": "서울 경복궁의 정문"},
    {"name": "한강", "description_type": "강", "description": "서울을 관통하는 대한민국 최대 강"},
]

# 한국어 질문 템플릿
QUESTION_TEMPLATES = [
    "{name}의 {description_type}은 무엇인가요?",
    "{name}에 대해 설명해주세요.",
    "{name}의 {description_type}에 대한 정보를 알려주세요.",
    "{name}은 어떤 {description_type}인가요?",
    "{name}의 {description_type}에 대해 알고 싶습니다.",
]

# 한국어 답변 템플릿
ANSWER_TEMPLATES = [
    "{name}의 {description_type}은 {description}입니다.",
    "{name}은 {description}입니다.",
    "{name}에 대한 {description_type}은 {description}입니다.",
]

def generate_entities(num_entities: int) -> List[Dict]:
    """
    지정된 수의 한국어 가상 개체 생성
    """
    entities = []
    base_entities = KOREAN_ENTITIES.copy()
    
    # 기본 엔티티 추가
    entities.extend(base_entities)
    
    # 필요한 만큼 추가 엔티티 생성
    for i in range(len(base_entities), num_entities):
        entity_id = i - len(base_entities) + 1
        entity = {
            "name": f"개체_{entity_id}",
            "description_type": random.choice(["특징", "정보", "설명", "유형", "종류"]),
            "description": f"한국어 지식 베이스용 테스트 개체 {entity_id}에 대한 설명입니다."
        }
        entities.append(entity)
    
    return entities[:num_entities]

def generate_qa_pair(entity: Dict) -> Dict:
    """
    엔티티에 대한 질문-답변 쌍 생성
    """
    # 질문 생성
    question_template = random.choice(QUESTION_TEMPLATES)
    question = question_template.format(
        name=entity["name"], 
        description_type=entity["description_type"]
    )
    
    # 답변 생성
    answer_template = random.choice(ANSWER_TEMPLATES)
    answer = answer_template.format(
        name=entity["name"], 
        description_type=entity["description_type"],
        description=entity["description"]
    )
    
    # 키 문자열 생성 (KBLaM 형식)
    key_string = f"{entity['name']}의 {entity['description_type']}"
    
    return {
        "name": entity["name"],
        "description_type": entity["description_type"],
        "description": entity["description"],
        "Q": question,
        "A": answer,
        "key_string": key_string
    }

def main():
    parser = argparse.ArgumentParser(description="한국어 지식 베이스 및 질문-답변 쌍 생성")
    parser.add_argument("--num_entities", type=int, default=1000, help="생성할 엔티티 수")
    parser.add_argument("--output_dir", type=str, default="datasets/synthetic_data", help="출력 디렉토리")
    parser.add_argument("--output_name", type=str, default="korean_data", help="출력 파일 이름")
    args = parser.parse_args()
    
    # 엔티티 생성
    entities = generate_entities(args.num_entities)
    
    # 각 엔티티에 대한 질문-답변 쌍 생성
    data = [generate_qa_pair(entity) for entity in entities]
    
    # 출력 디렉토리 확인 및 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # JSON 파일로 저장
    output_path = os.path.join(args.output_dir, f"{args.output_name}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"{len(data)}개의 엔티티로 구성된 한국어 데이터셋이 {output_path}에 저장되었습니다.")

if __name__ == "__main__":
    main() 