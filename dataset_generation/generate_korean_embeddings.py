import argparse
import json
import os
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser(description="한국어 데이터셋을 위한 임베딩 생성")
    parser.add_argument("--data_path", type=str, default="datasets/synthetic_data/korean_data.json", help="데이터셋 경로")
    parser.add_argument("--output_dir", type=str, default="datasets/synthetic_data", help="출력 디렉토리")
    parser.add_argument("--model_name", type=str, default="jhgan/ko-sroberta-multitask", help="한국어 문장 임베딩 모델")
    args = parser.parse_args()
    
    # 데이터 로드
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 임베딩 모델 로드
    print(f"한국어 임베딩 모델 {args.model_name} 로드 중...")
    model = SentenceTransformer(args.model_name)
    print("모델 로드 완료!")
    
    # 키와 값 텍스트 추출
    key_texts = [item['key_string'] for item in data]
    value_texts = [item['description'] for item in data]
    answer_texts = [item['A'] for item in data]
    
    # 임베딩 생성
    print("키 텍스트 임베딩 생성 중...")
    key_embeddings = model.encode(key_texts, show_progress_bar=True)
    
    print("값 텍스트 임베딩 생성 중...")
    value_embeddings = model.encode(value_texts, show_progress_bar=True)
    
    print("답변 텍스트 임베딩 생성 중...")
    answer_embeddings = model.encode(answer_texts, show_progress_bar=True)
    
    # 출력 파일 경로
    output_prefix = os.path.splitext(os.path.basename(args.data_path))[0]
    model_name_short = args.model_name.split('/')[-1]
    
    # 임베딩 저장
    key_output_path = os.path.join(args.output_dir, f"{output_prefix}_{model_name_short}_embd_key.npy")
    value_output_path = os.path.join(args.output_dir, f"{output_prefix}_{model_name_short}_embd_value.npy")
    answer_output_path = os.path.join(args.output_dir, f"{output_prefix}_{model_name_short}_embd_answer.npy")
    
    np.save(key_output_path, key_embeddings)
    np.save(value_output_path, value_embeddings)
    np.save(answer_output_path, answer_embeddings)
    
    print(f"임베딩이 다음 파일들에 저장되었습니다:")
    print(f"- 키 임베딩: {key_output_path}")
    print(f"- 값 임베딩: {value_output_path}")
    print(f"- 답변 임베딩: {answer_output_path}")

if __name__ == "__main__":
    main() 