#!/bin/bash

# 1. 한국어 데이터셋 생성
echo "한국어 데이터셋 생성 중..."
python dataset_generation/gen_korean_data.py --num_entities 1000 --output_name korean_data

# 2. 한국어 임베딩 생성
echo "한국어 임베딩 생성 중..."
python dataset_generation/generate_korean_embeddings.py --data_path datasets/synthetic_data/korean_data.json

# 3. Gemma3 모델을 한국어 데이터로 학습
echo "Gemma3 모델 학습 시작..."
python experiments/train.py \
    --dataset korean_data \
    --dataset_dir datasets/synthetic_data \
    --N 1000 \
    --B 16 \
    --total_steps 601 \
    --encoder_spec jhgan/ko-sroberta-multitask \
    --key_embd_src key \
    --use_cached_embd \
    --use_data_aug \
    --llm_type gemma3 \
    --hf_model_spec google/gemma-3b \
    --kb_size 200 \
    --kb_token_layer_frequency 3 \
    --model_save_dir output/korean_gemma3_test \
    --gradient_accm_step 10 \
    --lr 5e-5 