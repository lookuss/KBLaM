#!/bin/bash

# 수정된 Gemma 모델 테스트 스크립트

# 1. 필요한 디렉토리 생성
echo "필요한 디렉토리 생성 중..."
mkdir -p datasets/synthetic_data

# 2. 작은 합성 데이터셋 생성
echo "작은 테스트 데이터셋 생성 중..."
python dataset_generation/gen_synthetic_data.py --num_entities 20 --output_name mini_test

# 3. 임베딩 생성 (all-MiniLM-L6-v2 사용)
echo "임베딩 생성 중..."
python dataset_generation/generate_kb_embeddings.py \
    --data_path datasets/synthetic_data/mini_test.json \
    --embedding_model all-MiniLM-L6-v2 \
    --output_name mini_test

# 4. Gemma 모델 학습 (매우 짧은 단계로)
echo "Gemma 모델 학습 시작..."
python experiments/train.py \
    --dataset mini_test \
    --dataset_dir datasets/synthetic_data \
    --N 20 \
    --B 2 \
    --total_steps 10 \
    --encoder_spec all-MiniLM-L6-v2 \
    --key_embd_src key \
    --use_cached_embd \
    --llm_type gemma3 \
    --hf_model_spec google/gemma-3-4b-it \
    --kb_size 10 \
    --kb_token_layer_frequency 3 \
    --model_save_dir output/fixed_gemma_test \
    --gradient_accm_step 1 \
    --max_seq_len 256 \
    --lr 5e-5

# 5. 간단한 평가
echo "Gemma 모델 평가 시작..."
python experiments/eval.py generation \
    --encoder_spec all-MiniLM-L6-v2 \
    --save_dir output/fixed_gemma_test \
    --exp_config_name fixed_gemma_results \
    --dataset_dir datasets/synthetic_data \
    --test_dataset mini_test.json \
    --kb_size 10 \
    --model_dir output/fixed_gemma_test/stage1_lr_5e-05KeyFromkey_all-MiniLM-L6-v2_mini_test_gemma3_step_10 \
    --encoder_dir output/fixed_gemma_test/stage1_lr_5e-05KeyFromkey_all-MiniLM-L6-v2_mini_test_gemma3_step_10_encoder/encoder.pt \
    --llm_type gemma3 \
    --llm_base_dir google/gemma-3-4b-it \
    --eval_mode kb \
    --topk_size 5 \
    --kb_layer_frequency 3 \
    --kb_scale_factor 1.0 \
    --precomputed_embed_keys_path datasets/synthetic_data/mini_test_all-MiniLM-L6-v2_embd_key.npy \
    --precomputed_embed_values_path datasets/synthetic_data/mini_test_all-MiniLM-L6-v2_embd_value.npy \
    --seed 42 