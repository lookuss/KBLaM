#!/bin/bash

# 최소한의 설정으로 Gemma3 모델이 작동하는지 확인하는 스크립트

# 작은 합성 데이터셋 생성 (10개 엔티티만 생성)
echo "작은 테스트 데이터셋 생성 중..."
python dataset_generation/gen_synthetic_data.py --num_entities 10 --output_name tiny_test

# 임베딩 생성
echo "임베딩 생성 중..."
python dataset_generation/generate_kb_embeddings.py \
    --data_path datasets/synthetic_data/tiny_test.json \
    --embedding_model all-MiniLM-L6-v2 \
    --output_name tiny_test

# Gemma3 모델 매우 짧은 학습 (5 스텝만 실행)
echo "Gemma3 모델 최소 학습 시작..."
python experiments/train.py \
    --dataset tiny_test \
    --dataset_dir datasets/synthetic_data \
    --N 10 \
    --B 2 \
    --total_steps 5 \
    --encoder_spec all-MiniLM-L6-v2 \
    --key_embd_src key \
    --use_cached_embd \
    --llm_type gemma3 \
    --hf_model_spec google/gemma-3-4b-it\
    --kb_size 5 \
    --kb_token_layer_frequency 3 \
    --model_save_dir output/minimal_gemma3_test \
    --gradient_accm_step 1 \
    --max_seq_len 256 \
    --lr 5e-5

# 간단한 추론 테스트
echo "Gemma3 모델 작동 여부 확인 중..."
python experiments/eval.py generation \
    --encoder_spec all-MiniLM-L6-v2 \
    --save_dir output/minimal_gemma3_test \
    --exp_config_name minimal_gemma3_results \
    --dataset_dir datasets/synthetic_data \
    --test_dataset tiny_test.json \
    --kb_size 5 \
    --model_dir output/minimal_gemma3_test/stage1_lr_5e-05KeyFromkey_all-MiniLM-L6-v2_tiny_test_gemma3_step_5 \
    --encoder_dir output/minimal_gemma3_test/stage1_lr_5e-05KeyFromkey_all-MiniLM-L6-v2_tiny_test_gemma3_step_5_encoder/encoder.pt \
    --llm_type gemma3 \
    --llm_base_dir google/gemma-3-4b-it\
    --eval_mode kb \
    --topk_size 5 \
    --kb_layer_frequency 3 \
    --kb_scale_factor 1.0 \
    --precomputed_embed_keys_path datasets/synthetic_data/tiny_test_all-MiniLM-L6-v2_embd_key.npy \
    --precomputed_embed_values_path datasets/synthetic_data/tiny_test_all-MiniLM-L6-v2_embd_value.npy \
    --seed 42 