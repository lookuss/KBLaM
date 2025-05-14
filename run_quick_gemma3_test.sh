#!/bin/bash

# 기존 합성 데이터셋을 사용하여 Gemma3 모델 테스트

# 1. 데이터 확인
echo "데이터셋 확인 중..."
ls -la datasets/synthetic_data/

# 2. Gemma3 모델 학습 - 기존 합성 데이터로 적은 스텝 수로 테스트
echo "Gemma3 모델 학습 시작..."
python experiments/train.py \
    --dataset synthetic_data \
    --N 1000 \
    --B 8 \
    --total_steps 50 \
    --encoder_spec OAI \
    --use_cached_embd \
    --key_embd_src key \
    --use_data_aug \
    --llm_type gemma3 \
    --hf_model_spec google/gemma-3-4b-it\
    --kb_size 100 \
    --kb_token_layer_frequency 3 \
    --model_save_dir output/quick_gemma3_test \
    --gradient_accm_step 4 \
    --max_seq_len 512 \
    --lr 5e-5

# 3. Gemma3 모델 간단 평가
echo "Gemma3 모델 평가 시작..."
python experiments/eval.py generation \
    --encoder_spec OAI \
    --save_dir output/quick_gemma3_test \
    --exp_config_name quick_gemma3_results \
    --dataset_dir datasets/synthetic_data \
    --test_dataset synthetic_data.json \
    --kb_size 50 \
    --model_dir output/quick_gemma3_test/stage1_lr_5e-05KeyFromkey_OAI_synthetic_data_gemma3_step_50 \
    --encoder_dir output/quick_gemma3_test/stage1_lr_5e-05KeyFromkey_OAI_synthetic_data_gemma3_step_50_encoder/encoder.pt \
    --llm_type gemma3 \
    --llm_base_dir google/gemma-3-4b-it\
    --eval_mode kb \
    --topk_size 20 \
    --kb_layer_frequency 3 \
    --kb_scale_factor 1.0 \
    --precomputed_embed_keys_path datasets/synthetic_data/synthetic_data_oai_embd_key.npy \
    --precomputed_embed_values_path datasets/synthetic_data/synthetic_data_oai_embd_value.npy \
    --seed 42 