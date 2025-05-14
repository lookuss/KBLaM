#!/bin/bash

# Gemma3 모델을 한국어 데이터로 평가하는 스크립트
python experiments/eval.py generation \
    --encoder_spec jhgan/ko-sroberta-multitask \
    --save_dir output/korean_gemma3_test \
    --exp_config_name korean_gemma3_results \
    --dataset_dir datasets/synthetic_data \
    --test_dataset korean_data.json \
    --kb_size 100 \
    --model_dir output/korean_gemma3_test/stage1_lr_5e-05KeyFromkey_jhgan/ko-sroberta-multitask_korean_data_gemma3_step_600 \
    --encoder_dir output/korean_gemma3_test/stage1_lr_5e-05KeyFromkey_jhgan/ko-sroberta-multitask_korean_data_gemma3_step_600_encoder/encoder.pt \
    --llm_type gemma3 \
    --llm_base_dir google/gemma-3-4b-it\
    --eval_mode kb \
    --topk_size 50 \
    --kb_layer_frequency 3 \
    --kb_scale_factor 1.0 \
    --precomputed_embed_keys_path datasets/synthetic_data/korean_data_ko-sroberta-multitask_embd_key.npy \
    --precomputed_embed_values_path datasets/synthetic_data/korean_data_ko-sroberta-multitask_embd_value.npy \
    --seed 42 