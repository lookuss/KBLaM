#!/bin/bash

# Gemma3 모델을 KBLaM으로 평가하는 스크립트
python experiments/eval.py \
    --encoder_spec OAI \
    --model_save_dir output/gemma3_test \
    --data_dir datasets/synthetic_data \
    --kb_size 250 \
    --model_ckpt_path output/gemma3_test/checkpoints/model_600 \
    --encoder_path output/gemma3_test/checkpoints/encoder.pt \
    --llm_type gemma3 \
    --llm_base_dir google/gemma-3b \
    --eval_mode kb \
    --topk_size 250 \
    --kb_scale_factor 1.0 