#!/bin/bash

# Gemma3 모델을 KBLaM으로 훈련하는 스크립트
python experiments/train.py \
    --dataset synthetic_data \
    --N 120000 \
    --B 20 \
    --total_steps 601 \
    --encoder_spec OAI \
    --use_oai_embd \
    --key_embd_src key \
    --use_data_aug \
    --llm_type gemma3 \
    --hf_model_spec google/gemma-3-4b-it \
    --kb_size 250 \
    --kb_token_layer_frequency 3 \
    --model_save_dir output/gemma3_test \
    --gradient_accm_step 20 