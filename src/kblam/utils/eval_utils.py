from typing import Optional

import numpy as np
import torch
import transformers

from KBLaM.src.kblam.models.kblam_config import KBLaMConfig
from KBLaM.src.kblam.models.llama3_model import KblamLlamaForCausalLM
from KBLaM.src.kblam.models.phi3_model import KBLaMPhi3ForCausalLM
from KBLaM.src.kblam.models.gemma3_model import KBLaMGemma3ForCausalLM

instruction_prompts = """
Please answer questions based on the given text with format: "The {property} of {name} is {description}"
"""

instruction_prompts_multi_entities = """
Please answer questions based on the given text with format: "The {property} of {name1} is {description}; The {property} of {name2} is {description}; ..."
"""

zero_shot_prompt = """
Please answer the question in a very compact manner with format: The {property} of {name} is {description}
"""

zero_shot_prompt_multi_entities = """
Please answer the question in a very compact manner with format: "The {property} of {name1} is {description}; The {property} of {name2} is {description}; ...
"""


def _prune_for_llama(S: str) -> str:
    S = S.replace("<|eot_id|>", "")
    S = S.replace("<|start_header_id|>assistant<|end_header_id|>", "")
    S = S.replace("<|start_header_id|>user<|end_header_id|>", "")
    S = S.replace("<|end_of_text|>", "")
    return S


def _prune_for_phi3(S: str) -> str:
    S = S.replace("<|end|>", "")
    S = S.replace("<|assistant|>", "")
    S = S.replace("<|user|>", "")
    return S


def _prune_for_gemma3(S: str) -> str:
    S = S.replace("<start_of_turn>model", "")
    S = S.replace("<end_of_turn>", "")
    S = S.replace("<start_of_turn>user", "")
    return S


def softmax(x: np.array, axis: int) -> np.array:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis)


def _format_Q_llama(Q: str):
    return (
        "<|start_header_id|>user<|end_header_id|> "
        + Q
        + "<|eot_id|>"
        + "<|start_header_id|>assistant<|end_header_id|>"
    )


def _format_Q_phi3(Q: str):
    return "<|user|>\n" + Q + "<|end|>\n" + "<|assistant|>\n"


def _format_Q_gemma3(Q: str):
    return "<start_of_turn>user\n" + Q + "<end_of_turn>\n<start_of_turn>model\n"


model_question_format_mapping = {
    KblamLlamaForCausalLM: _format_Q_llama,
    KBLaMPhi3ForCausalLM: _format_Q_phi3,
    KBLaMGemma3ForCausalLM: _format_Q_gemma3,
}

model_prune_format_mapping = {
    KblamLlamaForCausalLM: _prune_for_llama,
    KBLaMPhi3ForCausalLM: _prune_for_phi3,
    KBLaMGemma3ForCausalLM: _prune_for_gemma3,
    "llama3": _prune_for_llama, 
    "phi3": _prune_for_phi3,
    "gemma3": _prune_for_gemma3,
}


def answer_question(
    tokenizer: transformers.PreTrainedTokenizer,
    model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM | KBLaMGemma3ForCausalLM,
    Q: str,
    kb=None,
    kb_config: Optional[KBLaMConfig] = None,
    topk_size: int = -1,
):
    for m in model_question_format_mapping:
        if isinstance(model, m):
            input_str = model_question_format_mapping[m](Q)
    tokenizer_output = tokenizer(input_str, return_tensors="pt", padding=True).to(
        "cuda"
    )
    input_ids, attention_masks = (
        tokenizer_output["input_ids"],
        tokenizer_output["attention_mask"],
    )

    with torch.autograd.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            kb_kvs=kb,
            max_new_tokens=150,
            tokenizer=tokenizer,
            output_attentions=True,
            kb_config=kb_config,
            topk_size=topk_size,
        ).squeeze()
    outputs = tokenizer.decode(outputs, skip_special_tokens=False)

    for m in model_prune_format_mapping:
        if isinstance(model, m):
            pruned_output = model_prune_format_mapping[m](outputs)
            return pruned_output
    
    # 인스턴스 매칭이 안되면 모델 타입으로 찾기
    if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
        model_type = model.config.model_type
        if model_type == 'gemma':
            return _prune_for_gemma3(outputs)
        elif model_type == 'llama':
            return _prune_for_llama(outputs)
        elif model_type == 'phi':
            return _prune_for_phi3(outputs)
    
    # 기본값 반환
    return outputs
