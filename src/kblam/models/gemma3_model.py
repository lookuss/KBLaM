import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union
import os
import numpy as np

# Gemma3ForCausalLM 대신 GemmaForCausalLM 사용
from transformers import GemmaForCausalLM

from kblam.models.kblam_config import KBLaMConfig


class KBAdapter(nn.Module):
    """
    Knowledge Base adapter for Gemma model
    """
    def __init__(self, hidden_size, kb_size, kb_layer_norm=True, key_value_dim=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.kb_size = kb_size
        self.kb_layer_norm = kb_layer_norm

        self.query_proj = nn.Linear(
            hidden_size, kb_size if key_value_dim is None else key_value_dim, bias=True
        )

        if kb_layer_norm:
            self.kb_layernorm = nn.LayerNorm(
                kb_size if key_value_dim is None else key_value_dim,
                elementwise_affine=False,
                bias=False,
            )
        else:
            self.kb_layernorm = nn.Identity()

    def forward(self, hidden_states, kb_embeddings=None):
        """
        hidden_states (batch_size, seq_len, hidden_size): 모델에서 나온 은닉 상태
        kb_embeddings (kb_size, kb_dim): KB 임베딩
        """
        # 쿼리 생성
        query = self.query_proj(hidden_states)  # (batch_size, seq_len, kb_size)
        query = self.kb_layernorm(query)

        if kb_embeddings is None:
            return query, None, None

        # key와 value의 임베딩을 가져옴
        key_embds, value_embds = kb_embeddings

        # 어텐션 스코어 계산
        kb_attn_weight = torch.matmul(
            query, key_embds.transpose(-1, -2)
        )  # (batch_size, seq_len, n_entities)

        kb_attn_probs = F.softmax(kb_attn_weight, dim=-1)
        kb_context = torch.matmul(kb_attn_probs, value_embds)  # (batch_size, seq_len, embed_dim)

        return query, kb_attn_probs, kb_context


class KBLaMGemma3ForCausalLM(GemmaForCausalLM):
    """
    Gemma 모델에 KB 기능을 추가한 클래스
    """
    def __init__(self, config, kb_config=None):
        super().__init__(config)
        
        # KB config 설정
        self.kb_config = kb_config if kb_config is not None else KBLaMConfig()
        
        # 모델 은닉층 크기
        hidden_size = config.hidden_size
        
        # KB 어댑터 생성
        self.kb_adapters = nn.ModuleList(
            [
                KBAdapter(
                    hidden_size=hidden_size,
                    kb_size=self.kb_config.kb_size,
                    kb_layer_norm=self.kb_config.kb_layer_norm,
                )
                if i % self.kb_config.kb_token_layer_frequency == 0
                else None
                for i in range(config.num_hidden_layers)
            ]
        )
        
        # KB 어탠션을 위한 추가 레이어
        self.kb_projection = nn.Linear(
            self.kb_config.kb_size, config.hidden_size, bias=False
        )
        
        # 분리된 쿼리 헤드가 필요한 경우
        if self.kb_config.sep_query_head:
            # 분리된 쿼리 헤드를 위한 별도의 어댑터
            self.kb_query_head = nn.ModuleList(
                [
                    KBAdapter(
                        hidden_size=hidden_size,
                        kb_size=self.kb_config.kb_size,
                        kb_layer_norm=self.kb_config.kb_layer_norm,
                    )
                    if i % self.kb_config.kb_token_layer_frequency == 0
                    else None
                    for i in range(config.num_hidden_layers)
                ]
            )
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        kb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        topk_size: int = -1,
        kb_kvs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # 호환성을 위한 별칭
        kb_config: Optional[KBLaMConfig] = None,  # 평가 시 사용되는 설정
        tokenizer=None,  # 디코딩 등에 사용
        save_attention_weights: bool = False,  # 어텐션 가중치 저장 여부
        attention_save_loc: str = None,  # 어텐션 저장 경로
        attention_file_base_name: str = None,  # 어텐션 파일 이름
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # kb 및 kb_kvs 처리 (둘 다 같은 기능)
        if kb_kvs is not None and kb is None:
            kb = kb_kvs
        
        # kb_config가 제공된 경우 임시로 설정 변경
        original_kb_config = None
        if kb_config is not None:
            original_kb_config = self.kb_config
            self.kb_config = kb_config

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Gemma 모델의 내부 구현에 맞게 forward 호출
        model_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,  # 여기서 None으로 설정하고 나중에 loss를 계산
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # KB 적용을 위해 항상 히든 스테이트 출력 필요
            return_dict=True,
        )

        sequence_output = model_outputs.logits
        hidden_states = model_outputs.hidden_states

        # KB 처리 로직 (kb가 제공된 경우)
        kb_logits = None
        if kb is not None:
            all_kb_attns = []
            for i, adapter in enumerate(self.kb_adapters):
                if adapter is not None:
                    query_head = self.kb_query_head[i] if hasattr(self, 'kb_query_head') and self.kb_config.sep_query_head else adapter
                    layer_hidden_states = hidden_states[i + 1]  # +1 because the first element is the embedding layer
                    
                    # Top-k KB 엔티티를 사용하는 경우
                    use_kb = kb
                    if topk_size > 0 and topk_size < kb[0].shape[0]:
                        with torch.no_grad():
                            query, _, _ = query_head(layer_hidden_states, None)
                            # 평균 쿼리 계산
                            avg_query = query.mean(dim=1, keepdim=True)
                            # key와의 유사도 계산
                            sim = torch.matmul(avg_query, kb[0].transpose(-1, -2))
                            # top-k 인덱스 선택
                            _, topk_indices = torch.topk(sim.squeeze(1), topk_size, dim=-1)
                            # topk 엔티티 선택
                            topk_kb = (
                                torch.stack([kb[0][topk_indices[b]] for b in range(topk_indices.shape[0])]),
                                torch.stack([kb[1][topk_indices[b]] for b in range(topk_indices.shape[0])]),
                            )
                            use_kb = topk_kb
                    
                    # KB 어텐션 적용
                    _, kb_attn, kb_context = adapter(layer_hidden_states, use_kb)
                    
                    if kb_attn is not None:
                        all_kb_attns.append(kb_attn)
                        
                        # 어텐션 가중치 저장 (평가용)
                        if save_attention_weights and attention_save_loc:
                            attn_np = kb_attn.cpu().detach().numpy()
                            attn_file = f"{attention_file_base_name}_{i}.npy"
                            attn_path = os.path.join(attention_save_loc, attn_file)
                            np.save(attn_path, attn_np)
                    
                    if kb_context is not None:
                        # KB 컨텍스트 투영
                        kb_proj_context = self.kb_projection(kb_context)
                        
                        # 스케일 팩터 적용
                        kb_proj_context = kb_proj_context * self.kb_config.kb_scale_factor
                        
                        # 원래 출력에 KB 컨텍스트 추가
                        sequence_output = sequence_output + kb_proj_context

        # KB 설정을 원래대로 복원
        if original_kb_config is not None:
            self.kb_config = original_kb_config

        # Loss 계산 (레이블이 제공된 경우)
        loss = None
        if labels is not None:
            # loss 계산 로직 (Gemma 모델의 구현에 맞게 조정)
            shift_logits = sequence_output[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (sequence_output,) + model_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=sequence_output,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        ) 