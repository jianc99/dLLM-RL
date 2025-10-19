# adpated from SADR https://github.com/JetAstra/SDAR/blob/main/generate.py

import argparse
import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def top_k_logits(logits, k):
    if k <= 0:
        return logits
    else:
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)


def top_p_logits(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool),
                                 -1, sorted_indices, sorted_mask)
    logits = logits.masked_fill(mask_indices, float('-inf'))
    return logits


def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    orig_shape = logits.shape[:-1]    # [batch, block]
    vocab_size = logits.shape[-1]

    logits = logits.reshape(-1, vocab_size)  # [batch*block, vocab]

    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        logits = top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = top_p_logits(logits, top_p)
    probs = F.softmax(logits, dim=-1)  # shape: [batch*block, vocab]
    assert probs.dim() == 2
    token = torch.multinomial(probs, num_samples=1)  # [batch*block, 1]
    token_prob = torch.gather(probs, -1, token)     # [batch*block, 1]

    return token.view(*orig_shape), token_prob.view(*orig_shape)


def get_num_transfer_tokens(block_length, steps):
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


@torch.no_grad()
def block_diffusion_generate(
        model,
        prompt,
        mask_id,
        gen_length=128,
        block_length=8,
        denoising_steps=8,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        remasking_strategy='low_confidence_dynamic',
        confidence_threshold=0.85,
        stopping_criteria_idx=None
    ):

    model.eval()
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length +
                  block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(
        num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                               .repeat_interleave(block_length, dim=1).unsqueeze(0)
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    x = torch.full((1, total_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    global_nfp = 0

    # Prefill stage
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:,
                                                       :prefill_length, :prefill_length]
        if cur_attn_mask.dim() == 3:
            cur_attn_mask = cur_attn_mask[:, None, :, :]
        cur_position_ids = position_ids[:, :prefill_length]
        model(cur_x,
              attention_mask=cur_attn_mask,
              position_ids=cur_position_ids,
              past_key_values=past_key_values,
              use_cache=True,
              store_kv=True)

    num_transfer_tokens = get_num_transfer_tokens(
        block_length, denoising_steps)

    # Decode stage
    for num_block in range(prefill_blocks, num_blocks):
        cur_x = x[:, num_block*block_length:(num_block+1)*block_length].clone()
        cur_attn_mask = block_diffusion_attention_mask[
            :, num_block*block_length:(num_block+1)*block_length, :(num_block+1)*block_length
        ]
        if cur_attn_mask.dim() == 3:
            cur_attn_mask = cur_attn_mask[:, None, :, :]
        cur_position_ids = position_ids[:, num_block *
                                        block_length:(num_block+1)*block_length]
        for step in range(denoising_steps + 1):
            mask_index = (cur_x == mask_id)
            if mask_index.sum() == 0:
                # Store kv cache
                model(cur_x,
                      attention_mask=cur_attn_mask,
                      position_ids=cur_position_ids,
                      past_key_values=past_key_values,
                      use_cache=True,
                      store_kv=True)
                global_nfp += 1
                break

            # Denosing
            logits = model(cur_x,
                           attention_mask=cur_attn_mask,
                           position_ids=cur_position_ids,
                           past_key_values=past_key_values,
                           use_cache=True,
                           store_kv=False).logits
            global_nfp += 1

            # Sampling
            x0, x0_p = sample_with_temperature_topk_topp(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            # Sampling strategy
            if remasking_strategy == 'sequential':
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(cur_x.shape[0]):
                    if mask_index[j].any():
                        first_mask_index = mask_index[j].nonzero(as_tuple=True)[
                            0].min().item()
                        transfer_index[j, first_mask_index:first_mask_index +
                                       num_transfer_tokens[step]] = True
                    else:
                        raise ValueError(
                            "No mask tokens found in the current block.")

            elif remasking_strategy == 'low_confidence_static':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    _, idx = torch.topk(
                        confidence[j], num_transfer_tokens[step])
                    transfer_index[j, idx] = True

            elif remasking_strategy == 'low_confidence_dynamic':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    high_conf_mask = confidence[j] > confidence_threshold
                    num_high_confidence = high_conf_mask.sum()
                    if num_high_confidence >= num_transfer_tokens[step]:
                        transfer_index[j] = high_conf_mask
                    else:
                        _, idx = torch.topk(
                            confidence[j], num_transfer_tokens[step])
                        transfer_index[j, idx] = True
            else:
                raise ValueError(
                    f"Unknown remasking strategy: {remasking_strategy}")

            cur_x[transfer_index] = x0[transfer_index]

        x[:, num_block*block_length:(num_block+1)*block_length] = cur_x
        if stopping_criteria_idx is not None and any(stop_idx in x[:, prompt_length:] for stop_idx in stopping_criteria_idx):
            break
    print(f"Total number of forward passes: {global_nfp}, avg number of tokens per forward pass: {((x != mask_id).sum().item() - prompt_length)/global_nfp:.2f}")
    return x


@torch.no_grad()
def block_diffusion_generate_ar_verify(
        model,
        prompt,
        mask_id,
        gen_length=128,
        block_length=8,
        denoising_steps=8,
        temperature=1.0,
        AR_temperature=0.6,
        top_k=0,
        top_p=1.0,
        confidence_threshold=0.85,
        stopping_criteria_idx=None
    ):

    model.eval()
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length +
                  block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.ones(
        block_length, block_length, device=model.device)

    position_ids = torch.arange(total_length+block_length, device=model.device).unsqueeze(0)

    ar_mask = torch.tril(torch.ones(
        total_length+block_length, total_length+block_length, device=model.device))
    ar_attention_mask = ar_mask.unsqueeze(0).unsqueeze(1)

    x = torch.full((1, total_length+block_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    prefill_length = prompt_length

    # Prefill stage
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = ar_attention_mask[:, :, :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]
        ar_logits =model(cur_x,
              attention_mask=cur_attn_mask,
              position_ids=cur_position_ids,
              past_key_values=past_key_values,
              use_cache=True,
              store_kv=True).logits[:, -1, :]
    
    bonus_token, _ = sample_with_temperature_topk_topp(
            ar_logits,
            temperature=AR_temperature,
            top_k=top_k,
            top_p=top_p
        )
    bonus_token = bonus_token.squeeze(-1)
    x[:, prefill_length:prefill_length+1] = bonus_token

    cur_index = prefill_length

    num_transfer_tokens = get_num_transfer_tokens(
        block_length, denoising_steps)

    global_nfp = 0

    # Decode stage
    while True:
        cur_x = x[:, cur_index:cur_index+block_length].clone()
        cur_ar_attention_mask = ar_attention_mask[:, :, cur_index:cur_index+block_length, :cur_index+block_length]
        cur_attn_mask = torch.cat((ar_attention_mask[:, :, cur_index:cur_index+block_length, :cur_index], block_mask.unsqueeze(0).unsqueeze(1)), dim=-1)
        cur_position_ids = position_ids[:, cur_index:cur_index+block_length]
        for step in range(denoising_steps + 1):
            mask_index = (cur_x == mask_id)
            if mask_index.sum() == 0:
                # Store kv cache
                logits = model(cur_x,
                      attention_mask=cur_ar_attention_mask,
                      position_ids=cur_position_ids,
                      past_key_values=past_key_values,
                      use_cache=True,
                      store_kv=False).logits
                AR_tokens, _ = sample_with_temperature_topk_topp(
                    logits,
                    temperature=0.6,
                    top_k=top_k,
                    top_p=top_p
                )
                incorrect = (AR_tokens[:, :-1] != cur_x[:, 1:])
                # Give each position a numeric index
                idxs = torch.arange(incorrect.size(1), device=incorrect.device).unsqueeze(0).expand_as(incorrect)
                # Replace correct positions with a large number, then take the min per row
                # -> first mismatch index per sequence
                first_incorrect = torch.where(incorrect, idxs, torch.full_like(idxs, incorrect.size(1))).min(dim=1).values
                first_incorrect[first_incorrect == incorrect.size(1)] = cur_x.size(1) - 1
                # cur_index += first_incorrect[0].item() + 1
                verified_tokens = cur_x[:, :first_incorrect[0].item()+1]
                cur_x = verified_tokens
                cur_position_ids = cur_position_ids[:, :first_incorrect[0].item()+1]
                cur_ar_attention_mask = cur_ar_attention_mask[:, :, :first_incorrect[0].item()+1, :cur_index+first_incorrect[0].item()+1]
                model(cur_x,
                      attention_mask=cur_ar_attention_mask,
                      position_ids=cur_position_ids,
                      past_key_values=past_key_values,
                      use_cache=True,
                      store_kv=True)
                bonus_token = AR_tokens[:, first_incorrect[0].item():first_incorrect[0].item()+1]
                x[:, cur_index:cur_index+first_incorrect[0].item()+1] = verified_tokens
                x[:, cur_index+first_incorrect[0].item()+1:cur_index+first_incorrect[0].item()+2] = bonus_token
                cur_index += first_incorrect[0].item() + 1
                # print(f"Current generated length: {cur_index-prompt_length}")
                global_nfp += 1
                break

            # Denosing
            logits = model(cur_x,
                           attention_mask=cur_attn_mask,
                           position_ids=cur_position_ids,
                           past_key_values=past_key_values,
                           use_cache=True,
                           store_kv=False).logits
            global_nfp += 1

            # Sampling
            x0, x0_p = sample_with_temperature_topk_topp(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            confidence = torch.where(mask_index, x0_p, -torch.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            original_cur_x = cur_x.clone()
            for j in range(confidence.shape[0]):
                high_conf_mask = confidence[j] > confidence_threshold
                num_high_confidence = high_conf_mask.sum()
                if num_high_confidence >= num_transfer_tokens[step]:
                    transfer_index[j] = high_conf_mask
                else:
                    _, idx = torch.topk(
                        confidence[j], num_transfer_tokens[step])
                    transfer_index[j, idx] = True
            cur_x[transfer_index] = x0[transfer_index]
            cur_x[:, 0] = original_cur_x[:, 0]  # Keep the first token unchanged

        if stopping_criteria_idx is not None and any(stop_idx in x[:, prompt_length:] for stop_idx in stopping_criteria_idx):
            break
        if cur_index >= total_length:
            break
    print(f"Total number of forward passes: {global_nfp}, avg number of tokens per forward pass: {((x != mask_id).sum().item() - prompt_length)/global_nfp:.2f}")
    return x