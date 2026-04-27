import requests
import time
from tqdm.auto import tqdm
import torch
from datasets import load_dataset

def compute_text_perplexity(completions):

    import torch
    import math
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from torch.nn import functional as F
    from tqdm.auto import tqdm

    if completions is None or len(completions) == 0:
        return [], float('nan')

    model_name = 'gpt2'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载 tokenizer / model（padding 用 eos）
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    batch_size = 8
    ppls = []
    with torch.no_grad():
        for i in tqdm(range(0, len(completions), batch_size)):
            batch_texts = completions[i:i+batch_size]
            enc = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc.get('attention_mask', torch.ones_like(input_ids)).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B, T, V)

            # 计算逐 token 交叉熵：对 logits 和 labels 右移一位
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()

            B, Tm, V = shift_logits.shape
            # 计算 token-level loss (不归约)
            loss_flat = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1), reduction='none')
            loss = loss_flat.view(B, Tm)

            # 对每个样本按 mask 求平均负对数似然
            token_counts = shift_mask.sum(dim=1).float()
            # 防止除以 0
            token_counts_clamped = token_counts.clone()
            token_counts_clamped[token_counts_clamped == 0] = 1.0

            per_sample_nll = (loss * shift_mask).sum(dim=1) / token_counts_clamped
            per_sample_ppl = torch.exp(per_sample_nll)

            # 将 token_counts 为 0 的样本设为 nan
            per_sample_ppl = per_sample_ppl.cpu().tolist()
            for idx_in_batch, cnt in enumerate(token_counts.cpu().tolist()):
                if cnt == 0:
                    ppls.append(float('nan'))
                else:
                    ppls.append(float(per_sample_ppl[idx_in_batch]))

    # 计算平均 PPL（忽略 nan）
    import numpy as _np
    valid = [x for x in ppls if not (_np.isnan(x) or _np.isinf(x))]
    mean_ppl = float(_np.mean(valid)) if len(valid) > 0 else float('nan')
    return ppls, mean_ppl




def perspective_api_request(text, max_retries=5, retry_delay=3):

    payload = {
        "comment": {"text": text},
        "requestedAttributes": {
            k:{} for k in PERSPECTIVE_COMPS
        }
    }

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(PERSPECTIVE_URL, json=payload, proxies=proxies, timeout=15)
            r.raise_for_status()
            data = r.json()["attributeScores"]
            return [data[k]["summaryScore"]["value"] for k in PERSPECTIVE_COMPS]
        except Exception as e:
            print(f"第 {attempt} 次请求失败：{e}")
            if hasattr(e, 'response') and e.response is not None:
                print("响应内容：", e.response.text)
            if attempt < max_retries:
                print(f"{retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                print("多次重试仍失败。")
                return [0.5 for k in PERSPECTIVE_COMPS]

def evaluate_toxicity(texts):

    scores = {k:[] for k in PERSPECTIVE_COMPS}
    for text in tqdm(texts):
        score_tuple = perspective_api_request(text)
        for k, v in zip(PERSPECTIVE_COMPS, score_tuple):
            scores[k].append(v)
    
    avg_scores = {k: sum(v) / len(v) for k, v in scores.items()}
    return scores, avg_scores