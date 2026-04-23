"""
#####################################################################################################################

    trust probing project - 2025

    execute a model with a prompt and retrieve activations, logprobs

#####################################################################################################################
"""

import  os
import  sys
import  string
import  math
import  numpy           as np
import  torch
import  torch.nn.functional as F
from    propmt_samples  import pr_yes, pr_no 


hf_file         = "../data/.hf.txt"         # file with the current HF API access key
model           = None
tokenizer       = None

model_name      = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def set_hf():
    """
    Get the hugginface client

    """
    from    transformers    import AutoTokenizer, AutoModelForCausalLM
    from    huggingface_hub import login

    global model
    global tokenizer
    key             = open( hf_file, 'r' ).read().rstrip()
    login( token=key )

    tokenizer       = AutoTokenizer.from_pretrained( model_name, use_fast=True )
    model           = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map  = "auto",
            torch_dtype = torch.bfloat16,
    )
    tokenizer.pad_token_id      = tokenizer.eos_token_id
    model.config.pad_token_id   = model.config.eos_token_id


# ===================================================================================================================
# tiny utilities
# ===================================================================================================================

def _ensure_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]

def build_variants( base ):
    """
    generate variants of "no", "yes" systematically
    args:
        base [str] in {"yes","no"}
    """
    prefixes = ["", " ", "  "]      # 0/1/2 leading spaces
    suffixes = ["", "."]            # optional period
    cases    = [base, base.capitalize()]  # yes/Yes
    variants = {f"{p}{c}{s}" for p in prefixes for c in cases for s in suffixes}
    return sorted(variants)


# ===================================================================================================================
# 
# main functions
#
# last_token_layerwise_reps()       extract hidden activations
# continuation_logprobs_batched()   extract logprobs of completion
# class_logprobs_from_variants()    aggregates variants of yes/no completions
# make_extractor()                  make an extractor_fn that can be plugged into the collector script
# ===================================================================================================================

@torch.no_grad()
def last_token_layerwise_reps( inp, drop_embedding=True ):
    """
    inp: tokenizer(...) dict on device, add_special_tokens=False
    Returns torch.Tensor of shape (L, H) where L = #transformer blocks
    (embedding layer removed by default).
    """
    fw      = model(**inp, output_hidden_states=True, use_cache=True)
    hs      = fw.hidden_states                                      # tuple: (emb, layer1, ..., layerN)
    layers  = hs[1:] if drop_embedding else hs                      # drop embedding layer if requested
    reps    = torch.stack([h[:, -1, :] for h in layers], dim=0)     # (L, 1, H)
    return reps.squeeze( 1 )                                        # (L, H)

@torch.no_grad()
def continuation_logprobs_batched( prompt, candidates):
    """
    Compute sequence log-prob for each candidate continuation under the model,
    using teacher forcing on (prompt + candidate). Batches all candidates.
    Returns: dict {candidate_str: logprob_float}
    """
    device      = next(model.parameters()).device
    # Build a batch: one row per candidate
    enc_prompt  = tokenizer(prompt, add_special_tokens=False)
    plen        = len(enc_prompt["input_ids"])

    # Encode each (prompt + cand) to tensors; keep per-cand lengths
    batch_ids   = []
    cand_lens   = []
    for c in candidates:
        enc     = tokenizer(prompt + c, add_special_tokens=False)
        ids     = enc["input_ids"]
        batch_ids.append(ids)
        cand_lens.append(len(ids) - plen)

    # Pad to batch
    maxlen      = max(len(ids) for ids in batch_ids)
    pad_id      = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    input_ids   = []
    attention_mask = []
    for ids in batch_ids:
        pad_n   = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad_n)
        attention_mask.append([1]*len(ids) + [0]*pad_n)

    input_ids   = torch.tensor(input_ids, device=device, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, device=device, dtype=torch.long)

    out         = model(input_ids=input_ids, attention_mask=attention_mask)
    # next-token logits; shift for teacher forcing
    logits      = out.logits[:, :-1, :]  # (B, T-1, V)
    logprobs    = F.log_softmax(logits.float(), dim=-1)

    # Gather logprobs of each candidate token sequence
    cand2logp   = {}
    for i, (ids, Lc) in enumerate(zip(batch_ids, cand_lens)):
        # indices within this sequence
        # positions that *predict* the candidate tokens start at plen-1
        start   = plen - 1
        end     = start + Lc  # exclusive in slicing
        # target candidate token ids
        cand_token_ids = torch.tensor(ids[plen:plen+Lc], device=device).unsqueeze(0).unsqueeze(-1)  # (1, Lc, 1)
        # extract the relevant slice for this batch row
        row_lp  = logprobs[i:i+1, start:end, :]  # (1, Lc, V)
        token_logps = row_lp.gather(2, cand_token_ids).squeeze(0).squeeze(-1)  # (Lc,)
        cand2logp[candidates[i]] = float(token_logps.sum().item())
    return cand2logp


def class_logprobs_from_variants(cand2logp, yes_variants, no_variants, normalize_by_count=True):
    """
    Aggregate multiple variants with log-sum-exp.
    Returns: logp_yes, logp_no, p_yes, p_no
    """
    def lme(vals):  # log-mean-exp
        if not vals: return float('-inf')
        m = max(vals)
        return m + math.log(sum(math.exp(v - m) for v in vals)) - (math.log(len(vals)) if normalize_by_count else 0.0)

    vy      = [ cand2logp[c] for c in yes_variants if c in cand2logp ]
    vn      = [ cand2logp[c] for c in no_variants  if c in cand2logp ]
    ly, ln  = lme(vy), lme(vn)

    m       = max( ly, ln )
    p_yes   = math.exp( ly - m ) / ( math.exp(ly - m) + math.exp(ln - m) )
    return ly, ln, p_yes, 1.0 - p_yes

    def lse(vals):
        if not vals:
            return float('-inf')
        m   = max(vals)
        return m + math.log( sum( math.exp(v - m) for v in vals ) )

    ly      = lse( [ cand2logp[c] for c in yes_variants if c in cand2logp ] )
    ln      = lse( [ cand2logp[c] for c in no_variants if c in cand2logp ] )
    # normalize
    m       = max( ly, ln )
    p_yes   = math.exp( ly - m ) / ( math.exp(ly - m) + math.exp(ln - m) )
    p_no    = 1.0 - p_yes
    return ly, ln, p_yes, p_no


def make_extractor( no_variants=False ):
    """
    returns an extractor_fn(prompt) -> (reps(L,H) float32 tensor, aux dict)
    aux dict contains: logp_yes, logp_no, p_yes, p_no, logodds_yes_no
    args:
         no_variants    [bool] do not include variants of "yes"/"no"
    """
    model.eval()
    if no_variants:
        yes_variants    = "yes"
        no_variants     = "no"
    else:
        yes_variants    = build_variants( "yes" )
        no_variants     = build_variants( "no" )
    yes_variants        = _ensure_list(yes_variants)
    no_variants         = _ensure_list(no_variants)

    @torch.no_grad()
    def extractor_fn(prompt: str):
        device          = next( model.parameters() ).device
        # 1) decision-state activations (layer-wise last-token reps)
        inp             = tokenizer( prompt, return_tensors="pt", add_special_tokens=False ).to(device)
        reps            = last_token_layerwise_reps( inp, drop_embedding=True)  # (L,H) float32

        # 2) robust yes/no sequence log-probs
        cands           = yes_variants + no_variants
        cand2logp       = continuation_logprobs_batched( prompt, cands )
        ly, ln, p_yes, p_no = class_logprobs_from_variants( cand2logp, yes_variants, no_variants )

        aux = {
            "logp_yes": float( ly ),
            "logp_no":  float( ln ),
            "p_yes":    float( p_yes ),
            "p_no":     float( p_no ),
            "logodds_yes_no": float( ly - ln ),
        }
        return reps.float(), aux
    return extractor_fn

