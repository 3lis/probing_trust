"""
#####################################################################################################################

    trust probing project - 2025

    collect activation from dataset of prompts, and archive it

#####################################################################################################################
"""

import  os
import  hashlib
import  sys
import  string
import  torch
import  numpy           as np
import  pandas          as pd
import  xarray          as xr
from    numcodecs   import Blosc


model_map       = None
scen_map        = {"school":0, "fire":1, "farm":2}

zarr_path       = "../res/activation.zarr"
manifest_path   = "../res/manifest.parquet"

# ===================================================================================================================
# data layout
# A) Tensor store: xarray + zarr
# Save a single 3-D array of activations with dims (sample, layer, hidden).
# Attach lightweight per-sample coords (labels, model, scenario, prompt_id) directly to the array.
# Keep the full prompt text in a separate manifest (Parquet) keyed by prompt_id to avoid bloating the tensor store.
# This gives you:
# Appendable, compressed, chunked storage.
# Super fast filtering like “all samples from school on model llama-8b”.
# Clean per-layer slices for probes.
# Shapes/Types
# repr: (N, L, H) float16 (storage) — convert to float32 on read for numerics.
# Coords on sample:
# yes_no: int8
# model_code: int16 (map string ↔ code in manifest)
# scen_code: int8 (map string ↔ code in manifest)
# prompt_id: fixed-length string or int64 (hash)
# Compression: Blosc/Zstd with bit-shuffle works well.
#
# B) Manifest: Parquet
# A tidy table with columns:
# prompt_id (key)
# prompt (full text)
# model_code, model_name
# scen_code, scen_name
# Any extra per-prompt attributes (e.g., token counts, p_yes, loglik_yes/no…)
# 
#
# ===================================================================================================================


COMP = Blosc( cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE )

def prompt_hash( s: str ) -> str:
    """
    short, stable id for prompts (same across models/scenarios)
    """
    return hashlib.blake2b( s.encode("utf-8" ), digest_size=8 ).hexdigest()

def to_float16_storage( x_torch ) -> np.ndarray:
    """
    # (L,H) torch -> float16 numpy for compact storage
    """
    return x_torch.detach().cpu().to(torch.float16).numpy()
    
def make_encoding( ds_batch, H, chunks_sample=256, chunks_hidden=512 ):
    """
    build an encoding
    """
    enc = {}

    # main tensor: (sample, layer, hidden)
    if "repr" in ds_batch:
        enc["repr"] = {
            "dtype": "float16",
            "compressor": COMP,
            "chunks": (chunks_sample, 1, min(chunks_hidden, H)),
        }

    # labels / metadata (chunk on sample only)
    for name in ("yes_no", "prompt_id", "scen", "model"):
        if name in ds_batch:
            enc[name] = {
                "compressor": None,
                "chunks": (chunks_sample,),
            }
    return enc


def init_or_append_zarr(
    zarr_path: str,
    batch_repr_f16: np.ndarray,     # (B, L, H) float16
    sample_ids,                     # list[str], len B
    yes_no,                         # np.int8, (B,)
    model_code,                     # np.int16, (B,)
    model_name,                     # str
    scen_code,                      # np.int8,  (B,)
    prompt_ids,                     # list[str], len B
    layer_index=None                # np.int32, (L,)
):
    """
    build an xarray Dataset for this batch
    """
    group   = model_name
    L       = batch_repr_f16.shape[1]
    ds_batch = xr.Dataset(
        data_vars={
            "repr":      (("sample","layer","hidden"), batch_repr_f16),
            "yes_no":    (("sample",), yes_no.astype(np.int8)),
            "model_code":(("sample",), model_code.astype(np.int16)),
            "scen_code": (("sample",), scen_code.astype(np.int8)),
            "prompt_id": (("sample",), np.array(prompt_ids, dtype="U16")),
        },
        coords={
            "sample": np.array(sample_ids, dtype="U24"),
            "layer":  np.array(
                layer_index if layer_index is not None
                else np.arange(1, L+1, dtype=np.int32)
            ),
        },
        attrs={"note":"trust representations; last-token per layer (no embeddings)"}
    )

    H       = batch_repr_f16.shape[2]
    enc     = make_encoding(ds_batch, H, chunks_sample=256, chunks_hidden=512)
    group_exists = True

    if not os.path.exists(zarr_path):
        group_exists    = False
    else:
        try:
                ex = xr.open_zarr( zarr_path, group=group )
        except Exception:
            group_exists = False
            ex = None

    if not group_exists:
        ds_batch.to_zarr(
            zarr_path,
            group=group,
            mode="w",
            encoding=enc,
            zarr_version=2,          # <<< force v2
        )
    else:
        ds_batch.to_zarr(
            zarr_path,
            group=group,
            mode="a",
            append_dim="sample",
            zarr_version=2,          # <<< force v2
        )


def _flush(buffers, buffers_aux, zarr_path, manifest_path):
    """
    stack activations
    """
    batch_repr = np.stack([b[0] for b in buffers], axis=0)      # (B,L,H)
    sample_ids = [b[1] for b in buffers]
    yes_no     = np.array([b[2] for b in buffers], dtype=np.int8)
    model_code = np.array([b[3] for b in buffers], dtype=np.int16)
    scen_code  = np.array([b[4] for b in buffers], dtype=np.int8)
    prompt_ids = [b[5] for b in buffers]

    model_name  = buffers[ 0 ][ 7 ]
    # Write/append Zarr tensor store
    init_or_append_zarr(
        zarr_path, batch_repr, sample_ids, yes_no, model_code, model_name, scen_code, prompt_ids
    )

    # Append/update Parquet manifest (metadata + yes/no scores)
    man_batch = pd.DataFrame({
        "prompt_id": prompt_ids,
        "prompt":    [b[6] for b in buffers],
        "model_code":[b[3] for b in buffers],
        "model_name":[b[7] for b in buffers],
        "scen_code": [b[4] for b in buffers],
        "scen_name": [b[8] for b in buffers],
        "logp_yes":  [aux["logp_yes"] for aux in buffers_aux],
        "logp_no":   [aux["logp_no"]  for aux in buffers_aux],
        "p_yes":     [aux["p_yes"]    for aux in buffers_aux],
        "p_no":      [aux["p_no"]     for aux in buffers_aux],
        "logodds_yes_no": [aux["logodds_yes_no"] for aux in buffers_aux],
    })

    if not os.path.exists(manifest_path):
        man_batch.to_parquet(manifest_path, index=False, compression="zstd")
    else:
        # simple append (fast). If you want strict de-dup, read/concat/drop_duplicates.
        existing = pd.read_parquet(manifest_path)
        merged = pd.concat([existing, man_batch], ignore_index=True)
        merged.to_parquet(manifest_path, index=False, compression="zstd")



def collect_and_store(
    df: pd.DataFrame,                 # columns: prompt, yes_no, model, scen
    extractor_fn,                     # returns (reps(L,H) torch.float32, aux dict)
    zarr_path: str,                   # e.g., "trust_repr.zarr"
    manifest_path: str,               # e.g., "trust_manifest.parquet"
    model_map: dict,                  # {"Meta-Llama-3.1-8B-Instruct": 1, ...}
    scen_map: dict,                   # {"school":0,"fire":1,"farm":2}
    batch_size: int = 128
):
    """
    collector with aux (logp_yes/no, probs, logodds)
    """
    buffers = []          # per-sample tuples (see _flush)
    buffers_aux = []      # just aux dicts aligned with buffers

    n_rows          = len( df )
    for i, row in df.iterrows():
        pr          = row["prompt"]
        yes_no      = int(row["yes_no"])
        model_name  = row["model"]
        scen_name   = row["scen"]

        reps, aux   = extractor_fn(pr)    # reps: (L,H) torch; aux: dict with probs/logprobs
        rep_f16     = to_float16_storage(reps)

        pid         = prompt_hash(pr)
        mid         = model_map[model_name]
        sid         = scen_map[scen_name]
        sample_id   = f"{pid}_{mid}_{sid}"
        print( f"done prompt {i:4d}\t model: {model_name} scenario: {scen_name}  sample_id: {sample_id}" )

        buffers.append((
            rep_f16,      # 0
            sample_id,    # 1
            yes_no,       # 2
            mid,          # 3
            sid,          # 4
            pid,          # 5
            pr,           # 6
            model_name,   # 7
            scen_name     # 8
        ))
        buffers_aux.append(aux)

        if len(buffers) >= batch_size:
            print( "flushing buffers" )
            _flush(buffers, buffers_aux, zarr_path, manifest_path)
            buffers.clear()
            buffers_aux.clear()

    if buffers:
        _flush(buffers, buffers_aux, zarr_path, manifest_path)

    return True
