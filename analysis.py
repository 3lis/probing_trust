"""
#####################################################################################################################

    trust probing project - 2025

    analyzing prototypes and probes

#####################################################################################################################
"""

import  os
import  sys
import  json5
import  time
import  datetime
import  torch
import  zarr
import  numpy   as np
import  xarray  as xr
import  pandas  as pd
import  json
import  joblib

import  run_model
import  plot

from    dataclasses                     import dataclass
from    typing                          import Dict, Any, Optional
from    sklearn.metrics                 import roc_auc_score, accuracy_score

from    models                          import models_short_name

zarr_path       = "../res/activation.zarr"
manifest_path   = "../res/activation.parquet"
dir_data        = "../data"
dir_res         = "../res"
dir_stat        = "../stat"
dir_probe       = "../probes"
dir_plots       = "../plots"

model           = None
tokenizer       = None
scenarios       = None
curr_models     = None

VERBOSE         = False


# ===================================================================================================================
#
# common utilities
#
# load_store()
# select_slices()
# _get_Xy()
# 
# ===================================================================================================================

def load_store( model_name ):
    """
    access to zarr archive, using  groups
    args:
        model_name  [str] model short name
    """
    # first check for existence of the specified group
    root        = zarr.open( zarr_path, mode="r" )
    groups      = list( root.group_keys() )
    assert model_name in groups, f"error in load_store_group: group {model_name} not in {zarr_path}" 
    # open the data associated with the specified group
    ds          = xr.open_zarr( zarr_path, group=model_name )
    man         = pd.read_parquet( manifest_path )
    model_map   = (man[["model_code","model_name"]]
                 .drop_duplicates().set_index("model_name")["model_code"].to_dict())
    scen_map    = (man[["scen_code","scen_name"]]
                 .drop_duplicates().set_index("scen_name")["scen_code"].to_dict())
    return ds, man, model_map, scen_map


def select_slices( ds, *, model_code=None, scen_codes=None ):
    """
    select slices to use in the analysis
    args:
        ds          [xarray.core.dataset.Dataset]
        model_code  [int] model code or None for all
        scen_codes  [list] with int codes of scenarios or None for all
    returns:
        sub         [xarray.core.dataset.Dataset]
    """
    sub = ds
    if model_code is not None:
        sub = sub.where(sub["model_code"] == int(model_code), drop=True)
    if scen_codes is not None:
        mask = None
        for c in scen_codes:
            m = (sub["scen_code"] == int(c))
            mask = m if mask is None else (mask | m)
        sub = sub.where(mask, drop=True)
    return sub


def _get_Xy( sub, layer_k ):
    """
    get all vectors of activation and labels yes/no from the slice, at a given layer
    args:
        sub     [xarray.core.dataset.Dataset] the slice from which to get vectors
        layer_k [int] the specified layer
    returns:
        [tuple] of:
        X       [np.array] with shape (n,H)
        y       [np.array] with shape (n,)
    """
    X = sub["repr"].sel(layer=layer_k).astype(np.float32).load().values  # (n,H)
    y = sub["yes_no"].values.astype(int)                                  # (n,)
    return X, y



# ===================================================================================================================
#
# probe utilities
#
# ProbeArtifact
# load_probe_artifact()
# probe_fname()
# load_probe()
# probe_margin()
# probe_proba()
# eval_probe()
# 
# ===================================================================================================================
@dataclass
class ProbeArtifact:
    """
    object containing a probe, suitable for saving/loading
    """
    idx: np.ndarray          # selected feature indices (global, length k)
    w_std: np.ndarray        # LR weights in standardized space (k,)
    b: float                 # LR intercept
    mean: np.ndarray         # StandardScaler.mean_ for selected cols (k,)
    scale: np.ndarray        # StandardScaler.scale_ for selected cols (k,)
    meta: Dict[str, Any]     # anything you want (layer, model, C, AUC, etc.)


def load_probe_artifact(path: str) -> ProbeArtifact:
    """
    load an object containing a probe from file
    NOTE the trick to prevent error for probes saved manually within ipython!
    """
    import __main__
    # bind the class name where the pickle expects it to be
    __main__.ProbeArtifact = ProbeArtifact

    return joblib.load(path)


def probe_fname( model_name, layer, ptype="stable", scen='' ):
    """
    construct the proper filename of a probe
    """
    if ptype == "common":
        fname  = f"{ptype}_{model_name}_{layer}.joblib"
    else:
        fname  = f"{ptype}_{model_name}_{scen}_{layer}.joblib"
    path   = os.path.join( dir_probe, fname )
    return path


def load_probe( model_name, layer, ptype="common", scen='' ):
    """
    load an object containing a probe using info about the filename
    """
    path   = probe_fname( model_name, layer, ptype=ptype, scen=scen )
    if not os.path.isfile( path ):
        if VERBOSE:
            print( f"probe file {path} not found" )
        return None
    return load_probe_artifact( path )


def probe_margin( art, X ):
    """
    convolve the probe on a given vector
    Use on raw full-layer features X_full: (n, H)
    """
    Xs = (X[:, art.idx] - art.mean) / (art.scale + 1e-12)
    return Xs @ art.w_std + art.b

def probe_proba(art, X ):
    """
    returns the probability margins applying a probe to the given vector
    """
    z = probe_margin( art, X )
    return 1.0 / (1.0 + np.exp(-z))


def eval_probe(art, X, y ):
    """
    evaluate AUC/ACC of a probe over a set of activation
    """
    p       = probe_proba( art, X )
    auc     = roc_auc_score( y, p )
    acc     = accuracy_score( y, (p > 0.5 ).astype(int) )
    return auc, acc


# ===================================================================================================================
# 
# causal effects on discrimination for a non-final layer
# 
# ===================================================================================================================

def causal_drop( art, X, y=None, s=None ):
    """
    discrimination drop caused by mean ablation in the probe's standardized space.
    baseline margin: m_before = (Z @ w) + b
    after ablation:  m_after  = (Z_abl @ w) + b, with Z_abl[:, ablated] = 0
    where Z = (X[:, idx] - mean) / scale and (w,b) are (w_std, b).
    NOTE that ablation is computed with a quick trick, taking into account that
    being S the subspace of idx:
        Z' w + b = ( Z - Z[;S]P_S ) w + b = Z w + b - Sum( Z[;j]w_j
    in fact, when s=None, it simply holds:
        m_after = b
        drop    = Z w

    args:
        art [ProbeArtifact] the probe
        X   [np.array] (n,H) raw pre-norm activations at probe's layer
        y   [np.array] (n,) in {0,1} for oriented drop, or None
        s   [int] only top-|w| s features; if None, ablate all

    returns:
        [dict] with mean/min/max drop, oriented drop (if y provided)
    """
    idx = np.asarray(art.idx, dtype=int)
    w   = np.asarray(art.w_std, dtype=np.float32)
    b   = float(art.b)

    # standardize once (n,k)
    Z   = (X[:, idx] - art.mean) / (art.scale + 1e-12)

    # baseline margins
    m_before = (Z @ w) + b

    # choose which *local* columns (0..k-1) to ablate
    k   = len(idx)
    if isinstance(s, int) and 0 < s < k:
        # ablate top-|w| s features
        ablate_local = np.argsort(-np.abs(w))[:s]
    else:
        ablate_local = np.arange(k)

    # efficient mean ablation: setting Z[:, j]=0 removes that term from the dot
    # so the drop equals sum_j (Z[:, j] * w[j]) over ablated j.
    drop    = np.sum(Z[:, ablate_local] * w[ablate_local], axis=1).astype(np.float32)

    # convert drops into probabilities, for better readability
    m_after     = m_before - drop
    p_before    = 1 / ( 1+np.exp( -m_before ) )
    p_after     = 1 / ( 1+np.exp( -m_after ) )
    p_drop      = p_before - p_after
    p_mean_drop = float( p_drop.mean() )
    p_std_drop  = float( p_drop.std() )
    
    out = {
        "k_total": int(k),
        "k_ablated": int(len(ablate_local)),
        "mean_drop": float(drop.mean()),
        "min_drop":  float(drop.min()),
        "max_drop":  float(drop.max()),
        "p_mean_drop":  p_mean_drop,
        "p_std_drop":  p_std_drop
    }
    if y is not None:
        y_or = (y * 2 - 1).astype(np.int8)   # 1->+1, 0->-1
        out["mean_drop_oriented"] = float((drop * y_or).mean())
        p_drop_or   = (p_before - p_after) * y_or
        out["p_mean_drop_or"] = float( p_drop_or.mean() )
        out["p_std_drop_or"] = float( p_drop_or.std() )
    return out


def causal_flip(art, X, y, s=None):
    """
    flip in yes/no discrimination caused by ablation of the probe neurons
    mean ablation in the probe's standardized space is the same as in causal_drop(),
    see comments there
    args:
        art [ProbeArtifact] the probe
        X   [np.array] (n,H) raw pre-norm activations at probe's layer
        y   [np.array] (n,) in {0,1} for oriented drop, or None
        s   [int] only top-|w| s features; if None, ablate all

    returns:
        [dict] with flipping rates
    """
    idx = np.asarray( art.idx, int )
    w   = np.asarray( art.w_std, np.float32 )
    Z   = ( X[:, idx] - art.mean ) / ( art.scale + 1e-12 )

    k   = len( idx )
    ablate_local = ( np.argsort(-np.abs(w) )[:s] if isinstance(s, int) and 0 < s < k
                    else np.arange(k))

    m_before = (Z @ w) + float(art.b)
    # the clever algebra trick for mean ablation
    drop     = np.sum(Z[:, ablate_local] * w[ablate_local], axis=1).astype(np.float32)
    m_after  = m_before - drop

    y0      = ( m_before >= 0 ).astype(int)
    y1      = ( m_after >= 0  ).astype(int)

    out     = dict()
    for t in ( 0,1 ):       # NOTE: 0=distrusted, 1=entrusted
        mask                        = ( y==t )
        out[f"true_{t}_flip_rate"]  = float( np.mean( y0[mask] != y1[mask] ) )
        out[f"true_{t}_from1to0"]   = float( np.mean( ( y0[mask]==1 ) & ( y1[mask]==0 ) ) )
        out[f"true_{t}_from0to1"]   = float( np.mean( ( y0[mask]==0 ) & ( y1[mask]==1 ) ) )
    return out
    


# ===================================================================================================================
#
# interventionist methods - last layer
#
# preliminary functions that require running a model, and store vectors on flie
# further analysis can be carried on without the need of GPU, reading vectors from file
#
# load_model()
# one_token_id()
# lm_head_pair_readout_pre()
# save_lm_readout_pre()
# load_lm_readout_pre()
# ===================================================================================================================


def load_model( model_name ):
    """
    load a model
    args:
        model_name  [str] model short name
    """
    global model, tokenizer
    short       = list( models_short_name.values() )
    assert model_name in short, f"error: {model_name} not in models_short_name"
    long        = list( models_short_name.keys() )
    model_long  = long[ short.index( model_name ) ]
    run_model.model_name    = model_long
    run_model.set_hf()
    model       = run_model.model
    tokenizer   = run_model.tokenizer


def one_token_id( s ):
    """
        tokinze a word, ensuring converts to one single token
    """
    ids = tokenizer.encode( s, add_special_tokens=False )
    assert len( ids ) == 1, f"'{s}' splits into {len(ids)} pieces"
    return ids[ 0 ]


def lm_head_pair_readout_pre( a, b ):
    """
    return w_pre and c, that can be used to compute the a-vs-b logit diff:
        diff = X w_pre + c
    where X is pre-norm final hidden state of the last token.
    args:
        a   [str] word for "yes" response
        b   [str] word for "no" response
    returns:
        [tuple]  with w_pre, c
    """
    W       = model.lm_head.weight.detach().float().cpu().numpy()       # (V,H)
    b_all   = ( model.lm_head.bias.detach().float().cpu().numpy()
             if model.lm_head.bias is not None else None )              # (V,)
    g       = model.model.norm.weight.detach().float().cpu().numpy()    # (H,)
    ia, ib  = one_token_id( a), one_token_id( b)
    w_pre   = g * ( W[ia] - W[ib] )                                     # (H,)
    c       = float( (b_all[ia] - b_all[ib] ) if b_all is not None else 0.0 )
    return w_pre.astype( np.float32 ), c


def save_lm_readout_pre( model_name ):
    """
    utility to save the pre-sinaptic weights vector of the last layer, and offset, to file
    NOTE: this function cannot be wrapped in a loop, because loading a model is memory intensive
    therefore should be executed manually for each model - here are their signatures (so far):
        save_lm_readout_pre( "ll2-7" )
        save_lm_readout_pre( "ll2-13" )
        save_lm_readout_pre( "ll3-8" )
        save_lm_readout_pre( "ph3m" )
        save_lm_readout_pre( "qw1-7" )
        save_lm_readout_pre( "qw2-7" )
        save_lm_readout_pre( "qw2-14" )
    """
    global model
    if model is None:
        load_model( model_name )
    w_pre, c    = lm_head_pair_readout_pre( "yes", "no" )
    fname       = os.path.join( dir_data, f"lm_head_{model_name}_w_pre.npy" )
    np.save( fname, w_pre )
    fname       = os.path.join( dir_data, f"lm_head_{model_name}_c.npy" )
    np.save( fname, c )


def load_lm_readout_pre( model_name ):
    """
    utility to load pre-sinaptic weights vector of the last layer, and offset, from file
    args:
        model_name  [str] model short name
    returns:
        [tuple]  with w_pre, c or None
    """
    fname       = os.path.join( dir_data, f"lm_head_{model_name}_w_pre.npy" )
    if not os.path.isfile( fname ):
        if VERBOSE:
            print( f"file {fname} with weights vector not found" )
        return None
    w_pre       = np.load( fname )
    fname       = os.path.join( dir_data, f"lm_head_{model_name}_c.npy" )
    c           = np.load( fname )
    return w_pre, float( c )



# ===================================================================================================================
#
# interventionist methods - last layer
#
# On the final layer, yes/no decision is Dlogit = h · (W_yes − W_no) + const
# Zeroing the selected neurons in h should reduce Dlogit toward the other class
#
# pair_steer_dir_in_art()
# alpha_to_flip_pair_score()
# sigma_normalized_dir()
# required_alpha_quantiles_sigma()
# flip_curve_global_alpha()
# flip_curves_bidir()
# flip_curve_symmetric()
# flip_curve_by_class()
# flip_curve_random_baseline_proj()
# intervention_flip_curve()
# ===================================================================================================================

def pair_steer_dir_in_art( w_pre, art, H ):
    """
    cmopute the steepest pair-change using only art.idx neurons:
        v = normalize( P_art(w_pre) )
    args:
        w_pre   [np.array] (H,)
        art     [ProbeArtifact] the probe
        H       [int] hidden vector dimension
    returns:
        [np.array] (H,)
    """
    v = np.zeros(H, dtype=np.float32)
    v[art.idx] = w_pre[art.idx]
    n = np.linalg.norm(v) + 1e-12
    return v / n


def alpha_to_flip_pair_score(h, v, w_pre, c=0.0, margin=0.0 ):
    """
    this function essentialy solves the 1-D equation for the amount necessary to move along
    the v direction, in order to flip the pair score, possibly with a given margin
    given
        s( h ) = hT w + c
    and a state edit
        h' = h + alpha v
    then
        s( h' ) = (h + alpha v)T w + c = s( h ) + alpha (vT w )
    and we compute the minimum alpha such that s(h') crosses 0 (possibly with margin)
    args:
        h       [np.array] (H,)
        v       [np.array] (H,)
        w_pre   [np.array] (H,)
        c       [float]
        margin  [float]
    returns:
        [float] alpha
    """
    s       = float( h @ w_pre + c )    # current score
    denom   = float( v @ w_pre )        # strenght of the direction of the score (none if 0)
    if abs( denom ) < 1e-9:
        return np.inf                   # no way to flip the pair score
    target  = -np.sign(s) * margin      # just 0 if no margin is required
    return ( target - s ) / denom       # solve the equation in alpha


def sigma_normalized_dir(w_pre, art, X_train):
    # std of raw pre-norm states on the probe dims
    std = X_train[:, art.idx].std(axis=0, ddof=1).astype(np.float32) + 1e-12
    v = np.zeros(X_train.shape[1], np.float32)
    v[art.idx] = w_pre[art.idx]
    # scale so that mean squared z-score step = 1  → α is in “σ-units”
    scale = np.sqrt(np.mean((v[art.idx] / std)**2)) + 1e-12
    return v / scale

def required_alpha_quantiles_sigma(X, w_pre, c, art, qs=(10,25,50,75,90,95,99)):
    """
    evaluate alphas - expressed as multiplier of sigma, necessary to flip a quantile of samples
    """
    v_sig = sigma_normalized_dir(w_pre, art, X)
    s0    = (X @ w_pre) + c
    denom = float(v_sig @ w_pre)  # constant across examples
    alphas = (-s0) / (denom + 1e-12)   # margin target = 0
    alphas = np.abs(alphas[np.isfinite(alphas)])
    return np.percentile(alphas, qs).tolist(), v_sig


def flip_curve_global_alpha(X, w_pre, c, v_sigma, alphas):
    """
    compute the curve of flipping rates at given alphas
    """
    s0 = (X @ w_pre) + c
    flips = []
    for a in alphas:
        s1 = ((X + a*v_sigma) @ w_pre) + c
        flips.append(float(np.mean(np.sign(s0) != np.sign(s1))))
    return np.array(flips)


def flip_curves_bidir(X, w_pre, c, art, v, max_alpha=None, n=25):
    """
    returns symmetric alpha grid and flip-rate curves for +v and -v directions.
    args:
        X       [np.array] (n,H) raw pre-norm activations at last layer
        w_pre   [np.array] (H,)
        c       [float]
        art     [ProbeArtifact] the probe
        v       [np.array] (H,) projected/normalized direction in probe subspace
        max_alpha  [float] for flipping
        n       [int] number of alpha's steps in one direction
    returns:
        [tuple]  with (positive) alphas, flip curve positive, flip curve egative
    """
    # choose a ceiling from the percentile helper, or fall back if None
    if max_alpha is None:
        # ~95th percentile of required alphas
        qs, _       = required_alpha_quantiles_sigma( X, w_pre, c, art )
        max_alpha   = max( 2.0, float(qs[-2]) )
    if max_alpha > 20:
        max_alpha   = 10 * np.ceil( max_alpha / 10 )       # ceil to multiple of 10
    alphas      = np.linspace( 0.0, max_alpha, n )

    def curve_for_dir( sign=+1 ):
        flips = []
        for a in alphas:
            X_edit = X + (sign * a) * v[None, :]
            s0 = X @ w_pre + c
            s1 = X_edit @ w_pre + c
            flips.append( float(np.mean(np.sign(s0) != np.sign(s1))) )
        return np.asarray(flips)

    curve_pos   = curve_for_dir(+1)   # push toward +v
    curve_neg   = curve_for_dir(-1)   # push toward -v
    return alphas, curve_pos, curve_neg


def flip_curve_symmetric( X, w_pre, c, art, v, max_alpha=None, n=12 ):
    """
    returns symmetric alpha grid and flip-rate curves for +v and -v directions.
    wrapper on flip_curves_bidir()
    args:
        X       [np.array] (n,H) raw pre-norm activations at last layer
        w_pre   [np.array] (H,)
        c       [float]
        art     [ProbeArtifact] the probe
        v       [np.array] (H,) projected/normalized direction in probe subspace
        max_alpha  [float] for flipping
        clip_pct[float] percentile to clip alpha
    returns:
        [tuple]  with symmetric alphas, flip curve in both directions
    """
    alphas_pos, curve_pos, curve_neg = flip_curves_bidir(X, w_pre, c, art, v, max_alpha, n=n )
    alphas_sym  = np.r_[-alphas_pos[::-1][:-1], alphas_pos]  # e.g., [-A,...,0,...,+A], avoiding two 0's
    curve_sym   = np.r_[curve_neg[::-1][:-1],   curve_pos]
    return alphas_sym, curve_sym


def flip_curve_by_class(X, y, w_pre, c, v, alphas):
    s0 = X @ w_pre + c
    out0, out1 = [], []
    for a in alphas:
        s1 = (X + a*v[None,:]) @ w_pre + c
        flips = np.sign(s0) != np.sign(s1)
        out0.append(float(np.mean(flips[y==0])))
        out1.append(float(np.mean(flips[y==1])))
    return np.array(out0), np.array(out1)


def flip_curve_random_baseline_proj(X, w_pre, c, k, alphas, trials=100, seed=0):
    """
    compute the curve of flipping rates at given alphas, for pseudo-probes made by k
    neurons, chosen randomly
    args:
        X       [np.array] (n,H) raw pre-norm activations at last layer
        w_pre   [np.array] (H,)
        c       [float]
        k       [int] number of neurons in the probe
        alphas  [np.array] (A,) the alphas to use for steering
        trials  [int] number of random steps
        seed    [float]
    returns:
        [tuple]  with mean and std flip curve
    """
    rng     = np.random.default_rng(seed)
    H       = X.shape[1]
    curves  = []
    for _ in range(trials):
        ridx    = rng.choice(H, size=k, replace=False)
        v       = np.zeros(H, np.float32); v[ridx] = w_pre[ridx]
        std     = X[:, ridx].std(axis=0, ddof=1).astype(np.float32) + 1e-12
        scale   = np.sqrt(np.mean((v[ridx]/std)**2)) + 1e-12
        v       /= scale
        curves.append( flip_curve_global_alpha( X, w_pre, c, v, alphas ) )
    curves  = np.stack(curves, axis=0)
    return curves.mean(axis=0), curves.std(axis=0)


def intervention_flip_curve( X, y, w_pre, c, art, n=16 ):
    """
    collect the curve of flip rates at various alphas, compared with random intervention
    """
    H           = X.shape[1]
    v           = pair_steer_dir_in_art( w_pre, art, H )

    # compute the curve filp-rate/alpha for random neurons
    alphas, flip = flip_curve_symmetric(X, w_pre, c, art, v, max_alpha=None, n=n )
    mu, sd = flip_curve_random_baseline_proj(X, w_pre, c, len(art.idx), alphas )
    f0_neg, f1_neg = flip_curve_by_class(X, y, w_pre, c, -v, alphas )  # push toward "no"
    f0_pos, f1_pos = flip_curve_by_class(X, y, w_pre, c, +v, alphas )  # push toward "yes"
    results         = {
        "alphas":       alphas,
        "flip":         flip,
        "random_mu":    mu,
        "random_sd":    sd,
        "f0_neg":       f0_neg,
        "f1_neg":       f1_neg,
        "f0_pos":       f0_pos,
        "f1_pos":       f1_pos,
    }
    df          = pd.DataFrame( results )

    return df



# ===================================================================================================================
#
# main execution functions
#
# ===================================================================================================================


def exec_test_probe( model_name="ll3-8", ptype="stable", df_path=None, do_plot=True ):
    """
    test all probes for a model, looping on all scenarios, for all layers found,
    compute also the probe common to all scenarios, for each layer
    args:
        model_name  [str]
        ptype       [str]
        df_path     [str] path to save the dataset or None
        do_plot     [bool] generate plot as well
    """
    ds, man, model_map, scen_map = load_store( model_name )
    model_code  = model_map[ model_name ]
    scenes      = list( scen_map.keys() ) + [ "common" ]

    # in turn, pick one scenario as training, and the other two as test, pick all three for common probes
    results         = []
    for scen in scenes:
        print( f"{scen}: evaluation..." )
        if scen == "common":
            sub = select_slices( ds )                   # pick all samples
        else:
            train_scen  = scen_map[ scen ]
            test_scen   = list( scen_map.values() )
            test_scen.remove( train_scen )
            sub = select_slices( ds, scen_codes=test_scen )
        layers  = np.array( sub["layer" ] )             # pick all layers
        layers.sort()
        for layer in list( layers ):
            if VERBOSE:
                print( f"layer: {layer}" )
            X,y     = _get_Xy( sub, layer )
            if scen == "common":
                art     = load_probe( model_name, layer, ptype="common", scen='' )
            else:
                art     = load_probe( model_name, layer, ptype=ptype, scen=scen )
            if art is None:
                continue
            drop        = causal_drop( art, X, y )
            auc, acc    = eval_probe( art, X, y )
            flip        = causal_flip( art, X, y )
            d           = { "layer": layer, "model": model_name, "scen": scen }
            d[ "acc" ]              = acc
            d[ "auc" ]              = auc
            d[ "drop_mean" ]        = drop[ "p_mean_drop" ]
            d[ "drop_std" ]         = drop[ "p_std_drop" ]
            d[ "drop_or_mean" ]     = drop[ "p_mean_drop_or" ]
            d[ "drop_or_std" ]      = drop[ "p_std_drop_or" ]
            d[ "entrust_flip" ]     = flip[ "true_1_flip_rate" ]        # entrust flipped to distrust
            d[ "distrust_flip" ]    = flip[ "true_0_flip_rate" ]        # distrust flipped to entrust
            results.append( d )

    df          = pd.DataFrame( results )
    if df_path is not None:
        df.to_pickle( df_path )
    if do_plot:
        basename = os.path.join( dir_plots, f"plt_{model_name}_" )
        dfc      = df[ df["scen"]=="common" ].copy()
        plot.plot_trust_flip( dfc, basename=basename, title=model_name )    # plot of flipping rate per layer

    return df


def plot_inter( model_name="ll3-8" ):
    """
    plot interventionist analysis for the last layer only on a model
    args:
        model_name  [str]
    """
    ds, man, model_map, scen_map = load_store( model_name )
    sub         = select_slices( ds )
    layers      = np.array( sub["layer" ] )             # pick all layers
    layers.sort()
    layer       = layers[ -1 ]
    X,y         = _get_Xy( sub, layer )
    art         = load_probe( model_name, layer, ptype="common", scen='' )
    if art is None:
        return False
    readout     = load_lm_readout_pre( model_name )                     # requires a pre-loaded weight vector
    if readout is None:
        return False
    w_pre, c    = readout
    dfi         = intervention_flip_curve( X, y, w_pre, c, art )
    basename    = os.path.join( dir_plots, f"plt_{model_name}_{layer}_" )
    plot.plot_intervention( dfi, basename=basename, title=model_name )

    return True


def plot_probe( model_name="ll3-8", n_layers=10 ):
    """
    plot single neurons of probes for a model, and on a selection of layers
    args:
        model_name  [str]
        n_layers    [int] number of last layers to process
    """
    ds, man, model_map, scen_map = load_store( model_name )
    model_code  = model_map[ model_name ]
    sub         = select_slices( ds )
    layers      = np.array( sub["layer" ] )             # pick all layers
    layers.sort()
    if len( layers ) < n_layers:   n_layers = 0
    for layer in layers[ -n_layers : ]:
        X,y         = _get_Xy( sub, layer )
        art         = load_probe( model_name, layer, ptype="common", scen='' )
        if art is None:
            continue
        basename    = os.path.join( dir_plots, f"plt_{model_name}_{layer}_" )
        suptitle    = f"layer {layer}"
        plot.mono_plots_for_neurons( X, y, art, basename=basename, suptitle=suptitle )


def plot_proto( model_name="ll3-8" ):
    """
    plot prototype discrimination for a model versus layers
    args:
        model_name  [str]
    """
    fname       = os.path.join( dir_res, f"{model_name}_proto.pkl" )
    df          = pd.read_pickle( fname )
    basename    = os.path.join( dir_plots, f"plt_{model_name}_" )
    plot.plot_prototype_auc_by_layer( df, basename=basename, title=model_name )


def stat_probe( df ):
    """
    extract statistics of the probe results per model on the best layer each
    args:
        df  [pd.DataFrame] as that returned by exec_test_probe, concatenated over models
            necessary columns are layer, model, scen, acc, auc, drop_or_mean
    """

    def infer_total_layers( df ):
        """
        get the number of layers per model, handles 0- or 1-based layer indexing,
        assumes 'model' and 'layer' columns exist.
        args:
            df  [pd.DataFrame]
        returns
            [dict] {model: total_layers}
        """
        out = {}
        for model, g in df.groupby("model"):
            layers = sorted(pd.unique(g["layer"]))
            if len(layers) == 0:
                continue
            # If layers start at 0, total = max + 1; else assume 1-based and use max
            total = int(layers[-1] + 1) if layers[0] == 0 else int(layers[-1])
            out[model] = total
        return out

    def add_rel_depth( df, totals ):
        """
        add the relative depth columns
        args:
            df      [pd.DataFrame]
            totals  [dict] as returned by infer_total_layers()
        returns
            [pd.DataFrame] with the added columns
        """
        out     = df.copy()
        for scen in scenarios:
            col = f"{scen}_layer"
            if col in out:
                new_col = f"{scen}_rel_layer"
                out[ new_col ] = out.apply(
                    lambda r: r[ col ] / totals.get( r[ "model" ], np.nan ), axis=1
                )
        # nice ordering
        def block( s ):
            return [ f"{s}_layer", f"{s}_rel_layer", f"{s}_drop_or_abs", f"{s}_acc" ]

        cols    = ( [c for c in block(s) if c in out.columns] for s in scenarios )
        cols    = ["model"] + sum( cols, [] )
        return out[ cols ]

    # strip the common pseudo-scenario, if there
    if "common" in df[ "scen" ].unique():
        df = df[ df[ "scen" ] != "common" ].copy()

    totals      = infer_total_layers( df )

    df          = df.copy()
    df["drop_or_abs"] = df["drop_or_mean"].abs()

    # best layer per (model, scen) by |oriented drop| ---
    best        = ( df.sort_values( ["model","scen","drop_or_abs"], ascending=[True,True,False] )
              .groupby( ["model","scen"], as_index=False )
              .head(1)[ ["model","scen","layer","drop_or_abs","acc"]] )

    # unstack scenarios -> columns become a MultiIndex (field, scen).
    wide        = ( best.set_index( [ "model", "scen" ] )
                [ [ "layer", "drop_or_abs", "acc" ] ]
                .unstack( "scen" ) )

    # ensure deterministic scen order & flatten MultiIndex columns cleanly
    wide        = wide.sort_index(axis=1, level=1)
    wide.columns = [ f"{sc}_{field}" for field, sc in wide.columns ]  # (field, scen) -> "scen_field"
    wide        = wide.reset_index()
    stat        = add_rel_depth( wide, totals )
    stat        = stat.round(3)
    return stat


def latex_probe_stat( stat: pd.DataFrame,
                        depth_as_pct=False,
                        depth_decimals=1,
                        acc_decimals=3,
                        drop_decimals=3,
                        group_gap="23pt",
                        fname=None) -> str:
    """
    format the statistics returned by stat_probe in LaTeX
    args:
        stat            [pd.DataFrame] expected columns:
                      'model',
                      'fire_rel_layer','fire_acc','fire_drop_or_abs',
                      'farm_rel_layer','farm_acc','farm_drop_or_abs',
                      'school_rel_layer','school_acc','school_drop_or_abs'
        depth_as_pct    [bool] relative depth in percentage
        depth_decimals  [int] decimals for relative depth
        acc_decimals    [int] decimals for ACC
        drop_decimals   [int] decimals for drop
        group_gap       [str] gap between scenarios data
    """
    # formatting helpers
    def fmt_depth(x):
        if pd.isna(x): return ""
        if depth_as_pct: return f"{100*x:.{depth_decimals}f}\\%"
        return f"{x:.{depth_decimals}f}"

    def fmt_acc(x):
        return "" if pd.isna(x) else f"{x:.{acc_decimals}f}"

    def fmt_drop(x):
        return "" if pd.isna(x) else f"{x:.{drop_decimals}f}"

    scen_cols = {
        "fire":   ("fire_rel_layer","fire_acc","fire_drop_or_abs"),
        "farm":   ("farm_rel_layer","farm_acc","farm_drop_or_abs"),
        "school": ("school_rel_layer","school_acc","school_drop_or_abs"),
    }
    # select & order rows
    df = stat.set_index("model").loc[list( curr_models )].reset_index()


    # header
    # column spec: model + (ccc @gap ccc @gap ccc)
    colspec = f"l ccc @{{\\hspace{{{group_gap}}}}} ccc @{{\\hspace{{{group_gap}}}}} ccc"

    # header line (two-tier)
    sub_head    = " & depth & ACC & drop"
    top =  "\\begin{tabular}{" + colspec + "}\n\\toprule\n"
    top += "model & "
    top += "\\multicolumn{3}{c}{fire} & "
    top += "\\multicolumn{3}{c}{farm} & "
    top += "\\multicolumn{3}{c}{school} \\\\\n"
    top += "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}\n"
    top += 3 * sub_head
    top += " \\\\\n\\midrule\n"

    # tail
    tail = "\n\\bottomrule\n\\end{tabular}\n"

    # rows
    lines = []
    for _, r in df.iterrows():
        row = [ r["model"] ]
        for scen in scenarios:
            dcol, acol, tcol = scen_cols[ scen ]
            row += [ fmt_depth( r.get(dcol, float("nan")) ),
                    fmt_acc( r.get(acol, float("nan")) ),
                    fmt_drop( r.get(tcol, float("nan")) )]
        lines.append( " & ".join( row ) + r" \\")
    body = "\n".join( lines )

    tex = top + body + tail

    if fname:
        with open( fname, "w" ) as f:
            f.write( tex )
    return tex

