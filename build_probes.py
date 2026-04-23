"""
#####################################################################################################################

    trust probing project - 2025

    construct prototypes vectors and small probes from activation of a model

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
import  run_model
import  json
import  joblib
from    dataclasses                 import dataclass
from    typing                      import Dict, Any, Optional

from sklearn.metrics                import roc_auc_score, accuracy_score
from sklearn.linear_model           import LogisticRegression
from sklearn.discriminant_analysis  import LinearDiscriminantAnalysis
from sklearn.model_selection        import StratifiedKFold
from sklearn.covariance             import LedoitWolf
from sklearn.preprocessing          import StandardScaler

dir_probe       = "../probes"
zarr_path       = "../res/activation.zarr"
manifest_path   = "../res/manifest.parquet"


# ===================================================================================================================
#
# common utiities
#
# load_store()
# load_store_group()
# select_slices()
# _get_Xy()
# _l2n()
# new_probe_artifact()
# scen_consensus()
# 
# ===================================================================================================================

def load_store():
    """
    legacy access to zarr archive, before using groups
    """
    ds          = xr.open_zarr( zarr_path )
    man         = pd.read_parquet( manifest_path )
    model_map   = (man[["model_code","model_name"]]
                 .drop_duplicates().set_index("model_name")["model_code"].to_dict())
    scen_map    = (man[["scen_code","scen_name"]]
                 .drop_duplicates().set_index("scen_name")["scen_code"].to_dict())
    return ds, man, model_map, scen_map


def load_store_group( group ):
    """
    new access to zarr archive, using  groups
    args:
        group   [str] the group, should be identical to the model short name
    """
    # first check for existence of the specified group
    root        = zarr.open( zarr_path, mode="r" )
    groups      = list( root.group_keys() )
    assert group in groups, f"error in load_store_group: group {group} not in {zarr_path}" 
    # open the data associated with the specified group
    ds          = xr.open_zarr( zarr_path, group=group )
    man         = pd.read_parquet( manifest_path )
    model_map   = (man[["model_code","model_name"]]
                 .drop_duplicates().set_index("model_name")["model_code"].to_dict())
    scen_map    = (man[["scen_code","scen_name"]]
                 .drop_duplicates().set_index("scen_name")["scen_code"].to_dict())
    return ds, man, model_map, scen_map


def select_slices( ds, *, model_code=None, scen_codes=None ):
    """
    select all required slices, for model code and scenarios
    NOTE that the model selection is actually redundant with the new zarr organized by groups
    args:
        ds      [xarray.core.dataset.Dataset] the slice from which to get vectors
        model_code  [int] code of the model
        scen_codes  [list] with [int] codes of scenes
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


def _l2n( X, eps=1e-12 ):
    """
    row-wise L2 normalization, eps avoid division by 0
    """
    n = np.linalg.norm( X, axis=1, keepdims=True ) + eps
    return X / n


@dataclass
class ProbeArtifact:
    """
    object containing a probe, suitable for saving/loading
    NOTE the @dataclass decorator that allows an implicit __init__()
    """
    idx: np.ndarray          # selected feature indices (global, length k)
    w_std: np.ndarray        # LR weights in standardized space (k,)
    b: float                 # LR intercept
    mean: np.ndarray         # StandardScaler.mean_ for selected cols (k,)
    scale: np.ndarray        # StandardScaler.scale_ for selected cols (k,)
    meta: Dict[str, Any]     # anything you want (layer, model, C, AUC, etc.)


def new_probe_artifact(
        idx: np.ndarray,
        scaler,              # fitted StandardScaler on X[:, idx]
        clf,                 # fitted LogisticRegression
        meta: Optional[Dict[str, Any]] = None,
        dtype=np.float32
    ) -> ProbeArtifact:
    """
    create the object containing a probe
    """
    idx = np.asarray(idx, dtype=np.int32)
    return ProbeArtifact(
        idx=idx,
        w_std=clf.coef_.ravel().astype(dtype, copy=True),
        b=float(clf.intercept_[0]),
        mean=scaler.mean_.astype(dtype, copy=True),
        scale=scaler.scale_.astype(dtype, copy=True),
        meta=dict(meta or {})
    )


def scen_consensus( freq_list ):
    """
    find neurons with good frequencies across all scenarios
    args:
        freq_list [list] with (H,) vectors of frequencies for all scenarios
    """
    F       = np.stack( freq_list )
    meanF   = F.mean(0)             # mean stability
    minF    = F.min(0)              # worst-case stability
    votes   = (F >= 0.5).sum(0)     # how many scenarios ≥ 0.5

    # strict core: stable in all three, or relax to 2-of-3 with good mean
    core_strict = np.where(minF >= 0.5)[0]
    core_relax  = np.where((votes >= 2) & (meanF >= 0.45))[0]

    return core_strict, core_relax



# ===================================================================================================================
#
# extraction and evaluation of prototype vectors for all layers
#
# best_threshold()
# eval_proto_train_test()
# eval_maha_train_test()
# eval_lr_train_test()
# eval_lda_train_test()
# run_prototypes()
# 
# ===================================================================================================================

def best_threshold( scores_tr, y_tr ):
    """
    choose threshold that maximizes training accuracy
    """
    order = np.argsort(scores_tr)
    s = scores_tr[order]
    y = y_tr[order]

    # compute candidate thresholds between consecutive scores; include extremes (np.inf)
    edges = np.r_[-np.inf, (s[:-1] + s[1:]) / 2.0, np.inf]      # array of all valid thresholds

    # predict positive if score > thr
    best_acc, best_thr = -1.0, 0.0              # initialize accuracy to worst value, and threshold
 
    # cumulative positives for fast scan
    # number of true positive and false positive when threshold at each edge (predict > thr)
    tp = np.cumsum(y[::-1])[::-1]               # note the double reversal using slicing
    fp = np.cumsum((1 - y)[::-1])[::-1]
    P = y.sum()
    N = (1 - y).sum()
    # map edge i to counts when threshold between s[i-1], s[i]
    for i in range(len(edges)):
        if i == 0:
            # thr=-inf => predict all positive
            acc = (P) / (P + N)
        elif i == len(edges)-1:
            # thr=+inf => predict none positive
            acc = (N) / (P + N)
        else:
            # scores greater than s[i-1] are predicted positive
            # index of first element greater than s[i-1] is i
            tp_i = tp[i-1] if i-1 < len(tp) else 0
            fp_i = fp[i-1] if i-1 < len(fp) else 0
            acc = (tp_i + (N - fp_i)) / (P + N)
        if acc > best_acc:
            best_acc, best_thr = acc, edges[i]
    return best_thr


def eval_proto_train_test( Xtr, ytr, Xte, yte ):
    """
    simple linear classification using prototype vectors for yes/no, and cosine similarity
    args:
        Xtr     [np.array] with shape (n,H) training vectors
        ytr     [np.array] with shape (n,)  training labels
        Xte     [np.array] with shape (n,H) test vectors
        yte     [np.array] with shape (n,)  test labels
    returns:
        [tuple] of:
        auc     [float] R.O.C. A.U.C score
        acc     [float] accuracy score
    """
    Xtr_n   = _l2n( Xtr )
    Xte_n   = _l2n( Xte )

    # compute the yes/no prototypes, and keep them in the unit sphere
    # note the need of keepdims=True for the mean, because _l2n expects two dimensions
    py      = _l2n( Xtr_n[ytr==1].mean(0, keepdims=True) )[0]
    pn      = _l2n( Xtr_n[ytr==0].mean(0, keepdims=True) )[0]

    # now compute cosine similarity of vectors
    s_tr    = Xtr_n @ py - Xtr_n @ pn
    s_te    = Xte_n @ py - Xte_n @ pn

    # derive the best threshold to apply when computing accuracy
    thr     = best_threshold( s_tr, ytr )
    auc     = roc_auc_score( yte, s_te )
    acc     = accuracy_score( yte, (s_te > thr).astype(int) )

    return auc, acc


def eval_maha_train_test( Xtr, ytr, Xte, yte ):
    """
    classification using Mahalanobis distance between prototype vectors for yes/no, and cosine similarity
        d_M( x, mu ) = sqrt( (x-mu)^T Sigma^-1 (x-mu) )
    args:
        Xtr     [np.array] with shape (n,H) training vectors
        ytr     [np.array] with shape (n,)  training labels
        Xte     [np.array] with shape (n,H) test vectors
        yte     [np.array] with shape (n,)  test labels
    returns:
        [tuple] of:
        auc     [float] R.O.C. A.U.C score
        acc     [float] accuracy score
    """
    mu_y = Xtr[ytr==1].mean(0)
    mu_n = Xtr[ytr==0].mean(0)
    cov  = LedoitWolf().fit(Xtr).covariance_    # compute the covariance matrix, with shape (H,H)
    inv  = np.linalg.pinv( cov )                # compute inverse, or pseudoinverse if does not exist

    def md(X, mu):                              # the actual Mahalanobis distance
        d   = X - mu
        mu2 = np.sum( (d @ inv) * d, axis=1 )
        return np.sqrt( mu2 )

    s_tr    = md(Xtr, mu_n) - md(Xtr, mu_y)  # larger => closer to yes
    s_te    = md(Xte, mu_n) - md(Xte, mu_y)
    auc     = roc_auc_score(yte, s_te)
    acc     = accuracy_score(yte, (s_te > 0.0).astype(int))  # <- fixed threshold
    return auc, acc


def eval_lr_train_test(Xtr, ytr, Xte, yte):
    """
    classification using logistic regression
    args:
        Xtr     [np.array] with shape (n,H) training vectors
        ytr     [np.array] with shape (n,)  training labels
        Xte     [np.array] with shape (n,H) test vectors
        yte     [np.array] with shape (n,)  test labels
    returns:
        [tuple] of:
        auc     [float] R.O.C. A.U.C score
        acc     [float] accuracy score
    """
    # first standardize all vector features
    scaler  = StandardScaler().fit( Xtr )
    Xtr_s   = scaler.transform( Xtr )
    Xte_s   = scaler.transform( Xte )

    # now compute a classifier with LogisticRegression on the training data
    clf     = LogisticRegression(max_iter=300, solver="lbfgs", C=1.0)
    clf.fit(Xtr_s, ytr)
    p_tr    = clf.predict_proba(Xtr_s)[:,1]
    p_te    = clf.predict_proba(Xte_s)[:,1]

    # derive the best threshold to apply when computing accuracy
    thr     = best_threshold(p_tr, ytr)
    auc     = roc_auc_score( yte, p_te )
    acc     = accuracy_score( yte, (p_te > thr).astype(int) )
    return auc, acc


def eval_lda_train_test( Xtr, ytr, Xte, yte ):
    """
    classification using linear discriminant analysis
    args:
        Xtr     [np.array] with shape (n,H) training vectors
        ytr     [np.array] with shape (n,)  training labels
        Xte     [np.array] with shape (n,H) test vectors
        yte     [np.array] with shape (n,)  test labels
    returns:
        [tuple] of:
        auc     [float] R.O.C. A.U.C score
        acc     [float] accuracy score
    """
    lda     = LinearDiscriminantAnalysis(solver="svd")
    lda.fit(Xtr, ytr)
    z_tr    = lda.transform(Xtr).ravel()
    z_te    = lda.transform(Xte).ravel()

    # derive the best threshold to apply when computing accuracy
    thr     = best_threshold( z_tr, ytr )
    auc     = roc_auc_score( yte, z_te )
    acc     = accuracy_score( yte, (z_te > thr).astype(int) )
    return auc, acc


def run_prototypes( ds, mcode, tr_code, te_codes, skip=( "maha", "lda" ) ):
    """
    run all evaluations of prototype vectors over layers, with separated training and testing scenarios
    args:
        ds       [xarray.core.dataset.Dataset]
        mcode    [int] model code
        tr_code  [int] code of the scenario used for training
        te_codes [list] codes of the scenarios used for test
        skip     [list] of methods to skip, possible values "proto", "maha", "lr", "lda"
    """
    # slices
    sub_tr = select_slices(ds, model_code=mcode, scen_codes=[tr_code])
    sub_te = select_slices(ds, model_code=mcode, scen_codes=te_codes)

    rows = []
    for k in sub_tr.layer.values.tolist():
        row     = { "layer": int(k) }
        Xtr, ytr = _get_Xy(sub_tr, k)
        Xte, yte = _get_Xy(sub_te, k)
        if not "proto" in skip:
#           print( f"layer {k:2d} - doing eval_proto_train_test..." )
            auc, acc = eval_proto_train_test(Xtr, ytr, Xte, yte)
            row[ "AUC_proto" ]  = auc
            row[ "ACC_proto" ]  = acc
        if not "maha" in skip:
#           print( f"layer {k:2d} - doing eval_maha_train_test..." )
            auc, acc = eval_maha_train_test(Xtr, ytr, Xte, yte)
            row[ "AUC_maha" ]   = auc
            row[ "ACC_maha" ]   = acc
        if not "lr" in skip:
#           print( f"layer {k:2d} - doing eval_lr_train_test..." )
            auc, acc = eval_lr_train_test(Xtr, ytr, Xte, yte)
            row[ "AUC_LR" ]     = auc
            row[ "ACC_LR" ]     = acc
        if not "lda" in skip:
#           print( f"layer {k:2d} - doing eval_lda_train_test..." )
            auc, acc = eval_lda_train_test(Xtr, ytr, Xte, yte)
            row[ "AUC_LDA" ]    = auc
            row[ "ACC_LDA" ]    = acc

        rows.append( row )

    df = pd.DataFrame(rows)
    return df



# ===================================================================================================================
# 
# looking for neurons with sparse probing
#
# topk_by_effectsize()
# sparse_lr_path()
# pick_min_k()
# stability_selection()
# train_lr_subset()
# eval_lr_subset()
#
# 
# ===================================================================================================================

def topk_by_effectsize(Xtr, ytr, k):
    """
    rank neurons by their effect size in classifying yes/no, and take the best k ones
    args:
        Xtr     [np.array] with shape (n,H) training vectors
        ytr     [np.array] with shape (n,)  training labels
        k       [int] number of best neurons
    returns:
        [np.array] of [int]
    """
    pos     = Xtr[ytr==1]
    neg     = Xtr[ytr==0]
    mu_p    = pos.mean(0)
    mu_n    = neg.mean(0)
    var_p   = pos.var(0, ddof=1)
    var_n   = neg.var(0, ddof=1)
    pooled  = np.sqrt(0.5*(var_p + var_n) + 1e-12)
    d       = np.abs((mu_p - mu_n) / (pooled + 1e-12))
    return np.argsort( -d )[ :k ]

def sparse_lr_path(Xtr, ytr, Xte, yte,
        C_grid=None,
        k_cap=50,
        delta_auc_eps=1e-3,
        patience=3,
        max_iter=5000,
        class_weight=None):
    """
    find sparse representations using logistic regression with L1 norm and regularization C,
    that, for small values, drive many coefficients of the weight vector to zero
    args:
        Xtr             [np.array] with shape (n,H) training vectors
        ytr             [np.array] with shape (n,)  training labels
        Xte             [np.array] with shape (n,H) test vectors
        yte             [np.array] with shape (n,)  test labels
        C_grid          [np.array] with values of regularizer C
        k_cap           [int] number of best neurons that starts to be too large
        delta_auc_eps   [float] minimum improvement of AUC over loops in LogisticRegression
        patience        [int] allowed number of loops with no significant improvement of AUC
        max_iter        [int] allowed number of interations in LogisticRegression
    returns:
        [list] of [dict]
    """
    if C_grid is None:
        C_grid = np.logspace( -3, 1, 20 )   # values to be used for C
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    best_auc = -1.0                         # used for early stopping logic
    stall = 0                               # used for early stopping logic0
    rows = []
    clf_prev = None                         # provides the previous 
    for C in C_grid:
        clf = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=C,
            warm_start=True,
            max_iter=max_iter,
            tol=1e-5,
            class_weight=class_weight
        )
        if clf_prev is not None:
            clf.coef_ = clf_prev.coef_.copy()
            clf.intercept_ = clf_prev.intercept_.copy()
        clf.fit(Xtr_s, ytr)
        w = clf.coef_.ravel()
        idx = np.flatnonzero(w != 0.0)
        k = idx.size
        p_te = clf.predict_proba(Xte_s)[:,1]
        auc = roc_auc_score(yte, p_te)
        acc = accuracy_score(yte, (p_te > 0.5).astype(int))
        rows.append({"C": C, "k": int(k), "AUC": auc, "ACC": acc, "idx": idx, "scaler": scaler, "clf": clf})
        # early stopping logic
        if auc > best_auc + 1e-12:
            stall, best_auc = 0, auc
        else:
            stall += 1
        if k > k_cap and stall >= patience and (best_auc - auc) < delta_auc_eps:
            break
        if k > 2 * k_cap:
            break
        clf_prev = clf                      # NOTE: even if this is a reference assignement, works because of warm_start

    rows = sorted(rows, key=lambda r: (r["k"], r["C"]))
    return rows


def pick_min_k( rows, eps=0.01 ):
    """
    takes the minimum number of neurons which AUC is within eps of the best one
    args:
        rows    [list] of [dict], as returned by sparse_lr_path()
    returns:
        [int]
    """
    best_auc = max(r["AUC"] for r in rows)
    candidates = [r for r in rows if r["AUC"] >= best_auc - eps]
    return min(candidates, key=lambda r: (r["k"], -r["AUC"]))


def stability_selection( Xtr, ytr, C,
        scaler=None,
        n_repeats=50,
        subsample=0.5,
        random_state=0,
        class_weight=None):
    """
    check for stable neurons:
    refit the same LogisticRegression of the current best subvector on many random subsamples
    and count how often each neuron gets a non-zero weight
    args:
        Xtr             [np.array] with shape (n,H) training vectors
        ytr             [np.array] with shape (n,)  training labels
        C               [float] with the regularizer of the best found vector
        scaler          [StandardScaler()] the same used when finding the best vector
        n_repeats       [int] how many subsets to test
        subsample       [float] fraction of neurons in the subsets
        random_state    [int] random seed
    returns:
        [np.array] (H,) of frequencies
    """
    rng = np.random.default_rng( random_state )
    if scaler is None:
        scaler = StandardScaler().fit(Xtr)
    Xs = scaler.transform(Xtr)
    n, H = Xs.shape
    freq = np.zeros( H, dtype=np.float32 )  # frequencies initialized to 0
    m = int(max(2, round(subsample * n)))   # size of the subsamples
    C   = C * ( n / m )                     # make C a bit larger than the original best one
    # pick random subsets of neurons and fit them
    for _ in range(n_repeats):
        idx = rng.choice(n, size=m, replace=False)
        clf = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=2000, class_weight=class_weight)
        clf.fit(Xs[idx], ytr[idx])
        freq += (clf.coef_.ravel() != 0.0)  # counts how many time a neuron is included in the fit
    freq /= n_repeats                       # turn counts into frequencies
    return freq


def pick_stable_indices_target(
    freq: np.ndarray,
    pi_grid: np.ndarray | None = None,
    target_k: int = 6,
    k_tol: int = 1,              # if you want to accept k in [target_k, target_k + k_tol] without trimming
    topk_fallback: int = 16,     # used if no pi hits target_k
):
    """
    Choose a stable subset by frequency thresholding toward a *target* size.

    Strategy:
      1) Sweep pi from high->low; find the largest pi with k = #(freq>=pi) >= target_k.
      2) From those indices, keep the top-`target_k` by frequency (ties broken by index).
      3) If no pi reaches target_k but some k>0, take the highest-pi non-empty set and
         keep top-min(k, target_k).
      4) If freq is all zeros, return the topK by raw freq (which will all be 0),
         or a single best index if K=0.

    Returns
    -------
    idx : np.ndarray[int]   # selected indices (len == target_k whenever possible)
    pi_used : float         # the threshold pi that produced the pool (nan if fallback)
    k_raw : int             # raw count before trimming at the chosen pi
    """
    if pi_grid is None:
        pi_grid = np.linspace(0.8, 0.1, 20)  # 0.80, 0.75, ..., 0.30

    order       = np.argsort( -freq )        # descending by frequency
    best_idx    = None
    best_pi     = np.nan
    best_k_raw  = 0

    # Pass 1: find the tightest (largest) pi with >= target_k items
    for pi in pi_grid:
        idx     = np.flatnonzero( freq >= pi )
        k       = len( idx )
        if k >= target_k:
            # enforce exact target_k via top-k by frequency among those passing the threshold
            # (we keep stable AND most frequent)
            if k_tol > 0 and k <= target_k + k_tol:
                chosen      = idx  # accept as-is within tolerance
            else:
                # rank the passing indices by freq (desc), then take top target_k
                idx_sorted  = idx[ np.argsort( -freq[idx], kind="stable" ) ]
                chosen      = idx_sorted[ :target_k ]
            return np.asarray(chosen, dtype=int), float(pi), int(k)

        # track the best non-empty high-pi candidate in case we never hit target_k
        if k > 0 and best_idx is None:
            best_idx, best_pi, best_k_raw = idx, float(pi), int(k)

    # Pass 2: fallback to highest-pi non-empty set (trim or pad)
    if best_idx is not None:
        idx_sorted      = best_idx[ np.argsort(-freq[best_idx], kind="stable") ]
        k_keep          = min( len( idx_sorted ), target_k )
        return np.asarray( idx_sorted[:k_keep], dtype=int ), best_pi, int(len(best_idx) )

    # Pass 3: ultimate fallback — take topK by freq globally (even if all zeros)
    nz = order[freq[order] > 0]
    if nz.size > 0:
        k_keep          = min( target_k, nz.size, topk_fallback )
        return np.asarray( nz[:k_keep], dtype=int ), float("nan"), int( nz.size )

    # Everything is zero: just return the single top index
    return np.asarray( [ int(order[0])], dtype=int ), float( "nan" ), 1


def train_lr_subset( Xtr, ytr, idx, class_weight=None ):
    """
    train with logistic regression a subset of neurons, so to return the classifier with
    only the necessary weights
    args:
        Xtr     [np.array] with shape (n,H) training vectors
        ytr     [np.array] with shape (n,)  training labels
        idx     [np.array] with indexes of the subset of neurons
    returns:
        [tuple] of:
        scaler  [StandardScaler]
        clf     [LogisticRegression]
    """
    if len(idx) == 0:
        raise ValueError("Subset has 0 features; relax stability threshold or increase effect_top_k.")
    scaler      = StandardScaler().fit( Xtr[:, idx] )      # fit on train once
    Xtr_s       = scaler.transform( Xtr[:, idx] )
    clf         = LogisticRegression( max_iter=1000, solver="lbfgs", class_weight=class_weight )
    clf.fit( Xtr_s, ytr )
    return scaler, clf


def fit_pooled_probe(ds, scen_map, layer_k, idx, model_code=2, class_weight=None):
    """
    train with logistic regression a common subset of neurons on all scenarios, and construct
    a probe object
    """
    scen_names  = list( scen_map.keys() )
    # 1) pool data across scenarios
    Xs, ys, ws = [], [], []
    n_per_s = []
    for s in scen_names:
        scen_codes  = [ scen_map[s] ]
        sub         = select_slices( ds, model_code=model_code, scen_codes=scen_codes )
        X, y        = _get_Xy( sub, layer_k )
        Xs.append(X); ys.append(y); n_per_s.append(len(y))
    X_all = np.vstack(Xs); y_all = np.concatenate(ys)
    # scenario-balanced sample weights
    n_total = sum(n_per_s)
    w_per_s = [ (1.0/len(scen_names))/n for n in n_per_s ]  # each scenario gets total weight 1/S
    offset = 0
    for n,w in zip(n_per_s, w_per_s):
        ws.append(np.full(n, w, dtype=np.float32))
        offset += n
    w_all = np.concatenate(ws)

    # 2) fit scaler and LR on the selected columns
    idx = np.asarray(idx, dtype=int)
    scaler = StandardScaler().fit(X_all[:, idx])
    Xs_all = scaler.transform(X_all[:, idx])

    clf = LogisticRegression(
        penalty="l2", solver="lbfgs",
        C=1.0, max_iter=5000, tol=1e-6, class_weight=class_weight
    )
    clf.fit(Xs_all, y_all, sample_weight=w_all)

    # 3) build ProbeArtifact (as per our earlier helper)
    art = new_probe_artifact(
        idx=idx, scaler=scaler, clf=clf,
        meta={
            "layer": layer_k,
            "k": int(len(idx)),
            "penalty": "l2",
            "C": 1.0,
            "scenarios": scen_names,
            "sample_weight": "scenario-balanced",
        }
    )
    return art


def eval_lr_subset(Xtr, ytr, Xte, yte, idx, class_weight=None):
    """
    evaluate the logistic regression on a subset of neurons on the training set
    args:
        Xtr     [np.array] with shape (n,H) training vectors
        ytr     [np.array] with shape (n,)  training labels
        Xte     [np.array] with shape (n,H) test vectors
        yte     [np.array] with shape (n,)  test labels
        idx     [np.array] with indexes of the subset of neurons
    returns:
        [tuple] of:
        auc     [float] R.O.C. A.U.C score
        acc     [float] accuracy score
    """
    if len(idx) == 0:
        raise ValueError("Subset has 0 features; relax stability threshold or increase effect_top_k.")
    scaler, clf = train_lr_subset( Xtr, ytr, idx, class_weight=class_weight )
    Xte_s   = scaler.transform(Xte[:, idx])
    p       = clf.predict_proba(Xte_s)[:,1]
    auc     = roc_auc_score( yte, p )
    acc     = accuracy_score( yte, (p > 0.5 ).astype(int) )
    return auc, acc



# ===================================================================================================================
# 
# causal drop in discrimination for a non-final layer
# 
# ===================================================================================================================

def margins( scaler, clf, X, idx ):
    """
    full margin = w·x + b (b cancels in differences but keep it for completeness)
    """
    Xs      = scaler.transform( X[:, idx] )
    return (Xs @ clf.coef_.ravel()) + float( clf.intercept_[0] )


def causal_drop_lr_partial(Xtr, ytr, Xte, yte, idx_sel, s=None, n_random=200, seed=0, class_weight=None):
    """
    compute the drop in discrimination caused by the alation of probe's neurons, for comparision
    compute also a mean drop due to random ablation of same number of neurons in the layer
    """
    rng         = np.random.default_rng(seed)
    k           = len( idx_sel )
    if s is None:
        s       = k
    scaler, clf = train_lr_subset(Xtr, ytr, idx_sel, class_weight=class_weight)
    w           = clf.coef_.ravel()
    order       = np.argsort(-np.abs(w))   # top-|w| dims within the selected set
    top         = order[:s]

    # baseline margins
    m_before    = margins(scaler, clf, Xte, idx_sel)
    y_or        = (yte * 2 - 1)

    # ablate TOP-s (mean ablation in standardized space)
    Xs      = scaler.transform(Xte[:, idx_sel])
    Xa      = Xs.copy()
    Xa[:, top] = 0.0
    ma      = (Xa @ w) + float(clf.intercept_[0])
    d_sel   = m_before - ma
    res_sel = {
        "k_total": k, "s": s,
        "mean_drop_sel": float(np.mean(d_sel)),
        "mean_drop_sel_oriented": float(np.mean(y_or * d_sel)),
        "flip_rate_sel": float(np.mean(np.sign(m_before) != np.sign(ma))),
    }

    # RANDOM-s ablations within the selected set
    drops_rand, drops_rand_or, flips_rand = [], [], []
    for _ in range(n_random):
        pick    = rng.choice(k, size=s, replace=False)
        Xr      = Xs.copy()
        Xr[:, pick] = 0.0
        mr      = (Xr @ w) + float(clf.intercept_[0])
        d       = m_before - mr
        drops_rand.append(np.mean(d))
        drops_rand_or.append(np.mean(y_or * d))
        flips_rand.append(np.mean(np.sign(m_before) != np.sign(mr)))

    res_rand = {
        "mean_drop_rand": float(np.mean(drops_rand)),
        "mean_drop_rand_oriented": float(np.mean(drops_rand_or)),
        "flip_rate_rand": float(np.mean(flips_rand)),
    }
    return {**res_sel, **res_rand}


def per_neuron_contrib(Xtr, ytr, Xte, yte, idx_sel):
    """
    compute the discrimination contribution for each single neuron in a probe
    args:
        Xtr     [np.array] with shape (n,H) training vectors
        ytr     [np.array] with shape (n,)  training labels
        Xte     [np.array] with shape (n,H) test vectors
        yte     [np.array] with shape (n,)  test labels
        idx_sel [np.array] with indexes of the subset of neurons
    returns:
        [dict] with neuron index as key and contribution as value
    """
    scaler, clf = train_lr_subset(Xtr, ytr, idx_sel)
    w = clf.coef_.ravel()
    b = float(clf.intercept_[0])
    Xs = scaler.transform(Xte[:, idx_sel])
    base = Xs @ w + b
    y_or = (yte * 2 - 1)

    contrib = dict()
    for j in range(len(idx_sel)):
        Xa = Xs.copy(); Xa[:, j] = 0.0
        mj = Xa @ w + b
        drop = base - mj
        contrib[ idx_sel[j] ]   = float( np.mean(y_or * drop) )
    return contrib


# ===================================================================================================================
# 
# greedy search of probes taking into account all scenarios
# 
# ===================================================================================================================


def baseline_auc_L2(train_X, train_y, test_X, test_y, C=1.0):
    # L2 logistic on *all* dims -> strong, scenario-agnostic baseline
    scaler = StandardScaler().fit(train_X)
    trS = scaler.transform(train_X); teS = scaler.transform(test_X)
    clf = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000, C=C)
    clf.fit(trS, train_y)
    p = clf.predict_proba(teS)[:,1]
    return roc_auc_score(test_y, p)

def auc_subset(split, idx):
    Xtr, ytr, Xte, yte = split
    ss = StandardScaler().fit(Xtr[:, idx])
    trS = ss.transform(Xtr[:, idx]); teS = ss.transform(Xte[:, idx])
    clf = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000)
    clf.fit(trS, ytr)
    p = clf.predict_proba(teS)[:,1]
    return roc_auc_score(yte, p)

# --- main greedy builder ---
def greedy_consensus_subset_layer(
    ds,
    scen_map,
    model_code=2,
    layer_k=32,
    consensus_pool=None,       # list/array of candidate neuron IDs (global indices)
    seed=None,                 # starting set (e.g., core_relax)
    tol=0.01,                  # within 0.01 AUC of baseline for each split
    C_baseline=1.0,            # L2 strength for baseline LR
):
    scen_names      = scen_map.keys()
    # assume single model code in dataset; if multiple, filter appropriately
    # build splits: train on s, test on others
    splits = []
    baselines = []
    for i, s_tr in enumerate(scen_names):
        print( f"{s_tr}: processing..." )
        train_scen  = scen_map[ s_tr ]
        test_scen   = list( scen_map.values() )
        test_scen.remove( train_scen )
        # load data & split between training ant testing
        sub_tr      = select_slices( ds, model_code=model_code, scen_codes=[train_scen])
        sub_te      = select_slices( ds, model_code=model_code, scen_codes=test_scen)
        Xtr, ytr    = _get_Xy(sub_tr, layer_k)
        Xte, yte    = _get_Xy(sub_te, layer_k)
        splits.append((Xtr, ytr, Xte, yte))
        # baseline AUC on all dims
        baselines.append(baseline_auc_L2(Xtr, ytr, Xte, yte, C=C_baseline))
    baselines = np.array(baselines)

    pool = np.array(sorted(set(consensus_pool)), dtype=int)
    chosen = [] if seed is None else list(seed)
    remaining = [j for j in pool if j not in chosen]

    history = []
    def current_aucs(idx_list):
        if len(idx_list) == 0:
            return np.zeros(len(splits))
        return np.array([auc_subset(splits[k], np.array(idx_list, dtype=int)) for k in range(len(splits))])

    # start status
    aucs = current_aucs(chosen)
    history.append({"k": len(chosen), "chosen": chosen.copy(), "aucs": aucs.copy()})

    # greedy loop
    while True:
        gaps = baselines - aucs
        if np.all(gaps <= tol):
            break
        # evaluate each candidate by worst-case AUC if added
        best_c, best_minauc, best_aucs = None, -1.0, None
        for c in remaining:
            trial = chosen + [c]
            aucs_c = current_aucs(trial)
            minauc = aucs_c.min()  # maximin criterion
            if minauc > best_minauc + 1e-12:
                best_c, best_minauc, best_aucs = c, minauc, aucs_c
        if best_c is None:
            # no improvement (shouldn't happen often)
            break
        chosen.append(best_c)
        remaining.remove(best_c)
        aucs = best_aucs
        history.append({"k": len(chosen), "chosen": chosen.copy(), "aucs": aucs.copy()})

    report = {
        "layer": layer_k,
        "baselines": baselines,
        "tol": tol,
        "final_k": len(chosen),
        "final_idx": np.array(chosen, dtype=int),
        "final_aucs": aucs,
        "history": history,
    }
    return report


# ===================================================================================================================
# main execution functions
#
# sparse_probe_runner()
# exec_prototypes()
# exec_build_probe()
# ===================================================================================================================

def sparse_probe_runner(
        ds,
        model_code,
        train_scen,
        test_scen,
        layer_k,
        effect_top_k        = None,
        C_grid              = None,
        l1_eps              = 0.01,
        stability_repeats   = 30,
        subsample           = 0.7,
        min_k               = 2,
        target_k            = 6,
        class_weight        = None
    ):
    """
    wrapper on the functions searching for a small probe for the specified layer
    args:
        ds                  [xarray.core.dataset.Dataset]
        model_code          [int] model code
        train_scen          [int] code of training scenario
        test_scen           [list] with int codes of testing scenarios
        layer_k             [int] the requested layer
        effect_top_k        [int] to prescreen neurons, None = use all neurons
        C_grid              [np.array] with regularization values to probe, or None
        l1_eps              [float] the margin with respect to the best accuracy to pick a minimal subset of neurons
        stability_repeats   [int] number of repetitions of random training subsets in stability_selection()
        subsample           [float] fraction of samples over the total, to be used in stability_selection()
        min_k               [int] minimum number of neurons in the stability selection
        target_k            [int] target number of neurons in the stability selection
        class_weight        [str] e.g., "balanced" if classes skewed
    returns:
        [dict] with all results of the search
    """
    # load data & split between training ant testing
    sub_tr          = select_slices( ds, model_code=model_code, scen_codes=[train_scen])
    sub_te          = select_slices( ds, model_code=model_code, scen_codes=test_scen)
    Xtr, ytr        = _get_Xy(sub_tr, layer_k)
    Xte, yte        = _get_Xy(sub_te, layer_k)

    # optional pre-screen by effect size a number of neurons
    # if used, the numbering of neurons will be altered, and idx_sel allows recovering
    # the original indeces
    H               = Xtr.shape[1]
    idx_sel         = np.arange(H)                              # by default use all neurons
    if effect_top_k is not None and effect_top_k < H:
        idx_sel     = topk_by_effectsize(Xtr, ytr, k=effect_top_k)
        Xtr         = Xtr[:, idx_sel]                              # selection of neurons
        Xte         = Xte[:, idx_sel]                              # selection of neurons

    # sparse search by L1 linear regression
    rows            = sparse_lr_path(Xtr, ytr, Xte, yte, C_grid=C_grid, class_weight=class_weight)
    best            = pick_min_k(rows, eps=l1_eps)          # minimal-k within eps of best AUC
    idx_l1_reduced  = best["idx"]
    idx_l1_full     = idx_sel[idx_l1_reduced]              # map back to full feature indices

    # stability selection in the neighborhood of the best C, computing the frequency of neurons
    # being discriminative over random subsets of the training data
    freq            = stability_selection( Xtr, ytr,
                            C=best["C"],
                            scaler=best["scaler"],
                            n_repeats=stability_repeats,
                            subsample=subsample,
                            random_state=0,
                            class_weight=class_weight
                    )
    idx_stable_reduced, pi_used, _ = pick_stable_indices_target( freq, target_k=target_k )
    idx_stable_full = idx_sel[idx_stable_reduced]

    # test the subsets of neurons found so far
    sc, clf         = train_lr_subset(Xtr, ytr, idx_l1_full )
    l1_art          = new_probe_artifact( idx_l1_full, sc, clf, meta={"type":"l1"} )
    sc, clf         = train_lr_subset(Xtr, ytr, idx_stable_full )
    stable_art      = new_probe_artifact( idx_stable_full, sc, clf, meta={"type":"stable"} )

    # test the subsets of neurons found so far
    auc_l1, acc_l1  = eval_lr_subset( Xtr, ytr, Xte, yte, idx_l1_full, class_weight=class_weight )
    auc_st, acc_st  = eval_lr_subset( Xtr, ytr, Xte, yte, idx_stable_full, class_weight=class_weight )

    # causal (non-final layer) effect using the logistic margin axis
    # more causal analysis done with analysis.py
    cd_l1           = causal_drop_lr_partial( Xtr, ytr, Xte, yte, idx_l1_full, s=None )
    cd_st           = causal_drop_lr_partial( Xtr, ytr, Xte, yte, idx_stable_full, s=1 )

    per_neuron      = per_neuron_contrib( Xtr, ytr, Xte, yte, idx_l1_full )

    # 7) report
    report = {
        "n_features_total": int(H),
        "L1_min_k": int(len(idx_l1_full)),
        "L1_min_k_AUC/ACC": (float(auc_l1), float(acc_l1)),
        "Stable_pi": float(pi_used),
        "Stable_k": int(len(idx_stable_full)),
        "Stable_AUC/ACC": (float(auc_st), float(acc_st)),
        "C_best": float(best["C"]),
        "freq": freq,
        "Effect_top_k": int(effect_top_k) if effect_top_k is not None else None,
        "CausalDrop_L1": cd_l1,
        "CausalDrop_Stable": cd_st,
        "Frontier_top5": [
            {k: (float(r[k]) if k in ("AUC","ACC") else r[k]) for k in ("k","AUC","ACC","C")}
            for r in sorted(rows, key=lambda r: (-r["AUC"], r["k"]))[:5]
        ],
        "Idx_L1": idx_l1_full,          # numpy array of selected neuron indices
        "Idx_Stable": idx_stable_full,  # numpy array of selected sdtable neuron indices
        "L1_probe": l1_art,             # probe artifact for L1
        "Stable_probe": stable_art,     # probe artifact for stable
        "Per_Neuron": per_neuron,       # contributions of discrimination per neuron
    }
    return report


def common_probe( ds, scen_map, freq, idxs, model_code, layer_k ):
    """
    compute the common probe for a layer, using the frequencies
    args:
    args:
        ds          [xarray.core.dataset.Dataset] the slice from which to get vectors
        scen_map    [dict] with scenarios names -> code
        freq        [list] with 3 frequencies lists for each scenario, at layer_k layer
        idxs        [list] with neuron indeces collected for probes trained on each scenario
        model_code  [int] code of the model
        layer_k     [int]
    """
    consensus_pool  = list( set( idxs ) )
    _, core_relax   = scen_consensus( freq )
    consensus       = greedy_consensus_subset_layer( ds, scen_map,
            model_code=model_code,
            layer_k=layer_k,
            consensus_pool=consensus_pool,
            seed=core_relax,
            tol=0.001
    )
    idx             = consensus[ "final_idx" ]
    if len( idx ):
        art         = fit_pooled_probe( ds, scen_map, layer_k, idx, model_code=model_code )
        return art
    print( "Warning: no common probe found" )
    return None


def exec_prototypes( model_name="ll3-8", with_group=True ):
    """
    execute the prototype evaluation for a model, looping on all scenarios
    args:
        model_name  [str]
        save        [bool]
        with_group  [bool] when False use legacy zarr format (exists for ll3-8 only),
                    if True use the new zarr organized by groups, that correspond to models
    """
    if with_group:
        ds, man, model_map, scen_map = load_store_group( model_name )
    else:
        ds, man, model_map, scen_map = load_store()
    model_code  = model_map[ model_name ]


    # in turn, pick one scenario as training, and the other two as test
    dfs             = []
    for scen in scen_map.keys():
        print( f"{scen}: evaluation..." )
        train_scen  = scen_map[ scen ]
        test_scen   = list( scen_map.values() )
        test_scen.remove( train_scen )
        df          = run_prototypes( ds, model_code, train_scen, test_scen )
        df["scen"]  = scen
        dfs.append( df )

    df_proto    = pd.concat( dfs, ignore_index=True )

    return df_proto


def exec_build_probe( model_name="ll3-8", layers=(28,32), save=True, with_group=True ):
    """
    execute the probe search and evaluation for a model,
    looping on all scenarios, and on a selection of layers
    compute also the probe common to all scenarios, for each layer
    args:
        model_name  [str]
        layers      [tuple]
        save        [bool]
        with_group  [bool] when False use legacy zarr format (exists for ll3-8 only),
                    if True use the new zarr organized by groups, that correspond to models
    """
    if with_group:
        ds, man, model_map, scen_map = load_store_group( model_name )
    else:
        ds, man, model_map, scen_map = load_store()
    model_code  = model_map[ model_name ]

    def from_report( r ):
        """
        helper extracting minimal information from a report
        """
        d                   = dict()
        d[ "scen" ]         = scen
        d[ "layer_k" ]      = layer_k
        d[ "L1_idx" ]       = r[ "Idx_L1" ]
        d[ "L1_acc" ]       = r[ "L1_min_k_AUC/ACC" ][ 1 ]
        d[ "L1_drop" ]      = r[ "CausalDrop_L1" ][ "mean_drop_sel_oriented" ]
        d[ "L1_flip" ]      = r[ "CausalDrop_L1" ][ "flip_rate_sel" ]
        d[ "Stable_idx" ]   = r[ "Idx_Stable" ]
        d[ "Stable_acc" ]   = r[ "Stable_AUC/ACC" ][ 1 ]
        d[ "Stable_drop" ]  = r[ "CausalDrop_L1" ][ "mean_drop_sel_oriented" ]
        d[ "Stable_flip" ]  = r[ "CausalDrop_Stable" ][ "flip_rate_sel" ]
        return d

    def do_save( r ):
        """
        helper for saving probes
        """
        l1_art                          = r[ "L1_probe" ]
        stable_art                      = r[ "Stable_probe" ]
        l1_art.meta[ "scen" ]           = scen
        l1_art.meta[ "model" ]          = model_name
        l1_art.meta[ "layer_k" ]        = layer_k
        stable_art.meta[ "scen" ]       = scen
        stable_art.meta[ "model" ]      = model_name
        stable_art.meta[ "layer_k" ]    = layer_k
        fname                           = f"l1_{model_name}_{scen}_{layer_k}.joblib"
        path                            = os.path.join( dir_probe, fname )
        joblib.dump(l1_art, path)
        fname                           = f"stable_{model_name}_{scen}_{layer_k}.joblib"
        path                            = os.path.join( dir_probe, fname )
        joblib.dump(stable_art, path)


    # in turn, pick one scenario as training, and the other two as test
    dfs             = []
    probes          = []
    freqs           = dict()
    idxs            = dict()
    for l in layers:
        freqs[ l ]  = []
        idxs[ l ]   = []
    for scen in scen_map.keys():
        print( f"{scen}: evaluation..." )
        train_scen  = scen_map[ scen ]
        test_scen   = list( scen_map.values() )
        test_scen.remove( train_scen )
        for layer_k in layers:
            print( f"{scen}: probing layer {layer_k}" )
            report  = sparse_probe_runner( ds,
                model_code          = model_code,
                train_scen          = train_scen,
                test_scen           = test_scen,
                layer_k             = layer_k
            )
            freqs[ layer_k ].append( report[ "freq" ] )
            idxs[ layer_k ]         += list( report[ "Idx_Stable" ] )
            
            probes.append( from_report( report ) )
            if save:
                do_save( report )

    for l in layers:
        art     = common_probe( ds, scen_map, freqs[ l ], idxs[ l ], model_code, l )
        if art is not None:
            fname   = f"common_{model_name}_{l}.joblib"
            path    = os.path.join( dir_probe, fname )
            joblib.dump( art, path )

    df_probe    = pd.DataFrame( probes )

    return df_probe
