"""

functions for plotting results

"""

import  os
import  pickle
import  numpy               as np
import  copy
import  colorsys
import  matplotlib.pyplot   as plt
import  matplotlib.transforms as mtransforms
from    matplotlib          import rcParams
from    matplotlib.patches  import Patch, Polygon
from    matplotlib.lines    import Line2D
from    matplotlib.ticker   import FuncFormatter
from    matplotlib.cm       import ScalarMappable
from    matplotlib.colors   import PowerNorm, LinearSegmentedColormap, BoundaryNorm, ListedColormap
from    sklearn.metrics     import roc_auc_score, roc_curve
from    math                import sqrt

bfigsize        = ( 18.0, 6.0 )                         # figure size for bar plots
ffigsize        = ( 12.0, 5.0 )                         # figure size for flip bars
rfigsize        = ( 14.0, 8.0 )                         # figure size for radar plot
labelspacing    = 1.1
extension       = ".pdf"

# the elegant tableau-10 series of colors in matplotlib
# NOTE: there are - of course - just 10 colors
tab_colors  = [ 'tab:blue', 'tab:orange', 'tab:green', 'tab:red',
          'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
          'tab:olive', 'tab:cyan' ]

# markers
scen_marker = ( 'o',  '*',  'D',  'P',  'X',  'h', 'H', '<', '>', 'x' )
entrust_col = '#1b5e20'
distrust_col= '#b91c1c'
line_style  = [ '-', '--', '-', '--' ]

xlabel_rot  = 45                                        # X labels rotation, used in multiple runs box plots
char_len    = 0.007                                     # typical length of a character in legend, in plot units



# ===================================================================================================================
#   plotting functions
#
# ===================================================================================================================

def best_threshold_1d(s, y):
    """Return threshold t* maximizing J = TPR - FPR (Youden)"""
    fpr, tpr, thr = roc_curve(y, s)
    J = tpr - fpr
    j = np.argmax(J)
    t_star = thr[j]
    return float(t_star)

def per_neuron_contrib_arrays(X, y, art):
    # X: (n,H) raw; art: ProbeArtifact
    idx = art.idx.astype(int)
    Z   = (X[:, idx] - art.mean) / (art.scale + 1e-12)      # (n,k)
    C   = Z * art.w_std                                     # (n,k) contributions
    return C, y.astype(int), idx

def plot_mono_hist( ax, s, y, bins=40, title="", fsz=12, show_ylabel=False ):
    """
    histograms of activation for entrust/distrust samples of a single neuron,
    oriented with entrust to the right
    args:
        ax          [matplotlib.axes._axes.Axes]
        s           [np.array] (n, 1) raw pre-norm activations for a single neuron
        y           [np.array] (n,) class labels per sample
        bins        [int] bins of the histograms
        show_ylabel [bool]
    """
    # orient so larger = more likely y=1
    if s[y==1].mean() < s[y==0].mean():
        s = -s
    # Youden threshold
    t_star = best_threshold_1d( s, y )
    # common bin range
    lo = np.percentile(s, 0.5)
    hi = np.percentile(s, 99.5)
    lo, hi = float(lo), float(hi)
    # histograms
    ax.hist(
        s[y==1],
        bins=bins,
        range=(lo, hi),
        density=True,
        alpha=0.6,
        label="entrust",
        color=entrust_col,
    )
    ax.hist(
        s[y==0],
        bins=bins,
        range=(lo, hi),
        density=True,
        alpha=0.6,
        label="distrust",
        color=distrust_col,
    )
    # zero and best threshold
    ax.axvline( t_star, color="k", linestyle=":", linewidth=1)
    # cosmetics
    ax.tick_params( axis='both', labelsize=fsz )
    if show_ylabel:
        ax.set_ylabel("density", fontsize=fsz+2 )
    ax.set_xlabel("pre-activation", fontsize=fsz+2 )
    ax.set_title( title, fontsize=fsz+3 )


def mono_plots_for_neurons(X, y, art, max_neurons=6, bins=40, suptitle=None, basename="plot_" ):
    """
    histograms of activation for entrust/distrust samples of a single neuron, for all neurons in a probe
    args:
        X           [np.array] (n,H) raw pre-norm activations
        y           [np.array] (n,) class labels per sample
        art         [ProbeArtifact] the probe
        max_neurons [int] maximum number of neurons shown in one plot
        bins        [int] bins of the histograms
        suptitle    [str]
        basename    [str]
    """
    C, y, idx   = per_neuron_contrib_arrays( X, y, art )
    k           = C.shape[1]
    take        = min(k, max_neurons)
    order       = np.argsort(-np.abs(C.mean(axis=0)))[:take]  # top by |mean contrib|
    rows        = 1 if take < 5 else 2
    cols        = int(np.ceil( take / rows ))
    fig, axes   = plt.subplots( rows, cols, figsize=(4*cols, 3*rows), squeeze=False, sharey=True )
    match take:
        case    1:
            fsz = 11
        case    2:
            fsz = 12
        case    _:
            fsz = 14

    first       = True
    for ax, j in zip( axes.ravel(), order ):
        neuron  = idx[ j ]
        s = X[:, neuron].astype(np.float32)
        plot_mono_hist(ax, s, y, bins=bins, title=f"neuron {neuron}", fsz=fsz, show_ylabel=first )
        first   = False

    for kfree in range(take, rows*cols):
        axes.ravel()[kfree].axis('off')
    if suptitle: fig.suptitle( suptitle, fontsize=fsz+4 )
    # NOTE: legend fails for probes with 5 neurons: axes[-1,-1] is empty, but cases are so rare that
    # it is not worth fixing
    if take != 5:
        axes[-1,-1].legend( loc="best", frameon=False, fontsize=fsz )
    plt.tight_layout()
    fname   = basename + "act_hist" + extension
    plt.savefig( fname )
    plt.close()


def plot_trust_flip( df, base=64, n_bins=32, title="", basename="plot_" ):
    """
    bar plot of trust flip rates in both directions (entrust->distrust/distrust->entrust)
    versus layer
    args:
        df  [pandas.core.frame.DataFrame] already sliced for desired model/scene, should have
            at least the columns 'layer', 'entrust_flip','distrust_flip'
        basename    [str]
    """
    sub     = df.sort_values('layer').copy()

    layers  = sub['layer'].to_numpy()
    f_e     = sub['entrust_flip'].to_numpy().astype(float)
    f_d     = sub['distrust_flip'].to_numpy().astype(float)
    e_max   = float(f_e.max()) if len(f_e) else 1.0
    d_max   = float(f_d.max()) if len(f_d) else 1.0

    nl      = len( layers )
    fig, ax = plt.subplots( figsize=( 0.4 * nl, 5.0 ) )

    def symmetric_power_edges( n_bins=32, base=64, gamma=0.5 ):
        """
        help function for constructing a colormapper dense at the edges
        returns n_bins+1 edges in [0,1], dense near 0 and 1
        """
        t           = np.linspace( 0, 1, n_bins + 1 )
        s           = base * t
        y           = np.where(s <= base/2, (s**gamma)/base, 1.0 - (( base - s)**gamma)/base )
        y[0], y[-1] = 0.0, 1.0
        return y

    def discrete_from_cmap( base_cmap, n ):
        xs = (np.arange(n) + 0.5) / n   # sample midpoints
        return ListedColormap(base_cmap(xs), name=f"{base_cmap.name}_discrete_{n}")


    bounds      = symmetric_power_edges(n_bins=n_bins, base=base, gamma=0.5)
    bn          = BoundaryNorm(bounds, len(bounds)-1, clip=True)
    Greens_disc = discrete_from_cmap(plt.cm.Greens, len(bounds)-1)
    Reds_disc   = discrete_from_cmap(plt.cm.Reds,   len(bounds)-1)
    colors_e    = Greens_disc(bn(f_e))
    colors_d    = Reds_disc(bn(f_d))

# patch for IEEE paper plots
# used with output directory ../twocol_plots
    colors_e    = "tab:green"
    colors_d    = "tab:red"

    bars_e = ax.bar(layers,  f_e, width=0.7, color=colors_e, edgecolor="white", lw=0.6 )
    bars_d = ax.bar(layers, -f_d, width=0.7, color=colors_d, edgecolor="white", lw=0.6 )

# after drawing the bars ...

# 1) lock symmetric limits with a small pad
    pad = 0.02    # tiny breathing room
    ax.set_ylim(-1.0 - pad, 1.0 + pad)

# 2) fixed major ticks every 0.25 including ±1.0 and 0
#   maj = np.array([-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00])
    maj = np.array([-1.00,        -0.50,        0.00,       0.50,       1.00])
    ax.set_yticks(maj)

# label with absolute values on both halves
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{abs(v):.2f}"))

# optional: lighter minor ticks every 0.125
    minr = np.arange(-1.0, 1.0 + 1e-9, 0.125)
    ax.set_yticks(minr, minor=True)
    ax.grid(axis="y", which="major", alpha=0.3 )
    ax.grid(axis="y", which="minor", alpha=0.12 )
    ax.tick_params( axis='both', labelsize=20 )

# 3) add side labels near the axis for clarity
    ymax = 1.0          # since we locked to +-1
    x_ax = -2.6 / nl    # nudges the text left of the y-axis, normalized by the number of layers
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(x_ax, +0.5*ymax, r"entrust$\rightarrow$",  transform=trans, ha="right", va="center",
            rotation=90, fontsize=26, color="#1b5e20")
    ax.text(x_ax, -0.5*ymax, r"distrust$\rightarrow$", transform=trans, ha="right", va="center",
            rotation=90, fontsize=26, color="#7f1d1d")

# keep the main y-label too (for units), no, too much space
#   ax.set_ylabel('flip rate', fontsize=22)
    ax.set_xlabel('layer', fontsize=22)
    ax.set_title( title, fontsize=28 )

    plt.tight_layout()
    fname   = basename + "flip_hist" + extension
    plt.savefig( fname )
    return True


def plot_intervention( df, n_dense=801, title="", basename="plot_" ):
    """
    bar plot of trust flip rates in both directions (entrust->distrust/distrust->entrust)
    for the last layer, compared with random intervention
    args:
        df  [pandas.core.frame.DataFrame] already sliced for desired model/scene, should have
            at least the columns 'layer', 'entrust_flip','distrust_flip'
        basename    [str]
    """
# assume your DataFrame is named df with columns: ["alphas","flip","random_mu","random_sd"]

# ---- interpolation grid
    a = df["alphas"].to_numpy()
    flip = df["flip"].to_numpy()
    mu = df["random_mu"].to_numpy()
    sd = df["random_sd"].to_numpy()

# dense symmetric grid
    a_dense = np.linspace(a.min(), a.max(), n_dense )

# choose either simple linear interp (robust)...
    flip_i = np.interp(a_dense, a, flip)
    mu_i   = np.interp(a_dense, a, mu)
    sd_i   = np.interp(a_dense, a, sd)

# ...OR (optional) use a smoothing spline if you have SciPy installed
# from scipy.interpolate import UnivariateSpline
# s_fac = len(a)*0.002  # tweak smoothness
# flip_i = UnivariateSpline(a, flip, s=s_fac)(a_dense)
# mu_i   = UnivariateSpline(a, mu,   s=s_fac)(a_dense)
# sd_i   = UnivariateSpline(a, sd,   s=s_fac)(a_dense)

# ---- compute bands and deltas
    mu_pm1 = mu_i + 0.5 * sd_i
    mu_mm1 = mu_i - 0.5 * sd_i

# ---- figure with two panels
    fig, ax = plt.subplots( figsize=ffigsize )

# Panel 1: main curves + uncertainty bands
    ax.fill_between(a_dense, mu_mm1, mu_pm1, color='tab:pink', alpha=0.25 )
    ax.plot(a_dense, mu_i,  linewidth=2, color='tab:brown', label="random flip" )
    ax.plot(a_dense, flip_i, linewidth=2, color='tab:green', label="measured flip")

# cosmetics
    ax.axvline(0, linewidth=1, linestyle=":", alpha=0.7)
    ax.set_ylabel( "flip rate", fontsize=22 )
    ax.set_title( title, fontsize=28  )
    ax.legend( loc="best", frameon=False, fontsize=24 )
    ax.tick_params( axis='both', labelsize=22 )

# Panel 2: “excess over random” (measured − random_mean)
    ax.axhline(0, linewidth=1, linestyle=":", alpha=0.7)
    ax.set_xlabel( r"steering strength $\alpha$", fontsize=24 )
    ax.set_ylabel( r"$\Delta$ flip", fontsize=24 )

    plt.tight_layout()
    fname   = basename + "interv_flip" + extension
    plt.savefig( fname )
    plt.close()


def plot_prototype_auc_by_layer( df, title="", basename="plot_" ):
    """
    df columns: ['layer','AUC_proto','scen'] (plus others ignored)
    Plots AUC_proto vs layer for each scenario, highlights per-scenario peaks,
    """
    scen_order = sorted(df['scen'].unique())
    plt.figure(figsize=(8,5))
    ax = plt.gca()

    for i, scen in enumerate( scen_order ):
        d = df[df['scen']==scen].sort_values('layer')
        x = d['layer'].to_numpy()
        y = d['AUC_proto'].to_numpy()

        # line + markers
        ax.plot(x, y, marker=scen_marker[ i ], color=tab_colors[ i ], linewidth=2, label=scen)

        # annotate peak layer
        i_star = np.argmax(y)

    ax.tick_params( axis='both', labelsize=22 )

    ax.set_xlabel( "layer", fontsize=22 )
    ax.set_ylabel( "AUC", fontsize=22 )
    ax.set_title( title, fontsize=28 )
    ax.set_ylim(0.5, 1.0)
    ax.grid( True, alpha=0.2 )
    ax.legend( frameon=False, fontsize=22 )
    plt.tight_layout()
    fname   = basename + "proto" + extension
    plt.savefig( fname )
    plt.close()

