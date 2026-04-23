###############################################3
#
# example configuration for final analysis
###############################################3
kwargs      = {
    'execution':        "analysis",
#   'analyses':         [ "proto", "probe", "flip", "plot_flip" ],
    'analyses':         [ "probe", "flip", "inter", "plot_flip" ],
#   'analyses':         [ "flip", "plot_flip" ],
#   'analyses':         [ "flip", "plot_flip" ],
#   'analyses':         [ "proto" ],
#   'models':           [ "ll2-7", "ll2-13", "ph3m" ],
#   'models':           [ "ll2-7" ],
    'n_layers':         10,

}
