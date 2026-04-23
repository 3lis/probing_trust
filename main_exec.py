"""
#####################################################################################################################

    trust probing project - 2025

    main execution file
    this script is the main interface for all the processing steps in the pipeline,
    that are selected with the "execution" parameter in the configuration file
    the possible execution modes are:

    "prompt":       collect prompts from the simulations in trust_prop, and store them in a dataset
    "activation":   run a model on prompts from the dataset above, and store activations for all layers
    "proto":        build prototype vectors for all layers that best respond to trust/distrust
    "probe":        as above, in addition extract small probes for a selection of layers
    "analysis"      analyze the performances of prototypes and probes, producing textual and graphic results
                    in turn, this execution comprises several options, specified with the list analyses,
                    possible values are:
                        "proto"         plot prototype AUC by layers, separated by scenarios
                        "probe"         plot histograms of activation for single neurons of probes, for last layers
                        "flip"          executes analysis.exec_test_probe(), necessary for the remaining specs
                        "save_flip"     save results of analysis.exec_test_probe() in ../probes
                        "inter"         plot curve of intervention flip on the last layer, compared with random
                        "plot_flip"     plot bars of flip rates in both directions per layer
                        "tex_drop"      TeX table of discrimination drops at best layer per scenario and model

#####################################################################################################################
"""

import  os
import  sys
import  pickle
import  copy
import  time
import  datetime
import  shutil
import  random
import  pandas          as pd

import  load_cnfg
import  collect_act
import  collect_prompts
import  run_model
import  build_probes
import  analysis

from    models      import models, models_short_name, models_family

now_time            = None
frmt_response       = "%y-%m-%d_%H-%M-%S"   # datetime format for response filenames
time_file           = "timing.dat"          # for profiling

cnfg                = None                  # configuration object

PROFILE             = False

dir_data            = "../data"             # with the original json file, all prompts, and models' w_pre, c
dir_res             = "../res"              # zarr file of activation, prototypes of all models
dir_probe           = "../probes"           # all (1200!) probes in joblib format
dir_plots           = "../plots"            # all plot results
dir_plots           = "../twocol_plots"     # alternative directory for flip plots in two colors only
dir_stat            = "../stat"             # LaTeX tables with statistcs
f_zarr              = "activation.zarr"     # all model's activation in response to prompts
f_manifest          = "activation.parquet"  # corresponding info
f_prompt            = "prompts.pkl"         # all prompts, of all models and scenarios
f_proto             = "proto.pkl"           # suffix of prototype result files in Pandas
f_probe             = "probe_info.pkl"      # suffix of probe info files in Pandas
prompt_path         = None
zarr_path           = None
manifest_path       = None
dir_current         = None

# scenarios and their code in zarr
scen_map            = {"school":0, "fire":1, "farm":2}

# models currently used
curr_models         = ( "ll2-7", "ll2-13", "ll3-8", "ph3m", "qw1-7", "qw2-7", "qw2-14" )

# possible execution modes
executions          = ( "prompt", "activation", "probe", "proto", "analysis" )

# possible analyses
analyses            = ( "probe", "proto", "inter", "flip", "plot_flip", "tex_drop" )

# ===================================================================================================================
#
#   Basic utilities
#   init_paths()
#   init_cnfg()
#
# ===================================================================================================================

def init_paths():
    """ -------------------------------------------------------------------------------------------------------------
    Set paths to proper directories
    ------------------------------------------------------------------------------------------------------------- """
    global prompt_path, zarr_path, manifest_path

    prompt_path     = os.path.join( dir_data, f_prompt )
    zarr_path       = os.path.join( dir_res, f_zarr )
    manifest_path   = os.path.join( dir_res, f_manifest )


def init_cnfg():
    """
    Set global parameters from command line and python configuration file
    Execute this function before init_paths()
    """
    global cnfg, now_time, f_zarr, f_manifest, f_prompt, f_proto, f_probe


    # minimal verification of integrity of the models' information 
    assert len( models ) == len( models_short_name ) == len( models_family ), \
                      "error in models.py: mismatched models' info"

    cnfg            = load_cnfg.Config()                    # instantiate the configuration object

    # load parameters from command line
    line_kwargs     = load_cnfg.read_args()                 # read the arguments in the command line
    cnfg.load_from_line( line_kwargs )                      # and parse their value into the configuration

    if cnfg.MODEL is not None and cnfg.MODEL < 0:
        print( "ID    model                                 interface  short name" )
        for i, m in enumerate( models ):
            s   = ''
            if m in models_short_name:
                s   = models_short_name[ m ]
            if len( m ) > 40:
                m   = m[ : 26 ] + "<...>" + m[ -9 : ]
            print( f"{i:>2d}   {m:<43} {s}" )
        sys.exit()

    # load parameters from file
    if cnfg.CONFIG is None:
        print( "cannot execute anything without a configuration file" )
        sys.exit()
    exec( "import " + cnfg.CONFIG )                     # exec the import statement
    file_kwargs     = eval( cnfg.CONFIG + ".kwargs" )   # assign the content to a variable
    cnfg.load_from_file( file_kwargs )                  # read the configuration file,

    # overwrite command line arguments
    if cnfg.MODEL is not None:
        cnfg.model_id       = cnfg.MODEL

    # if a model is used, from its index derive the complete model name and usage mode
    if hasattr( cnfg, 'model_id' ):
        assert cnfg.model_id < len( models ), f"error: model # {cnfg.model_id} not available"
        cnfg.model          = models[ cnfg.model_id ]
        cnfg.model_short    = models_short_name[ cnfg.model ]
        cnfg.model_family   = models_family[ cnfg.model ]

    if hasattr( cnfg, 'f_zarr' ):
        f_zarr              = cnfg.f_zarr
    if hasattr( cnfg, 'f_manifest' ):
        f_manifest          = cnfg.f_manifest
    if hasattr( cnfg, 'f_prompt' ):
        f_prompt            = cnfg.f_prompt
    if hasattr( cnfg, 'f_proto' ):
        f_proto             = cnfg.f_proto
    if hasattr( cnfg, 'f_probe' ):
        f_probe             = cnfg.f_probe

    # export information from config
    build_probes.VERBOSE    = cnfg.VERBOSE
    analysis.VERBOSE        = cnfg.VERBOSE

    # export misc globals
    analysis.curr_models    = curr_models
    analysis.scenarios      = list( scen_map.keys() )

    # ensure path consistency
    analysis.dir_data       = dir_data
    analysis.dir_res        = dir_res 
    analysis.dir_probe      = dir_probe
    analysis.dir_plots      = dir_plots
    analysis.dir_stat       = dir_stat 

    # check for valid execution mode
    if hasattr( cnfg, 'execution' ):
        assert cnfg.execution in executions, f"error: execution {cnfg.execution} not implemented"
    else:
        cnfg.execution  = "nothing"

    # check for validity of the specified analysis types
    if cnfg.execution == "analysis":                    # in this case multiple models are used
        if not hasattr( cnfg, "models" ):
            cnfg.models     = curr_models               # use all current models
        if not hasattr( cnfg, "analyses" ):
            cnfg.analyses   = analyses                  # use all possible analyses
        if "plot_flip" in cnfg.analyses:
            assert "flip" in cnfg.analyses, '"plot_flip" analysis requires "flip"'
        if "tex_drop" in cnfg.analyses:
            assert "flip" in cnfg.analyses, '"tex_drop" analysis requires "flip"'

    # string used for composing directory of results
    now_time                = time.strftime( frmt_response )



# ===================================================================================================================
#
#   main functions
#
#   do_prompt()
#   do_activation()
#   do_probe()
#   do_analysis()
#
# ===================================================================================================================

def do_prompt():
    """
    launch the collection of prompts
    it is assumed that the configuration file contains the specification of the range of results
    for which prompts are collected
    """
    init_paths()
    collect_prompts.dump_file       = prompt_path
    if hasattr( cnfg, 'res_range' ):
        collect_prompts.res_range   = cnfg.res_range
    collect_prompts.collect_data()
    return True


def do_activation():
    """
    launch the collection of model's activation to prompts
    for which prompts are collected
    """
    assert hasattr( cnfg, 'model' ), "error: no model specified for collecting activation"
    init_paths()
    assert os.path.isfile( prompt_path ), "error: file with prompts not found"
    run_model.model_name    = cnfg.model
    run_model.set_hf()
    no_variants             = cnfg.model_short == "qw2-14"  # this model has memory issues
    extractor_fn            = run_model.make_extractor( no_variants=no_variants )
    model_map               = dict()
    for i,k in enumerate( models ):
        s                   = models_short_name[ k ]
        model_map[ s ]      = i
    df                      = pd.read_pickle( prompt_path )
    dfm                     = df[ df[ "model" ] == cnfg.model_short ].copy()

    collect_act.collect_and_store(
        df              = dfm,
        extractor_fn    = extractor_fn,
        zarr_path       = zarr_path,
        manifest_path   = manifest_path,
        model_map       = model_map,
        scen_map        = scen_map,
        batch_size      = 128,
    )
    return True


def do_probe( no_probe=False ):
    """
    launch the building of prototypes and probes for a model
    args:
        no_probe [bool] to prototypes only
    """
    assert hasattr( cnfg, 'model_short' ), "error: no model specified for building probes"
    init_paths()
    build_probes.dir_probe  = dir_probe
    build_probes.zarr_path  = zarr_path
    build_probes.manifest_path  = manifest_path
    df                      = build_probes.exec_prototypes( cnfg.model_short )
    fname                   = f"{cnfg.model_short}_{f_proto}"
    df_path                 = os.path.join( dir_res, fname )
    df.to_pickle( df_path )
    if no_probe:
        return True
    assert hasattr( cnfg, 'layers' ), "error: no list of layers specified for building probes"
    df                      = build_probes.exec_build_probe( cnfg.model_short, cnfg.layers )
    fname                   = f"{cnfg.model_short}_{f_probe}"
    df_path                 = os.path.join( dir_probe, fname )
    df.to_pickle( df_path )

    return True


def do_analysis():
    """
    performs the analysis on already existing probes and prototypes
    it is assumed that the configuration file contains specification on models and kind of analysis
    """
    init_paths()
    analysis.zarr_path          = zarr_path
    analysis.manifest_path      = manifest_path
    collect_prompts.dump_file   = prompt_path

    n_layers                    = 10                # a specification for probe plots
    if hasattr( cnfg, 'n_layers' ):
        n_layers                = cnfg.n_layers

    dfs                         = []                # used only by tex_drop analysis
    for m in cnfg.models:
        if cnfg.VERBOSE:
            print( f"doing analysis for model {m}" )
        if "proto" in cnfg.analyses:
            analysis.plot_proto( model_name=m )
        if "probe" in cnfg.analyses:
            analysis.plot_probe( model_name=m, n_layers=n_layers )
        if "flip" in cnfg.analyses:
            do_plot     = "plot_flip" in cnfg.analyses
            if "save_flip" in cnfg.analyses:
                fname       = f"{m}_flip_{f_probe}"
                df_path     = os.path.join( dir_probe, fname )
            else:
                df_path     = None
            df          = analysis.exec_test_probe( model_name=m, do_plot=do_plot, df_path=df_path )
        if "inter" in cnfg.analyses:
            analysis.plot_inter( model_name=m )
        if "tex_drop" in cnfg.analyses:
            dfs.append( df )

    if "tex_drop" in cnfg.analyses:
        df      = pd.concat( dfs, ignore_index=True )
        stat    = analysis.stat_probe( df )
        fname   = os.path.join( dir_stat, "drop.tex" )
        analysis.latex_probe_stat( stat, fname=fname )

    return True


# ===================================================================================================================
#
#   MAIN
#
# ===================================================================================================================

if __name__ == '__main__':
    init_cnfg()
    match cnfg.execution:
        case "nothing":
            print( "Program instructed to DO NOTHING" )
        case "prompt":
            do_prompt()
        case "activation":
            do_activation()
        case "probe":
            if PROFILE:
                import cProfile
                cProfile.run( "do_probe()", time_file )
            else:
                do_probe()
        case "proto":
            do_probe( no_probe=True )
        case "analysis":
            do_analysis()

