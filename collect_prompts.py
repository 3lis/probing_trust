"""
#####################################################################################################################

    trust probing project - 2025

    retrieve prompts from trust_prop results

#####################################################################################################################
"""

import  os
import  sys
import  numpy    as np
import  pandas   as pd

res         = "../../trust_prop/res"
log         = "runs.log"
dump_file   = "../data/prompts_ll3_8.pkl"
old_query   = """ Therefore, please reply with "<yes/no>" only.\n"""
# a query that ensure the single next token in a clean "yes/no"
new_query   = "Reply with exactly one word: yes or no (lowercase, no punctuation). Answer: "



# specification of the executions to analyze
# if res_range is empty all executions found in ../res are analyzed
# if res_range has only one entry, it is the first execution to process, followed by all the others
# if res_range has two entries, these are the boundaries of the executions to analyze
# if res_range has one list, than all and only the entries in the inner list are analyzed
# if res_range has one or more tuples, than tuples should have two entries, which are boundaries of multiple ranges
res_range           = [ [
    "25-09-02_15-12-02",    # ll3-8      fire
    "25-09-03_06-45-53",    # ll3-8      farm
    "25-09-04_05-15-49",    # ll3-8      school
] ]


# ===================================================================================================================
#
#   information retrieval functions
#
#   get_info()
#   get_prompt()
#   scan_result()
#
# ===================================================================================================================

def get_info( lines ):
    """
    retrieve essential info
    """

    model       = "---"
    scenario    = "---"
    augm        = "---"
    easiness    = "0"
    found       = False
    for l in lines:
        if l.startswith( "model_short" ):
            model       = l.split()[ -1 ]
            found       = True
        if l.startswith( "scenario" ):
            scenario    = l.split()[ -1 ].replace( "scenario_", '' )
        if l.startswith( "easiness" ):
            easiness    = l.split()[ -1 ]
        if l.startswith( "augmentation" ):
            augm        = l.split()[ -1 ]

    if not found:
        return None

    return model, scenario, augm, easiness


def get_prompt( lines ):
    """
    retrieve one prompt
    args:
        lines   [list] with lines in the file
    return:
        [tuple] of completion [str], prompt [str], remaining lines [list]
    """
    i       = 0
    while i < len( lines ):             # search for a new dialog turn inside full capacity
        l       = lines[ i ]
        if "full capacity stage" in l:
            break
        i       += 1
    i       += 1
    while i < len( lines ):             # search for beiginning of prompt
        l       = lines[ i ]
        if l.startswith( "USER:" ):
            prompt  = ''
            break
        i       += 1
    i       += 1
    turn    = False                     # 2-turns dialog
    while i < len( lines ):             # prompt continuation
        l       = lines[ i ]
        if l.startswith( "---------" ): # prompt end
            break
        if l.startswith( "ASSIST" ):    # it was a preliminary dialgo turn, search again
            turn    = True
            break
        prompt  += l
        i       += 1
    if turn:                            # search for the second dialg turn
        while i < len( lines ):
            l       = lines[ i ]
            if l.startswith( "USER:" ):
                prompt  = ''
                break
            i       += 1
        i       += 1
        while i < len( lines ):         # prompt continuation
            l       = lines[ i ]
            if l.startswith( "---------" ): # prompt end
                end     = True
                break
            prompt  += l
            i       += 1
    i       += 1
    while i < len( lines ):             # search for beiginning of completion
        l       = lines[ i ]
        if l.startswith( "Completion #" ):
            break
        i       += 1
    i       += 1
    if i >= len( lines ):               # no more lines
        return None, None, []

    compl       = lines[ i ].split()
    if "yes" in compl[ 0 ].lower():
        return "yes", prompt, lines[ i : ]
    if "no" in compl[ 0 ].lower():
        return "no", prompt, lines[ i : ]
    return None, None, lines[ i : ]


def scan_result( res_name ):
    """
    retrieve all prompts from one result log
    args:
        res [str]   name, in timestamp, of the result
    return:
        [pandas.core.frame.DataFrame] the data in pandas DataFrame
    """
    n_info  = 50                            # number of heading lines
    fname   = os.path.join( res, res_name, log )
    if not os.path.isfile( fname ):
        print( f"{res_name}  is not a file" )
        return None
    with open( fname, 'r' ) as fd:
        lines   = fd.readlines()
    if not len( lines ):
        print( f"file {res_name}  has no lines" )
        return None
    # get the basic information from the heading lines
    info        = get_info ( lines[ :n_info ] )
    if info is None:
        print( f"no info found in {res_name}" )
        return None

    pr_yes  = []
    pr_no   = []

    # scan all remaining lines in the file searching for prompts
    lines   = lines[ n_info: ]
    while len( lines ):
        yn, pr, lines   = get_prompt( lines )
        if yn == "yes":
            pr_yes.append( pr.replace( old_query, new_query ) )     # change the final query in the prompt
        if yn == "no":
            pr_no.append( pr.replace( old_query, new_query ) )      # change the final query in the prompt

    d       = dict()
    model, scen, augm, _    = info
    # balance the number of positive and negative prompts
    n       = min( len( pr_yes ), len( pr_no ) )
    d[ "prompt" ]   = np.array( pr_yes[ :n ] + pr_no[ :n ] )
    d[ "yes_no" ]   = np.array( n * [ 1 ] + n * [ 0 ] )
    d[ "model" ]    = np.array( 2 * n * [ model ] )
    d[ "scen" ]     = np.array( 2 * n * [ scen ] )
    d[ "augm" ]     = np.array( 2 * n * [ augm ] )
    df              = pd.DataFrame( d )

    return df



# ===================================================================================================================
#
#   main functions
#
#   select_data()
#   collect_data()
#
# ===================================================================================================================

def select_data():
    """
    build the list of results to collect for statistics

    return:             [list] with directories in ../res
    """
    list_res    = sorted( os.listdir( res ) )
    if not len( res_range ):
        return list_res

    if isinstance( res_range[ 0 ], list ):
        return res_range[ 0 ]

    if isinstance( res_range[ 0 ], tuple ):
        multi_res   = []
        for r in res_range:
            assert len( r ) == 2, "entries in res_range should be tuples with exactely two items"
            first   = r[ 0 ]
            last    = r[ -1 ]
            assert first in list_res, f"first specified result {first} not found"
            assert last in list_res, f"last specified result {last} not found"
            i_first     = list_res.index( first )
            i_last      = list_res.index( last )
            multi_res   += list_res[ i_first : i_last+1 ]
        return multi_res

    if len( res_range ) == 1:
        first   = res_range[ 0 ]
        assert first in list_res, f"first specified result {first} not found"
        i_first     = list_res.index( first )
        return list_res[ i_first : ]

    if len( res_range ) == 2:
        first   = res_range[ 0 ]
        last    = res_range[ -1 ]
        assert first in list_res, f"first specified result {first} not found"
        assert last in list_res, f"last specified result {last} not found"
        i_first     = list_res.index( first )
        i_last      = list_res.index( last )
        return list_res[ i_first : i_last+1 ]

    print( "if you want to specify single results to be collected, include them in a list inside res_range\n" )
    return []


def collect_data():
    """
    collect all prompts with their additional information, and save on file in Pandas format
    """

    list_res    = select_data()
    n_res       = len( list_res )
    print( f"scanning for {n_res} execution results\n" )
    dfs         = []

    for f in list_res:                          # scan all selected results
        df      = scan_result( f )
        n_rec   = len( df )
        dfs.append( df )
        print( f"{f}  done with {n_rec} records" )

    df              = pd.concat( dfs, ignore_index=True )
    df["yes_no"]    = df["yes_no"].astype("int8")
    df["model"]     = df["model"].astype("category")
    df["scen"]      = df["scen"].astype("category")

    df.to_pickle( dump_file )                   # save all prompts in Pandas format
