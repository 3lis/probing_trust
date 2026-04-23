"""
#####################################################################################################################

    trust project - 2024

    Configuration of parameters from file and command line

#####################################################################################################################
"""

import  os
from    argparse        import ArgumentParser


class Config( object ):
    """
    Parameters accepted by the software:
    (many parameters can be ser in the configuration file as well as with command line flags)

    CONFIG                  [str] name of configuration file (without path nor extension) (DEFAULT=None)
    DEBUG                   [bool] debug mode: print prompts only, do not call OpenAI
    MODEL                   [int] index in the list of possible models (DEFAULT=0)
    VERBOSE                 [int] write additional information, -v standard, -vv for debugging

    execution               [str] one of "prompt", "activation", "probe", "proto", "analysis"
    f_prompt                [str] filname of the prompt dataset
    f_zarr                  [str] path of zarr archive
    f_manifest              [str] path of zarr archive
    init_dialog             [list] titles of initial dialogues
    model_id                [int] index in the list of possible models (overwritten by MODEL)
    scenario                [str] name of json file with the scenario (no extension)
    layer_k                 [int] specific layer to process
    layers                  [list] list of layers to process
    """

    def load_from_line( self, line_kwargs ):
        """
        Load parameters from command line arguments

        params:
            line_kwargs:        [dict] parameteres read from arguments passed in command line
        """
        for key, value in line_kwargs.items():
            setattr( self, key, value )


    def load_from_file( self, file_kwargs ):
        """
        Load parameters from a python file.
        Check the correctness of parameteres, set defaults.

        params:
            file_kwargs:        [dict] parameteres coming from a python module (file)
        """
        for key, value in file_kwargs.items():
            setattr( self, key, value )


    def __str__( self ):
        """
        Visualize the list of all parameters

        return:     [str]
        """
        s   = ''
        d   = self.__dict__

        for k in d:
            if isinstance( d[ k ], dict ):
                s   += f"{k}:\n"
                for j in d[ k ]:
                    s   += f"{'':5}{j:<30}{d[ k ][ j ]}\n"
            else:
                s   += f"{k:<35}{d[ k ]}\n"
        return s


def read_args():
    """
    Parse the command-line arguments defined by flags
    
    return:         [dict] key = name of parameter, value = value of parameter
    """
    parser      = ArgumentParser()

    parser.add_argument(
            '-c',
            '--config',
            action          = 'store',
            dest            = 'CONFIG',
            type            = str,
            default         = None,
            help            = "Name of configuration file (without path nor extension)"
    )
    parser.add_argument(
            '-D',
            '--debug',
            action          = 'store_true',
            dest            = 'DEBUG',
            help            = "Debug mode: print prompts only, do not call OpenAI"
    )
    parser.add_argument(
            '-m',
            '--model',
            action          = 'store',
            dest            = 'MODEL',
            type            = int,
            default         = None,
            help            = "Index in the list of possible models (default=0) (-1 to print all)",
    )
    parser.add_argument(
            '-v',
            '--verbose',
            action          = 'count',
            dest            = 'VERBOSE',
            default         = 0,
            help            = "Write additional information, -v standard, -vv more"
    )
    return vars( parser.parse_args() )


