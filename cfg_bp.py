# example configutation for executing build_probes
kwargs      = {
    'model_id':          7,
    'execution':        "probe",
#   'execution':        "proto",
#   'layers':           list( range( 20, 41 ) ) #ll2-13
#   'layers':           list( range( 20, 32 ) ) #most
#   'layers':           list( range( 20, 29 ) ) #qw2-7
#   'layers':           list( range( 10, 21 ) ) #additional lower layers
    'layers':           list( range( 10, 49 ) ) #qw2-14
}
