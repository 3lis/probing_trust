"""
#####################################################################################################################

    trust project - 2025

    list of models in use and their properties

#####################################################################################################################
"""

models                  = (                     # available models (first one is the default)
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "google/gemma-2-9b-it",
        "Qwen/Qwen1.5-7B-Chat",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct-1M",
)
models_short_name       = {                     # short name identifying a model, as used in log.txt
        "meta-llama/Llama-2-7b-chat-hf"         : "ll2-7",
        "meta-llama/Llama-2-13b-chat-hf"        : "ll2-13",
        "meta-llama/Meta-Llama-3.1-8B-Instruct" : "ll3-8",
        "google/gemma-2-9b-it"                  : "gem2-9",
        "microsoft/Phi-3-mini-4k-instruct"      : "ph3m",
        "Qwen/Qwen1.5-7B-Chat"                  : "qw1-7",
        "Qwen/Qwen2.5-7B-Instruct"              : "qw2-7",
        "Qwen/Qwen2.5-14B-Instruct-1M"          : "qw2-14",
}
models_family           = {                     # which family a model belongs to
        "meta-llama/Llama-2-7b-chat-hf"         : "meta",
        "meta-llama/Llama-2-13b-chat-hf"        : "meta",
        "meta-llama/Meta-Llama-3.1-8B-Instruct" : "meta",
        "google/gemma-2-9b-it"                  : "google",
        "microsoft/Phi-3-mini-4k-instruct"      : "microsoft",
        "Qwen/Qwen1.5-7B-Chat"                  : "qwen",
        "Qwen/Qwen2.5-7B-Instruct"              : "qwen",
        "Qwen/Qwen2.5-14B-Instruct-1M"          : "qwen",
}
