# Mechanistic Evidence for Trust-Related Representations in LLMs

This repository accompanies the paper:

**Mechanistic Evidence for Trust-Related Representations in Large Language Models**

It provides the code used to collect activations, construct prototype vectors and sparse probes, and perform the analyses reported in the paper.

## Overview

The pipeline consists of four main stages:

1. **Prompt collection**  
   Extraction and preprocessing of prompts from simulation outputs.

2. **Activation extraction**  
   Execution of language models on prompts and storage of layer-wise hidden representations.

3. **Probe construction**  
   Computation of:
   - prototype vectors (full-state linear readouts)
   - sparse probes (l1-based selection and stability selection)

4. **Analysis**  
   Evaluation of:
   - decoding performance (AUC, accuracy)
   - causal effects (ablation, flip rates)
   - intervention at the final layer

The main entry point is `main_exec.py`, which orchestrates all stages via a configuration file.

## Repository Structure

- `main_exec.py` — main execution script controlling the pipeline
- `collect_prompts.py` — extraction of prompts from simulation logs
- `collect_act.py` — collection and storage of activations (Zarr + Parquet)
- `run_model.py` — model execution and extraction of hidden states and log-probabilities
- `build_probes.py` — construction of prototype vectors and sparse probes
- `analysis.py` — evaluation and causal analyses (drop, flip, intervention)
- `plot.py` — plotting utilities
- `models.py` — list of supported models and identifiers
- `load_cnfg.py` — configuration handling

Example configuration files:

- `cfg_pr.py` — prompt collection
- `cfg_a.py` — activation extraction
- `cfg_bp.py` — probe construction
- `cfg_aM.py` — analysis

## Data Storage

Activations are stored using:

- **Zarr** for tensor data (`repr`) with shape `(sample, layer, hidden)`
- **Parquet** for metadata such as prompts, labels, and probabilities

This design allows efficient slicing by layer, model, and scenario.

## Usage

All operations are controlled through `main_exec.py` with a configuration file:

```bash
python main_exec.py --config cfg_name
```

### Execution modes

The following modes are supported:

- `prompt` — collect prompts
- `activation` — compute and store activations
- `proto` — compute prototype vectors
- `probe` — construct sparse probes
- `analysis` — run evaluation and generate plots

## Typical Workflow

1. **Collect prompts**

```bash
python main_exec.py --config cfg_pr
```

2. **Extract activations**

```bash
python main_exec.py --config cfg_a
```

3. **Build probes**

```bash
python main_exec.py --config cfg_bp
```

4. **Run analysis**

```bash
python main_exec.py --config cfg_aM
```

## Models

The repository supports multiple instruction-tuned models, including:

- Llama-2 (7B, 13B)
- Llama-3.1 (8B)
- Phi-3-mini
- Qwen (1.5, 2.5)

Model definitions and short names are specified in `models.py`.

## Notes

- Activation extraction requires access to the corresponding Hugging Face models and credentials.
- Large models may require substantial GPU memory.
- Intermediate results are written to directories such as `../res`, `../probes`, and `../plots`.

## Reproducibility

The code implements the procedures described in the paper, including:

- cross-scenario generalization
- stability selection for sparse probes
- consensus probe construction
- causal ablation and intervention analyses

Exact numerical results may vary depending on hardware, library versions, and stochastic subsampling.

## License

This repository is provided for research purposes.
