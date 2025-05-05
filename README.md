# NeoDynamML
AI-Enhanced Molecular Dynamics Pipeline for Neoantigen Dynamics Analysis in Cancer Immunotherapy
NeoDynamML is a comprehensive end-to-end pipeline that integrates cutting-edge AI/ML techniques with molecular dynamics simulations to analyze neoantigen dynamics for cancer immunotherapy research. The pipeline identifies potential neoantigens from mutation data, predicts their binding to HLA molecules, and uses molecular dynamics simulations enhanced by machine learning to assess stability and immunogenicity. It also employs graph neural networks to model peptide-HLA interactions and uses transformers to predict T-cell reactivity, ultimately generating detailed reports on the most promising neoantigens for personalized cancer vaccines.

## System Requirements

### Hardware Requirements
- **CPU**: Minimum 8 cores recommended, 16+ cores for optimal performance
- **RAM**: Minimum 32GB, 64GB+ recommended for large datasets
- **GPU**: NVIDIA GPU with 8GB+ VRAM (Tesla V100/A100, RTX 3080+ recommended)
- **Storage**: 500GB+ SSD storage (depending on dataset size)

### Software Requirements
- **Operating System**: Linux (Ubuntu 20.04+, CentOS 8+)
- **CUDA**: Version 11.4+ (for GPU acceleration)
- **Python**: Version 3.8+

## Installation

### Option 1: Using conda (Recommended)

```bash
# Create a new conda environment
conda create -n neodynamml python=3.9
conda activate neodynamml

# Clone the repository
git clone https://github.com/yourorg/neodynamml.git
cd neodynamml

# Install dependencies
conda install -c conda-forge -c bioconda -c nvidia -f environment.yml

# Install NeoDynamML
pip install -e .
```

### Option 2: Using Docker

```bash
# Pull the NeoDynamML Docker image
docker pull yourorg/neodynamml:latest

# Run the container with GPU support
docker run --gpus all -v /path/to/data:/data -it yourorg/neodynamml:latest
```

### Option 3: Manual Installation

```bash
# Clone the repository
git clone https://github.com/yourorg/neodynamml.git
cd neodynamml

# Install Python dependencies
pip install -r requirements.txt

# Install additional dependencies
# GROMACS for MD simulations
sudo apt-get update && sudo apt-get install -y gromacs cuda-toolkit

# Install NeoDynamML
pip install -e .
```

## Pipeline Workflow

NeoDynamML operates through a series of interconnected modules:

1. **Variant Analysis**: Processes genomic data to identify somatic mutations
2. **Neoantigen Prediction**: Identifies candidate neoantigens from mutation data
3. **HLA Binding Prediction**: Assesses neoantigen binding affinity to HLA molecules
4. **MD Simulation Preparation**: Sets up molecular dynamics simulations
5. **ML-Enhanced MD Simulation**: Runs simulations with ML optimization
6. **GNN-Based Interaction Analysis**: Models peptide-HLA interactions using GNNs
7. **T-Cell Reactivity Prediction**: Uses transformer models to predict immunogenicity
8. **Result Integration & Reporting**: Generates comprehensive reports

## Usage Guide

### Quick Start

For a basic run of the pipeline on sample data:

```bash
# Activate environment
conda activate neodynamml

# Run basic workflow with default parameters
neodynamml run --input samples/sample_mutations.vcf --hla samples/hla_types.txt --output results/
```

### Input Data Requirements

#### Required Input Files

1. **Mutation Data**: Variant Call Format (VCF) file or MAF (Mutation Annotation Format) file
   ```
   samples/patient001_mutations.vcf
   ```

2. **HLA Type Information**: Text file with HLA alleles
   ```
   # Format: patient_id HLA-A HLA-A HLA-B HLA-B HLA-C HLA-C
   patient001 A*02:01 A*24:02 B*07:02 B*35:01 C*04:01 C*07:02
   ```

3. **Reference Genome**: Reference genome in FASTA format
   ```
   references/GRCh38.fa
   ```

#### Optional Input Files

1. **Gene Expression Data**: TPM or FPKM values in TSV format
   ```
   samples/patient001_expression.tsv
   ```

2. **Existing MD Trajectory Data**: For re-analysis or extension
   ```
   samples/previous_md_trajectories/
   ```

### Configuration

Create a configuration file `config.yaml` with your desired parameters:

```yaml
# Basic configuration
run_name: "patient001_analysis"
output_dir: "results/patient001"
threads: 16
gpu_device: 0

# Input files
vcf_file: "data/patient001_mutations.vcf"
hla_file: "data/patient001_hla.txt"
reference_genome: "references/GRCh38.fa"
expression_data: "data/patient001_expression.tsv"

# Neoantigen prediction
min_peptide_length: 8
max_peptide_length: 11
binding_affinity_threshold: 500  # nM
expression_threshold: 1.0  # TPM

# MD simulation parameters
simulation_time: 100  # ns
temperature: 310  # K
pressure: 1.0  # bar
timestep: 0.002  # ps
water_model: "TIP3P"

# ML parameters
gnn_model: "peptide_hla_gnn_v2"
transformer_model: "tcell_reactivity_v1"
confidence_threshold: 0.75

# Advanced options
enable_adaptive_sampling: true
save_intermediate_results: true
generate_visualization: true
```

### Running the Pipeline

#### Full Pipeline

```bash
# Run the full pipeline with a configuration file
neodynamml run --config config.yaml
```

#### Individual Modules

You can also run specific modules of the pipeline:

```bash
# Run only the neoantigen prediction module
neodynamml modules neoantigen-prediction --config config.yaml

# Run only the MD simulation module
neodynamml modules md-simulation --config config.yaml --input results/patient001/neoantigen_predictions.csv
```

### Advanced Options

#### Parallel Processing

```bash
# Run with distributed processing on multiple nodes
neodynamml run --config config.yaml --distributed --nodes 4
```

#### Checkpoint and Resume

```bash
# Enable checkpointing
neodynamml run --config config.yaml --checkpointing

# Resume from a checkpoint
neodynamml run --config config.yaml --resume results/patient001/checkpoints/checkpoint_20230501_143022
```

#### Custom Models

```bash
# Use custom ML models
neodynamml run --config config.yaml --gnn-model path/to/custom_gnn_model.pt --transformer-model path/to/custom_transformer.pt
```

## Output Files

### Primary Output

- `neoantigen_report.html`: Interactive HTML report of neoantigen candidates
- `top_neoantigens.csv`: Prioritized list of neoantigens with scores
- `summary_statistics.json`: Statistical summary of the analysis

### Module-Specific Outputs

- `variant_analysis/`: Processed mutation data
- `neoantigen_candidates/`: Predicted neoantigen peptides
- `hla_binding/`: HLA binding prediction results
- `md_simulations/`: Molecular dynamics trajectories
- `peptide_hla_complexes/`: 3D structure files of peptide-HLA complexes
- `gnn_analysis/`: Graph neural network interaction analyses
- `tcell_predictions/`: T-cell reactivity predictions

### Visualization Files

- `visualizations/`: Contains interactive plots and 3D visualizations
  - `binding_affinity_heatmap.png`: Heatmap of binding affinities
  - `3d_structures/`: PDB files of key structures
  - `md_trajectory_analysis/`: Dynamic property plots from MD simulations
  - `interaction_networks/`: Network visualizations of peptide-HLA interactions

## Module Descriptions

### 1. Variant Analysis Module

Processes VCF/MAF files to identify somatic mutations and their potential impact.

```bash
neodynamml modules variant-analysis --vcf data/mutations.vcf --output results/variant_analysis
```

### 2. Neoantigen Prediction Module

Predicts neoantigens from mutation data, considering peptide length and expression.

```bash
neodynamml modules neoantigen-prediction --variants results/variant_analysis --output results/neoantigen_candidates
```

### 3. HLA Binding Prediction Module

Assesses binding affinity between candidate neoantigens and patient HLA types.

```bash
neodynamml modules hla-binding --neoantigens results/neoantigen_candidates --hla data/hla_types.txt --output results/hla_binding
```

### 4. MD Simulation Preparation Module

Prepares structural models for molecular dynamics simulations.

```bash
neodynamml modules md-preparation --binding-results results/hla_binding --output results/md_prep
```

### 5. ML-Enhanced MD Simulation Module

Runs molecular dynamics simulations with machine learning optimizations.

```bash
neodynamml modules md-simulation --prepared-structures results/md_prep --output results/md_simulations
```

### 6. GNN Interaction Analysis Module

Models peptide-HLA interactions using graph neural networks.

```bash
neodynamml modules gnn-analysis --md-trajectories results/md_simulations --output results/gnn_analysis
```

### 7. T-Cell Reactivity Prediction Module

Predicts T-cell reactivity using transformer models.

```bash
neodynamml modules tcell-prediction --binding-data results/hla_binding --gnn-data results/gnn_analysis --output results/tcell_predictions
```

### 8. Result Integration & Reporting Module

Combines results from all modules into comprehensive reports.

```bash
neodynamml modules reporting --all-results results/ --output results/final_report
```

## Customization

### Adding Custom HLA Alleles

To add support for rare or custom HLA alleles:

```bash
neodynamml utils add-hla-allele --name "HLA-A*02:278" --sequence path/to/sequence.fasta
```

### Custom Scoring Functions

Create a custom scoring Python script:

```python
# custom_score.py
def custom_neoantigen_score(binding_affinity, stability_score, reactivity_score):
    # Your custom scoring logic
    return combined_score
```

Then use it in the pipeline:

```bash
neodynamml run --config config.yaml --custom-scoring custom_score.py
```

### Plugin System

NeoDynamML supports plugins for extending functionality:

```bash
# Install a plugin
pip install neodynamml-plugin-proteomics

# Use plugin in pipeline
neodynamml run --config config.yaml --use-plugin proteomics
```

## Performance Considerations

### Memory Optimization

For large datasets:

```bash
neodynamml run --config config.yaml --low-memory --chunk-size 1000
```

### GPU Acceleration

Specify which GPU to use:

```bash
neodynamml run --config config.yaml --gpu 0,1  # Use first and second GPU
```

### Disk Space Management

Clean intermediate files:

```bash
neodynamml utils clean-intermediate --results-dir results/patient001/ --keep-essential
```

## API Documentation

NeoDynamML provides a Python API for integration into other workflows:

```python
from neodynamml import Pipeline

# Initialize pipeline with config
pipeline = Pipeline(config_file="config.yaml")

# Run full analysis
results = pipeline.run()

# Access results
top_neoantigens = results.get_top_candidates(n=10)
print(top_neoantigens)

# Run specific module
binding_results = pipeline.run_module("hla_binding")
```

For detailed API documentation:

```bash
# Generate and view API documentation
neodynamml docs --server
```

## Use Cases

### 1. Single Patient Analysis

```bash
neodynamml run --vcf patient001.vcf --hla patient001_hla.txt --output patient001_results/
```

### 2. Cohort Analysis

```bash
neodynamml cohort --patients-dir patients/ --output cohort_results/
```

### 3. Integrating with Existing Data

```bash
neodynamml run --config config.yaml --existing-data previous_analysis/
```

## Troubleshooting

### Common Issues

#### CUDA Error

```
Error: CUDA driver version is insufficient for CUDA runtime version
```

Solution:
```bash
# Update CUDA drivers
neodynamml utils update-cuda
```

#### Memory Error During MD Simulation

```
Error: Out of memory during MD simulation
```

Solution:
```bash
# Reduce system size or use lower memory options
neodynamml modules md-simulation --low-memory --reduced-precision
```

#### Missing HLA Structures

```
Error: Structure for HLA-A*68:01 not found
```

Solution:
```bash
# Download missing HLA structures
neodynamml utils download-hla-structures --allele "HLA-A*68:01"
```

### Logging and Debugging

```bash
# Enable debug logging
neodynamml run --config config.yaml --log-level debug --log-file debug.log
```


### Development Setup

```bash
# Clone repository with development dependencies
git clone https://github.com/yourorg/neodynamml.git
cd neodynamml
pip install -e ".[dev]"

# Run tests
pytest tests/
```

