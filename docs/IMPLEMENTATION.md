# VacuaGym Implementation Guide

Complete guide to the VacuaGym pipeline from data download to active learning.

**Version**: v0.1.0-alpha
**Last Updated**: 2025-12-26

---

## Pipeline Overview

VacuaGym implements a complete ML pipeline for string theory vacuum exploration:

```
Raw Data → Parsed Tables → Features → Labels → Splits → Models → Active Learning
(Phase 1)   (Phase 1)      (Phase 2) (Phase 3) (Phase 4) (Phase 5) (Phase 6)
```

---

## Phase 1: Data Ingestion

### Download Scripts (scripts/01-03)

**Purpose**: Download public geometry datasets

**Scripts**:
1. `01_download_ks.py` - Kreuzer-Skarke reflexive polytopes
2. `02_download_cicy3.py` - CICY threefold list (7,890)
3. `03_download_fth6d.py` - F-theory 6D toric bases (61,539)

**Usage**:
```bash
python scripts/01_download_ks.py
python scripts/02_download_cicy3.py
python scripts/03_download_fth6d.py
```

**Output**:
- `data/raw/ks_reflexive_polytopes/`
- `data/raw/cicy3_7890/`
- `data/raw/ftheory_6d_toric_bases_61539/`

### Parser Scripts (scripts/10-12)

**Purpose**: Convert raw files to standardized Parquet tables

**Scripts**:
1. `10_parse_ks.py` - Parse KS polytopes
2. `11_parse_cicy3.py` - Parse CICY configurations
3. `12_parse_fth6d.py` - Parse F-theory bases

**Usage**:
```bash
python scripts/10_parse_ks.py
python scripts/11_parse_cicy3.py
python scripts/12_parse_fth6d.py
```

**Output**:
- `data/processed/tables/ks_polytopes.parquet`
- `data/processed/tables/cicy3_configs.parquet`
- `data/processed/tables/fth6d_bases.parquet`

**Data Schema**:
- Polytope ID / Config ID / Base ID
- Hodge numbers (h^{1,1}, h^{2,1})
- Topological invariants (Euler characteristic, etc.)
- Geometric features

---

## Phase 2: Feature Engineering

### Feature Building (scripts/20)

**Purpose**: Create ML-ready features from parsed data

**Script**: `20_build_features.py`

**Features Generated**:

**For KS Polytopes**:
- Hodge numbers and ratios
- Topological invariants
- Complexity measures
- Normalized features

**For CICY Configurations**:
- Configuration matrix statistics
- Hodge number combinations
- Three-generation indicators
- Matrix complexity scores

**For F-theory Bases**:
- Graph structural features (nodes, edges, degree)
- Connectivity metrics (clustering, paths)
- Centrality measures
- Spectral features

**Usage**:
```bash
python scripts/20_build_features.py
```

**Output**:
- `data/processed/tables/ks_features.parquet`
- `data/processed/tables/cicy3_features.parquet`
- `data/processed/tables/fth6d_graph_features.parquet`

---

## Phase 3: Label Generation ⭐ KEY NOVELTY

### Toy EFT Stability Simulation (scripts/30)

**Purpose**: Generate stability labels using physics simulation

**Script**: `30_generate_labels_toy_eft.py`

**Method**:

1. **For each geometry**:
   - Sample synthetic flux parameters (F3, H3, g_s, etc.)
   - Build toy EFT scalar potential:
     ```
     V(φ) = V_flux + V_pert - V_np
     ```
   - Run numerical minimization to find critical points
   - Compute Hessian matrix at critical points
   - Analyze eigenvalues

2. **Label Assignment**:
   - `stable`: All eigenvalues > 0 (local minimum)
   - `unstable`: All eigenvalues < 0 (local maximum)
   - `saddle`: Mixed eigenvalues
   - `failed`: Minimization didn't converge

**Usage**:
```bash
python scripts/30_generate_labels_toy_eft.py
```

**Output**:
- `data/processed/labels/toy_eft_stability.parquet`

**Columns**:
- `geometry_id`, `dataset`, `n_moduli`
- `stability` (label)
- `potential_value` (V at critical point)
- `eigenvalues` (Hessian eigenvalues)
- `minimization_success` (bool)

**Why This Matters**:

This is simulation-based labeling - we generate labels computationally rather than hand-labeling. This enables:
- Large-scale labeled datasets
- Physics-motivated labels
- Controlled experiments
- Active learning (label on demand)

---

## Phase 4: Benchmark Splits

### Creating Standard Splits (scripts/40)

**Purpose**: Create reproducible train/val/test splits

**Script**: `40_make_splits.py`

**Split Types**:

1. **IID Split**: Random 70/15/15 stratified split
2. **OOD Complexity**: Train on simple, test on complex geometries
3. **OOD Dataset**: Train on KS+CICY, test on F-theory
4. **OOD Hodge**: Stratified by Hodge number bins

**Usage**:
```bash
python scripts/40_make_splits.py
```

**Output** (`data/processed/splits/`):
- `iid_split.json`
- `ood_complexity_split.json`
- `ood_dataset_fth6d.json`
- `ood_dataset_cicy3.json`

**Format**:
```json
{
  "train": [indices...],
  "val": [indices...],
  "test": [indices...],
  "split_type": "iid",
  "train_size": 7000,
  "val_size": 1500,
  "test_size": 1500
}
```

---

## Phase 5: Baseline Models

### Tabular Models (scripts/50)

**Purpose**: Baseline classification on tabular features

**Script**: `50_train_baseline_tabular.py`

**Models**:
- Logistic Regression
- Random Forest (100 trees)
- MLP (64→32 hidden units)

**Datasets**: KS, CICY (tabular features)

**Usage**:
```bash
python scripts/50_train_baseline_tabular.py
```

**Output** (`runs/<timestamp>/tabular/`):
- `metrics.json` - Performance metrics
- `*.pkl` - Trained models

### Graph Models (scripts/51)

**Purpose**: Baseline GNN classification on graph structures

**Script**: `51_train_baseline_graph.py`

**Models**:
- GraphSAGE (2-layer, 64 hidden)
- GCN (2-layer, 64 hidden)

**Dataset**: F-theory bases (graph structure)

**Usage**:
```bash
python scripts/51_train_baseline_graph.py
```

**Output** (`runs/<timestamp>/graph/`):
- `metrics.json` - Performance metrics
- `*.pt` - Trained PyTorch models

**Metrics Tracked**:
- Accuracy (train/val/test)
- F1 Score (macro)
- Classification report (per-class metrics)

---

## Phase 6: Active Learning ⭐ IMPACT DEMONSTRATION

### Active Learning Loop (scripts/60)

**Purpose**: Demonstrate compute-efficient vacuum search

**Script**: `60_active_learning_scan.py`

**Algorithm**:

```
1. Initialize with small labeled set (100 samples)
2. Loop:
   a. Train model on labeled set
   b. Select uncertain samples from unlabeled pool
   c. Run toy EFT simulation to label selected samples
   d. Add to labeled set
   e. Repeat
```

**Selection Strategies**:
- **Entropy**: Select samples with highest prediction entropy
- **Margin**: Select samples with smallest margin between top classes
- **Random**: Baseline (random selection)

**Usage**:
```bash
python scripts/60_active_learning_scan.py
```

**Output** (`runs/<timestamp>/active_learning/`):
- `active_learning_results.json` - Iteration history
- `final_model.pkl` - Final trained model

**Impact**:

Shows that ML can reduce the number of simulations needed by intelligently selecting which candidates to evaluate. This is the "discovery loop" - using ML to guide physics simulations.

---

## Running the Full Pipeline

### Quick Start (Using Existing Data)

If data is already downloaded:

```bash
# Parse data
python scripts/10_parse_ks.py
python scripts/11_parse_cicy3.py
python scripts/12_parse_fth6d.py

# Build features
python scripts/20_build_features.py

# Generate labels (this takes time!)
python scripts/30_generate_labels_toy_eft.py

# Create splits
python scripts/40_make_splits.py

# Train baselines
python scripts/50_train_baseline_tabular.py
python scripts/51_train_baseline_graph.py

# Run active learning
python scripts/60_active_learning_scan.py
```

### Full Pipeline (From Scratch)

```bash
# 1. Download data
python scripts/01_download_ks.py
python scripts/02_download_cicy3.py
python scripts/03_download_fth6d.py

# 2-6. Same as above
...
```

### Using Make

```bash
make data     # Download and process data
make features # Build features
make labels   # Generate labels
make train    # Train baselines
```

---

## Notebooks & Tutorials

### Quickstart Tutorial

`notebooks/01_quickstart_tutorial.ipynb`

**Contents**:
1. Load datasets
2. Explore features and Hodge numbers
3. Examine stability labels
4. Train a simple model
5. Analyze feature importance

**Usage**:
```bash
jupyter notebook notebooks/01_quickstart_tutorial.ipynb
```

---

## Key Design Decisions

### Why Parquet?

- Fast columnar storage
- Efficient compression
- Cross-language support
- Handles nested data

### Why Toy EFT?

- Computationally feasible
- Physics-motivated
- Generates realistic labels
- Enables large-scale experiments

### Why Active Learning?

- Demonstrates practical impact
- Reduces computational cost
- Guides expensive simulations
- Real-world applicability

---

## Performance & Scalability

### Dataset Sizes

- **KS**: ~1000 samples (placeholder, full dataset ~474M)
- **CICY**: ~7,890 configurations
- **F-theory**: ~1000 samples (placeholder, full dataset ~61,539)

### Computation Time

- **Parsing**: Minutes
- **Feature building**: Seconds
- **Label generation**: Hours (depends on n_samples)
- **Training baselines**: Minutes
- **Active learning**: Hours

### Optimization Tips

- Reduce `n_samples` in label generation for faster iteration
- Use parallel processing for label generation
- Cache intermediate results
- Use GPU for graph models

---

## Troubleshooting

### Common Issues

**1. Missing dependencies**
```bash
pip install -r requirements.txt
```

**2. Data not found**
```bash
# Run download scripts first
python scripts/01_download_*.py
```

**3. PyTorch Geometric not installed**
```bash
pip install torch torch-geometric
```

**4. Out of memory**
- Reduce batch size
- Process datasets in chunks
- Use smaller subsets for testing

---

## Next Steps

1. **Run full pipeline** with real data
2. **Experiment with EFT parameters** in script 30
3. **Add new features** in script 20
4. **Try different models** in scripts 50-51
5. **Customize active learning** strategy in script 60
6. **Contribute** new datasets or benchmarks

---

## Citation

If you use VacuaGym, cite:

```bibtex
@software{vacuagym2025,
  author = {Ahamed, Towsif},
  title = {VacuaGym: ML Benchmarks for String Theory Compactifications},
  year = {2025},
  url = {https://github.com/towsif/vacua-gym},
  version = {0.1.0-alpha}
}
```

And cite the original data sources (see data/external/mirrors_and_checksums/LICENSE_NOTES.md).

---

**Questions?** towsif.kuet.ac.bd@gmail.com
