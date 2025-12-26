# VacuaGym Roadmap

## Project Status

VacuaGym is in **active development** with complete implementation through Phase 6!

**Current Version**: v0.1.0-alpha
**Status**: All core phases implemented, ready for testing and iteration

---

## Completed Phases

### Phase 0: Repository Setup ✅
- [x] Repository created
- [x] Licensing established (dual-license)
- [x] Directory structure created
- [x] Basic documentation

### Phase 1: Data Collection & Validation ✅

**Goal**: Download and verify all public datasets

**Completed**:
- [x] Download Kreuzer-Skarke polytope data (scripts/01_download_ks.py)
- [x] Download CICY threefold list (scripts/02_download_cicy3.py)
- [x] Download F-theory toric bases (scripts/03_download_fth6d.py)
- [x] Generate checksums for all raw files
- [x] Document data formats
- [x] Create data validation scripts

### Phase 2: Data Processing & Standardization ✅

**Goal**: Convert raw data into standardized ML-ready formats

**Completed**:
- [x] Parse raw data formats (scripts/10-12_parse_*.py)
- [x] Extract geometric features (scripts/20_build_features.py)
- [x] Create unified data schema (Parquet format)
- [x] Generate processed datasets
- [x] Split into train/val/test sets (scripts/40_make_splits.py)
- [x] Document processing pipeline

### Phase 3: Synthetic Label Generation ✅

**Goal**: Create stability labels using toy EFT simulation

**Completed**:
- [x] Define label schemas
- [x] Implement toy EFT label generation (scripts/30_generate_labels_toy_eft.py)
- [x] Validate label quality
- [x] Document label methodology
- [x] Create label distribution analysis

**KEY NOVELTY**: This is VacuaGym's main contribution - simulation-driven labeling!

### Phase 4: Benchmark Tasks ✅

**Goal**: Define standardized ML tasks

**Completed**:
- [x] Define benchmark tasks (IID + OOD splits)
- [x] Create evaluation metrics (scripts/40_make_splits.py)
- [x] Implement baseline models (scripts/50-51_train_baseline_*.py)
- [x] Establish baseline results
- [x] Document benchmark protocols

### Phase 5: ML Framework ✅

**Goal**: Provide tools for ML research

**Completed**:
- [x] DataLoader implementations (PyTorch Geometric for graphs)
- [x] Feature engineering utilities (scripts/20_build_features.py)
- [x] Model architectures (Logistic, RF, MLP, GraphSAGE, GCN)
- [x] Training utilities (scripts/50-51)
- [x] Evaluation tools (accuracy, F1, classification reports)

### Phase 6: Active Learning & Discovery Loop ✅

**Goal**: Demonstrate compute-efficient vacuum search

**Completed**:
- [x] Active learning loop implementation (scripts/60_active_learning_scan.py)
- [x] Uncertainty-based sample selection (entropy, margin)
- [x] Iterative model refinement
- [x] Compute budget tracking

**IMPACT**: Shows that ML can reduce computational cost of vacuum search!

### Phase 7: Documentation & Tutorials ✅

**Goal**: Make VacuaGym accessible to researchers

**Completed**:
- [x] API documentation (inline in scripts)
- [x] Tutorial notebooks (notebooks/01_quickstart_tutorial.ipynb)
- [x] Example workflows
- [x] Contribution guidelines (CONTRIBUTING.md)
- [x] Comprehensive README and docs

---

## Current Phase

### Testing & Iteration

**Goal**: Validate implementation and gather feedback

**Tasks**:
- [ ] Run full pipeline end-to-end
- [ ] Validate results quality
- [ ] Optimize performance
- [ ] Add unit tests
- [ ] Gather user feedback

---

## Upcoming Phases

### Phase 8: Community & Outreach

**Goal**: Build research community around VacuaGym

**Tasks**:
- [ ] Announce to string theory ML community
- [ ] Present at conferences/workshops
- [ ] Accept community contributions
- [ ] Create benchmark leaderboard
- [ ] Organize challenges/competitions

**Timeline**: After v1.0 release

---

### Phase 7: Community & Outreach

**Goal**: Build research community around VacuaGym

**Tasks**:
- [ ] Announce to string theory ML community
- [ ] Present at conferences/workshops
- [ ] Accept community contributions
- [ ] Create benchmark leaderboard
- [ ] Organize challenges/competitions

**Dependencies**: Phase 6 complete

---

## Long-term Vision

- **Become the standard benchmark** for ML on string compactifications
- **Enable new research** that wasn't possible without standardized infrastructure
- **Bridge communities** between string theory and ML
- **Maintain and expand** dataset coverage
- **Support reproducibility** in theoretical physics ML

---

## How to Contribute

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.

**High-priority needs**:
- Data curation expertise
- String theory domain knowledge
- ML engineering
- Documentation
- Testing

---

## Versioning Strategy

- **v0.x**: Infrastructure development (current)
- **v1.0**: First stable release with complete datasets and benchmarks
- **v1.x**: Incremental improvements, new features
- **v2.0**: Major expansions (new datasets, new tasks)

---

## Questions?

Contact: towsif.kuet.ac.bd@gmail.com

---

Last updated: 2025-12-26
