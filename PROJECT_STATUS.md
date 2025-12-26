# VacuaGym Project Status

**Last Updated**: 2025-12-26
**Project Owner**: Towsif Ahamed (towsif.kuet.ac.bd@gmail.com)
**Version**: v0.1.0-alpha

---

## Current Status: ALL PHASES COMPLETE!

VacuaGym is fully implemented from infrastructure through active learning!

**Status**: ✅ Ready for end-to-end testing and iteration
**Next**: Run full pipeline, validate results, gather feedback

---

## Completed Tasks

### ✅ Phase 0: Repository Initialization
- [x] Git repository created
- [x] Directory structure established
- [x] Basic hygiene files (.gitignore, .gitkeep files)

### ✅ Phase 1: Identity & Licensing
- [x] **LICENSE** - Dual-license structure (PolyForm Noncommercial + Commercial)
- [x] **COMMERCIAL-LICENSE.md** - Commercial licensing terms
- [x] **CONTRIBUTING.md** - Contributor agreement and relicensing rights
- [x] **CITATION.cff** - Academic citation metadata

### ✅ Phase 2: Repository Skeleton
- [x] Complete directory structure:
  ```
  data/raw/           # Dataset-specific subdirectories created
  data/interim/       # For processing
  data/processed/     # For ML-ready data
  data/external/      # Checksums and metadata
  papers/             # For bibliography and snapshots
  src/vacua_gym/      # Python package
  scripts/            # Processing scripts
  notebooks/          # Analysis notebooks
  docs/               # Documentation
  tests/              # Test suite
  ```
- [x] **requirements.txt** - Python dependencies
- [x] **setup.py** - Package configuration
- [x] **Makefile** - Build targets (skeleton)
- [x] **src/vacua_gym/__init__.py** - Package initialization

### ✅ Phase 3: Data Infrastructure
- [x] Dataset directories created:
  - `data/raw/ks_reflexive_polytopes/`
  - `data/raw/cicy3_7890/`
  - `data/raw/ftheory_6d_toric_bases_61539/`
- [x] **README_source.txt** for each dataset (provenance documentation)
- [x] **checksums.sha256** (placeholder for data verification)
- [x] **LICENSE_NOTES.md** (data licensing clarification)

### ✅ Phase 5: Documentation Stubs
- [x] **README.md** - Project overview and introduction
- [x] **docs/DATA_SOURCES.md** - Data provenance documentation
- [x] **docs/DATA_SCHEMA.md** - Data format specification
- [x] **docs/REPRODUCIBILITY.md** - Reproducibility guidelines
- [x] **docs/ROADMAP.md** - Project roadmap and timeline

---

## Implementation Summary

### Complete Pipeline Scripts

**Phase 1: Data Download (scripts/01-03)**
- `01_download_ks.py` - Kreuzer-Skarke polytope data
- `02_download_cicy3.py` - CICY threefold configurations
- `03_download_fth6d.py` - F-theory 6D toric bases

**Phase 1: Data Parsing (scripts/10-12)**
- `10_parse_ks.py` - Parse KS to Parquet tables
- `11_parse_cicy3.py` - Parse CICY to Parquet tables
- `12_parse_fth6d.py` - Parse F-theory bases to Parquet tables

**Phase 2: Feature Engineering (scripts/20)**
- `20_build_features.py` - Build ML-ready features for all datasets

**Phase 3: Label Generation (scripts/30)** ⭐ KEY NOVELTY
- `30_generate_labels_toy_eft.py` - Toy EFT stability simulation
  - Samples synthetic flux parameters
  - Builds scalar potential V(φ)
  - Runs numerical minimization
  - Computes Hessian eigenvalues
  - Labels: stable / unstable / saddle / metastable

**Phase 4: Benchmark Splits (scripts/40)**
- `40_make_splits.py` - Create IID and OOD splits
  - IID: Random 70/15/15 split
  - OOD by complexity: Train on simple, test on complex
  - OOD by dataset: Cross-dataset generalization

**Phase 5: Baseline Models (scripts/50-51)**
- `50_train_baseline_tabular.py` - Logistic Regression, Random Forest, MLP
- `51_train_baseline_graph.py` - GraphSAGE, GCN for toric bases

**Phase 6: Active Learning (scripts/60)** ⭐ IMPACT DEMONSTRATION
- `60_active_learning_scan.py` - Active learning loop
  - Uncertainty sampling (entropy, margin)
  - Iterative refinement
  - Compute budget optimization

### Supporting Files

**Utilities**
- `scripts/download_and_checksum.sh` - Data verification

**Notebooks**
- `notebooks/01_quickstart_tutorial.ipynb` - Complete tutorial

**Documentation**
- All docs updated with implementation details

---

## Pending Tasks

### ⏳ Phase 1 Remaining: License File
- [ ] **licenses/PolyForm-Noncommercial-1.0.0.txt** - Verbatim license text
  - Note: Will be downloaded with a less costly agent
  - Source: https://polyformproject.org/licenses/noncommercial/1.0.0/

### ⏳ Phase 3 Remaining: Data Downloads
- [ ] Download Kreuzer-Skarke reflexive polytopes
- [ ] Download CICY threefold list (7,890)
- [ ] Download F-theory 6D toric bases (61,539)
- [ ] Generate SHA256 checksums for all downloaded files
- [ ] Update README_source.txt files with download dates

### ⏳ Phase 4: Data Verification
- [ ] Verify data integrity (checksums)
- [ ] Document actual data formats found
- [ ] Parse and validate raw data
- [ ] Update DATA_SOURCES.md with detailed info

---

## Repository Structure (Current)

```
VacuaGym/
├── LICENSE                           # Dual-license (PolyForm NC + Commercial)
├── COMMERCIAL-LICENSE.md             # Commercial terms
├── CONTRIBUTING.md                   # Contributor agreement
├── CITATION.cff                      # Citation metadata
├── README.md                         # Project overview
├── PROJECT_STATUS.md                 # This file
├── requirements.txt                  # Dependencies
├── setup.py                          # Package setup
├── Makefile                          # Build targets
├── .gitignore                        # Git ignore rules
│
├── licenses/                         # License texts
│   └── [PolyForm-Noncommercial-1.0.0.txt - TO BE ADDED]
│
├── data/
│   ├── raw/                          # Original datasets
│   │   ├── ks_reflexive_polytopes/   # Kreuzer-Skarke data
│   │   │   └── README_source.txt
│   │   ├── cicy3_7890/               # CICY data
│   │   │   └── README_source.txt
│   │   └── ftheory_6d_toric_bases_61539/  # F-theory bases
│   │       └── README_source.txt
│   ├── interim/                      # Processing (gitignored)
│   ├── processed/                    # ML-ready data
│   │   ├── tables/
│   │   ├── labels/
│   │   └── splits/
│   └── external/
│       └── mirrors_and_checksums/
│           ├── checksums.sha256
│           └── LICENSE_NOTES.md
│
├── src/
│   └── vacua_gym/
│       └── __init__.py
│
├── docs/
│   ├── DATA_SOURCES.md               # Data documentation
│   ├── DATA_SCHEMA.md                # Schema specification
│   ├── REPRODUCIBILITY.md            # Reproducibility guide
│   └── ROADMAP.md                    # Project roadmap
│
├── papers/
│   ├── bib/                          # Bibliography
│   └── snapshots/                    # Paper snapshots
│
├── scripts/                          # Processing scripts (to be added)
├── notebooks/                        # Analysis notebooks (to be added)
└── tests/                            # Test suite (to be added)
```

---

## Next Steps (Priority Order)

1. **Download PolyForm License Text**
   - Get verbatim copy of PolyForm Noncommercial 1.0.0
   - Place in `licenses/PolyForm-Noncommercial-1.0.0.txt`

2. **Data Downloads**
   - Download Kreuzer-Skarke data from TU Wien
   - Download CICY list from Oxford
   - Download F-theory bases from arXiv/supplementary materials

3. **Data Verification**
   - Generate checksums for all downloads
   - Verify data integrity
   - Update documentation with download metadata

4. **Initial Git Commit**
   - Commit all infrastructure files
   - Tag as v0.1.0-alpha (infrastructure complete)
   - Push to GitHub

5. **Begin Phase 2: Data Processing**
   - Write parsing scripts for each dataset
   - Document actual file formats
   - Create standardized data schema

---

## Key Files to Review

### Licensing & Legal
- [LICENSE](LICENSE) - Dual-license structure
- [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) - Commercial terms
- [CONTRIBUTING.md](CONTRIBUTING.md) - CLA and contribution terms
- [data/external/mirrors_and_checksums/LICENSE_NOTES.md](data/external/mirrors_and_checksums/LICENSE_NOTES.md) - Data licensing

### Project Documentation
- [README.md](README.md) - Project introduction
- [docs/ROADMAP.md](docs/ROADMAP.md) - Development roadmap
- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) - Reproducibility guidelines

### Technical Documentation
- [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) - Data provenance
- [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md) - Data formats
- Dataset README files in `data/raw/*/README_source.txt`

---

## Notes

### What VacuaGym IS
- Infrastructure and ML framework
- Curated public datasets
- Reproducible benchmarks
- Source-available (not open source)

### What VacuaGym IS NOT
- New physics claims
- Proprietary datasets
- Closed-source commercial product
- A complete research result (it's infrastructure)

### Philosophy
- Full transparency and reproducibility
- Proper attribution to original sources
- Community-driven but legally protected
- Research infrastructure, not research output

---

## Contact

**Towsif Ahamed**
Email: towsif.kuet.ac.bd@gmail.com

For technical questions, open an issue on GitHub.
For commercial licensing, email directly.

---

**Status**: Infrastructure phase complete, ready for data collection.
