# VacuaGym Quick Start Guide

**Status**: Infrastructure phase - no data yet

---

## For New Contributors

### 1. Clone the Repository
```bash
git clone https://github.com/towsif/vacua-gym.git
cd vacua-gym
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install VacuaGym in development mode
pip install -e .
```

### 3. Understand the Project
Read these files in order:
1. [README.md](README.md) - Project overview
2. [PROJECT_STATUS.md](PROJECT_STATUS.md) - Current status
3. [docs/ROADMAP.md](docs/ROADMAP.md) - Development plan
4. [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute

---

## Current Status

**Phase**: Infrastructure setup (90% complete)

**What's Done**:
- Repository structure
- Licensing framework
- Documentation stubs
- Package skeleton

**What's Next**:
- Download datasets (Kreuzer-Skarke, CICY, F-theory bases)
- Generate checksums
- Parse raw data formats
- Create processing pipeline

---

## Repository Structure

```
VacuaGym/
├── src/vacua_gym/      # Python package (empty, to be filled)
├── data/raw/           # Raw datasets (to be downloaded)
├── data/processed/     # ML-ready data (to be created)
├── scripts/            # Processing scripts (to be written)
├── notebooks/          # Analysis notebooks (to be created)
├── docs/               # Documentation (stubs created)
└── tests/              # Test suite (to be written)
```

---

## For Users (Future)

**Note**: VacuaGym is not ready for use yet. This section describes the future workflow.

### Installation (Future)
```bash
pip install vacua-gym  # Not yet published
```

### Basic Usage (Future)
```python
import vacua_gym

# Load a dataset
dataset = vacua_gym.datasets.load_cicy()

# Access features and labels
features = dataset.features
labels = dataset.labels

# Use in PyTorch
loader = vacua_gym.get_dataloader(dataset, batch_size=32)
```

---

## Data Download Instructions (To Be Implemented)

**Current Status**: Manual download required

### Kreuzer-Skarke Data
```bash
# Instructions to be added after testing download
# Expected location: data/raw/ks_reflexive_polytopes/
```

### CICY Data
```bash
# Instructions to be added after testing download
# Expected location: data/raw/cicy3_7890/
```

### F-theory Bases
```bash
# Instructions to be added after testing download
# Expected location: data/raw/ftheory_6d_toric_bases_61539/
```

---

## Development Workflow

### Running Tests (Future)
```bash
make test
# or
pytest tests/
```

### Building Documentation (Future)
```bash
make docs
```

### Data Processing (Future)
```bash
make data
```

---

## Important Files

### Legal/Licensing
- [LICENSE](LICENSE) - Dual-license terms
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contributor agreement

### Documentation
- [README.md](README.md) - Project overview
- [PROJECT_STATUS.md](PROJECT_STATUS.md) - Current status
- [docs/ROADMAP.md](docs/ROADMAP.md) - Development roadmap

### Configuration
- [requirements.txt](requirements.txt) - Dependencies
- [setup.py](setup.py) - Package configuration
- [Makefile](Makefile) - Build commands

---

## Getting Help

- **Technical Questions**: Open an issue on GitHub
- **Commercial Licensing**: towsif.kuet.ac.bd@gmail.com
- **Contributing**: Read [CONTRIBUTING.md](CONTRIBUTING.md) first

---

## Before You Contribute

1. Read [CONTRIBUTING.md](CONTRIBUTING.md)
2. Understand the dual-license model
3. Agree to the contributor license agreement
4. Open an issue to discuss your contribution
5. Submit a pull request

---

## Citation

If you use VacuaGym in your research (when ready):

```bibtex
@software{vacuagym2025,
  author = {Ahamed, Towsif},
  title = {VacuaGym: ML Benchmarks for String Theory Compactifications},
  year = {2025},
  url = {https://github.com/towsif/vacua-gym}
}
```

Also cite the original data sources! See [data/external/mirrors_and_checksums/LICENSE_NOTES.md](data/external/mirrors_and_checksums/LICENSE_NOTES.md).

---

**Last Updated**: 2025-12-26
**Project Status**: Infrastructure phase - not ready for use
**Contact**: towsif.kuet.ac.bd@gmail.com
