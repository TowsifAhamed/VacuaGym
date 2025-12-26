# Reproducibility Guide

## Overview

VacuaGym is designed for full reproducibility. This document describes how to reproduce all results.

---

## Principles

1. **Full Provenance**: Every data source is documented
2. **Checksums**: All downloaded files are verified
3. **Version Control**: Data processing code is versioned
4. **Deterministic**: Random seeds are fixed
5. **Transparent**: All steps are documented

---

## Reproducing Data Downloads

### Prerequisites
- Internet connection
- Sufficient disk space (estimate: TBD)
- Python 3.8+

### Steps

1. Clone repository
2. Run download scripts (to be implemented)
3. Verify checksums
4. Process raw data

**Detailed instructions**: [To be filled]

---

## Reproducing Data Processing

[To be documented]

---

## Reproducing Experiments

[To be documented]

---

## Reproducing Benchmarks

[To be documented]

---

## Environment Setup

### Using pip
```bash
pip install -r requirements.txt
```

### Using conda
[To be documented]

---

## Data Provenance Tracking

All raw data files include:
- Source URL
- Download date
- SHA256 checksum
- Original file name

See `data/raw/*/README_source.txt` for details.

---

## Checksums

After downloading data, verify with:
```bash
cd data/external/mirrors_and_checksums
sha256sum -c checksums.sha256
```

---

## Known Issues

[To be documented as they arise]

---

## Contact

Questions about reproducibility: towsif.kuet.ac.bd@gmail.com
