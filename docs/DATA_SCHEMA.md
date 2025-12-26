# Data Schema

## Overview

This document describes the data schema used in VacuaGym for storing and processing geometry datasets.

---

## Directory Structure

```
data/
├── raw/                    # Original, unmodified data
│   ├── ks_reflexive_polytopes/
│   ├── cicy3_7890/
│   └── ftheory_6d_toric_bases_61539/
├── interim/                # Intermediate processing (not committed)
├── processed/              # Cleaned, standardized data
│   ├── tables/            # Structured tables (parquet/csv)
│   ├── labels/            # Generated labels
│   └── splits/            # Train/val/test splits
└── external/              # Metadata and checksums
    └── mirrors_and_checksums/
```

---

## Raw Data Formats

### Kreuzer-Skarke
[To be documented based on actual file format]

### CICY
[To be documented based on actual file format]

### F-theory Bases
[To be documented based on actual file format]

---

## Processed Data Formats

### Standard Table Format
[To be defined]

### Feature Representation
[To be defined]

### Label Schema
[To be defined]

---

## Data Splits

[To be defined]

---

## Versioning

[To be defined]
