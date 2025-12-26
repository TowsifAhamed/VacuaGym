# VacuaGym

**A source-available, reproducible, simulation-driven benchmark and ML framework for exploring string/M-theory compactifications**

---

## What is VacuaGym?

VacuaGym is infrastructure for machine learning research on string theory compactifications. It provides:

- **Public Geometry Datasets**: Curated collections of Calabi-Yau manifolds and related geometric objects
- **Reproducible Benchmarks**: Standardized tasks for ML research on theoretical physics
- **ML Framework**: Tools for exploring stability, moduli spaces, and physical properties

## Key Features

- **Source-Available**: Full transparency for academic research
- **Dataset-Driven**: Built on publicly available geometry databases
- **Reproducible**: Complete provenance tracking and version control
- **Benchmark Suite**: Standardized tasks for comparing ML approaches

## Data Sources

VacuaGym uses public datasets from the string theory and algebraic geometry communities:

- **Kreuzer-Skarke Database**: Reflexive polytopes for Calabi-Yau threefolds
- **CICY List**: Complete intersection Calabi-Yau threefolds
- **F-theory Bases**: Toric base surfaces for elliptic fibrations

All original data sources are properly cited and attributed. See documentation for details.

## Mirrors and checksums

We provide a small set of mirrored dataset pages and a checksum file under `data/external/mirrors_and_checksums` for reproducibility. Use the provided script to mirror sources and regenerate checksums:

```
bash scripts/download_and_checksum.sh
```

Files currently mirrored by the script (as of this update):

- `Kreuzer-Skarke` site index (mirrored to `data/external/mirrors_and_checksums/mirrors/kreuzer_sk`)
- `CICY` list (mirrored to `data/external/mirrors_and_checksums/mirrors/cicy_list`)
- `F-theory 6D` paper abstract page (mirrored to `data/external/mirrors_and_checksums/mirrors/f_theory_6d`)

The PolyForm Noncommercial 1.0.0 license text is included in the repository; no manual action is required for the license.

Dataset verification and status

Use the mirror script to (re)fetch dataset pages and regenerate checksums:

```
bash scripts/download_and_checksum.sh
```

To verify checksums:

```
sha256sum -c data/external/mirrors_and_checksums/checksums.sha256
```

Current mirror status (updated):

- `Kreuzer-Skarke`: several data files mirrored under `data/external/mirrors_and_checksums/mirrors/kreuzer_sk/` (including `W/` and `pub/misc/` files).
- `CICY`: `cicylist.txt` and index mirrored under `data/external/mirrors_and_checksums/mirrors/cicy_list/`.
- `F-theory 6D`: only the paper abstract and TeX source were mirrored; the full dataset of toric bases (61,539) was not available from the paper source. See the mirrors README for next steps.

## License

VacuaGym is **source-available** software with a dual-license model:

- **Noncommercial Use**: [PolyForm Noncommercial License 1.0.0](LICENSE)
  - Free for academic research, education, and personal use

- **Commercial Use**: Requires separate commercial license
  - See [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) for details
  - Contact: towsif.kuet.ac.bd@gmail.com

## Project Status

This project is under active development. We are building:

1. Data collection and validation infrastructure
2. Benchmark task definitions
3. Baseline ML models
4. Documentation and tutorials

## Contributing

We welcome contributions from the community. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting pull requests.

By contributing, you agree to license your contributions under the noncommercial license and grant relicensing rights for commercial use.

## Citation

If you use VacuaGym in your research, please cite:

```bibtex
@software{vacuagym2025,
  author = {Ahamed, Towsif},
  title = {VacuaGym: ML Benchmarks for String Theory Compactifications},
  year = {2025},
  url = {https://github.com/towsif/vacua-gym}
}
```

## Contact

**Towsif Ahamed**
towsif.kuet.ac.bd@gmail.com

---

**Disclaimer**: This is infrastructure for ML research. We make no claims about new physics. VacuaGym provides datasets, benchmarks, and tools that were previously unavailable to the community.
