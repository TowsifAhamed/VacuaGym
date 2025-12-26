# Mirrors and checksums

This folder contains locally mirrored pages/files and a checksum manifest to aid reproducible data acquisition for VacuaGym.

What was mirrored by `scripts/download_and_checksum.sh`:

- `mirrors/kreuzer_sk/` — Kreuzer-Skarke Calabi-Yau site index and pages
- `mirrors/cicy_list/` — CICY threefold list index and linked files (plain-text and Mathematica index pages)
- `mirrors/f_theory_6d/` — arXiv abstract page for Morrison & Taylor (1201.1943)


Notes on license and datasets

- The PolyForm Noncommercial 1.0.0 license text is already included in the repository; no manual license action is required.

Generating and verifying checksums:

- The script `scripts/download_and_checksum.sh` will (re)mirror the above sources and regenerate `checksums.sha256`.
- To verify checksums locally:

```
sha256sum -c data/external/mirrors_and_checksums/checksums.sha256
```

Status notes:

- `Kreuzer-Skarke` and `CICY` dataset files have been mirrored (see `mirrors/` subdirectories).
- `F-theory 6D` dataset: the paper abstract and TeX source were mirrored, but the full toric-bases dataset (61,539 bases) was not present in the paper source. If you have a preferred authoritative host for the F-theory bases (author website, supplementary archive, or an institutional mirror), provide it and I will mirror and checksum it.
