# Provenance: real ESA OPS-SAT data

`dataset_partial.csv` is a real subset of the public **OPSSAT-AD** benchmark
(Ruszczak, Kotowski, Evans, Nalepa — "The OPS-SAT benchmark for detecting
anomalies in satellite telemetry", 2024, arXiv:2407.04730), released by
ESA / KP Labs / Opole University of Technology / Silesian University of
Technology on Zenodo:

- Landing page: https://zenodo.org/records/12588359
- DOI: 10.5281/zenodo.12588359
- License: CC-BY-4.0
- File fetched: `dataset.csv` (the featurized, per-segment file — one row
  per manually labeled telemetry segment, with real channel IDs from the
  ESA OPS-SAT CubeSat and a real binary `anomaly` label)

**What is real here:** every value in this CSV — channel names
(`CADC0872`, `CADC0892`, `CADC0874`, `CADC0884`, `CADC0873`, ...),
segment duration/length, mean, variance, std, kurtosis, skew, peak counts,
and the `anomaly` / `train` labels — is copied verbatim from ESA's real
released dataset.

**What is NOT included:** the companion `segments.csv` (the raw
per-timestamp telemetry samples, ~584 MB) could not be downloaded in this
sandboxed environment (only `raw.githubusercontent.com`-class domains are
reachable from the code container, and `web_fetch` truncates large files).
`dataset_partial.csv` keeps a real 14-row slice across 5 distinct real
channels (not ~500 rows as an earlier draft of this note claimed — corrected
during code review; always trust `wc -l` over a prior description).

**How GaganMitra uses it:** `EsaAdbAdapter` reconstructs a per-timestamp
telemetry series for each segment by sampling from each segment's *real*
measured mean/std/duration (see `load_real_opssat_segments()` in
`src/adapters/esa_adb_adapter.py`). The reconstructed per-timestamp values
are therefore synthetic, but every parameter driving that reconstruction —
channel identity, segment timing, and the anomaly ground truth — is real
ESA data, not invented. This is disclosed in the adapter's docstring and in
the README; do not describe it as "raw real telemetry" without that
caveat.

If you can download the full `segments.csv` yourself (e.g. from a machine
with unrestricted internet access), point `EsaAdbAdapter` at it directly for
a fully real-data run — see the `load_real_opssat_segments()` docstring for
where to plug it in.
