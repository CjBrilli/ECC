## Overview

DSN Doppler measurements contain signatures of:

- Solar wind turbulence (scintillation)  
- Large-scale structures (CIRs)  
- Transient events (CMEs)  

This project extracts those signals through a three-stage correction pipeline:

1. Remove solar elongation dependence  
2. Remove CIR-scale background structure  
3. Detect transient enhancements  

The result is a robust method for identifying solar disturbances using radio tracking data.

---

## Method Summary

### 1. Phase Scintillation

Doppler residuals are converted into phase:

\[
\phi(t) = 2\pi \int f(t)\,dt
\]
[
\phi(t) = 2\pi \int f(t),dt
]

Band-limited phase RMS is computed using a power spectral density (PSD) method over:

\[
3 \times 10^{-4} \le f \le 3 \times 10^{-2} \ \text{Hz}
\]

---

### 2. Elongation Correction

Phase scintillation depends strongly on solar elongation (SEP).

A quiet baseline is constructed:

\[
\phi_{\text{expected}} = f(\text{elongation})
\]

The signal is normalised:

\[
\text{phase\_ratio} = \frac{\phi_{\text{observed}}}{\phi_{\text{expected}}}
\]

---

### 3. CIR Detection

CIRs are identified as long-duration enhancements using:

- 12-hour smoothing  
- hysteresis thresholds  
- minimum duration constraint (> 24 hours)  

---

### 4. CME / Transient Detection

CIR background is removed:

\[
\text{clean\_signal} = \frac{\text{phase\_ratio}}{\text{phase\_smooth}}
\]

Transient events are identified via:

- thresholding (clean_signal > 3)  
- duration filtering (0.25–24 hours)  

---

## Repository Structure

Repository Structure
ECC/
│
├── config/              # Configuration files and parameters
├── data_links/          # References to external datasets
├── inputs/              # Input data (local or linked)
├── notebooks/           # Analysis notebooks
│   ├── daily_rms/
│   ├── phase_scintillation/
│   ├── detection_pipeline/
│   └── multi_year_analysis/
│
├── src/                 # Core processing modules
│   ├── io.py
│   ├── phase.py
│   ├── spectral.py
│   ├── tec_model.py
│   └── detection.py
│
├── README.md
├── LICENSE
└── .gitignore

---

## Workflow

### Notebook 1 — Daily RMS Analysis
- Computes Doppler RMS per day  
- Merges with solar elongation  
- Compares with DSN theoretical model  

### Notebook 2 — Phase Scintillation
- Converts Doppler to phase  
- Computes PSD-based phase RMS in sliding windows  
- Produces `phase_windows_<year>.csv`  

### Notebook 3 — Detection Pipeline
- Removes elongation dependence  
- Detects CIR regions  
- Detects CME-like transients  
- Produces event catalogues  

### Notebook 4 — Multi-Year Analysis
- Compares results across 2010–2014  
- Generates summary statistics and figures  

---

## Key Parameters

| Parameter            | Value                | Description |
|---------------------|---------------------|-------------|
| Phase band          | 3e-4 – 3e-2 Hz      | Scintillation frequency range |
| Window size         | 20 min              | Phase computation |
| Step size           | 10 min              | Window overlap |
| CIR smoothing       | 12 hr               | Background scale |
| CIR thresholds      | 1.4 / 1.2           | Hysteresis |
| Transient threshold | 3.0                 | CME detection |
| Transient step      | 20 min              | Critical for correct results |
---

### Horizons Solar Elongation File

The pipeline requires a plain text JPL Horizons ephemeris file containing solar elongation values.

The parser expects:

- a data block between `$$SOE` and `$$EOE`
- timestamps in the first two columns
- solar elongation stored in the Horizons `S-O-T /r` field

In each data row, the elongation is read as the numeric value immediately before `/L` or `/T`.

Example:

```text
$$SOE
2010-Jan-01 00:00  ...  2.6407 /L  ...
2010-Jan-02 00:00  ...  2.4124 /L  ...
$$EOE

---

## Outputs

For each year:

**Daily metrics**
- Doppler RMS vs SEP  

**Phase windows**
- Band-limited phase scintillation  

**CIR catalogue**
- Start/end times  
- Duration  
- Signal strength  

**Transient event catalogue**
- CME-like detections  
- Peak and median signals  

---

## Validation

The pipeline is validated by:

- Reproducing consistent CIR counts across years  
- Stable transient detection after parameter alignment  
- Physically consistent separation of:
  - elongation effects  
  - CIR background  
  - transient disturbances  

---

## Key Insight

This method demonstrates that DSN Doppler tracking data contains measurable and separable signatures of solar wind structures, including CMEs.

---

## Requirements

- Python 3.10+
- NumPy
- pandas
- matplotlib
- SciPy

---

## Usage

1. Select a year in the config file:

   ```python
   from config.settings_2011 import *

---
## Author

Caleb

Developed as part of astrophysical data analysis using DSN tracking data of Venus Express.

Affiliation: University of Tasmania, Honours Research Project
Supervisor: Dr. Guifre Moleras

# DSN-Based Detection of CIRs and CMEs from Doppler Scintillation

This repository implements a complete, reproducible pipeline for detecting solar wind structures — specifically Co-rotating Interaction Regions (CIRs) and Coronal Mass Ejections (CMEs) — using Deep Space Network (DSN) Doppler tracking data of *Venus Express (VEX)*.

The method uses phase scintillation analysis and a physically motivated normalisation pipeline to isolate heliospheric disturbances from geometric and background effects.

---
## License

This project is licensed under the MIT License. See the LICENSE file for details.
---
