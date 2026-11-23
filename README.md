# Google Pixel 9 Camera Characterization Pipeline

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV 5.0+](https://img.shields.io/badge/opencv-5.0%2B-green.svg)](https://opencv.org/)

**Author:** Natalie Montano  
**Course:** EE 513 — Portland State University (Fall 2026)  
**Repository:** [https://github.com/nmmont/camera_characterization/](https://github.com/nmmont/camera_characterization/)

---

## 1. Problem Statement

Smartphone cameras apply non-linear heavy internal processing (noise reduction, sharpening, and color grading) before saving standard JPEGs. These steps obscure the underlying optical and sensor physics of the hardware.

To measure the physical performance of the Google Pixel 9 primary camera ($6.90\text{ mm}$, $f/1.68$, $25\text{ mm}$ full-frame equivalent, 50 MP Quad-Bayer sensor binned $2\times 2$ to 12.5 MP raw), this project bypasses the ISP by analyzing raw 10-bit Bayer DNG captures acquired via Open Camera (`Camera2 API`). The pipeline characterizes four core parts in the linear stages of a typical ISP —**geometry**, **spatial resolution**, **sensor noise & dynamic range**, and **color accuracy**—and verifies every stage against analytic solutions, synthetic models, and an automated unit test suite.

---

## 2. Methodology

Every analytical module is validated on synthetic ground truth prior to processing hardware DNGs, and operates directly on linear raw sensor data (`black_level = 63.9841 ± 0.0138 DN`, `white_level = 1023 DN`, `GBRG` CFA layout):

1. **Geometric Calibration & Distortion Modeling (`geometry.py`):**
   * Performs multi-view planar Zhang calibration (`cv2.calibrateCameraExtended`) across $N = 68$ raw checkerboard captures ($6 \times 9$ inner corners, $19.0 \pm 0.5\text{ mm}$ squares) spanning working distances from $0.3\text{ m}$ to $0.6\text{ m}$ (excluding $0.2\text{ m}$ close-range captures due to boundary clipping and elevated $1.753\text{ px}$ RMS residuals).
   * Solves for the intrinsic camera matrix $K$ and 5 distortion coefficients ($k_1, k_2, p_1, p_2, k_3$), cross-validating effective pixel pitch ($2.2654\ \mu\text{m}$) against the $35\text{ mm}$ crop-factor estimate ($2.34\ \mu\text{m}$) and verifying straight-line undistortion improvement.

2. **Spatial Resolution & Slanted-Edge MTF50 (`resolution.py`):**
   * Evaluates un-demosaiced single-plane green (`G1`) ROIs extracted from $N = 25$ raw captures ($5$ frames across each of 5 field zones: Center, Top-Left, Top-Right, Bottom-Left, Bottom-Right) of an ISO 12233 slanted-edge target at a $1.0\text{ m}$ working distance (ISO 22, $1/10\text{ s}$).
   * Enforces automated pre-MTF quality control checks (`run_qc_audit`): edge tilt angle within $3^\circ\text{–}7^\circ$ ($R^2 > 0.90$), zero black/white clipping, and $\ge 20\text{ px}$ flat dark/bright margins before computing the super-sampled Edge Spread Function (ESF), Hann-windowed Line Spread Function (LSF), and MTF50.

3. **Sensor Noise, Photon Transfer Curve & Dynamic Range (`noise.py`, `dynamics.py`):**
   * Characterizes base ISO 22 temporal noise within a central $256 \times 256$ `G1` patch across $24$ flat-field DNGs ($11$ exposure pairs from $1/45\text{ s}$ to $1.5\text{ s}$) and $30$ lens-capped dark frames ($N = 5$ across $6$ exposure times from $1/39695\text{ s}$ to $16.0\text{ s}$).
   * Uses the two-frame difference method ($\sigma^2 = \frac{1}{2}\text{Var}[A - B]$) below $70\%$ saturation to eliminate Fixed-Pattern Noise (FPN) and PRNU, fitting conversion gain $K$, read noise floor $\sigma_r$, thermal dark current $I_{\text{dark}}$, full-well capacity, and dynamic range ($\text{SNR} = 1$).

4. **Color Calibration & CIEDE2000 Evaluation (`color.py`):**
   * Groups un-demosaiced $2\times 2$ Bayer superpixels into quarter-resolution linear RGB $[R, G_{\text{avg}}, B]$, divides by an unsaturated flat-field white reference sheet for lens shading correction, applies von Kries neutral white balance on Patch 21 (Neutral 6.5), and fits a $3\times 3$ Color Correction Matrix (CCM) via linear least squares into CIE 1931 XYZ (D50 PCS).
   * Evaluates perceptual color error ($\Delta E_{00}$) across repeated LED captures ($N = 3$) and cross-tests generalization across Incandescent, Overhead Fluorescent, and Outdoor Daylight illuminants.

---

## 3. Master Results Summary (Google Pixel 9)

| Subsystem | Parameter Name | Measured Value | Uncertainty ($\pm 1\sigma$) | Units | Method / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hardware** | Sensor & Raw Format | $4080 \times 3072$ (`GBRG`) | — | px | 10-bit DNG, $2\times 2$ binned from 50 MP |
| **Geometry** | Focal Length ($f_x, f_y$) | $3048.03,\ 3043.67$ | $6.54,\ 6.52$ | px | Zhang calibration ($N = 68$ views, $0.3\text{–}0.6\text{ m}$) |
| **Geometry** | Mean Focal Length ($f_{\text{mean}}$) | $3045.85$ | $4.62$ | px | Weighted mean ($f_{\text{mm}} = 6.90 \pm 0.10\text{ mm}$) |
| **Geometry** | Principal Point ($c_x, c_y$) | $1509.05,\ 2065.80$ | $2.81,\ 2.68$ | px | Optical center offset |
| **Geometry** | Radial Distortion ($k_1, k_2, k_3$) | $+0.0418,\ -0.1081,\ +0.1003$ | $0.0018,\ 0.0056,\ 0.0062$ | — | 5-parameter model |
| **Geometry** | Tangential Distortion ($p_1, p_2$) | $+0.00116,\ +0.00124$ | $0.00023,\ 0.00020$ | — | Lens tilt & decentering |
| **Geometry** | Reprojection Error (RMS) | $0.9595$ | $0.05$ | px | Global fit residual ($33.7\%$ edge bend reduction) |
| **Geometry** | Effective Pixel Pitch | $2.2654$ | $0.0034$ | $\mu\text{m}$ | $f_{\text{mm}} / f_{\text{px}}$ ($3.2\%$ diff vs. $35\text{ mm}$ crop) |
| **Resolution** | Center MTF50 | $0.3239$ | $0.0631$ | $\text{cy/green-px}$ | ISO 12233 slanted-edge ($0.0715\ \text{cy/}\mu\text{m}$, $N=5$) |
| **Resolution** | Corner Mean MTF50 | $0.2381$ | $0.0761$ | $\text{cy/green-px}$ | Mean across 4 corners ($0.0526\ \text{cy/}\mu\text{m}$, $N=20$) |
| **Resolution** | Corner Worst MTF50 | $0.1804$ | $0.0301$ | $\text{cy/green-px}$ | Top-Left corner ROI ($0.0398\ \text{cy/}\mu\text{m}$) |
| **Noise & DR** | Conversion Gain ($K$) | $0.01476$ | $0.000037$ | $\text{DN/e}^-$ | Pair-differencing PTC fit ($67.7\ \text{e}^-/\text{DN}$) |
| **Noise & DR** | Read Noise Floor ($\sigma_r$) | $57.76$ | $2.89$ | $\text{e}^-$ | $0.853\text{ DN}$ (matches vendor `NoiseProfile` within $0.6\%$) |
| **Noise & DR** | Dark Current ($I_{\text{dark}}$) | $16.46$ | $0.31$ | $\text{e}^-/\text{s}$ | Dark exposure series ($1/39695\text{ s}$ to $16.0\text{ s}$) |
| **Noise & DR** | Full-Well Capacity ($N_{\text{sat}}$) | $64,965.7$ | $--$ | $\text{e}^-$ | Linear capacity $(1023 - 63.98) / K$ |
| **Noise & DR** | Dynamic Range | $10.14$ | $0.07$ | stops | $61.02\text{ dB}$ at $\text{SNR}=1$ (vendor: $10.13$ stops) |
| **Color** | Uncalibrated Error (LED) | $12.34$ | $4.72$ | $\Delta E_{00}$ | White-balanced raw RGB mapped to sRGB |
| **Color** | Calibrated Error (LED) | $3.53$ | $0.06$ | $\Delta E_{00}$ | von Kries WB + $3\times 3$ CCM ($N=3$ LED charts) |
| **Color** | Worst-Patch Error (LED) | $11.19$ | $0.13$ | $\Delta E_{00}$ | Patch 15 (Red) across $N=3$ LED captures |

---

## 4. Repository Structure

```text
camera_characterization/
├── config.yaml                              # Paths, checkerboard specs, slanted-edge ROIs, & ColorChecker bboxes
├── calibration_working_distance_analysis.py # Analyzes 0.2m to 0.6m captures for optimal image dataset for geometric analysis
├── camera_characterization_pipeline.py      # Interactive end-to-end characterization notebook/script
├── src/
│   └── camera_toolkit/
│       ├── main.py                      # CLI entrypoint executing all stages & exporting deliverables
│       ├── utils.py                     # Environment validation, EXIF/DNG I/O, & results.json exporter
│       ├── geometry.py                  # Zhang calibration, distortion profiling, & pixel pitch metrics
│       ├── resolution.py                # QC audits, ESF/LSF/MTF50 solver, & diffraction limit modeling
│       ├── noise.py                     # Pair-difference PTC, read noise, dark current, & vendor profile audit
│       ├── dynamics.py                  # Full-well capacity & engineering/usable dynamic range
│       ├── color.py                     # Quarter-res linear RGB, LSC, von Kries WB, CCM, & CIEDE2000
│       ├── simulator.py                 # Synthetic forward models for ground-truth verification
│       ├── plots.py                     # Diagnostic figures
│       └── report_generator.py          # Automated Markdown report generator
├── test/
│   ├── conftest.py                      # Shared test fixtures and path configuration
│   ├── test_core_io.py                  # EXIF parsing, version gating, and JSON schema tests
│   ├── test_geometry.py                 # Synthetic checkerboard & straightness verification tests
│   ├── test_resolution.py               # Analytic Gaussian MTF50 & QC gate tests
│   ├── test_noise.py                    # Synthetic PTC gain, read noise, & dark current recovery tests
│   ├── test_dynamics.py                 # Full-well capacity & dynamic range propagation tests
│   └── test_color.py                    # CIEDE2000 & noiseless CCM recovery tests
└── outputs/
    ├── results.json                     # Machine-readable calibrated parameters & uncertainties
    └── characterization_report.md       # Generated summary report
```

---

## 5. Running the Pipeline & Tests

### Prerequisites
* **Python** $\ge 3.12$
* **OpenCV** $\ge 5.0.0$
* Core scientific packages: `numpy`, `rawpy`, `colour-science`, `pyyaml`, `matplotlib`, `tifffile`

### Execute the Full Characterization Pipeline
Run from the repository root to process all stages, generate diagnostic plots, and populate `outputs/results.json`:
```bash
python src/camera_toolkit/main.py
```
*(Optional flags: `--config path/to/config.yaml` or `--no-plots` to export `results.json` and `characterization_report.md` without rendering PNG figures.)*

### Run the Unit Test Suite (33 Tests)
Verify all mathematical solvers, synthetic ground-truth benchmarks, and runtime environment checks:
```bash
python -m unittest discover -s test -p "test_*.py" -v
```
