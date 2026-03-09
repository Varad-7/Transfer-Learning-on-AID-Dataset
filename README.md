# Transfer Learning on AID Dataset — GNR 638 Assignment 2

> **Course:** GNR 638 — Machine Learning for Remote Sensing II  
> **Institute:** IIT Bombay | **Semester:** Spring 2025–26  
> **Team:** Shikhar Verma (22B2201) · Varad Vikas Patil (22B2270) · Aarushi Agarwal (21D070003)

---

## Overview

This project evaluates three pre-trained CNN architectures — **ResNet50**, **EfficientNet-B0**, and **ConvNeXt-Tiny** — on the [AID (Aerial Image Dataset)](https://captain-whu.github.io/DiRS/) with 30 land-cover classes across five transfer-learning scenarios:

| #   | Scenario                                                        | Script                       |
| --- | --------------------------------------------------------------- | ---------------------------- |
| 1   | Linear Probe Transfer (frozen backbone)                         | `scenario_1_linear_probe.py` |
| 2   | Fine-Tuning Strategies (4 unfreezing strategies)                | `scenario_2_finetuning.py`   |
| 3   | Few-Shot Learning (100 / 20 / 5 % data)                         | `scenario_3_fewshot.py`      |
| 4   | Corruption Robustness (Gaussian noise, motion blur, brightness) | `scenario_4_corruption.py`   |
| 5   | Layer-Wise Feature Probing + PCA                                | `scenario_5_layerwise.py`    |

---

## Repository Structure

```
.
├── scenario_1_linear_probe.py   # Scenario 1 entry point
├── scenario_2_finetuning.py     # Scenario 2 entry point
├── scenario_3_fewshot.py        # Scenario 3 entry point
├── scenario_4_corruption.py     # Scenario 4 entry point
├── scenario_5_layerwise.py      # Scenario 5 entry point
├── regen_s2_plot.py             # Regenerate Scenario 2 plots from saved JSON
├── regen_s3_plot.py             # Regenerate Scenario 3 plots from saved JSON
├── regen_s4_plot.py             # Regenerate Scenario 4 plots from saved JSON
├── regen_s5_plot.py             # Regenerate Scenario 5 plots from saved JSON
├── run_all.sh                   # Run all 5 scenarios in sequence
├── setup_env.sh                 # Create venv and install dependencies
├── requirements.txt             # Python dependencies
├── src/
│   ├── dataset.py               # Data loading, splits, corruption transforms
│   ├── models.py                # Model factory, freeze/unfreeze helpers, feature extraction
│   ├── train_utils.py           # Training loop, evaluation, gradient norm utilities
│   └── visualize.py             # All plotting functions
├── train_data/                  # AID dataset (place here — see below)
│   ├── Airport/
│   ├── Beach/
│   └── ...  (30 class folders)
├── outputs/                     # Auto-generated results (JSON + plots)
│   ├── scenario_1/
│   ├── scenario_2/
│   ├── scenario_3/
│   ├── scenario_4/
│   └── scenario_5/
└── report/
    ├── main.tex                 # LaTeX report
    ├── generate_latex.py        # Auto-generate LaTeX from result JSONs
    └── images/                  # Plots used in the report (copied from outputs/)
```

---

## Prerequisites

- Python 3.9+
- macOS / Linux / **Windows 10 or 11**
- GPU: Apple Silicon MPS, CUDA GPU, or CPU (slow)
- `pdflatex` (optional — only needed to compile the report)
- **Windows only:** [Git for Windows](https://git-scm.com/download/win) and [Python from python.org](https://www.python.org/downloads/windows/)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Varad-7/Transfer-Learning-on-AID-Dataset.git
cd Transfer-Learning-on-AID-Dataset
```

### 2. Prepare the dataset

Download the AID dataset and place the 30 class folders directly inside `train_data/`:

```
train_data/
├── Airport/
├── BareLand/
├── BaseballField/
...
└── Viaduct/
```

Each class folder should contain `.jpg` images.

### 3. Create the virtual environment and install dependencies

**macOS / Linux:**
```bash
bash setup_env.sh
```

Or manually on macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (Command Prompt or PowerShell):**
```bat
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Windows + CUDA:** If you have an NVIDIA GPU, install the CUDA-enabled PyTorch build before the rest of the requirements:
> ```bat
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> pip install -r requirements.txt
> ```

---

## Running the Code

### Run All Scenarios at Once

**macOS / Linux:**
```bash
bash run_all.sh
```

**Windows** — `run_all.sh` is a Bash script and does not run natively. Use one of these options:

*Option A — Git Bash (recommended):*
```bash
bash run_all.sh
```

*Option B — Command Prompt / PowerShell (manual equivalent):*
```bat
venv\Scripts\activate
python scenario_1_linear_probe.py
python scenario_2_finetuning.py
python scenario_3_fewshot.py
python scenario_4_corruption.py
python scenario_5_layerwise.py
```

This runs all 5 scenarios sequentially with default settings.

---

### Run Individual Scenarios

Activate the environment first:

**macOS / Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bat
venv\Scripts\activate
```

#### Scenario 1 — Linear Probe Transfer

Trains only the classifier head on a frozen backbone.

```bash
python scenario_1_linear_probe.py
```

Outputs saved to `outputs/scenario_1/`.

---

#### Scenario 2 — Fine-Tuning Strategies

Compares four unfreezing strategies: `linear_probe`, `last_block`, `selective_20pct`, `full_ft`.

```bash
# All models, all strategies (default)
python scenario_2_finetuning.py

# Single model
python scenario_2_finetuning.py --model resnet50

# Custom epochs and batch size
python scenario_2_finetuning.py --model efficientnet_b0 --epochs 10 --batch_size 16
```

| Argument       | Default | Description                                              |
| -------------- | ------- | -------------------------------------------------------- |
| `--model`      | `all`   | `resnet50`, `efficientnet_b0`, `convnext_tiny`, or `all` |
| `--epochs`     | `15`    | Number of training epochs                                |
| `--batch_size` | `32`    | Batch size                                               |

Outputs saved to `outputs/scenario_2/`.

---

#### Scenario 3 — Few-Shot Learning

Trains on 100 %, 20 %, and 5 % of the training data to study data-scarcity resilience.

```bash
# All models (default)
python scenario_3_fewshot.py

# Single model
python scenario_3_fewshot.py --model resnet50

# Custom epoch counts
python scenario_3_fewshot.py --epochs_max 30 --epochs_few 20
```

| Argument       | Default | Description                       |
| -------------- | ------- | --------------------------------- |
| `--model`      | `all`   | Model selection                   |
| `--epochs_max` | `20`    | Epochs for 100 % data fraction    |
| `--epochs_few` | `10`    | Epochs for 20 % and 5 % fractions |
| `--batch_size` | `32`    | Batch size                        |

Outputs saved to `outputs/scenario_3/`.

---

#### Scenario 4 — Corruption Robustness

Trains on clean data, then evaluates under 5 corruption types.

```bash
# All models (default)
python scenario_4_corruption.py

# Single model, custom epochs
python scenario_4_corruption.py --model convnext_tiny --epochs 10 --batch_size 16
```

| Argument       | Default | Description                    |
| -------------- | ------- | ------------------------------ |
| `--model`      | `all`   | Model selection                |
| `--epochs`     | `15`    | Epochs for clean-data training |
| `--batch_size` | `32`    | Batch size                     |

Outputs saved to `outputs/scenario_4/`.

---

#### Scenario 5 — Layer-Wise Feature Probing

Extracts features from early, middle, and late layers; trains linear classifiers; generates PCA plots.

```bash
# All models (default)
python scenario_5_layerwise.py

# Single model
python scenario_5_layerwise.py --model efficientnet_b0 --batch_size 32
```

| Argument       | Default | Description     |
| -------------- | ------- | --------------- |
| `--model`      | `all`   | Model selection |
| `--batch_size` | `32`    | Batch size      |

Outputs saved to `outputs/scenario_5/`.

---

### Regenerate Plots from Saved Results

If you already have result JSONs and only want to regenerate the plots:

```bash
python regen_s2_plot.py
python regen_s3_plot.py
python regen_s4_plot.py
python regen_s5_plot.py
```

> Works identically on macOS, Linux, and Windows.

---

## Compiling the Report

**macOS / Linux:**
```bash
cd report
pdflatex main.tex
pdflatex main.tex   # Run twice to resolve cross-references
```

**Windows:**

Install a LaTeX distribution such as [MiKTeX](https://miktex.org/download) or [TeX Live](https://tug.org/texlive/), then:
```bat
cd report
pdflatex main.tex
pdflatex main.tex
```

Alternatively, open `report/main.tex` in [TeXstudio](https://www.texstudio.org/) or [Overleaf](https://www.overleaf.com) and compile from the GUI.

The compiled PDF will be `report/main.pdf`.

---

## Hardware Notes

| Platform | Device | Notes |
|----------|--------|-------|
| macOS (Apple Silicon) | M1 / M2 / M3 | MPS acceleration used automatically |
| Linux / Windows | NVIDIA GPU (CUDA) | CUDA used automatically if available |
| Any | CPU only | Works, but significantly slower |

- Models were developed and tested on an **Apple M2 (8 GB unified memory)** device.
- ConvNeXt-Tiny requires a reduced batch size (`--batch_size 16` or `8`) to avoid memory exhaustion on 8 GB devices.
- On CUDA GPUs with ≥ 8 GB VRAM, the default batch size of 32 works for all models.
- Training times per scenario per model are approximately 5–20 minutes depending on hardware.
- **Windows note:** `num_workers > 0` in DataLoader can cause issues on Windows due to multiprocessing differences. The scripts already use `num_workers=0` to avoid this.

---

## Results Summary

| Model           | Linear Probe | Best Fine-Tune       | 5 % Few-Shot |
| --------------- | ------------ | -------------------- | ------------ |
| ResNet50        | 76.98 %      | 96.78 % (full FT)    | 76.98 %      |
| EfficientNet-B0 | 79.27 %      | 96.57 % (full FT)    | 71.48 %      |
| ConvNeXt-Tiny   | 91.85 %      | 95.07 % (last block) | 4.22 %       |

---

## License

This project is submitted as academic coursework for GNR 638 at IIT Bombay.
