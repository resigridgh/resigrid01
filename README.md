# Homework 01: Gradient Descent and Optimization


This repository implements a multi-layer binary classification model trained using gradient descent and PyTorch automatic differentiation, developed for **CPE 487/587 – Homework 1 (Gradient Descent and Optimization)**.

---

# Requirements

- Python 3.12  
- PyTorch  
- Matplotlib  

# Install dependencies:

uv add torch matplotlib torchdiffeq


# Project Structure for Homework 01

- resigrid01/
  - src/
    - mypythonpackage/
      - `__init__.py`
      - deepl/
        - `__init__.py`
        - `two_layer_binary_classification.py`
  - scripts/
      - `binaryclassification_impl.py`
- `README.md`


GPU is used automatically if available.


This script:

- Generates a random dataset
- Trains a 4-layer neural network using gradient descent
- Computes binary cross-entropy loss
- Plots loss versus epochs
- Saves the plot as a PDF file

# Output
A file is generated with the format:

crossentropyloss_YYYYMMDDhhmmss.pdf

This file shows training loss versus epochs.


## HW02Q7


---

### 1. Clone the Repository

```bash
git clone https://github.com/resigridgh/resigrid01.git
cd resigrid01
```

---

### 2. Create and Activate a Virtual Environment

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Linux / macOS**
```bash
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
.venv\Scripts\activate
```

---

### 3. Install the Package (Required for src-layout)

This project uses a `src/` package structure.  
You must install the package in editable mode so Python can find `mypythonpackage`.

```bash
pip install -e .
```


---

### 4. Install Dependencies

Install required libraries:

```bash
pip install torch matplotlib manim numpy
```

Verify Manim installation:

```bash
manim --version
```

---

### 5. Execute the Assignment (Background Execution)

The training takes a long time, so a background script is provided.

Make the script executable:

```bash
chmod +x hw02q7_background.sh
```

Run the assignment:

```bash
./hw02q7_background.sh
```




---
# HW02Q8 
---

## Step 1 

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
```


---

## Step 2 — Install Dependencies

```bash
uv add torch torchvision torchaudio
uv add numpy pandas matplotlib scikit-learn onnx
```

Install the local package:

```bash
uv pip install -e .
```

---

## Step 3 — Download Dataset

```bash
chmod +x Scripts/malwaredatadownload.sh
./Scripts/malwaredatadownload.sh
```

This will create:

```
data/Android_Malware.csv
```

---

## Step 4 — Run Experiments and Create Boxplots

```bash
chmod +x Scripts/multiclass_impl.sh
./Scripts/multiclass_impl.sh
```


---

## Step 5 — Final Output

After completion, the HW02Q8 results are in:

```
outputs/plots/
```

You should see:

```
boxplot_accuracy_*.png
boxplot_f1_*.png
boxplot_precision_*.png
boxplot_recall_*.png
```





















