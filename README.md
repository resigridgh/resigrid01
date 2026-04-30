# HW 01: Gradient Descent and Optimization

---

## Step 1. Clone the Repository

```bash
git clone https://github.com/resigridgh/resigrid01.git
cd resigrid01
```

---

## Step 2. Create and Activate a Virtual Environment

# first to clone this repo and synchronize all the necessary package
```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
uv build
```
## Step 3. Run the bash script
```bash
cd Script
chmod +x binary_classh.sh
nohup ./binary_classh.sh > training_log.out 2>&1 &
```
### 


# Output
A file is generated with the format in the Scripts folder:

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

# first to clone this repo and synchronize all the necessary package
```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
uv build
```

### 3. Install the Package (Required for src-layout)

This project uses a `src/` package structure.  
You must install the package in editable mode so Python can find `mypythonpackage`.

```bash
uv pip install -e .
```


---

### 4. Install Dependencies

Install required libraries:

```bash
uv pip install torch matplotlib manim numpy
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
git clone https://github.com/resigridgh/resigrid01.git
cd resigrid01
uv venv --python 3.12
source .venv/bin/activate
uv sync
uv build
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


---
# HW03Q6 
---

## 🚀 How to Run

Follow these steps from the root of the repository:

## Step 1 - Clone the repo

```bash
git clone https://github.com/resigridgh/resigrid01.git
cd resigrid01
uv venv --python 3.12
source .venv/bin/activate
uv sync
uv build
```
## Step 2 — Go to repo root
```bash
cd ~/resigrid01
```
## Step 3 — Make the script executable

```bash
chmod +x Scripts/imagenet_impl.sh
```
## Step 4 — Run the script
```bash
Scripts/imagenet_impl.sh
```
### Should See:
```bash
example_train_image.png
example_val_image.png
```

---
# HW03Q7 
---

## Step 1 - Clone the repo

```bash
git clone https://github.com/resigridgh/resigrid01.git
cd resigrid01
uv venv --python 3.12
source .venv/bin/activate
uv sync
uv build
```
## Step 2 — Go to repo root
```bash
cd resigrid01
```

## Step 3
```bash
cd Scrips
```

## Step 4
```
python acc_classifier_impl.py
```
## You Shouls See:
```bash
acc_model.onnx
accuracy_vs_epochs.png
loss_vs_epochs.png
```
In the Scripts directory. 



---
# HW04 
---

## 🚀 How to Run

Follow these steps from the root of the repository:

## Step 1 - Clone the repo

```bash
git clone https://github.com/resigridgh/resigrid01.git
cd resigrid01
uv venv --python 3.12
source .venv/bin/activate
uv sync
uv build
```
## Step 2 — Go to `Scripts` folder
```bash
cd ~/resigrid01/Scripts
```
## Step 3 — Make the script executable

```bash
chmod +x run_genmodels.sh
```
## Step 4 — Run the script in the background
```bash
nohup ./run_genmodels.sh > run_${TS}.log 2>&1 &
```
### Should See following files in the `outputs/gen_eval_1000` folder:
```bash
comparison_barplot_2026-04-17_17-57-27.png
vae_samples_25_2026-04-17_17-57-27.png
gan_samples_25_2026-04-17_17-57-27.png
diffusion_samples_25_2026-04-17_17-57-27.png
```








