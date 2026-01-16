# Homework 01: Gradient Descent and Optimization

This repository implements a multi-layer binary classification model trained using gradient descent and PyTorch automatic differentiation, developed for **CPE 487/587 – Homework 1 (Gradient Descent and Optimization)**.

---

## Requirements

- Python 3.12  
- PyTorch  
- Matplotlib  

# Install dependencies:

!pip install torch matplotlib

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

Installation

- Clone the repository through the following instructions:

git clone https://github.com/resigridgh/resigrid01

cd resigrid01

- Running the Code
 
Copy the code contains in binaryclassification_impl.py from the scripts directory in your notebook. 

Here is an example: https://colab.research.google.com/drive/1OMqp9houE9cf1uHFv__4Gry25-8LCLBb?usp=sharing

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













