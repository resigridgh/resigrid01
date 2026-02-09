# HW02 Q7: Weight Matrix Visualization
import os
from mypythonpackage.deepl.two_layer_binary_classification import binary_classification
from mypythonpackage.animation import animate_weight_heatmap
import matplotlib.pyplot as plt


# 1) Create media folder 

os.makedirs("media", exist_ok=True)


# 2) Train model and receive weight histories

W1_hist, W2_hist, W3_hist, W4_hist, losses = binary_classification(
    d=200,
    n=40000,
    epochs=5000,
    lr=0.01
)

print("Training finished. Creating animations...")


# 3) Create 4 animations 


# W1
animate_weight_heatmap(
    W1_hist,
    save_path="media/W1.mp4",
    title="W1 Weight Evolution"
)

# W2
animate_weight_heatmap(
    W2_hist,
    save_path="media/W2.mp4",
    title="W2 Weight Evolution"
)

# W3
animate_weight_heatmap(
    W3_hist,
    save_path="media/W3.mp4",
    title="W3 Weight Evolution"
)

# W4 (transpose) 
animate_weight_heatmap(
    W4_hist.transpose(1,2),
    save_path="media/W4.mp4",
    title="W4 Weight Evolution"
)
