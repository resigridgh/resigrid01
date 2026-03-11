# binaryclassification_impl.py

import sys
import os
sys.path.append(os.path.abspath("../src/mypythonpackage"))

from deepl.two_layer_binary_classification import binary_classification
import matplotlib.pyplot as plt
from datetime import datetime

W1, W2, W3, W4, losses = binary_classification(d=10, n=1000, epochs=5000, lr=0.001)

# Plot loss
plt.figure()
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')

# Save figure
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"crossentropyloss_{timestamp}.pdf"

plt.savefig(filename, bbox_inches='tight')

print(f"Plot saved as {filename}")
