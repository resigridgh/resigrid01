from mypythonpackage import deepl
import matplotlib.pyplot as plt
losses, W1, W2, W3, W4 = deepl.binary_classification(200, 40000, epochs = 1000)

from datetime import datetime

plt.plot(losses.cpu().detach().numpy())

plt.savefig("lossfunction_" + str(datetime.now()).replace(" ", "_").replace(":", "-").replace(".","__") + ".pdf")

print("Training complete.")
