from model.regression import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
url = "https://raw.githubusercontent.com/rahulbhadani/CPE486586_FA25/main/Data/Hydropower.csv"
df = pd.read_csv(url)

# Select features
X = df['Benefit-Cost-Ratio'].values
y = df['AnnualProduction'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train model
model = LinearRegression(lr=0.001, epochs=3000)
r2 = model.fit(X_train, y_train, X_test, y_test, verbose=True)

# Summary and plot
model.summary()
model.plot_results(X, y, save_path="LinearRegression_Result.pdf", dpi=300)

