#############################################
### IC # 5 | Blood Pressures OLS Regression #
#-------------------------------------------#
# Sabrina Akter | Sabrina127@usf.edu        #
# 10/7/26                                   #
#############################################
# Creates scatterplot and performs          #
# Linear regression for a given dataset     #
# Saving outputs as png files and           #
# returning regression stats.               #
# ------------------------------            #
# (C) S. Akter, 2026 | For research         #
# and educational use only, not for         #
# Clinical decision-making                  #
#############################################

# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Environment variables
CSV_PATH = "Blood_Pressure_clean.csv"
X_COL = "sys_mmhg"
Y_COL = "dias_mmhg"

# Load CSV and perform sanity check
df = pd.read_csv(CSV_PATH)
data = df[[X_COL, Y_COL]].dropna()

# Create scatterplot
fig, ax = plt.subplots(figsize=(8, 10))
ax.scatter(data[X_COL], data[Y_COL], alpha=0.6, edgecolor="k", linewidth=0.3)
ax.set_xlabel(X_COL)
ax.set_ylabel(Y_COL)
ax.set_title(f"{Y_COL} vs. {X_COL} (n = {len(data)})")
ax.grid(alpha=0.3)
fig.tight_layout()

# Save scatterplot
scatter_file = f"{X_COL}_vs_{Y_COL}_scatter.png"
fig.savefig(scatter_file, dpi=300)
print(f"Saved scatterplot -> {scatter_file}")

# Perform linear regression
X = sm.add_constant(data[X_COL])
model = sm.OLS(data[Y_COL], X).fit()

# Print regression summary
print("OLS Regression Summary:")
print(model.summary())

# Plot regression line on existing axis
x_line = np.linspace(data[X_COL].min(), data[X_COL].max(), num=100)
y_line = model.params["const"] + model.params[X_COL] * x_line
ax.plot(x_line, y_line, color="crimson", linewidth=2, label=f"OLS Line (R^2 = {model.rsquared:.3f})")
ax.set_title(f"{Y_COL} vs. {X_COL} with OLS Line (n = {len(data)})")
ax.legend()

#save new scatterplot
ols_file = f"{X_COL}_vs_{Y_COL}_ols.png"
fig.savefig(ols_file, dpi=300)
print(f"Saved scatterplot with OLS Line -> {ols_file}")

# Save final plot with regression line
regression_file = f"{X_COL}_vs_{Y_COL}_regression.png"
fig.savefig(regression_file, dpi=300)
print(f"Saved plot with regression line -> {regression_file}")