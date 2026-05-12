import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# Phase 1: Data Loading & Preparation
# ==========================================
file_path = r'C:\Users\tim83\.cache\kagglehub\datasets\saurabhshahane\oil-price\versions\1\EE_Codes\SP500_OilDate for Prof. Narayan.xls'
df = pd.read_excel(file_path)

# Convert the 'obs' column (e.g., '2003M01') into a usable Date format
df['Date'] = pd.to_datetime(df['obs'], format='%YM%m')
df = df.sort_values('Date')

# Grab the last 10 years of data (10 years * 12 months = 120 rows)
df_recent = df.tail(120).copy()

# Calculate Monthly Returns (percentage change) for stationarity
df_recent['WTI_Return'] = df_recent['Oil'].pct_change()
df_recent['SP500_Return'] = df_recent['SP500'].pct_change()

# Drop the first row since pct_change() leaves a NaN value
df_recent = df_recent.dropna(subset=['WTI_Return', 'SP500_Return'])

# ==========================================
# Phase 2: Z-Scores & Outlier Removal
# ==========================================
df_recent['WTI_Zscore'] = stats.zscore(df_recent['WTI_Return'])
df_recent['SP500_Zscore'] = stats.zscore(df_recent['SP500_Return'])

# Define threshold (3 standard deviations) and filter outliers
z_threshold = 3
df_clean = df_recent[(np.abs(df_recent['WTI_Zscore']) < z_threshold) & 
                     (np.abs(df_recent['SP500_Zscore']) < z_threshold)].copy()

print(f"--- DATA CLEANING ---")
print(f"Original 10-year data points: {len(df_recent)}")
print(f"Data points after removing 'black swan' outliers: {len(df_clean)}\n")

df_clean.to_csv('cleaned_oil_sp500_data.csv', index=False)

# ==========================================
# Phase 3: Exploratory Data Analysis (EDA)
# ==========================================
# 1. Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='WTI_Return', y='SP500_Return', data=df_clean, alpha=0.7, color='blue')
plt.title('Monthly Returns (2004-2013): WTI Crude Oil vs S&P 500')
plt.xlabel('WTI Crude Monthly Return')
plt.ylabel('S&P 500 Monthly Return')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)

# Calculate and plot the line of best fit
m, b = np.polyfit(df_clean['WTI_Return'], df_clean['SP500_Return'], 1)
plt.plot(df_clean['WTI_Return'], m * df_clean['WTI_Return'] + b, color='red', linestyle='-', linewidth=2)
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(6, 4))
corr_matrix = df_clean[['WTI_Return', 'SP500_Return']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Pearson Correlation Heatmap')
plt.show()


# 3. Time Series Plot of Monthly Returns
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_xlabel('Date')
ax1.set_ylabel('WTI Monthly Return', color='tab:orange')
ax1.plot(df_clean['Date'], df_clean['WTI_Return'], color='tab:orange', 
         linewidth=1.2, alpha=0.8, label='WTI Crude Oil')
ax1.tick_params(axis='y', labelcolor='tab:orange')
ax1.axhline(0, color='black', linestyle='--', linewidth=0.6)

ax2 = ax1.twinx()
ax2.set_ylabel('S&P 500 Monthly Return', color='steelblue')
ax2.plot(df_clean['Date'], df_clean['SP500_Return'], color='steelblue', 
         linewidth=1.2, alpha=0.8, label='S&P 500')
ax2.tick_params(axis='y', labelcolor='steelblue')

fig.suptitle('WTI Crude Oil vs S&P 500: Monthly Returns (2004–2013)', fontsize=14)
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.88))
plt.tight_layout()
plt.show()

# ==========================================
# Phase 4: Statistical Modeling
# ==========================================
print("--- LINEAR REGRESSION (Predicting S&P 500 Returns) ---")
X_lin = df_clean['WTI_Return']
Y_lin = df_clean['SP500_Return']
X_lin_sm = sm.add_constant(X_lin) # Add y-intercept

ols_model = sm.OLS(Y_lin, X_lin_sm).fit()
print(ols_model.summary())
print("\n" + "="*50 + "\n")

print("--- LOGISTIC REGRESSION (Predicting Market Direction) ---")
# Create binary outcome: 1 if market went up/flat, 0 if it went down
df_clean['SP500_Up'] = (df_clean['SP500_Return'] >= 0).astype(int)

X_log = df_clean[['WTI_Return']] 
Y_log = df_clean['SP500_Up']

log_model = LogisticRegression()
log_model.fit(X_log, Y_log)

Y_pred = log_model.predict(X_log)
print("Confusion Matrix:")
print(confusion_matrix(Y_log, Y_pred))
print("\nClassification Report:")
print(classification_report(Y_log, Y_pred))

# ==========================================
# Phase 5: Additional Data Visualizations
# ==========================================

# --- FIGURE 4: Visual Confusion Matrix ---
# Generate the matrix values
cm = confusion_matrix(Y_log, Y_pred)

plt.figure(figsize=(7, 5))
# Use seaborn heatmap to make it visual. 'fmt="d"' ensures whole numbers.
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False,
            xticklabels=['Predicted Down', 'Predicted Up'],
            yticklabels=['Actual Down', 'Actual Up'],
            annot_kws={"size": 16}) # Make the numbers nice and big

plt.title('Logistic Regression: Market Direction Confusion Matrix', fontsize=14, pad=15)
plt.ylabel('True Market Movement', fontsize=12)
plt.xlabel('Predicted Market Movement', fontsize=12)

# This text box helps explain the visual to the reader
plt.figtext(0.5, -0.05, "Note: Model failed to predict any market downturns (0 True Negatives).", 
            wrap=True, horizontalalignment='center', fontsize=10, color='darkred')
plt.show()