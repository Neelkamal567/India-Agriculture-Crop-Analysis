import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score



# ------------------Load dataset------------------

df = pd.read_csv("C:/Users/Neelkamal/Downloads/archive (3)/India Agriculture Crop Production.csv")
df


# Basic info
print("Shape:", df.shape)
print("First 5 rows: ",df.head())
print("\nColumn Names:", df.columns.tolist())
print(df.info())
print("Data Types:", df.dtypes)
print(df.describe())

# -------- PREPROCESSING ---------------

# Missing values
print(df.isnull().sum())

# Drop missing values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

#  Drop rows where Production or Area is null
df.dropna(subset=['Production', 'Area'], inplace=True)

print(df.isnull().sum())

# Reset index
df = df.reset_index(drop=True)

# Fix data types
df['Year'] = df['Year'].str.split('-').str[0].astype(int)

#check
print(df['Year'].head())
print(df['Year'].dtype)


#Outlier treatment using IQR on Production
Q1 = df['Production'].quantile(0.25)
Q3 = df['Production'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Production'] >= Q1 - 1.5 * IQR) & (df['Production'] <= Q3 + 1.5 * IQR)]

print("\nShape after preprocessing:", df.shape)
print("\nCleaned Data Sample:\n", df.head())


# OBJECTIVE 1 — STATE-WISE TOTAL PRODUCTION
state_prod = df.groupby('State')['Production'].sum().sort_values(ascending=False).head(10)
print(state_prod)

plt.figure(figsize=(6,4))
sns.barplot(x=state_prod.values, y=state_prod.index, palette='viridis')
plt.title('Top 10 States by Total Crop Production', fontsize=14, fontweight='bold')
plt.xlabel('Total Production (Tonnes)')
plt.ylabel('State')
plt.tight_layout()
plt.show()


# OBJECTIVE 2 — SEASON-WISE PRODUCTION COMPARISON

season_prod = df.groupby('Season')['Production'].sum().sort_values(ascending=False)
print(season_prod)


plt.figure(figsize=(6,4))
sns.barplot(x=season_prod.index, y=season_prod.values, palette='Set2' )
plt.title('Season-wise Total Production', fontweight='bold')
plt.xlabel('Total Production (Tonnes)')
plt.ylabel('State')
plt.xticks(rotation=30)
plt.show()


#OBJECTIVE 3 — TOP CROPS BY PRODUCTION VOLUME

crop_prod = df.groupby('Crop')['Production'].sum().sort_values(ascending=False).head(15)
print(crop_prod)

plt.figure(figsize=(8,4))
sns.barplot(x=crop_prod.values, y=crop_prod.index, palette='rocket')
plt.title('Top 15 Crops by Total Production Volume', fontsize=14, fontweight='bold')
plt.xlabel('Total Production (Tonnes)')
plt.ylabel('Crop')
plt.tight_layout()
plt.show()


# OBJECTIVE 4 — YEAR-WISE PRODUCTION TREND


df['Year'] = df['Year'].astype(str).str[:4].astype(int)
yearly_prod = df.groupby('Year')['Production'].sum()
print(yearly_prod)

plt.figure(figsize=(6, 4))
plt.plot(yearly_prod.index, yearly_prod.values, marker='o', linewidth=2,
         color='steelblue', markerfacecolor='orange', markersize=6)
plt.fill_between(yearly_prod.index, yearly_prod.values, alpha=0.15, color='steelblue')
plt.title('Year-wise Total Crop Production Trend', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Total Production (Tonnes)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# OBJECTIVE 5 — CORRELATION: AREA vs PRODUCTION + HEATMAP

corr = df[['Area', 'Production', 'Yield', 'Year']].corr()
print("Correlation Matrix:\n", corr)


plt.figure(figsize=(8, 5))
plt.scatter(df['Area'], df['Production'],alpha=0.1, color='teal', s=5)
plt.xscale('log')
plt.grid(True)
plt.title('Area vs Production',fontweight='bold')
plt.xlabel('Area (Hectare)')
plt.ylabel('Production (Tonnes)')
plt.show()

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap='coolwarm',linewidths=0.5,)
plt.title('Correlation Heatmap', fontweight='bold')
plt.show()

# OBJECTIVE 6 — OUTLIER DETECTION (BOXPLOTS)

plt.figure(figsize=(6,5))
sns.boxplot(y=df['Production'],color='salmon')
plt.title('Production Outliers',fontweight='bold')
plt.ylabel('Production(Tonnes)')
plt.show()
 
plt.figure(figsize=(6,5))
sns.boxplot(y=df['Area'],color='skyblue')
plt.title('Area Outliers',fontweight='bold')
plt.ylabel('Area(Hectare)')
plt.show()

#IQR
for col in ['Production', 'Area']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
print(outliers)

# OBJECTIVE 7 — DISTRICT-LEVEL ANALYSIS (Top State)

top_state = df.groupby('State')['Production'].sum().idxmax()
print(f"Top Producing State: {top_state}")

state_df = df[df['State'] == top_state]
district_prod = state_df.groupby('District')['Production'].sum().sort_values(ascending=False).head(10)
print(district_prod)

plt.figure(figsize=(6, 4))
sns.barplot(x=district_prod.values, y=district_prod.index, palette='mako')
plt.title(f'Top 10 Districts in {top_state} by Production', fontsize=13, fontweight='bold')
plt.xlabel('Total Production (Tonnes)')
plt.ylabel('District')
plt.tight_layout()
plt.show()


# ============================================================
# OBJECTIVE 8 — MACHINE LEARNING: LINEAR REGRESSION
# ============================================================


# Prepare data
ml_df = df[['State', 'Crop', 'Season', 'Area', 'Year', 'Production']].copy()
ml_df.dropna(inplace=True)

# Label Encoding
le = LabelEncoder()
for col in ['State', 'Crop', 'Season']:
    ml_df[col] = le.fit_transform(ml_df[col])

# Features and Target
X = ml_df[['Area', 'Year', 'State', 'Crop', 'Season']]
y = ml_df['Production']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions00.
y_pred = model.predict(X_test)

# Evaluation
r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nModel Evaluation:")
print(f"  R² Score : {r2:.4f}")
print(f"  MAE      : {mae:.2f}")
print(f"  RMSE     : {rmse:.2f}")

# Coefficients
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print("\nFeature Coefficients:\n", coef_df)

# Actual vs Predicted Plot
plt.figure(figsize=(10, 5))
plt.scatter(y_test[:300], y_pred[:300], alpha=0.5, color='steelblue', s=15)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.title('Actual vs Predicted Production (Linear Regression)', fontweight='bold')
plt.xlabel('Actual Production')
plt.ylabel('Predicted Production')
plt.legend()
plt.tight_layout()
plt.show()

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 4))
plt.scatter(y_pred[:300], residuals[:300], alpha=0.4, color='coral', s=15)
plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
plt.title('Residual Plot', fontweight='bold')
plt.xlabel('Predicted Production')
plt.ylabel('Residuals')
plt.tight_layout()
plt.show()





