# EDA
# ============================================================
# EXPLORATORY DATA ANALYSIS (EDA) PROJECT
# Using Python, Pandas, NumPy, Matplotlib and Seaborn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. CREATE DATASET
# ============================================================

data = {
    "Name": [
        "Thunder", "Lightning", "Dragon", "Titan",
        "Viper", "Cyclone", "Storm", "Rocket",
        "Eagle", "Phoenix", "Twister", "Raptor"
    ],

    "Location": [
        "USA", "USA", "UK", "USA",
        "Canada", "UK", "USA", "Canada",
        "USA", "UK", "Canada", "USA"
    ],

    "Speed": [
        95, 110, 85, 120,
        90, 100, 105, 115,
        80, 88, 92, 125
    ],

    "Height": [
        120, 150, 100, 180,
        110, 130, 140, 160,
        90, 105, 115, 190
    ],

    "Length": [
        3000, 3500, 2800, 4000,
        2900, 3200, 3300, 3700,
        2500, 2700, 3000, 4200
    ],

    "Year_Built": [
        2015, 2018, 2012, 2020,
        2016, 2014, 2019, 2021,
        2010, 2013, 2017, 2022
    ]
}

df = pd.DataFrame(data)


# ============================================================
# 2. DATA UNDERSTANDING
# ============================================================

print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

print("\n========== LAST 5 ROWS ==========\n")
print(df.tail())

print("\n========== DATASET SHAPE ==========\n")
print(df.shape)

print("\n========== COLUMN NAMES ==========\n")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========\n")
print(df.dtypes)

print("\n========== STATISTICAL SUMMARY ==========\n")
print(df.describe())

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========\n")
print(df.duplicated().sum())


# ============================================================
# 3. DATA PREPARATION
# ============================================================

# Remove unnecessary columns if present
df = df.drop(columns=["Year_Built"])

# Rename columns for clarity
df = df.rename(columns={
    "Name": "Roller_Coaster",
    "Speed": "Speed_kmh",
    "Height": "Height_m",
    "Length": "Length_m"
})

# Remove duplicate rows
df = df.drop_duplicates()

# Handle missing numerical values
numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )

# Handle missing categorical values
if df["Location"].isnull().sum() > 0:
    df["Location"] = df["Location"].fillna(
        df["Location"].mode()[0]
    )


print("\n========== CLEANED DATASET ==========\n")
print(df)


# ============================================================
# 4. UNIVARIATE ANALYSIS
# ============================================================

# Speed distribution

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Speed_kmh"],
    bins=6,
    kde=True
)

plt.title("Distribution of Roller Coaster Speed")
plt.xlabel("Speed (km/h)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# Height distribution

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Height_m"],
    bins=6,
    kde=True
)

plt.title("Distribution of Roller Coaster Height")
plt.xlabel("Height (m)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ============================================================
# 5. FEATURE RELATIONSHIP - SPEED VS HEIGHT
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Height_m",
    y="Speed_kmh",
    hue="Location",
    s=100
)

plt.title("Relationship Between Height and Speed")
plt.xlabel("Height (m)")
plt.ylabel("Speed (km/h)")

plt.tight_layout()
plt.show()


# ============================================================
# 6. PAIR PLOT
# ============================================================

sns.pairplot(
    df,
    vars=[
        "Speed_kmh",
        "Height_m",
        "Length_m"
    ],
    hue="Location"
)

plt.show()


# ============================================================
# 7. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(8, 6))

correlation = df[
    [
        "Speed_kmh",
        "Height_m",
        "Length_m"
    ]
].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()


# ============================================================
# 8. TOP 5 FASTEST ROLLER COASTERS
# ============================================================

top_5 = df.sort_values(
    by="Speed_kmh",
    ascending=False
).head(5)

print("\n========== TOP 5 FASTEST ROLLER COASTERS ==========\n")

print(
    top_5[
        [
            "Roller_Coaster",
            "Location",
            "Speed_kmh"
        ]
    ]
)


# ============================================================
# 9. FASTEST ROLLER COASTER BY LOCATION
# ============================================================

fastest_by_location = (
    df.groupby("Location")["Speed_kmh"]
    .max()
    .sort_values(ascending=False)
)

print(
    "\n========== FASTEST ROLLER COASTER SPEED BY LOCATION ==========\n"
)

print(fastest_by_location)


# ============================================================
# 10. BAR CHART - FASTEST COASTER BY LOCATION
# ============================================================

plt.figure(figsize=(8, 5))

fastest_by_location.plot(
    kind="bar"
)

plt.title("Fastest Roller Coaster Speed by Location")
plt.xlabel("Location")
plt.ylabel("Maximum Speed (km/h)")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ============================================================
# 11. ANSWERING A DATA QUESTION
# ============================================================

print("\n========== DATA QUESTION ==========")

print(
    "Question: Which roller coaster is the fastest?"
)

fastest = df.loc[
    df["Speed_kmh"].idxmax()
]

print(
    "\nFastest Roller Coaster:",
    fastest["Roller_Coaster"]
)

print(
    "Location:",
    fastest["Location"]
)

print(
    "Speed:",
    fastest["Speed_kmh"],
    "km/h"
)


# ============================================================
# 12. FINAL EDA SUMMARY
# ============================================================

print("\n============================================")
print("       EDA PROJECT COMPLETED")
print("============================================")

print("\nNumber of rows:", len(df))

print("Number of columns:", len(df.columns))

print(
    "\nAverage Speed:",
    round(df["Speed_kmh"].mean(), 2),
    "km/h"
)

print(
    "Average Height:",
    round(df["Height_m"].mean(), 2),
    "m"
)

print(
    "Average Length:",
    round(df["Length_m"].mean(), 2),
    "m"
)
print("\nEDA completed successfully.")
