import pandas as pd
import numpy as np
import joblib   # ✅ pickle ki jagah joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------
df = pd.read_csv("Monthly District Avg RainFall 1901 - 2017.csv")

print("Dataset Shape:", df.shape)

# --------------------------------------------------
# 2. FEATURE ENGINEERING
# --------------------------------------------------

df['PreMonsoon'] = df[['Jan','Feb','Mar','Apr','May']].sum(axis=1)
df['Rainfall_Variability'] = df[['Jan','Feb','Mar','Apr','May']].std(axis=1)
df['Dry_Spell'] = (df[['Jan','Feb','Mar','Apr','May']] < 10).sum(axis=1)

df = df.dropna()

# --------------------------------------------------
# 3. CREATE FLOOD LABEL
# --------------------------------------------------
threshold = df['Monsoon'].quantile(0.80)
df['FLOOD'] = (df['Monsoon'] > threshold).astype(int)

print("\nFlood Distribution:\n", df['FLOOD'].value_counts())

# --------------------------------------------------
# 4. TIME-BASED SPLIT
# --------------------------------------------------
train = df[df['Year'] <= 2000]
test = df[df['Year'] > 2000]

features = ['Jan','Feb','Mar','Apr','May',
            'PreMonsoon','Rainfall_Variability','Dry_Spell']

X_train = train[features]
y_train = train['FLOOD']

X_test = test[features]
y_test = test['FLOOD']

# --------------------------------------------------
# 5. TRAIN OPTIMIZED MODEL (SIZE REDUCED)
# --------------------------------------------------
best_model = RandomForestClassifier(
    n_estimators=80,      
    max_depth=10,         
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

best_model.fit(X_train, y_train)

probs = best_model.predict_proba(X_test)[:,1]
y_pred = (probs > 0.40).astype(int)

print("\nModel Evaluation (Threshold = 0.40)")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# 6. SAVE MODEL (COMPRESSED)
# --------------------------------------------------
joblib.dump(best_model, "flood_model.pkl", compress=3)

print("\nModel saved successfully as flood_model.pkl")