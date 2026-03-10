# =========================================
# NASA LANDSLIDE PREDICTION - TRAIN + SAVE
# =========================================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve

# -----------------------------------------
# 1️⃣ LOAD DATA
# -----------------------------------------
df = pd.read_csv("global_landslide_catalog.csv")
df.columns = df.columns.str.lower()

print("Original Shape:", df.shape)

# -----------------------------------------
# 2️⃣ FILTER RAIN EVENTS
# -----------------------------------------
df = df[df['landslide_trigger'].str.lower().isin(
    ['rain', 'downpour', 'continuous_rain', 'monsoon']
)]

print("After Rain Filter:", df.shape)

# -----------------------------------------
# 3️⃣ DATE FEATURE ENGINEERING
# -----------------------------------------
df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce', format='mixed')
df = df.dropna(subset=['event_date'])

df['year'] = df['event_date'].dt.year
df['month'] = df['event_date'].dt.month
df['day'] = df['event_date'].dt.day

# -----------------------------------------
# 4️⃣ TARGET CREATION
# -----------------------------------------
df = df.dropna(subset=['landslide_size'])

df['target'] = df['landslide_size'].str.lower().apply(
    lambda x: 1 if x in ['large','very_large','catastrophic'] else 0
)

print("Target Distribution:\n", df['target'].value_counts())

# -----------------------------------------
# 5️⃣ FEATURE ENGINEERING
# -----------------------------------------
for col in ['country_name', 'landslide_category', 'landslide_setting']:
    top = df[col].value_counts().nlargest(10).index
    df[col] = df[col].where(df[col].isin(top), 'other')

selected_cols = [
    'latitude',
    'longitude',
    'year',
    'month',
    'day',
    'country_name',
    'landslide_category',
    'landslide_setting'
]

df = df[selected_cols + ['target']]
df = df.fillna('unknown')
df = pd.get_dummies(df, drop_first=True)

X = df.drop('target', axis=1)
y = df['target']

# -----------------------------------------
# 6️⃣ IMPUTER
# -----------------------------------------
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# -----------------------------------------
# 7️⃣ TRAIN TEST SPLIT
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------------------
# 8️⃣ SMOTE-TOMEK
# -----------------------------------------
smk = SMOTETomek(random_state=42)
X_train, y_train = smk.fit_resample(X_train, y_train)

# -----------------------------------------
# 9️⃣ SCALER
# -----------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# -----------------------------------------
# 🔟 XGBOOST (BEST MODEL)
# -----------------------------------------
xgb_model = XGBClassifier(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=3,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, xgb_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_threshold_xgb = thresholds[np.argmax(f1_scores)]

print("\nBest Threshold:", best_threshold_xgb)
print("ROC-AUC:", roc_auc_score(y_test, xgb_probs))

# -----------------------------------------
# 💾 SAVE EVERYTHING
# -----------------------------------------

pickle.dump(xgb_model, open("landslide_model.pkl", "wb"))
pickle.dump(imputer, open("landslide_imputer.pkl", "wb"))
pickle.dump(scaler, open("landslide_scaler.pkl", "wb"))
pickle.dump(df.drop('target', axis=1).columns.tolist(), open("landslide_columns.pkl", "wb"))
pickle.dump(best_threshold_xgb, open("landslide_threshold.pkl", "wb"))

print("\n✅ Landslide Model + Preprocessors Saved Successfully!")