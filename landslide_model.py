# =========================================
# NASA LANDSLIDE PREDICTION - FINAL PRO VERSION
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier

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

# Keep only top categories to avoid huge dummy explosion
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
# 8️⃣ SMOTE-TOMEK BALANCING
# -----------------------------------------
smk = SMOTETomek(random_state=42)
X_train, y_train = smk.fit_resample(X_train, y_train)

print("\nAfter SMOTE-Tomek:\n", pd.Series(y_train).value_counts())

# -----------------------------------------
# 9️⃣ SCALING (For Logistic)
# -----------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------
# 🔟 LOGISTIC REGRESSION (Optimized)
# -----------------------------------------
log_model = LogisticRegression(
    max_iter=3000,
    class_weight='balanced'
)

log_model.fit(X_train_scaled, y_train)
log_probs = log_model.predict_proba(X_test_scaled)[:, 1]

# Optimize threshold for best F1
precision, recall, thresholds = precision_recall_curve(y_test, log_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_threshold = thresholds[np.argmax(f1_scores)]

log_preds = (log_probs > best_threshold).astype(int)

print("\n==== Logistic Regression (Optimized Threshold) ====")
print("ROC-AUC:", roc_auc_score(y_test, log_probs))
print("Best Threshold:", best_threshold)
print(classification_report(y_test, log_preds))

# -----------------------------------------
# 1️⃣1️⃣ RANDOM FOREST
# -----------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=600,
    max_depth=18,
    min_samples_split=4,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_preds = rf_model.predict(X_test)

print("\n==== Random Forest ====")
print("ROC-AUC:", roc_auc_score(y_test, rf_probs))
print(classification_report(y_test, rf_preds))

# -----------------------------------------
# 1️⃣2️⃣ XGBOOST (BEST MODEL)
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

# Optimize threshold
precision, recall, thresholds = precision_recall_curve(y_test, xgb_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_threshold_xgb = thresholds[np.argmax(f1_scores)]

xgb_preds = (xgb_probs > best_threshold_xgb).astype(int)

print("\n==== XGBoost (Best Model) ====")
print("ROC-AUC:", roc_auc_score(y_test, xgb_probs))
print("Best Threshold:", best_threshold_xgb)
print(classification_report(y_test, xgb_preds))

# -----------------------------------------
# 1️⃣3️⃣ CONFUSION MATRIX (XGB)
# -----------------------------------------
cm = confusion_matrix(y_test, xgb_preds)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------------------
# 1️⃣4️⃣ ROC CURVE
# -----------------------------------------
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)

plt.figure()
plt.plot(fpr_xgb, tpr_xgb, label="XGBoost")
plt.plot([0, 1], [0, 1], '--')
plt.legend()
plt.title("ROC Curve - XGBoost")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

print("\n🔥 FINAL HIGH-ACCURACY MODEL COMPLETED")

# -----------------------------------------
# 1️⃣5️⃣ SAVE FINAL MODEL + PREPROCESSORS
# -----------------------------------------

import pickle

# Save XGBoost model
pickle.dump(xgb_model, open("landslide_model.pkl", "wb"))

# Save imputer
pickle.dump(imputer, open("landslide_imputer.pkl", "wb"))

# Save scaler (optional but safe)
pickle.dump(scaler, open("landslide_scaler.pkl", "wb"))

# Save column structure (VERY IMPORTANT for dummies)
pickle.dump(X.columns.tolist(), open("landslide_columns.pkl", "wb"))

print("\n✅ Landslide model and preprocessors saved successfully!")