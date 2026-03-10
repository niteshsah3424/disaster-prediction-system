import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, confusion_matrix, roc_curve)
from sklearn.model_selection import cross_val_score

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------
df = pd.read_csv("Monthly District Avg RainFall 1901 - 2017.csv")

print("Dataset Shape:", df.shape)

# --------------------------------------------------
# 2. FEATURE ENGINEERING
# --------------------------------------------------

# Pre-monsoon rainfall intensity
df['PreMonsoon'] = df[['Jan','Feb','Mar','Apr','May']].sum(axis=1)

# Rainfall variability (instability indicator)
df['Rainfall_Variability'] = df[['Jan','Feb','Mar','Apr','May']].std(axis=1)

# Dry spell count (low rainfall months)
df['Dry_Spell'] = (df[['Jan','Feb','Mar','Apr','May']] < 10).sum(axis=1)

df = df.dropna()

# --------------------------------------------------
# 3. CREATE FLOOD LABEL (Top 20%)
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
# 5. MODELS WITH CLASS BALANCING
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=300,
                                            class_weight='balanced',
                                            random_state=42),
    "SVM": SVC(probability=True, class_weight='balanced')
}

print("\n================ MODEL COMPARISON ================")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]
    
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))
    print(classification_report(y_test, preds))

# --------------------------------------------------
# 6. BEST MODEL (Random Forest) + Threshold Tuning
# --------------------------------------------------
best_model = RandomForestClassifier(n_estimators=300,
                                    class_weight='balanced',
                                    random_state=42)

best_model.fit(X_train, y_train)

probs = best_model.predict_proba(X_test)[:,1]

# Lower threshold to improve flood recall
y_pred = (probs > 0.40).astype(int)

print("\nAfter Threshold Tuning (0.40):")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# 7. CONFUSION MATRIX
# --------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --------------------------------------------------
# 8. ROC CURVE
# --------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, probs)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1])
plt.title("ROC Curve - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# --------------------------------------------------
# 9. FEATURE IMPORTANCE
# --------------------------------------------------
importances = best_model.feature_importances_

plt.figure()
plt.barh(features, importances)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.show()

# --------------------------------------------------
# 10. CROSS VALIDATION (Training Data)
# --------------------------------------------------
cv_scores = cross_val_score(best_model, X_train, y_train,
                            cv=5, scoring='roc_auc')

print("\nCross-Validation ROC-AUC Scores:", cv_scores)
print("Average CV ROC-AUC:", np.mean(cv_scores))

import pickle

pickle.dump(best_model, open("flood_model.pkl", "wb"))