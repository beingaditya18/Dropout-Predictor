import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(__file__)
csv_path = os.path.join(BASE_DIR, "students_500_realistic.csv")
model_path = os.path.join(BASE_DIR, "models", "dropout_rf.pkl")
acc_path = os.path.join(BASE_DIR, "models", "accuracy.txt")

# 1. Load dataset
df = pd.read_csv(csv_path)

# 2. Features used
features = [
    "attendance",
    "result_pct",
    "assignments_submitted",
    "engagement_score",
    "family_income",
    "distance_km",
    "fee_delay",
    "class"
]

X = df[features].values
y = df["dropout"].values

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train RandomForest
clf = RandomForestClassifier(n_estimators=300, random_state=42)
clf.fit(X_train, y_train)

# 5. Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print(classification_report(y_test, y_pred))

# 6. Save model + metadata
meta = {
    "features": features,
    "feature_means": {f: float(df[f].mean()) for f in features},
    "feature_importances": {f: float(imp) for f, imp in zip(features, clf.feature_importances_)}
}

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
with open(model_path, "wb") as f:
    pickle.dump({"model": clf, "meta": meta}, f)

with open(acc_path, "w") as f:
    f.write(str(acc))

print(f"✅ Model retrained and saved to {model_path}")
print(f"✅ Accuracy saved to {acc_path}")
