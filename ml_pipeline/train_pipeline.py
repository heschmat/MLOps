import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
import joblib

from .transformers import MaxHRImputer, SexBinaryEncoder

CFG = {
    "target_col":  "heart_disease",
    "data_path": './ml_pipeline/data/heart_small.csv',
    "model_path": "./app/model/rf_pipeline.joblib",
    "rnd_seed": 19
}

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(CFG['data_path'])

X = df.drop(columns=CFG['target_col'])
y = df[CFG['target_col']]

# -----------------------------
# Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=CFG['rnd_seed'], stratify=y
)

# -----------------------------
# Pipeline
# -----------------------------
pipeline = Pipeline(steps=[
    ("sex_encoder", SexBinaryEncoder()),
    ("max_hr_imputer", MaxHRImputer()),
    ("model", RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        class_weight={0: 1, 1: 3},
        random_state=CFG['rnd_seed'],
        n_jobs=-1
    ))
])

# -----------------------------
# Train
# -----------------------------
pipeline.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
probs = pipeline.predict_proba(X_test)[:, 1]
# preds = pipeline.predict(X_test)
preds = (probs >= 0.4).astype(int)

score_recall = recall_score(y_test, preds)
print(f"Recal score: {score_recall:.4f}")

# -----------------------------
# Save pipeline
# -----------------------------
joblib.dump(pipeline, CFG['model_path'])
print(f"Pipeline saved at: {CFG['model_path']}")


if __name__ == "__main__":
    pass
