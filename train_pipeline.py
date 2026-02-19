import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
import joblib

CFG = {
    "target_col":  "heart_disease",
    "data_path": './heart_small.csv',
    "rnd_seed": 19
}

# -----------------------------
# Custom transformer
# -----------------------------
class MaxHRImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        mask = X["max_hr"].isna()
        X.loc[mask, "max_hr"] = 220 - X.loc[mask, "age"]
        return X


# -----------------------------
# Binary encoder for sex
# -----------------------------
class SexBinaryEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["sex"] = X["sex"].map({"male": 1, "female": 0})
        return X


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
joblib.dump(pipeline, "rf_pipeline.joblib")
print("Pipeline saved as rf_pipeline.joblib")
