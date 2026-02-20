from sklearn.base import BaseEstimator, TransformerMixin

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
