import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "student-mat.csv", sep=";")
TARGET = "G3"

def run_experiment(drop_previous_grades=False):
    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET].copy()

    if drop_previous_grades:
        X = X.drop(columns=["G1", "G2"])

    num = X.select_dtypes(include=["number"]).columns.tolist()
    cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), num),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat)
    ])

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    Xtr = pre.fit_transform(Xtr)
    Xte = pre.transform(Xte)

    names = pre.get_feature_names_out()
    selector = SelectFromModel(
        RandomForestRegressor(
            n_estimators=400, random_state=42, n_jobs=-1,
            max_features="sqrt"
        ),
        threshold="median"
    )
    selector.fit(Xtr, ytr)
    Xtr = selector.transform(Xtr)
    Xte = selector.transform(Xte)

    model = RandomForestRegressor(
        n_estimators=600, random_state=42, n_jobs=-1,
        max_features="sqrt"
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    selected = names[selector.get_support()]
    imp = pd.Series(model.feature_importances_, index=selected)
    imp = imp.sort_values(ascending=False)

    return {
        "MAE": mean_absolute_error(yte, pred),
        "RMSE": np.sqrt(mean_squared_error(yte, pred)),
        "R2": r2_score(yte, pred),
        "selected": len(selected),
        "importance": imp,
        "model": model,
        "pre": pre,
        "selector": selector,
        "actual": yte.to_numpy(),
        "pred": pred
    }

with_grades = run_experiment(False)
without_grades = run_experiment(True)

results = pd.DataFrame([
    {
        "Experiment": "With G1 and G2",
        "MAE": with_grades["MAE"],
        "RMSE": with_grades["RMSE"],
        "R2": with_grades["R2"],
        "Selected_Features": with_grades["selected"]
    },
    {
        "Experiment": "Without G1 and G2",
        "MAE": without_grades["MAE"],
        "RMSE": without_grades["RMSE"],
        "R2": without_grades["R2"],
        "Selected_Features": without_grades["selected"]
    }
])

results.to_csv(BASE / "experiment_results.csv", index=False)
with_grades["importance"].rename("importance").to_csv(
    BASE / "feature_importance_with_G1_G2.csv"
)
without_grades["importance"].rename("importance").to_csv(
    BASE / "feature_importance_without_G1_G2.csv"
)

plt.figure(figsize=(10, 6))
with_grades["importance"].head(12).sort_values().plot(kind="barh")
plt.title("Top Predictors of Final Student Performance")
plt.xlabel("Random Forest Importance")
plt.tight_layout()
plt.savefig(BASE / "feature_importance.png", dpi=200)
plt.close()

plt.figure(figsize=(7, 6))
plt.scatter(with_grades["actual"], with_grades["pred"], alpha=0.7)
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Actual vs Predicted Final Grade")
plt.tight_layout()
plt.savefig(BASE / "actual_vs_predicted.png", dpi=200)
plt.close()

joblib.dump(
    {
        "preprocessor": with_grades["pre"],
        "selector": with_grades["selector"],
        "model": with_grades["model"]
    },
    BASE / "research_model.joblib"
)

print("=" * 65)
print("RESEARCH EXPERIMENT")
print("=" * 65)
print(f"Dataset: {len(df)} students, {df.shape[1]-1} predictors")
print("\nMODEL 1: Including previous grades G1 and G2")
print(f"MAE  = {with_grades['MAE']:.4f}")
print(f"RMSE = {with_grades['RMSE']:.4f}")
print(f"R2   = {with_grades['R2']:.4f}")
print(f"Selected features = {with_grades['selected']}")
print("\nMODEL 2: Excluding G1 and G2")
print(f"MAE  = {without_grades['MAE']:.4f}")
print(f"RMSE = {without_grades['RMSE']:.4f}")
print(f"R2   = {without_grades['R2']:.4f}")
print(f"Selected features = {without_grades['selected']}")
print("\nTop predictors without G1/G2:")
for k, v in without_grades["importance"].head(12).items():
    print(f"{k:45s} {v:.5f}")
