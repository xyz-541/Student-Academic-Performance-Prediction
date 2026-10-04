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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

BASE = Path(__file__).resolve().parent
DATA = BASE / "student-mat.csv"
df = pd.read_csv(DATA, sep=";")

# Target: final mathematics grade G3 (0-20).
target = "G3"
X = df.drop(columns=[target])
y = df[target]

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols)
], remainder="drop")

# Embedded feature selection: retain features with importance >= median.
selector_estimator = RandomForestRegressor(
    n_estimators=400, random_state=42, n_jobs=-1, max_features="sqrt"
)
selector = SelectFromModel(selector_estimator, threshold="median")

model = RandomForestRegressor(
    n_estimators=600, random_state=42, n_jobs=-1, max_features="sqrt"
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Fit preprocessing on training data only preventing leakage.
X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()
selector.fit(X_train_p, y_train)
X_train_s = selector.transform(X_train_p)
X_test_s = selector.transform(X_test_p)

model.fit(X_train_s, y_train)
pred = model.predict(X_test_s)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

selected_names = feature_names[selector.get_support()]
importance = pd.Series(model.feature_importances_, index=selected_names).sort_values(ascending=False)

print("=" * 60)
print("STUDENT ACADEMIC PERFORMANCE MODEL")
print("=" * 60)
print(f"Dataset shape: {df.shape}")
print(f"Target: {target}")
print(f"Numeric features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")
print(f"Encoded features: {len(feature_names)}")
print(f"Selected features: {len(selected_names)}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2:  {r2:.4f}")
print()
print("Top selected predictors:")
for name, value in importance.head(15).items():
    print(f"{name:45s} {value:.5f}")

# Save reusable artifacts.
joblib.dump(
    {"preprocessor": preprocessor, "selector": selector, "model": model},
    BASE / "student_performance_model.joblib"
)

pd.DataFrame({
    "actual_G3": y_test.to_numpy(),
    "predicted_G3": np.round(pred, 3)
}).to_csv(BASE / "predictions.csv", index=False)

importance.rename("importance").to_csv(BASE / "feature_importance.csv")
