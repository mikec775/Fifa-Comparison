import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

fifa14 = pd.read_csv('data/fifa14.csv', low_memory=False)
fifa24 = pd.read_csv('data/fifa24.csv', encoding="latin1", low_memory=False)

cols = ["short_name", "long_name", "age", "overall", "potential"]

def clean(df):
    df = df[cols].copy()
    for c in ["age", "overall", "potential"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["name_clean"] = df["long_name"].str.lower().str.strip()
    df = df.dropna(subset=["age", "overall", "potential", "name_clean"])
    return df.drop_duplicates(subset="name_clean")

fifa14_clean = clean(fifa14)
fifa24_clean = clean(fifa24)

fifa14_sub = fifa14_clean.rename(columns={
                                    "short_name": "Player_name_14", 
                                    "age": "Age_14", 
                                    "overall": "Overall_14", 
                                    "potential": "Potential_14"
                                    }).drop(columns=["long_name"])

fifa24_sub = fifa24_clean.rename(columns={
                                    "short_name": "Player_name_24", 
                                    "age": "Age_24", 
                                    "overall": "Overall_24", 
                                    "potential": "Potential_24"
                                    }).drop(columns=["long_name"])

players = fifa14_sub.merge(fifa24_sub, how="inner", on="name_clean")
players = players[players["Age_24"] <= 30].copy()

players["potential_diff"] = players["Potential_24"] - players["Potential_14"]
players["actual"] = players["Overall_24"] - players["Overall_14"]

print(players.head())
print(f"Matched players: {len(players)}")

print(players[[
    "Player_name_14",
    "Overall_24",
    "Potential_24",
    "potential_diff",
    "actual"
]].describe())

plt.figure(figsize=(10, 6))

plt.scatter(players["potential_diff"], players["actual"], alpha=0.5)
plt.xlabel("Potential difference")
plt.ylabel("Overall difference")
plt.title("Potential difference vs. Overall difference for players under 30 in FIFA 24 vs FIFA 14")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

features = ["Age_14", "Overall_14", "Potential_14"]
target = "Overall_24"

ml = players.dropna(subset=features + [target]).copy()
x, y = ml[features], ml[target]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

def metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred)
    }

results = {}

results["Baseline: Overall_14"] = metrics(y_test, X_test["Overall_14"])
results["Baseline: Potential_14"] = metrics(y_test, X_test["Potential_14"])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )
}

fitted = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    fitted[name] = model
    results[name] = metrics(y_test, model.predict(X_test))

results_df = pd.DataFrame(results).T.sort_values("mae")         
print("\nResults:\n")
print(results_df.round(3))

print("Cross fold mae:")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    cv_mae = cross_val_score(model, x, y, scoring="neg_mean_absolute_error", cv=kf)
    print(f"{name}: cv mae = {cv_mae.mean():.2f} ± {cv_mae.std():.2f}")


models_rows = [i for i in results_df.index if not i.startswith("Baseline")]
best_name = models_rows[0]
y_pred = fitted[best_name].predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(y_test, y_pred, alpha=0.4)

lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]

axes[0].plot(lims, lims, 'r--', label="Perfect prediction")
axes[0].set_xlabel("Actual overall 24")
axes[0].set_ylabel("Predicted overall 24")
axes[0].set_title(f"{best_name}: predicted vs. actual")
axes[0].legend()
axes[0].grid(alpha=0.2)

res = y_test - y_pred
axes[1].hist(res, bins=30, edgecolor="black")
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set_xlabel("Prediction error")
axes[1].set_ylabel("Players")
axes[1].set_title("Error Distribution")
axes[1].grid(alpha=0.2)

plt.tight_layout()
plt.show()

print("Random forrest feature importances:")
print(pd.Series(fitted["Random Forest"].feature_importances_, index=features).sort_values(ascending=False))

print("Linear Regression coefficients:")
print(pd.Series(fitted["Linear Regression"].coef_, index=features))