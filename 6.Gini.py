import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ==========================================
# 1. ПОВТОРЯЕМ ПРЕДЫДУЩИЕ ШАГИ (ЗАГРУЗКА И ТРАНСФОРМАЦИЯ)
# ==========================================
file_path = os.path.join("data", "cs-training.csv")
df = pd.read_csv(file_path, index_col=0)

df = df[df["age"] >= 18]
delay_columns = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]
for col in delay_columns:
    df.loc[df[col] >= 96, col] = None

target = "SeriousDlqin2yrs"
features = [
    col
    for col in df.columns
    if col not in [target, "NumberRealEstateLoansOrLines"]
]

df_binned = pd.DataFrame(index=df.index)
df_binned[target] = df[target]

for col in features:
    if df[col].nunique() <= 10:
        df_binned[col] = df[col].astype(str).fillna("Missing")
    else:
        try:
            df_binned[col] = (
                pd.qcut(df[col], q=5, duplicates="drop")
                .astype(str)
                .fillna("Missing")
            )
        except Exception:
            df_binned[col] = df[col].astype(str).fillna("Missing")


def get_woe_map(data, feature, target_var):
    grouped = data.groupby(feature, observed=False)[target_var].agg(["count", "sum"])
    grouped.columns = ["Total", "Bad"]
    grouped["Good"] = grouped["Total"] - grouped["Bad"]
    total_bad = data[target_var].sum()
    total_good = data[target_var].count() - total_bad
    share_bad = (grouped["Bad"] / total_bad).replace(0, 0.0001)
    share_good = (grouped["Good"] / total_good).replace(0, 0.0001)
    return np.log(share_good / share_bad).to_dict()


df_woe = pd.DataFrame(index=df.index)
df_woe[target] = df[target]

for col in features:
    woe_map = get_woe_map(df_binned, col, target)
    df_woe[col + "_woe"] = df_binned[col].map(woe_map)

# ==========================================
# 2. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ
# ==========================================
X = df_woe.drop(columns=[target])
y = df_woe[target]

# Разделяем в пропорции 70% на 30%
# random_state=42 гарантирует, что при каждом запуске разделение будет одинаковым
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==========================================
# 3. ОБУЧЕНИЕ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ
# ==========================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ==========================================
# 4. РАСЧЕТ МЕТРИК КАЧЕСТВА (ROC-AUC И GINI)
# ==========================================
# Считаем вероятности дефолта для обучающей и тестовой выборки
train_preds = model.predict_proba(X_train)[:, 1]
test_preds = model.predict_proba(X_test)[:, 1]

# Считаем ROC-AUC
auc_train = roc_auc_score(y_train, train_preds)
auc_test = roc_auc_score(y_test, test_preds)

# Переводим в коэффициент Джини
gini_train = 2 * auc_train - 1
gini_test = 2 * auc_test - 1

print("--- РЕЗУЛЬТАТЫ ОБУЧЕНИЯ МОДЕЛИ ---")
print(f"Gini на обучающей выборке (Train): {gini_train:.4f}")
print(f"Gini на тестовой выборке (Test):  {gini_test:.4f}")

print("\n--- КОЭФФИЦИЕНТЫ МОДЕЛИ ---")
intercept = model.intercept_[0]
print(f"Константа (Intercept): {intercept:.4f}")

coefficients = pd.DataFrame(
    {"Признак": X.columns, "Коэффициент (Beta)": model.coef_[0]}
)
print(coefficients.to_string(index=False))