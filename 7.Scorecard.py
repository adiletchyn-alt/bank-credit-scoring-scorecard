import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# =====================================================================
# 1. ЗАГРУЗКА, ОЧИСТКА И ВОССТАНОВЛЕНИЕ ОБУЧЕННОЙ МОДЕЛИ
# =====================================================================
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


# Сохраняем карты WoE, они понадобятся для финальных баллов
woe_maps_all = {}
df_woe = pd.DataFrame(index=df.index)
df_woe[target] = df[target]

for col in features:
    woe_maps_all[col] = get_woe_map(df_binned, col, target)
    df_woe[col + "_woe"] = df_binned[col].map(woe_maps_all[col])

X = df_woe.drop(columns=[target])
y = df_woe[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Получаем коэффициенты
intercept = model.intercept_[0]
coef_dict = dict(zip(X.columns, model.coef_[0]))

# =====================================================================
# 2. МАСШТАБИРОВАНИЕ В БАЛЛЫ (SCORECARD SCALING)
# =====================================================================
target_score = 600
target_odds = 50
pdo = 20

# Математические коэффициенты масштабирования
factor = pdo / np.log(2)
offset = target_score - factor * np.log(target_odds)

# Количество признаков
n_features = len(features)

# Базовая константа баллов (делится поровну между всеми признаками для красоты)
base_score = offset / n_features

scorecard_rows = []

# Рассчитываем баллы для каждого бакета каждого признака
for col in features:
    beta = coef_dict[col + "_woe"]
    col_woe_map = woe_maps_all[col]

    for bucket, woe_val in col_woe_map.items():
        # Формула расчета балла. Мы меняем знак на минус перед beta,
        # чтобы компенсировать отрицательные веса модели и сделать баллы логичными.
        score = base_score - (beta * woe_val * factor)

        scorecard_rows.append(
            {
                "Признак": col,
                "Группа (Бакет)": bucket,
                "WoE": round(woe_val, 4),
                "Балл (Score)": int(round(score)),
            }
        )

# Превращаем в датафрейм для красивого вывода
scorecard_df = pd.DataFrame(scorecard_rows)

print("--- СКОРОКАРТА (SCORECARD) ДЛЯ БАНКА ---")
# Выведем пример карты баллов для первых двух признаков (например, для возраста и дохода)
print(scorecard_df[scorecard_df["Признак"].isin(["age", "MonthlyIncome"])].to_string(index=False))

# Сохраняем всю карту баллов в Excel для резюме/портфолио
scorecard_df.to_excel("credit_scorecard.xlsx", index=False)
print("\nПолная скорокарта сохранена в файл: credit_scorecard.xlsx")
