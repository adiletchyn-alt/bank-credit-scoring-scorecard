import os
import numpy as np
import pandas as pd

# ==========================================
# 1. ЗАГРУЗКА И БАЗОВАЯ ОЧИСТКА
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

# Исключаем слабый признак (IV < 0.02)
features = [
    col
    for col in df.columns
    if col not in [target, "NumberRealEstateLoansOrLines"]
]

# ==========================================
# 2. ПОДГОТОВКА И ТРАНСФОРМАЦИЯ ДАННЫХ
# ==========================================
df_binned = pd.DataFrame(index=df.index)
df_binned[target] = df[target]

# Проводим биннинг аналогично прошлому шагу
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

# Словарь, где мы будем хранить таблицы WoE для каждого признака
woe_maps = {}
df_woe = pd.DataFrame(index=df.index)
df_woe[target] = df[target]


# Функция формирования словаря соответствия Группа -> Значение WoE
def get_woe_map(data, feature, target_var):
    grouped = data.groupby(feature, observed=False)[target_var].agg(["count", "sum"])
    grouped.columns = ["Total", "Bad"]
    grouped["Good"] = grouped["Total"] - grouped["Bad"]

    total_bad = data[target_var].sum()
    total_good = data[target_var].count() - total_bad

    share_bad = grouped["Bad"] / total_bad
    share_good = grouped["Good"] / total_good

    share_bad = share_bad.replace(0, 0.0001)
    share_good = share_good.replace(0, 0.0001)

    woe = np.log(share_good / share_bad)
    return woe.to_dict()


# Трансформируем датасет: заменяем текстовые интервалы на числа WoE
for col in features:
    woe_maps[col] = get_woe_map(df_binned, col, target)
    df_woe[col + "_woe"] = df_binned[col].map(woe_maps[col])

print("--- ТРАНСФОРМАЦИЯ ЗАВЕРШЕНА ---")
print("Пример преобразованных данных (первые 5 строк):")
print(df_woe.head())