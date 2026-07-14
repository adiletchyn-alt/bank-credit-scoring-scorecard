import os
import numpy as np
import pandas as pd

# ==========================================
# 1. ЗАГРУЗКА И ОЧИСТКА ДАННЫХ
# ==========================================
file_path = os.path.join("data", "cs-training.csv")
df = pd.read_csv(file_path, index_col=0)

# Базовая очистка
df = df[df["age"] >= 18]
delay_columns = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]
for col in delay_columns:
    df.loc[df[col] >= 96, col] = None

target = "SeriousDlqin2yrs"


# ==========================================
# 2. УНИВЕРСАЛЬНАЯ ФУНКЦИЯ WoE / IV
# ==========================================
def calculate_woe_iv(data, feature, target_var):
    # Временно заполняем системные пропуски строкой 'Missing'
    # Это выделит клиентов без данных в отдельную аналитическую группу
    s = data[feature].fillna("Missing")

    grouped = data.groupby(s, observed=False)[target_var].agg(["count", "sum"])
    grouped.columns = ["Total", "Bad"]
    grouped["Good"] = grouped["Total"] - grouped["Bad"]

    total_bad = data[target_var].sum()
    total_good = data[target_var].count() - total_bad

    grouped["Share_Bad"] = grouped["Bad"] / total_bad
    grouped["Share_Good"] = grouped["Good"] / total_good

    # Защита от деления на ноль
    grouped["Share_Bad"] = grouped["Share_Bad"].replace(0, 0.0001)
    grouped["Share_Good"] = grouped["Share_Good"].replace(0, 0.0001)

    grouped["WoE"] = np.log(grouped["Share_Good"] / grouped["Share_Bad"])
    grouped["IV_bucket"] = (grouped["Share_Good"] - grouped["Share_Bad"]) * grouped[
        "WoE"
    ]

    return grouped["IV_bucket"].sum()


# ==========================================
# 3. АВТОМАТИЧЕСКИЙ БИННИНГ ДЛЯ ВСЕХ КОЛОНОК
# ==========================================
features_to_analyze = [col for col in df.columns if col != target]
iv_results = {}

for col in features_to_analyze:
    # Создаем временную колонку для сгруппированных данных
    temp_col_name = col + "_binned"

    # Если уникальных значений мало (например, количество просрочек или детей),
    # то группируем как есть, без разбиения на интервалы
    if df[col].nunique() <= 10:
        df[temp_col_name] = df[col].astype(str)
    else:
        # Если значений много (доход, возраст, коэффициенты), бьем на 5 равных групп (квантилей)
        # duplicates='drop' защищает от ошибок, если группы пересекаются
        try:
            df[temp_col_name] = pd.qcut(
                df[col], q=5, duplicates="drop"
            ).astype(str)
        except Exception:
            df[temp_col_name] = df[col].astype(str)

    # Считаем IV для получившихся групп
    iv_results[col] = calculate_woe_iv(df, temp_col_name, target)

# Превращаем результаты в красивую таблицу и сортируем по убыванию силы признака
iv_df = pd.DataFrame(list(iv_results.items()), columns=["Признак", "Information_Value (IV)"])
iv_df = iv_df.sort_values(by="Information_Value (IV)", ascending=False).reset_index(drop=True)

print("--- ИТОГОВЫЙ РЕЙТИНГ СИЛЫ ПРИЗНАКОВ (IV) ---")
print(iv_df)