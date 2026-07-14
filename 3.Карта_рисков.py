import os
import numpy as np
import pandas as pd

# ==========================================
# 1. ЗАГРУЗКА И ОЧИСТКА ДАННЫХ
# ==========================================
file_path = os.path.join("data", "cs-training.csv")
df = pd.read_csv(file_path, index_col=0)

# Фильтруем возраст
df = df[df["age"] >= 18]

# Заменяем аномальные коды просрочек 96 и 98 на пропуски (None)
delay_columns = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]
for col in delay_columns:
    df.loc[df[col] >= 96, col] = None


# ==========================================
# 2. БИННИНГ (РАЗБИЕНИЕ НА ГРУППЫ)
# ==========================================
# Нарезаем возраст на стандартные банковские когорты
age_bins = [18, 30, 45, 60, 75, 120]
df["age_group"] = pd.cut(df["age"], bins=age_bins, include_lowest=True)

# Переводим интервалы в текст для удобства работы алгоритма
df["age_group"] = df["age_group"].astype(str)


# ==========================================
# 3. ФУНКЦИЯ РАСЧЕТА WoE И IV
# ==========================================
def calculate_woe_iv(data, feature, target):
    # Считаем общее количество клиентов и число дефолтов в каждой группе
    grouped = data.groupby(feature, observed=False)[target].agg(["count", "sum"])
    grouped.columns = ["Total_Customers", "Bad_Customers"]

    # Good — клиенты без дефолта
    grouped["Good_Customers"] = (
        grouped["Total_Customers"] - grouped["Bad_Customers"]
    )

    # Общие итоги по всей базе для расчета долей
    total_bad = data[target].sum()
    total_good = data[target].count() - total_bad

    # Доля плохих и хороших клиентов группы от общего объема рынка
    grouped["Share_Bad"] = grouped["Bad_Customers"] / total_bad
    grouped["Share_Good"] = grouped["Good_Customers"] / total_good

    # Защита от деления на 0
    grouped["Share_Bad"] = grouped["Share_Bad"].replace(0, 0.0001)
    grouped["Share_Good"] = grouped["Share_Good"].replace(0, 0.0001)

    # Расчет WoE и IV по формулам математической статистики рисков
    grouped["WoE"] = np.log(grouped["Share_Good"] / grouped["Share_Bad"])
    grouped["IV_bucket"] = (
        grouped["Share_Good"] - grouped["Share_Bad"]
    ) * grouped["WoE"]

    total_iv = grouped["IV_bucket"].sum()

    return grouped[["Total_Customers", "Bad_Customers", "WoE"]], total_iv


# Запускаем расчет для созданных нами групп возраста
woe_table, iv_value = calculate_woe_iv(df, "age_group", "SeriousDlqin2yrs")

print("--- ТАБЛИЦА WoE ДЛЯ ВОЗРАСТА ---")
print(woe_table)
print(f"\nОбщее значение IV для признака Возраст: {iv_value:.4f}")