import os
import pandas as pd

# 1. Загрузка данных
file_path = os.path.join("data", "cs-training.csv")
df = pd.read_csv(file_path, index_col=0)

print(f"Размер базы ДО очистки: {df.shape[0]} строк")

# 2. Очистка возраста: убираем строки, где возраст меньше 18 лет
df = df[df["age"] >= 18]

# 3. Очистка аномальных просрочек:
# Создаем список колонок, где встречаются странные значения 96 и 98
delay_columns = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]

# Заменяем значения 96 и 98 на NaN (пропуски) во всех трех колонках сразу
for col in delay_columns:
    df.loc[df[col] >= 96, col] = None

print(f"Размер базы ПОСЛЕ очистки возраста: {df.shape[0]} строк")
print("\nПроверка пропусков после очистки просрочек:")
print(df[delay_columns].isnull().sum())