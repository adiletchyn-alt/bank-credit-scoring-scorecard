import pandas as pd
import streamlit as st

# ==========================================
# 1. НАСТРОЙКА СТРАНИЦЫ И ЗАГРУЗКА ДАННЫХ
# ==========================================
st.set_page_config(page_title="Кредитный Конвейер Банка", page_icon="🏦", layout="centered")

st.title("🏦 Кредитный скоринг: Оценка заемщика")
st.write("Передвигайте ползунки, чтобы рассчитать скоринговый балл клиента и получить автоматическое решение.")

# Загружаем нашу карту баллов из сохраненного Excel-файла
@st.cache_data
def load_scorecard():
    return pd.read_excel("credit_scorecard.xlsx")

try:
    scorecard_df = pd.read_excel("credit_scorecard.xlsx")
except FileNotFoundError:
    st.error("Ошибка: Файл 'credit_scorecard.xlsx' не найден! Убедитесь, что вы запустили прошлый скрипт Scorecard.py.")
    st.stop()


# Вспомогательная функция для поиска баллов по интервалам
def get_points_from_interval(val, data, feature_name):
    # Фильтруем карту баллов только для нужного признака
    feature_rows = data[data["Признак"] == feature_name]
    
    for _, row in feature_rows.iterrows():
        bucket_str = row["Группа (Бакет)"]
        
        # Пропускаем бакет Missing в данном упрощенном интерактивном калькуляторе
        if bucket_str == "Missing":
            continue
            
        # Парсим строку интервала вида "(39.0, 48.0]"
        bucket_str = bucket_str.replace("(", "").replace("]", "")
        left, right = map(float, bucket_str.split(", "))
        
        # Проверяем, входит ли введенное число в интервал
        if left < val <= right:
            return int(row["Балл (Score)"])
            
    # Если значение выбивается (например, слишком молодой или старый), возвращаем балл крайней группы
    return int(feature_rows.iloc[0]["Балл (Score)"])

# ==========================================
# 2. ИНТЕРФЕЙС ПОЛЬЗОВАТЕЛЯ (ВВОД ДАННЫХ)
# ==========================================
st.subheader("📋 Данные анкеты клиента")

# Ползунок для выбора возраста от 18 до 100 лет
age_input = st.slider("Возраст клиента (лет):", min_value=18, max_value=100, value=35, step=1)

# Ползунок для ежемесячного дохода от 0 до 50 000 долларов/рублей
income_input = st.slider("Ежемесячный доход:", min_value=0, max_value=50000, value=5000, step=500)

# ==========================================
# 3. РАСЧЕТ БАЛЛОВ И ВЕРДИКТ
# ==========================================
# Находим баллы для возраста и дохода на основе нашей модели
age_points = get_points_from_interval(age_input, scorecard_df, "age")
income_points = get_points_from_interval(income_input, scorecard_df, "MonthlyIncome")

# В этой модели мы взяли только 2 признака для демо-версии. Считаем сумму:
total_score = age_points + income_points

st.markdown("---")
st.subheader("📊 Результат оценки")

# Отображаем баллы в виде красивых карточек
col1, col2, col3 = st.columns(3)
col1.metric("Баллы за возраст", f"+{age_points}")
col2.metric("Баллы за доход", f"+{income_points}")
col3.metric("ИТОГОВЫЙ БАЛЛ", f"{total_score}")

# Установим пороговое значение одобрения (Cut-off Score).
# Сумма минимальных баллов (48+50)=98, максимальных (67+59)=126. Возьмем порог в 105 баллов.
cutoff_score = 105

if total_score >= cutoff_score:
    st.success(f"🟢 **РЕШЕНИЕ: КРЕДИТ ОДОБРЕН!** (Финальный балл {total_score} >= порога отсечения {cutoff_score})")
else:
    st.error(f"🔴 **РЕШЕНИЕ: ОТКАЗ В ВЫДАЧЕ.** (Финальный балл {total_score} ниже порога отсечения {cutoff_score})")