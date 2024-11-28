import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_files
import streamlit as st

# Заголовок приложения
st.title("Анализ тональности текста")

# Загрузка и обработка данных
@st.cache_resource
def load_and_train_model():
    # Загрузка данных
    train_data = load_files('aclImdb/train', categories=['pos', 'neg'])
    test_data = load_files('aclImdb/test', categories=['pos', 'neg'])

    X_train, y_train = train_data.data, train_data.target
    X_test, y_test = test_data.data, test_data.target

    # Векторизация текста
    vectorizer = CountVectorizer(binary=True, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Обучение модели
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Точность модели
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    return model, vectorizer, accuracy, classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])

# Загрузка модели
with st.spinner("Обучение модели, пожалуйста, подождите..."):
    model, vectorizer, accuracy, class_report = load_and_train_model()

# Отображение точности модели
st.write(f"**Точность модели:** {accuracy:.2f}")

# Форма для ввода текста
st.header("Проверка тональности текста")
user_input = st.text_area("Введите текст для анализа тональности:")

if st.button("Анализировать"):
    if user_input.strip():
        text_vec = vectorizer.transform([user_input])
        prediction = model.predict(text_vec)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.write(f"**Результат анализа:** {sentiment}")
    else:
        st.warning("Пожалуйста, введите текст.")

# Кнопка для отображения отчета по классификации
if st.button("Показать отчет по классификации"):
    st.text("Отчет по классификации:")
    st.text(class_report)
