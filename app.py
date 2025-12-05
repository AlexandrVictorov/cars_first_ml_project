from sklearn.linear_model import LinearRegression
import pickle
import os
import streamlit as st
from sklearn.linear_model import Ridge
import pandas as pd


st.set_page_config(
    page_title="Предсказание цены авто",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource  # Кэшируем модель (загружается только один раз)
def load_model():
    with open('models/linear_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    return model, feature_names



st.header("Рассчет стоимости автомобиля по параметрам 🚗")   
st.subheader("📝 Введите параметры автомобиля")
year = st.number_input("Год авто", min_value=1970, max_value=2025) 
km_driven = st.number_input("Пробег", min_value=0) 
mileage = st.slider("Экономия топлива каждые 100км", 5, 40, 20) 
engine = st.number_input("Объем двигателя в кубических сантиметрах СС", min_value=600, max_value=2000)
max_power = st.number_input("Мощность в лошадиных силах", min_value=40, max_value=300)
seats = st.number_input("Количество сидений", min_value=0)

model, feature_names = load_model()
#st.text(model)
#st.text(feature_names) 

if st.button('💰 Рассчитать стоимость', type='primary'):
    try:
        values = [year, km_driven, mileage, engine, max_power, seats]
        df_input = pd.DataFrame([values], columns=feature_names)
        
        st.text(f"Данные для предсказания: {df_input}")
        
        prediction = model.predict(df_input)
        
        if prediction > 0:
           st.success(f"### Предсказанная стоимость: **В местной валюте {prediction[0]:,.2f}**")
        else:
           st.error('Результат отрицательный, попробуйте изменить данные.')
        
    except Exception as e:
        st.error(f"❌ Ошибка: {str(e)}")
