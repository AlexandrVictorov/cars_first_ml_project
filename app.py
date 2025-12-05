from sklearn.linear_model import LinearRegression
import pickle
import os
import streamlit as st
from sklearn.linear_model import Ridge
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


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

def simple_pairplot_plotly(df, numeric_columns): #построение графика паирплот с плотли
    df_numeric = df[numeric_columns].dropna()

    # Создаём scatter matrix
    fig = px.scatter_matrix(df_numeric, dimensions=numeric_columns, height=800, title="Pairplot: взаимосвязи между признаками")
    
    #Код ниже взял в deepseek так как с плотли никогда не работал.
    fig.update_traces(
        diagonal_visible=False,  # Уберём стандартные диагонали
        showupperhalf=False,     # Покажем только нижнюю половину
        marker=dict(
            size=4,
            opacity=0.6,
            line=dict(width=0.5, color='white')
        )
    )
    
    # Добавляем гистограммы на диагональ отдельно
    for i in range(len(numeric_columns)):
        fig.add_trace(
            go.Histogram(
                x=df_numeric[numeric_columns[i]],
                xaxis=f'x{i+1}',
                yaxis=f'y{i+1}',
                showlegend=False,
                marker_color=px.colors.qualitative.Plotly[i % 10]
            )
        )
    
    fig.update_layout(
        showlegend=False,
        hovermode='closest'
    )
    
    return fig

st.header("Рассчет стоимости автомобиля по параметрам 🚗")   
st.subheader("📝 Введите параметры автомобиля")
year = st.number_input("Год авто", min_value=1970, max_value=2025) 
km_driven = st.number_input("Пробег", min_value=0) 
mileage = st.slider("Экономия топлива каждые 100км", 5, 40, 20) 
engine = st.number_input("Объем двигателя в кубических сантиметрах СС", min_value=600, max_value=2000)
max_power = st.number_input("Мощность в лошадиных силах", min_value=40, max_value=300)
seats = st.number_input("Количество сидений", min_value=2)

model, feature_names = load_model()
#st.text(model)
#st.text(feature_names) 

if st.button('💰 Рассчитать стоимость', type='primary'):
    try:
        values = [year, km_driven, mileage, engine, max_power, seats]
        df_input = pd.DataFrame([values], columns=feature_names)
        print(df_input)
        print()
        st.text(f"Данные для предсказания: {df_input}")
        
        prediction = model.predict(df_input)
        
        if prediction > 0:
           st.success(f"### Предсказанная стоимость: **В местной валюте {prediction[0]:,.2f}**")
        else:
           st.error('Результат отрицательный, попробуйте изменить данные.')
        
    except Exception as e:
        st.error(f"❌ Ошибка: {str(e)}")

st.header("📊 Визуализация датасета, на чем обучалась модель линейной регрессии.")

df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')

print("Train data shape:", df_train.shape)
print("Test data shape: ", df_test.shape)

df_train['mileage'] = df_train['mileage'].str.split().str[0].astype(float)
df_train['engine'] = df_train['engine'].str.split().str[0].astype(float)
df_train['max_power'] = df_train['max_power'].str.replace('bhp', '').str.split().str[0].astype(float)
#то же самое для теста
df_test['mileage'] = df_test['mileage'].str.split().str[0].astype(float)
df_test['engine'] = df_test['engine'].str.split().str[0].astype(float)
df_test['max_power'] = df_test['max_power'].str.replace('bhp', '').str.split().str[0].astype(float)

#ЗАПОЛНЯЮ ПРОПУСКИ  
null_sers = ['mileage', 'engine', 'max_power', 'seats']
for ser in null_sers:
  median = df_train[ser].median()
  df_train[ser] = df_train[ser].fillna(median)
  df_test[ser] = df_test[ser].fillna(median)

#удаляю дубли 
without_target = df_train.drop('selling_price', axis=1)
df_train = df_train.drop_duplicates(subset=without_target.columns, keep='first') #удаляю повторяющиеся строки
df_train = df_train.reset_index(drop=True) #обновляю индексы

df_train = df_train.drop('torque', axis=1)
df_test = df_test.drop('torque', axis=1)

df_train['engine'] = df_train['engine'].astype(int)
df_test['engine'] = df_test['engine'].astype(int)

df_train['seats'] = df_train['seats'].astype(int)
df_test['seats'] = df_test['seats'].astype(int)
print(df_train['selling_price'].describe())

numeric_category = ['year', 'selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
#вызываю функцию для построения паирплот
st.subheader("Pairplot для тренеровочного датасета.")
fig = simple_pairplot_plotly(df_train, numeric_category)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Pairplot для тестового датасета.")
fig = simple_pairplot_plotly(df_test, numeric_category)
st.plotly_chart(fig, use_container_width=True)

#получаю веса модели
coefficients = model.coef_#[0]
coeff_df = pd.DataFrame({'Признак': feature_names, 'Коэффициент': coefficients,})
print(coeff_df)

st.subheader("📊 Визуализация весов модели.")

#код ниже для плотли сгенерировал в deepseek
fig1 = px.bar(
    coeff_df, 
    x='Признак', 
    y='Коэффициент',
    title='Веса признаков в модели',
    color='Коэффициент',  # Цвет зависит от значения
    color_continuous_scale=['red', 'gray', 'green'],  # Красный для отрицательных, зеленый для положительных
    text='Коэффициент'
)

fig1.update_traces(
    texttemplate='%{text:.2f}',  # Формат чисел
    textposition='outside'
)

fig1.update_layout(
    xaxis_title="Признаки",
    yaxis_title="Значение коэффициента",
    showlegend=False,
    height=500
)

st.plotly_chart(fig1, use_container_width=True)