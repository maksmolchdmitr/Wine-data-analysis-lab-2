import numpy as np
import pandas as pd


def classify_data_types_and_create_new_variable(df):
    # Определение и классификация типов данных
    print("Типы данных в наборе данных:")
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"{col}: категориальный")
        else:
            print(f"{col}: числовой")

    # Создание новой переменной 'цена_за_балл'
    df['цена_за_балл'] = df['price'] / df['points']
    return df


def create_frequency_table_and_compare_means(df):
    # Создание частотной таблицы для переменной 'страна'
    frequency_table = df['country'].value_counts()
    print("Частотная таблица для переменной 'страна':", frequency_table, sep="\n")

    # Сравнение средней цены в выборке и по всему набору данных
    sample_size = 100
    sample_data = df.sample(n=sample_size, random_state=42)
    sample_mean_price = sample_data['price'].mean()
    population_mean_price = df['price'].mean()
    print(f"Средняя цена в выборке: {sample_mean_price:.2f}")
    print(f"Средняя цена по всему набору данных: {population_mean_price:.2f}")


def calculate_descriptive_statistics(df):
    # Расчет описательных статистик для числовых переменных
    descriptive_stats = df.describe(include=[np.number])
    print("Описательные статистики для числовых переменных:", descriptive_stats, sep="\n")

    # Обсуждение мер центральной тенденции
    print("Обсуждение мер центральной тенденции:")
    print("""
    Для переменной "points" лучше всего подходит медиана (50% квантиль) в качестве меры центральной тенденции.
    Т.к. диапазон оценок довольно узок (от 80 до 100), и среднее значение может быть немного смещено
    вверх из-за наличия нескольких максимальных оценок, равных 100.
    Медиана же будет отражать точку, где ровно половина вин имеет оценку ниже этой точки, и половина - выше, что будет
    более информативным в этом контексте.
    
    Для переменной "price" лучше всего подходит mean в качестве меры центральной тенденции.
    Т.к. распределение цен довольно симметрично вокруг среднего значения, и нет существенных
    признаков наличия выбросов или сильной асимметрии.
    (Среднее значение цен (37.55) расположено довольно близко к медиане (32.00))
""")


def calculate_z_scores_and_discuss_outliers(df):
    # Расчет Z-баллов для переменной 'цена'
    mean = df['price'].mean()
    df['z_баллы_для_price'] = (df['price'] - mean) / df['price'].std()
    print("Z-баллы для переменной 'price':", df['z_баллы_для_price'].describe(), df['z_баллы_для_price'], sep="\n")

    # Обсуждение порога и влияния выбросов
    print()
    print("Обсуждение порога и влияния выбросов:")
    print("Как порог для выбросов, можно использовать значение |z| > 3, так как оно соответствует примерно "
          "0,3% от всего нормального распределения. Этот порог позволяет отсеять крайние значения, которые могут "
          "искажать результаты анализа.")
    threshold = 3
    print("Кол-во выбросов для колонки 'price' = ", len(df[abs(df['z_баллы_для_price']) > threshold]))
    print(len(df))
    df = df[abs(df['z_баллы_для_price']) <= threshold]
    print(len(df))


data_frame = pd.read_csv('winemag-data_first150k.csv')
data_frame.drop(columns=["Unnamed: 0"], inplace=True)
data_frame.dropna(inplace=True)
print("Описание датасета:", data_frame.info, sep="\n")
print("Колонки:", data_frame.columns, sep="\n")
classify_data_types_and_create_new_variable(data_frame)
create_frequency_table_and_compare_means(data_frame)
calculate_descriptive_statistics(data_frame)
calculate_z_scores_and_discuss_outliers(data_frame)
