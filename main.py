import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def fill_na_with_mean(df):
    # Определить, какие столбцы являются числовыми, а какие - категориальными
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    # Заменить все NaN значения на среднее значение для числовых столбцов
    dataframe_filled = df.copy()
    dataframe_filled[numeric_columns] = dataframe_filled[numeric_columns].fillna(df[numeric_columns].mean())

    # Заменить все NaN значения на самое частое значение для категориальных столбцов
    dataframe_filled[categorical_columns] = dataframe_filled[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

    return dataframe_filled


def classify_data_types_and_create_new_variable(df):
    # Определение и классификация типов данных
    print("Типы данных в наборе данных:")
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"{col}: категориальный")
        else:
            print(f"{col}: числовой")

    # Создание новой переменной 'point_price'
    df['point_price'] = df['price'] / df['points']
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
    df['price_z_score'] = (df['price'] - mean) / df['price'].std()
    print("Z-баллы для переменной 'price':", df['price_z_score'].describe(), df['price_z_score'], sep="\n")

    # Обсуждение порога и влияния выбросов
    print()
    print("Обсуждение порога и влияния выбросов:")
    print("Как порог для выбросов, можно использовать значение |z| > 3, так как оно соответствует примерно "
          "0,3% от всего нормального распределения. Этот порог позволяет отсеять крайние значения, которые могут "
          "искажать результаты анализа.")
    threshold = 3
    print("Кол-во выбросов для колонки 'price' = ", len(df[abs(df['price_z_score']) > threshold]))


def calculate_correlation(df):
    # Вычислить матрицу корреляции
    correlation_matrix = df.corr(numeric_only=True)

    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
    plt.show()


def plot_histogram_range(df):
    # Гистограмма для переменной оценки
    plt.figure(figsize=(10, 6))
    sns.histplot(df["points"], kde=True)
    plt.title("Распределение оценок")
    plt.xlabel("Оценка")
    plt.ylabel("Частота")
    plt.show()

    # Диаграмма размаха для переменной цены
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df["price"])
    plt.title("Распределение цен")
    plt.xlabel("Цена")
    plt.ylabel("")
    plt.show()


def plot_scatter(df):
    # Диаграмма рассеяния для визуализации взаимосвязи между ценой и оценками
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="points", y="price", data=df, hue="country")
    plt.title("Взаимосвязь между ценой и оценками")
    plt.xlabel("Оценка")
    plt.ylabel("Цена")
    plt.show()


def linear_regression(df):
    # Построить модель линейной регрессии
    X = sm.add_constant(df["points"])
    model = sm.OLS(df["price"], X).fit()

    # Вывести значение R² и его интерпретацию
    r_squared = model.rsquared
    print(f"Значение R²: {r_squared}")
    print("R² представляет собой долю вариации зависимой переменной, объясненной независимой переменной в модели.")


def logistic_regression(df):
    # Подготовить данные для логистической регрессии
    X = df[["points", "price"]]
    y = df["country"]

    # Разбить данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Построить модель логистической регрессии
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Оценить точность модели
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность модели: {accuracy}")


data_frame = pd.read_csv('winemag-data_first150k.csv')
data_frame.drop(columns=["Unnamed: 0"], inplace=True)
# data_frame.dropna(inplace=True)
data_frame = fill_na_with_mean(data_frame)
print("Описание датасета:", data_frame.info, sep="\n")
print("Колонки:", data_frame.columns, sep="\n")
classify_data_types_and_create_new_variable(data_frame)
create_frequency_table_and_compare_means(data_frame)
calculate_descriptive_statistics(data_frame)
calculate_z_scores_and_discuss_outliers(data_frame)
# data_frame = data_frame[abs(data_frame['price_z_score']) <= 3]
calculate_correlation(data_frame)
plot_histogram_range(data_frame)
plot_scatter(data_frame)
linear_regression(data_frame)
logistic_regression(data_frame)


