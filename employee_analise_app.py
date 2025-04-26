import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import warnings

# Настройки
warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="HR Analytics System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("📊 HR Analytics Dashboard")
st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)


# Функции загрузки данных
@st.cache_data
def load_sample_data():
    """Загрузка встроенных демо-данных"""
    data = {
        'employee_id': np.arange(1, 501),
        'age': np.random.randint(22, 65, size=500),
        'gender': np.random.choice(['Male', 'Female'], size=500, p=[0.55, 0.45]),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'],
                                      size=500, p=[0.2, 0.5, 0.25, 0.05]),
        'department': np.random.choice(['HR', 'IT', 'Finance', 'Marketing', 'Operations'],
                                       size=500, p=[0.1, 0.3, 0.2, 0.2, 0.2]),
        'position': np.random.choice(['Junior', 'Middle', 'Senior', 'Manager', 'Director'],
                                     size=500, p=[0.3, 0.4, 0.2, 0.08, 0.02]),
        'salary': np.round(np.random.normal(5000, 2000, 500)).clip(2000, 15000),
        'experience': np.random.randint(1, 20, size=500),
        'performance_score': np.random.randint(60, 101, size=500),
        'satisfaction_score': np.round(np.random.normal(7.5, 1.5, 500)).clip(1, 10),
        'overtime': np.random.choice(['Yes', 'No'], size=500, p=[0.35, 0.65]),
        'attrition': np.random.choice(['Yes', 'No'], size=500, p=[0.2, 0.8]),
        'training_hours': np.random.randint(0, 41, size=500)
    }
    df = pd.DataFrame(data)
    df['salary'] = df['salary'].astype(int)
    df['satisfaction_score'] = df['satisfaction_score'].astype(int)
    return df


@st.cache_data
def load_uploaded_data(uploaded_file):
    """Загрузка данных из CSV файла"""
    try:
        df = pd.read_csv(uploaded_file)

        # Проверка обязательных полей
        required_columns = {'age', 'salary', 'department', 'position'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            st.error(f"В файле отсутствуют обязательные колонки: {missing}")
            return None

        # Преобразование дат
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass

        return df
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {str(e)}")
        return None


# Загрузка данных
st.sidebar.header("Загрузка данных")
data_source = st.sidebar.radio(
    "Источник данных",
    ["Демо-данные", "Загрузить CSV"],
    index=0
)

df = None

if data_source == "Демо-данные":
    df = load_sample_data()
    st.sidebar.success("Загружены демонстрационные данные (500 сотрудников)")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Выберите CSV файл с данными сотрудников",
        type=["csv"],
        help="Файл должен содержать как минимум колонки: age, salary, department, position"
    )
    if uploaded_file is not None:
        df = load_uploaded_data(uploaded_file)

if df is None:
    st.warning("Пожалуйста, загрузите данные для анализа")
    st.stop()

# Основной интерфейс
tab1, tab2, tab3, tab4 = st.tabs(["Обзор данных", "Визуализация", "Стат. анализ", "ML Анализ"])

with tab1:
    st.header("Обзор данных")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Основная информация")
        st.write(f"Всего сотрудников: {len(df)}")
        st.write(f"Отделов: {df['department'].nunique()}")
        st.write(f"Средняя зарплата: ${df['salary'].mean():,.2f}")
        st.write(f"Уровень текучести: {df['attrition'].value_counts(normalize=True)['Yes']:.1%}")

    st.subheader("Просмотр данных")
    rows_to_show = st.slider("Количество строк для отображения", 5, 100, 10)
    st.dataframe(df.head(rows_to_show), use_container_width=True)

    st.subheader("Распределение по отделам")
    dept_dist = df['department'].value_counts().reset_index()
    dept_dist.columns = ['Department', 'Count']
    fig = px.bar(dept_dist, x='Department', y='Count', color='Department')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Визуализация данных")
    plot_type = st.selectbox(
        "Тип визуализации",
        ["Гистограмма", "Боксплот", "Scatter Plot", "Корреляция", "Тепловая карта"]
    )
    
    if plot_type == "Гистограмма":
        col1, col2 = st.columns(2)
        with col1:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            feature = st.selectbox("Выберите признак", numeric_cols)
        with col2:
            hue = st.selectbox("Группировать по", ['None'] + df.select_dtypes(exclude=np.number).columns.tolist())
        fig = px.histogram(
            df,
            x=feature,
            color=None if hue == 'None' else hue,
            nbins=20,
            marginal="box",
            title=f"Распределение {feature}"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Боксплот":
        col1, col2 = st.columns(2)
        with col1:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            feature = st.selectbox("Выберите признак", numeric_cols)
        with col2:
            category = st.selectbox("Группировать по", df.select_dtypes(exclude=np.number).columns.tolist())
        fig = px.box(
            df,
            x=category,
            y=feature,
            color=category,
            title=f"Распределение {feature} по {category}"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Scatter Plot":
        col1, col2, col3 = st.columns(3)
        with col1:
            x_feature = st.selectbox("X ось", df.select_dtypes(include=np.number).columns.tolist())
        with col2:
            y_feature = st.selectbox("Y ось", df.select_dtypes(include=np.number).columns.tolist())
        with col3:
            hue = st.selectbox("Цвет", ['None'] + df.select_dtypes(exclude=np.number).columns.tolist())
        fig = px.scatter(
            df,
            x=x_feature,
            y=y_feature,
            color=None if hue == 'None' else hue,
            hover_data=df.columns.tolist(),
            title=f"{x_feature} vs {y_feature}"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Корреляция":
        numeric_df = df.select_dtypes(include=np.number)
        if len(numeric_df.columns) < 2:
            st.warning("Недостаточно числовых признаков для корреляционного анализа")
        else:
            corr_matrix = numeric_df.corr()
            fig = go.Figure(
                data=go.Heatmap(
                    z=corr_matrix,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1
                )
            )
            fig.update_layout(title="Корреляционная матрица")
            st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Тепловая карта":
        # Выбор колонок для тепловой карты
        pivot_col1 = st.selectbox("Колонка для строк", df.select_dtypes(exclude=np.number).columns.tolist())
        pivot_col2 = st.selectbox("Колонка для столбцов", df.select_dtypes(exclude=np.number).columns.tolist())
        value_col = st.selectbox("Значение", df.select_dtypes(include=np.number).columns.tolist())
        try:
            # Установка значений по умолчанию
            default_columns = df.select_dtypes(exclude=np.number).columns.tolist()
            default_numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
            pivot_col1 = default_columns[0] if default_columns else None
            pivot_col2 = default_columns[1] if len(default_columns) > 1 else None
            value_col = default_numeric_columns[0] if default_numeric_columns else None
            # Преобразование данных, если они содержат списки
            df[pivot_col1] = df[pivot_col1].apply(lambda x: str(x) if isinstance(x, list) else x)
            df[pivot_col2] = df[pivot_col2].apply(lambda x: str(x) if isinstance(x, list) else x)
            # Удаление пропущенных значений
            df = df.dropna(subset=[pivot_col1, pivot_col2, value_col])
            # Создание сводной таблицы
            pivot_df = df.pivot_table(
                index=pivot_col1,
                columns=pivot_col2,
                values=value_col,
                aggfunc='mean'
            ).fillna(0)
            # Визуализация тепловой карты
            fig = px.imshow(
                pivot_df,
                labels=dict(x=pivot_col2, y=pivot_col1, color=value_col),
                title=f"Среднее {value_col} по {pivot_col1} и {pivot_col2}"
            )
            # Отображение только одного графика с уникальным ключом
            st.plotly_chart(fig, use_container_width=True, key="heatmap_plot")
        except ValueError as e:
            st.error(f"Ошибка при создании сводной таблицы: {e}")

with tab3:
    st.header("Статистический анализ")

    analysis_type = st.selectbox(
        "Тип анализа",
        ["Описательная статистика", "t-тест", "ANOVA", "Анализ текучести"]
    )

    if analysis_type == "Описательная статистика":
        st.subheader("Общая статистика")
        st.dataframe(df.describe(include='all').T, use_container_width=True)

        st.subheader("Статистика по отделам")
        department = st.selectbox("Выберите отдел", ['All'] + df['department'].unique().tolist())
        if department == 'All':
            st.dataframe(df.groupby('department').describe().T, use_container_width=True)
        else:
            st.dataframe(df[df['department'] == department].describe().T, use_container_width=True)

    elif analysis_type == "t-тест":
        st.subheader("Парный t-тест")

        col1, col2, col3 = st.columns(3)

        with col1:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            feature = st.selectbox("Признак для теста", numeric_cols)

        with col2:
            category = st.selectbox("Категориальный признак", df.select_dtypes(exclude=np.number).columns.tolist())

        with col3:
            unique_values = df[category].unique()
            if len(unique_values) != 2:
                st.warning("Выберите бинарную категорию")
                st.stop()
            group1, group2 = st.select_slider(
                "Выберите группы для сравнения",
                options=unique_values,
                value=(unique_values[0], unique_values[1])
            )

        group1_data = df[df[category] == group1][feature]
        group2_data = df[df[category] == group2][feature]

        t_stat, p_val = stats.ttest_ind(group1_data, group2_data, equal_var=False)

        st.metric(f"Среднее для {group1}", f"{group1_data.mean():.2f}")
        st.metric(f"Среднее для {group2}", f"{group2_data.mean():.2f}")
        st.metric("t-статистика", f"{t_stat:.4f}")
        st.metric("p-значение", f"{p_val:.4f}")

        if p_val < 0.05:
            st.success("Различия статистически значимы (p < 0.05)")
        else:
            st.error("Различия не статистически значимы (p >= 0.05)")

    elif analysis_type == "ANOVA":
        st.subheader("Однофакторный дисперсионный анализ")

        col1, col2 = st.columns(2)

        with col1:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            feature = st.selectbox("Признак для анализа", numeric_cols)

        with col2:
            category = st.selectbox("Категория для группировки",
                                    df.select_dtypes(exclude=np.number).columns.tolist())

        groups = df[category].unique()
        if len(groups) < 2:
            st.warning("Недостаточно групп для ANOVA")
            st.stop()

        samples = [df[df[category] == group][feature] for group in groups]
        f_stat, p_val = stats.f_oneway(*samples)

        st.metric("F-статистика", f"{f_stat:.4f}")
        st.metric("p-значение", f"{p_val:.4f}")

        if p_val < 0.05:
            st.success("Есть статистически значимые различия между группами (p < 0.05)")

            # Post-hoc тест Тьюки
            from statsmodels.stats.multicomp import pairwise_tukeyhsd

            tukey = pairwise_tukeyhsd(
                endog=df[feature],
                groups=df[category],
                alpha=0.05
            )

            st.subheader("Post-hoc анализ (Тьюки)")
            st.text(str(tukey))
        else:
            st.error("Нет статистически значимых различий между группами (p >= 0.05)")

    elif analysis_type == "Анализ текучести":
        st.subheader("Анализ факторов текучести кадров")

        attrition_rate = df['attrition'].value_counts(normalize=True)['Yes']
        st.metric("Общий уровень текучести", f"{attrition_rate:.1%}")

        st.subheader("Текучесть по категориям")
        category = st.selectbox("Категория",
                                df.select_dtypes(exclude=np.number).columns.tolist())

        attrition_by_category = df.groupby(category)['attrition'].apply(
            lambda x: (x == 'Yes').mean()
        ).sort_values(ascending=False)

        fig = px.bar(
            attrition_by_category.reset_index(),
            x=category,
            y='attrition',
            color=category,
            labels={'attrition': 'Уровень текучести'},
            title=f"Уровень текучести по {category}"
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        # Логистическая регрессия для анализа факторов
        if st.checkbox("Показать анализ значимости факторов"):
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import LabelEncoder

            # Подготовка данных
            X = df.drop(columns=['attrition'])
            y = df['attrition'].map({'Yes': 1, 'No': 0})

            # Кодирование категориальных переменных
            cat_cols = X.select_dtypes(exclude=np.number).columns
            le = LabelEncoder()
            for col in cat_cols:
                X[col] = le.fit_transform(X[col])

            # Обучение модели
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)

            # Важность признаков
            importance = pd.DataFrame({
                'feature': X.columns,
                'coefficient': model.coef_[0],
                'abs_coefficient': np.abs(model.coef_[0])
            }).sort_values('abs_coefficient', ascending=False)

            fig = px.bar(
                importance.head(10),
                x='coefficient',
                y='feature',
                orientation='h',
                title="Важность признаков для прогнозирования текучести"
            )
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Машинное обучение и продвинутая аналитика")

    ml_type = st.selectbox(
        "Тип анализа",
        ["Кластеризация", "PCA", "Прогнозирование текучести"]
    )

    if ml_type == "Кластеризация":
        st.subheader("Кластерный анализ методом k-средних")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        features = st.multiselect(
            "Выберите признаки для кластеризации",
            numeric_cols,
            default=['salary', 'performance_score', 'satisfaction_score']
        )

        if len(features) < 2:
            st.warning("Выберите хотя бы 2 признака")
            st.stop()

        n_clusters = st.slider("Количество кластеров", 2, 5, 3)

        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        df['cluster'] = clusters

        # Визуализация
        if len(features) >= 3:
            fig = px.scatter_3d(
                df,
                x=features[0],
                y=features[1],
                z=features[2],
                color='cluster',
                title=f"Кластеризация по {', '.join(features[:3])}",
                hover_data=df.columns.tolist()
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.scatter(
                df,
                x=features[0],
                y=features[1],
                color='cluster',
                title=f"Кластеризация по {features[0]} и {features[1]}",
                hover_data=df.columns.tolist()
            )
            st.plotly_chart(fig, use_container_width=True)

        # Оценка качества кластеризации
        silhouette = silhouette_score(X_scaled, clusters)
        st.metric("Silhouette Score", f"{silhouette:.3f}")

        # Характеристики кластеров
        st.subheader("Средние значения по кластерам")
        cluster_stats = df.groupby('cluster')[features].mean()
        st.dataframe(cluster_stats.style.background_gradient(cmap='Blues'), use_container_width=True)

    elif ml_type == "PCA":
        st.subheader("Анализ главных компонент (PCA)")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        features = st.multiselect(
            "Выберите признаки для PCA",
            numeric_cols,
            default=['age', 'salary', 'experience', 'performance_score', 'satisfaction_score']
        )

        if len(features) < 2:
            st.warning("Выберите хотя бы 2 признака")
            st.stop()

        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X_scaled)

        df_pca = pd.DataFrame(
            data=principal_components,
            columns=['PC1', 'PC2']
        )

        # Добавляем категориальную переменную для цвета
        hue = st.selectbox("Цвет по категории", ['None'] + df.select_dtypes(exclude=np.number).columns.tolist())

        fig = px.scatter(
            df_pca,
            x='PC1',
            y='PC2',
            color=None if hue == 'None' else df[hue],
            title="PCA Результат",
            hover_data=df.columns.tolist()
        )
        st.plotly_chart(fig, use_container_width=True)

        # Объясненная дисперсия
        explained_var = pca.explained_variance_ratio_
        st.metric("Объясненная дисперсия (PC1)", f"{explained_var[0]:.1%}")
        st.metric("Объясненная дисперсия (PC2)", f"{explained_var[1]:.1%}")

        # Loadings
        st.subheader("Нагрузки (Loadings)")
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=['PC1', 'PC2'],
            index=features
        )
        st.dataframe(loadings.style.background_gradient(cmap='RdBu', axis=None, vmin=-1, vmax=1),
                     use_container_width=True)

    elif ml_type == "Прогнозирование текучести":
        st.subheader("Прогнозирование текучести кадров")

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix

        # Подготовка данных
        X = df.drop(columns=['attrition'])
        y = df['attrition'].map({'Yes': 1, 'No': 0})

        # Кодирование категориальных переменных
        cat_cols = X.select_dtypes(exclude=np.number).columns
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Обучение модели
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Оценка модели
        y_pred = model.predict(X_test)

        st.subheader("Метрики модели")
        st.text(classification_report(y_test, y_pred))

        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(
            cm,
            labels=dict(x="Предсказанный", y="Истинный", color="Количество"),
            x=['No', 'Yes'],
            y=['No', 'Yes'],
            text_auto=True,
            title="Матрица ошибок"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Важность признаков
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        fig = px.bar(
            importance.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title="Топ-10 важных признаков"
        )
        st.plotly_chart(fig, use_container_width=True)

# Футер
st.sidebar.markdown("---")
st.sidebar.markdown("""
**HR аналитическая система**  
Версия 1.0  
© 2023 приложение Джолдошовой М.
""")

# Экспорт данных
if st.sidebar.button("Экспорт данных в CSV"):
    filename = f"data/employee_data_export.csv"
    df.to_csv(filename, index=False)
    st.sidebar.success(f"Данные экспортированы в {filename}")