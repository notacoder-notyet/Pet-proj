# Predictive Maintenance Classification

## Описание проекта

Задача — **бинарная классификация отказов оборудования** по показаниям датчиков (температура воздуха и процесса, скорость вращения, крутящий момент, износ инструмента) и типу продукта.

Датасет: [Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) (AI4I 2020, Kaggle). Целевая колонка — `Target` (`0` — нет отказа, `1` — отказ). Класс отказа редкий (~3.4%), поэтому основная метрика — **F1-score**, а не Accuracy.

## Структура проекта


ml-predictive-maintenance/
├── data/
│   ├── raw/          # исходный датасет
│   └── processed/    # обработанные данные (если есть)
├── notebooks/
│   └── 01_eda_and_modeling.ipynb   # основной ноутбук с EDA и обучением
├── src/
│   └── predict.py    # скрипт для инференса
├── models/
│   └── best_model.pkl  # сохранённая модель
├── requirements.txt
└── README.md
```

## Результаты

Сравнение моделей на тестовой выборке (20%, `random_state=42`):

| Модель | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9675 | 0.6364 | 0.1029 | 0.1772 |
| Random Forest | 0.9780 | 0.7069 | 0.6029 | 0.6508 |
| XGBoost | 0.9750 | 0.6047 | 0.7647 | 0.6753 |
| **XGBoost Tuned** | **0.9760** | **0.6190** | **0.7647** | **0.6842** |

Гиперпараметры XGBoost Tuned подобраны через `GridSearchCV` (`scoring='f1'`, `cv=3`).

## Выводы

Для продакшена выбрана **XGBoost Tuned**: у неё лучший **F1-score** (0.6842) и разумный баланс Precision/Recall. Logistic Regression почти не находит отказы (Recall 0.10). Random Forest точнее по Precision, но пропускает больше отказов. Базовый XGBoost близко к тюнингу, сетка даёт небольшой прирост F1.

В промышленных задачах обслуживания **важнее Recall**: лучше поймать все отказы даже ценой ложных тревог, чем пропустить реальную поломку. Accuracy здесь обманчива из-за дисбаланса классов.

## Как запустить проект

### 1. Клонировать репозиторий

```bash
git clone https://github.com/notacoder-notyet/Pet-proj.git
cd Pet-proj
```

### 2. Создать окружение и поставить зависимости

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Данные

CSV в git не хранится. Скачайте датасет с Kaggle и положите файл сюда:

```text
data/raw/predictive_maintenance.csv
```

### 4. Обучение и EDA

```bash
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

Выполните ячейки сверху вниз. Последний блок сохранит модель в `models/best_model.pkl`.

### 5. Инференс

```bash
python src/predict.py
```

Скрипт загружает `models/best_model.pkl` и печатает класс (`0`/`1`) и вероятность отказа на демо-примере.
