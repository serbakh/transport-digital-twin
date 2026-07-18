"""
Навчання та оцінювання моделей машинного навчання
на синтетичних даних цифрового двійника
транспортно-експедиторського процесу.

Основні задачі:
1. Регресія — прогнозування добового тоннажу.
2. Класифікація — прогнозування потенційного вузького місця.
3. Перевірка узагальнення моделей на нові виробничі конфігурації.
4. Оцінювання впливу обсягу синтетичних даних на якість моделей.
5. Аналіз важливості вхідних ознак.
6. Збереження навчених моделей для модуля рекомендацій.
"""

from pathlib import Path

import joblib
import matplotlib

# Неінтерактивний backend дозволяє зберігати графіки
# у середовищах без графічного інтерфейсу.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

# -----------------------------------------------------------------------------
# 1. Шляхи до вхідних і вихідних даних
# -----------------------------------------------------------------------------

DATA_PATH = Path("data/synthetic/scenario_summary.csv")
OUTPUT_DIR = Path("data/ml")
MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 2. Основні параметри навчання
# -----------------------------------------------------------------------------

TEST_SIZE = 0.25
RANDOM_STATE = 42
N_ESTIMATORS = 300


# -----------------------------------------------------------------------------
# 3. Завантаження та перевірка синтетичних даних
# -----------------------------------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Файл із синтетичними даними не знайдено: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

if df.empty:
    raise ValueError("Файл scenario_summary.csv не містить записів.")

features = [
    "number_of_trucks",
    "number_of_loaders",
    "unload_points",
    "external_scale_load",
    "road_delay_percent",
]

required_columns = features + [
    "total_tons",
    "bottleneck",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "У наборі даних відсутні обов'язкові стовпці: "
        + ", ".join(missing_columns)
    )

missing_values = int(df[required_columns].isna().sum().sum())

if missing_values > 0:
    raise ValueError(
        f"У навчальному наборі виявлено пропуски: {missing_values}"
    )

print("=" * 68)
print("Підготовка синтетичних даних для машинного навчання")
print("=" * 68)
print(f"Кількість виробничих сценаріїв : {len(df)}")
print(f"Кількість вхідних ознак        : {len(features)}")
print("Вхідні ознаки:")

for feature in features:
    print(f"  - {feature}")

print("\nРозподіл класів вузького місця:")

bottleneck_distribution = (
    df["bottleneck"]
    .value_counts()
    .sort_index()
)

for class_name, count in bottleneck_distribution.items():
    print(f"  {class_name}: {count} сценаріїв")


# -----------------------------------------------------------------------------
# 4. Формування матриці вхідних ознак
# -----------------------------------------------------------------------------

X = df[features]


# -----------------------------------------------------------------------------
# 5. Регресійна модель: прогнозування добового тоннажу
# -----------------------------------------------------------------------------

y_regression = df["total_tons"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X,
    y_regression,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

reg_model = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

reg_model.fit(X_train_reg, y_train_reg)

y_pred_reg = reg_model.predict(X_test_reg)

regression_mae = mean_absolute_error(y_test_reg, y_pred_reg)
regression_r2 = r2_score(y_test_reg, y_pred_reg)

print("\n" + "=" * 68)
print("1. Регресія: прогнозування добового тоннажу")
print("=" * 68)
print(f"Навчальна вибірка : {len(X_train_reg)} сценаріїв")
print(f"Тестова вибірка   : {len(X_test_reg)} сценаріїв")
print(f"MAE               : {regression_mae:.2f} т")
print(f"R²                : {regression_r2:.3f}")

# Допоміжна перевірка: лінійна регресія як базова модель,
# що обґрунтовує вибір Random Forest (нелінійність залежності).
baseline_model = LinearRegression()
baseline_model.fit(X_train_reg, y_train_reg)
baseline_pred = baseline_model.predict(X_test_reg)
baseline_r2 = r2_score(y_test_reg, baseline_pred)

print(f"R² (Linear Regression, довідково) : {baseline_r2:.3f}")

# -----------------------------------------------------------------------------
# 5.1. Допоміжна перевірка: стійкість оцінок Random Forest,
# за допомогою 5-fold крос-валідації (замість єдиного розбиття).
# -----------------------------------------------------------------------------

cv_scheme = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_r2_scores = cross_val_score(
    RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
    ),
    X, y_regression, cv=cv_scheme, scoring="r2",
)
cv_mae_scores = -cross_val_score(
    RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
    ),
    X, y_regression, cv=cv_scheme, scoring="neg_mean_absolute_error",
)

print(
    f"5-fold CV: R² = {cv_r2_scores.mean():.3f} ± {cv_r2_scores.std():.3f}, "
    f"MAE = {cv_mae_scores.mean():.2f} ± {cv_mae_scores.std():.2f} т"
)


# -----------------------------------------------------------------------------
# 6. Класифікація: прогнозування вузького місця
# -----------------------------------------------------------------------------

y_bottleneck = df["bottleneck"]

(
    X_train_bottleneck,
    X_test_bottleneck,
    y_train_bottleneck,
    y_test_bottleneck,
) = train_test_split(
    X,
    y_bottleneck,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_bottleneck,
)

bottleneck_model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1,
)

bottleneck_model.fit(X_train_bottleneck, y_train_bottleneck)

y_pred_bottleneck = bottleneck_model.predict(X_test_bottleneck)

bottleneck_accuracy = accuracy_score(
    y_test_bottleneck,
    y_pred_bottleneck,
)

bottleneck_f1 = f1_score(
    y_test_bottleneck,
    y_pred_bottleneck,
    average="weighted",
    zero_division=0,
)

print("\n" + "=" * 68)
print("2. Класифікація: прогнозування вузького місця")
print("=" * 68)
print(f"Accuracy          : {bottleneck_accuracy:.3f}")
print(f"F1-score weighted : {bottleneck_f1:.3f}")
print("\nЗвіт класифікації:")
print(
    classification_report(
        y_test_bottleneck,
        y_pred_bottleneck,
        zero_division=0,
    )
)

# -----------------------------------------------------------------------------
# 7. Допоміжна перевірка: стійкість оцінок класифікатора, за допомогою стратифікованої 5-fold крос-валідації
# -----------------------------------------------------------------------------

cv_scheme_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_acc_scores = cross_val_score(
    RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
                            class_weight="balanced", n_jobs=-1),
    X, y_bottleneck, cv=cv_scheme_clf, scoring="accuracy",
)
print(f"5-fold CV: Accuracy = {cv_acc_scores.mean():.3f} ± {cv_acc_scores.std():.3f}")

# -----------------------------------------------------------------------------
# 8. Перевірка узагальнення за кількістю самоскидів
# -----------------------------------------------------------------------------

truck_train_df = df[df["number_of_trucks"] <= 16].copy()
truck_test_df = df[df["number_of_trucks"] >= 17].copy()

X_train_trucks = truck_train_df[features]
X_test_trucks = truck_test_df[features]

truck_reg_model = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

truck_reg_model.fit(
    X_train_trucks,
    truck_train_df["total_tons"],
)

truck_pred_tons = truck_reg_model.predict(X_test_trucks)

truck_mae = mean_absolute_error(
    truck_test_df["total_tons"],
    truck_pred_tons,
)

truck_r2 = r2_score(
    truck_test_df["total_tons"],
    truck_pred_tons,
)

print("\n" + "=" * 68)
print("3. Перевірка узагальнення: кількість самоскидів")
print("=" * 68)
print("Навчання : 4–16 самоскидів")
print("Перевірка: 17–20 самоскидів")
print(f"MAE      : {truck_mae:.2f} т")
print(f"R²       : {truck_r2:.3f}")

truck_bottleneck_model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1,
)

truck_bottleneck_model.fit(
    X_train_trucks,
    truck_train_df["bottleneck"],
)

truck_pred_bottleneck = truck_bottleneck_model.predict(X_test_trucks)

truck_bottleneck_accuracy = accuracy_score(
    truck_test_df["bottleneck"],
    truck_pred_bottleneck,
)

truck_bottleneck_f1 = f1_score(
    truck_test_df["bottleneck"],
    truck_pred_bottleneck,
    average="weighted",
    zero_division=0,
)

print("\nВузьке місце:")
print(f"Accuracy          : {truck_bottleneck_accuracy:.3f}")
print(f"F1-score weighted : {truck_bottleneck_f1:.3f}")
print(
    classification_report(
        truck_test_df["bottleneck"],
        truck_pred_bottleneck,
        zero_division=0,
    )
)


# -----------------------------------------------------------------------------
# 9. Перевірка узагальнення за кількістю точок розвантаження
# -----------------------------------------------------------------------------

unload_train_df = df[df["unload_points"] <= 2].copy()
unload_test_df = df[df["unload_points"] == 3].copy()

X_train_unload = unload_train_df[features]
X_test_unload = unload_test_df[features]

unload_reg_model = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

unload_reg_model.fit(
    X_train_unload,
    unload_train_df["total_tons"],
)

unload_pred_tons = unload_reg_model.predict(X_test_unload)

unload_mae = mean_absolute_error(
    unload_test_df["total_tons"],
    unload_pred_tons,
)

unload_r2 = r2_score(
    unload_test_df["total_tons"],
    unload_pred_tons,
)

print("\n" + "=" * 68)
print("4. Перевірка узагальнення: точки розвантаження")
print("=" * 68)
print("Навчання : 1–2 точки розвантаження")
print("Перевірка: 3 точки розвантаження")
print(f"MAE      : {unload_mae:.2f} т")
print(f"R²       : {unload_r2:.3f}")

unload_bottleneck_model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1,
)

unload_bottleneck_model.fit(
    X_train_unload,
    unload_train_df["bottleneck"],
)

unload_pred_bottleneck = unload_bottleneck_model.predict(X_test_unload)

unload_bottleneck_accuracy = accuracy_score(
    unload_test_df["bottleneck"],
    unload_pred_bottleneck,
)

unload_bottleneck_f1 = f1_score(
    unload_test_df["bottleneck"],
    unload_pred_bottleneck,
    average="weighted",
    zero_division=0,
)

print("\nВузьке місце:")
print(f"Accuracy          : {unload_bottleneck_accuracy:.3f}")
print(f"F1-score weighted : {unload_bottleneck_f1:.3f}")
print(
    classification_report(
        unload_test_df["bottleneck"],
        unload_pred_bottleneck,
        zero_division=0,
    )
)


# -----------------------------------------------------------------------------
# 10. Вплив обсягу синтетичних даних на якість моделей
# -----------------------------------------------------------------------------

sample_sizes = [50, 100, 250, 500, 750, 1000]

learning_train_df, learning_test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["bottleneck"],
)

X_learning_test = learning_test_df[features]
y_learning_test_reg = learning_test_df["total_tons"]
y_learning_test_bottleneck = learning_test_df["bottleneck"]

learning_results = []

for sample_size in sample_sizes:
    if sample_size > len(learning_train_df):
        continue

    sample_df = learning_train_df.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )

    X_sample = sample_df[features]

    sample_reg_model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    sample_reg_model.fit(X_sample, sample_df["total_tons"])
    sample_pred_reg = sample_reg_model.predict(X_learning_test)

    sample_mae = mean_absolute_error(
        y_learning_test_reg,
        sample_pred_reg,
    )

    sample_r2 = r2_score(
        y_learning_test_reg,
        sample_pred_reg,
    )

    sample_bottleneck_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    sample_bottleneck_model.fit(
        X_sample,
        sample_df["bottleneck"],
    )

    sample_pred_bottleneck = sample_bottleneck_model.predict(
        X_learning_test
    )

    sample_bottleneck_accuracy = accuracy_score(
        y_learning_test_bottleneck,
        sample_pred_bottleneck,
    )

    sample_bottleneck_f1 = f1_score(
        y_learning_test_bottleneck,
        sample_pred_bottleneck,
        average="weighted",
        zero_division=0,
    )

    learning_results.append({
        "sample_size": sample_size,
        "mae_total_tons": round(sample_mae, 2),
        "r2_total_tons": round(sample_r2, 3),
        "accuracy_bottleneck": round(
            sample_bottleneck_accuracy,
            3,
        ),
        "f1_bottleneck": round(
            sample_bottleneck_f1,
            3,
        ),
    })

learning_df = pd.DataFrame(learning_results)

print("\n" + "=" * 68)
print("5. Вплив обсягу синтетичних даних")
print("=" * 68)
print(learning_df.to_string(index=False))

learning_curve_path = OUTPUT_DIR / "ml_learning_curve.csv"

learning_df.to_csv(
    learning_curve_path,
    index=False,
    encoding="utf-8-sig",
)

# -----------------------------------------------------------------------------
# 11. Аналіз важливості вхідних ознак
# -----------------------------------------------------------------------------

feature_importance_df = pd.DataFrame({
    "feature": features,
    "importance_total_tons": reg_model.feature_importances_,
    "importance_bottleneck": bottleneck_model.feature_importances_,
})

feature_importance_df = (
    feature_importance_df
    .sort_values(
        by="importance_total_tons",
        ascending=False,
    )
    .reset_index(drop=True)
)

print("\n" + "=" * 68)
print("6. Важливість вхідних ознак")
print("=" * 68)
print(feature_importance_df.to_string(index=False))

feature_importance_path = OUTPUT_DIR / "ml_feature_importance.csv"

feature_importance_df.to_csv(
    feature_importance_path,
    index=False,
    encoding="utf-8-sig",
)

# -----------------------------------------------------------------------------
# 12. Формування фінальних моделей на повному наборі даних
#
# Метрики якості вище розраховано на окремій тестовій вибірці.
# Після завершення оцінювання фінальні моделі повторно навчаються
# на всіх доступних синтетичних сценаріях, щоб використати повний
# обсяг даних у модулі підтримки управлінських рішень.
# -----------------------------------------------------------------------------

final_reg_model = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

final_reg_model.fit(
    X,
    y_regression,
)

final_bottleneck_model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1,
)

final_bottleneck_model.fit(
    X,
    y_bottleneck,
)


# -----------------------------------------------------------------------------
# 13. Збереження фінальних моделей і переліку ознак
# -----------------------------------------------------------------------------

reg_model_path = MODEL_DIR / "total_tons_model.joblib"
bottleneck_model_path = MODEL_DIR / "bottleneck_model.joblib"
features_path = MODEL_DIR / "model_features_names.joblib"

joblib.dump(
    final_reg_model,
    reg_model_path,
)

joblib.dump(
    final_bottleneck_model,
    bottleneck_model_path,
)

joblib.dump(
    features,
    features_path,
)

# -----------------------------------------------------------------------------
# 14. Збереження узагальнених показників якості моделей
# -----------------------------------------------------------------------------

model_metrics_df = pd.DataFrame([
    {
        "model": "total_tons_regression",
        "mae": round(regression_mae, 2),
        "r2": round(regression_r2, 3),
        "accuracy": None,
        "f1": None,
    },
    {
        "model": "bottleneck_classification",
        "mae": None,
        "r2": None,
        "accuracy": round(bottleneck_accuracy, 3),
        "f1": round(bottleneck_f1, 3),
    },
])

model_metrics_path = OUTPUT_DIR / "ml_model_metrics.csv"

model_metrics_df.to_csv(
    model_metrics_path,
    index=False,
    encoding="utf-8-sig",
)


# -----------------------------------------------------------------------------
# 15. Контрольний індикатор завершення
# -----------------------------------------------------------------------------

print("\n" + "=" * 68)
print("Навчання та оцінювання моделей успішно завершено")
print("=" * 68)
print("Збережено моделі:")
print(f"  - {reg_model_path}")
print(f"  - {bottleneck_model_path}")
print(f"  - {features_path}")
print("Збережено результати:")
print(f"  - {learning_curve_path}")
print(f"  - {feature_importance_path}")
print(f"  - {model_metrics_path}")
print("=" * 68)
