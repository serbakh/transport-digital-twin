import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, classification_report


DATA_PATH = "data/synthetic/scenario_summary.csv"
DAILY_TARGET_TONS = 3500


df = pd.read_csv(DATA_PATH)

# Целевая переменная для классификации
df["plan_completed"] = (df["total_tons"] >= DAILY_TARGET_TONS).astype(int)

# ------------------------------
# Визначення вузького місця
# ------------------------------

def detect_bottleneck(row):
    waits = {
        "tare": row["avg_wait_tare"],
        "loader": row["avg_wait_loader"],
        "gross": row["avg_wait_gross"],
        "unload": row["avg_wait_unload"],
    }

    return max(waits, key=waits.get)


df["bottleneck"] = df.apply(
    detect_bottleneck,
    axis=1
)

# Входные признаки
features = [
    "number_of_trucks",
    "number_of_loaders",
    "unload_points",
    "external_scale_load",
]

X = df[features]

# ------------------------------
# 1. Регрессия: прогноз тоннажа
# ------------------------------

y_reg = df["total_tons"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_reg,
    test_size=0.25,
    random_state=42
)

reg_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

reg_model.fit(X_train, y_train)

y_pred = reg_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Регресія: прогноз total_tons ===")
print(f"MAE: {mae:.2f} т")
print(f"R²: {r2:.3f}")


# ------------------------------
# 2. Классификация: выполнение плана
# ------------------------------

y_clf = df["plan_completed"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_clf,
    test_size=0.25,
    random_state=42,
    stratify=y_clf
)

clf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

clf_model.fit(X_train, y_train)

y_pred = clf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== Класифікація: виконання плану 3500 т ===")
print(f"Accuracy: {accuracy:.3f}")
print(f"F1-score: {f1:.3f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ------------------------------
# 3. Класифікація вузького місця
# ------------------------------

y_bottleneck = df["bottleneck"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_bottleneck,
    test_size=0.25,
    random_state=42,
    stratify=y_bottleneck
)

bottleneck_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

bottleneck_model.fit(X_train, y_train)

y_pred = bottleneck_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n=== Класифікація вузького місця ===")
print(f"Accuracy: {accuracy:.3f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ------------------------------
# 4. Extrapolation test:
# навчання на 4–16 самоскидах,
# перевірка на 17–20 самоскидах
# ------------------------------

train_df = df[df["number_of_trucks"] <= 16]
test_df = df[df["number_of_trucks"] >= 17]

X_train = train_df[features]
X_test = test_df[features]

# Регресія total_tons
y_train_reg = train_df["total_tons"]
y_test_reg = test_df["total_tons"]

extra_reg_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

extra_reg_model.fit(X_train, y_train_reg)
y_pred_reg = extra_reg_model.predict(X_test)

extra_mae = mean_absolute_error(y_test_reg, y_pred_reg)
extra_r2 = r2_score(y_test_reg, y_pred_reg)

print("\n=== Extrapolation test: total_tons ===")
print("Train: trucks <= 16")
print("Test: trucks >= 17")
print(f"MAE: {extra_mae:.2f} т")
print(f"R²: {extra_r2:.3f}")


# Класифікація plan_completed
y_train_clf = train_df["plan_completed"]
y_test_clf = test_df["plan_completed"]

extra_clf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

extra_clf_model.fit(X_train, y_train_clf)
y_pred_clf = extra_clf_model.predict(X_test)

extra_accuracy = accuracy_score(y_test_clf, y_pred_clf)
extra_f1 = f1_score(y_test_clf, y_pred_clf)

print("\n=== Extrapolation test: plan_completed ===")
print(f"Accuracy: {extra_accuracy:.3f}")
print(f"F1-score: {extra_f1:.3f}")
print(classification_report(y_test_clf, y_pred_clf, zero_division=0))


# Класифікація bottleneck
y_train_bottleneck = train_df["bottleneck"]
y_test_bottleneck = test_df["bottleneck"]

extra_bottleneck_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

extra_bottleneck_model.fit(X_train, y_train_bottleneck)
y_pred_bottleneck = extra_bottleneck_model.predict(X_test)

extra_bottleneck_accuracy = accuracy_score(
    y_test_bottleneck,
    y_pred_bottleneck
)

print("\n=== Extrapolation test: bottleneck ===")
print(f"Accuracy: {extra_bottleneck_accuracy:.3f}")
print(classification_report(y_test_bottleneck, y_pred_bottleneck, zero_division=0))


# ------------------------------
# 5. Extrapolation test:
# train = 1-2 точки вигрузки
# test = 3 точки вигрузки
# ------------------------------

train_df = df[df["unload_points"] <= 2]
test_df = df[df["unload_points"] == 3]

X_train = train_df[features]
X_test = test_df[features]

# ------------------------------
# Регресія total_tons
# ------------------------------

y_train_reg = train_df["total_tons"]
y_test_reg = test_df["total_tons"]

extra_reg_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

extra_reg_model.fit(X_train, y_train_reg)

y_pred_reg = extra_reg_model.predict(X_test)

extra_mae = mean_absolute_error(
    y_test_reg,
    y_pred_reg
)

extra_r2 = r2_score(
    y_test_reg,
    y_pred_reg
)

print("\n=== Extrapolation test: unload_points ===")
print("Train: unload_points = 1,2")
print("Test : unload_points = 3")
print(f"MAE: {extra_mae:.2f} т")
print(f"R²: {extra_r2:.3f}")

# ------------------------------
# Виконання плану
# ------------------------------

y_train_clf = train_df["plan_completed"]
y_test_clf = test_df["plan_completed"]

extra_clf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

extra_clf_model.fit(
    X_train,
    y_train_clf
)

y_pred_clf = extra_clf_model.predict(X_test)

print("\n=== Extrapolation test: plan_completed ===")
print(
    classification_report(
        y_test_clf,
        y_pred_clf,
        zero_division=0
    )
)

# ------------------------------
# Вузьке місце
# ------------------------------

y_train_bottleneck = train_df["bottleneck"]
y_test_bottleneck = test_df["bottleneck"]

extra_bottleneck_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

extra_bottleneck_model.fit(
    X_train,
    y_train_bottleneck
)

y_pred_bottleneck = extra_bottleneck_model.predict(
    X_test
)

print("\n=== Extrapolation test: bottleneck ===")
print(
    classification_report(
        y_test_bottleneck,
        y_pred_bottleneck,
        zero_division=0
    )
)

# ------------------------------
# 5. Вплив обсягу синтетичних даних
# на якість ML-моделі
# ------------------------------

sample_sizes = [100, 250, 500, 750]

train_pool_df, test_df = train_test_split(
    df,
    test_size=0.25,
    random_state=42,
    stratify=df["plan_completed"]
)

X_test = test_df[features]

y_test_reg = test_df["total_tons"]
y_test_clf = test_df["plan_completed"]
y_test_bottleneck = test_df["bottleneck"]

learning_results = []

for size in sample_sizes:
    if size > len(train_pool_df):
        continue

    sample_df = train_pool_df.sample(
        n=size,
        random_state=42
    )

    X_sample = sample_df[features]

    y_sample_reg = sample_df["total_tons"]
    y_sample_clf = sample_df["plan_completed"]
    y_sample_bottleneck = sample_df["bottleneck"]

    # Регресія total_tons
    reg = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    reg.fit(X_sample, y_sample_reg)
    y_pred_reg = reg.predict(X_test)

    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)

    # Класифікація plan_completed
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    clf.fit(X_sample, y_sample_clf)
    y_pred_clf = clf.predict(X_test)

    acc = accuracy_score(y_test_clf, y_pred_clf)
    f1 = f1_score(y_test_clf, y_pred_clf)

    # Класифікація bottleneck
    bottleneck_clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    bottleneck_clf.fit(X_sample, y_sample_bottleneck)
    y_pred_bottleneck = bottleneck_clf.predict(X_test)

    bottleneck_acc = accuracy_score(
        y_test_bottleneck,
        y_pred_bottleneck
    )

    learning_results.append({
        "sample_size": size,
        "mae_total_tons": round(mae, 2),
        "r2_total_tons": round(r2, 3),
        "accuracy_plan_completed": round(acc, 3),
        "f1_plan_completed": round(f1, 3),
        "accuracy_bottleneck": round(bottleneck_acc, 3)
    })

learning_df = pd.DataFrame(learning_results)

print("\n=== Вплив обсягу синтетичних даних ===")
print(learning_df)

learning_df.to_csv(
    "data/synthetic/ml_learning_curve.csv",
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------
# Графік 5.1. MAE від обсягу даних
# ------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    learning_df["sample_size"],
    learning_df["mae_total_tons"],
    marker="o"
)

plt.title(
    "Вплив обсягу синтетичних даних на MAE"
)

plt.xlabel(
    "Кількість сценаріїв для навчання"
)

plt.ylabel(
    "MAE прогнозу тоннажу, т"
)

plt.grid(True)

plt.savefig(
    "data/synthetic/ml_learning_mae.png"
)

plt.close()


# ------------------------------
# Графік 5.2. Accuracy bottleneck
# ------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    learning_df["sample_size"],
    learning_df["accuracy_bottleneck"],
    marker="o"
)

plt.title(
    "Вплив обсягу синтетичних даних на Accuracy bottleneck"
)

plt.xlabel(
    "Кількість сценаріїв для навчання"
)

plt.ylabel(
    "Accuracy"
)

plt.grid(True)

plt.savefig(
    "data/synthetic/ml_learning_bottleneck.png"
)

plt.close()

print("\nГрафіки збережено:")
print("data/synthetic/ml_learning_mae.png")
print("data/synthetic/ml_learning_bottleneck.png")

print("\nФайл збережено:")
print("data/synthetic/ml_learning_curve.csv")

# ------------------------------
# 6. Важливість ознак
# ------------------------------

feature_importance = pd.DataFrame({
    "feature": features,
    "importance_total_tons": reg_model.feature_importances_,
    "importance_plan_completed": clf_model.feature_importances_,
    "importance_bottleneck": bottleneck_model.feature_importances_,
})

feature_importance = feature_importance.sort_values(
    by="importance_total_tons",
    ascending=False
)

print("\n=== Важливість ознак ===")
print(feature_importance)

feature_importance.to_csv(
    "data/synthetic/ml_feature_importance.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nФайл збережено:")
print("data/synthetic/ml_feature_importance.csv")