"""
Формування рекомендацій щодо мінімальної конфігурації ресурсів
на основі прогнозів моделей машинного навчання.

Модуль:
1. Завантажує навчені моделі прогнозування тоннажу та вузького місця.
2. Формує простір допустимих конфігурацій ресурсів.
3. Прогнозує добовий тоннаж для кожної конфігурації.
4. Перевіряє виконання договірних планів 3000 та 4000 т/добу.
5. Обирає мінімальну конфігурацію ресурсів із заданим запасом продуктивності.
"""

from itertools import product
from pathlib import Path

import joblib
import pandas as pd


# -----------------------------------------------------------------------------
# 1. Шляхи до моделей і результатів
# -----------------------------------------------------------------------------

REG_MODEL_PATH = Path("models/total_tons_model.joblib")
BOTTLENECK_MODEL_PATH = Path("models/bottleneck_model.joblib")
FEATURES_PATH = Path("models/model_features_names.joblib")

OUTPUT_DIR = Path("data/ml")
OUTPUT_PATH = OUTPUT_DIR / "resource_recommendations.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 2. Основні параметри модуля рекомендацій
# -----------------------------------------------------------------------------

# Запас продуктивності враховує прогнозну похибку ML-моделі
# та природну мінливість транспортно-експедиторського процесу.
# Конфігурація рекомендується лише тоді, коли прогнозований тоннаж
# перевищує планове значення щонайменше на 5%.
SAFETY_MARGIN_PERCENT = 5


# -----------------------------------------------------------------------------
# 3. Перевірка наявності навчених моделей
# -----------------------------------------------------------------------------

required_files = [
    REG_MODEL_PATH,
    BOTTLENECK_MODEL_PATH,
    FEATURES_PATH,
]

missing_files = [
    path
    for path in required_files
    if not path.exists()
]

if missing_files:
    raise FileNotFoundError(
        "Не знайдено файли навчених моделей: "
        + ", ".join(str(path) for path in missing_files)
        + ". Спочатку виконайте train_model.py."
    )


# -----------------------------------------------------------------------------
# 4. Завантаження моделей і переліку ознак
# -----------------------------------------------------------------------------

reg_model = joblib.load(REG_MODEL_PATH)
bottleneck_model = joblib.load(BOTTLENECK_MODEL_PATH)
features = joblib.load(FEATURES_PATH)


# -----------------------------------------------------------------------------
# 5. Простір конфігурацій ресурсів
# -----------------------------------------------------------------------------

TRUCK_VALUES = list(range(4, 21))
LOADER_VALUES = [1, 2, 3]
UNLOAD_POINT_VALUES = [1, 2, 3]
EXTERNAL_SCALE_LOAD_VALUES = [0, 5, 50]
ROAD_DELAY_VALUES = [0, 5, 15, 30]


# -----------------------------------------------------------------------------
# 6. Договірні виробничі плани
# -----------------------------------------------------------------------------

contract_scenarios = [
    {"ship_tons": 5000, "target_tons": 3000},
    {"ship_tons": 10000, "target_tons": 4000},
]


# -----------------------------------------------------------------------------
# 7. Формування всіх допустимих конфігурацій
# -----------------------------------------------------------------------------

configurations = []

for (
    number_of_trucks,
    number_of_loaders,
    unload_points,
    external_scale_load,
    road_delay_percent,
) in product(
    TRUCK_VALUES,
    LOADER_VALUES,
    UNLOAD_POINT_VALUES,
    EXTERNAL_SCALE_LOAD_VALUES,
    ROAD_DELAY_VALUES,
):
    configurations.append({
        "number_of_trucks": number_of_trucks,
        "number_of_loaders": number_of_loaders,
        "unload_points": unload_points,
        "external_scale_load": external_scale_load,
        "road_delay_percent": road_delay_percent,
    })

config_df = pd.DataFrame(configurations)

missing_feature_columns = [
    feature
    for feature in features
    if feature not in config_df.columns
]

if missing_feature_columns:
    raise ValueError(
        "У таблиці конфігурацій відсутні ознаки: "
        + ", ".join(missing_feature_columns)
    )


# -----------------------------------------------------------------------------
# 8. Прогнозування тоннажу та вузького місця
# -----------------------------------------------------------------------------

config_df["predicted_total_tons"] = reg_model.predict(
    config_df[features]
)

config_df["predicted_bottleneck"] = bottleneck_model.predict(
    config_df[features]
)


# -----------------------------------------------------------------------------
# 9. Пошук мінімальної конфігурації ресурсів
# -----------------------------------------------------------------------------

def find_min_resource_configuration(
    dataframe,
    target_tons,
    external_scale_load,
    road_delay_percent,
):
    """
    Обирає мінімальну конфігурацію ресурсів, яка за прогнозом
    забезпечує виконання заданого добового плану із встановленим
    запасом продуктивності.

    Зовнішнє навантаження вагової та дорожня затримка розглядаються
    як задані умови, а не як ресурси, що можуть бути змінені підприємством.

    Мінімізація виконується лексикографічно:
    1. мінімальна кількість самоскидів;
    2. мінімальна кількість мехлопат;
    3. мінімальна кількість точок розвантаження;
    4. найбільший прогнозований тоннаж серед однакових конфігурацій.
    """

    required_tons_with_margin = (
        target_tons
        * (1 + SAFETY_MARGIN_PERCENT / 100)
    )

    candidates = dataframe[
        (dataframe["external_scale_load"] == external_scale_load)
        & (dataframe["road_delay_percent"] == road_delay_percent)
        & (
            dataframe["predicted_total_tons"]
            >= required_tons_with_margin
        )
    ].copy()

    if candidates.empty:
        return None

    candidates = candidates.sort_values(
        by=[
            "number_of_trucks",
            "number_of_loaders",
            "unload_points",
            "predicted_total_tons",
        ],
        ascending=[True, True, True, False],
    )

    return candidates.iloc[0]


# -----------------------------------------------------------------------------
# 10. Виведення рекомендації
# -----------------------------------------------------------------------------

def print_recommendation(
    ship_tons,
    target_tons,
    external_scale_load,
    road_delay_percent,
    recommendation,
):
    """Виводить рекомендовану конфігурацію та прогнозований результат."""

    print("\n" + "=" * 68)
    print(f"Обсяг суднової партії         : {ship_tons} т")
    print(f"Планова норма                 : {target_tons} т/добу")
    print(f"Необхідний запас              : {SAFETY_MARGIN_PERCENT}%")
    print(f"Зовнішнє навантаження вагової : {external_scale_load}%")
    print(f"Дорожня затримка              : до {road_delay_percent}%")
    print("-" * 68)

    if recommendation is None:
        print(
            "Рекомендація: у заданому просторі конфігурацію "
            "із необхідним запасом не знайдено."
        )
        return

    predicted_tons = float(recommendation["predicted_total_tons"])
    target_completion = (
        predicted_tons / target_tons * 100
        if target_tons > 0
        else None
    )
    reserve_tons = predicted_tons - target_tons
    reserve_percent = (
        reserve_tons / target_tons * 100
        if target_tons > 0
        else None
    )
    estimated_days = (
        ship_tons / predicted_tons
        if predicted_tons > 0
        else None
    )

    print("Рекомендована мінімальна конфігурація:")
    print(f"  Самоскиди              : {int(recommendation['number_of_trucks'])}")
    print(f"  Мехлопати              : {int(recommendation['number_of_loaders'])}")
    print(f"  Точки розвантаження    : {int(recommendation['unload_points'])}")

    print("\nПрогнозований результат:")
    print(f"  Добовий тоннаж         : {predicted_tons:.1f} т")
    if target_completion is not None:
        print(f"  Виконання плану        : {target_completion:.1f}%")
    print(f"  Запас продуктивності   : {reserve_tons:.1f} т")
    if reserve_percent is not None:
        print(f"  Запас відносно плану   : {reserve_percent:.1f}%")
    if estimated_days is not None:
        print(f"  Орієнтовна тривалість  : {estimated_days:.2f} доби")
    else:
        print("  Орієнтовна тривалість  : не визначено")
    print(
        f"  Потенційне вузьке місце: "
        f"{recommendation['predicted_bottleneck']}"
    )


# -----------------------------------------------------------------------------
# 11. Формування рекомендацій для виробничих планів
# -----------------------------------------------------------------------------

recommendations = []

for contract_scenario in contract_scenarios:
    for external_scale_load in EXTERNAL_SCALE_LOAD_VALUES:
        for road_delay_percent in ROAD_DELAY_VALUES:

            recommendation = find_min_resource_configuration(
                config_df,
                contract_scenario["target_tons"],
                external_scale_load,
                road_delay_percent,
            )

            print_recommendation(
                contract_scenario["ship_tons"],
                contract_scenario["target_tons"],
                external_scale_load,
                road_delay_percent,
                recommendation,
            )

            required_tons_with_margin = (
                contract_scenario["target_tons"]
                * (1 + SAFETY_MARGIN_PERCENT / 100)
            )

            if recommendation is None:
                recommendations.append({
                    "ship_tons": contract_scenario["ship_tons"],
                    "target_tons": contract_scenario["target_tons"],
                    "safety_margin_percent": SAFETY_MARGIN_PERCENT,
                    "required_tons_with_margin": round(
                        required_tons_with_margin,
                        1,
                    ),
                    "external_scale_load": external_scale_load,
                    "road_delay_percent": road_delay_percent,
                    "configuration_found": False,
                })
                continue

            predicted_tons = float(
                recommendation["predicted_total_tons"]
            )

            target_completion = (
                predicted_tons
                / contract_scenario["target_tons"]
                * 100
                if contract_scenario["target_tons"] > 0
                else None
            )

            reserve_tons = (
                predicted_tons
                - contract_scenario["target_tons"]
            )

            reserve_percent = (
                reserve_tons
                / contract_scenario["target_tons"]
                * 100
                if contract_scenario["target_tons"] > 0
                else None
            )

            estimated_loading_days = (
                contract_scenario["ship_tons"]
                / predicted_tons
                if predicted_tons > 0
                else None
            )

            recommendations.append({
                "ship_tons": contract_scenario["ship_tons"],
                "target_tons": contract_scenario["target_tons"],
                "safety_margin_percent": SAFETY_MARGIN_PERCENT,
                "required_tons_with_margin": round(
                    required_tons_with_margin,
                    1,
                ),
                "external_scale_load": external_scale_load,
                "road_delay_percent": road_delay_percent,
                "configuration_found": True,
                "number_of_trucks": int(
                    recommendation["number_of_trucks"]
                ),
                "number_of_loaders": int(
                    recommendation["number_of_loaders"]
                ),
                "unload_points": int(
                    recommendation["unload_points"]
                ),
                "predicted_total_tons": round(
                    predicted_tons,
                    1,
                ),
                "target_completion_percent": (
                    round(target_completion, 1)
                    if target_completion is not None
                    else None
                ),
                "production_reserve_tons": round(
                    reserve_tons,
                    1,
                ),
                "production_reserve_percent": (
                    round(reserve_percent, 1)
                    if reserve_percent is not None
                    else None
                ),
                "estimated_loading_days": (
                    round(estimated_loading_days, 2)
                    if estimated_loading_days is not None
                    else None
                ),
                "predicted_bottleneck": (
                    recommendation["predicted_bottleneck"]
                ),
            })


# -----------------------------------------------------------------------------
# 12. Збереження результатів
# -----------------------------------------------------------------------------

recommendation_df = pd.DataFrame(recommendations)

recommendation_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


# -----------------------------------------------------------------------------
# 13. Контрольний індикатор завершення
# -----------------------------------------------------------------------------

found_recommendations = int(
    recommendation_df["configuration_found"]
    .fillna(False)
    .sum()
)

print("\n" + "=" * 68)
print("Формування рекомендацій успішно завершено")
print("=" * 68)
print(f"Перевірено конфігурацій : {len(config_df)}")
print(f"Сформовано сценаріїв    : {len(recommendation_df)}")
print(f"Знайдено рекомендацій   : {found_recommendations}")
print(f"Запас продуктивності    : {SAFETY_MARGIN_PERCENT}%")
print(f"Збережено файл          : {OUTPUT_PATH}")
print("=" * 68)
