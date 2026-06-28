import pandas as pd


DATA_PATH = "data/synthetic/scenario_summary.csv"


def find_min_resource_configuration(df, target_tons):
    """
    Пошук мінімальної конфігурації ресурсів,
    яка забезпечує виконання планового тоннажу.
    """

    candidates = df[
        df["total_tons"] >= target_tons
    ].copy()

    if candidates.empty:
        return None

    candidates["total_avg_wait"] = (
        candidates["avg_wait_tare"]
        + candidates["avg_wait_loader"]
        + candidates["avg_wait_gross"]
        + candidates["avg_wait_unload"]
    )

    candidates = candidates.sort_values(
        by=[
            "number_of_trucks",
            "number_of_loaders",
            "unload_points",
            "external_scale_load",
            "total_avg_wait"
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True
        ]
    )

    return candidates.iloc[0]


def print_recommendation(ship_tons, target_tons, recommendation):
    print("\n" + "=" * 60)
    print(f"Судно: {ship_tons} т")
    print(f"Планова норма: {target_tons} т/добу")

    if recommendation is None:
        print("Рекомендація: конфігурацію не знайдено")
        return

    print("Рекомендована мінімальна конфігурація:")
    print(f"- Самоскиди: {int(recommendation['number_of_trucks'])}")
    print(f"- Мехлопати: {int(recommendation['number_of_loaders'])}")
    print(f"- Точки вигрузки: {int(recommendation['unload_points'])}")
    print(f"- Зовнішнє завантаження вагової: {int(recommendation['external_scale_load'])}%")

    print("\nОчікуваний результат:")
    print(f"- Тоннаж: {recommendation['total_tons']:.1f} т/добу")
    print(f"- Виконання плану: {recommendation['target_completion']:.1f}%")
    print(f"- Середній цикл: {recommendation['average_cycle_time']:.1f} хв")

    print("\nСередні очікування:")
    print(f"- Тарування: {recommendation['avg_wait_tare']:.2f} хв")
    print(f"- Навантаження: {recommendation['avg_wait_loader']:.2f} хв")
    print(f"- Брутто: {recommendation['avg_wait_gross']:.2f} хв")
    print(f"- Розвантаження: {recommendation['avg_wait_unload']:.2f} хв")

    print("\nУтилізація ресурсів:")
    print(f"- Мехлопати: {recommendation['loader_utilization']:.1f}%")
    print(f"- Вагова: {recommendation['scale_utilization']:.1f}%")
    print(f"- Вигрузка: {recommendation['unload_utilization']:.1f}%")


df = pd.read_csv(DATA_PATH)

contract_scenarios = [
    {"ship_tons": 5000, "target_tons": 3000},
    {"ship_tons": 10000, "target_tons": 4000},
]

scale_load_scenarios = [0, 30, 60]

recommendations = []

for scenario in contract_scenarios:
    for scale_load in scale_load_scenarios:

        filtered_df = df[
            df["external_scale_load"] == scale_load
        ]

        recommendation = find_min_resource_configuration(
            filtered_df,
            scenario["target_tons"]
        )

        print_recommendation(
            scenario["ship_tons"],
            scenario["target_tons"],
            recommendation
        )

        print(f"Умова: зовнішнє завантаження вагової {scale_load}%")

        if recommendation is not None:
            row = recommendation.to_dict()
            row["ship_tons"] = scenario["ship_tons"]
            row["contract_target_tons"] = scenario["target_tons"]
            row["selected_external_scale_load"] = scale_load
            recommendations.append(row)

recommendation_df = pd.DataFrame(recommendations)

recommendation_df.to_csv(
    "data/synthetic/resource_recommendations.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nФайл збережено:")
print("data/synthetic/resource_recommendations.csv")