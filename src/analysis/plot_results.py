"""
Побудова аналітичних графіків за результатами
дискретно-подієвого моделювання транспортно-експедиторського процесу.

Файл формує компактний набір графіків, придатних для використання
у статті та для аналізу результатів цифрового двійника.

Графіки:
1. Добовий тоннаж залежно від кількості самоскидів.
2. Середній час очікування за основними операціями.
3. Завантаженість виробничих ресурсів.
4. Вплив зовнішнього навантаження вагової на продуктивність.
5. Вплив дорожніх затримок на продуктивність.
"""

from pathlib import Path

import matplotlib

# Неінтерактивний backend дозволяє зберігати графіки
# у середовищах без графічного інтерфейсу.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------------------------------------------------------
# 1. Шляхи до вхідних даних і результатів
# -----------------------------------------------------------------------------

DATA_PATH = Path("data/synthetic/scenario_summary.csv")
PLOT_DIR = Path("data/plots")

# Папка для графіків створюється автоматично,
# якщо вона ще не існує.
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 2. Базові умови для порівняльних графіків
#
# Значення можна змінювати для аналізу іншої конфігурації.
# -----------------------------------------------------------------------------

BASE_NUMBER_OF_LOADERS = 2
BASE_UNLOAD_POINTS = 2
BASE_EXTERNAL_SCALE_LOAD = 0
BASE_ROAD_DELAY_PERCENT = 0


# -----------------------------------------------------------------------------
# 3. Завантаження та перевірка даних
# -----------------------------------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Файл із результатами моделювання не знайдено: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

if df.empty:
    raise ValueError(
        "Файл scenario_summary.csv не містить записів."
    )

required_columns = [
    "number_of_trucks",
    "number_of_loaders",
    "unload_points",
    "external_scale_load",
    "road_delay_percent",
    "total_tons",
    "avg_wait_tare",
    "avg_wait_loader",
    "avg_wait_gross",
    "avg_wait_unload",
    "loader_utilization",
    "scale_utilization",
    "unload_utilization",
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

missing_values = int(
    df[required_columns].isna().sum().sum()
)

if missing_values > 0:
    raise ValueError(
        f"У наборі даних виявлено пропуски: {missing_values}"
    )


# -----------------------------------------------------------------------------
# 4. Допоміжні функції
# -----------------------------------------------------------------------------

def save_plot(file_name):
    """
    Зберігає поточний графік у папку результатів
    та закриває фігуру.
    """

    output_path = PLOT_DIR / file_name

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return output_path


def require_non_empty(dataframe, description):
    """
    Перевіряє, чи містить вибірка записи
    для побудови конкретного графіка.
    """

    if dataframe.empty:
        raise ValueError(
            f"Не знайдено даних для побудови графіка: {description}"
        )


generated_files = []

# -----------------------------------------------------------------------------
# 5. Графік 1. Добовий тоннаж залежно від кількості самоскидів
# -----------------------------------------------------------------------------

productivity_df = df[
    (df["number_of_loaders"] == BASE_NUMBER_OF_LOADERS)
    & (df["external_scale_load"] == BASE_EXTERNAL_SCALE_LOAD)
    & (df["road_delay_percent"] == BASE_ROAD_DELAY_PERCENT)
].copy()

require_non_empty(
    productivity_df,
    "добовий тоннаж залежно від кількості самоскидів",
)

plt.figure(figsize=(8, 5))

for unload_points in sorted(
    productivity_df["unload_points"].unique()
):
    subset = productivity_df[
        productivity_df["unload_points"] == unload_points
    ].sort_values("number_of_trucks")

    unload_label = {
        1: "1 точка розвантаження",
        2: "2 точки розвантаження",
        3: "3 точки розвантаження",
    }.get(
        unload_points,
        f"{unload_points} точок розвантаження",
    )

    plt.plot(
        subset["number_of_trucks"],
        subset["total_tons"],
        marker="o",
        label=unload_label,
    )

plt.title(
    "Добовий тоннаж залежно від кількості самоскидів"
)
plt.xlabel("Кількість самоскидів")
plt.ylabel("Добовий тоннаж, т")
plt.grid(True)

plt.legend(
    title="Кількість точок розвантаження"
)

generated_files.append(
    save_plot("01_productivity_by_trucks.png")
)

# -----------------------------------------------------------------------------
# 6. Графік 2. Середній час очікування за основними операціями
# -----------------------------------------------------------------------------

wait_df = df[
    (df["number_of_loaders"] == BASE_NUMBER_OF_LOADERS)
    & (df["unload_points"] == BASE_UNLOAD_POINTS)
    & (df["external_scale_load"] == BASE_EXTERNAL_SCALE_LOAD)
    & (df["road_delay_percent"] == BASE_ROAD_DELAY_PERCENT)
].sort_values("number_of_trucks")

require_non_empty(
    wait_df,
    "середній час очікування за операціями",
)

plt.figure(figsize=(8, 5))

wait_columns = {
    "avg_wait_tare": "Тарування",
    "avg_wait_loader": "Навантаження",
    "avg_wait_gross": "Зважування брутто",
    "avg_wait_unload": "Розвантаження",
}

for column, label in wait_columns.items():
    plt.plot(
        wait_df["number_of_trucks"],
        wait_df[column],
        marker="o",
        label=label,
    )

plt.title(
    "Середній час очікування за основними операціями"
)
plt.xlabel("Кількість самоскидів")
plt.ylabel("Середній час очікування, хв")
plt.grid(True)
plt.legend()

generated_files.append(
    save_plot("02_average_waiting_times.png")
)


# -----------------------------------------------------------------------------
# 7. Графік 3. Завантаженість виробничих ресурсів
# -----------------------------------------------------------------------------

utilization_df = df[
    (df["number_of_loaders"] == BASE_NUMBER_OF_LOADERS)
    & (df["unload_points"] == BASE_UNLOAD_POINTS)
    & (df["external_scale_load"] == BASE_EXTERNAL_SCALE_LOAD)
    & (df["road_delay_percent"] == BASE_ROAD_DELAY_PERCENT)
].sort_values("number_of_trucks")

require_non_empty(
    utilization_df,
    "завантаженість виробничих ресурсів",
)

plt.figure(figsize=(8, 5))

utilization_columns = {
    "loader_utilization": "Мехлопати",
    "scale_utilization": "Вагова",
    "unload_utilization": "Розвантаження",
}

for column, label in utilization_columns.items():
    plt.plot(
        utilization_df["number_of_trucks"],
        utilization_df[column],
        marker="o",
        label=label,
    )

plt.title(
    "Завантаженість виробничих ресурсів"
)
plt.xlabel("Кількість самоскидів")
plt.ylabel("Завантаженість ресурсу, %")
plt.grid(True)
plt.legend()

generated_files.append(
    save_plot("03_resource_utilization.png")
)


# -----------------------------------------------------------------------------
# 8. Графік 4. Вплив зовнішнього навантаження вагової
# -----------------------------------------------------------------------------

scale_load_df = df[
    (df["number_of_loaders"] == BASE_NUMBER_OF_LOADERS)
    & (df["unload_points"] == BASE_UNLOAD_POINTS)
    & (df["road_delay_percent"] == BASE_ROAD_DELAY_PERCENT)
].copy()

require_non_empty(
    scale_load_df,
    "вплив зовнішнього навантаження вагової",
)

plt.figure(figsize=(8, 5))

for external_scale_load in sorted(
    scale_load_df["external_scale_load"].unique()
):
    subset = scale_load_df[
        scale_load_df["external_scale_load"]
        == external_scale_load
    ].sort_values("number_of_trucks")

    plt.plot(
        subset["number_of_trucks"],
        subset["total_tons"],
        marker="o",
        label=f"{external_scale_load}%",
    )

plt.title(
    "Вплив зовнішнього навантаження вагової "
    "на добовий тоннаж"
)
plt.xlabel("Кількість самоскидів")
plt.ylabel("Добовий тоннаж, т")
plt.grid(True)
plt.legend(title="Зовнішнє навантаження")

generated_files.append(
    save_plot("04_external_scale_load_effect.png")
)


# -----------------------------------------------------------------------------
# 9. Графік 5. Вплив дорожніх затримок
# -----------------------------------------------------------------------------

road_delay_df = df[
    (df["number_of_loaders"] == BASE_NUMBER_OF_LOADERS)
    & (df["unload_points"] == BASE_UNLOAD_POINTS)
    & (
        df["external_scale_load"]
        == BASE_EXTERNAL_SCALE_LOAD
    )
].copy()

require_non_empty(
    road_delay_df,
    "вплив дорожніх затримок",
)

plt.figure(figsize=(8, 5))

for road_delay_percent in sorted(
    road_delay_df["road_delay_percent"].unique()
):
    subset = road_delay_df[
        road_delay_df["road_delay_percent"]
        == road_delay_percent
    ].sort_values("number_of_trucks")

    plt.plot(
        subset["number_of_trucks"],
        subset["total_tons"],
        marker="o",
        label=f"до {road_delay_percent}%",
    )

plt.title(
    "Вплив дорожніх затримок на добовий тоннаж"
)
plt.xlabel("Кількість самоскидів")
plt.ylabel("Добовий тоннаж, т")
plt.grid(True)
plt.legend(title="Дорожня затримка")

generated_files.append(
    save_plot("05_road_delay_effect.png")
)


# -----------------------------------------------------------------------------
# 10. Контрольний індикатор завершення
# -----------------------------------------------------------------------------

print("=" * 68)
print("Побудову графіків успішно завершено")
print("=" * 68)
print(f"Вхідний файл       : {DATA_PATH}")
print(f"Папка результатів  : {PLOT_DIR}")
print(f"Створено графіків  : {len(generated_files)}")
print("Збережено файли:")

for file_path in generated_files:
    print(f"  - {file_path}")

print("=" * 68)
