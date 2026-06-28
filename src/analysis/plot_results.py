import pandas as pd
import matplotlib.pyplot as plt

# Завантаження результатів сценаріїв
df = pd.read_csv("data/synthetic/scenario_summary.csv")

# ------------------------------
# Графік 1. Тоннаж від кількості самоскидів
# ------------------------------

for loaders in sorted(df["number_of_loaders"].unique()):
    for unload_points in sorted(df["unload_points"].unique()):

        subset = df[
            (df["number_of_loaders"] == loaders)
            & (df["external_scale_load"] == 0)
            & (df["unload_points"] == unload_points)
        ]

        plt.figure(figsize=(8, 5))

        plt.plot(
            subset["number_of_trucks"],
            subset["total_tons"],
            marker="o"
        )

        plt.title(
            f"Продуктивність системи ({loaders} мехлопата, {unload_points} точка вигрузки)"
        )

        plt.xlabel("Кількість самоскидів")
        plt.ylabel("Тонн за добу")

        plt.grid(True)

        plt.savefig(
            f"data/synthetic/productivity_loader_{loaders}_unload_{unload_points}.png"
        )

        plt.close()

# ------------------------------
# Графік 2. Очікування навантаження
# ------------------------------

for loaders in sorted(df["number_of_loaders"].unique()):
    for unload_points in sorted(df["unload_points"].unique()):

        subset = df[
            (df["number_of_loaders"] == loaders)
            & (df["external_scale_load"] == 0)
            & (df["unload_points"] == unload_points)
        ]

        plt.figure(figsize=(8, 5))

        plt.plot(
            subset["number_of_trucks"],
            subset["avg_wait_loader"],
            marker="o"
        )

        plt.title(
            f"Середнє очікування навантаження ({loaders} мехлопата, {unload_points} точка вигрузки)"
        )

        plt.xlabel("Кількість самоскидів")
        plt.ylabel("Очікування навантаження, хв")

        plt.grid(True)

        plt.savefig(
            f"data/synthetic/wait_loader_{loaders}_unload_{unload_points}.png"
        )

        plt.close()

# ------------------------------
# Графік 3. Очікування розвантаження
# ------------------------------

for loaders in sorted(df["number_of_loaders"].unique()):
    for unload_points in sorted(df["unload_points"].unique()):

        subset = df[
            (df["number_of_loaders"] == loaders)
            & (df["external_scale_load"] == 0)
            & (df["unload_points"] == unload_points)
        ]

        plt.figure(figsize=(8, 5))

        plt.plot(
            subset["number_of_trucks"],
            subset["avg_wait_unload"],
            marker="o"
        )

        plt.title(
            f"Середнє очікування розвантаження ({loaders} мехлопата, {unload_points} точка вигрузки)"
        )

        plt.xlabel("Кількість самоскидів")
        plt.ylabel("Очікування розвантаження, хв")

        plt.grid(True)

        plt.savefig(
            f"data/synthetic/wait_unload_loader_{loaders}_unload_{unload_points}.png"
        )

        plt.close()
# ------------------------------
# Графік 4. Максимальне очікування навантаження
# ------------------------------

for loaders in sorted(df["number_of_loaders"].unique()):
    for unload_points in sorted(df["unload_points"].unique()):

        subset = df[
            (df["number_of_loaders"] == loaders)
            & (df["external_scale_load"] == 0)
            & (df["unload_points"] == unload_points)
        ]

        plt.figure(figsize=(8, 5))

        plt.plot(
            subset["number_of_trucks"],
            subset["max_wait_loader"],
            marker="o"
        )

        plt.title(
            f"Максимальне очікування навантаження ({loaders} мехлопата, {unload_points} точка вигрузки)"
        )

        plt.xlabel("Кількість самоскидів")
        plt.ylabel("Максимальне очікування навантаження, хв")

        plt.grid(True)

        plt.savefig(
            f"data/synthetic/max_wait_loader_{loaders}_unload_{unload_points}.png"
        )

        plt.close()


# ------------------------------
# Графік 5. Максимальне очікування розвантаження
# ------------------------------

for loaders in sorted(df["number_of_loaders"].unique()):
    for unload_points in sorted(df["unload_points"].unique()):

        subset = df[
            (df["number_of_loaders"] == loaders)
            & (df["external_scale_load"] == 0)
            & (df["unload_points"] == unload_points)
        ]

        plt.figure(figsize=(8, 5))

        plt.plot(
            subset["number_of_trucks"],
            subset["max_wait_unload"],
            marker="o"
        )

        plt.title(
            f"Максимальне очікування розвантаження ({loaders} мехлопата, {unload_points} точка вигрузки)"
        )

        plt.xlabel("Кількість самоскидів")
        plt.ylabel("Максимальне очікування розвантаження, хв")

        plt.grid(True)

        plt.savefig(
            f"data/synthetic/max_wait_unload_{loaders}_unload_{unload_points}.png"
        )

        plt.close()
# ------------------------------
# Графік 6. Очікування тарування
# ------------------------------

for loaders in sorted(df["number_of_loaders"].unique()):
    for unload_points in sorted(df["unload_points"].unique()):

        subset = df[
            (df["number_of_loaders"] == loaders)
            & (df["external_scale_load"] == 0)
            & (df["unload_points"] == unload_points)
        ]

        plt.figure(figsize=(8, 5))

        plt.plot(
            subset["number_of_trucks"],
            subset["avg_wait_tare"],
            marker="o"
        )

        plt.title(
            f"Середнє очікування тарування ({loaders} мехлопата, {unload_points} точка вигрузки)"
        )

        plt.xlabel("Кількість самоскидів")
        plt.ylabel("Очікування тарування, хв")

        plt.grid(True)

        plt.savefig(
            f"data/synthetic/wait_tare_loader_{loaders}_unload_{unload_points}.png"
        )

        plt.close()
# ------------------------------
# Графік 7. Очікування зважування брутто
# ------------------------------

for loaders in sorted(df["number_of_loaders"].unique()):
    for unload_points in sorted(df["unload_points"].unique()):

        subset = df[
            (df["number_of_loaders"] == loaders)
            & (df["external_scale_load"] == 0)
            & (df["unload_points"] == unload_points)
        ]

        plt.figure(figsize=(8, 5))

        plt.plot(
            subset["number_of_trucks"],
            subset["avg_wait_gross"],
            marker="o"
        )

        plt.title(
            f"Середнє очікування зважування брутто ({loaders} мехлопата, {unload_points} точка вигрузки)"
        )

        plt.xlabel("Кількість самоскидів")
        plt.ylabel("Очікування зважування брутто, хв")

        plt.grid(True)

        plt.savefig(
            f"data/synthetic/wait_gross_loader_{loaders}_unload_{unload_points}.png"
        )

        plt.close()

      
print("Графіки збережено.")