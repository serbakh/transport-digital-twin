import simpy
import pandas as pd
import random

# Основні параметри моделі
START_TIME = 8 * 60 + 15
SIMULATION_DURATION = 24 * 60
SIMULATION_END = START_TIME + SIMULATION_DURATION

NUMBER_OF_TRUCKS = 9

DAILY_TARGET_TONS = 3500

TARE_TIME = 3
GROSS_TIME = 3
RETURN_TIME = 20

TRUCK_SCENARIOS = list(range(4, 21))
LOADER_SCENARIOS = [1, 2, 3]
SCALE_EXTERNAL_LOAD_SCENARIOS = [0, 10, 20, 30, 40, 50, 60]
UNLOAD_POINT_SCENARIOS = [1, 2, 3]

trip_results = []
scenario_results = []

def external_scale_traffic(env, scale, external_load_percent):
    """
    Фонове навантаження вагової стороннім транспортом.
    """

    if external_load_percent == 0:
        return

    # Чим більший відсоток, тим частіше сторонній транспорт займає вагову
    interval = 300 / external_load_percent

    while env.now < SIMULATION_END:
        yield env.timeout(interval)

        yield env.process(wait_until_work_time(env))

        with scale.request() as request:
            yield request
            yield env.timeout(GROSS_TIME)
            
def is_break_time(current_minute):
    """
    Перевіряє, чи поточний час потрапляє у перерву або пересмінку.
    Операції, що вже почалися, не перериваються.
    Нові операції під час перерви не стартують.
    """

    minute_in_day = current_minute % 1440

    # Пересмінка 08:00–08:15
    if 480 <= minute_in_day < 495:
        return True

    # Обід денний 12:00–13:00
    if 720 <= minute_in_day < 780:
        return True

    # Пересмінка 20:00–20:15
    if 1200 <= minute_in_day < 1215:
        return True

    # Обід нічний 00:00–01:00
    if 0 <= minute_in_day < 60:
        return True

    return False


def wait_until_work_time(env):
    """
    Якщо поточний момент потрапляє у перерву, процес очікує,
    доки робота знову стане доступною.
    """

    while is_break_time(env.now):
        yield env.timeout(1)


def truck_process(env, truck_id, loader, scale, unloading_point, number_of_trucks, number_of_loaders, external_scale_load, unload_points, resource_stats):
    """
    Безперервна робота одного самоскида протягом доби.
    """

    trip_id = 0

    while env.now < SIMULATION_END:
        trip_id += 1
        start_time = env.now

        cargo_weight = round(random.uniform(27, 28), 2)

        load_time = round(random.uniform(8, 10), 1)

        travel_time = round(random.uniform(18, 25), 1)

        unload_time = round(random.uniform(12, 15), 1)

        wait_unload = 0
        wait_tare = 0
        wait_loader = 0
        wait_gross = 0

        print(f"{env.now}: Самоскид {truck_id}, рейс {trip_id} прибув на тарування")

        # Тарування
        yield env.process(wait_until_work_time(env))
        tare_queue_enter = env.now

        with scale.request() as request:
            yield request
            wait_tare = env.now - tare_queue_enter

            yield env.process(wait_until_work_time(env))
            print(f"{env.now}: Самоскид {truck_id}, рейс {trip_id} почав тарування")
            resource_stats["scale_busy_time"] += TARE_TIME
            yield env.timeout(TARE_TIME)
            print(f"{env.now}: Самоскид {truck_id}, рейс {trip_id} завершив тарування")

        # Навантаження
        yield env.process(wait_until_work_time(env))
        loader_queue_enter = env.now

        with loader.request() as request:
            yield request
            wait_loader = env.now - loader_queue_enter

            yield env.process(wait_until_work_time(env))
            print(f"{env.now}: Самоскид {truck_id}, рейс {trip_id} почав навантаження")
            resource_stats["loader_busy_time"] += load_time
            yield env.timeout(load_time)
            print(f"{env.now}: Самоскид {truck_id}, рейс {trip_id} завершив навантаження")

        # Зважування брутто
        yield env.process(wait_until_work_time(env))
        gross_queue_enter = env.now

        with scale.request() as request:
            yield request
            wait_gross = env.now - gross_queue_enter

            yield env.process(wait_until_work_time(env))
            print(f"{env.now}: Самоскид {truck_id}, рейс {trip_id} почав зважування брутто")
            resource_stats["scale_busy_time"] += GROSS_TIME
            yield env.timeout(GROSS_TIME)
            print(f"{env.now}: Самоскид {truck_id}, рейс {trip_id} завершив зважування брутто")

        # Рух до причалу
        yield env.timeout(travel_time)
        print(f"{env.now}: Самоскид {truck_id}, рейс {trip_id} прибув до причалу")

        # Розвантаження
        yield env.process(wait_until_work_time(env))
        unload_queue_enter = env.now

        with unloading_point.request() as request:
            yield request
            wait_unload = env.now - unload_queue_enter

            yield env.process(wait_until_work_time(env))
            print(f"{env.now}: Самоскид {truck_id}, рейс {trip_id} почав розвантаження")
            resource_stats["unload_busy_time"] += unload_time
            yield env.timeout(unload_time)
            print(f"{env.now}: Самоскид {truck_id}, рейс {trip_id} завершив розвантаження")

        finish_time = env.now
        total_time = finish_time - start_time

        # Запис результатів рейсу
        if finish_time <= SIMULATION_END:
            trip_results.append({
                "unload_points": unload_points,
                "wait_unload": wait_unload,
                "scenario_trucks": number_of_trucks,
                "scenario_loaders": number_of_loaders,
                "external_scale_load": external_scale_load,
                "truck_id": truck_id,
                "trip_id": trip_id,
                "cargo_weight": cargo_weight,

                "wait_tare": wait_tare,
                "wait_loader": wait_loader,
                "wait_gross": wait_gross,

                "tare_time": TARE_TIME,
                "load_time": load_time,
                "gross_time": GROSS_TIME,
                "travel_time": travel_time,
                "unload_time": unload_time,
                "return_time": RETURN_TIME,

                "start_time": start_time,
                "finish_time": finish_time,
                "total_time": total_time
            })

        # Повернення самоскида на склад
        yield env.timeout(RETURN_TIME)
        
def run_scenario(number_of_trucks, number_of_loaders, external_scale_load, unload_points):
    """
    Запуск одного сценарію симуляції.
    """

    global trip_results
    trip_results = []

    env = simpy.Environment(initial_time=START_TIME)
    
    loader = simpy.Resource(env, capacity=number_of_loaders)
    scale = simpy.Resource(env, capacity=1)
    unloading_point = simpy.Resource(env, capacity=unload_points)
    resource_stats = {
        "loader_busy_time": 0,
        "scale_busy_time": 0,
        "unload_busy_time": 0
    }

    env.process(external_scale_traffic(env, scale, external_scale_load))

    for truck_id in range(1, number_of_trucks + 1):
        env.process(
            truck_process(
                env,
                truck_id,
                loader,
                scale,
                unloading_point,
                number_of_trucks,
                number_of_loaders,
                external_scale_load,
                unload_points,
                resource_stats
            )
        )

    env.run(until=SIMULATION_END)

    df = pd.DataFrame(trip_results)

    total_trips = len(df)
    total_tons = df["cargo_weight"].sum() if total_trips > 0 else 0
    target_completion = total_tons / DAILY_TARGET_TONS * 100 if DAILY_TARGET_TONS > 0 else 0
    average_cycle_time = df["total_time"].mean() if total_trips > 0 else 0

    avg_wait_tare = df["wait_tare"].mean() if total_trips > 0 else 0
    avg_wait_loader = df["wait_loader"].mean() if total_trips > 0 else 0
    avg_wait_gross = df["wait_gross"].mean() if total_trips > 0 else 0
    avg_wait_unload = df["wait_unload"].mean() if total_trips > 0 else 0
    
    max_wait_tare = df["wait_tare"].max() if total_trips > 0 else 0
    max_wait_loader = df["wait_loader"].max() if total_trips > 0 else 0
    max_wait_gross = df["wait_gross"].max() if total_trips > 0 else 0
    max_wait_unload = df["wait_unload"].max() if total_trips > 0 else 0

    loader_utilization = (
        resource_stats["loader_busy_time"]
        / (SIMULATION_DURATION * number_of_loaders)
        * 100
    )

    scale_utilization = (
        resource_stats["scale_busy_time"]
        / SIMULATION_DURATION
        * 100
    )

    unload_utilization = (
        resource_stats["unload_busy_time"]
        / (SIMULATION_DURATION * unload_points)
        * 100
    )

    return {
        "number_of_trucks": number_of_trucks,
        "number_of_loaders": number_of_loaders,
        "total_trips": total_trips,
        "total_tons": round(total_tons, 1),
        "target_completion": round(target_completion, 1),
        "average_cycle_time": round(average_cycle_time, 1),
        "avg_wait_tare": round(avg_wait_tare, 2),
        "avg_wait_loader": round(avg_wait_loader, 2),
        "avg_wait_gross": round(avg_wait_gross, 2),
        "max_wait_tare": round(max_wait_tare, 2),
        "max_wait_loader": round(max_wait_loader, 2),
        "max_wait_gross": round(max_wait_gross, 2),
        "max_wait_unload": round(max_wait_unload, 2),
        "loader_utilization": round(loader_utilization, 2),
        "scale_utilization": round(scale_utilization, 2),
        "unload_utilization": round(unload_utilization, 2),
        "external_scale_load": external_scale_load,
        "unload_points": unload_points,
        "avg_wait_unload": round(avg_wait_unload, 2),
    }, df

all_trips = []

for external_scale_load in SCALE_EXTERNAL_LOAD_SCENARIOS:
    for number_of_loaders in LOADER_SCENARIOS:
        for unload_points in UNLOAD_POINT_SCENARIOS:
            for number_of_trucks in TRUCK_SCENARIOS:

                summary, trips_df = run_scenario(
                    number_of_trucks,
                    number_of_loaders,
                    external_scale_load,
                    unload_points
                )

                scenario_results.append(summary)
                all_trips.append(trips_df)
                

            print(
                f"Scenario: trucks={number_of_trucks}, "
                f"loaders={number_of_loaders}, "
                f"scale_load={external_scale_load}%, "
                f"tons={summary['total_tons']}, "
                f"completion={summary['target_completion']}%, "
                f"avg_cycle={summary['average_cycle_time']} хв, "
                f"avg_wait_loader={summary['avg_wait_loader']} хв"
            )

scenario_df = pd.DataFrame(scenario_results)
trips_df = pd.concat(all_trips, ignore_index=True)


print("Кількість сценаріїв:", len(scenario_df))

scenario_df.to_csv(
    "data/synthetic/scenario_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

trips_df.to_csv(
    "data/synthetic/trips_all_scenarios.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nПідсумкова таблиця сценаріїв:")
print(scenario_df)

print("\nТОП-10 сценаріїв за тоннажем:")

top_df = scenario_df.sort_values(
    by="total_tons",
    ascending=False
)

print("\nСценарії, що виконують план 3500 т/добу:")

plan_df = scenario_df[
    scenario_df["total_tons"] >= 3500
]

print(plan_df)
print(top_df.head(10))
print("\nФайли збережено:")
print("data/synthetic/scenario_summary.csv")
print("data/synthetic/trips_all_scenarios.csv")

