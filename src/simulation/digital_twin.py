"""
Дискретно-подієва імітаційна модель транспортно-експедиторського процесу.

Призначення файлу:
1. Відтворити виробничий цикл доставки вантажу від складу до борта судна.
2. Провести серію сценарних експериментів зі зміною кількості ресурсів.
3. Згенерувати синтетичні дані для подальшого навчання моделей машинного навчання.

Логіка процесу одного самоскида:
тарування -> навантаження -> зважування брутто -> рух до причалу -> розвантаження -> повернення на склад.

Одиниця модельного часу: хвилина.

Методологічні примітки:
- random.seed() навмисно не фіксується, оскільки модель використовується як генератор
  нових синтетичних наборів даних, а не як одноразовий відтворюваний приклад.
- Дорожня ділянка не моделюється як окремий ресурс із чергою. Її вплив задано через
  сценарний коефіцієнт випадкової затримки, щоб не створювати штучне вузьке місце.
- Перерви та перезмінки враховуються під час виконання операцій за правилом порогу:
  якщо до завершення операції залишається не більше FINISH_THRESHOLD_MIN хвилин, операція завершується;
  якщо більше — призупиняється до закінчення перерви.
"""

import simpy
import pandas as pd
import random
import time
import os
import numpy as np

# -----------------------------------------------------------------------------
# 1. Глобальні параметри моделі
# -----------------------------------------------------------------------------

# Початок активної роботи моделі: 08:15.
# Час задається у хвилинах від початку доби: 8 * 60 + 15 = 495 хвилин.
START_TIME = 8 * 60 + 15

# Тривалість моделювання — 24 години.
SIMULATION_DURATION = 24 * 60

# Кінцевий момент моделювання у хвилинах.
# Оскільки старт не з 00:00, а з 08:15, кінець дорівнює 08:15 наступної доби.
SIMULATION_END = START_TIME + SIMULATION_DURATION

# Поріг завершення операції перед перервою, хв.
# Якщо до завершення операції залишається не більше 1 хвилини,
# операція завершується без зупинки.
# Якщо залишок більший — виконання призупиняється до завершення перерви.
FINISH_THRESHOLD_MIN = 1

# Постійні технологічні тривалості, хв.
# Тарування — зважування порожнього самоскида.
TARE_TIME = 3

# Зважування брутто — зважування завантаженого самоскида.
GROSS_TIME = 3

# Тарування та зважування брутто тривають по 3 хвилини,
# тобто не довше встановленого порогу завершення операції.
# Тому вже розпочате зважування не переривається через початок
# перерви або перезмінки, але нова операція під час перерви
# не розпочинається.

# Повернення самоскида від причалу назад на склад.
RETURN_TIME = 20

# -----------------------------------------------------------------------------
# 2. Простір сценаріїв
# -----------------------------------------------------------------------------

# Кількість самоскидів у сценарних експериментах.
# Саме цей параметр дозволяє дослідити, як парк машин впливає на продуктивність
# та чи виникає перенасичення системи транспортом.
TRUCK_SCENARIOS = list(range(4, 21))

# Кількість мехлопат / навантажувальних ресурсів.
# Зміна цього параметра дозволяє перевірити, чи є навантаження обмежувальним етапом.
LOADER_SCENARIOS = [1, 2, 3]

# Зовнішнє навантаження вагової, %.
# 0% — вагова використовується тільки досліджуваним процесом;
# 5% — незначна стороння активність;
# 50% — суттєве спільне використання вагової іншими потоками транспорту.
# Вагова не задається як гарантоване вузьке місце, а використовується як фактор шуму.
SCALE_EXTERNAL_LOAD_SCENARIOS = [0, 5, 50]

# Кількість точок розвантаження на причалі / біля борта судна.
# Дозволяє дослідити, як пропускна здатність розвантаження впливає на систему.
UNLOAD_POINT_SCENARIOS = [1, 2, 3]

# Сценарії зовнішньої нестабільності дорожньої ділянки, %.
# Це не ресурс і не черга, а випадкове збільшення часу дороги до причалу.
# Такий підхід дозволяє врахувати затримки на маршруті, в зоні в'їзду до порту
# або біля прохідної, не створюючи штучного вузького місця.
ROAD_DELAY_SCENARIOS = [0, 5, 15, 30]

# Договірні плани 3000 та 4000 т/добу не вводяться в симуляцію як сценарії.
# Цифровий двійник генерує фактичну продуктивність конфігурацій,
# а перевірка виконання плану виконується окремим модулем рекомендацій.

# -----------------------------------------------------------------------------
# 3. Глобальні списки для накопичення результатів
# -----------------------------------------------------------------------------

# Детальні результати по кожному рейсу в межах поточного сценарію.
trip_results = []

# Підсумкові результати по кожному сценарію.
scenario_results = []


# -----------------------------------------------------------------------------
# 4. Допоміжний процес: зовнішнє навантаження вагової
# -----------------------------------------------------------------------------

def external_scale_traffic(env, scale, external_load_percent, resource_stats):
    """
    Імітує фонове навантаження вагової стороннім транспортом.

    Логіка:
    - якщо зовнішнє навантаження дорівнює 0%, вагова використовується лише самоскидами моделі;
    - якщо навантаження більше 0%, через певні інтервали з'являється сторонній транспорт;
    - сторонній транспорт займає вагову на час, еквівалентний одному зважуванню.

    Це не означає, що вагова навмисно робиться вузьким місцем. Параметр потрібен,
    щоб перевірити чутливість процесу до спільного використання ресурсу.
    """

    if external_load_percent == 0:
        return

    
    # Інтервал між зверненнями стороннього транспорту розраховується так,
    # щоб частка часу зайнятості вагової зовнішнім потоком відповідала
    # заданому сценарному значенню external_load_percent.
    idle_interval = (
        GROSS_TIME
        * (100 - external_load_percent)
        / external_load_percent
    )

    while env.now < SIMULATION_END:
        yield env.timeout(idle_interval)
        yield env.process(wait_until_work_time(env))

        with scale.request() as request:
            yield request
            resource_stats["scale_external_busy_time"] += GROSS_TIME
            yield env.process(perform_operation_with_breaks(env, GROSS_TIME))


# -----------------------------------------------------------------------------
# 5. Календар роботи: перерви та пересмінки
# -----------------------------------------------------------------------------

# Добові інтервали перерв у хвилинах від початку доби.
# Формат: (початок, кінець). Кінець не включається.
BREAK_INTERVALS = [
    (0, 60),       # Нічний обід 00:00–01:00.
    (480, 495),    # Пересмінка 08:00–08:15.
    (720, 780),    # Денний обід 12:00–13:00.
    (1200, 1215),  # Пересмінка 20:00–20:15.
]


def is_break_time(current_minute):
    """
    Перевіряє, чи поточний модельний час потрапляє у перерву або пересмінку.
    """

    minute_in_day = current_minute % 1440

    for start, end in BREAK_INTERVALS:
        if start <= minute_in_day < end:
            return True

    return False


def time_until_break_end(current_minute):
    """
    Обчислює, скільки хвилин залишилося до завершення поточної перерви.

    На відміну від покрокового очікування по 1 хвилині, ця функція дозволяє
    одразу перейти до кінця перерви. Це зменшує кількість подій SimPy при
    великій кількості сценаріїв і не змінює логіку моделі.
    """

    minute_in_day = current_minute % 1440

    for start, end in BREAK_INTERVALS:
        if start <= minute_in_day < end:
            return end - minute_in_day

    return 0


def wait_until_work_time(env):
    """
    Затримує старт нової операції, якщо поточний момент припадає на перерву.

    Повертає кількість хвилин очікування. Це дозволяє окремо враховувати
    простої через календар роботи, не змішуючи їх із чергою на ресурс.
    """

    waited = 0
    while is_break_time(env.now):
        remaining_in_break = time_until_break_end(env.now)
        yield env.timeout(remaining_in_break)
        waited += remaining_in_break
    return waited


def time_until_next_break(current_minute):
    """
    Обчислює час до найближчої майбутньої перерви.

    Якщо зараз уже перерва, повертається 0.
    Якщо до кінця модельної доби перерви немає, повертається велике число.
    """

    if is_break_time(current_minute):
        return 0

    current_day_start = current_minute - (current_minute % 1440)
    candidates = []

    # Перевіряємо поточну та наступну добу, бо моделювання переходить через 00:00.
    for day_shift in (0, 1440):
        day_start = current_day_start + day_shift
        for start, _ in BREAK_INTERVALS:
            break_start = day_start + start
            if break_start > current_minute:
                candidates.append(break_start - current_minute)

    return min(candidates) if candidates else 10**9


def perform_operation_with_breaks(env, duration):
    """
    Виконує технологічну операцію з урахуванням перерв і пересмінок.

    Правило моделі:
    - якщо операція може завершитися до перерви — вона завершується звичайно;
    - якщо перерва починається до завершення операції, оцінюється залишок;
    - якщо залишок <= FINISH_THRESHOLD_MIN, операція завершується без зупинки;
    - якщо залишок > FINISH_THRESHOLD_MIN, операція зупиняється до закінчення перерви.

    Повертає час простою через перерви під час перебування на ресурсі.
    Черга на ресурс рахується окремо.
    """

    remaining = duration
    break_wait = 0

    # Якщо ресурс отримано саме під час перерви, транспорт фізично утримує ресурс,
    # але операція не починається до відновлення роботи.
    break_wait += yield env.process(wait_until_work_time(env))

    while remaining > 0:
        to_break = time_until_next_break(env.now)

        # Перерва не заважає завершити операцію.
        if remaining <= to_break:
            yield env.timeout(remaining)
            remaining = 0

        # Перерва почнеться раніше, ніж завершиться операція.
        else:
            remaining_after_break_start = remaining - to_break

            # Якщо залишилось зовсім мало, операцію завершуємо без зупинки.
            if remaining_after_break_start <= FINISH_THRESHOLD_MIN:
                yield env.timeout(remaining)
                remaining = 0

            # Якщо залишок суттєвий, працюємо до перерви, потім чекаємо її завершення.
            else:
                yield env.timeout(to_break)
                remaining -= to_break
                break_wait += yield env.process(wait_until_work_time(env))

    return break_wait


# -----------------------------------------------------------------------------
# 6. Основний процес: робота одного самоскида
# -----------------------------------------------------------------------------

def truck_process(
    env,
    truck_id,
    loader,
    scale,
    unloading_point,
    number_of_trucks,
    number_of_loaders,
    external_scale_load,
    unload_points,
    road_delay_percent,
    resource_stats,
):
    """
    Описує безперервну роботу одного самоскида протягом модельної доби.

    Кожен самоскид багаторазово проходить повний цикл:
    1. Тарування.
    2. Навантаження.
    3. Зважування брутто.
    4. Рух до причалу.
    5. Розвантаження.
    6. Повернення на склад.

    У межах кожного рейсу фіксуються:
    - маса вантажу;
    - тривалість операцій;
    - очікування в чергах;
    - простої через перерви;
    - повний час циклу;
    - параметри сценарію, в якому виконано рейс.
    """

    trip_id = 0

    while env.now < SIMULATION_END:
        trip_id += 1
        start_time = env.now

        # Випадкова маса вантажу, т.
        cargo_weight = round(random.uniform(27, 28), 2)

        # Випадкова тривалість навантаження, хв.
        load_time = round(random.uniform(8, 10), 1)

        # Базова тривалість руху від складу до причалу, хв.
        base_travel_time = round(random.uniform(18, 25), 1)

        # Випадкова додаткова затримка дороги.
        # Наприклад, road_delay_percent=30 означає, що час дороги може випадково
        # збільшитись у межах від 0% до 30% від базового часу.
        road_delay_time = round(base_travel_time * random.uniform(0, road_delay_percent) / 100, 1)
        travel_time = round(base_travel_time + road_delay_time, 1)

        # Випадкова тривалість розвантаження біля борта судна, хв.
        unload_time = round(random.uniform(12, 15), 1)

        # Ініціалізація очікувань.
        # queue — очікування ресурсу в черзі;
        # pre_break — простій через перерву до постановки в чергу;
        # break — простій через календар роботи після отримання ресурсу;
        # total — сумарне очікування, яке включає pre_break + queue + break.
        wait_tare_queue = wait_tare_pre_break = wait_tare_break = wait_tare_total = 0
        wait_loader_queue = wait_loader_pre_break = wait_loader_break = wait_loader_total = 0
        wait_gross_queue = wait_gross_pre_break = wait_gross_break = wait_gross_total = 0
        wait_unload_queue = wait_unload_pre_break = wait_unload_break = wait_unload_total = 0

        # ------------------------------------------------------------------
        # 6.1. Тарування
        # ------------------------------------------------------------------
        # Якщо попередня дія завершилася під час перерви, самоскид ще не стає
        # у чергу на ресурс. Цей простій фіксується окремо як pre_break, щоб
        # баланс часу рейсу не втрачав очікування перед постановкою в чергу.
        wait_tare_pre_break = yield env.process(wait_until_work_time(env))
        tare_queue_enter = env.now

        with scale.request() as request:
            yield request
            wait_tare_queue = env.now - tare_queue_enter
            resource_stats["scale_internal_busy_time"] += TARE_TIME
            wait_tare_break = yield env.process(perform_operation_with_breaks(env, TARE_TIME))
            wait_tare_total = wait_tare_pre_break + wait_tare_queue + wait_tare_break

        # ------------------------------------------------------------------
        # 6.2. Навантаження
        # ------------------------------------------------------------------
        wait_loader_pre_break = yield env.process(wait_until_work_time(env))
        loader_queue_enter = env.now

        with loader.request() as request:
            yield request
            wait_loader_queue = env.now - loader_queue_enter
            resource_stats["loader_busy_time"] += load_time
            wait_loader_break = yield env.process(perform_operation_with_breaks(env, load_time))
            wait_loader_total = wait_loader_pre_break + wait_loader_queue + wait_loader_break

        # ------------------------------------------------------------------
        # 6.3. Зважування брутто
        # ------------------------------------------------------------------
        wait_gross_pre_break = yield env.process(wait_until_work_time(env))
        gross_queue_enter = env.now

        with scale.request() as request:
            yield request
            wait_gross_queue = env.now - gross_queue_enter
            resource_stats["scale_internal_busy_time"] += GROSS_TIME
            wait_gross_break = yield env.process(perform_operation_with_breaks(env, GROSS_TIME))
            wait_gross_total = wait_gross_pre_break + wait_gross_queue + wait_gross_break

        # ------------------------------------------------------------------
        # 6.4. Рух до причалу
        # ------------------------------------------------------------------
        # Дорога не є ресурсом із чергою. Вона моделюється як часова затримка,
        # яка може збільшуватись залежно від сценарію road_delay_percent.
        yield env.timeout(travel_time)

        # ------------------------------------------------------------------
        # 6.5. Розвантаження
        # ------------------------------------------------------------------
        wait_unload_pre_break = yield env.process(wait_until_work_time(env))
        unload_queue_enter = env.now

        with unloading_point.request() as request:
            yield request
            wait_unload_queue = env.now - unload_queue_enter
            resource_stats["unload_busy_time"] += unload_time
            wait_unload_break = yield env.process(perform_operation_with_breaks(env, unload_time))
            wait_unload_total = wait_unload_pre_break + wait_unload_queue + wait_unload_break

        finish_time = env.now
        total_time = finish_time - start_time

        # ------------------------------------------------------------------
        # 6.6. Запис результатів рейсу
        # ------------------------------------------------------------------
        if finish_time <= SIMULATION_END:
            trip_results.append({
                "unload_points": unload_points,
                "road_delay_percent": road_delay_percent,
                "scenario_trucks": number_of_trucks,
                "scenario_loaders": number_of_loaders,
                "external_scale_load": external_scale_load,
                "truck_id": truck_id,
                "trip_id": trip_id,
                "cargo_weight": cargo_weight,
                "wait_tare": wait_tare_total,
                "wait_loader": wait_loader_total,
                "wait_gross": wait_gross_total,
                "wait_unload": wait_unload_total,
                "wait_tare_queue": wait_tare_queue,
                "wait_loader_queue": wait_loader_queue,
                "wait_gross_queue": wait_gross_queue,
                "wait_unload_queue": wait_unload_queue,
                "wait_tare_pre_break": wait_tare_pre_break,
                "wait_loader_pre_break": wait_loader_pre_break,
                "wait_gross_pre_break": wait_gross_pre_break,
                "wait_unload_pre_break": wait_unload_pre_break,
                "wait_tare_break": wait_tare_break,
                "wait_loader_break": wait_loader_break,
                "wait_gross_break": wait_gross_break,
                "wait_unload_break": wait_unload_break,
                "tare_time": TARE_TIME,
                "load_time": load_time,
                "gross_time": GROSS_TIME,
                "base_travel_time": base_travel_time,
                "road_delay_time": road_delay_time,
                "travel_time": travel_time,
                "unload_time": unload_time,
                "return_time": RETURN_TIME,
                "start_time": start_time,
                "finish_time": finish_time,
                "total_time": total_time,
            })

        # ------------------------------------------------------------------
        # 6.7. Повернення самоскида на склад
        # ------------------------------------------------------------------
        # Повернення виконується лише тоді, коли до завершення моделювання
        # ще залишається модельний час.
        if env.now < SIMULATION_END:
            yield env.timeout(
                min(RETURN_TIME, SIMULATION_END - env.now)
            )


# -----------------------------------------------------------------------------
# 7. Запуск одного сценарію
# -----------------------------------------------------------------------------

def run_scenario(
    number_of_trucks,
    number_of_loaders,
    external_scale_load,
    unload_points,
    road_delay_percent,
):
    """
    Запускає один сценарій симуляції з фіксованою конфігурацією ресурсів.

    Вхідні параметри сценарію:
    - number_of_trucks: кількість самоскидів;
    - number_of_loaders: кількість мехлопат;
    - external_scale_load: рівень зовнішнього навантаження вагової;
    - unload_points: кількість точок розвантаження;
    - road_delay_percent: рівень випадкової нестабільності дороги.

    Добовий план навантаження не є параметром симуляції. Він застосовується
    пізніше у модулі рекомендацій для вибору мінімальної конфігурації ресурсів
    під 3000 або 4000 т/добу.

    Результат:
    - summary: агреговані показники сценарію;
    - df: детальна таблиця рейсів у межах сценарію.
    """

    global trip_results
    trip_results = []

    env = simpy.Environment(initial_time=START_TIME)

    loader = simpy.Resource(env, capacity=number_of_loaders)
    scale = simpy.Resource(env, capacity=1)
    unloading_point = simpy.Resource(env, capacity=unload_points)

    resource_stats = {
        "loader_busy_time": 0,
        "scale_internal_busy_time": 0,
        "scale_external_busy_time": 0,
        "unload_busy_time": 0,
    }

    env.process(external_scale_traffic(env, scale, external_scale_load, resource_stats))

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
                road_delay_percent,
                resource_stats,
            )
        )

    env.run(until=SIMULATION_END)

    df = pd.DataFrame(trip_results)

    # ------------------------------------------------------------------
    # 7.1. Агреговані показники сценарію
    # ------------------------------------------------------------------

    total_trips = len(df)
    total_tons = df["cargo_weight"].sum() if total_trips > 0 else 0
    average_cycle_time = df["total_time"].mean() if total_trips > 0 else 0

    avg_wait_tare = df["wait_tare"].mean() if total_trips > 0 else 0
    avg_wait_loader = df["wait_loader"].mean() if total_trips > 0 else 0
    avg_wait_gross = df["wait_gross"].mean() if total_trips > 0 else 0
    avg_wait_unload = df["wait_unload"].mean() if total_trips > 0 else 0
    avg_travel_time = df["travel_time"].mean() if total_trips > 0 else 0
    avg_road_delay_time = df["road_delay_time"].mean() if total_trips > 0 else 0

    max_wait_tare = df["wait_tare"].max() if total_trips > 0 else 0
    max_wait_loader = df["wait_loader"].max() if total_trips > 0 else 0
    max_wait_gross = df["wait_gross"].max() if total_trips > 0 else 0
    max_wait_unload = df["wait_unload"].max() if total_trips > 0 else 0

    # ------------------------------------------------------------------
    # 7.2. Завантаженість ресурсів
    # ------------------------------------------------------------------

    loader_utilization = (
        resource_stats["loader_busy_time"]
        / (SIMULATION_DURATION * number_of_loaders)
        * 100
    )

    # Внутрішнє використання вагової — тільки досліджуваними самоскидами.
    scale_internal_utilization = (
        resource_stats["scale_internal_busy_time"]
        / SIMULATION_DURATION
        * 100
    )

    # Загальне використання вагової — досліджуваний процес + зовнішній транспорт.
    scale_utilization = (
        (resource_stats["scale_internal_busy_time"] + resource_stats["scale_external_busy_time"])
        / SIMULATION_DURATION
        * 100
    )

    unload_utilization = (
        resource_stats["unload_busy_time"]
        / (SIMULATION_DURATION * unload_points)
        * 100
    )

    # Простий допоміжний індикатор вузького місця.
    # Це не остаточний управлінський висновок, а швидка ознака ресурсу
    # з найбільшою завантаженістю у конкретному сценарії.
    utilization_map = {
        "loader": loader_utilization,
        "scale": scale_utilization,
        "unload": unload_utilization,
    }
    bottleneck = max(utilization_map, key=utilization_map.get)

    return {
        "number_of_trucks": number_of_trucks,
        "number_of_loaders": number_of_loaders,
        "unload_points": unload_points,
        "external_scale_load": external_scale_load,
        "road_delay_percent": road_delay_percent,
        "total_trips": total_trips,
        "total_tons": round(total_tons, 1),
        "average_cycle_time": round(average_cycle_time, 1),
        "avg_travel_time": round(avg_travel_time, 2),
        "avg_road_delay_time": round(avg_road_delay_time, 2),
        "avg_wait_tare": round(avg_wait_tare, 2),
        "avg_wait_loader": round(avg_wait_loader, 2),
        "avg_wait_gross": round(avg_wait_gross, 2),
        "avg_wait_unload": round(avg_wait_unload, 2),
        "max_wait_tare": round(max_wait_tare, 2),
        "max_wait_loader": round(max_wait_loader, 2),
        "max_wait_gross": round(max_wait_gross, 2),
        "max_wait_unload": round(max_wait_unload, 2),
        "loader_utilization": round(loader_utilization, 2),
        "scale_internal_utilization": round(scale_internal_utilization, 2),
        "scale_utilization": round(scale_utilization, 2),
        "unload_utilization": round(unload_utilization, 2),
        "bottleneck": bottleneck,
    }, df


# -----------------------------------------------------------------------------
# 8. Масовий запуск усіх сценаріїв
# -----------------------------------------------------------------------------

run_started_at = time.time()
all_trips = []

# Повний перебір сценарного простору:
# зовнішнє навантаження вагової × мехлопати × точки розвантаження
# × нестабільність дороги × самоскиди.
# Договірні плани 3000/4000 т/добу тут не перебираються: вони застосовуються
# окремо після генерації продуктивності сценаріїв.
for external_scale_load in SCALE_EXTERNAL_LOAD_SCENARIOS:
    for number_of_loaders in LOADER_SCENARIOS:
        for unload_points in UNLOAD_POINT_SCENARIOS:
            for road_delay_percent in ROAD_DELAY_SCENARIOS:
                for number_of_trucks in TRUCK_SCENARIOS:

                    summary, trips_df = run_scenario(
                        number_of_trucks,
                        number_of_loaders,
                        external_scale_load,
                        unload_points,
                        road_delay_percent,
                    )

                    scenario_results.append(summary)
                    all_trips.append(trips_df)


# -----------------------------------------------------------------------------
# 9. Формування синтетичних наборів даних
# -----------------------------------------------------------------------------

scenario_df = pd.DataFrame(scenario_results)
trips_df = pd.concat(all_trips, ignore_index=True)



# -----------------------------------------------------------------------------
# 10. Валідація синтетичних даних
# -----------------------------------------------------------------------------

print("\n" + "=" * 58)
print("Валідація синтетичних даних")
print("=" * 58)

# Списки використовуються для накопичення всіх виявлених проблем.
# Програма не зупиняється після першої помилки, а виконує всі перевірки
# та наприкінці формує повний звіт.
validation_errors = []
validation_warnings = []


# -----------------------------------------------------------------------------
# 10.1. Перевірка пропусків і нескінченних значень
#
# Набори даних не повинні містити порожніх клітинок NaN, а також значень
# positive infinity або negative infinity. Такі значення можуть спричинити
# помилки під час навчання моделей машинного навчання.
# -----------------------------------------------------------------------------

scenario_missing = int(scenario_df.isna().sum().sum())
trips_missing = int(trips_df.isna().sum().sum())

scenario_infinite = int(
    np.isinf(
        scenario_df.select_dtypes(include="number").to_numpy()
    ).sum()
)

trips_infinite = int(
    np.isinf(
        trips_df.select_dtypes(include="number").to_numpy()
    ).sum()
)

if scenario_missing > 0:
    validation_errors.append(
        f"У таблиці сценаріїв виявлено пропуски: {scenario_missing}"
    )

if trips_missing > 0:
    validation_errors.append(
        f"У таблиці рейсів виявлено пропуски: {trips_missing}"
    )

if scenario_infinite > 0:
    validation_errors.append(
        f"У таблиці сценаріїв виявлено нескінченні значення: "
        f"{scenario_infinite}"
    )

if trips_infinite > 0:
    validation_errors.append(
        f"У таблиці рейсів виявлено нескінченні значення: "
        f"{trips_infinite}"
    )

if (
    scenario_missing == 0
    and trips_missing == 0
    and scenario_infinite == 0
    and trips_infinite == 0
):
    print("[OK] Пропуски та нескінченні значення відсутні")
else:
    print("[ERROR] Виявлено пропуски або нескінченні значення")


# -----------------------------------------------------------------------------
# 10.2. Перевірка від'ємних значень
#
# Час технологічних операцій, очікування, маса вантажу, тоннаж та показники
# використання ресурсів не можуть бути від'ємними.
# -----------------------------------------------------------------------------

scenario_non_negative_columns = [
    "total_trips",
    "total_tons",
    "average_cycle_time",
    "avg_travel_time",
    "avg_road_delay_time",
    "avg_wait_tare",
    "avg_wait_loader",
    "avg_wait_gross",
    "avg_wait_unload",
    "max_wait_tare",
    "max_wait_loader",
    "max_wait_gross",
    "max_wait_unload",
    "loader_utilization",
    "scale_internal_utilization",
    "scale_utilization",
    "unload_utilization",
]

trip_non_negative_columns = [
    "cargo_weight",
    "wait_tare",
    "wait_loader",
    "wait_gross",
    "wait_unload",
    "wait_tare_queue",
    "wait_loader_queue",
    "wait_gross_queue",
    "wait_unload_queue",
    "wait_tare_pre_break",
    "wait_loader_pre_break",
    "wait_gross_pre_break",
    "wait_unload_pre_break",
    "wait_tare_break",
    "wait_loader_break",
    "wait_gross_break",
    "wait_unload_break",
    "tare_time",
    "load_time",
    "gross_time",
    "base_travel_time",
    "road_delay_time",
    "travel_time",
    "unload_time",
    "return_time",
    "total_time",
]

negative_values_found = False

for column in scenario_non_negative_columns:
    negative_count = int((scenario_df[column] < 0).sum())

    if negative_count > 0:
        negative_values_found = True
        validation_errors.append(
            f"Стовпець scenario_df['{column}'] містить "
            f"від'ємні значення: {negative_count}"
        )

for column in trip_non_negative_columns:
    negative_count = int((trips_df[column] < 0).sum())

    if negative_count > 0:
        negative_values_found = True
        validation_errors.append(
            f"Стовпець trips_df['{column}'] містить "
            f"від'ємні значення: {negative_count}"
        )

if not negative_values_found:
    print("[OK] Від'ємні значення відсутні")
else:
    print("[ERROR] Виявлено від'ємні значення")


# -----------------------------------------------------------------------------
# 10.3. Перевірка допустимих діапазонів
#
# Перевіряється відповідність значень діапазонам, закладеним у модель:
# маса вантажу, тривалість операцій, параметри сценаріїв і завантаженість
# виробничих ресурсів.
# -----------------------------------------------------------------------------

range_errors_found = False


def check_range(dataframe, column, minimum, maximum, table_name):
    """
    Перевіряє, чи всі значення стовпця перебувають у заданому діапазоні.
    """

    global range_errors_found

    invalid_count = int(
        (
            (dataframe[column] < minimum)
            | (dataframe[column] > maximum)
        ).sum()
    )

    if invalid_count > 0:
        range_errors_found = True
        validation_errors.append(
            f"{table_name}['{column}']: значень поза діапазоном "
            f"{minimum}–{maximum}: {invalid_count}"
        )


# Діапазони технологічних параметрів окремих рейсів.
check_range(trips_df, "cargo_weight", 27, 28, "trips_df")
check_range(trips_df, "load_time", 8, 10, "trips_df")
check_range(trips_df, "base_travel_time", 18, 25, "trips_df")
check_range(trips_df, "unload_time", 12, 15, "trips_df")

# Постійні технологічні параметри.
check_range(trips_df, "tare_time", TARE_TIME, TARE_TIME, "trips_df")
check_range(trips_df, "gross_time", GROSS_TIME, GROSS_TIME, "trips_df")
check_range(trips_df, "return_time", RETURN_TIME, RETURN_TIME, "trips_df")

# Завантаженість ресурсів має перебувати в межах від 0 до 100%.
check_range(
    scenario_df,
    "loader_utilization",
    0,
    100,
    "scenario_df",
)

check_range(
    scenario_df,
    "scale_internal_utilization",
    0,
    100,
    "scenario_df",
)

check_range(
    scenario_df,
    "scale_utilization",
    0,
    100,
    "scenario_df",
)

check_range(
    scenario_df,
    "unload_utilization",
    0,
    100,
    "scenario_df",
)

# Перевірка належності параметрів до визначеного простору сценаріїв.
scenario_value_checks = {
    "number_of_trucks": TRUCK_SCENARIOS,
    "number_of_loaders": LOADER_SCENARIOS,
    "unload_points": UNLOAD_POINT_SCENARIOS,
    "external_scale_load": SCALE_EXTERNAL_LOAD_SCENARIOS,
    "road_delay_percent": ROAD_DELAY_SCENARIOS,
}

for column, allowed_values in scenario_value_checks.items():
    invalid_count = int(
        (~scenario_df[column].isin(allowed_values)).sum()
    )

    if invalid_count > 0:
        range_errors_found = True
        validation_errors.append(
            f"scenario_df['{column}'] містить значення, "
            f"не передбачені простором сценаріїв: {invalid_count}"
        )

if not range_errors_found:
    print("[OK] Значення перебувають у допустимих діапазонах")
else:
    print("[ERROR] Виявлено значення поза допустимими діапазонами")


# -----------------------------------------------------------------------------
# 10.4. Перевірка логічної узгодженості
#
# Перевіряються внутрішні співвідношення між показниками:
# - загальне очікування дорівнює сумі його складових;
# - час дороги дорівнює базовому часу та додатковій затримці;
# - загальний час циклу відповідає сумі операцій і очікувань;
# - кожна конфігурація сценарію є унікальною;
# - кількість сценаріїв відповідає заданому простору параметрів.
# -----------------------------------------------------------------------------

logical_errors_found = False
time_tolerance = 0.11

wait_components = {
    "wait_tare": [
        "wait_tare_queue",
        "wait_tare_pre_break",
        "wait_tare_break",
    ],
    "wait_loader": [
        "wait_loader_queue",
        "wait_loader_pre_break",
        "wait_loader_break",
    ],
    "wait_gross": [
        "wait_gross_queue",
        "wait_gross_pre_break",
        "wait_gross_break",
    ],
    "wait_unload": [
        "wait_unload_queue",
        "wait_unload_pre_break",
        "wait_unload_break",
    ],
}

# Перевірка складових очікування.
for total_column, component_columns in wait_components.items():
    calculated_wait = trips_df[component_columns].sum(axis=1)

    invalid_count = int(
        (
            abs(trips_df[total_column] - calculated_wait)
            > time_tolerance
        ).sum()
    )

    if invalid_count > 0:
        logical_errors_found = True
        validation_errors.append(
            f"Порушено баланс '{total_column}': "
            f"некоректних записів {invalid_count}"
        )

# Перевірка часу руху до причалу.
calculated_travel_time = (
    trips_df["base_travel_time"]
    + trips_df["road_delay_time"]
)

invalid_travel_count = int(
    (
        abs(
            trips_df["travel_time"]
            - calculated_travel_time
        )
        > time_tolerance
    ).sum()
)

if invalid_travel_count > 0:
    logical_errors_found = True
    validation_errors.append(
        f"Порушено баланс часу дороги: "
        f"некоректних записів {invalid_travel_count}"
    )

# Перевірка повного часу транспортного циклу.
# Час повернення не включається, оскільки finish_time фіксується
# після розвантаження та до початку повернення самоскида на склад.
calculated_cycle_time = (
    trips_df["tare_time"]
    + trips_df["load_time"]
    + trips_df["gross_time"]
    + trips_df["travel_time"]
    + trips_df["unload_time"]
    + trips_df["wait_tare"]
    + trips_df["wait_loader"]
    + trips_df["wait_gross"]
    + trips_df["wait_unload"]
)

invalid_cycle_count = int(
    (
        abs(
            trips_df["total_time"]
            - calculated_cycle_time
        )
        > time_tolerance
    ).sum()
)

if invalid_cycle_count > 0:
    logical_errors_found = True
    validation_errors.append(
        f"Порушено баланс повного часу циклу: "
        f"некоректних записів {invalid_cycle_count}"
    )

# Перевірка унікальності конфігурацій сценаріїв.
scenario_key_columns = [
    "number_of_trucks",
    "number_of_loaders",
    "unload_points",
    "external_scale_load",
    "road_delay_percent",
]

duplicate_scenarios = int(
    scenario_df.duplicated(
        subset=scenario_key_columns
    ).sum()
)

if duplicate_scenarios > 0:
    logical_errors_found = True
    validation_errors.append(
        f"Виявлено дублікати виробничих сценаріїв: "
        f"{duplicate_scenarios}"
    )

# Очікувана кількість сценаріїв визначається добутком
# кількості значень усіх сценарних параметрів.
expected_scenarios = (
    len(TRUCK_SCENARIOS)
    * len(LOADER_SCENARIOS)
    * len(SCALE_EXTERNAL_LOAD_SCENARIOS)
    * len(UNLOAD_POINT_SCENARIOS)
    * len(ROAD_DELAY_SCENARIOS)
)

if len(scenario_df) != expected_scenarios:
    logical_errors_found = True
    validation_errors.append(
        f"Кількість сценаріїв не відповідає очікуваній: "
        f"отримано {len(scenario_df)}, "
        f"очікувалося {expected_scenarios}"
    )

# Кожний збережений рейс повинен завершитися до завершення моделювання.
unfinished_trips = int(
    (trips_df["finish_time"] > SIMULATION_END).sum()
)

if unfinished_trips > 0:
    logical_errors_found = True
    validation_errors.append(
        f"Виявлено рейси, завершені після закінчення моделювання: "
        f"{unfinished_trips}"
    )

if not logical_errors_found:
    print("[OK] Логічна узгодженість даних підтверджена")
else:
    print("[ERROR] Виявлено порушення логічної узгодженості")


# -----------------------------------------------------------------------------
# 10.5. Підсумок валідації
#
# Якщо критичних помилок не виявлено, виконання програми продовжується
# і синтетичні набори даних зберігаються у CSV-файли.
# Якщо виявлено хоча б одну критичну помилку, збереження даних скасовується.
# -----------------------------------------------------------------------------

print("-" * 58)
print(f"Записів сценаріїв  : {len(scenario_df)}")
print(f"Записів рейсів     : {len(trips_df)}")
print(f"Помилок валідації  : {len(validation_errors)}")
print(f"Попереджень         : {len(validation_warnings)}")

if validation_errors:
    print("Статус валідації    : FAILED")
    print("-" * 58)

    for error_number, error_message in enumerate(
        validation_errors,
        start=1,
    ):
        print(f"{error_number}. {error_message}")

    print("=" * 58)

    raise ValueError(
        "Валідація синтетичних даних завершилася з помилками. "
        "Файли не збережено."
    )

print("Статус валідації    : PASSED")
print("=" * 58)


# -----------------------------------------------------------------------------
# 11.  Збереження синтетичних наборів даних
# -----------------------------------------------------------------------------
scenario_path = "data/synthetic/scenario_summary.csv"
trips_path = "data/synthetic/trips_all_scenarios.csv"

# Захист від запуску проєкту на новому комп'ютері або після очищення папок.
# Якщо директорії ще немає, вона буде створена автоматично.
os.makedirs("data/synthetic", exist_ok=True)

# Збереження агрегованих результатів сценаріїв.
# Цей файл використовується для навчання моделей ML на рівні сценаріїв.
scenario_df.to_csv(
    scenario_path,
    index=False,
    encoding="utf-8-sig",
)

# Збереження деталізованих результатів рейсів.
# Цей файл використовується для аналізу черг, циклів та поведінки окремих машин.
trips_df.to_csv(
    trips_path,
    index=False,
    encoding="utf-8-sig",
)

# -----------------------------------------------------------------------------
# 12. Контрольний індикатор завершення
# -----------------------------------------------------------------------------

elapsed = time.time() - run_started_at

print("=" * 58)
print("Моделювання успішно завершено")
print("=" * 58)
print(f"Згенеровано сценаріїв : {len(scenario_df)}")
print(f"Згенеровано рейсів    : {len(trips_df)}")
print(f"Час виконання         : {elapsed:.1f} с")
print("Збережено файли:")
print(f"  • {scenario_path}")
print(f"  • {trips_path}")
print("=" * 58)