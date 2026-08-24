"""Правдоподобные случайные показания датчиков / Plausible random sensor samples."""

from __future__ import annotations

import random

# Диапазоны из AI4I 2020 (train), слегка расширены / Dataset ranges, slightly widened
AIR_K = (295.3, 304.5)
PROCESS_K = (305.7, 313.8)
SPEED_RPM = (1168.0, 2886.0)
TORQUE_NM = (3.8, 76.6)
WEAR_MIN = (0.0, 253.0)
TYPE_WEIGHTS = [("L", 0.60), ("M", 0.30), ("H", 0.10)]


def _uniform(bounds: tuple[float, float], ndigits: int) -> float:
    return round(random.uniform(*bounds), ndigits)


def random_sensor_sample() -> dict:
    """Случайный, но реалистичный набор полей как в CSV / Random but realistic CSV-like row."""
    type_code = random.choices(
        [t for t, _ in TYPE_WEIGHTS],
        weights=[w for _, w in TYPE_WEIGHTS],
        k=1,
    )[0]
    serial = random.randint(10000, 29999)
    return {
        "Product ID": f"{type_code}{serial}",
        "Type": type_code,
        "Air temperature [K]": _uniform(AIR_K, 1),
        "Process temperature [K]": _uniform(PROCESS_K, 1),
        "Rotational speed [rpm]": round(random.uniform(*SPEED_RPM)),
        "Torque [Nm]": _uniform(TORQUE_NM, 1),
        "Tool wear [min]": round(random.uniform(*WEAR_MIN)),
    }


def format_sample(sample: dict) -> str:
    """Текст с параметрами для чата / Human-readable parameter dump."""
    return (
        f"Type: {sample['Type']}\n"
        f"Product ID: {sample['Product ID']}\n"
        f"Air temperature: {sample['Air temperature [K]']} K\n"
        f"Process temperature: {sample['Process temperature [K]']} K\n"
        f"Rotational speed: {sample['Rotational speed [rpm]']} rpm\n"
        f"Torque: {sample['Torque [Nm]']} Nm\n"
        f"Tool wear: {sample['Tool wear [min]']} min"
    )
