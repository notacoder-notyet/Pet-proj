"""Telegram-бот: показания датчиков -> предсказание отказа.
Telegram bot: sensor readings -> failure prediction.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from src.dialog_log import log_event
from src.predict import load_model, predict
from src.schemas import SensorReading
from src.sensor_sample import format_sample, random_sensor_sample

load_dotenv()

# Состояния диалога / Conversation states
TYPE, PRODUCT_ID, AIR, PROCESS, SPEED, TORQUE, WEAR = range(7)

TYPE_KEYBOARD = ReplyKeyboardMarkup([["L", "M", "H"]], one_time_keyboard=True, resize_keyboard=True)

# Подсказки: типичные значения AI4I, не жёсткие пределы модели
# Prompts: typical AI4I values, not hard model limits
PROMPT_AIR = "Температура воздуха в Кельвинах (в датасете обычно 295–305 K, это ~22–32 °C). Например 298.1"
PROMPT_PROCESS = "Температура процесса, K (обычно 306–314 K). Например 308.6"
PROMPT_SPEED = "Скорость вращения, rpm (обычно 1200–2200, в данных до ~2900). Например 1551"
PROMPT_TORQUE = "Крутящий момент, Nm (обычно 4–77). Например 42.8"
PROMPT_WEAR = "Износ инструмента, минуты работы (в данных 0–253, среднее ~108). Например 108"

HELP_TEXT = (
    "Предсказание отказа станка по одному JSON-объекту.\n"
    "Отправьте в чат одно сообщение — весь набор полей сразу.\n\n"
    "Обязательные ключи (имена как в датасете):\n"
    "• Product ID — строка, например M14860\n"
    "• Type — L, M или H\n"
    "• Air temperature [K] — число, Кельвин (~295–305, 298 K ≈ 25 °C)\n"
    "• Process temperature [K] — число, Кельвин (~306–314)\n"
    "• Rotational speed [rpm] — число, об/мин (~1200–2200)\n"
    "• Torque [Nm] — число, Н·м (~4–77)\n"
    "• Tool wear [min] — число, минуты работы (0–253)\n\n"
    "Пример (можно скопировать и поменять цифры):\n"
    '{"Product ID": "M14860", "Type": "M", '
    '"Air temperature [K]": 298.1, "Process temperature [K]": 308.6, '
    '"Rotational speed [rpm]": 1551, "Torque [Nm]": 42.8, "Tool wear [min]": 108}\n\n'
    "Правила JSON: двойные кавычки у ключей, точка в дробях, без лишней запятой в конце.\n\n"
    "Другие команды:\n"
    "/predict — те же поля, но по одному сообщению\n"
    "/demo — случайный правдоподобный пример и сразу расчёт\n"
    "/cancel — прервать пошаговый ввод"
)

# Мягкие границы ввода в боте / Soft input bounds in the bot
AIR_RANGE = (250.0, 400.0)
PROCESS_RANGE = (250.0, 400.0)
SPEED_RANGE = (500.0, 5000.0)
TORQUE_RANGE = (0.1, 150.0)
WEAR_RANGE = (0.0, 500.0)


def _format_result(pred: int, prob: float) -> str:
    """Человекочитаемый ответ / Human-readable reply."""
    if pred == 1:
        status = "ОТКАЗ (класс 1)"
        hint = "Модель видит высокий риск. Имеет смысл проверить станок."
    else:
        status = "Норма (класс 0)"
        hint = "Отказ не предсказан. Это не гарантия, только оценка модели."
    bar_len = 10
    filled = min(bar_len, max(0, round(prob * bar_len)))
    bar = "█" * filled + "░" * (bar_len - filled)
    return (
        f"{status}\n"
        f"Вероятность отказа: {prob:.1%}  [{bar}]\n"
        f"{hint}"
    )


def _parse_float(text: str) -> float:
    return float(text.strip().replace(",", "."))


def _parse_in_range(text: str, low: float, high: float, field_ru: str) -> float:
    value = _parse_float(text)
    if not (low <= value <= high):
        raise ValueError(
            f"{field_ru}: {value} вне допустимого диапазона {low:g}–{high:g}. "
            "Проверьте единицы (температура — Кельвин, не Цельсий и не rpm)."
        )
    return value


def _friendly_error(exc: Exception) -> str:
    """Короткое сообщение без traceback и ссылок Pydantic / Short message without Pydantic URLs."""
    text = str(exc)
    if "validation error" in text.lower() or "ValidationError" in type(exc).__name__:
        return (
            "Не получилось принять значения. Частые причины:\n"
            "• температура должна быть в Кельвинах (около 298, не 25 и не 12000)\n"
            "• Type — только L, M или H\n"
            "• все поля — числа, кроме Type и Product ID\n"
            f"Детали: {text.splitlines()[0]}"
        )
    return f"Ошибка: {text}"


def _infer_local(raw: dict) -> tuple[int, float]:
    reading = SensorReading.model_validate(raw)
    artifact = load_model()
    return predict(reading.to_raw_dict(), artifact=artifact)


def _infer_via_api(raw: dict, api_url: str) -> tuple[int, float]:
    reading = SensorReading.model_validate(raw)
    url = api_url.rstrip("/") + "/predict"
    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, json=reading.model_dump(by_alias=True))
        response.raise_for_status()
        data = response.json()
    return int(data["prediction"]), float(data["probability"])


def run_inference(raw: dict) -> tuple[int, float]:
    api_url = os.getenv("API_URL", "").strip()
    if api_url:
        return _infer_via_api(raw, api_url)
    return _infer_local(raw)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    log_event("command_start", update)
    await update.message.reply_text("Бот предиктивного обслуживания.\n\n" + HELP_TEXT)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    log_event("command_help", update)
    await update.message.reply_text(HELP_TEXT)


async def demo_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    sample = random_sensor_sample()
    log_event("command_demo", update, extra={"sample": sample})
    await update.message.reply_text("Случайный правдоподобный набор (как в датасете AI4I):\n\n" + format_sample(sample))
    try:
        pred, prob = run_inference(sample)
        log_event("demo_result", update, extra={"prediction": pred, "probability": prob, "sample": sample})
        await update.message.reply_text(_format_result(pred, prob))
    except Exception as exc:
        log_event("demo_error", update, extra={"error": str(exc), "sample": sample})
        await update.message.reply_text(_friendly_error(exc))


async def predict_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    log_event("predict_start", update)
    await update.message.reply_text(
        "Тип продукта: L (low), M (medium) или H (high).",
        reply_markup=TYPE_KEYBOARD,
    )
    return TYPE


async def got_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    value = (update.message.text or "").strip().upper()
    if value not in {"L", "M", "H"}:
        await update.message.reply_text("Нужно L, M или H.", reply_markup=TYPE_KEYBOARD)
        return TYPE
    context.user_data["Type"] = value
    await update.message.reply_text(
        "Product ID, например M14860. Можно «-»: сгенерирую из типа (на предсказание почти не влияет).",
        reply_markup=ReplyKeyboardRemove(),
    )
    return PRODUCT_ID


async def got_product_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = (update.message.text or "").strip()
    type_code = context.user_data.get("Type", "M")
    if text in {"-", "—", ""}:
        text = f"{type_code}{10000 + (update.effective_user.id % 19999)}"
        await update.message.reply_text(f"Product ID: {text}")
    context.user_data["Product ID"] = text
    await update.message.reply_text(PROMPT_AIR)
    return AIR


async def got_air(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        context.user_data["Air temperature [K]"] = _parse_in_range(
            update.message.text or "", *AIR_RANGE, "Температура воздуха"
        )
    except ValueError as exc:
        await update.message.reply_text(str(exc) + "\n" + PROMPT_AIR)
        return AIR
    await update.message.reply_text(PROMPT_PROCESS)
    return PROCESS


async def got_process(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        context.user_data["Process temperature [K]"] = _parse_in_range(
            update.message.text or "", *PROCESS_RANGE, "Температура процесса"
        )
    except ValueError as exc:
        await update.message.reply_text(str(exc) + "\n" + PROMPT_PROCESS)
        return PROCESS
    await update.message.reply_text(PROMPT_SPEED)
    return SPEED


async def got_speed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        context.user_data["Rotational speed [rpm]"] = _parse_in_range(
            update.message.text or "", *SPEED_RANGE, "Скорость"
        )
    except ValueError as exc:
        await update.message.reply_text(str(exc) + "\n" + PROMPT_SPEED)
        return SPEED
    await update.message.reply_text(PROMPT_TORQUE)
    return TORQUE


async def got_torque(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        context.user_data["Torque [Nm]"] = _parse_in_range(
            update.message.text or "", *TORQUE_RANGE, "Крутящий момент"
        )
    except ValueError as exc:
        await update.message.reply_text(str(exc) + "\n" + PROMPT_TORQUE)
        return TORQUE
    await update.message.reply_text(PROMPT_WEAR)
    return WEAR


async def got_wear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        context.user_data["Tool wear [min]"] = _parse_in_range(
            update.message.text or "", *WEAR_RANGE, "Износ"
        )
    except ValueError as exc:
        await update.message.reply_text(str(exc) + "\n" + PROMPT_WEAR)
        return WEAR
    raw: dict[str, Any] = dict(context.user_data)
    log_event("predict_input", update, extra={"input": raw})
    await update.message.reply_text("Считаю вероятность отказа...")
    try:
        pred, prob = run_inference(raw)
        log_event("predict_result", update, extra={"input": raw, "prediction": pred, "probability": prob})
        await update.message.reply_text(_format_result(pred, prob))
    except Exception as exc:
        log_event("predict_error", update, extra={"input": raw, "error": str(exc)})
        await update.message.reply_text(_friendly_error(exc))
    context.user_data.clear()
    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    log_event("predict_cancel", update)
    context.user_data.clear()
    await update.message.reply_text("Отменено.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END


async def json_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()
    if not text.startswith("{"):
        log_event("unknown_text", update)
        await update.message.reply_text("Не понял сообщение. Команды: /help, /predict, /demo.")
        return
    try:
        raw = json.loads(text)
        log_event("json_input", update, extra={"input": raw})
        pred, prob = run_inference(raw)
        log_event("json_result", update, extra={"input": raw, "prediction": pred, "probability": prob})
        await update.message.reply_text(_format_result(pred, prob))
    except json.JSONDecodeError:
        log_event("json_parse_error", update)
        await update.message.reply_text("Не получилось прочитать JSON. Проверьте кавычки и запятые.")
    except Exception as exc:
        log_event("json_error", update, extra={"error": str(exc)})
        await update.message.reply_text(_friendly_error(exc))


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "Задайте TELEGRAM_BOT_TOKEN в .env (BotFather -> /newbot).\n"
            "Set TELEGRAM_BOT_TOKEN in .env (BotFather -> /newbot)."
        )

    app = Application.builder().token(token).build()
    conv = ConversationHandler(
        entry_points=[CommandHandler("predict", predict_start)],
        states={
            TYPE: [MessageHandler(filters.TEXT & ~filters.COMMAND, got_type)],
            PRODUCT_ID: [MessageHandler(filters.TEXT & ~filters.COMMAND, got_product_id)],
            AIR: [MessageHandler(filters.TEXT & ~filters.COMMAND, got_air)],
            PROCESS: [MessageHandler(filters.TEXT & ~filters.COMMAND, got_process)],
            SPEED: [MessageHandler(filters.TEXT & ~filters.COMMAND, got_speed)],
            TORQUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, got_torque)],
            WEAR: [MessageHandler(filters.TEXT & ~filters.COMMAND, got_wear)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("demo", demo_cmd))
    app.add_handler(conv)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, json_message))

    print("Telegram-бот запущен. Логи: logs/telegram_dialogs.jsonl")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
