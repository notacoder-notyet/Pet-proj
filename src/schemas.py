"""Схема входных показаний датчиков / Schema for raw sensor readings."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class SensorReading(BaseModel):
    """Сырые поля как в CSV датасета / Raw fields matching the dataset CSV."""

    model_config = ConfigDict(populate_by_name=True)

    udi: Optional[int] = Field(default=None, alias="UDI")
    product_id: str = Field(alias="Product ID", min_length=2, examples=["M14860"])
    type: Literal["L", "M", "H"] = Field(alias="Type")
    air_temperature_k: float = Field(alias="Air temperature [K]")
    process_temperature_k: float = Field(alias="Process temperature [K]")
    rotational_speed_rpm: float = Field(alias="Rotational speed [rpm]", gt=0)
    torque_nm: float = Field(alias="Torque [Nm]")
    tool_wear_min: float = Field(alias="Tool wear [min]", ge=0)

    def to_raw_dict(self) -> dict:
        """Словарь в формате ноутбука / Dict in notebook/CSV format."""
        payload = {
            "Product ID": self.product_id,
            "Type": self.type,
            "Air temperature [K]": self.air_temperature_k,
            "Process temperature [K]": self.process_temperature_k,
            "Rotational speed [rpm]": self.rotational_speed_rpm,
            "Torque [Nm]": self.torque_nm,
            "Tool wear [min]": self.tool_wear_min,
        }
        if self.udi is not None:
            payload["UDI"] = self.udi
        return payload


class PredictionResponse(BaseModel):
    """Ответ модели / Model response."""

    prediction: int = Field(description="0 = нет отказа, 1 = отказ / 0 = no failure, 1 = failure")
    probability: float = Field(description="Вероятность отказа / Failure probability")
    label: str
