from fastapi import BackgroundTasks
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional, Dict

# Data models
class SensorReading(BaseModel):
    timestamp: datetime
    sensor_id: str
    sensor_type: Optional[str] = "industrial"
    temperature: float
    humidity: float
    pressure: float
    vibration: float
    power_consumption: float
    status: Optional[str] = "normal"

class PredictionRequest(BaseModel):
    temperature: float
    humidity: float
    pressure: float
    vibration: float
    power_consumption: float
    sensor_type: Optional[str] = "industrial"

class TrainingRequest(BaseModel):
    model_types: Optional[List[str]] = ["all"]
    n_samples: Optional[int] = 5000
    retrain: Optional[bool] = True

class ModelPrediction(BaseModel):
    model_name: str
    is_anomaly: bool
    confidence: float
    predicted_status: str
    anomaly_score: Optional[float] = None

class EnsemblePrediction(BaseModel):
    ensemble_prediction: bool
    confidence: float
    individual_predictions: List[ModelPrediction]
    feature_importance: Optional[Dict[str, float]] = None