from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter
from typing import Dict, Any, List
from datetime import datetime
from schemas import SensorReading, PredictionRequest, TrainingRequest, ModelPrediction, EnsemblePrediction
from ml_pipeline import AdvancedSensorMLPipeline

router = APIRouter()

# Initialize ML Pipeline
ml_pipeline = AdvancedSensorMLPipeline()

@router.on_event("startup")
async def startup_event():
    """Initialize with comprehensive model training"""
    print("Initializing Advanced Sensor ML Pipeline...")
    
    # Train models in background
    metrics = ml_pipeline.train_all_models(n_samples=3000)
    
    print("="*60)
    print("MODEL TRAINING RESULTS")
    print("="*60)
    for model_name, metric in metrics.items():
        print(f"{model_name.upper()}:")
        print(f"  Accuracy: {metric['accuracy']:.4f}")
        if metric['auc_score']:
            print(f"  AUC Score: {metric['auc_score']:.4f}")
        if metric['cv_mean']:
            print(f"  CV Score: {metric['cv_mean']:.4f} (+/- {metric['cv_std']*2:.4f})")
        print()
    
    print("Advanced ML Pipeline ready!")

@router.get("/")
async def root():
    return {
        "message": "Advanced Sensor Data ML API",
        "status": "running",
        "models_trained": ml_pipeline.is_trained,
        "available_models": list(ml_pipeline.models.keys()) if ml_pipeline.is_trained else []
    }

@router.post("/sensors/readings", response_model=Dict[str, Any])
async def add_sensor_reading(reading: SensorReading):
    """Add a new sensor reading"""
    sensor_data.append(reading)
    return {
        "message": "Reading added successfully",
        "id": len(sensor_data),
        "timestamp": reading.timestamp
    }

@router.get("/sensors/readings")
async def get_sensor_readings(limit: int = 100):
    """Get recent sensor readings"""
    return sensor_data[-limit:]

@router.get("/sensors/readings/{sensor_id}")
async def get_sensor_readings_by_id(sensor_id: str, limit: int = 100):
    """Get readings for a specific sensor"""
    filtered_data = [r for r in sensor_data if r.sensor_id == sensor_id]
    return filtered_data[-limit:]

@router.post("/predict/anomaly", response_model=EnsemblePrediction)
async def predict_anomaly(request: PredictionRequest):
    """Advanced ensemble anomaly prediction"""
    if not ml_pipeline.is_trained:
        raise HTTPException(status_code=400, detail="Models not trained yet")
    
    try:
        # Prepare features (same order as training)
        features = [
            request.temperature,
            request.humidity,
            request.pressure,
            request.vibration,
            request.power_consumption
        ]
        
        # Calculate engineered features
        temp_humidity_ratio = request.temperature / max(request.humidity, 1)
        pressure_normalized = (request.pressure - 1013) / 50
        vibration_power_correlation = request.vibration * request.power_consumption / 100
        thermal_efficiency = request.power_consumption / max(request.temperature, 1)
        stability_index = 1 / (1 + request.vibration + abs(pressure_normalized))
        
        # Add engineered features
        features.extend([
            temp_humidity_ratio,
            pressure_normalized,
            vibration_power_correlation,
            thermal_efficiency,
            stability_index
        ])
        
        # Add sensor type encoding (dummy implementation)
        sensor_type_features = [0, 0, 0, 0]  # Default to industrial
        if request.sensor_type == 'environmental':
            sensor_type_features[0] = 1
        elif request.sensor_type == 'industrial':
            sensor_type_features[1] = 1
        elif request.sensor_type == 'mechanical':
            sensor_type_features[2] = 1
        
        features.extend(sensor_type_features)
        
        # Make ensemble prediction
        prediction = ml_pipeline.ensemble_predict(features)
        
        return prediction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/train/models")
async def train_models(background_tasks: BackgroundTasks, request: TrainingRequest):
    """Train or retrain models"""
    
    def train_background():
        try:
            metrics = ml_pipeline.train_all_models(n_samples=request.n_samples)
            if "model_performance_history" not in globals():
                global model_performance_history
                model_performance_history = []
            model_performance_history.append({
                'timestamp': datetime.now(),
                'metrics': metrics,
                'n_samples': request.n_samples
            })
        except Exception as e:
            print(f"Background training error: {str(e)}")
    
    background_tasks.add_task(train_background)
    return {
        "message": "Model training started in the background.",
        "n_samples": request.n_samples,
        "retrain": request.retrain
    }