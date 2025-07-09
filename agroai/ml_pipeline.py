import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
from schemas import SensorReading, ModelPrediction, EnsemblePrediction
import warnings
warnings.filterwarnings('ignore')


# In-memory storage
sensor_data = []
model_performance_history = []

class AdvancedSensorMLPipeline:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoder = LabelEncoder()
        self.pca = PCA(n_components=0.95)
        self.kmeans = KMeans(n_clusters=4, random_state=42)
        self.feature_names = []
        self.is_trained = False
        self.training_history = {}
        self.model_metrics = {}
        
    def generate_comprehensive_data(self, n_samples=5000):
        """Generate realistic sensor data with multiple patterns"""
        np.random.seed(42)
        data = []
        
        sensor_types = ['industrial', 'environmental', 'mechanical', 'electrical']
        
        for i in range(n_samples):
            # Time-based patterns
            time_factor = np.sin(i / 100) * 0.1  # Daily cycles
            seasonal_factor = np.sin(i / 1000) * 0.05  # Seasonal variations
            
            # Choose sensor type
            sensor_type = np.random.choice(sensor_types)
            
            # Base parameters by sensor type
            sensor_configs = {
                'industrial': {
                    'temp': (45, 8), 'humidity': (40, 15), 'pressure': (1020, 30),
                    'vibration': (0.2, 0.1), 'power': (150, 50)
                },
                'environmental': {
                    'temp': (22, 5), 'humidity': (65, 20), 'pressure': (1013, 25),
                    'vibration': (0.05, 0.03), 'power': (80, 30)
                },
                'mechanical': {
                    'temp': (35, 10), 'humidity': (50, 25), 'pressure': (1000, 40),
                    'vibration': (0.3, 0.15), 'power': (200, 75)
                },
                'electrical': {
                    'temp': (30, 6), 'humidity': (55, 18), 'pressure': (1015, 20),
                    'vibration': (0.1, 0.05), 'power': (120, 40)
                }
            }
            
            config = sensor_configs[sensor_type]
            
            # Generate anomalies with different patterns
            anomaly_prob = 0.12 if sensor_type == 'industrial' else 0.08
            is_anomaly = np.random.random() < anomaly_prob
            
            if is_anomaly:
                anomaly_types = ['spike', 'drift', 'noise', 'correlation', 'degradation']
                anomaly_type = np.random.choice(anomaly_types)
                
                if anomaly_type == 'spike':
                    multipliers = (1.8, 1.5, 0.7, 3.0, 2.0)
                elif anomaly_type == 'drift':
                    multipliers = (1.3, 1.2, 0.85, 1.8, 1.6)
                elif anomaly_type == 'noise':
                    multipliers = (1.0, 1.0, 1.0, 4.0, 1.0)
                elif anomaly_type == 'correlation':
                    multipliers = (0.6, 1.6, 1.3, 1.5, 0.8)
                else:  # degradation
                    multipliers = (1.2, 1.1, 0.9, 2.0, 1.4)
            else:
                multipliers = (1.0, 1.0, 1.0, 1.0, 1.0)
            
            # Generate sensor values
            temp_base, temp_std = config['temp']
            humidity_base, humidity_std = config['humidity']
            pressure_base, pressure_std = config['pressure']
            vibration_base, vibration_std = config['vibration']
            power_base, power_std = config['power']
            
            temperature = np.random.normal(
                temp_base * multipliers[0], 
                temp_std * (1.5 if is_anomaly else 1.0)
            ) + time_factor * 5 + seasonal_factor * 3
            
            humidity = np.random.normal(
                humidity_base * multipliers[1], 
                humidity_std * (1.3 if is_anomaly else 1.0)
            ) + time_factor * 10 + seasonal_factor * 5
            
            pressure = np.random.normal(
                pressure_base * multipliers[2], 
                pressure_std * (1.2 if is_anomaly else 1.0)
            ) + time_factor * 15 + seasonal_factor * 8
            
            vibration = np.random.normal(
                vibration_base * multipliers[3], 
                vibration_std * (2.0 if is_anomaly else 1.0)
            ) + time_factor * 0.02
            
            power_consumption = np.random.normal(
                power_base * multipliers[4], 
                power_std * (1.4 if is_anomaly else 1.0)
            ) + time_factor * 20 + seasonal_factor * 10
            
            # Advanced feature engineering
            temp_humidity_ratio = temperature / max(humidity, 1)
            pressure_normalized = (pressure - 1013) / 50
            vibration_power_correlation = vibration * power_consumption / 100
            thermal_efficiency = power_consumption / max(temperature, 1)
            stability_index = 1 / (1 + vibration + abs(pressure_normalized))
            
            # Determine status
            if is_anomaly:
                if multipliers[0] > 1.5 or multipliers[3] > 2.5:
                    status = 'critical'
                elif multipliers[0] > 1.2 or multipliers[3] > 1.5:
                    status = 'warning'
                else:
                    status = 'maintenance_required'
            else:
                status = 'normal'
            
            data.append({
                'sensor_id': f'sensor_{i % 25}',
                'sensor_type': sensor_type,
                'temperature': temperature,
                'humidity': humidity,
                'pressure': pressure,
                'vibration': vibration,
                'power_consumption': power_consumption,
                'temp_humidity_ratio': temp_humidity_ratio,
                'pressure_normalized': pressure_normalized,
                'vibration_power_correlation': vibration_power_correlation,
                'thermal_efficiency': thermal_efficiency,
                'stability_index': stability_index,
                'status': status,
                'is_anomaly': is_anomaly,
                'timestamp': datetime.now() - timedelta(minutes=n_samples-i)
            })
        
        return pd.DataFrame(data)
    
    def prepare_features(self, df):
        """Prepare comprehensive feature matrix"""
        numeric_features = [
            'temperature', 'humidity', 'pressure', 'vibration', 'power_consumption',
            'temp_humidity_ratio', 'pressure_normalized', 'vibration_power_correlation',
            'thermal_efficiency', 'stability_index'
        ]
        
        X = df[numeric_features].copy()
        
        # Handle sensor type encoding
        if 'sensor_type' in df.columns:
            sensor_dummies = pd.get_dummies(df['sensor_type'], prefix='sensor_type')
            X = pd.concat([X, sensor_dummies], axis=1)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return X.values, self.feature_names
    
    def train_classical_models(self, X_train, X_test, y_train, y_test):
        """Train multiple classical ML models with hyperparameter tuning"""
        models_config = {
            'random_forest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150, 
                learning_rate=0.1, 
                max_depth=8,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf', 
                C=1.0, 
                gamma='scale',
                probability=True, 
                random_state=42
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=1000,
                random_state=42
            ),
            'isolation_forest': IsolationForest(
                contamination=0.12, 
                random_state=42,
                n_estimators=150
            )
        }
        
        results = {}
        
        for name, model in models_config.items():
            print(f"Training {name}...")
            
            try:
                if name == 'isolation_forest':
                    model.fit(X_train)
                    y_pred = model.predict(X_test)
                    y_pred = (y_pred == -1).astype(int)
                    accuracy = accuracy_score(y_test, y_pred)
                    auc_score = None
                    probabilities = None
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    probabilities = model.predict_proba(X_test)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    auc_score = roc_auc_score(y_test, probabilities[:, 1])
                
                # Cross-validation score
                if name != 'isolation_forest':
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                else:
                    cv_mean = cv_std = None
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'auc_score': auc_score,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'predictions': y_pred,
                    'probabilities': probabilities,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                print(f"  {name} - Accuracy: {accuracy:.4f}")
                if auc_score:
                    print(f"  {name} - AUC: {auc_score:.4f}")
                if cv_mean:
                    print(f"  {name} - CV Score: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
                    
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        return results
    
    def build_advanced_neural_network(self, input_shape):
        """Build advanced neural network with regularization"""
        model = Sequential([
            Dense(256, activation='relu', input_shape=(input_shape,)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def build_lstm_model(self, timesteps, features):
        """Build LSTM for time series anomaly detection"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(timesteps, features)),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_sequences(self, data, timesteps=15):
        """Create sequences for LSTM training"""
        X_seq, y_seq = [], []
        
        for i in range(len(data) - timesteps):
            X_seq.append(data[i:(i + timesteps)])
            y_seq.append(data[i + timesteps])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_deep_models(self, X_train, X_test, y_train, y_test):
        """Train deep learning models"""
        results = {}
        
        # Advanced Neural Network
        print("Training Advanced Neural Network...")
        try:
            nn_model = self.build_advanced_neural_network(X_train.shape[1])
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            history = nn_model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=64,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
            nn_pred_proba = nn_model.predict(X_test)
            nn_pred = (nn_pred_proba > 0.5).astype(int).flatten()
            nn_accuracy = accuracy_score(y_test, nn_pred)
            nn_auc = roc_auc_score(y_test, nn_pred_proba)
            
            results['neural_network'] = {
                'model': nn_model,
                'accuracy': nn_accuracy,
                'auc_score': nn_auc,
                'predictions': nn_pred,
                'probabilities': nn_pred_proba,
                'history': history.history
            }
            
            print(f"  Neural Network - Accuracy: {nn_accuracy:.4f}, AUC: {nn_auc:.4f}")
            
        except Exception as e:
            print(f"Error training Neural Network: {str(e)}")
        
        # LSTM Model
        print("Training LSTM...")
        try:
            timesteps = 15
            
            # Prepare sequences
            combined_data = np.column_stack([X_train, y_train.reshape(-1, 1)])
            X_train_seq, y_train_seq = self.create_sequences(combined_data, timesteps)
            
            combined_test = np.column_stack([X_test, y_test.reshape(-1, 1)])
            X_test_seq, y_test_seq = self.create_sequences(combined_test, timesteps)
            
            if len(X_train_seq) > 0 and len(X_test_seq) > 0:
                X_train_lstm = X_train_seq[:, :, :-1]
                y_train_lstm = y_train_seq[:, -1]
                X_test_lstm = X_test_seq[:, :, :-1]
                y_test_lstm = y_test_seq[:, -1]
                
                lstm_model = self.build_lstm_model(timesteps, X_train_lstm.shape[2])
                
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True
                )
                
                lstm_history = lstm_model.fit(
                    X_train_lstm, y_train_lstm,
                    epochs=80,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                lstm_pred_proba = lstm_model.predict(X_test_lstm)
                lstm_pred = (lstm_pred_proba > 0.5).astype(int).flatten()
                lstm_accuracy = accuracy_score(y_test_lstm, lstm_pred)
                lstm_auc = roc_auc_score(y_test_lstm, lstm_pred_proba)
                
                results['lstm'] = {
                    'model': lstm_model,
                    'accuracy': lstm_accuracy,
                    'auc_score': lstm_auc,
                    'predictions': lstm_pred,
                    'probabilities': lstm_pred_proba,
                    'history': lstm_history.history,
                    'timesteps': timesteps
                }
                
                print(f"  LSTM - Accuracy: {lstm_accuracy:.4f}, AUC: {lstm_auc:.4f}")
                
        except Exception as e:
            print(f"Error training LSTM: {str(e)}")
        
        return results
    
    def get_feature_importance(self, model, model_name):
        """Extract feature importance for different model types"""
        if hasattr(model, 'feature_importances_'):
            return dict(zip(self.feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_') and model_name == 'svm':
            return dict(zip(self.feature_names, abs(model.coef_[0])))
        else:
            return None
    
    def ensemble_predict(self, features):
        """Make ensemble predictions using all trained models"""
        if not self.is_trained:
            raise ValueError("Models not trained yet")
        
        # Prepare features
        feature_array = np.array(features).reshape(1, -1)
        feature_scaled = self.scalers['main'].transform(feature_array)
        
        predictions = []
        ensemble_votes = []
        
        for model_name, model_data in self.models.items():
            try:
                model = model_data['model']
                
                if model_name == 'isolation_forest':
                    # Isolation Forest returns -1 for anomalies, 1 for normal
                    pred = model.predict(feature_scaled)[0]
                    is_anomaly = pred == -1
                    confidence = abs(model.decision_function(feature_scaled)[0])
                    anomaly_score = model.decision_function(feature_scaled)[0]
                    
                elif model_name in ['neural_network', 'lstm']:
                    # Deep learning models
                    if model_name == 'lstm':
                        timesteps = model_data.get('timesteps', 15)
                        if len(sensor_data) >= timesteps:
                            # Use recent data for LSTM
                            recent_data = []
                            for i in range(timesteps):
                                if i < len(sensor_data):
                                    reading = sensor_data[-(i+1)]
                                    recent_features = [
                                        reading.temperature, reading.humidity, reading.pressure,
                                        reading.vibration, reading.power_consumption
                                    ]
                                    recent_data.append(recent_features)
                            
                            if len(recent_data) == timesteps:
                                lstm_input = np.array(recent_data).reshape(1, timesteps, -1)
                                pred_proba = model.predict(lstm_input)[0][0]
                            else:
                                pred_proba = 0.5  # Default if not enough data
                        else:
                            pred_proba = 0.5  # Default if not enough data
                    else:
                        pred_proba = model.predict(feature_scaled)[0][0]
                    
                    is_anomaly = pred_proba > 0.5
                    confidence = abs(pred_proba - 0.5) * 2
                    anomaly_score = pred_proba
                    
                else:
                    # Classical ML models
                    pred = model.predict(feature_scaled)[0]
                    pred_proba = model.predict_proba(feature_scaled)[0]
                    
                    is_anomaly = bool(pred)
                    confidence = max(pred_proba)
                    anomaly_score = pred_proba[1]
                
                # Determine status
                if is_anomaly:
                    if confidence > 0.8:
                        status = 'critical'
                    elif confidence > 0.6:
                        status = 'warning'
                    else:
                        status = 'maintenance_required'
                else:
                    status = 'normal'
                
                predictions.append(ModelPrediction(
                    model_name=model_name,
                    is_anomaly=is_anomaly,
                    confidence=confidence,
                    predicted_status=status,
                    anomaly_score=anomaly_score
                ))
                
                ensemble_votes.append(1 if is_anomaly else 0)
                
            except Exception as e:
                print(f"Error in prediction for {model_name}: {str(e)}")
                continue
        
        # Ensemble decision (majority vote with confidence weighting)
        if ensemble_votes:
            weighted_votes = []
            for i, pred in enumerate(predictions):
                weight = pred.confidence
                vote = 1 if pred.is_anomaly else 0
                weighted_votes.append(vote * weight)
            
            ensemble_decision = sum(weighted_votes) / sum(pred.confidence for pred in predictions)
            ensemble_prediction = ensemble_decision > 0.5
            ensemble_confidence = abs(ensemble_decision - 0.5) * 2
        else:
            ensemble_prediction = False
            ensemble_confidence = 0.0
        
        # Feature importance from best performing model
        best_model = max(self.models.items(), key=lambda x: x[1].get('accuracy', 0))
        feature_importance = self.get_feature_importance(best_model[1]['model'], best_model[0])
        
        return EnsemblePrediction(
            ensemble_prediction=ensemble_prediction,
            confidence=ensemble_confidence,
            individual_predictions=predictions,
            feature_importance=feature_importance
        )
    
    def train_all_models(self, n_samples=5000):
        """Train all models with comprehensive pipeline"""
        print(f"Starting comprehensive model training with {n_samples} samples...")
        
        # Generate training data
        df = self.generate_comprehensive_data(n_samples)
        
        # Prepare features
        X, feature_names = self.prepare_features(df)
        y = df['is_anomaly'].astype(int).values
        
        # Scale features
        self.scalers['main'] = StandardScaler()
        X_scaled = self.scalers['main'].fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train classical models
        classical_results = self.train_classical_models(X_train, X_test, y_train, y_test)
        
        # Train deep learning models
        deep_results = self.train_deep_models(X_train, X_test, y_train, y_test)
        
        # Combine results
        self.models = {**classical_results, **deep_results}
        
        # Calculate overall metrics
        self.model_metrics = {}
        for model_name, result in self.models.items():
            self.model_metrics[model_name] = {
                'accuracy': result['accuracy'],
                'auc_score': result.get('auc_score'),
                'cv_mean': result.get('cv_mean'),
                'cv_std': result.get('cv_std')
            }
        
        self.is_trained = True
        
        # Store training history
        self.training_history = {
            'timestamp': datetime.now(),
            'n_samples': n_samples,
            'n_features': X.shape[1],
            'metrics': self.model_metrics
        }
        
        print("Model training completed successfully!")
        return self.model_metrics
