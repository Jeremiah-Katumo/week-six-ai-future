import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Activity, AlertTriangle, TrendingUp, Database, Zap, Thermometer } from 'lucide-react';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL;

function SensorDashboard() {
    const [sensorData, setSensorData] = useState([]);
    const [analytics, setAnalytics] = useState(null);
    const [modelStatus, setModelStatus] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [predictionForm, setPredictionForm] = useState({
        temperature: 25,
        humidity: 60,
        pressure: 1013,
        vibration: 0.1,
        power_consumption: 100
    });

    useEffect(() => {
        fetchSensorData();
        fetchAnalytics();
        fetchModelStatus();
    }, []);

    const fetchSensorData = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/sensors/readings?limit=50`);
            const data = await response.json();
            setSensorData(data);
        } catch (error) {
            console.error('Error fetching sensor data:', error);
        }
    };

    const fetchAnalytics = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/sensors/analytics`);
            const data = await response.json();
            setAnalytics(data);
        } catch (error) {
            console.error('Error fetching analytics:', error);
        }
    };

    const fetchModelStatus = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/model/status`);
            const data = await response.json();
            setModelStatus(data);
        } catch (error) {
            console.error('Error fetching model status:', error);
        }
    };

    const handlePrediction = async () => {
        setLoading(true);

        try {
            const response = await fetch(`${API_BASE_URL}/predict/anomaly`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(predictionForm),
            });

            const data = await response.json();
            setPrediction(data);
        } catch (error) {
            console.error('Error making prediction:', error);
        } finally {
            setLoading(false);
        }
    };

    const addSampleReading = async () => {
        const sampleReading = {
            timestamp: new Date().toISOString(),
            sensor_id: `sensor_${Math.floor(Math.random() * 10)}`,
            temperature: 20 + Math.random() * 10,
            humidity: 50 + Math.random() * 20,
            pressure: 1000 + Math.random() * 50,
            vibration: 0.05 + Math.random() * 0.1,
            power_consumption: 80 + Math.random() * 40,
            status: Math.random() > 0.8 ? 'anomaly' : 'normal'
        };

        try {
            await fetch(`${API_BASE_URL}/sensors/readings`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(sampleReading),
            });

            fetchSensorData();
            fetchAnalytics();
        } catch (error) {
            console.error('Error adding reading:', error);
        }
    };

    const chartData = sensorData.map((reading, index) => ({
        index,
        temperature: reading.temperature,
        humidity: reading.humidity,
        pressure: reading.pressure / 10, // Scale for visibility
        vibration: reading.vibration * 100, // Scale for visibility
        power: reading.power_consumption,
        status: reading.status === 'anomaly' ? 1 : 0
    }));

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                            <Activity className="h-8 w-8 text-blue-600" />
                            <h1 className="text-3xl font-bold text-gray-800">Sensor Data Dashboard</h1>
                        </div>
                        <button
                            onClick={addSampleReading}
                            className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
                        >
                            Add Sample Reading
                        </button>
                    </div>
                </div>

                {/* Analytics Cards */}
                {analytics && (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
                        <div className="bg-white rounded-lg shadow-lg p-6">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-gray-600">Total Readings</p>
                                    <p className="text-2xl font-bold text-blue-600">{analytics.total_readings}</p>
                                </div>
                                <Database className="h-8 w-8 text-blue-600" />
                            </div>
                        </div>

                        <div className="bg-white rounded-lg shadow-lg p-6">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-gray-600">Avg Temperature</p>
                                    <p className="text-2xl font-bold text-green-600">{analytics.average_temperature?.toFixed(1)}°C</p>
                                </div>
                                <Thermometer className="h-8 w-8 text-green-600" />
                            </div>
                        </div>

                        <div className="bg-white rounded-lg shadow-lg p-6">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-gray-600">Avg Humidity</p>
                                    <p className="text-2xl font-bold text-purple-600">{analytics.average_humidity?.toFixed(1)}%</p>
                                </div>
                                <TrendingUp className="h-8 w-8 text-purple-600" />
                            </div>
                        </div>

                        <div className="bg-white rounded-lg shadow-lg p-6">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-gray-600">Recent Anomalies</p>
                                    <p className="text-2xl font-bold text-red-600">{analytics.recent_anomalies}</p>
                                </div>
                                <AlertTriangle className="h-8 w-8 text-red-600" />
                            </div>
                        </div>
                    </div>
                )}

                {/* Charts */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                    <div className="bg-white rounded-lg shadow-lg p-6">
                        <h2 className="text-xl font-semibold text-gray-800 mb-4">Sensor Readings Over Time</h2>
                        <ResponsiveContainer width="100%" height={300}>
                            <LineChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="index" />
                                <YAxis />
                                <Tooltip />
                                <Line type="monotone" dataKey="temperature" stroke="#3B82F6" name="Temperature" />
                                <Line type="monotone" dataKey="humidity" stroke="#8B5CF6" name="Humidity" />
                                <Line type="monotone" dataKey="power" stroke="#EF4444" name="Power" />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>

                    <div className="bg-white rounded-lg shadow-lg p-6">
                        <h2 className="text-xl font-semibold text-gray-800 mb-4">Anomaly Detection</h2>
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="index" />
                                <YAxis />
                                <Tooltip />
                                <Bar dataKey="status" fill="#EF4444" name="Anomaly Status" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Prediction Form */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="bg-white rounded-lg shadow-lg p-6">
                        <h2 className="text-xl font-semibold text-gray-800 mb-4">Anomaly Prediction</h2>
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700">Temperature (°C)</label>
                                <input
                                    type="number"
                                    step="0.1"
                                    value={predictionForm.temperature}
                                    onChange={(e) => setPredictionForm({ ...predictionForm, temperature: parseFloat(e.target.value) })}
                                    className="mt-1 block w-full rounded-md border-gray-300 border p-2 focus:border-blue-500 focus:outline-none"
                                />
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700">Humidity (%)</label>
                                <input
                                    type="number"
                                    step="0.1"
                                    value={predictionForm.humidity}
                                    onChange={(e) => setPredictionForm({ ...predictionForm, humidity: parseFloat(e.target.value) })}
                                    className="mt-1 block w-full rounded-md border-gray-300 border p-2 focus:border-blue-500 focus:outline-none"
                                />
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700">Pressure (hPa)</label>
                                <input
                                    type="number"
                                    step="0.1"
                                    value={predictionForm.pressure}
                                    onChange={(e) => setPredictionForm({ ...predictionForm, pressure: parseFloat(e.target.value) })}
                                    className="mt-1 block w-full rounded-md border-gray-300 border p-2 focus:border-blue-500 focus:outline-none"
                                />
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700">Vibration</label>
                                <input
                                    type="number"
                                    step="0.01"
                                    value={predictionForm.vibration}
                                    onChange={(e) => setPredictionForm({ ...predictionForm, vibration: parseFloat(e.target.value) })}
                                    className="mt-1 block w-full rounded-md border-gray-300 border p-2 focus:border-blue-500 focus:outline-none"
                                />
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700">Power Consumption (W)</label>
                                <input
                                    type="number"
                                    step="0.1"
                                    value={predictionForm.power_consumption}
                                    onChange={(e) => setPredictionForm({ ...predictionForm, power_consumption: parseFloat(e.target.value) })}
                                    className="mt-1 block w-full rounded-md border-gray-300 border p-2 focus:border-blue-500 focus:outline-none"
                                />
                            </div>

                            <button
                                onClick={handlePrediction}
                                disabled={loading}
                                className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50 transition-colors"
                            >
                                {loading ? 'Predicting...' : 'Predict Anomaly'}
                            </button>
                        </div>
                    </div>

                    {/* Results */}
                    <div className="bg-white rounded-lg shadow-lg p-6">
                        <h2 className="text-xl font-semibold text-gray-800 mb-4">Prediction Results</h2>

                        {prediction && (
                            <div className="space-y-4">
                                <div className={`p-4 rounded-lg ${prediction.is_anomaly ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'} border`}>
                                    <div className="flex items-center space-x-2">
                                        {prediction.is_anomaly ? (
                                            <AlertTriangle className="h-5 w-5 text-red-600" />
                                        ) : (
                                            <Zap className="h-5 w-5 text-green-600" />
                                        )}
                                        <span className={`font-medium ${prediction.is_anomaly ? 'text-red-800' : 'text-green-800'}`}>
                                            {prediction.is_anomaly ? 'Anomaly Detected' : 'Normal Operation'}
                                        </span>
                                    </div>
                                </div>

                                <div className="grid grid-cols-2 gap-4">
                                    <div className="bg-gray-50 p-3 rounded-lg">
                                        <p className="text-sm text-gray-600">Predicted Status</p>
                                        <p className="font-semibold text-gray-800">{prediction.predicted_status}</p>
                                    </div>

                                    <div className="bg-gray-50 p-3 rounded-lg">
                                        <p className="text-sm text-gray-600">Confidence</p>
                                        <p className="font-semibold text-gray-800">{(prediction.confidence * 100).toFixed(1)}%</p>
                                    </div>

                                    <div className="bg-gray-50 p-3 rounded-lg col-span-2">
                                        <p className="text-sm text-gray-600">Anomaly Score</p>
                                        <p className="font-semibold text-gray-800">{prediction.anomaly_score.toFixed(4)}</p>
                                    </div>
                                </div>
                            </div>
                        )}

                        {modelStatus && (
                            <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                                <h3 className="font-medium text-blue-800 mb-2">Model Status</h3>
                                <p className="text-sm text-blue-700">
                                    Model Trained: {modelStatus.model_trained ? 'Yes' : 'No'}
                                </p>
                                {modelStatus.metrics && (
                                    <p className="text-sm text-blue-700">
                                        Accuracy: {(modelStatus.metrics.accuracy * 100).toFixed(1)}%
                                    </p>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

export default SensorDashboard;