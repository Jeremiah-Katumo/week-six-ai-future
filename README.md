# AgroAI Edge

AgroAI Edge is a modular platform for real-time sensor data analytics and anomaly detection at the edge, designed for agriculture and industrial IoT deployments. It features a FastAPI backend for data ingestion, analytics, and machine learning, and a React-based dashboard for visualization and live anomaly prediction.

---

## Features

- **Edge Sensor Data Ingestion:** Collect and store readings from various edge-deployed sensors.
- **Real-Time Analytics:** View statistics such as average temperature, humidity, and recent anomalies.
- **On-Device Machine Learning:** Train and evaluate anomaly detection models at the edge.
- **Live Anomaly Prediction:** Predict anomalies in real-time using ensemble ML models.
- **Interactive Dashboard:** Visualize sensor data, analytics, and prediction results.
- **Configurable via .env:** Easily adapt to different environments and deployments.

---

## Tech Stack

- **Backend:** FastAPI, Python, dotenv
- **Frontend:** React, Recharts, Tailwind CSS, Lucide Icons
- **ML/Analytics:** scikit-learn, numpy, pandas
- **Deployment:** Docker (optional), .env configuration

---

## Getting Started

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd week-six-ai-future
```

### 2. Backend Setup

#### a. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

#### b. Install dependencies

```bash
pip install -r requirements.txt
```

#### c. Configure environment variables

Create a `.env` file in `agroai/`:

```
API_BASE_URL=http://localhost:8000
ORIGIN_BASE_URL=http://localhost:3000
SECRET_KEY=your_secret_key
```

#### d. Run the FastAPI server

```bash
uvicorn agroai.main:app --reload
```

---

### 3. Frontend Setup

```bash
cd agroai-app
npm install
```

Create a `.env` file in `agroai-app/`:

```
REACT_APP_API_BASE_URL=http://localhost:8000
```

Start the React app:

```bash
npm start
```

---

## Usage

- Access the dashboard at [http://localhost:3000](http://localhost:3000)
- The backend API runs at [http://localhost:8000](http://localhost:8000)
- Use the dashboard to view analytics, add sample readings, and predict anomalies.

---

## Project Structure

```
week-six-ai-future/
│
├── agroai/                # FastAPI backend
│   ├── main.py
│   ├── routes.py
│   ├── pipeline.py
│   └── ...
│
├── agroai-app/            # React frontend
│   ├── src/
│   │   └── components/
│   │       └── SensorDashboard.jsx
│   └── ...
│
├── requirements.txt
└── README.md
```

---

## API Endpoints (Backend)

- `GET /` — API status
- `GET /sensors/readings` — Get sensor readings
- `POST /sensors/readings` — Add a sensor reading
- `GET /sensors/analytics` — Get analytics summary
- `POST /predict/anomaly` — Predict anomaly from input
- `POST /train/models` — Train/retrain ML models

---

## Contributing

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

---

## License

MIT License

---

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://react.dev/)
- [Recharts](https://recharts.org/)
- [Lucide Icons](https://lucide.dev/)