from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from routes import router

load_dotenv()

app = FastAPI()

# connect to react app
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get('ORIGIN_BASE_URL'),  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cors
@app.get('/')
async def root():
    return {
        "message": "Welcome to the AgroAI API",
        "status": "running"
    }
    
app.include_router(router)
