# API LIKE CODE, USED TO RUN THE MIND PIPELINE
from fastapi import FastAPI, Request, Response, HTTPException, WebSocket, WebSocketDisconnect
from typing import List
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np
from tabulate import tabulate
from scipy import sparse
import os
import asyncio
import sys
import json
import dotenv
from enum import Enum
from datetime import datetime
#OJO A ESTE
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from IRQ.indexer import *
from IRQ.query_eng import *
from mind.query_generator import QueryGenerator
from utils.utils import clear_screen, print_doc
from CLIemulator import CLI

mind = FastAPI()
dotenv.load_dotenv()
templates = Jinja2Templates(directory="backend/templates")

mind.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Be more restrictive in production, configure firewall rules instead
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MindStatus(str, Enum):
    idle = "idle"
    initializing = "initializing"
    initialized = "initialized"
    running = "running"
    completed = "completed"
    failed = "failed"

class InitData(BaseModel):
    dataset: str

current_status = {"state": MindStatus.idle, "last_updated": None}

@mind.get("/test_root")
def test_root():
    return {"message": "src API is up"}

@mind.get("/status")
def get_status():
    return current_status

@mind.post("/initialize")
def initialize(data: InitData):
    if current_status["state"] in [MindStatus.initializing, MindStatus.initialized, MindStatus.running, MindStatus.completed]:
        return {"message": f"MIND already in '{current_status['state']}' state."}

    current_status["state"] = MindStatus.initializing
    current_status["last_updated"] = datetime.now()

    # try:
    dataset_path = os.getenv("DATASET_PATH", "/Data/3_joined_data")

    dataset = data.dataset
    dataset_path = os.path.join(dataset_path, dataset)
    print(f"Initializing MIND with dataset at: {dataset_path}")

    ### NO ESTOY SEGURO
    global cli 
    cli = CLI(dataset_name = dataset)

    current_status["state"] = MindStatus.initialized

    current_status["last_updated"] = datetime.now()
    return {"message": 'MIND initialized successfully.'}
    # except Exception as e:
    #     print(f"Error during initialization: {e}")
    #     current_status["state"] = MindStatus.failed
    #     current_status["last_updated"] = datetime.now()
    #     return {"error": str(e)}, 500

@mind.get("/explore")
def explore():
    '''
    Explore the topics in the MIND dataset.
    '''
    if current_status["state"] != MindStatus.initialized:
        raise HTTPException(status_code=400, detail="MIND not initialized. Please initialize first.")

    try:
        topics = cli.get_topics()
        return {"topics": topics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@mind.post("/run")
def run_mind(request: Request):
    '''
    Run the mind pipeline with the provided request data.
    '''
    pass

