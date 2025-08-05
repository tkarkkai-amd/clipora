# main.py
import yaml
import time
import zipfile
import os
import uuid
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

import train
import job_db
from clipora.config import TrainConfig, parse_yaml_to_config

ZIPPED_DATA_EXTRACT_PATH = os.getenv("ZIPPED_DATA_EXTRACT_PATH", "/tmp/extracted_data/")
FILE_DOWNLOAD_PATH = os.getenv("FILE_DOWNLOAD_PATH", "/tmp/downloaded_data/")

os.makedirs(ZIPPED_DATA_EXTRACT_PATH, exist_ok=True)
os.makedirs(FILE_DOWNLOAD_PATH, exist_ok=True)

# --- Lifespan Manager for Executor and DB ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup: Create process pool and initialize the database
    # will be used to run CPU-intensive tasks in a separate process
    # so that API will stay responsive
    job_db.init_db()
    app.state.process_pool = ProcessPoolExecutor()
    yield
    # On shutdown: Gracefully close the process pool
    app.state.process_pool.shutdown()

# --- (Modified) Background Task (runs in a separate process) ---
def train_job(job_id: str, config: TrainConfig, zip_path: str):
    """
    This function is CPU-intensive and runs in a separate process.
    It communicates status by calling functions from the `db` module.
    """
    try:
        job_db.update_job_status(job_id, "extracting", f"Extracting {os.path.basename(zip_path)}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ZIPPED_DATA_EXTRACT_PATH)

        train_dataset_fname = os.path.basename(config.train_dataset)
        config.train_dataset = os.path.join(ZIPPED_DATA_EXTRACT_PATH, train_dataset_fname)
        eval_dataset_fname = os.path.basename(config.eval_dataset)
        config.eval_dataset = os.path.join(ZIPPED_DATA_EXTRACT_PATH, eval_dataset_fname)

        job_db.update_job_status(job_id, "training", "Model training in progress...")
        train.dummy_training(job_id)
        # train.main(config)

        job_db.update_job_status(job_id, "complete", "Training finished successfully.")
    except Exception as e:
        job_db.update_job_status(job_id, "failed", str(e))

# --- FastAPI App ---
app = FastAPI(
    title="Training API",
    description="An API to submit training jobs with status tracking via SQLite.",
    version="2.1.0",
    lifespan=lifespan
)

origins = [
    "http://localhost",
    "http://localhost:8000", # If you serve the html with `python -m http.server`
    "null"  # Allow requests from local files (i.e., opening the HTML with file://)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# --- Training Endpoint ---
@app.post("/train/", summary="Submit a training job", status_code=status.HTTP_202_ACCEPTED)
async def train_model(
    config_str: str = Form(..., description="A YAML string for training config."),
    file: UploadFile = File(..., description="A ZIP file with the dataset.")
):
    job_id = str(uuid.uuid4())
    
    try:
        config_dict = yaml.safe_load(config_str)
        config = TrainConfig(**config_dict)
    except yaml.YAMLError:
        raise HTTPException(status_code=400, detail="Config is not valid YAML.")
    
    if file.content_type not in ["application/zip", "application/x-zip-compressed"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Expected ZIP.")

    save_path = os.path.join(FILE_DOWNLOAD_PATH, f"{job_id}.zip")
    try:
        with open(save_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)
    finally:
        await file.close()

    # Create the initial job record in the database
    job_db.create_job(job_id)

    # Submit the job to the process pool
    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        app.state.process_pool,
        train_job,
        job_id, config, save_path
    )
    
    return {
        "message": "Training job accepted.",
        "job_id": job_id,
        "status_url": app.url_path_for("get_status", job_id=job_id)
    }

# --- (Modified) Status Endpoint ---
@app.get("/status/{job_id}", summary="Get job status", name="get_status")
async def get_status(job_id: str):
    """
    Polls the database to get the current status of the training job.
    """
    job = job_db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return job

@app.get("/")
def read_root():
    return {"message": "Welcome to the Training API. POST to /train/ to submit a job."}