# main.py
# 1. Save this code in a file named `main.py`.
#
# 2. Install the necessary libraries:
#    pip install "fastapi[all]" python-multipart
#
# 3. Run the application from your terminal:
#    uvicorn main:app --reload --port 8080
#
# 4. Use the provided cURL command to test the endpoint.

import yaml
import time
import zipfile
import os
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, ValidationError
import train
from clipora.config import TrainConfig, parse_yaml_to_config


# Define the Pydantic model based on the provided training configuration.
class TrainingConfigPydantic(BaseModel):
    # Model and Weights
    model_name: str
    pretrained: str
    compile: bool
    seed: int

    # Environment
    device: str
    output_dir: str

    # Logging (wandb)
    wandb: bool
    wandb_project: Optional[str] = None # Optional, might only be needed if wandb is True

    # Dataset
    train_dataset: str
    eval_dataset: str
    datatype: str
    csv_separator: str
    image_col: str
    text_col: str
    shuffle: bool

    # LoRA Parameters
    lora_rank: int
    lora_alpha: int
    lora_dropout: float

    # Training Hyperparameters
    batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    use_8bit_adam: bool
    learning_rate: float
    epochs: int
    warmup: float

    # Saving and Evaluation
    save_interval: int
    eval_interval: int
    eval_steps: int

ZIPPED_DATA_EXTRACT_PATH = "/Users/tman/work/received_data/"

def simple_extract(zip_path, extract_to=ZIPPED_DATA_EXTRACT_PATH):
    """Simple extraction with minimal error handling."""
    try:
        print(f"Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
    
def train_job(config: TrainConfig, zip_path):
    simple_extract(zip_path, ZIPPED_DATA_EXTRACT_PATH)
    train_dataset_fname = os.path.basename(config.train_dataset)
    config.train_dataset = os.path.join(ZIPPED_DATA_EXTRACT_PATH, train_dataset_fname)
    eval_dataset_fname = os.path.basename(config.eval_dataset)
    config.eval_dataset = os.path.join(ZIPPED_DATA_EXTRACT_PATH, eval_dataset_fname)
    train.main(config)

# simple_extract("/Users/tman/work/bridge_data.zip", ZIPPED_DATA_EXTRACT_PATH)
# config = parse_yaml_to_config("api_example_config.yml")

# train_job(config)

# Initialize the FastAPI application
app = FastAPI(
    title="Training API",
    description="An API to submit training data (ZIP) and a configuration (YAML).",
    version="1.2.0",
)

@app.post("/train/", summary="Receive Training Data and Config")
async def train_model(
    background_tasks: BackgroundTasks,
    config_str: str = Form(..., description="A YAML string containing the training configuration."),
    file: UploadFile = File(..., description="A ZIP file containing the training dataset.")
):
    """
    This endpoint accepts a training job, saves the data, and starts the training
    process in the background.

    1.  **config_str**: A form field containing a YAML string that conforms to the `TrainingConfig` model.
    2.  **file**: An uploaded ZIP file containing the dataset.
    """
    # --- Config Validation ---
    try:
        config_dict = yaml.safe_load(config_str)
        config = TrainConfig(**config_dict)
    except yaml.YAMLError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Config is not a valid YAML string."
        )
    # except ValidationError as e:
    #     raise HTTPException(
    #         status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    #         detail={"message": "Invalid config structure.", "errors": e.errors()}
    #     )

    # --- File Validation ---
    if file.content_type not in ["application/zip", "application/x-zip-compressed"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Expected a ZIP file, but got '{file.content_type}'."
        )
        
    # --- Save the uploaded file to disk in chunks ---
    save_path = f"fastapidata_{int(time.time())}.zip"
    try:
        with open(save_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024): # 1MB chunks
                buffer.write(chunk)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"There was an error saving the file: {e}"
        )
    finally:
        await file.close()

    # --- Add the training function to run in the background ---
    # The API will return a response immediately, and this function will
    # run separately.
    background_tasks.add_task(train_job, config, save_path)

    return {
        "message": "Training job accepted and started in the background.",
        "validated_config": config.__dict__,
        "file_info": {
            "filename": file.filename,
            "saved_path": save_path
        }
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to the Training API. Send a POST request to /train/ to submit a job."}
