# tests/test_main.py

import pytest
import yaml
import io
import zipfile
from fastapi.testclient import TestClient
from api_main import app
from pathlib import Path

@pytest.fixture
def fixtures_path() -> Path:
    """Returns the absolute path to the test fixtures directory."""
    return Path(__file__).parent / "fixtures"

# Create a TestClient instance based on your FastAPI app
client = TestClient(app)

# A sample valid training configuration for tests
SAMPLE_CONFIG = {
    'model_name_or_path': 'openai/clip-vit-base-patch32',
    'train_dataset': 'train_data.csv',
    'eval_dataset': 'eval_data.csv',
    'datatype': 'custom', # Requires a file upload
    'output_dir': '/tmp/test_output'
}

# --- Fixtures ---

@pytest.fixture
def mock_db_and_executor(mocker):
    """
    Mocks the database module and the ProcessPoolExecutor to prevent
    actual db writes and process creation during tests.
    """
    # Mock the entire job_db module
    mock_job_db = mocker.patch('main.job_db')
    
    # Mock the executor to run the job immediately and in the same process
    mock_executor = mocker.patch('main.asyncio.get_running_loop')
    # Configure the mock to just call the function directly instead of in a pool
    mock_executor.return_value.run_in_executor.side_effect = lambda pool, func, *args: func(*args)

    # Mock the actual train_job function to avoid running it
    mocker.patch('main.train_job')
    
    return mock_job_db

@pytest.fixture
def dummy_zip_file():
    """Creates an in-memory zip file for upload testing."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        zipf.writestr("train_data.csv", "image,caption\nimg1.jpg,a cat")
        zipf.writestr("eval_data.csv", "image,caption\neval_img1.jpg,a dog")
    zip_buffer.seek(0)
    return ("dataset.zip", zip_buffer, "application/zip")

# --- Test Cases ---

def test_train_with_zip_success(mock_db_and_executor, dummy_zip_file):
    """
    Tests the happy path for the /train/ endpoint with a zip file.
    """
    config_str = yaml.dump(SAMPLE_CONFIG)
    
    response = client.post(
        "/train/",
        data={"config_str": config_str},
        files={"file": dummy_zip_file}
    )

    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert "Training job accepted" in data["message"]

    # Verify that the DB was called to create the job
    mock_db_and_executor.create_job.assert_called_once()
    # Verify the background job was scheduled
    assert app.state.process_pool.submit.called


def test_train_with_hf_dataset_success(mock_db_and_executor):
    """
    Tests the happy path for the /train/ endpoint using a HuggingFace dataset (no file).
    """
    hf_config = SAMPLE_CONFIG.copy()
    hf_config['datatype'] = 'hf'
    config_str = yaml.dump(hf_config)

    response = client.post(
        "/train/",
        data={"config_str": config_str}
        # No file is passed
    )

    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    
    # Verify the DB and executor were called
    mock_db_and_executor.create_job.assert_called_once()
    assert app.state.process_pool.submit.called


def test_train_fails_if_zip_is_required_but_not_provided(mock_db_and_executor):
    """
    Tests that the endpoint returns a 400 error if a custom dataset
    is specified without a file upload.
    """
    config_str = yaml.dump(SAMPLE_CONFIG)

    response = client.post(
        "/train/",
        data={"config_str": config_str} # No file attached
    )

    assert response.status_code == 400
    assert "Custom data requires a ZIP file upload" in response.json()["detail"]


def test_train_fails_with_invalid_yaml(mock_db_and_executor):
    """
    Tests that the endpoint returns a 400 error for malformed YAML.
    """
    invalid_config_str = "key: value\n  bad-indent"
    response = client.post(
        "/train/",
        data={"config_str": invalid_config_str},
    )

    assert response.status_code == 400
    assert "Config is not valid YAML" in response.json()["detail"]


def test_get_status_found(mock_db_and_executor):
    """
    Tests retrieving the status of an existing job.
    """
    job_id = "test-job-123"
    job_details = {
        "id": job_id,
        "status": "training",
        "detail": "Training in progress... 25%",
        "created_at": "2025-08-06T12:00:00Z"
    }
    # Configure the mock to return a specific job
    mock_db_and_executor.get_job.return_value = job_details

    response = client.get(f"/status/{job_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == job_id
    assert data["status"] == "training"
    assert data["progress"] == 25.0
    mock_db_and_executor.get_job.assert_called_with(job_id)


def test_get_status_not_found(mock_db_and_executor):
    """
    Tests retrieving the status of a non-existent job.
    """
    job_id = "non-existent-job"
    # Configure the mock to simulate a job not being found
    mock_db_and_executor.get_job.return_value = None

    response = client.get(f"/status/{job_id}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found."
    mock_db_and_executor.get_job.assert_called_with(job_id)