# db.py
import os
import sqlite3
import datetime
from typing import Dict, Optional

DB_PATH = os.getenv("DB_PATH", "/tmp/training_jobs.db")

def get_db_connection():
    """
    Establishes a database connection.
    Enables WAL mode for better concurrency and sets a timeout.
    """
    conn = sqlite3.connect(DB_PATH, timeout=15)
    # WAL mode allows concurrent readers and one writer, preventing many locking issues.
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn

def init_db():
    """Initializes the database table if it doesn't exist."""
    conn = get_db_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                detail TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()

def create_job(job_id: str):
    """Creates a new job record with a 'queued' status."""
    conn = get_db_connection()
    try:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO jobs (id, status, detail, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, "queued", "Job is waiting to start.", now, now)
        )
        conn.commit()
    finally:
        conn.close()

def update_job_status(job_id: str, status: str, detail: str = ""):
    """Updates the status and detail of an existing job."""
    conn = get_db_connection()
    try:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        conn.execute(
            "UPDATE jobs SET status = ?, detail = ?, updated_at = ? WHERE id = ?",
            (status, detail, now, job_id)
        )
        conn.commit()
    finally:
        conn.close()

def get_job(job_id: str) -> Optional[Dict]:
    """Fetches a job record by its ID and returns it as a dictionary."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        if row:
            # Convert the sqlite3.Row object to a standard dictionary
            return dict(row)
        return None
    finally:
        conn.close()