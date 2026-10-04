import os
import json
import re
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import requests
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="LabPulse AI", description="Open-weight bench assistant for wet-lab logs")

app.mount("/static", StaticFiles(directory="static"), name="static")

MONGODB_URI = os.getenv("MONGODB_URI", "")
GEMMA_API_URL = os.getenv("GEMMA_API_URL", "https://api-inference.huggingface.co/models/google/gemma-2-2b-it")
HF_TOKEN = os.getenv("HF_TOKEN", "")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# MongoDB Atlas connection with safe fallback
mongo_client = None
db = None
if MONGODB_URI:
    try:
        mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
        db = mongo_client["labpulse"]
        db.command("ping")
        print("Connected to MongoDB Atlas successfully.")
    except Exception as e:
        print(f"MongoDB offline/failed: {e}. Using in-memory fallback.")
        db = None

fallback_storage = []

class RawNotePayload(BaseModel):
    raw_notes: str = Field(...)

class StructuredExperiment(BaseModel):
    assay_type: str
    target_sample: str
    annealing_temp_celsius: Optional[float]
    optical_density_ratio: Optional[float]
    cycle_count: Optional[int]
    anomalies: List[str]
    clean_summary: str
    timestamp: str

EXTRACTION_PROMPT = """You are LabPulse AI, an automated molecular biology laboratory logger.
Analyze these raw notes and extract the parameters into valid JSON:
{{
  "assay_type": "PCR or Spectrophotometry or Gel Electrophoresis or Other",
  "target_sample": "gene or sample name",
  "annealing_temp_celsius": number or null,
  "optical_density_ratio": number or null,
  "cycle_count": integer or null,
  "anomalies": ["list of issues or contamination"],
  "clean_summary": "1-sentence formal summary"
}}

Raw Notes:
{notes}
"""

def fallback_heuristic_parser(notes: str) -> dict:
    """Offline heuristic parser used when no API token or local Ollama is active"""
    assay = "PCR" if "pcr" in notes.lower() else ("Spectrophotometry" if ("od" in notes.lower() or "260" in notes) else "Assay Protocol")
    
    # Annealing temp extraction
    temp_match = re.search(r"(\d{2}(?:\.\d+)?)\s*(?:°?C|celsius)", notes, re.IGNORECASE)
    temp = float(temp_match.group(1)) if temp_match else None

    # OD ratio extraction
    od_match = re.search(r"(?:a?260\s*/\s*a?280|od|ratio)[^\d]*(\d\.\d+)", notes, re.IGNORECASE)
    od = float(od_match.group(1)) if od_match else None

    # Cycles
    cycle_match = re.search(r"(\d{1,2})\s*cycles?", notes, re.IGNORECASE)
    cycles = int(cycle_match.group(1)) if cycle_match else None

    # Anomalies
    anomalies = []
    if "smear" in notes.lower(): anomalies.append("Smear observed in gel lane")
    if "expired" in notes.lower(): anomalies.append("Buffer/reagent expired")
    if "contamination" in notes.lower(): anomalies.append("Possible sample contamination")
    if "leak" in notes.lower(): anomalies.append("Buffer leak reported")
    if not anomalies: anomalies.append("None detected")

    return {
        "assay_type": assay,
        "target_sample": "Extracted Sample Colony" if "colony" in notes.lower() else "Plasmid DNA",
        "annealing_temp_celsius": temp,
        "optical_density_ratio": od,
        "cycle_count": cycles,
        "anomalies": anomalies,
        "clean_summary": f"Completed {assay} evaluation under standard protocol parameters."
    }

def query_gemma(prompt: str, notes: str) -> dict:
    # 1. Try local Ollama if running
    try:
        ollama_res = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": "gemma2:2b", "prompt": prompt, "stream": False},
            timeout=2.0
        )
        if ollama_res.status_code == 200:
            text = ollama_res.json().get("response", "")
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
    except Exception:
        pass

    # 2. Try Hugging Face Gemma 2 API if token provided
    if HF_TOKEN.strip():
        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            payload = {
                "inputs": f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
                "parameters": {"max_new_tokens": 512, "temperature": 0.1}
            }
            res = requests.post(GEMMA_API_URL, headers=headers, json=payload, timeout=12)
            if res.status_code == 200:
                raw_text = res.json()[0].get("generated_text", "")
                match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
        except Exception as e:
            print(f"HF inference error: {e}")

    # 3. Graceful offline fallback
    return fallback_heuristic_parser(notes)

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

@app.post("/api/parse", response_model=StructuredExperiment)
def parse_bench_notes(payload: RawNotePayload):
    prompt = EXTRACTION_PROMPT.format(notes=payload.raw_notes)
    parsed_data = query_gemma(prompt, payload.raw_notes)
    
    structured = StructuredExperiment(
        assay_type=parsed_data.get("assay_type", "Standard Protocol"),
        target_sample=parsed_data.get("target_sample", "Unspecified"),
        annealing_temp_celsius=parsed_data.get("annealing_temp_celsius"),
        optical_density_ratio=parsed_data.get("optical_density_ratio"),
        cycle_count=parsed_data.get("cycle_count"),
        anomalies=parsed_data.get("anomalies", []),
        clean_summary=parsed_data.get("clean_summary", "Run completed and processed."),
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    )
    
    record = structured.model_dump()
    if db is not None:
        try:
            db["runs"].insert_one(record.copy())
        except Exception:
            fallback_storage.append(record)
    else:
        fallback_storage.append(record)
        
    return structured

@app.get("/api/runs")
def get_recent_runs():
    if db is not None:
        try:
            return list(db["runs"].find({}, {"_id": 0}).sort("timestamp", -1).limit(10))
        except Exception:
            return fallback_storage[-10:]
    return fallback_storage[-10:]