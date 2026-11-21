import os
import requests
import urllib3
from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
from app.pipeline import process_dataset, process_single_query

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI(title="Sain Team Кореша")

class RunRequest(BaseModel):
    case: int = Field(..., ge=1, le=4, description="Номер кейса от 1 до 4")
    dataset_code: str = Field(..., min_length=1, description="Код набора данных")

class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1)
    expert_id: str
    user_id: Optional[str] = "u_default"

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/run")
def run(req: RunRequest, authorization: str | None = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"error": "Missing or invalid Authorization header"})
    
    token = authorization.replace("Bearer ", "").strip()
    
    api_url = os.environ.get("HACK_API_URL", "https://hack-api.be-healthy.tech")
    url = f"{api_url}/case/{req.case}/getData?dataset_code={req.dataset_code}"
    
    try:
        response = requests.get(
            url, 
            headers={"Authorization": f"Bearer {token}"}, 
            timeout=30, 
            verify=False 
        )
        response.raise_for_status()
        full_data = response.json()
    except requests.exceptions.HTTPError as e:
        return JSONResponse(
            status_code=e.response.status_code, 
            content={"error": f"API Error: {e.response.text}"}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Network error: {str(e)}"})

    dataset = full_data.get("dataset", {})
    
    try:
        final_answers = process_dataset(dataset)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Pipeline error: {str(e)}"})
    
    return JSONResponse(content=final_answers)

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        answer = process_single_query(text=req.text, expert_id=req.expert_id)
        return JSONResponse(content=answer)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
def health():
    return {"status": "ok"}