import io
import time
import logging
import requests
import gc
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from PIL import Image
from ultralytics import RTDETR
import os
from dotenv import load_dotenv

# ==========================================
# 1. LOGGING CONFIGURATION (Bonus Points)
# ==========================================
LOG_FILE = Path("app_audit.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ]
)
logger = logging.getLogger("civic_api")

# ==========================================
# 2. APP & MODEL INITIALIZATION
# ==========================================
app = FastAPI(
    title="Civic Infrastructure AI API",
    description="Vision and Reasoning API for Road Damage Detection"
)

# Load environment variables from the .env file
load_dotenv()

# Fetch the key securely
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY is missing! Check your .env file.")

# Configuration
MODEL_PATH = "best_finetuned.pt"
BUCKET_NAME = "road-damage-models-bucket"
BLOB_NAME = "best_finetuned.pt"

def download_model_from_gcs():
    if not os.path.exists(MODEL_PATH):
        logger.info(f"Attempting to download model from GCS bucket: {BUCKET_NAME}")
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(BUCKET_NAME)
            blob = bucket.blob(BLOB_NAME)
            blob.download_to_filename(MODEL_PATH)
            logger.info("Model downloaded successfully from GCS.")
        except Exception as e:
            logger.error(f"CRITICAL GCS DOWNLOAD FAILED: {str(e)}")
            raise RuntimeError(f"GCS Download Error: {e}")

# Trigger download before loading into RT-DETR
download_model_from_gcs()

# Declare model globally upfront
model = None

try:
    model = RTDETR(MODEL_PATH)
    logger.info("RT-DETR model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise RuntimeError(f"Could not load model weights: {e}")

# ==========================================
# 3. HELPER FUNCTIONS (No Frameworks)
# ==========================================
def call_groq(prompt: str, max_tokens: int = 250) -> str:
    """Helper to communicate with Groq API natively using standard requests."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "openai/gpt-oss-20b", 
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1, 
        "max_tokens": max_tokens
    }
    
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        logger.error(f"Groq API Error: {response.text}")
        raise HTTPException(status_code=502, detail="External LLM API failure.")
    
def extract_detections(result) -> list:
    """Extracts bounding boxes and confidence from RT-DETR output."""
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        detections.append({
            "class_name": model.names[cls_id],
            "confidence": round(float(box.conf[0].item()), 4),
            "bbox": [round(x, 2) for x in box.xyxy[0].tolist()]
        })
    return detections

# Root health check endpoint for deployment stability
@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Civic Infrastructure AI API is running successfully."
    }

# ==========================================
# 4. ENDPOINT 1: VISION ONLY (/detect)
# ==========================================
@app.post("/detect")
async def detect_damage(file: UploadFile = File(...)):
    start_time = time.time()
    logger.info(f"POST /detect hit | File: {file.filename}")
    
    if not file.content_type.startswith("image/"):
        logger.warning("Invalid file type uploaded to /detect.")
        raise HTTPException(status_code=400, detail="Must be an image.")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Run Inference
    # Change imgsz to 416 to lower memory usage on Render's 512MB free tier
    results = model.predict(source=image, conf=0.25, imgsz=640, device="cpu")
    detections = extract_detections(results[0])
    
    # Cleanup memory to protect low-RAM free tiers
    del results
    gc.collect()
    
    latency = (time.time() - start_time) * 1000
    logger.info(f"Detection complete in {latency:.2f}ms | Found {len(detections)} items.")
    
    return {
        "status": "success",
        "total_detections": len(detections),
        "detections": detections
    }

# ==========================================
# 5. ENDPOINT 2: REASONING LAYER (/reason)
# ==========================================
@app.post("/reason")
async def natural_language_reasoning(file: UploadFile = File(...), question: str = Form(...)):
    start_time = time.time()
    logger.info(f"POST /reason hit | File: '{file.filename}' | Question: '{question}'")

    # ----------------------------------------
    # STAGE 0: VALIDATION (Check file type first)
    # ----------------------------------------
    if not file.content_type.startswith("image/"):
        logger.warning("Invalid file type uploaded to /reason.")
        raise HTTPException(status_code=400, detail="Must be an image.")

    # ----------------------------------------
    # STAGE 1: INTENT ROUTING
    # ----------------------------------------
    intent_prompt = f"""
    You are an intent routing classifier. A user asked this question: {question}
    Does this question require analyzing an image of road damage, potholes, cracks, or civic infrastructure?
    Reply strictly with exactly "YES" or "NO". Do not output any other text.
    """
    
    intent = call_groq(intent_prompt, max_tokens=10)
    
    if "NO" in intent.upper():
        logger.info("Intent Routing: Unrelated query. Bypassing vision model.")
        return {
            "status": "unrelated", 
            "response": "This question is not related to road infrastructure or the image provided."
        }
    
    logger.info("Intent Routing: Query validated. Proceeding to vision model.")

    # ----------------------------------------
    # STAGE 2: STRUCTURED VISION INFERENCE
    # ----------------------------------------
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    results = model.predict(source=image, conf=0.25, imgsz=640, device='cpu')
    detections = extract_detections(results[0])
    
    del results
    gc.collect()
    logger.info(f"RT-DETR Inference finished | Found {len(detections)} object(s).")

    # ----------------------------------------
    # STAGE 3: CONFIDENCE GUARDRAIL
    # ----------------------------------------
    max_conf = max([d['confidence'] for d in detections]) if detections else 0.0
    if detections and max_conf < 0.35:
        logger.warning(f"Guardrail TRIPPED! Max confidence {max_conf:.2f} is below the 0.35 safety threshold.")
        return {
            "status": "low_confidence", 
            "response": "The detection results are insufficient to answer confidently. Human verification is required."
        }

    # ----------------------------------------
    # STAGE 4: STRUCTURED REASONING
    # ----------------------------------------
    
    reasoning_prompt = f"""
    You are a civic infrastructure AI assistant. 
    Here is the exact road damage data detected by our vision model in the uploaded image:
    {detections}
    
    Based ONLY on this data, answer the user's question concisely in plain language. 
    Do not invent new damages or assume information not in the JSON. 
    
    CRITICAL RULES:
    1. If the detection list is empty ([]), state professionally that no road damage or issues were detected by the system.
    2. If the user asks about an object or animal completely unrelated to road infrastructure, state clearly that it is not present in the image.
    3. Never return a blank response.
    
    User Question: {question}
    """
    
    final_answer = call_groq(reasoning_prompt, max_tokens=250)
    
    latency = (time.time() - start_time) * 1000
    logger.info(f"Request finalized successfully in {latency:.2f}ms.\n" + "-"*50)
    
    return {
        "status": "success",
        "response": final_answer,
    }
