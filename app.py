# app.py - Hugging Face Spaces Version
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import logging
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MindCare AI Backend",
    description="Emotion Analysis API on Hugging Face",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
TOKENIZER = None
MODEL_LOADED = False

def load_model_once():
    """Load model once"""
    global MODEL, TOKENIZER, MODEL_LOADED
    
    if MODEL_LOADED:
        return MODEL, TOKENIZER
    
    try:
        logger.info("🔄 Loading emotion model...")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        
        model_name = "Kruti1234/mindcare-text-emotion"
        
        logger.info(f"📥 Loading from: {model_name}")
        TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        MODEL = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        MODEL.eval()
        for param in MODEL.parameters():
            param.requires_grad = False
        
        MODEL_LOADED = True
        gc.collect()
        
        logger.info("✅ Model loaded successfully!")
        return MODEL, TOKENIZER
        
    except Exception as e:
        logger.error(f"❌ Failed: {str(e)}")
        raise

class TextAnalysisRequest(BaseModel):
    text: str
    user_id: str = "user123"

class TextAnalysisResponse(BaseModel):
    status: str
    data: Dict[str, Any]
    processing_time: float

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting MindCare AI on Hugging Face...")
    load_model_once()
    logger.info("✅ Ready!")

@app.get("/")
async def root():
    return {
        "message": "MindCare AI Backend - Hugging Face",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": MODEL_LOADED
    }

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "mindcare-backend-hf",
        "model_loaded": MODEL_LOADED
    }

@app.post("/api/v1/analyze-text", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    import time
    import torch
    
    start_time = time.time()
    
    try:
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if len(request.text) > 1000:
            raise HTTPException(status_code=400, detail="Text too long")
        
        model, tokenizer = load_model_once()
        
        with torch.no_grad():
            inputs = tokenizer(
                request.text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        emotion_map = {0: "joy", 1: "sadness", 2: "anger", 3: "anxiety", 4: "neutral"}
        primary_emotion = emotion_map.get(predicted_class, "neutral")
        
        all_probabilities = {
            emotion_map[i]: float(probabilities[i].item())
            for i in range(len(probabilities))
        }
        
        wellness_scores = {"joy": 8.5, "neutral": 5.0, "anxiety": 3.5, "sadness": 3.0, "anger": 2.5}
        wellness_score = wellness_scores.get(primary_emotion, 5.0)
        
        interpretation = f"Analysis indicates {primary_emotion} emotion. "
        if wellness_score >= 7:
            interpretation += "You're experiencing positive emotions!"
        elif wellness_score >= 4:
            interpretation += "Your emotional state is balanced."
        else:
            interpretation += "Consider reaching out for support."
        
        crisis_detected = wellness_score < 3.0
        processing_time = time.time() - start_time
        
        response_data = {
            "emotion": {
                "primary": primary_emotion,
                "confidence": float(confidence),
                "all_probabilities": all_probabilities
            },
            "wellness_score": float(wellness_score),
            "interpretation": interpretation,
            "crisis_indicators": {
                "crisis_detected": crisis_detected,
                "recommendation": "Seek professional support" if crisis_detected else "Continue monitoring"
            }
        }
        
        del inputs, outputs, probabilities
        gc.collect()
        
        logger.info(f"✅ {primary_emotion} ({confidence:.1%})")
        
        return {
            "status": "success",
            "data": response_data,
            "processing_time": round(processing_time, 3)
        }
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))