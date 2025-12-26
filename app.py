# # app.py - Hugging Face Spaces Version
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Dict, Any
# import logging
# import gc

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI(
#     title="MindCare AI Backend",
#     description="Emotion Analysis API on Hugging Face",
#     version="1.0.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# MODEL = None
# TOKENIZER = None
# MODEL_LOADED = False

# def load_model_once():
#     """Load model once"""
#     global MODEL, TOKENIZER, MODEL_LOADED
    
#     if MODEL_LOADED:
#         return MODEL, TOKENIZER
    
#     try:
#         logger.info("🔄 Loading emotion model...")
#         from transformers import AutoModelForSequenceClassification, AutoTokenizer
#         import torch
        
#         model_name = "Kruti1234/mindcare-text-emotion"
        
#         logger.info(f"📥 Loading from: {model_name}")
#         TOKENIZER = AutoTokenizer.from_pretrained(model_name)
#         MODEL = AutoModelForSequenceClassification.from_pretrained(model_name)
        
#         MODEL.eval()
#         for param in MODEL.parameters():
#             param.requires_grad = False
        
#         MODEL_LOADED = True
#         gc.collect()
        
#         logger.info("✅ Model loaded successfully!")
#         return MODEL, TOKENIZER
        
#     except Exception as e:
#         logger.error(f"❌ Failed: {str(e)}")
#         raise

# class TextAnalysisRequest(BaseModel):
#     text: str
#     user_id: str = "user123"

# class TextAnalysisResponse(BaseModel):
#     status: str
#     data: Dict[str, Any]
#     processing_time: float

# @app.on_event("startup")
# async def startup_event():
#     logger.info("🚀 Starting MindCare AI on Hugging Face...")
#     load_model_once()
#     logger.info("✅ Ready!")

# @app.get("/")
# async def root():
#     return {
#         "message": "MindCare AI Backend - Hugging Face",
#         "version": "1.0.0",
#         "status": "running",
#         "model_loaded": MODEL_LOADED
#     }

# @app.get("/api/v1/health")
# async def health_check():
#     return {
#         "status": "healthy",
#         "service": "mindcare-backend-hf",
#         "model_loaded": MODEL_LOADED
#     }

# @app.post("/api/v1/analyze-text", response_model=TextAnalysisResponse)
# async def analyze_text(request: TextAnalysisRequest):
#     import time
#     import torch
    
#     start_time = time.time()
    
#     try:
#         if not request.text or len(request.text.strip()) == 0:
#             raise HTTPException(status_code=400, detail="Text cannot be empty")
        
#         if len(request.text) > 1000:
#             raise HTTPException(status_code=400, detail="Text too long")
        
#         model, tokenizer = load_model_once()
        
#         with torch.no_grad():
#             inputs = tokenizer(
#                 request.text,
#                 return_tensors="pt",
#                 padding=True,
#                 truncation=True,
#                 max_length=128
#             )
            
#             outputs = model(**inputs)
#             probabilities = torch.softmax(outputs.logits, dim=1)[0]
#             predicted_class = torch.argmax(probabilities).item()
#             confidence = probabilities[predicted_class].item()
        
#         emotion_map = {0: "joy", 1: "sadness", 2: "anger", 3: "anxiety", 4: "neutral"}
#         primary_emotion = emotion_map.get(predicted_class, "neutral")
        
#         all_probabilities = {
#             emotion_map[i]: float(probabilities[i].item())
#             for i in range(len(probabilities))
#         }
        
#         wellness_scores = {"joy": 8.5, "neutral": 5.0, "anxiety": 3.5, "sadness": 3.0, "anger": 2.5}
#         wellness_score = wellness_scores.get(primary_emotion, 5.0)
        
#         interpretation = f"Analysis indicates {primary_emotion} emotion. "
#         if wellness_score >= 7:
#             interpretation += "You're experiencing positive emotions!"
#         elif wellness_score >= 4:
#             interpretation += "Your emotional state is balanced."
#         else:
#             interpretation += "Consider reaching out for support."
        
#         crisis_detected = wellness_score < 3.0
#         processing_time = time.time() - start_time
        
#         response_data = {
#             "emotion": {
#                 "primary": primary_emotion,
#                 "confidence": float(confidence),
#                 "all_probabilities": all_probabilities
#             },
#             "wellness_score": float(wellness_score),
#             "interpretation": interpretation,
#             "crisis_indicators": {
#                 "crisis_detected": crisis_detected,
#                 "recommendation": "Seek professional support" if crisis_detected else "Continue monitoring"
#             }
#         }
        
#         del inputs, outputs, probabilities
#         gc.collect()
        
#         logger.info(f"✅ {primary_emotion} ({confidence:.1%})")
        
#         return {
#             "status": "success",
#             "data": response_data,
#             "processing_time": round(processing_time, 3)
#         }
        
#     except Exception as e:
#         logger.error(f"❌ Error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# app.py - MindCare Backend with TEXT + AUDIO Support
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import logging
import gc
import io
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MindCare AI Backend",
    description="Text + Audio Emotion Analysis API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models
TEXT_MODEL = None
TEXT_TOKENIZER = None
AUDIO_MODEL = None
AUDIO_LABEL_ENCODER = None
MODELS_LOADED = {"text": False, "audio": False}

def load_text_model():
    """Load text emotion model (BERT)"""
    global TEXT_MODEL, TEXT_TOKENIZER, MODELS_LOADED
    
    if MODELS_LOADED["text"]:
        return TEXT_MODEL, TEXT_TOKENIZER
    
    try:
        logger.info("📝 Loading text emotion model...")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        model_name = "Kruti1234/mindcare-text-emotion"
        TEXT_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        TEXT_MODEL = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        TEXT_MODEL.eval()
        for param in TEXT_MODEL.parameters():
            param.requires_grad = False
        
        MODELS_LOADED["text"] = True
        gc.collect()
        
        logger.info("✅ Text model loaded!")
        return TEXT_MODEL, TEXT_TOKENIZER
        
    except Exception as e:
        logger.error(f"❌ Text model failed: {str(e)}")
        raise

def load_audio_model():
    """Load audio emotion model (CNN)"""
    global AUDIO_MODEL, AUDIO_LABEL_ENCODER, MODELS_LOADED
    
    if MODELS_LOADED["audio"]:
        return AUDIO_MODEL, AUDIO_LABEL_ENCODER
    
    try:
        logger.info("🎤 Loading audio emotion model...")
        from huggingface_hub import hf_hub_download
        import tensorflow as tf
        import pickle
        
        repo_id = "Kruti1234/mindcare-audio-emotion"
        
        # Download model
        model_path = hf_hub_download(repo_id=repo_id, filename="mindcare_audio_emotion_model.h5")
        AUDIO_MODEL = tf.keras.models.load_model(model_path)
        logger.info("✅ Audio model loaded!")
        
        # Download label encoder
        encoder_path = hf_hub_download(repo_id=repo_id, filename="audio_label_encoder.pkl")
        with open(encoder_path, 'rb') as f:
            AUDIO_LABEL_ENCODER = pickle.load(f)
        logger.info("✅ Audio label encoder loaded!")
        
        MODELS_LOADED["audio"] = True
        gc.collect()
        
        return AUDIO_MODEL, AUDIO_LABEL_ENCODER
        
    except Exception as e:
        logger.error(f"❌ Audio model failed: {str(e)}")
        raise

# Request/Response models
class TextAnalysisRequest(BaseModel):
    text: str
    user_id: str = "user123"

class TextAnalysisResponse(BaseModel):
    status: str
    data: Dict[str, Any]
    processing_time: float

class AudioAnalysisResponse(BaseModel):
    status: str
    data: Dict[str, Any]
    processing_time: float

@app.on_event("startup")
async def startup_event():
    """Pre-load models on startup"""
    try:
        logger.info("🚀 Starting MindCare AI Backend...")
        logger.info("📝 Loading text model...")
        load_text_model()
        logger.info("🎤 Loading audio model...")
        load_audio_model()
        logger.info("✅ All models loaded!")
    except Exception as e:
        logger.error(f"⚠️ Startup error: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "MindCare AI Backend - Text + Audio",
        "version": "2.0.0",
        "status": "running",
        "models_loaded": MODELS_LOADED
    }

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "mindcare-backend",
        "text_model_loaded": MODELS_LOADED["text"],
        "audio_model_loaded": MODELS_LOADED["audio"]
    }

@app.post("/api/v1/analyze-text", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """Analyze emotion from text"""
    import time
    import torch
    
    start_time = time.time()
    
    try:
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if len(request.text) > 1000:
            raise HTTPException(status_code=400, detail="Text too long (max 1000 chars)")
        
        model, tokenizer = load_text_model()
        
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
            },
            "modality": "text"
        }
        
        del inputs, outputs, probabilities
        gc.collect()
        
        logger.info(f"✅ Text: {primary_emotion} ({confidence:.1%})")
        
        return {
            "status": "success",
            "data": response_data,
            "processing_time": round(processing_time, 3)
        }
        
    except Exception as e:
        logger.error(f"❌ Text analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analyze-audio", response_model=AudioAnalysisResponse)
async def analyze_audio(audio_file: UploadFile = File(...), user_id: str = "user123"):
    """Analyze emotion from audio"""
    import time
    import librosa
    
    start_time = time.time()
    
    try:
        # Validate file type
        if not audio_file.filename.endswith(('.wav', '.mp3', '.m4a', '.ogg')):
            raise HTTPException(status_code=400, detail="Invalid audio format. Use WAV, MP3, M4A, or OGG")
        
        # Read audio file
        audio_bytes = await audio_file.read()
        audio_data, sample_rate = librosa.load(io.BytesIO(audio_bytes), res_type='kaiser_fast', duration=3)
        
        # Extract MFCC features (same as training)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        
        # Pad or truncate to 174 time steps
        max_pad_len = 174
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        
        # Reshape for model (add batch and channel dimensions)
        features = mfccs.reshape(1, 40, 174, 1)
        
        # Load model and predict
        model, label_encoder = load_audio_model()
        
        predictions = model.predict(features, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Get emotion label
        primary_emotion = label_encoder.inverse_transform([predicted_class])[0]
        
        # All probabilities
        all_probabilities = {
            label_encoder.inverse_transform([i])[0]: float(predictions[0][i])
            for i in range(len(predictions[0]))
        }
        
        # Wellness score (map audio emotions to scores)
        wellness_map = {
            "neutral": 5.0, "calm": 6.0, "happy": 8.5,
            "sad": 3.0, "angry": 2.5, "fearful": 3.5, "disgust": 3.0
        }
        wellness_score = wellness_map.get(primary_emotion, 5.0)
        
        interpretation = f"Voice analysis indicates {primary_emotion} emotion. "
        if wellness_score >= 7:
            interpretation += "You're expressing positive emotions in your voice!"
        elif wellness_score >= 4:
            interpretation += "Your vocal tone suggests balanced emotions."
        else:
            interpretation += "Your voice suggests you might benefit from support."
        
        crisis_detected = wellness_score < 3.0
        processing_time = time.time() - start_time
        
        response_data = {
            "emotion": {
                "primary": primary_emotion,
                "confidence": confidence,
                "all_probabilities": all_probabilities
            },
            "wellness_score": wellness_score,
            "interpretation": interpretation,
            "crisis_indicators": {
                "crisis_detected": crisis_detected,
                "recommendation": "Seek professional support" if crisis_detected else "Continue monitoring"
            },
            "modality": "audio",
            "audio_duration": len(audio_data) / sample_rate
        }
        
        gc.collect()
        
        logger.info(f"✅ Audio: {primary_emotion} ({confidence:.1%})")
        
        return {
            "status": "success",
            "data": response_data,
            "processing_time": round(processing_time, 3)
        }
        
    except Exception as e:
        logger.error(f"❌ Audio analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
