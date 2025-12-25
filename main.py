from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="MindCare AI", version="2.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "MindCare AI API", "status": "online"}

if __name__ == "__main__":
    print("Starting server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)