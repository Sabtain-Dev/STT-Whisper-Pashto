# api/main.py

# This file serves as the main entry point for starting the FastAPI engine. It configures app wide metadata parameters and mounts your application routing blocks.
# (The App Gatekeeper): This file is the entry point. It initializes the FastAPI framework application instance, configures global settings (like security CORS rules), defines documentation metadata (title, version, docs URLs), and links the endpoint paths.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

# Define robust API Metadata specifications
app = FastAPI(
    title="Pashto Whisper STT API",
    description="Production-ready FastAPI backend for regional Pakistani Pashto Speech-to-Text inference.",
    version="1.3",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS Middleware for robust local/remote endpoint communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust explicitly to match your deploy schema ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include decoupled service endpoints matrix router
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    # Execute with optimal worker threading mapping for resource efficiency
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)