#!/usr/bin/env python3
"""
Orchestra Knowledge Capture — server entry point.

Usage:
    python3 server.py
    uvicorn server:app --reload --port 8080
"""
import sys
from pathlib import Path

# Ensure package root is on the path
sys.path.insert(0, str(Path(__file__).parent))

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config
from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

app = FastAPI(
    title="Orchestra Knowledge Capture",
    description="Pipe external knowledge into your Orchestra wiki.",
    version="0.1.0",
)

# Templates
templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))

# Static files (if ui/static/ exists)
static_dir = Path(__file__).parent / "ui" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Routes
app.include_router(router)

# Pass templates to routes module
from api import routes as _routes
_routes.templates = templates
_routes.kb_name = config.kb_name()


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=config.server_host(),
        port=config.server_port(),
        reload=False,
        log_level="info",
    )
