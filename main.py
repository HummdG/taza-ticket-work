"""
Entry point for the WhatsApp Flight Booking Bot
Run this file to start the modular application
"""

import uvicorn
from app.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 