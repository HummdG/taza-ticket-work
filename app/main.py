"""
Main FastAPI application for WhatsApp Flight Booking Bot
"""

import os
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

# Local imports
from .models.schemas import TestMessage, WebhookResponse
from .services.message_handler import process_user_message, create_twiml_response
from .services.memory_service import memory_manager

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="WhatsApp Flight Booking Bot", 
    version="1.0.0",
    description="A modular flight booking bot using Travelport API and LangGraph"
)


@app.get("/")
async def root():
    """Root endpoint with basic API information"""
    return {
        "message": "WhatsApp Flight Booking Bot API", 
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Flight Booking Bot",
        "version": "1.0.0"
    }


@app.get("/webhook")
async def webhook_verify(request: Request):
    """Handle WhatsApp webhook verification"""
    verify_token = request.query_params.get("hub.verify_token")
    expected_token = os.getenv("WEBHOOK_VERIFY_TOKEN", "your_verify_token")
    
    if verify_token == expected_token:
        challenge = request.query_params.get("hub.challenge", "")
        return PlainTextResponse(challenge)
    
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def webhook_handler(request: Request):
    """Handle incoming WhatsApp messages"""
    try:
        # Debug: Log all incoming data
        print(f"🔍 Incoming webhook request headers: {dict(request.headers)}")
        
        # Handle form data (Twilio format)
        form_data = await request.form()
        print(f"🔍 Form data received: {dict(form_data)}")
        
        message_text = form_data.get("Body", "")
        sender = form_data.get("From", "")
        
        # Extract media information for voice messages, images, etc.
        media_url = str(form_data.get("MediaUrl0", "")) if form_data.get("MediaUrl0") else ""
        media_content_type = str(form_data.get("MediaContentType0", "")) if form_data.get("MediaContentType0") else ""
        num_media = str(form_data.get("NumMedia", "0"))
        
        print(f"📱 Media info - URL: {media_url}, Type: {media_content_type}, Count: {num_media}")
        
        if not message_text:
            # Try JSON format (Meta WhatsApp Business API)
            try:
                json_data = await request.json()
                print(f"🔍 JSON data received: {json_data}")
                if json_data and "messages" in json_data.get("entry", [{}])[0].get("changes", [{}])[0].get("value", {}):
                    messages = json_data["entry"][0]["changes"][0]["value"]["messages"]
                    if messages:
                        message = messages[0]
                        message_text = message.get("text", {}).get("body", "")
                        sender = message.get("from", "")
                        
                        # Handle voice messages from Meta WhatsApp API
                        if message.get("type") == "audio":
                            audio_info = message.get("audio", {})
                            media_url = f"https://graph.facebook.com/v18.0/{audio_info.get('id')}"  # Will need auth
                            media_content_type = audio_info.get("mime_type", "audio/ogg")
                        
                        # Handle other media types from Meta WhatsApp API
                        elif message.get("type") in ["image", "video", "document"]:
                            media_info = message.get(message.get("type"), {})
                            media_url = f"https://graph.facebook.com/v18.0/{media_info.get('id')}"  # Will need auth
                            media_content_type = media_info.get("mime_type", "")
                            
            except Exception as json_error:
                print(f"⚠️ Could not parse JSON: {json_error}")
        
        if (message_text and sender) or (media_url and sender):
            # Use sender as user_id for memory management
            user_id = str(sender)
            
            if media_url and media_content_type:
                print(f"📱 Processing WhatsApp media message from {user_id}: Type: {media_content_type}")
                if media_content_type.startswith('audio'):
                    print(f"🎤 Voice message detected from {user_id}")
            else:
                print(f"📱 Processing WhatsApp text message from {user_id}: '{message_text}'")
            
            # Process the user message using the message handler with memory and media support
            bot_response = process_user_message(
                str(message_text), 
                user_id, 
                media_url=media_url if media_url else None,
                media_content_type=media_content_type if media_content_type else None
            )
            
            print(f"🤖 Generated response: '{bot_response}'")
            
            # Return TwiML response for Twilio
            twiml_response = create_twiml_response(bot_response)
            
            print(f"📡 Sending TwiML response: {twiml_response}")
            
            return PlainTextResponse(twiml_response, media_type="application/xml")
        
        return {"status": "ok"}
        
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/test")
async def test_endpoint(message: TestMessage):
    """Test endpoint for manual testing"""
    user_id = message.user_id or "test_user"
    response = process_user_message(
        message.message, 
        user_id,
        media_url=None,  # Could be extended to test media
        media_content_type=None
    )
    return {"response": response, "user_id": user_id}


@app.post("/test-webhook")
async def test_webhook_format():
    """Test the exact webhook response format"""
    test_response = "Hello! This is a test response from TazaTicket."
    twiml_response = create_twiml_response(test_response)
    print(f"📡 Test TwiML: {twiml_response}")
    return PlainTextResponse(twiml_response, media_type="application/xml")


@app.get("/memory/stats")
async def memory_stats():
    """Get memory usage statistics"""
    stats = memory_manager.get_memory_stats()
    return {"memory_stats": stats}


@app.post("/memory/clear/{user_id}")
async def clear_user_memory(user_id: str):
    """Clear memory for a specific user"""
    memory_manager.clear_user_memory(user_id)
    return {"message": f"Memory cleared for user: {user_id}"}


@app.post("/memory/cleanup")
async def cleanup_expired_memories():
    """Clean up expired memories"""
    memory_manager.cleanup_expired_memories()
    stats = memory_manager.get_memory_stats()
    return {"message": "Expired memories cleaned up", "current_stats": stats}


@app.get("/flight-collection/{user_id}")
async def get_flight_collection_state(user_id: str):
    """Get current flight collection state for a user"""
    is_collecting = memory_manager.is_collecting_flight_info(user_id)
    collection_state = memory_manager.get_flight_collection_state(user_id)
    return {
        "user_id": user_id,
        "is_collecting": is_collecting,
        "collection_state": collection_state
    }


@app.post("/flight-collection/clear/{user_id}")
async def clear_flight_collection_state(user_id: str):
    """Clear flight collection state for a user"""
    memory_manager.clear_flight_collection_state(user_id)
    return {"message": f"Flight collection state cleared for user: {user_id}"}


@app.get("/test/send-message")
async def test_send_message():
    """Test endpoint to send a direct message via Twilio"""
    try:
        from twilio.rest import Client
        import os
        
        # Get Twilio credentials
        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        twilio_number = os.getenv('TWILIO_WHATSAPP_NUMBER', 'whatsapp:+14155238886')
        
        if not all([account_sid, auth_token]):
            return {"error": "Missing Twilio credentials"}
        
        client = Client(account_sid, auth_token)
        
        message = client.messages.create(
            body="Test message from TazaTicket bot - direct send",
            from_=twilio_number,
            to='whatsapp:+447948623631'
        )
        
        return {
            "success": True,
            "message_sid": message.sid,
            "status": message.status,
            "body": message.body
        }
        
    except Exception as e:
        return {"error": str(e)}


@app.post("/test-voice")
async def test_voice_endpoint(message: dict):
    """Test endpoint for voice message testing"""
    user_id = message.get("user_id", "test_user")
    media_url = message.get("media_url")
    media_content_type = message.get("media_content_type", "audio/ogg")
    
    if not media_url:
        return {"error": "media_url is required for voice message testing"}
    
    response = process_user_message(
        "", 
        user_id,
        media_url=media_url,
        media_content_type=media_content_type
    )
    return {"response": response, "user_id": user_id, "transcription_attempted": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 