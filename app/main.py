"""
Main FastAPI application for WhatsApp Flight Booking Bot
"""

import os
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

# Twilio client import - ADD THIS
try:
    from twilio.rest import Client
    # Initialize Twilio client
    twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    
    if twilio_account_sid and twilio_auth_token:
        twilio_client = Client(twilio_account_sid, twilio_auth_token)
        print("✅ Twilio client initialized successfully")
    else:
        twilio_client = None
        print("⚠️ Twilio credentials not found")
except ImportError:
    twilio_client = None
    print("⚠️ Twilio library not installed")

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
    """Clean webhook handler that sends voice via TwiML"""
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
                            media_url = f"https://graph.facebook.com/v18.0/{audio_info.get('id')}"
                            media_content_type = audio_info.get("mime_type", "audio/ogg")
                        
                        # Handle other media types from Meta WhatsApp API
                        elif message.get("type") in ["image", "video", "document"]:
                            media_info = message.get(message.get("type"), {})
                            media_url = f"https://graph.facebook.com/v18.0/{media_info.get('id')}"
                            media_content_type = media_info.get("mime_type", "")
                            
            except Exception as json_error:
                print(f"⚠️ Could not parse JSON: {json_error}")
        
        if (message_text and sender) or (media_url and sender):
            # Use sender as user_id for memory management
            user_id = str(sender)
            
            is_voice_message = media_url and media_content_type and media_content_type.startswith('audio')
            
            if is_voice_message:
                print(f"🎤 Voice message detected from {user_id}")
            elif media_url and media_content_type:
                print(f"📱 Processing WhatsApp media message from {user_id}: Type: {media_content_type}")
            else:
                print(f"📱 Processing WhatsApp text message from {user_id}: '{message_text}'")
            
            # Process the user message using the message handler with media support
            text_response, voice_file_url = process_user_message(
                str(message_text), 
                user_id, 
                media_url=media_url if media_url else None,
                media_content_type=media_content_type if media_content_type else None
            )
            
            print(f"🤖 Generated text response: '{text_response}'")
            if voice_file_url:
                print(f"🎤 Generated voice file URL: {voice_file_url}")
            
            # ✨ SIMPLE FIX: Create TwiML response with media URL if it's a voice response
            if voice_file_url and is_voice_message:
                print(f"🎤 Sending voice response via TwiML with media: {voice_file_url}")
                twiml_response = create_twiml_response(text_response, voice_file_url)
            else:
                print(f"📝 Sending text response via TwiML")
                twiml_response = create_twiml_response(text_response)
            
            print(f"📡 Sending TwiML response: {twiml_response}")
            
            return PlainTextResponse(twiml_response, media_type="application/xml")
        
        return {"status": "ok"}
        
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def send_voice_message_via_twilio(to_number: str, from_number: str, voice_file_url: str, text_fallback: str = "") -> bool:
    """
    Send voice message using Twilio WhatsApp API
    
    Args:
        to_number: Recipient WhatsApp number (format: whatsapp:+1234567890)
        from_number: Sender WhatsApp number (format: whatsapp:+1234567890)
        voice_file_url: Public URL to the voice file
        text_fallback: Fallback text if voice fails
        
    Returns:
        bool: True if sent successfully, False otherwise
    """
    
    if not twilio_client:
        print("❌ Twilio client not available for sending voice messages")
        return False
    
    try:
        print(f"🎤 Sending voice message to {to_number}")
        print(f"🔗 Voice URL: {voice_file_url}")
        
        # Send message with voice file as media
        message = twilio_client.messages.create(
            to=to_number,
            from_=from_number,
            media_url=[voice_file_url],
            body=text_fallback if text_fallback else ""  # Optional text alongside voice
        )
        
        print(f"✅ Voice message sent successfully!")
        print(f"📞 Message SID: {message.sid}")
        print(f"📊 Status: {message.status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to send voice message via Twilio: {e}")
        
        # Log specific Twilio errors
        if hasattr(e, 'code'):
            print(f"🔍 Twilio Error Code: {e.code}")
        if hasattr(e, 'msg'):
            print(f"🔍 Twilio Error Message: {e.msg}")
        
        return False


@app.post("/test-send-voice")
async def test_send_voice_endpoint(request: dict):
    """Test endpoint to send voice message directly"""
    
    text = request.get("text", "Hello! This is a test voice message from TazaTicket.")
    language = request.get("language", "en")
    user_id = request.get("user_id", "test_user")
    to_number = request.get("to_number")  # e.g., "whatsapp:+447948623631"
    
    if not to_number:
        return {"error": "to_number is required (format: whatsapp:+1234567890)"}
    
    try:
        # Generate voice response
        from app.services.message_handler import generate_voice_response, upload_voice_file_to_accessible_url
        
        print(f"🎤 Generating test voice: '{text}' in language: {language}")
        
        # Generate voice file
        voice_file_path = generate_voice_response(text, language, user_id)
        
        if not voice_file_path:
            return {"error": "Voice generation failed"}
        
        # Upload to S3 and get URL
        voice_file_url = upload_voice_file_to_accessible_url(voice_file_path, user_id)
        
        if not voice_file_url:
            return {"error": "Voice upload failed"}
        
        # Send via Twilio
        twilio_whatsapp_number = os.getenv('TWILIO_WHATSAPP_NUMBER', 'whatsapp:+14155238886')
        
        voice_sent = await send_voice_message_via_twilio(
            to_number=to_number,
            from_number=twilio_whatsapp_number,
            voice_file_url=voice_file_url,
            text_fallback=text
        )
        
        # Cleanup
        try:
            os.unlink(voice_file_path)
        except:
            pass
        
        if voice_sent:
            return {
                "success": True,
                "message": "Voice message sent successfully!",
                "voice_url": voice_file_url,
                "to_number": to_number
            }
        else:
            return {"error": "Failed to send voice message via Twilio"}
            
    except Exception as e:
        return {"error": str(e)}

@app.get("/test-presigned/{user_id}")
async def test_presigned_url_endpoint(user_id: str, text: str = "Testing presigned URL access"):
    """Generate and test a presigned URL"""
    
    try:
        from app.services.message_handler import generate_voice_response, upload_voice_file_to_accessible_url
        
        # Generate voice file
        voice_file_path = generate_voice_response(text, "en", user_id)
        if not voice_file_path:
            return {"error": "Voice generation failed"}
        
        # Upload and get presigned URL
        presigned_url = upload_voice_file_to_accessible_url(voice_file_path, user_id)
        if not presigned_url:
            return {"error": "Upload failed"}
        
        # Test the URL
        import requests
        test_response = requests.head(presigned_url, timeout=5)
        
        # Cleanup
        try:
            os.unlink(voice_file_path)
        except:
            pass
        
        return {
            "success": True,
            "presigned_url": presigned_url,
            "url_accessible": test_response.status_code == 200,
            "content_type": test_response.headers.get('content-type'),
            "file_size": test_response.headers.get('content-length'),
            "expires_in": "2 hours",
            "instructions": "Copy the presigned_url and paste it in your browser to play the voice file"
        }
        
    except Exception as e:
        return {"error": str(e)}


@app.post("/send-voice-test")
async def send_voice_test():
    """Quick test to send voice message to your number"""
    
    if not twilio_client:
        return {"error": "Twilio client not available"}
    
    try:
        # Use the S3 URL from your logs that we know works
        test_voice_url = "https://tazaticket.s3.eu-north-1.amazonaws.com/voice/whatsapp_447948623631/20250721_225720_8fe25972.mp3"
        
        twilio_whatsapp_number = os.getenv('TWILIO_WHATSAPP_NUMBER', 'whatsapp:+14155238886')
        
        message = twilio_client.messages.create(
            from_=twilio_whatsapp_number,
            to='whatsapp:+447948623631',
            body='🎤 Test voice message from TazaTicket!',
            media_url=[test_voice_url]
        )
        
        return {
            "success": True,
            "message_sid": message.sid,
            "status": message.status,
            "voice_url": test_voice_url,
            "message": "Voice message sent! Check your WhatsApp."
        }
        
    except Exception as e:
        return {"error": str(e)}


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
        if not twilio_client:
            return {"error": "Twilio client not available"}
        
        twilio_number = os.getenv('TWILIO_WHATSAPP_NUMBER', 'whatsapp:+14155238886')
        
        message = twilio_client.messages.create(
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