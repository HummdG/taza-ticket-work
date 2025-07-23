"""
Main FastAPI application for WhatsApp Flight Booking Bot
"""

import os
import uvicorn
import asyncio
import sys
import time
from typing import Optional
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




# Also ensure your webhook handler has the right logging
@app.post("/webhook")
async def webhook_handler_fast_response(request: Request):
    """Fast webhook handler with immediate response and detailed async processing"""
    try:
        # Parse form data quickly
        form_data = await request.form()
        message_text = form_data.get("Body", "")
        sender = form_data.get("From", "")
        media_url = str(form_data.get("MediaUrl0", "")) if form_data.get("MediaUrl0") else ""
        media_content_type = str(form_data.get("MediaContentType0", "")) if form_data.get("MediaContentType0") else ""
        
        print(f"🔍 Incoming webhook request headers: {dict(request.headers)}")
        print(f"🔍 Form data received: {dict(form_data)}")
        print(f"📱 Media info - URL: {media_url}, Type: {media_content_type}, Count: {form_data.get('NumMedia', '0')}")
        
        if (message_text and sender) or (media_url and sender):
            user_id = str(sender)
            is_voice_message = media_url and media_content_type and media_content_type.startswith('audio')
            
            if is_voice_message:
                print(f"🎤 Voice message detected from {user_id}")
                print(f"📱 Media URL: {media_url}")
                print(f"🎵 Content Type: {media_content_type}")
                
                # Process voice message asynchronously (don't wait)
                print(f"⚡ Starting async processing - detailed logs will follow...")
                asyncio.create_task(process_voice_message_async(
                    user_id, message_text, media_url, media_content_type
                ))
                
                # Return immediate response to Twilio
                twiml_response = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>🎤 Processing your voice message...</Message>
</Response>"""
                print(f"📡 Sending immediate TwiML response to prevent timeout")
                return PlainTextResponse(twiml_response, media_type="application/xml")
            
            else:
                # Handle text messages normally (with full logging)
                print(f"📝 Processing WhatsApp text message from {user_id}: '{message_text}'")
                
                # Import and use the detailed processing
                from app.services.message_handler import process_user_message
                text_response, _ = process_user_message(str(message_text), user_id)
                
                print(f"🤖 Generated text response: '{text_response}'")
                print(f"📝 Sending text response via TwiML")
                
                twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{text_response}</Message>
</Response>"""
                print(f"📡 Sending TwiML response: {twiml_response}")
                
                return PlainTextResponse(twiml_response, media_type="application/xml")
        
        # Default response
        return PlainTextResponse("""<?xml version="1.0" encoding="UTF-8"?>
<Response></Response>""", media_type="application/xml")
        
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        import traceback
        traceback.print_exc()
        # Always return valid TwiML
        return PlainTextResponse("""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>Sorry, I'm having technical difficulties. Please try again.</Message>
</Response>""", media_type="application/xml")

# Replace your webhook functions in app/main.py with these enhanced versions
async def process_voice_message_async(user_id: str, message_text: str, media_url: str, media_content_type: str):
    """Process voice message asynchronously with immediate log flushing"""
    
    def flush_print(*args, **kwargs):
        """Print with immediate flush to terminal"""
        print(*args, **kwargs)
        sys.stdout.flush()  # Force immediate output
    
    flush_print(f"\n{'='*80}")
    flush_print(f"🎤 ASYNC VOICE PROCESSING STARTED")
    flush_print(f"👤 User: {user_id}")
    flush_print(f"🔗 Media URL: {media_url}")
    flush_print(f"🎵 Content Type: {media_content_type}")
    flush_print(f"{'='*80}")
    
    # Small delay to ensure logs are visible
    await asyncio.sleep(0.1)
    
    try:
        # Import the detailed processing function
        from app.services.message_handler import process_user_message
        
        flush_print(f"🎤 Starting detailed voice message processing...")
        flush_print(f"📱 Processing WhatsApp voice message from {user_id}")
        
        # Add periodic flushes during processing
        class FlushingProcessor:
            @staticmethod
            def process_with_flushing():
                # Hook into the processing to add flushes
                original_print = print
                
                def flushing_print(*args, **kwargs):
                    original_print(*args, **kwargs)
                    sys.stdout.flush()
                    time.sleep(0.01)  # Tiny delay to ensure output
                
                # Temporarily replace print
                import builtins
                builtins.print = flushing_print
                
                try:
                    # Process the voice message with flushing prints
                    result = process_user_message(
                        str(message_text), 
                        user_id,
                        media_url=media_url,
                        media_content_type=media_content_type
                    )
                    return result
                finally:
                    # Restore original print
                    builtins.print = original_print
        
        # Process with flushing
        text_response, voice_file_url = FlushingProcessor.process_with_flushing()
        
        flush_print(f"\n🎉 VOICE PROCESSING COMPLETED!")
        flush_print(f"🤖 Generated text response: '{text_response[:100]}{'...' if len(text_response) > 100 else ''}'")
        if voice_file_url:
            flush_print(f"🎤 Generated voice file URL: {voice_file_url}")
        
        flush_print(f"\n📡 SENDING RESPONSE VIA TWILIO API...")
        
        # Send response via direct Twilio API call
        if voice_file_url:
            flush_print(f"🎤 Sending voice response via direct Twilio API...")
            await send_voice_response_direct_api(user_id, voice_file_url, text_response)
        else:
            flush_print(f"📝 Sending text response via direct Twilio API...")
            await send_text_response_direct_api(user_id, text_response)
        
        flush_print(f"\n✅ ASYNC VOICE PROCESSING COMPLETE")
        flush_print(f"{'='*80}\n")
            
    except Exception as e:
        flush_print(f"\n❌ ERROR IN ASYNC VOICE PROCESSING")
        flush_print(f"🔍 Error: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        flush_print(f"🔄 Sending error message to user...")
        await send_text_response_direct_api(user_id, "Sorry, I couldn't process your voice message. Please try again.")
        flush_print(f"{'='*80}\n")



async def send_voice_response_direct_api(to_number: str, voice_url: str, fallback_text: str):
    """Send voice response using direct Twilio API with detailed logging"""
    
    try:
        from twilio.rest import Client
        import os
        
        print(f"🔧 Initializing Twilio client for voice response...")
        
        # Initialize Twilio client
        twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        twilio_whatsapp_number = os.getenv('TWILIO_WHATSAPP_NUMBER', 'whatsapp:+14155238886')
        
        if not all([twilio_account_sid, twilio_auth_token]):
            print("❌ Twilio credentials not available")
            return
        
        client = Client(twilio_account_sid, twilio_auth_token)
        
        print(f"🎤 Sending voice response via direct Twilio API...")
        print(f"   📞 From: {twilio_whatsapp_number}")
        print(f"   📱 To: {to_number}")
        print(f"   🔗 Voice URL: {voice_url}")
        
        # Send voice message via direct API
        message = client.messages.create(
            from_=twilio_whatsapp_number,
            to=to_number,
            media_url=[voice_url]
        )
        
        print(f"✅ Voice message sent successfully!")
        print(f"   📞 Message SID: {message.sid}")
        print(f"   📊 Status: {message.status}")
        print(f"   🔗 Media URL in message: {message.media}")
        
        if message.error_code:
            print(f"❌ Twilio error detected!")
            print(f"   🔍 Error Code: {message.error_code}")
            print(f"   📝 Error Message: {message.error_message}")
            print(f"🔄 Falling back to text response...")
            # Fallback to text
            await send_text_response_direct_api(to_number, fallback_text)
        else:
            print(f"🎉 Voice message delivery initiated successfully!")
        
    except Exception as e:
        print(f"❌ Error sending voice response via Twilio API:")
        print(f"   🔍 Error: {e}")
        import traceback
        traceback.print_exc()
        print(f"🔄 Falling back to text message...")
        # Fallback to text message
        await send_text_response_direct_api(to_number, fallback_text)


async def send_text_response_direct_api(to_number: str, text: str):
    """Send text response using direct Twilio API with detailed logging"""
    
    try:
        from twilio.rest import Client
        import os
        
        print(f"🔧 Initializing Twilio client for text response...")
        
        twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        twilio_whatsapp_number = os.getenv('TWILIO_WHATSAPP_NUMBER', 'whatsapp:+14155238886')
        
        if not all([twilio_account_sid, twilio_auth_token]):
            print("❌ Twilio credentials not available")
            return
        
        client = Client(twilio_account_sid, twilio_auth_token)
        
        print(f"📝 Sending text response via direct Twilio API...")
        print(f"   📞 From: {twilio_whatsapp_number}")
        print(f"   📱 To: {to_number}")
        print(f"   💬 Message: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        message = client.messages.create(
            from_=twilio_whatsapp_number,
            to=to_number,
            body=text
        )
        
        print(f"✅ Text message sent successfully!")
        print(f"   📞 Message SID: {message.sid}")
        print(f"   📊 Status: {message.status}")
        
        if message.error_code:
            print(f"❌ Twilio error detected!")
            print(f"   🔍 Error Code: {message.error_code}")
            print(f"   📝 Error Message: {message.error_message}")
        else:
            print(f"🎉 Text message delivery initiated successfully!")
        
    except Exception as e:
        print(f"❌ Error sending text response via Twilio API:")
        print(f"   🔍 Error: {e}")
        import traceback
        traceback.print_exc()




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