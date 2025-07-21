"""
Message handling service for processing all types of user requests with memory
"""

import os
import requests
import tempfile
from typing import Optional
from langchain_core.messages import HumanMessage
from ..models.schemas import FlightBookingState
from ..agents.flight_booking_agent import flight_booking_agent
from ..agents.general_conversation_agent import handle_general_conversation
from ..services.conversation_router import should_handle_as_flight_booking, should_collect_flight_info, analyze_flight_request_completeness
from ..services.memory_service import memory_manager
from ..services.flight_info_collector import flight_collector

# Import OpenAI for Whisper API
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except ImportError:
    print("⚠️ OpenAI library not installed. Voice messages will not work.")
    client = None


def process_user_message(user_message: str, user_id: str = "unknown", media_url: Optional[str] = None, media_content_type: Optional[str] = None) -> str:
    """
    Process any user message by routing to appropriate handler with memory
    
    Args:
        user_message: The user's message (could be empty for voice-only messages)
        user_id: Unique identifier for the user
        media_url: URL to media content (for voice messages, images, etc.)
        media_content_type: MIME type of the media
        
    Returns:
        str: Appropriate response based on message type
    """
    
    try:
        # Handle voice messages first
        if media_url and media_content_type and media_content_type.startswith('audio'):
            print(f"🎤 Processing voice message from user: {user_id}")
            
            # Transcribe the voice message
            transcribed_text = transcribe_voice_message(media_url, media_content_type)
            
            # Check if transcription failed
            if transcribed_text.startswith("🎤 I"):
                # Return error message directly
                memory_manager.add_conversation(user_id, "[Voice Message]", transcribed_text, "voice_error")
                return transcribed_text
            
            # Add emoji indicator for voice message and process transcribed text
            voice_indicator = "🎤 "
            response = process_transcribed_voice_message(transcribed_text, user_id, voice_indicator)
            
            # Save to memory with voice message indicator
            memory_manager.add_conversation(user_id, f"🎤 [Voice]: {transcribed_text}", response, "voice")
            return response
        
        # Handle other media types (images, documents, etc.)
        elif media_url and media_content_type:
            media_type = media_content_type.split('/')[0]
            if media_type == 'image':
                response = "🖼️ I can see you sent an image! While I can't analyze images yet, feel free to describe what you'd like help with regarding your travel plans."
            elif media_type == 'video':
                response = "🎬 I received your video! While I can't analyze videos yet, please let me know how I can help with your flight booking needs."
            elif media_type == 'application':  # Documents, PDFs, etc.
                response = "📄 I received your document! While I can't read documents yet, feel free to tell me what information you need help with for your travel."
            else:
                response = "📎 I received your file! While I can't process files yet, please let me know how I can assist you with flight booking."
            
            memory_manager.add_conversation(user_id, f"[{media_type.title()}]", response, "media")
            return response
        
        # Handle regular text messages (existing logic)
        # Get conversation context from memory
        conversation_context = memory_manager.get_conversation_context(user_id)
        
        # Check if user is currently collecting flight information
        is_collecting = memory_manager.is_collecting_flight_info(user_id)
        
        if is_collecting:
            print(f"🔄 Continuing flight info collection for user: {user_id}")
            response = handle_flight_info_collection(user_message, user_id, conversation_context)
            message_type = "flight_collection"
        
        # Check if this is a complete flight booking request
        elif should_handle_as_flight_booking(user_message, conversation_context):
            print(f"🛫 Routing to flight booking agent for user: {user_id}")
            response = process_flight_request(user_message, user_id, conversation_context)
            message_type = "flight"
        
        # Check if this is a partial flight request that needs more info
        elif should_collect_flight_info(user_message, conversation_context):
            print(f"📝 Starting flight info collection for user: {user_id}")
            response = start_flight_info_collection(user_message, user_id, conversation_context)
            message_type = "flight_collection"
        
        # Default to general conversation
        else:
            print(f"💬 Routing to general conversation agent for user: {user_id}")
            response = handle_general_conversation(user_message, user_id, conversation_context)
            message_type = "general"
        
        # Save the conversation to memory
        memory_manager.add_conversation(user_id, user_message, response, message_type)
        
        return response
            
    except Exception as e:
        print(f"❌ Error processing message: {e}")
        error_response = "😅 I'm having trouble processing your request right now. Please try again later!"
        
        # Still save error conversations to memory
        try:
            memory_manager.add_conversation(user_id, user_message, error_response, "error")
        except:
            pass
            
        return error_response


def start_flight_info_collection(user_message: str, user_id: str, conversation_context: str) -> str:
    """
    Start collecting flight information from a partial request
    
    Args:
        user_message: The user's message with partial flight info
        user_id: Unique identifier for the user
        conversation_context: Previous conversation context
        
    Returns:
        str: Question to collect missing information
    """
    
    try:
        # Extract available flight information
        has_intent, is_complete, extracted_info = analyze_flight_request_completeness(user_message, conversation_context)
        
        if not has_intent:
            # Shouldn't happen, but fallback to general conversation
            return handle_general_conversation(user_message, user_id, conversation_context)
        
        if is_complete:
            # Complete information, route to flight search
            return process_flight_request(user_message, user_id, conversation_context)
        
        # Save the collection state
        collection_state = {
            "collecting": True,
            "collected_info": extracted_info,
            "started_with": user_message
        }
        memory_manager.set_flight_collection_state(user_id, collection_state)
        
        # Generate question for missing information
        missing_fields = flight_collector.identify_missing_info(extracted_info)
        question = flight_collector.generate_question_for_missing_info(missing_fields, extracted_info)
        
        return question
        
    except Exception as e:
        print(f"❌ Error starting flight info collection: {e}")
        return "✈️ I'd be happy to help you find a flight! 😊 Could you tell me where you'd like to fly from, where you want to go, and when you'd like to travel?"


def handle_flight_info_collection(user_message: str, user_id: str, conversation_context: str) -> str:
    """
    Continue collecting flight information from user responses
    
    Args:
        user_message: The user's response with additional flight info
        user_id: Unique identifier for the user
        conversation_context: Previous conversation context
        
    Returns:
        str: Next question or flight search results
    """
    
    try:
        # Get current collection state
        collection_state = memory_manager.get_flight_collection_state(user_id)
        current_info = collection_state.get("collected_info", {})
        
        # Extract new information from user message
        new_info = flight_collector.extract_flight_info(user_message, conversation_context)
        
        # Merge with existing information
        merged_info = flight_collector.merge_flight_info(current_info, new_info)
        
        # Check if we now have complete information
        if flight_collector.is_flight_info_complete(merged_info):
            # Complete! Clear collection state and search for flights
            memory_manager.clear_flight_collection_state(user_id)
            
            # Create a summary and proceed to flight search
            summary = flight_collector.format_collected_info_summary(merged_info)
            
            # Construct a complete flight request message
            from_city = merged_info.get("from_city", "")
            to_city = merged_info.get("to_city", "")
            departure_date = merged_info.get("departure_date", "")
            
            complete_request = f"Find flights from {from_city} to {to_city} on {departure_date}"
            
            # Process as complete flight request
            flight_response = process_flight_request(complete_request, user_id, conversation_context)
            
            return f"{summary}\n\n{flight_response}"
        
        else:
            # Still missing information, update state and ask for more
            collection_state["collected_info"] = merged_info
            memory_manager.set_flight_collection_state(user_id, collection_state)
            
            # Generate next question
            missing_fields = flight_collector.identify_missing_info(merged_info)
            question = flight_collector.generate_question_for_missing_info(missing_fields, merged_info)
            
            return question
        
    except Exception as e:
        print(f"❌ Error handling flight info collection: {e}")
        # Clear collection state on error
        memory_manager.clear_flight_collection_state(user_id)
        return "😅 I had trouble processing that information. Let's start over - ✈️ could you tell me where you'd like to fly from, where you want to go, and when?"


def process_flight_request(user_message: str, user_id: str = "unknown", conversation_context: str = "") -> str:
    """
    Process a flight booking request using the LangGraph agent with memory
    
    Args:
        user_message: The user's flight request message
        user_id: Unique identifier for the user
        conversation_context: Previous conversation history
        
    Returns:
        str: Formatted response with flight information or error message
    """
    
    try:
        # Initialize state with memory context
        initial_state: FlightBookingState = {
        "messages": [HumanMessage(content=user_message)],
        "user_message": user_message,
        "user_id": user_id,
        "from_city": None,
        "to_city": None,
        "departure_date": None,
        "return_date": None,
        "passengers": 1,
        "passenger_age": 25,
        "raw_api_response": None,
        "cheapest_flight": None,
        "response_text": "",
        "conversation_context": conversation_context,
        "search_type": None,
        "date_range_start": None,
        "date_range_end": None,
        "range_description": None,
        "bulk_search_results": None,
        "search_dates": None,
        "best_departure_date": None,
    }
        
        # Run the agent
        final_state = flight_booking_agent.invoke(initial_state)
        
        # Save flight context to memory for future reference
        if final_state.get("cheapest_flight"):
            flight_context = {
                "last_search": {
                    "from_city": final_state.get("from_city"),
                    "to_city": final_state.get("to_city"),
                    "departure_date": final_state.get("departure_date"),
                    "return_date": final_state.get("return_date"),
                    "passengers": final_state.get("passengers"),
                }
            }
            memory_manager.add_flight_context(user_id, flight_context)
        
        # Return the response
        return final_state.get("response_text", "😔 I'm sorry, I couldn't process your flight request.")
        
    except Exception as e:
        print(f"❌ Error processing flight request: {e}")
        return "I'm having trouble processing your flight request right now. Please try again later."


def format_whatsapp_response(bot_response: str) -> str:
    """
    Format bot response for WhatsApp delivery
    
    Args:
        bot_response: The bot's response text
        
    Returns:
        str: WhatsApp-formatted response
    """
    # For now, just return the response as-is
    # In the future, this could handle WhatsApp-specific formatting
    return bot_response


def create_twiml_response(message: str) -> str:
    """
    Create TwiML response for Twilio WhatsApp integration
    
    Args:
        message: The message to send
        
    Returns:
        str: TwiML formatted response
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{message}</Message>
</Response>"""


def transcribe_voice_message(media_url: str, media_content_type: str) -> str:
    """
    Transcribe a voice message using OpenAI Whisper API
    
    Args:
        media_url: URL to download the voice message
        media_content_type: MIME type of the audio file
        
    Returns:
        str: Transcribed text from the voice message
    """
    
    if not client:
        return "🎤 I received your voice message, but voice transcription is not available right now. Could you please type your message instead?"
    
    try:
        print(f"🎤 Downloading voice message from: {media_url}")
        
        # Set up authentication for Twilio media URLs
        auth = None
        if "api.twilio.com" in media_url:
            # Use Twilio credentials for authentication
            twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
            
            if not twilio_account_sid or not twilio_auth_token:
                print("⚠️ Twilio credentials not found. Trying without authentication...")
            else:
                auth = (twilio_account_sid, twilio_auth_token)
                print(f"🔐 Using Twilio authentication for media download")
        
        # Download the audio file with authentication if needed
        response = requests.get(media_url, timeout=30, auth=auth)
        response.raise_for_status()
        
        print(f"✅ Media file downloaded successfully ({len(response.content)} bytes)")
        
        # Determine file extension based on content type
        extension = ".ogg"  # Default for WhatsApp voice messages
        if "mp4" in media_content_type:
            extension = ".mp4"
        elif "mpeg" in media_content_type:
            extension = ".mp3"
        elif "wav" in media_content_type:
            extension = ".wav"
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        
        print(f"🎤 Transcribing voice message...")
        
        # Transcribe using OpenAI Whisper
        with open(temp_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        if transcript and transcript.strip():
            print(f"✅ Voice message transcribed: {transcript}")
            return transcript.strip()
        else:
            return "🎤 I couldn't understand your voice message. Could you please try again or type your message?"
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading voice message: {e}")
        if "401" in str(e):
            return "🎤 I had trouble accessing your voice message. Please check the Twilio authentication settings and try again."
        return "🎤 I had trouble downloading your voice message. Could you please try sending it again?"
        
    except Exception as e:
        print(f"❌ Error transcribing voice message: {e}")
        return "🎤 I had trouble understanding your voice message. Could you please try again or type your message instead?"


def process_transcribed_voice_message(transcribed_text: str, user_id: str, voice_indicator: str) -> str:
    """
    Process transcribed voice message text through the normal message handling flow
    
    Args:
        transcribed_text: The transcribed text from voice message
        user_id: Unique identifier for the user
        voice_indicator: Emoji indicator for voice message
        
    Returns:
        str: Response with voice message indicator
    """
    
    # Get conversation context from memory
    conversation_context = memory_manager.get_conversation_context(user_id)
    
    # Check if user is currently collecting flight information
    is_collecting = memory_manager.is_collecting_flight_info(user_id)
    
    if is_collecting:
        print(f"🔄 Continuing flight info collection for voice user: {user_id}")
        response = handle_flight_info_collection(transcribed_text, user_id, conversation_context)
    
    # Check if this is a complete flight booking request
    elif should_handle_as_flight_booking(transcribed_text, conversation_context):
        print(f"🛫 Routing voice message to flight booking agent for user: {user_id}")
        response = process_flight_request(transcribed_text, user_id, conversation_context)
    
    # Check if this is a partial flight request that needs more info
    elif should_collect_flight_info(transcribed_text, conversation_context):
        print(f"📝 Starting flight info collection from voice for user: {user_id}")
        response = start_flight_info_collection(transcribed_text, user_id, conversation_context)
    
    # Default to general conversation
    else:
        print(f"💬 Routing voice message to general conversation agent for user: {user_id}")
        response = handle_general_conversation(transcribed_text, user_id, conversation_context)
    
    # Add voice indicator to response
    return f"{voice_indicator}{response}" 