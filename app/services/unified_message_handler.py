"""
Unified Message Handler for TazaTicket
Integrates the unified conversation agent with existing voice/text handling
Maintains backward compatibility while supporting enhanced slot filling
"""

import os
import json
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

from ..models.schemas import ConversationState, ConversationResponse
from ..agents.unified_conversation_agent import process_conversation_turn
from ..services.memory_service import memory_manager
from ..api.travelport import search_flights
from ..agents.flight_booking_agent import flight_booking_agent
from ..agents.general_conversation_agent import handle_general_conversation
from ..services.conversation_router import should_handle_as_flight_booking
from .message_handler import (
    SecureTazaTicketS3Handler, 
    transcribe_voice_message, 
    generate_voice_response,
    upload_voice_file_to_accessible_url,
    create_twiml_response
)


def detect_response_mode(media_url: Optional[str] = None, user_preference: str = "text") -> str:
    """Detect if user sent voice message to determine response mode"""
    if media_url and "audio" in str(media_url).lower():
        return "speech"
    return user_preference


def should_use_unified_agent(user_message: str, user_id: str) -> bool:
    """
    Determine if we should use the unified conversation agent
    Use unified agent for flight booking requests and active conversations
    """
    # Check if there's an active conversation with filled slots
    memory = memory_manager.get_user_memory(user_id)
    has_active = memory.has_active_conversation()
    
    # Check if this message is flight-related
    is_flight_related = should_handle_as_flight_booking(user_message)
    
    # Use unified agent if either condition is true
    return has_active or is_flight_related


async def process_unified_message(
    user_message: str,
    user_id: str,
    media_url: Optional[str] = None,
    media_content_type: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """
    Process message using unified conversation system
    Returns (text_response, audio_url)
    """
    
    try:
        # Handle voice input
        if media_url and media_content_type and "audio" in media_content_type:
            print(f"🎤 Processing voice message for user {user_id}")
            transcribed_text = transcribe_voice_message(media_url, media_content_type)
            if not transcribed_text:
                return "I couldn't understand the audio. Could you try again?", None
            user_message = transcribed_text
            response_mode = "speech"  # User sent voice, respond with voice
        else:
            response_mode = "text"
        
        # Get memory and current conversation state
        memory = memory_manager.get_user_memory(user_id)
        current_state = memory.get_conversation_state()
        
        # Add conversation history
        current_state['conversation_history'] = current_state.get('conversation_history', [])
        
        print(f"🧠 Current state summary: {memory.get_known_info_summary()}")
        
        # Initialize variables
        detected_language = "en"
        text_response = ""
        
        # Decide which agent to use
        if should_use_unified_agent(user_message, user_id):
            print(f"🤖 Using unified conversation agent")
            
            # Process with unified agent
            response = process_conversation_turn(
                user_message=user_message,
                current_state=current_state,
                user_mode=response_mode
            )
            
            # Store detected language
            detected_language = response.language.split('-')[0] if response.language else "en"
            
            # Update conversation state
            memory.update_conversation_state(response.state_update)
            
            # Add to conversation history
            conversation_exchange = [
                {"role": "user", "content": user_message, "timestamp": datetime.now().isoformat()},
                {"role": "assistant", "content": response.utterance, "timestamp": datetime.now().isoformat()}
            ]
            memory.update_conversation_state({"conversation_history": conversation_exchange})
            
            # Handle search action
            if response.action == "SEARCH" and response.search_payload:
                print(f"🔍 Executing flight search")
                try:
                    # Execute flight search
                    search_result = search_flights(response.search_payload)
                    
                    if search_result and "error" not in search_result:
                        # Store search results
                        memory.update_conversation_state({
                            "flight_results": search_result,
                            "search_stale": False
                        })
                        
                        # Format results for user
                        results_text = format_search_results(search_result)
                        final_response = f"{response.utterance}\n\n{results_text}"
                    else:
                        final_response = f"{response.utterance}\n\nI encountered an issue searching for flights. Please try again."
                        
                except Exception as e:
                    print(f"❌ Flight search error: {e}")
                    final_response = f"{response.utterance}\n\nI encountered an issue searching for flights. Please try again."
            else:
                final_response = response.utterance
            
            text_response = final_response
            
        else:
            print(f"🤖 Using general conversation agent")
            # Use general conversation agent for non-flight topics
            text_response = handle_general_conversation(user_message)
            response_mode = "text"  # General conversations default to text
        
        # Generate voice response if needed
        audio_url = None
        if response_mode == "speech":
            print(f"🔊 Generating voice response")
            try:
                # Generate voice file locally
                voice_file_path = generate_voice_response(text_response, detected_language, user_id)
                if voice_file_path:
                    # Upload to S3 and get public URL
                    audio_url = upload_voice_file_to_accessible_url(voice_file_path, user_id)
                    if audio_url:
                        # Clean up local temp file after upload
                        try:
                            import os
                            os.unlink(voice_file_path)
                            print(f"🧹 Cleaned up temporary file: {voice_file_path}")
                        except Exception as cleanup_error:
                            print(f"⚠️ Could not clean up temp file: {cleanup_error}")
                    else:
                        print("⚠️ Failed to upload voice file to S3")
                else:
                    print("⚠️ Failed to generate voice file")
            except Exception as voice_error:
                print(f"❌ Voice generation error: {voice_error}")
                audio_url = None
        
        # Store the exchange
        memory.add_message(user_message, text_response, "unified")
        
        return text_response, audio_url
        
    except Exception as e:
        print(f"❌ Error in unified message processing: {e}")
        error_msg = "I apologize, but I encountered an error. Could you please try again?"
        return error_msg, None


def format_search_results(search_result: Dict[str, Any]) -> str:
    """Format flight search results for user display"""
    try:
        if not search_result or "CatalogProductOfferingsResponse" not in search_result:
            return "No flights found for your search criteria."
        
        response_data = search_result["CatalogProductOfferingsResponse"]
        offerings = response_data.get("CatalogProductOfferings", [])
        
        if not offerings:
            return "No flights found for your search criteria."
        
        # Get the best (first) offering
        best_offering = offerings[0]
        journeys = best_offering.get("Product", {}).get("Journey", [])
        
        if not journeys:
            return "No flight details available."
        
        # Format journey information
        result_parts = ["✈️ Here are the best flights I found:\n"]
        
        for i, journey in enumerate(journeys[:2]):  # Show max 2 journeys
            journey_type = "Outbound" if i == 0 else "Return"
            segments = journey.get("Segment", [])
            
            if segments:
                first_segment = segments[0]
                last_segment = segments[-1]
                
                # Flight details
                dep_info = first_segment.get("DepartureAirport", {})
                arr_info = last_segment.get("ArrivalAirport", {})
                
                dep_time = first_segment.get("DepartureDateTime", "")[:16]  # YYYY-MM-DDTHH:MM
                arr_time = last_segment.get("ArrivalDateTime", "")[:16]
                
                result_parts.append(f"{journey_type}: {dep_info.get('value', 'N/A')} → {arr_info.get('value', 'N/A')}")
                result_parts.append(f"Departure: {dep_time.replace('T', ' ')}")
                result_parts.append(f"Arrival: {arr_time.replace('T', ' ')}")
                
                if len(segments) > 1:
                    result_parts.append(f"Stops: {len(segments) - 1}")
                else:
                    result_parts.append("Direct flight")
                
                result_parts.append("")  # Empty line
        
        # Add pricing if available
        pricing = best_offering.get("Pricing", {})
        if pricing:
            total_price = pricing.get("TotalPrice", {})
            if total_price:
                amount = total_price.get("value", "N/A")
                currency = total_price.get("currencyCode", "USD")
                result_parts.append(f"💰 Price: {amount} {currency}")
        
        result_parts.append("\nWould you like to see more options or book this flight?")
        
        return "\n".join(result_parts)
        
    except Exception as e:
        print(f"❌ Error formatting search results: {e}")
        return "I found some flights but couldn't format the details properly. Please try again."


# Keep existing message handler as fallback
async def process_user_message_enhanced(
    user_message: str,
    user_id: str,
    media_url: Optional[str] = None,
    media_content_type: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """
    Enhanced message processing that routes to unified agent or legacy system
    """
    try:
        # Always try unified agent first for better experience
        return await process_unified_message(
            user_message=user_message,
            user_id=user_id,
            media_url=media_url,
            media_content_type=media_content_type
        )
        
    except Exception as e:
        print(f"❌ Unified processing failed, falling back to legacy: {e}")
        
        # Fallback to original message handler
        from .message_handler import process_user_message
        return await process_user_message(
            user_message=user_message,
            user_id=user_id,
            media_url=media_url,
            media_content_type=media_content_type
        ) 