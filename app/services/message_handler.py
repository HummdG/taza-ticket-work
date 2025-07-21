"""
Message handling service for processing all types of user requests with memory
"""

from langchain_core.messages import HumanMessage
from ..models.schemas import FlightBookingState
from ..agents.flight_booking_agent import flight_booking_agent
from ..agents.general_conversation_agent import handle_general_conversation
from ..services.conversation_router import should_handle_as_flight_booking, should_collect_flight_info, analyze_flight_request_completeness
from ..services.memory_service import memory_manager
from ..services.flight_info_collector import flight_collector


def process_user_message(user_message: str, user_id: str = "unknown") -> str:
    """
    Process any user message by routing to appropriate handler with memory
    
    Args:
        user_message: The user's message
        user_id: Unique identifier for the user
        
    Returns:
        str: Appropriate response based on message type
    """
    
    try:
        # Get conversation context from memory
        conversation_context = memory_manager.get_conversation_context(user_id)
        
        # Check if user is currently collecting flight information
        is_collecting = memory_manager.is_collecting_flight_info(user_id)
        
        if is_collecting:
            print(f"🔄 Continuing flight info collection for user: {user_id}")
            response = handle_flight_info_collection(user_message, user_id, conversation_context)
            message_type = "flight_collection"
        
        # Check if this is a complete flight booking request
        elif should_handle_as_flight_booking(user_message):
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