"""
Conversation router service to classify user message intents
"""

import re
from typing import Dict, List, Tuple
from .flight_info_collector import flight_collector


def classify_message_intent(user_message: str) -> str:
    """
    Classify the intent of a user message - flight_booking or general_chat
    
    Args:
        user_message: The user's input message
        
    Returns:
        str: Intent classification - "flight_booking" or "general_chat"
    """
    
    # Convert to lowercase for easier matching
    message_lower = user_message.lower().strip()
    
    # Strong flight-related patterns
    strong_flight_indicators = [
        r'\bflight\b', r'\bfly\b', r'\bbook.*flight\b', r'\bflight.*book\b',
        r'\bairline\b', r'\bairport\b', r'\bticket\b', r'\breservation\b',
        r'\bitinerary\b', r'\btravel.*plan\b', r'\btrip.*plan\b'
    ]
    
    # Flight action phrases
    flight_action_phrases = [
        'fly to', 'fly from', 'flight to', 'flight from', 'book flight',
        'book a flight', 'search flight', 'find flight', 'need flight',
        'want to fly', 'going to', 'traveling to', 'trip to'
    ]
    
    # Check for strong flight indicators
    strong_flight_count = sum(1 for pattern in strong_flight_indicators if re.search(pattern, message_lower))
    
    # Check for flight action phrases
    has_flight_action = any(phrase in message_lower for phrase in flight_action_phrases)
    
    # Check for city/airport patterns with travel context
    city_pattern = r'\b([A-Z]{3}|new york|london|paris|tokyo|dubai|bangkok|singapore|sydney|rome|madrid|barcelona|berlin|amsterdam|zurich|vienna|prague|moscow|istanbul|cairo|mumbai|delhi|bangalore|lahore|karachi|islamabad)\b'
    has_cities = bool(re.search(city_pattern, user_message, re.IGNORECASE))
    
    # Classification for flight booking
    if strong_flight_count >= 1 or has_flight_action or (has_cities and any(word in message_lower for word in ['to', 'from'])):
        print(f"✅ Classified as flight_booking: {user_message}")
        return "flight_booking"
    
    # Everything else is general chat
    print(f"✅ Classified as general_chat: {user_message}")
    return "general_chat"


def should_handle_as_flight_booking(user_message: str) -> bool:
    """
    Determine if a message should be handled by the flight booking agent
    
    Args:
        user_message: The user's input message
        
    Returns:
        bool: True if flight booking, False for general chat
    """
    intent = classify_message_intent(user_message)
    return intent == "flight_booking"


def analyze_flight_request_completeness(user_message: str, conversation_context: str = "") -> Tuple[bool, bool, Dict]:
    """
    Analyze if a flight request is complete or needs more information
    
    Args:
        user_message: The user's message
        conversation_context: Previous conversation context
        
    Returns:
        Tuple of (has_flight_intent, is_complete, extracted_info)
    """
    
    # Extract flight information using the flight collector
    extracted_info = flight_collector.extract_flight_info(user_message, conversation_context)
    
    has_flight_intent = extracted_info.get("flight_intent", False)
    
    if not has_flight_intent:
        return False, False, {}
    
    # Check if all required information is present
    is_complete = flight_collector.is_flight_info_complete(extracted_info)
    
    return has_flight_intent, is_complete, extracted_info


def should_collect_flight_info(user_message: str, conversation_context: str = "") -> bool:
    """
    Determine if we should start/continue collecting flight information
    
    Args:
        user_message: The user's message
        conversation_context: Previous conversation context
        
    Returns:
        bool: True if we should collect flight info, False otherwise
    """
    
    # Quick safety check for obvious non-flight messages
    message_lower = user_message.lower().strip()
    
    # Common greetings should never trigger flight collection
    if message_lower in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]:
        return False
    
    # Other obvious non-flight patterns
    if message_lower in ["how are you", "how are you?", "what's up", "what's up?", "how's it going", "how's it going?"]:
        return False
    
    has_intent, is_complete, _ = analyze_flight_request_completeness(user_message, conversation_context)
    
    # Collect info if there's flight intent but request is incomplete
    return has_intent and not is_complete 