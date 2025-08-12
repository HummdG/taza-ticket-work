"""
Data models, schemas, and state definitions for the flight booking bot
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage


# Pydantic models for FastAPI
class TestMessage(BaseModel):
    message: str
    user_id: Optional[str] = "test_user"  # Default for testing


class WebhookResponse(BaseModel):
    status: str
    response: str


# Enhanced Conversation State for unified slot-filling system
class ConversationState(TypedDict):
    """Enhanced conversation state with unified slot management"""
    # Core conversation
    user_id: str
    user_message: str
    conversation_history: List[Dict[str, str]]  # Recent message exchanges
    
    # Required slots for flight booking
    origin: Optional[str]  # IATA city/airport code
    destination: Optional[str]  # IATA city/airport code
    dates: Optional[Dict[str, str]]  # {"depart": "YYYY-MM-DD", "return": "YYYY-MM-DD"} 
    passengers: Optional[int]
    trip_type: Optional[str]  # "one_way" | "return" | "multi_city"
    
    # Language and mode detection
    language: Optional[str]  # BCP47 language code
    response_mode: Optional[str]  # "text" | "speech"
    
    # State management
    search_stale: bool  # True if any slot changed and new search needed
    missing_slots: List[str]  # List of missing required slots
    
    # Flight search results
    search_payload: Optional[Dict[str, Any]]  # Built payload ready for API
    flight_results: Optional[Dict[str, Any]]  # API response
    
    # System state
    last_updated: Optional[str]  # ISO timestamp


# Response schema for the conversation system
class ConversationResponse(BaseModel):
    """Machine-readable conversation response"""
    response_mode: str  # "text" | "speech"
    language: str  # BCP47 language code
    action: str  # "ASK_MISSING" | "SEARCH" | "CONFIRM_TRIP" | "CLARIFY" | "SMALL_TALK" | "OTHER"
    missing_slots: List[str]  # Required slots that are missing
    state_update: Dict[str, Any]  # Updates to apply to conversation state
    utterance: str  # Natural response in user's language and mode
    search_payload: Optional[Dict[str, Any]] = None  # Only when action=SEARCH
    notes: Optional[str] = None  # Brief rationale for logs


# State Definition for LangGraph (keeping backward compatibility)
class FlightBookingState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    user_message: str
    user_id: str
    from_city: Optional[str]
    to_city: Optional[str]
    departure_date: Optional[str]
    return_date: Optional[str]
    passengers: int
    passenger_age: int
    # Explicitly track trip type to persist across workflow nodes
    trip_type: Optional[str]  # "one-way" or "round-trip"
    raw_api_response: Optional[Dict[str, Any]]
    cheapest_flight: Optional[Dict[str, Any]]
    response_text: str
    conversation_context: Optional[str]
    
    # Enhanced fields for bulk search functionality
    search_type: Optional[str]  # "specific" or "range"
    date_range_start: Optional[str]  # Start date for range searches
    date_range_end: Optional[str]  # End date for range searches
    range_description: Optional[str]  # Human-readable description of range
    bulk_search_results: Optional[Dict[str, Dict[str, Any]]]  # date -> api_response mapping
    search_dates: Optional[List[str]]  # List of dates that were searched
    best_departure_date: Optional[str]  # Best date found in range search


# Flight details response model
class FlightDetails(BaseModel):
    price: str = "N/A"
    currency: str = "USD"
    departure_time: str = "N/A"
    arrival_time: str = "N/A"
    duration: str = "N/A"
    airline: str = "N/A"
    baggage: str = "Check with airline"
    stops: str = "N/A"


class VoiceTestMessage(BaseModel):
    """Test message model for voice message testing"""
    message: str = ""  # Can be empty for voice-only messages
    user_id: Optional[str] = None
    media_url: Optional[str] = None
    media_content_type: Optional[str] = None