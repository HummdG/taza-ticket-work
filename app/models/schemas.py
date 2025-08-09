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


# State Definition for LangGraph
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