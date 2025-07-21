"""
Data models, schemas, and state definitions for the flight booking bot
"""

from typing import Dict, List, Optional, Union
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
    raw_api_response: Optional[Dict]
    cheapest_flight: Optional[Dict]
    response_text: str
    conversation_context: Optional[str]


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