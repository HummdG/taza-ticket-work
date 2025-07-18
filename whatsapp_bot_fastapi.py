"""
DEPRECATED - This file has been refactored into a modular structure.

The monolithic code has been split into the following modules:
- app/main.py - Main FastAPI application
- app/models/schemas.py - Data models and state definitions  
- app/api/travelport.py - API authentication and headers
- app/payloads/flight_search.py - API payload construction
- app/agents/flight_booking_agent.py - LangGraph workflow
- app/services/message_handler.py - Message processing logic

To run the new modular application:
python main.py

This file is kept for reference but should not be used in production.
"""

import os
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Dict, List, Optional, Union
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

# ⚠️ WARNING: This is the old monolithic version.
# Use the new modular structure in the 'app' directory instead.

# Load environment variables
load_dotenv()

# Configuration
CLIENT_ID = os.getenv("TRAVELPORT_APPLICATION_KEY")
CLIENT_SECRET = os.getenv("TRAVELPORT_APPLICATION_SECRET")
USERNAME = os.getenv("TRAVELPORT_USERNAME")
PASSWORD = os.getenv("TRAVELPORT_PASSWORD")
ACCESS_GROUP = os.getenv("TRAVELPORT_ACCESS_GROUP")

OAUTH_URL = "https://oauth.pp.travelport.com/oauth/oauth20/token"
CATALOG_URL = "https://api.pp.travelport.com/11/air/catalog/search/catalogproductofferings"

# Initialize FastAPI app
app = FastAPI(title="WhatsApp Flight Booking Bot", version="1.0.0")

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# State Definition
class FlightBookingState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    user_message: str
    from_city: Optional[str]
    to_city: Optional[str]
    departure_date: Optional[str]
    return_date: Optional[str]
    passengers: int
    passenger_age: int
    raw_api_response: Optional[Dict]
    cheapest_flight: Optional[Dict]
    response_text: str

# Pydantic models for API
class TestMessage(BaseModel):
    message: str

class WebhookResponse(BaseModel):
    status: str
    response: str

# Travelport API functions
def fetch_password_token():
    """Get OAuth token from Travelport API"""
    data = {
        "grant_type": "password",
        "username": USERNAME,
        "password": PASSWORD,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": "openid"
    }
    resp = requests.post(
        OAUTH_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=data
    )
    resp.raise_for_status()
    return resp.json()["access_token"]

def get_api_headers():
    """Build headers for Travelport API requests"""
    token = fetch_password_token()
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip, deflate",
        "Cache-Control": "no-cache",
        "Authorization": f"Bearer {token}",
        "XAUTH_TRAVELPORT_ACCESSGROUP": ACCESS_GROUP,
        "Accept-Version": "11",
        "Content-Version": "11",
    }

# Agent functions
def parse_travel_request(state: FlightBookingState) -> FlightBookingState:
    """Parse user message to extract travel details using LLM"""
    
    from datetime import datetime, timedelta
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    parsing_prompt = f"""
    Today's date is {today}. Extract flight booking details from this user message: "{state['user_message']}"
    
    Return ONLY a JSON object with these fields:
    {{
        "from_city": "3-letter airport code or null",
        "to_city": "3-letter airport code or null", 
        "departure_date": "YYYY-MM-DD format or null",
        "return_date": "YYYY-MM-DD format or null",
        "passengers": "number of passengers (default 1)",
        "passenger_age": "age of passenger (default 25)"
    }}
    
    Rules:
    - Today is {today}
    - Use standard 3-letter IATA airport codes (e.g., LHE for Lahore, ATH for Athens)
    - Parse relative dates: "tomorrow" = {tomorrow}, "next week" = 7 days from today
    - If return_date is not mentioned but it seems like a round trip, set it to 7 days after departure
    - If information is missing, use null
    - ALWAYS use dates in the future, never past dates
    
    Example: "I want to fly from Lahore to Athens tomorrow"
    Output: {{"from_city": "LHE", "to_city": "ATH", "departure_date": "{tomorrow}", "return_date": null, "passengers": 1, "passenger_age": 25}}
    """
    
    try:
        print(f"🤖 Calling LLM with message: {state['user_message']}")
        response = llm.invoke([HumanMessage(content=parsing_prompt)])
        content = response.content if isinstance(response.content, str) else str(response.content)
        print(f"🔤 LLM Raw Response: '{content}'")
        
        if not content or content.strip() == "":
            print("❌ LLM returned empty response")
            state["response_text"] = "I couldn't understand your flight request. Please provide details like: from city, to city, and travel dates."
            return state
        
        # Try to clean the content if it has extra text
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        parsed_data = json.loads(content)
        
        # Update state with parsed information
        state.update({
            "from_city": parsed_data.get("from_city"),
            "to_city": parsed_data.get("to_city"),
            "departure_date": parsed_data.get("departure_date"),
            "return_date": parsed_data.get("return_date"),
            "passengers": parsed_data.get("passengers", 1),
            "passenger_age": parsed_data.get("passenger_age", 25)
        })
        
        print(f"✅ Parsed travel details: {parsed_data}")
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON Parse Error: {e}")
        print(f"📝 Failed to parse content: '{content}'")
        state["response_text"] = "I couldn't understand your flight request. Please provide details like: from city, to city, and travel dates."
    except Exception as e:
        print(f"❌ Error parsing message: {e}")
        state["response_text"] = "I couldn't understand your flight request. Please provide details like: from city, to city, and travel dates."
    
    return state

def search_flights(state: FlightBookingState) -> FlightBookingState:
    """Search for flights using Travelport API"""
    
    # Check if we have required information
    if not state.get("from_city") or not state.get("to_city") or not state.get("departure_date"):
        state["response_text"] = "I need more information. Please specify: departure city, destination city, and travel date."
        return state
    
    try:
        # Build search criteria
        search_criteria = [{
            "@type": "SearchCriteriaFlight",
            "departureDate": state["departure_date"],
            "From": {"value": state["from_city"]},
            "To": {"value": state["to_city"]}
        }]
        
        # Add return flight if specified
        if state.get("return_date"):
            search_criteria.append({
                "@type": "SearchCriteriaFlight", 
                "departureDate": state["return_date"],
                "From": {"value": state["to_city"]},
                "To": {"value": state["from_city"]}
            })
        
        # Build API payload
        payload = {
            "@type": "CatalogProductOfferingsQueryRequest",
            "CatalogProductOfferingsRequest": {
                "@type": "CatalogProductOfferingsRequestAir",
                "maxNumberOfUpsellsToReturn": 1,
                "contentSourceList": ["GDS"],
                "PassengerCriteria": [{
                    "@type": "PassengerCriteria",
                    "number": state.get("passengers", 1),
                    "age": state.get("passenger_age", 25),
                    "passengerTypeCode": "ADT"
                }],
                "SearchCriteriaFlight": search_criteria,
                "SearchModifiersAir": {
                    "@type": "SearchModifiersAir",
                    "CarrierPreference": [{
                        "@type": "CarrierPreference",
                        "preferenceType": "Preferred",
                        "carriers": ["QR", "EY", "GF", "SV"]
                    }]
                },
                "CustomResponseModifiersAir": {
                    "@type": "CustomResponseModifiersAir",
                    "SearchRepresentation": "Journey"
                }
            }
        }
        
        # Make API call
        headers = get_api_headers()
        response = requests.post(CATALOG_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        api_result = response.json()
        state["raw_api_response"] = api_result
        
        # Debug: Print the API response structure
        print(f"🔍 API Response keys: {api_result.keys() if isinstance(api_result, dict) else 'Not a dict'}")
        if isinstance(api_result, dict) and 'CatalogProductOfferingsResponse' in api_result:
            catalog_response = api_result['CatalogProductOfferingsResponse']
            print(f"🔍 Catalog Response keys: {catalog_response.keys() if isinstance(catalog_response, dict) else 'Not a dict'}")
            if isinstance(catalog_response, dict) and 'CatalogProductOfferings' in catalog_response:
                offerings = catalog_response['CatalogProductOfferings']
                print(f"🔍 Offerings type: {type(offerings)}")
                if isinstance(offerings, list) and offerings:
                    print(f"🔍 First offering type: {type(offerings[0])}")
                    print(f"🔍 First offering sample: {str(offerings[0])[:200]}...")
        
        print(f"✅ Flight search completed. Found {len(api_result.get('CatalogProductOfferingsResponse', {}).get('CatalogProductOfferings', []))} options")
        
    except Exception as e:
        print(f"❌ Flight search error: {e}")
        state["response_text"] = f"Sorry, I couldn't search for flights at the moment. Error: {str(e)}"
    
    return state

def find_cheapest_flight(state: FlightBookingState) -> FlightBookingState:
    """Analyze API response to find cheapest flight and extract details"""
    
    if not state.get("raw_api_response"):
        state["response_text"] = "No flight data available to analyze."
        return state
    
    try:
        api_response = state["raw_api_response"]
        response_data = api_response.get("CatalogProductOfferingsResponse", {}) if api_response else {}
        catalog_offerings = response_data.get("CatalogProductOfferings", {}) if isinstance(response_data, dict) else {}
        offerings = catalog_offerings.get("CatalogProductOffering", []) if isinstance(catalog_offerings, dict) else []
        
        if not offerings:
            state["response_text"] = "No flights found for your search criteria."
            return state
        
        cheapest_flight = None
        lowest_price = float('inf')
        
        # Find the cheapest flight
        for offering in offerings:
            try:
                print(f"🔍 Analyzing offering: {type(offering)}")
                
                # Ensure offering is a dictionary
                if not isinstance(offering, dict):
                    print(f"⚠️ Offering is not a dict: {offering}")
                    continue
                
                # Extract price information from all options to find cheapest
                product_brand_options = offering.get("ProductBrandOptions", [])
                if not product_brand_options:
                    print(f"⚠️ No ProductBrandOptions found")
                    continue
                
                option_lowest_price = float('inf')
                best_option = None
                
                for option in product_brand_options:
                    product_brand_offerings = option.get("ProductBrandOffering", [])
                    
                    for brand_offering in product_brand_offerings:
                        best_price = brand_offering.get("BestCombinablePrice", {})
                        if isinstance(best_price, dict):
                            price = best_price.get("TotalPrice", 0)
                            if price and float(price) < option_lowest_price:
                                option_lowest_price = float(price)
                                best_option = brand_offering
                
                if not best_option:
                    print(f"⚠️ No valid price found in any ProductBrandOffering")
                    continue
                
                best_price = best_option.get("BestCombinablePrice", {})
                total_price = best_price.get("TotalPrice", 0)
                currency_info = best_price.get("CurrencyCode", {})
                currency = currency_info.get("value", "EUR") if isinstance(currency_info, dict) else "EUR"
                
                print(f"💰 Found best price for this offering: {total_price} {currency}")
                total_price = float(total_price) if total_price else 0
                
                if total_price > 0 and total_price < lowest_price:
                    lowest_price = total_price
                    cheapest_flight = offering
                    
            except (ValueError, TypeError, KeyError):
                continue
        
        if cheapest_flight:
            state["cheapest_flight"] = cheapest_flight
            
            # Extract flight details
            flight_details = extract_flight_details(cheapest_flight)
            state["response_text"] = format_flight_response(flight_details)
            
            print(f"✅ Found cheapest flight: ${lowest_price}")
        else:
            state["response_text"] = "I found flights but couldn't determine pricing. Please try again."
            
    except Exception as e:
        print(f"❌ Error analyzing flights: {e}")
        state["response_text"] = "Sorry, I had trouble analyzing the flight options."
    
    return state

def extract_flight_details(flight_offering: Dict) -> Dict:
    """Extract relevant details from a flight offering"""
    details = {
        "price": "N/A",
        "currency": "USD",
        "departure_time": "N/A",
        "arrival_time": "N/A", 
        "duration": "N/A",
        "airline": "N/A",
        "baggage": "Check with airline",
        "stops": "N/A"
    }
    
    try:
        # Extract price using correct Travelport API structure
        product_brand_options = flight_offering.get("ProductBrandOptions", [])
        if product_brand_options and isinstance(product_brand_options, list):
            first_option = product_brand_options[0]
            product_brand_offering = first_option.get("ProductBrandOffering", [])
            if product_brand_offering and isinstance(product_brand_offering, list):
                first_offering = product_brand_offering[0]
                best_price = first_offering.get("BestCombinablePrice", {})
                
                details["price"] = best_price.get("TotalPrice", "N/A")
                currency_info = best_price.get("CurrencyCode", {})
                details["currency"] = currency_info.get("value", "EUR") if isinstance(currency_info, dict) else "EUR"
        
        # Extract basic journey details from offering
        departure_city = flight_offering.get("Departure", "N/A")
        arrival_city = flight_offering.get("Arrival", "N/A")
        
        # Set basic available details  
        details["departure_time"] = f"From {departure_city}"
        details["arrival_time"] = f"To {arrival_city}"
        details["airline"] = "Check booking details"
        details["stops"] = "Check itinerary"
        
        # Extract baggage information
        baggage_info = flight_offering.get("Product", [{}])[0].get("BaggageAllowance", [])
        if baggage_info:
            details["baggage"] = f"{baggage_info[0].get('MaximumWeight', {}).get('value', 'Standard')} {baggage_info[0].get('MaximumWeight', {}).get('unit', 'kg')}"
            
    except Exception as e:
        print(f"Warning: Could not extract some flight details: {e}")
    
    return details

def format_flight_response(details: Dict) -> str:
    """Format flight details into a user-friendly response"""
    
    return f"""✈️ *Flight Found!*

💰 *Price:* {details['currency']} {details['price']}
🛫 *Departure:* {details['departure_time']}
🛬 *Arrival:* {details['arrival_time']}
✈️ *Airline:* {details['airline']}
🔄 *Stops:* {details['stops']}
🧳 *Baggage:* {details['baggage']}

Would you like me to search for more options or help you with booking?"""

# LangGraph workflow
def should_search_flights(state: FlightBookingState) -> str:
    """Decision function to determine next step"""
    if state.get("response_text") and "couldn't understand" in state["response_text"]:
        return "end"
    if not state.get("from_city") or not state.get("to_city") or not state.get("departure_date"):
        return "end"
    return "search"

def should_analyze_flights(state: FlightBookingState) -> str:
    """Decision function after flight search"""
    if state.get("raw_api_response"):
        return "analyze"
    return "end"

# Create the LangGraph workflow
workflow = StateGraph(FlightBookingState)
workflow.add_node("parse", parse_travel_request)
workflow.add_node("search", search_flights)
workflow.add_node("analyze", find_cheapest_flight)

workflow.set_entry_point("parse")
workflow.add_conditional_edges("parse", should_search_flights, {"search": "search", "end": END})
workflow.add_conditional_edges("search", should_analyze_flights, {"analyze": "analyze", "end": END})
workflow.add_edge("analyze", END)

flight_booking_agent = workflow.compile()

def process_flight_request(user_message: str) -> str:
    """Process a flight booking request using the LangGraph agent"""
    
    try:
        # Initialize state
        initial_state = FlightBookingState(
            messages=[HumanMessage(content=user_message)],
            user_message=user_message,
            from_city=None,
            to_city=None,
            departure_date=None,
            return_date=None,
            passengers=1,
            passenger_age=25,
            raw_api_response=None,
            cheapest_flight=None,
            response_text=""
        )
        
        # Run the agent
        final_state = flight_booking_agent.invoke(initial_state)
        
        # Return the response
        return final_state.get("response_text", "I'm sorry, I couldn't process your request.")
        
    except Exception as e:
        print(f"❌ Error processing request: {e}")
        return "I'm having trouble processing your request right now. Please try again later."

# FastAPI Routes
@app.get("/")
async def root():
    return {"message": "WhatsApp Flight Booking Bot API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Flight Booking Bot"}

@app.get("/webhook")
async def webhook_verify(request: Request):
    """Handle WhatsApp webhook verification"""
    verify_token = request.query_params.get("hub.verify_token")
    if verify_token == os.getenv("WEBHOOK_VERIFY_TOKEN", "your_verify_token"):
        return PlainTextResponse(request.query_params.get("hub.challenge", ""))
    raise HTTPException(status_code=403, detail="Verification failed")

@app.post("/webhook")
async def webhook_handler(request: Request):
    """Handle incoming WhatsApp messages"""
    try:
        # Handle form data (Twilio format)
        form_data = await request.form()
        message_text = form_data.get("Body", "")
        sender = form_data.get("From", "")
        
        if not message_text:
            # Try JSON format (Meta WhatsApp Business API)
            json_data = await request.json()
            if json_data and "messages" in json_data.get("entry", [{}])[0].get("changes", [{}])[0].get("value", {}):
                messages = json_data["entry"][0]["changes"][0]["value"]["messages"]
                if messages:
                    message_text = messages[0].get("text", {}).get("body", "")
                    sender = messages[0].get("from", "")
        
        if message_text and sender:
            # Process the flight request
            bot_response = process_flight_request(str(message_text))
            
            # Return TwiML response for Twilio
            twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{bot_response}</Message>
</Response>"""
            
            return PlainTextResponse(twiml_response, media_type="application/xml")
        
        return {"status": "ok"}
        
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/test")
async def test_endpoint(message: TestMessage):
    """Test endpoint for manual testing"""
    response = process_flight_request(message.message)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 