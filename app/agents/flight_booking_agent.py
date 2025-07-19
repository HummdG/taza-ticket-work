"""
LangGraph flight booking agent and workflow management
"""

import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Local imports
from ..models.schemas import FlightBookingState
from ..api.travelport import get_api_headers, CATALOG_URL
from ..payloads.flight_search import build_flight_search_payload


# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


def parse_travel_request(state: FlightBookingState) -> FlightBookingState:
    """Parse user message to extract travel details using LLM with conversation context"""
    
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Include conversation context if available
    context_section = ""
    if state.get("conversation_context"):
        context_section = f"\nPrevious conversation context:\n{state['conversation_context']}\n"
    
    parsing_prompt = f"""
    Today's date is {today}. Extract flight booking details from this user message: "{state['user_message']}"
    {context_section}
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
    - Use conversation context to fill in missing details (e.g., if user previously mentioned a city)
    
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
            state["response_text"] = "😅 I couldn't understand your flight request. Please provide details like: from city, to city, and travel dates."
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
        state["response_text"] = "😅 I couldn't understand your flight request. Please provide details like: from city, to city, and travel dates."
    except Exception as e:
        print(f"❌ Error parsing message: {e}")
        state["response_text"] = "😅 I couldn't understand your flight request. Please provide details like: from city, to city, and travel dates."
    
    return state


def search_flights(state: FlightBookingState) -> FlightBookingState:
    """Search for flights using Travelport API"""
    
    # Check if we have required information
    if not state.get("from_city") or not state.get("to_city") or not state.get("departure_date"):
        state["response_text"] = "I need more information. Please specify: departure city, destination city, and travel date."
        return state
    
    try:
        # Build API payload using the payload module
        # Type assertion since we've already checked these values are not None
        payload = build_flight_search_payload(
            from_city=str(state["from_city"]),
            to_city=str(state["to_city"]),
            departure_date=str(state["departure_date"]),
            return_date=state.get("return_date"),
            passengers=state.get("passengers", 1),
            passenger_age=state.get("passenger_age", 25)
        )
        
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
        state["response_text"] = f"😔 Sorry, I couldn't search for flights at the moment. Error: {str(e)}"
    
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
            
            # Extract flight details with search dates from state
            flight_details = extract_flight_details(cheapest_flight, state)
            state["response_text"] = format_flight_response(flight_details)
            
            print(f"✅ Found cheapest flight: ${lowest_price}")
        else:
            state["response_text"] = "✈️ I found flights but couldn't determine pricing. Please try again."
            
    except Exception as e:
        print(f"❌ Error analyzing flights: {e}")
        state["response_text"] = "😔 Sorry, I had trouble analyzing the flight options."
    
    return state


def extract_flight_details(flight_offering: Dict, state: Optional[FlightBookingState] = None) -> Dict:
    """Extract relevant details from a flight offering"""
    details = {
        "price": "N/A",
        "currency": "USD",
        "departure_time": "N/A",
        "arrival_time": "N/A", 
        "departure_date": "N/A",
        "return_date": "N/A",
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
        
        # Add search dates from state if available
        if state:
            departure_date = state.get("departure_date", "N/A")
            return_date_value = state.get("return_date")  # Can be None
            
            details["departure_date"] = departure_date
            
            # Handle return_date to ensure it's always a string
            if return_date_value:
                details["return_date"] = str(return_date_value)
            else:
                details["return_date"] = "One-way"
            
            # Format departure and arrival with dates
            if departure_date != "N/A":
                details["departure_time"] = f"From {departure_city} on {departure_date}"
                details["arrival_time"] = f"To {arrival_city}"
                if return_date_value:
                    details["arrival_time"] += f" (Return: {return_date_value})"
            else:
                details["departure_time"] = f"From {departure_city}"
                details["arrival_time"] = f"To {arrival_city}"
        else:
            # Fallback to basic format without dates
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
    
    # Format the response with dates
    response = f"""✈️ FLIGHT FOUND! ✈️

💰 Price: {details['currency']} {details['price']}
🛫 Departure: {details['departure_time']}
🛬 Arrival: {details['arrival_time']}
🏢 Airline: {details['airline']}
🔄 Stops: {details['stops']}
🧳 Baggage: {details['baggage']}

❓ Would you like me to search for more options or help you with booking?"""
    
    return response


# LangGraph workflow decision functions
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
def create_flight_booking_agent():
    """Create and compile the LangGraph flight booking agent"""
    
    workflow = StateGraph(FlightBookingState)
    workflow.add_node("parse", parse_travel_request)
    workflow.add_node("search", search_flights)
    workflow.add_node("analyze", find_cheapest_flight)

    workflow.set_entry_point("parse")
    workflow.add_conditional_edges("parse", should_search_flights, {"search": "search", "end": END})
    workflow.add_conditional_edges("search", should_analyze_flights, {"analyze": "analyze", "end": END})
    workflow.add_edge("analyze", END)

    return workflow.compile()


# Create the compiled agent
flight_booking_agent = create_flight_booking_agent() 