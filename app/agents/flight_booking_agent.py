"""
Enhanced flight booking agent with bulk date range searching - FIXED VERSION
"""

import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Local imports
from ..models.schemas import FlightBookingState
from ..api.travelport import get_api_headers, CATALOG_URL
from ..payloads.flight_search import build_flight_search_payload
from dotenv import load_dotenv
load_dotenv()


# Initialize LLM

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0
# )

llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0)


def parse_travel_request(state: FlightBookingState) -> FlightBookingState:
    """Enhanced parsing to detect date ranges vs specific dates"""
    
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Include conversation context if available
    context_section = ""
    if state.get("conversation_context"):
        context_section = f"\nPrevious conversation context:\n{state['conversation_context']}\n"
    
    parsing_prompt = f"""
    Today's date is {today}. Extract flight booking details from this user message: "{state['user_message']}"
    {context_section}
    
    Determine if the user is asking for:
    1. A specific date search
    2. A date range search (e.g., "cheapest in September", "best price next week")
    
    Return ONLY a JSON object with these fields:
    {{
        "from_city": "3-letter airport code or null",
        "to_city": "3-letter airport code or null", 
        "departure_date": "YYYY-MM-DD format or null (for specific date)",
        "return_date": "YYYY-MM-DD format or null",
        "passengers": "number of passengers (default 1)",
        "passenger_age": "age of passenger (default 25)",
        "search_type": "specific or range",
        "date_range_start": "YYYY-MM-DD format or null (start of range)",
        "date_range_end": "YYYY-MM-DD format or null (end of range)",
        "range_description": "text description of the range or null"
    }}
    
    Rules:
    - Today is {today}
    - Use standard 3-letter IATA airport codes
    - For phrases like "cheapest in September", "best price next week", "cheapest in the next month":
      - Set search_type to "range"
      - Set date_range_start and date_range_end appropriately
      - Set departure_date to null
    - For specific dates like "tomorrow", "August 15th":
      - Set search_type to "specific"
      - Set departure_date to the specific date
      - Set date_range_start and date_range_end to null
    - September 2025 = 2025-09-01 to 2025-09-30
    - Next week = 7 days starting from tomorrow
    - Next month = 30 days starting from today
    
    Examples:
    "Cheapest price from NYC to LAX in September" → search_type: "range", date_range_start: "2025-09-01", date_range_end: "2025-09-30"
    "Flight from NYC to LAX tomorrow" → search_type: "specific", departure_date: "{tomorrow}"
    """
    
    try:
        print(f"🤖 Enhanced parsing for: {state['user_message']}")
        response = llm.invoke([HumanMessage(content=parsing_prompt)])
        content = response.content if isinstance(response.content, str) else str(response.content)
        
        # Clean the response
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
            "passenger_age": parsed_data.get("passenger_age", 25),
            "search_type": parsed_data.get("search_type", "specific"),
            "date_range_start": parsed_data.get("date_range_start"),
            "date_range_end": parsed_data.get("date_range_end"),
            "range_description": parsed_data.get("range_description")
        })
        
        print(f"✅ Enhanced parsing result: {parsed_data}")
        
    except Exception as e:
        print(f"❌ Enhanced parsing error: {e}")
        state["response_text"] = "😅 I couldn't understand your flight request. Please provide details like: from city, to city, and travel dates."
    
    return state


def generate_date_range(start_date: str, end_date: str, max_searches: int = 15) -> List[str]:
    """Generate a list of dates to search within the given range"""
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    total_days = (end - start).days + 1
    
    if total_days <= max_searches:
        # Search every day if range is small enough
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        return dates
    else:
        # Sample dates across the range if it's too large
        step = total_days // max_searches
        dates = []
        for i in range(0, total_days, step):
            current = start + timedelta(days=i)
            if current <= end:
                dates.append(current.strftime("%Y-%m-%d"))
        
        # Always include the last date
        if dates[-1] != end_date:
            dates.append(end_date)
        
        return dates[:max_searches]


def search_single_date(from_city: str, to_city: str, departure_date: str, 
                      return_date: Optional[str], passengers: int, passenger_age: int) -> Tuple[str, Optional[Dict]]:
    """Search flights for a single date"""
    
    try:
        # Build API payload
        payload = build_flight_search_payload(
            from_city=from_city,
            to_city=to_city,
            departure_date=departure_date,
            return_date=return_date,
            passengers=passengers,
            passenger_age=passenger_age
        )
        
        # Make API call
        headers = get_api_headers()
        response = requests.post(CATALOG_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        return departure_date, response.json()
        
    except Exception as e:
        print(f"❌ Error searching date {departure_date}: {e}")
        return departure_date, None


def search_flights_bulk(state: FlightBookingState) -> FlightBookingState:
    """Search flights across a date range and find the globally cheapest option"""
    
    try:
        # FIX: Check for None values before calling generate_date_range
        start_date = state.get("date_range_start")
        end_date = state.get("date_range_end")
        
        if not start_date or not end_date:
            state["response_text"] = "Missing date range information for bulk search."
            return state
        
        # Generate dates to search
        dates_to_search = generate_date_range(
            start_date, 
            end_date,
            max_searches=15  # Limit to avoid API overload
        )
        
        print(f"🗓️ Searching {len(dates_to_search)} dates: {dates_to_search}")
        
        # Prepare search parameters
        from_city = str(state["from_city"]) if state.get("from_city") else ""
        to_city = str(state["to_city"]) if state.get("to_city") else ""
        return_date = state.get("return_date")
        passengers = state.get("passengers", 1)
        passenger_age = state.get("passenger_age", 25)
        
        # Use ThreadPoolExecutor for concurrent searches (be careful with rate limits)
        search_results = {}
        
        # Sequential search to avoid overwhelming the API
        for date in dates_to_search:
            print(f"🔍 Searching {date}...")
            try:
                search_date, result = search_single_date(
                    from_city, to_city, date, return_date, passengers, passenger_age
                )
                if result:
                    search_results[search_date] = result
                
                # Add small delay to respect rate limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"⚠️ Failed to search {date}: {e}")
                continue
        
        if not search_results:
            state["response_text"] = f"😔 No flights found in the date range {start_date} to {end_date}."
            return state
        
        # Store all results for analysis
        state["bulk_search_results"] = search_results
        state["search_dates"] = dates_to_search
        
        print(f"✅ Bulk search completed. Found results for {len(search_results)} dates")
        
    except Exception as e:
        print(f"❌ Bulk search error: {e}")
        state["response_text"] = f"😔 Error during bulk search: {str(e)}"
    
    return state


def search_flights_original(state: FlightBookingState) -> FlightBookingState:
    """Original single-date search function"""
    
    try:
        # FIX: Add None checks for state values
        from_city = state.get("from_city")
        to_city = state.get("to_city") 
        departure_date = state.get("departure_date")
        
        if not from_city or not to_city or not departure_date:
            state["response_text"] = "Missing required information for flight search."
            return state
        
        # Build API payload
        payload = build_flight_search_payload(
            from_city=str(from_city),
            to_city=str(to_city),
            departure_date=str(departure_date),
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
        
        print(f"✅ Single date search completed")
        
    except Exception as e:
        print(f"❌ Single date search error: {e}")
        state["response_text"] = f"😔 Sorry, I couldn't search for flights. Error: {str(e)}"
    
    return state


def search_flights(state: FlightBookingState) -> FlightBookingState:
    """Enhanced flight search supporting both specific dates and date ranges"""
    
    # Check if we have required information
    if not state.get("from_city") or not state.get("to_city"):
        state["response_text"] = "I need more information. Please specify: departure city and destination city."
        return state
    
    search_type = state.get("search_type", "specific")
    
    if search_type == "specific":
        # Original single date search
        if not state.get("departure_date"):
            state["response_text"] = "I need a departure date for your flight search."
            return state
        
        print(f"🔍 Searching specific date: {state['departure_date']}")
        return search_flights_original(state)
    
    elif search_type == "range":
        # New bulk date range search
        if not state.get("date_range_start") or not state.get("date_range_end"):
            state["response_text"] = "I need a valid date range for your search."
            return state
        
        print(f"🔍 Searching date range: {state['date_range_start']} to {state['date_range_end']}")
        return search_flights_bulk(state)
    
    else:
        state["response_text"] = "Invalid search type specified."
        return state


def analyze_bulk_search_results(state: FlightBookingState) -> FlightBookingState:
    """Analyze bulk search results to find the globally cheapest flight"""
    
    # FIX: Add proper None check and use .get() method
    bulk_results = state.get("bulk_search_results")
    
    if not bulk_results or not isinstance(bulk_results, dict):
        state["response_text"] = "No bulk search results to analyze."
        return state
    
    try:
        global_cheapest_flight = None
        global_lowest_price = float('inf')
        best_date = None
        
        # Analyze each date's results
        for search_date, api_response in bulk_results.items():
            print(f"🔍 Analyzing results for {search_date}")
            
            # FIX: Add comprehensive None checks
            if not api_response or not isinstance(api_response, dict):
                continue
                
            response_data = api_response.get("CatalogProductOfferingsResponse", {})
            if not response_data or not isinstance(response_data, dict):
                continue
                
            catalog_offerings = response_data.get("CatalogProductOfferings", {})
            if not catalog_offerings or not isinstance(catalog_offerings, dict):
                continue
                
            offerings = catalog_offerings.get("CatalogProductOffering", [])
            
            if not offerings:
                continue
            
            # Find cheapest flight for this date
            for offering in offerings:
                try:
                    # FIX: Add None check for offering
                    if not offering or not isinstance(offering, dict):
                        continue
                        
                    product_brand_options = offering.get("ProductBrandOptions", [])
                    if not product_brand_options:
                        continue
                    
                    for option in product_brand_options:
                        # FIX: Add None check for option
                        if not option or not isinstance(option, dict):
                            continue
                            
                        product_brand_offerings = option.get("ProductBrandOffering", [])
                        if not product_brand_offerings:
                            continue
                        
                        for brand_offering in product_brand_offerings:
                            # FIX: Add None check for brand_offering
                            if not brand_offering or not isinstance(brand_offering, dict):
                                continue
                                
                            best_price = brand_offering.get("BestCombinablePrice", {})
                            if isinstance(best_price, dict):
                                price = best_price.get("TotalPrice", 0)
                                if price and float(price) < global_lowest_price:
                                    global_lowest_price = float(price)
                                    global_cheapest_flight = offering
                                    best_date = search_date
                
                except Exception as e:
                    print(f"⚠️ Error analyzing offering for {search_date}: {e}")
                    continue
        
        if global_cheapest_flight and best_date:
            # Store the best results
            state["cheapest_flight"] = global_cheapest_flight
            state["best_departure_date"] = best_date
            state["raw_api_response"] = bulk_results[best_date]  # Set this for compatibility
            
            # Extract flight details
            flight_details = extract_flight_details(global_cheapest_flight, state, best_date)
            
            range_desc = state.get("range_description", f"{state.get('date_range_start')} to {state.get('date_range_end')}")
            
            response = f"""✈️ CHEAPEST FLIGHT FOUND! ✈️
🗓️ Best date in {range_desc}: {best_date}

💰 Price: {flight_details['currency']} {flight_details['price']}
🛫 Departure: {flight_details['departure_time']}
🛬 Arrival: {flight_details['arrival_time']}
🏢 Airline: {flight_details['airline']}
🔄 Stops: {flight_details['stops']}
🧳 Baggage: {flight_details['baggage']}

📊 Searched {len(bulk_results)} dates to find you the best price!

❓ Would you like me to search for more options or help you with booking?"""
            
            state["response_text"] = response
            
            print(f"✅ Found global cheapest flight: ${global_lowest_price} on {best_date}")
        else:
            state["response_text"] = "✈️ I found flights but couldn't determine the best pricing across the date range."
    
    except Exception as e:
        print(f"❌ Error analyzing bulk results: {e}")
        state["response_text"] = "😔 Error analyzing search results across the date range."
    
    return state


def find_cheapest_flight_original(state: FlightBookingState) -> FlightBookingState:
    """Original single-date analysis function"""
    
    if not state.get("raw_api_response"):
        state["response_text"] = "No flight data available to analyze."
        return state
    
    try:
        api_response = state["raw_api_response"]
        response_data = api_response.get("CatalogProductOfferingsResponse", {})
        catalog_offerings = response_data.get("CatalogProductOfferings", {}) 
        offerings = catalog_offerings.get("CatalogProductOffering", [])
        
        if not offerings:
            state["response_text"] = "No flights found for your search criteria."
            return state
        
        cheapest_flight = None
        lowest_price = float('inf')
        
        # Find the cheapest flight (same logic as original)
        for offering in offerings:
            try:
                product_brand_options = offering.get("ProductBrandOptions", [])
                if not product_brand_options:
                    continue
                
                for option in product_brand_options:
                    product_brand_offerings = option.get("ProductBrandOffering", [])
                    
                    for brand_offering in product_brand_offerings:
                        best_price = brand_offering.get("BestCombinablePrice", {})
                        if isinstance(best_price, dict):
                            price = best_price.get("TotalPrice", 0)
                            if price and float(price) < lowest_price:
                                lowest_price = float(price)
                                cheapest_flight = offering
                                
            except Exception:
                continue
        
        if cheapest_flight:
            state["cheapest_flight"] = cheapest_flight
            
            # Extract flight details
            flight_details = extract_flight_details(cheapest_flight, state)
            state["response_text"] = format_flight_response(flight_details)
            
            print(f"✅ Found cheapest flight: ${lowest_price}")
        else:
            state["response_text"] = "✈️ I found flights but couldn't determine pricing."
            
    except Exception as e:
        print(f"❌ Error in original analysis: {e}")
        state["response_text"] = "😔 Error analyzing flight results."
    
    return state


def find_cheapest_flight(state: FlightBookingState) -> FlightBookingState:
    """Enhanced analysis supporting both single and bulk search results"""
    
    search_type = state.get("search_type", "specific")
    
    if search_type == "specific":
        # Use original analysis for single date searches
        return find_cheapest_flight_original(state)
    
    elif search_type == "range":
        # New bulk analysis for date range searches
        return analyze_bulk_search_results(state)
    
    else:
        state["response_text"] = "Invalid search type for analysis."
        return state


def extract_flight_details(flight_offering: Dict, state: FlightBookingState, override_date: Optional[str] = None) -> Dict:
    """Enhanced flight details extraction"""
    
    details = {
        "price": "N/A",
        "currency": "EUR",
        "departure_time": "N/A",
        "arrival_time": "N/A",
        "airline": "Various",
        "stops": "N/A",
        "baggage": "Standard"
    }
    
    try:
        # Extract price information
        product_brand_options = flight_offering.get("ProductBrandOptions", [])
        if product_brand_options:
            for option in product_brand_options:
                product_brand_offerings = option.get("ProductBrandOffering", [])
                for brand_offering in product_brand_offerings:
                    best_price = brand_offering.get("BestCombinablePrice", {})
                    if isinstance(best_price, dict) and best_price.get("TotalPrice"):
                        details["price"] = str(best_price["TotalPrice"])
                        currency_info = best_price.get("CurrencyCode", {})
                        details["currency"] = currency_info.get("value", "EUR") if isinstance(currency_info, dict) else "EUR"
                        break
        
        # Use override date or state date for departure time
        departure_date = override_date or state.get("departure_date") or state.get("best_departure_date")
        
        if departure_date:
            departure_city = state.get("from_city", "")
            arrival_city = state.get("to_city", "")
            return_date = state.get("return_date")
            
            if return_date:
                details["departure_time"] = f"{departure_date} from {departure_city}"
                details["arrival_time"] = f"To {arrival_city} (Return: {return_date})"
            else:
                details["departure_time"] = f"{departure_date} from {departure_city}"
                details["arrival_time"] = f"To {arrival_city}"
        
    except Exception as e:
        print(f"Warning: Could not extract enhanced flight details: {e}")
    
    return details


def format_flight_response(details: Dict) -> str:
    """Enhanced flight response formatting"""
    
    response = f"""✈️ FLIGHT FOUND! ✈️

💰 Price: {details['currency']} {details['price']}
🛫 Departure: {details['departure_time']}
🛬 Arrival: {details['arrival_time']}
🏢 Airline: {details['airline']}
🔄 Stops: {details['stops']}
🧳 Baggage: {details['baggage']}

❓ Would you like me to search for more options or help you with booking?"""
    
    return response


# Updated workflow decision functions
def should_search_flights(state: FlightBookingState) -> str:
    """Enhanced decision function for routing"""
    if state.get("response_text") and "couldn't understand" in state["response_text"]:
        return "end"
    if not state.get("from_city") or not state.get("to_city"):
        return "end"
    
    search_type = state.get("search_type", "specific")
    if search_type == "specific" and not state.get("departure_date"):
        return "end"
    if search_type == "range" and (not state.get("date_range_start") or not state.get("date_range_end")):
        return "end"
    
    return "search"


def should_analyze_flights(state: FlightBookingState) -> str:
    """Enhanced decision function for analysis routing"""
    search_type = state.get("search_type", "specific")
    
    if search_type == "specific":
        return "analyze" if state.get("raw_api_response") else "end"
    elif search_type == "range":
        return "analyze" if state.get("bulk_search_results") else "end"
    else:
        return "end"


# Create the enhanced LangGraph workflow
def create_flight_booking_agent():
    """Create and compile the enhanced LangGraph flight booking agent"""
    
    workflow = StateGraph(FlightBookingState)
    workflow.add_node("parse", parse_travel_request)
    workflow.add_node("search", search_flights)
    workflow.add_node("analyze", find_cheapest_flight)

    workflow.set_entry_point("parse")
    workflow.add_conditional_edges("parse", should_search_flights, {"search": "search", "end": END})
    workflow.add_conditional_edges("search", should_analyze_flights, {"analyze": "analyze", "end": END})
    workflow.add_edge("analyze", END)

    return workflow.compile()


# Create the enhanced compiled agent with original name
flight_booking_agent = create_flight_booking_agent()