"""
Enhanced flight booking agent with bulk date range searching
"""

import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dotenv import load_dotenv

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Local imports
from ..models.schemas import FlightBookingState
from ..api.travelport import get_api_headers, CATALOG_URL
from ..payloads.flight_search import build_flight_search_payload

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
# Initialize LLM
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


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
        # Generate dates to search
        dates_to_search = generate_date_range(
            state["date_range_start"], 
            state["date_range_end"],
            max_searches=15  # Limit to avoid API overload
        )
        
        print(f"🗓️ Searching {len(dates_to_search)} dates: {dates_to_search}")
        
        # Prepare search parameters
        from_city = str(state["from_city"])
        to_city = str(state["to_city"])
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
            state["response_text"] = f"😔 No flights found in the date range {state['date_range_start']} to {state['date_range_end']}."
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
        # Build API payload
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
    
    bulk_results = state.get("bulk_search_results", {})
    
    if not bulk_results:
        state["response_text"] = "No bulk search results to analyze."
        return state
    
    try:
        global_cheapest_flight = None
        global_lowest_price = float('inf')
        best_date = None
        
        # Analyze each date's results
        for search_date, api_response in bulk_results.items():
            print(f"🔍 Analyzing results for {search_date}")
            
            response_data = api_response.get("CatalogProductOfferingsResponse", {})
            catalog_offerings = response_data.get("CatalogProductOfferings", {})
            offerings = catalog_offerings.get("CatalogProductOffering", [])
            
            if not offerings:
                continue
            
            # Find cheapest flight for this date
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


# Fix this function in your app/agents/flight_booking_agent.py

def find_cheapest_flight_original(state: FlightBookingState) -> FlightBookingState:
    """Original single-date analysis function with enhanced details"""
    
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
        
        # Find the cheapest flight
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
                                
            except Exception as e:
                print(f"⚠️ Error analyzing offering: {e}")
                continue
        
        if cheapest_flight:
            state["cheapest_flight"] = cheapest_flight
            
            # Extract enhanced flight details - PASS THE FULL STATE
            flight_details = extract_flight_details(cheapest_flight, state)
            state["response_text"] = format_flight_response(flight_details)
            
            # FIX: Change 'details' to 'flight_details'
            print(f"✅ Found cheapest flight: {flight_details['currency']} {flight_details['price']}")
        else:
            state["response_text"] = "✈️ I found flights but couldn't determine pricing."
            
    except Exception as e:
        print(f"❌ Error in flight analysis: {e}")
        state["response_text"] = "😔 Error analyzing flight results."
    
    return state


# Also update analyze_bulk_search_results to use enhanced details
def analyze_bulk_search_results(state: FlightBookingState) -> FlightBookingState:
    """Analyze bulk search results to find the globally cheapest flight with enhanced details"""
    
    bulk_results = state.get("bulk_search_results", {})
    
    if not bulk_results:
        state["response_text"] = "No bulk search results to analyze."
        return state
    
    try:
        global_cheapest_flight = None
        global_lowest_price = float('inf')
        best_date = None
        best_api_response = None
        
        # Analyze each date's results
        for search_date, api_response in bulk_results.items():
            print(f"🔍 Analyzing results for {search_date}")
            
            response_data = api_response.get("CatalogProductOfferingsResponse", {})
            catalog_offerings = response_data.get("CatalogProductOfferings", {})
            offerings = catalog_offerings.get("CatalogProductOffering", [])
            
            if not offerings:
                continue
            
            # Find cheapest flight for this date
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
                                if price and float(price) < global_lowest_price:
                                    global_lowest_price = float(price)
                                    global_cheapest_flight = offering
                                    best_date = search_date
                                    best_api_response = api_response
                
                except Exception as e:
                    print(f"⚠️ Error analyzing offering for {search_date}: {e}")
                    continue
        
        if global_cheapest_flight and best_date and best_api_response:
            # Store the best results
            state["cheapest_flight"] = global_cheapest_flight
            state["best_departure_date"] = best_date
            state["raw_api_response"] = best_api_response  # IMPORTANT: Set this for enhanced details
            
            # Extract enhanced flight details
            flight_details = extract_flight_details(global_cheapest_flight, state, best_date)
            
            range_desc = state.get("range_description", f"{state.get('date_range_start')} to {state.get('date_range_end')}")
            
            # Use the enhanced format_flight_response but add range info
            base_response = format_flight_response(flight_details)
            
            response = f"""🎉 BEST DEAL FOUND! 🎉
🗓️ Optimal date in {range_desc}: {best_date}

{base_response.replace('✈️ FLIGHT FOUND! ✈️', '')}

📊 Searched {len(bulk_results)} dates to find you the best price!"""
            
            state["response_text"] = response
            
            print(f"✅ Found global cheapest flight: {flight_details['currency']} {flight_details['price']} on {best_date}")
        else:
            state["response_text"] = "✈️ I found flights but couldn't determine the best pricing across the date range."
    
    except Exception as e:
        print(f"❌ Error analyzing bulk results: {e}")
        state["response_text"] = "😔 Error analyzing search results across the date range."
    
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
    """Enhanced flight details extraction with ReferenceList parsing and baggage information"""
    
    details = {
        "price": "N/A",
        "currency": "EUR",
        "departure_time": "N/A",
        "arrival_time": "N/A",
        "airline": "N/A",
        "stops": "N/A",
        "baggage": "Standard",
        "duration": "N/A",
        "flight_number": "N/A"
    }
    
    try:
        # Step 1: Extract price information, flight references, and terms & conditions reference
        cheapest_price = float('inf')
        best_flight_refs = []
        best_currency = "EUR"
        best_terms_and_conditions_ref = None
        
        product_brand_options = flight_offering.get("ProductBrandOptions", [])
        
        for option in product_brand_options:
            # Get flight references for this option
            flight_refs = option.get("flightRefs", [])
            
            product_brand_offerings = option.get("ProductBrandOffering", [])
            for brand_offering in product_brand_offerings:
                best_price = brand_offering.get("BestCombinablePrice", {})
                if isinstance(best_price, dict) and best_price.get("TotalPrice"):
                    price = float(best_price["TotalPrice"])
                    if price < cheapest_price:
                        cheapest_price = price
                        best_flight_refs = flight_refs
                        details["price"] = str(best_price["TotalPrice"])
                        
                        # Extract currency
                        currency_info = best_price.get("CurrencyCode", {})
                        if isinstance(currency_info, dict):
                            best_currency = currency_info.get("value", "EUR")
                        details["currency"] = best_currency
                        
                        # Extract terms and conditions reference for baggage lookup
                        terms_and_conditions = brand_offering.get("TermsAndConditions", {})
                        if isinstance(terms_and_conditions, dict):
                            best_terms_and_conditions_ref = terms_and_conditions.get("termsAndConditionsRef")
        
        print(f"🔍 Best flight refs: {best_flight_refs}")
        print(f"🔍 Best terms & conditions ref: {best_terms_and_conditions_ref}")
        
        # Step 2: Look up flight details in ReferenceList
        if best_flight_refs and state.get("raw_api_response"):
            reference_list = state["raw_api_response"].get("CatalogProductOfferingsResponse", {}).get("ReferenceList", [])
            
            # Find the flight reference list
            flight_reference_list = None
            terms_and_conditions_list = None
            
            for ref_list in reference_list:
                if ref_list.get("@type") == "ReferenceListFlight":
                    flight_reference_list = ref_list.get("Flight", [])
                elif ref_list.get("@type") == "ReferenceListTermsAndConditions":
                    terms_and_conditions_list = ref_list.get("TermsAndConditions", [])
            
            if flight_reference_list:
                print(f"🔍 Found {len(flight_reference_list)} flights in reference list")
                
                # Extract details for our specific flights
                flight_segments = []
                airlines = set()
                total_duration_minutes = 0
                flight_numbers = []
                
                for flight_ref in best_flight_refs:
                    for flight in flight_reference_list:
                        if flight.get("id") == flight_ref:
                            flight_segments.append(flight)
                            
                            # Extract airline
                            carrier = flight.get("carrier", "")
                            if carrier:
                                airlines.add(get_airline_name(carrier))
                                
                            # Extract flight number
                            number = flight.get("number", "")
                            if number:
                                flight_numbers.append(f"{carrier}{number}")
                            
                            # Extract duration
                            duration = flight.get("duration", "")
                            if duration:
                                # Parse ISO 8601 duration (PT3H40M)
                                minutes = parse_iso_duration(duration)
                                total_duration_minutes += minutes
                            
                            break
                
                # Step 3: Format the extracted flight details
                if flight_segments:
                    # Departure details (first segment)
                    first_flight = flight_segments[0]
                    departure_info = first_flight.get("Departure", {})
                    departure_location = departure_info.get("location", "")
                    departure_date = departure_info.get("date", "")
                    departure_time = departure_info.get("time", "")
                    departure_terminal = departure_info.get("terminal", "")
                    
                    if departure_date and departure_time:
                        formatted_departure = format_flight_datetime(departure_date, departure_time, departure_location, departure_terminal)
                        details["departure_time"] = formatted_departure
                    
                    # Arrival details (last segment)
                    last_flight = flight_segments[-1]
                    arrival_info = last_flight.get("Arrival", {})
                    arrival_location = arrival_info.get("location", "")
                    arrival_date = arrival_info.get("date", "")
                    arrival_time = arrival_info.get("time", "")
                    
                    if arrival_date and arrival_time:
                        formatted_arrival = format_flight_datetime(arrival_date, arrival_time, arrival_location)
                        details["arrival_time"] = formatted_arrival
                    
                    # Airline information
                    if airlines:
                        if len(airlines) == 1:
                            details["airline"] = list(airlines)[0]
                        else:
                            details["airline"] = f"Multiple: {', '.join(airlines)}"
                    
                    # Stops information
                    num_segments = len(flight_segments)
                    if num_segments == 1:
                        details["stops"] = "Direct flight"
                    else:
                        details["stops"] = f"{num_segments - 1} stop(s)"
                        
                        # Add layover information
                        if num_segments > 1:
                            layover_cities = []
                            for i in range(len(flight_segments) - 1):
                                layover_city = flight_segments[i].get("Arrival", {}).get("location", "")
                                if layover_city:
                                    layover_cities.append(layover_city)
                            
                            if layover_cities:
                                details["stops"] += f" via {', '.join(layover_cities)}"
                    
                    # Duration
                    if total_duration_minutes > 0:
                        hours = total_duration_minutes // 60
                        minutes = total_duration_minutes % 60
                        if hours > 0:
                            details["duration"] = f"{hours}h {minutes}m"
                        else:
                            details["duration"] = f"{minutes}m"
                    
                    # Flight numbers
                    if flight_numbers:
                        details["flight_number"] = ", ".join(flight_numbers)
                
                print(f"✅ Enhanced flight details extracted: {details}")
            else:
                print(f"⚠️ No flight reference list found")
            
            # Step 4: Extract baggage allowance information
            if best_terms_and_conditions_ref and terms_and_conditions_list:
                print(f"🔍 Looking up baggage allowance for terms ref: {best_terms_and_conditions_ref}")
                
                baggage_info = extract_baggage_allowance(best_terms_and_conditions_ref, terms_and_conditions_list)
                if baggage_info:
                    details["baggage"] = baggage_info
                    print(f"✅ Baggage allowance extracted: {baggage_info}")
            else:
                print(f"⚠️ No terms & conditions reference or list found for baggage lookup")
        else:
            print(f"⚠️ No flight refs or API response available")
            
    except Exception as e:
        print(f"❌ Error in enhanced flight details extraction: {e}")
        import traceback
        traceback.print_exc()
    
    return details


def extract_baggage_allowance(terms_ref: str, terms_and_conditions_list: List[Dict]) -> str:
    """
    Extract baggage allowance information from terms and conditions
    
    Args:
        terms_ref: Reference ID to look up (e.g., "T0")
        terms_and_conditions_list: List of terms and conditions from ReferenceList
        
    Returns:
        str: Formatted baggage allowance information
    """
    
    try:
        # Find the matching terms and conditions
        matching_terms = None
        for terms in terms_and_conditions_list:
            if terms.get("id") == terms_ref:
                matching_terms = terms
                break
        
        if not matching_terms:
            print(f"⚠️ No matching terms found for ref: {terms_ref}")
            return "Check with airline"
        
        baggage_allowances = matching_terms.get("BaggageAllowance", [])
        if not baggage_allowances:
            print(f"⚠️ No baggage allowances found in terms: {terms_ref}")
            return "Check with airline"
        
        # Process baggage allowances
        baggage_details = []
        
        for allowance in baggage_allowances:
            baggage_type = allowance.get("baggageType", "")
            validating_airline = allowance.get("validatingAirlineCode", "")
            baggage_items = allowance.get("BaggageItem", [])
            
            print(f"🧳 Processing baggage type: {baggage_type}")
            
            for item in baggage_items:
                included_in_price = item.get("includedInOfferPrice", "No")
                sold_by_weight = item.get("soldByWeightInd", False)
                measurements = item.get("Measurement", [])
                text_info = item.get("Text", "")
                
                # Extract weight information
                weight_info = ""
                for measurement in measurements:
                    if measurement.get("measurementType") == "Weight":
                        weight_value = measurement.get("value", 0)
                        weight_unit = measurement.get("unit", "")
                        
                        if weight_value > 0:
                            weight_info = f"{weight_value} {weight_unit}"
                        else:
                            weight_info = "No free allowance"
                
                # Format baggage information
                if baggage_type == "FirstCheckedBag":
                    if included_in_price == "Yes" and weight_info and weight_info != "No free allowance":
                        baggage_details.append(f"1st bag: {weight_info} included")
                    elif included_in_price == "Yes":
                        baggage_details.append(f"1st bag: Included")
                    else:
                        if weight_info:
                            baggage_details.append(f"1st bag: {weight_info} (fee applies)")
                        else:
                            baggage_details.append(f"1st bag: Fee applies")
                
                elif baggage_type == "CarryOn":
                    if weight_info and weight_info != "No free allowance":
                        baggage_details.append(f"Carry-on: {weight_info}")
                    else:
                        baggage_details.append(f"Carry-on: Standard allowance")
                
                # Add text information if available and meaningful
                if text_info and text_info not in ["CHGS MAY APPLY IF BAGS EXCEED TTL WT ALLOWANCE"]:
                    # Clean up common airline text
                    cleaned_text = text_info.replace("CHGS MAY APPLY IF BAGS EXCEED TTL WT ALLOWANCE", "").strip()
                    if cleaned_text:
                        baggage_details.append(cleaned_text)
        
        # Format final baggage information
        if baggage_details:
            # Remove duplicates while preserving order
            unique_details = []
            for detail in baggage_details:
                if detail not in unique_details:
                    unique_details.append(detail)
            
            if len(unique_details) == 1:
                return unique_details[0]
            else:
                return "; ".join(unique_details)
        else:
            # Fallback based on airline
            if validating_airline:
                airline_name = get_airline_name(validating_airline)
                return f"Check {airline_name} policy"
            else:
                return "Check with airline"
                
    except Exception as e:
        print(f"❌ Error extracting baggage allowance: {e}")
        import traceback
        traceback.print_exc()
        return "Check with airline"


def get_airline_name(carrier_code: str) -> str:
    """Convert IATA carrier code to airline name"""
    airline_names = {
        'QR': 'Qatar Airways',
        'EK': 'Emirates',
        'EY': 'Etihad Airways',
        'PK': 'Pakistan International Airlines', 
        'TK': 'Turkish Airlines',
        'BA': 'British Airways',
        'LH': 'Lufthansa',
        'AF': 'Air France',
        'KL': 'KLM',
        'SQ': 'Singapore Airlines',
        'CX': 'Cathay Pacific',
        'NH': 'All Nippon Airways',
        'JL': 'Japan Airlines',
        'AC': 'Air Canada',
        'DL': 'Delta Air Lines',
        'UA': 'United Airlines',
        'AA': 'American Airlines'
    }
    return airline_names.get(carrier_code, carrier_code)


def parse_iso_duration(duration_str: str) -> int:
    """Parse ISO 8601 duration string (PT3H40M) to minutes"""
    try:
        import re
        # Pattern to match PT3H40M format
        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?'
        match = re.match(pattern, duration_str)
        
        if match:
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            return hours * 60 + minutes
        
        return 0
    except:
        return 0


def format_flight_datetime(date_str: str, time_str: str, location: str = "", terminal: str = "") -> str:
    """Format flight date and time for display"""
    try:
        from datetime import datetime
        
        # Parse the date (2025-08-22)
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        
        # Parse the time (09:55:00)
        time_obj = datetime.strptime(time_str, "%H:%M:%S")
        
        # Format for display
        formatted_date = date_obj.strftime("%b %d")  # Aug 22
        formatted_time = time_obj.strftime("%H:%M")  # 09:55
        
        result = f"{formatted_date} at {formatted_time}"
        
        if location:
            result += f" ({location})"
        
        if terminal:
            result += f" Terminal {terminal}"
        
        return result
        
    except Exception as e:
        print(f"⚠️ Error formatting datetime: {e}")
        return f"{date_str} {time_str}"


def format_flight_response(details: Dict) -> str:
    """Enhanced flight response formatting with all details"""
    
    response = f"""✈️ FLIGHT FOUND! ✈️

💰 Price: {details['currency']} {details['price']}
🛫 Departure: {details['departure_time']}
🛬 Arrival: {details['arrival_time']}
🏢 Airline: {details['airline']}
✈️ Flight: {details['flight_number']}
🔄 Stops: {details['stops']}
⏱️ Duration: {details['duration']}
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