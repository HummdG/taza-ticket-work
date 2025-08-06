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
    """Enhanced parsing with better round-trip detection and duration calculation"""
    
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Include conversation context if available
    context_section = ""
    if state.get("conversation_context"):
        context_section = f"\nPrevious conversation context:\n{state['conversation_context']}\n"
    
    parsing_prompt = f"""
    Today's date is {today}. Extract flight booking details from this user message: "{state['user_message']}"
    {context_section}
    
    Pay special attention to round-trip requests and duration-based bookings.
    
    Return ONLY a JSON object with these fields:
    {{
        "from_city": "3-letter airport code or null",
        "to_city": "3-letter airport code or null", 
        "departure_date": "YYYY-MM-DD format or null (for specific date)",
        "return_date": "YYYY-MM-DD format or null (for return flights)",
        "passengers": "number of passengers (default 1)",
        "passenger_age": "age of passenger (default 25)",
        "search_type": "specific or range",
        "trip_type": "one-way or round-trip",
        "duration_days": "number of days for trip or null",
        "date_range_start": "YYYY-MM-DD format or null (start of range)",
        "date_range_end": "YYYY-MM-DD format or null (end of range)",
        "range_description": "text description of the range or null"
    }}
    
    Rules for round-trip detection and duration calculation:
    - "Round trip", "return trip", "coming back", "for X days" = trip_type: "round-trip"
    - "leaving tomorrow for 5 days" = departure: tomorrow, return: tomorrow + 5 days, trip_type: "round-trip"
    - "going for a week" = departure + 7 days for return
    - Always calculate return_date when trip_type is "round-trip" and duration_days is provided
    - Today is {today}, tomorrow is {tomorrow}
    - Use standard 3-letter IATA codes (KHI=Karachi, DXB=Dubai)
    
    Duration calculation examples:
    - "leaving tomorrow for 5 days" → departure: {tomorrow}, return: {(datetime.now() + timedelta(days=6)).strftime("%Y-%m-%d")}
    - "round trip for a week" → return date = departure + 7 days
    - "going for 3 days" with departure Aug 7 → return Aug 10
    
    Examples:
    "Round trip from Karachi to Dubai leaving tomorrow for 5 days" → 
    {{
        "from_city": "KHI", 
        "to_city": "DXB", 
        "departure_date": "{tomorrow}",
        "return_date": "{(datetime.now() + timedelta(days=6)).strftime("%Y-%m-%d")}",
        "trip_type": "round-trip",
        "duration_days": 5,
        "search_type": "specific"
    }}
    """
    
    try:
        print(f"🤖 Enhanced round-trip parsing for: {state['user_message']}")
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
        
        # Enhanced return date calculation for round-trips
        if parsed_data.get("trip_type") == "round-trip":
            if parsed_data.get("duration_days") and parsed_data.get("departure_date") and not parsed_data.get("return_date"):
                try:
                    dep_date = datetime.strptime(parsed_data["departure_date"], "%Y-%m-%d")
                    return_date = dep_date + timedelta(days=int(parsed_data["duration_days"]))
                    parsed_data["return_date"] = return_date.strftime("%Y-%m-%d")
                    print(f"✅ Calculated return date: {parsed_data['return_date']} (departure + {parsed_data['duration_days']} days)")
                except Exception as e:
                    print(f"⚠️ Error calculating return date: {e}")
            
            # Ensure search_type is specific for round-trips with exact dates
            if parsed_data.get("departure_date") and parsed_data.get("return_date"):
                parsed_data["search_type"] = "specific"
                print(f"✅ Round-trip detected: {parsed_data['departure_date']} to {parsed_data['return_date']}")
        
        # Update state with parsed information
        state.update({
            "from_city": parsed_data.get("from_city"),
            "to_city": parsed_data.get("to_city"),
            "departure_date": parsed_data.get("departure_date"),
            "return_date": parsed_data.get("return_date"),
            "passengers": parsed_data.get("passengers", 1),
            "passenger_age": parsed_data.get("passenger_age", 25),
            "search_type": parsed_data.get("search_type", "specific"),
            "trip_type": parsed_data.get("trip_type", "one-way"),
            "duration_days": parsed_data.get("duration_days"),
            "date_range_start": parsed_data.get("date_range_start"),
            "date_range_end": parsed_data.get("date_range_end"),
            "range_description": parsed_data.get("range_description")
        })
        
        print(f"✅ Enhanced parsing result: {parsed_data}")
        
    except Exception as e:
        print(f"❌ Enhanced parsing error: {e}")
        state["response_text"] = "😅 I couldn't understand your flight request. Please provide details like: from city, to city, and travel dates."
    
    return state

# Add this helper function
def extract_date_range_manually(message: str) -> dict:
    """Manual fallback for date range extraction"""
    import re
    from datetime import datetime, timedelta
    
    message_lower = message.lower()
    current_year = datetime.now().year
    
    # Pattern for "between X to Y" or "from X to Y"
    date_pattern = r'(?:between|from)\s+(\d{1,2})(?:st|nd|rd|th)?\s+to\s+(\d{1,2})(?:st|nd|rd|th)?\s+(\w+)'
    match = re.search(date_pattern, message_lower)
    
    if match:
        start_day = int(match.group(1))
        end_day = int(match.group(2))
        month_name = match.group(3)
        
        # Convert month name to number
        month_map = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
            'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
            'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        
        month_num = month_map.get(month_name.lower(), 8)  # Default to August
        
        try:
            start_date = f"{current_year}-{month_num:02d}-{start_day:02d}"
            end_date = f"{current_year}-{month_num:02d}-{end_day:02d}"
            
            return {
                "search_type": "range",
                "date_range_start": start_date,
                "date_range_end": end_date,
                "range_description": f"between {start_day}th to {end_day}th {month_name}"
            }
        except:
            pass
    
    return {}


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
    """Enhanced flight search supporting both one-way and round-trip flights"""
    
    if not state.get("from_city") or not state.get("to_city"):
        state["response_text"] = "I need more information. Please specify: departure city and destination city."
        return state
    
    search_type = state.get("search_type", "specific")
    trip_type = state.get("trip_type", "one-way")
    
    if search_type == "specific":
        if not state.get("departure_date"):
            state["response_text"] = "I need a departure date for your flight search."
            return state
        
        print(f"🔍 Searching {trip_type} flight: {state['departure_date']}")
        
        # Determine which payload to use based on trip type
        if trip_type == "round-trip" and state.get("return_date"):
            return search_roundtrip_flights(state)
        else:
            return search_flights_original(state)
    
    elif search_type == "range":
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

def find_cheapest_flight(state: FlightBookingState) -> FlightBookingState:
    """Corrected analysis - properly identifies complete journeys vs segments"""
    
    # Check if we have bulk search results first
    if state.get("bulk_search_results"):
        return analyze_bulk_search_results(state)
    
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
        
        print(f"🔍 Found {len(offerings)} catalog offerings")
        
        # Debug: Print offering structure
        for i, offering in enumerate(offerings):
            departure = offering.get("Departure", "")
            arrival = offering.get("Arrival", "")
            print(f"   📋 Offering {i+1}: {departure} → {arrival}")
        
        # Check if this is a round-trip request
        trip_type = state.get("trip_type", "one-way")
        has_return_date = bool(state.get("return_date"))
        from_city = str(state.get("from_city", ""))
        to_city = str(state.get("to_city", ""))
        
        print(f"🔍 Trip analysis: type={trip_type}, has_return={has_return_date}, route={from_city}→{to_city}")
        
        if (trip_type == "round-trip" or has_return_date) and len(offerings) >= 2:
            # Look for true round-trip: complete journeys in both directions
            outbound_offering = None
            return_offering = None
            
            for offering in offerings:
                departure = offering.get("Departure", "")
                arrival = offering.get("Arrival", "")
                
                # Complete outbound journey: starts from origin, ends at destination
                if departure == from_city and arrival == to_city:
                    outbound_offering = offering
                    print(f"   ✅ Found complete outbound journey: {departure} → {arrival}")
                
                # Complete return journey: starts from destination, ends at origin
                elif departure == to_city and arrival == from_city:
                    return_offering = offering
                    print(f"   ✅ Found complete return journey: {departure} → {arrival}")
            
            # Only process as round-trip if we have BOTH complete journeys
            if outbound_offering and return_offering:
                print(f"🔄 Processing true round-trip with complete journeys")
                return process_true_roundtrip(state, outbound_offering, return_offering)
            else:
                print(f"⚠️ Not a true round-trip - falling back to one-way processing")
                print(f"   Outbound complete journey: {bool(outbound_offering)}")
                print(f"   Return complete journey: {bool(return_offering)}")
        
        # Process as one-way (single complete journey, possibly with layovers)
        print(f"➡️ Processing as one-way journey")
        return process_oneway_journey(state, offerings)
            
    except Exception as e:
        print(f"❌ Error in flight analysis: {e}")
        import traceback
        traceback.print_exc()
        state["response_text"] = "😔 Error analyzing flight results."
    
    return state


def process_true_roundtrip(state: FlightBookingState, outbound_offering: Dict, return_offering: Dict) -> FlightBookingState:
    """Process a true round-trip with complete outbound and return journeys"""
    
    try:
        print(f"🔄 Processing true round-trip with separate complete journeys")
        
        # Extract details for each complete journey
        outbound_details = extract_complete_journey_details(outbound_offering, state, "outbound")
        return_details = extract_complete_journey_details(return_offering, state, "return")
        
        if not outbound_details or not return_details:
            state["response_text"] = "✈️ Found round-trip flights but couldn't extract details."
            return state
        
        # Calculate total price
        outbound_price = float(outbound_details.get("price", 0))
        return_price = float(return_details.get("price", 0))
        total_price = outbound_price + return_price
        
        # Combine details for round-trip response
        roundtrip_details = {
            "trip_type": "round-trip",
            "currency": outbound_details.get("currency", "EUR"),
            "price": f"{total_price:.2f}",
            
            # Outbound journey details
            "outbound_departure_time": outbound_details.get("departure_time", "N/A"),
            "outbound_arrival_time": outbound_details.get("arrival_time", "N/A"),
            "outbound_airline": outbound_details.get("airline", "N/A"),
            "outbound_flight_number": outbound_details.get("flight_number", "N/A"),
            "outbound_stops": outbound_details.get("stops", "N/A"),
            "outbound_duration": outbound_details.get("duration", "N/A"),
            "outbound_layover_details": outbound_details.get("layover_details", []),
            
            # Return journey details
            "return_departure_time": return_details.get("departure_time", "N/A"),
            "return_arrival_time": return_details.get("arrival_time", "N/A"),
            "return_airline": return_details.get("airline", "N/A"),
            "return_flight_number": return_details.get("flight_number", "N/A"),
            "return_stops": return_details.get("stops", "N/A"),
            "return_duration": return_details.get("duration", "N/A"),
            "return_layover_details": return_details.get("layover_details", []),
            
            # Combined details
            "baggage": outbound_details.get("baggage", "Standard"),
            "total_duration": calculate_roundtrip_total_duration(outbound_details, return_details)
        }
        
        state["cheapest_flight"] = {"outbound": outbound_offering, "return": return_offering}
        state["response_text"] = format_flight_response(roundtrip_details)
        
        print(f"✅ True round-trip processed: EUR {total_price:.2f} total")
        
        return state
        
    except Exception as e:
        print(f"❌ Error processing true round-trip: {e}")
        return process_oneway_journey(state, [outbound_offering])


def extract_complete_journey_details(offering: Dict, state: FlightBookingState, journey_type: str) -> Optional[Dict]:
    """Extract details for a complete journey (which may have multiple segments/layovers)"""
    
    try:
        # Find the cheapest option within this complete journey
        cheapest_price = float('inf')
        cheapest_option = None
        best_flight_refs = []
        best_terms_ref = None
        
        product_brand_options = offering.get("ProductBrandOptions", [])
        
        for option in product_brand_options:
            flight_refs = option.get("flightRefs", [])
            product_brand_offerings = option.get("ProductBrandOffering", [])
            
            for brand_offering in product_brand_offerings:
                best_price = brand_offering.get("BestCombinablePrice", {})
                if isinstance(best_price, dict):
                    price = best_price.get("TotalPrice", 0)
                    if price and float(price) < cheapest_price:
                        cheapest_price = float(price)
                        cheapest_option = brand_offering
                        best_flight_refs = flight_refs
                        
                        terms_conditions = brand_offering.get("TermsAndConditions", {})
                        if isinstance(terms_conditions, dict):
                            best_terms_ref = terms_conditions.get("termsAndConditionsRef")
        
        if not cheapest_option:
            print(f"⚠️ No valid options found in {journey_type} journey")
            return None
        
        # Extract detailed flight information for the complete journey
        journey_details = extract_journey_details_from_refs(
            best_flight_refs, 
            best_terms_ref,
            cheapest_option, 
            state
        )
        
        print(f"✅ {journey_type.title()} journey: {journey_details['currency']} {journey_details['price']}")
        print(f"   Route: {journey_details['departure_time']} → {journey_details['arrival_time']}")
        print(f"   Stops: {journey_details['stops']}")
        
        return journey_details
        
    except Exception as e:
        print(f"❌ Error extracting {journey_type} journey details: {e}")
        return None


def extract_journey_details_from_refs(flight_refs: List[str], terms_ref: Optional[str], price_option: Dict, state: FlightBookingState) -> Dict:
    """Extract details for a complete journey (possibly with multiple segments)"""
    
    details = {
        "price": "N/A",
        "currency": "EUR",
        "departure_time": "N/A",
        "arrival_time": "N/A",
        "airline": "N/A",
        "stops": "N/A",
        "baggage": "Standard",
        "duration": "N/A",
        "flight_number": "N/A",
        "layover_details": []
    }
    
    try:
        # Extract price information
        best_price = price_option.get("BestCombinablePrice", {})
        if isinstance(best_price, dict):
            details["price"] = str(best_price.get("TotalPrice", "N/A"))
            
            currency_info = best_price.get("CurrencyCode", {})
            if isinstance(currency_info, dict):
                details["currency"] = currency_info.get("value", "EUR")
        
        # Extract flight details from references
        if flight_refs and state.get("raw_api_response"):
            reference_list = state["raw_api_response"].get("CatalogProductOfferingsResponse", {}).get("ReferenceList", [])
            
            # Find flight reference list
            flight_reference_list = None
            terms_and_conditions_list = None
            
            for ref_list in reference_list:
                if ref_list.get("@type") == "ReferenceListFlight":
                    flight_reference_list = ref_list.get("Flight", [])
                elif ref_list.get("@type") == "ReferenceListTermsAndConditions":
                    terms_and_conditions_list = ref_list.get("TermsAndConditions", [])
            
            if flight_reference_list:
                # Get ALL flight segments for this complete journey
                flight_segments = []
                for flight_ref in flight_refs:
                    for flight in flight_reference_list:
                        if flight.get("id") == flight_ref:
                            flight_segments.append(flight)
                            break
                
                # Sort segments by departure time to get correct sequence
                flight_segments.sort(key=lambda x: (
                    x.get("Departure", {}).get("date", ""),
                    x.get("Departure", {}).get("time", "")
                ))
                
                print(f"🔍 Found {len(flight_segments)} segments for this journey")
                for i, segment in enumerate(flight_segments):
                    dep_loc = segment.get("Departure", {}).get("location", "")
                    arr_loc = segment.get("Arrival", {}).get("location", "")
                    dep_time = segment.get("Departure", {}).get("time", "")
                    print(f"   Segment {i+1}: {dep_loc} → {arr_loc} at {dep_time}")
                
                # Extract details from the complete journey (all segments combined)
                if flight_segments:
                    details.update(extract_complete_journey_info(flight_segments))
            
            # Extract baggage information
            if terms_ref and terms_and_conditions_list:
                baggage_info = extract_baggage_allowance(terms_ref, terms_and_conditions_list)
                if baggage_info:
                    details["baggage"] = baggage_info
    
    except Exception as e:
        print(f"⚠️ Error extracting journey details from refs: {e}")
    
    return details


def extract_complete_journey_info(flight_segments: List[Dict]) -> Dict:
    """Extract information for a complete journey with all its segments"""
    
    details = {}
    
    try:
        if not flight_segments:
            return details
        
        # Journey starts with first segment, ends with last segment
        first_segment = flight_segments[0]
        last_segment = flight_segments[-1]
        
        # Overall journey departure (first segment departure)
        departure_info = first_segment.get("Departure", {})
        details["departure_time"] = format_flight_datetime(
            departure_info.get("date", ""),
            departure_info.get("time", ""),
            departure_info.get("location", ""),
            departure_info.get("terminal", "")
        )
        
        # Overall journey arrival (last segment arrival)
        arrival_info = last_segment.get("Arrival", {})
        details["arrival_time"] = format_flight_datetime(
            arrival_info.get("date", ""),
            arrival_info.get("time", ""),
            arrival_info.get("location", "")
        )
        
        # Collect all airlines and flight numbers
        airlines = set()
        flight_numbers = []
        
        for segment in flight_segments:
            carrier = segment.get("carrier", "")
            if carrier:
                airlines.add(get_airline_name(carrier))
                
            number = segment.get("number", "")
            if number:
                flight_numbers.append(f"{carrier}{number}")
        
        if airlines:
            if len(airlines) == 1:
                details["airline"] = list(airlines)[0]
            else:
                details["airline"] = f"Multiple: {', '.join(airlines)}"
        
        if flight_numbers:
            details["flight_number"] = ", ".join(flight_numbers)
        
        # Stops and layover information for the complete journey
        num_segments = len(flight_segments)
        if num_segments == 1:
            details["stops"] = "Direct flight"
            details["layover_details"] = []
        else:
            # Calculate layovers between segments
            layover_info = calculate_layover_details(flight_segments)
            details["layover_details"] = layover_info.get("layover_details", [])
            
            stops_info = f"{num_segments - 1} stop(s)"
            if details["layover_details"]:
                layover_summary = []
                for layover in details["layover_details"]:
                    city = layover["city"]
                    duration = layover["duration"]
                    layover_summary.append(f"{city} ({duration})")
                stops_info += f" via {', '.join(layover_summary)}"
            
            details["stops"] = stops_info
        
        # Total journey duration (from first departure to last arrival)
        total_duration = calculate_total_flight_duration(flight_segments)
        if total_duration:
            details["duration"] = total_duration
        
        print(f"📋 Journey summary: {details['stops']}, Duration: {details['duration']}")
    
    except Exception as e:
        print(f"⚠️ Error extracting complete journey info: {e}")
    
    return details


def process_oneway_journey(state: FlightBookingState, offerings: List[Dict]) -> FlightBookingState:
    """Process one-way journey (single complete journey, possibly with layovers)"""
    
    try:
        print(f"➡️ Processing one-way journey from {len(offerings)} offerings")
        
        # Find the cheapest complete journey
        cheapest_offering = None
        lowest_price = float('inf')
        
        for offering in offerings:
            product_brand_options = offering.get("ProductBrandOptions", [])
            
            for option in product_brand_options:
                product_brand_offerings = option.get("ProductBrandOffering", [])
                
                for brand_offering in product_brand_offerings:
                    best_price = brand_offering.get("BestCombinablePrice", {})
                    if isinstance(best_price, dict):
                        price = best_price.get("TotalPrice", 0)
                        if price and float(price) < lowest_price:
                            lowest_price = float(price)
                            cheapest_offering = offering
        
        if cheapest_offering:
            state["cheapest_flight"] = cheapest_offering
            
            # Extract journey details using the existing function
            journey_details = extract_flight_details(cheapest_offering, state)
            state["response_text"] = format_flight_response(journey_details)
            
            print(f"✅ Found cheapest one-way journey: {journey_details['currency']} {journey_details['price']}")
            print(f"   Complete route: {journey_details['departure_time']} → {journey_details['arrival_time']}")
            print(f"   Stops: {journey_details['stops']}")
        else:
            state["response_text"] = "✈️ I found flights but couldn't determine pricing."
    
    except Exception as e:
        print(f"❌ Error processing one-way journey: {e}")
        state["response_text"] = "😔 Error analyzing flight results."
    
    return state



def search_flights(state: FlightBookingState) -> FlightBookingState:
    """Enhanced flight search ensuring proper round-trip API calls"""
    
    if not state.get("from_city") or not state.get("to_city"):
        state["response_text"] = "I need more information. Please specify: departure city and destination city."
        return state
    
    search_type = state.get("search_type", "specific")
    trip_type = state.get("trip_type", "one-way")
    
    print(f"🔍 Search type: {search_type}, Trip type: {trip_type}")
    print(f"🔍 From: {state.get('from_city')}, To: {state.get('to_city')}")
    print(f"🔍 Departure: {state.get('departure_date')}, Return: {state.get('return_date')}")
    
    if search_type == "specific":
        if not state.get("departure_date"):
            state["response_text"] = "I need a departure date for your flight search."
            return state
        
        # For round-trip flights, ensure we have a return date
        if trip_type == "round-trip" and not state.get("return_date"):
            state["response_text"] = "I need a return date for your round-trip flight search."
            return state
        
        print(f"🔍 Searching {trip_type} flight with proper API call")
        
        try:
            # Build API payload - this already handles round-trip when return_date is provided
            payload = build_flight_search_payload(
                from_city=str(state["from_city"]),
                to_city=str(state["to_city"]),
                departure_date=str(state["departure_date"]),
                return_date=state.get("return_date"),  # This is key for round-trip
                passengers=state.get("passengers", 1),
                passenger_age=state.get("passenger_age", 25)
            )
            
            print(f"✅ Built payload with return_date: {state.get('return_date')}")
            
            # Make API call
            headers = get_api_headers()
            response = requests.post(CATALOG_URL, headers=headers, json=payload)
            response.raise_for_status()
            
            api_result = response.json()
            state["raw_api_response"] = api_result
            
            print(f"✅ API call completed for {trip_type}")
            
        except Exception as e:
            print(f"❌ Flight search error: {e}")
            state["response_text"] = f"😔 Sorry, I couldn't search for flights. Error: {str(e)}"
    
    elif search_type == "range":
        if not state.get("date_range_start") or not state.get("date_range_end"):
            state["response_text"] = "I need a valid date range for your search."
            return state
        
        print(f"🔍 Searching date range: {state['date_range_start']} to {state['date_range_end']}")
        return search_flights_bulk(state)
    
    else:
        state["response_text"] = "Invalid search type specified."
        return state
    
    return state





def extract_roundtrip_flight_details(flight_offering: Dict, state: FlightBookingState) -> Dict:
    """Extract both outbound and return flight details from round-trip search"""
    
    details = {
        "price": "N/A",
        "currency": "EUR",
        "trip_type": "round-trip",
        
        # Outbound flight details
        "outbound_departure_time": "N/A",
        "outbound_arrival_time": "N/A", 
        "outbound_airline": "N/A",
        "outbound_flight_number": "N/A",
        "outbound_stops": "N/A",
        "outbound_duration": "N/A",
        "outbound_layover_details": [],
        
        # Return flight details
        "return_departure_time": "N/A",
        "return_arrival_time": "N/A",
        "return_airline": "N/A", 
        "return_flight_number": "N/A",
        "return_stops": "N/A",
        "return_duration": "N/A",
        "return_layover_details": [],
        
        # Combined details
        "baggage": "Standard",
        "total_duration": "N/A"
    }
    
    try:
        # Extract price information
        cheapest_price = float('inf')
        best_flight_refs = []
        best_currency = "EUR"
        best_terms_and_conditions_ref = None
        
        product_brand_options = flight_offering.get("ProductBrandOptions", [])
        
        for option in product_brand_options:
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
                        
                        currency_info = best_price.get("CurrencyCode", {})
                        if isinstance(currency_info, dict):
                            best_currency = currency_info.get("value", "EUR")
                        details["currency"] = best_currency
                        
                        terms_and_conditions = brand_offering.get("TermsAndConditions", {})
                        if isinstance(terms_and_conditions, dict):
                            best_terms_and_conditions_ref = terms_and_conditions.get("termsAndConditionsRef")
        
        # Extract flight details from reference list
        if best_flight_refs and state.get("raw_api_response"):
            reference_list = state["raw_api_response"].get("CatalogProductOfferingsResponse", {}).get("ReferenceList", [])
            
            flight_reference_list = None
            terms_and_conditions_list = None
            
            for ref_list in reference_list:
                if ref_list.get("@type") == "ReferenceListFlight":
                    flight_reference_list = ref_list.get("Flight", [])
                elif ref_list.get("@type") == "ReferenceListTermsAndConditions":
                    terms_and_conditions_list = ref_list.get("TermsAndConditions", [])
            
            if flight_reference_list:
                print(f"🔍 Processing {len(best_flight_refs)} flight references for round-trip")
                
                # Get all flight segments
                all_flight_segments = []
                for flight_ref in best_flight_refs:
                    for flight in flight_reference_list:
                        if flight.get("id") == flight_ref:
                            all_flight_segments.append(flight)
                            break
                
                # Sort by departure time
                all_flight_segments.sort(key=lambda x: (
                    x.get("Departure", {}).get("date", ""),
                    x.get("Departure", {}).get("time", "")
                ))
                
                # Split into outbound and return segments
                outbound_segments, return_segments = split_roundtrip_segments(
                    all_flight_segments, 
                    str(state["from_city"]), 
                    str(state["to_city"])
                )
                
                print(f"✅ Split segments: {len(outbound_segments)} outbound, {len(return_segments)} return")
                
                # Process outbound flight
                if outbound_segments:
                    outbound_details = process_flight_segments(outbound_segments, "outbound")
                    details.update(outbound_details)
                
                # Process return flight  
                if return_segments:
                    return_details = process_flight_segments(return_segments, "return")
                    details.update(return_details)
                
                # Calculate total trip duration
                if outbound_segments and return_segments:
                    total_duration = calculate_total_trip_duration(outbound_segments, return_segments)
                    if total_duration:
                        details["total_duration"] = total_duration
            
            # Extract baggage info
            if best_terms_and_conditions_ref and terms_and_conditions_list:
                baggage_info = extract_baggage_allowance(best_terms_and_conditions_ref, terms_and_conditions_list)
                if baggage_info:
                    details["baggage"] = baggage_info
                    
        print(f"✅ Round-trip flight details extracted")
        
    except Exception as e:
        print(f"❌ Error extracting round-trip flight details: {e}")
        import traceback
        traceback.print_exc()
    
    return details


def split_roundtrip_segments(all_segments: List[Dict], origin: str, destination: str) -> Tuple[List[Dict], List[Dict]]:
    """Split flight segments into outbound and return journeys"""
    
    outbound_segments = []
    return_segments = []
    
    try:
        # Simple approach: split based on dates and routes
        mid_point = len(all_segments) // 2
        
        # More sophisticated splitting based on route direction
        for i, segment in enumerate(all_segments):
            departure_location = segment.get("Departure", {}).get("location", "")
            arrival_location = segment.get("Arrival", {}).get("location", "")
            
            # If segment starts from origin area, it's likely outbound
            if departure_location == origin or (arrival_location == destination and i < mid_point):
                outbound_segments.append(segment)
            else:
                return_segments.append(segment)
        
        # Fallback: split in half
        if not outbound_segments or not return_segments:
            outbound_segments = all_segments[:mid_point]
            return_segments = all_segments[mid_point:]
            
    except Exception as e:
        print(f"⚠️ Error splitting round-trip segments: {e}")
        mid_point = len(all_segments) // 2
        outbound_segments = all_segments[:mid_point]
        return_segments = all_segments[mid_point:]
    
    return outbound_segments, return_segments


def process_flight_segments(segments: List[Dict], journey_type: str) -> Dict:
    """Process flight segments for either outbound or return journey"""
    
    prefix = f"{journey_type}_"
    details = {}
    
    try:
        if not segments:
            return details
        
        # Calculate layovers
        layover_info = calculate_layover_details(segments)
        details[f"{prefix}layover_details"] = layover_info.get("layover_details", [])
        
        # Get first and last segments
        first_segment = segments[0]
        last_segment = segments[-1]
        
        # Departure info
        departure_info = first_segment.get("Departure", {})
        details[f"{prefix}departure_time"] = format_flight_datetime(
            departure_info.get("date", ""),
            departure_info.get("time", ""),
            departure_info.get("location", ""),
            departure_info.get("terminal", "")
        )
        
        # Arrival info
        arrival_info = last_segment.get("Arrival", {})
        details[f"{prefix}arrival_time"] = format_flight_datetime(
            arrival_info.get("date", ""),
            arrival_info.get("time", ""),
            arrival_info.get("location", "")
        )
        
        # Airlines and flight numbers
        airlines = set()
        flight_numbers = []
        
        for segment in segments:
            carrier = segment.get("carrier", "")
            if carrier:
                airlines.add(get_airline_name(carrier))
                
            number = segment.get("number", "")
            if number:
                flight_numbers.append(f"{carrier}{number}")
        
        if airlines:
            details[f"{prefix}airline"] = ", ".join(airlines) if len(airlines) > 1 else list(airlines)[0]
        
        if flight_numbers:
            details[f"{prefix}flight_number"] = ", ".join(flight_numbers)
        
        # Stops information
        num_segments = len(segments)
        if num_segments == 1:
            details[f"{prefix}stops"] = "Direct flight"
        else:
            stops_info = f"{num_segments - 1} stop(s)"
            if layover_info.get("layover_details"):
                layover_summary = []
                for layover in layover_info["layover_details"]:
                    city = layover["city"]
                    duration = layover["duration"]
                    layover_summary.append(f"{city} ({duration})")
                stops_info += f" via {', '.join(layover_summary)}"
            details[f"{prefix}stops"] = stops_info
        
        # Duration
        total_duration = calculate_total_flight_duration(segments)
        if total_duration:
            details[f"{prefix}duration"] = total_duration
            
    except Exception as e:
        print(f"⚠️ Error processing {journey_type} segments: {e}")
    
    return details


def calculate_total_trip_duration(outbound_segments: List[Dict], return_segments: List[Dict]) -> Optional[str]:
    """Calculate total duration for the entire round trip"""
    
    try:
        if not outbound_segments or not return_segments:
            return None
        
        # Get outbound departure and return arrival
        outbound_departure = outbound_segments[0].get("Departure", {})
        return_arrival = return_segments[-1].get("Arrival", {})
        
        dep_date = outbound_departure.get("date", "")
        dep_time = outbound_departure.get("time", "")
        arr_date = return_arrival.get("date", "")
        arr_time = return_arrival.get("time", "")
        
        if all([dep_date, dep_time, arr_date, arr_time]):
            total_minutes = calculate_time_difference(dep_date, dep_time, arr_date, arr_time)
            if total_minutes and total_minutes > 0:
                # Convert to days, hours, minutes for round trips
                days = total_minutes // (24 * 60)
                remaining_minutes = total_minutes % (24 * 60)
                hours = remaining_minutes // 60
                minutes = remaining_minutes % 60
                
                duration_parts = []
                if days > 0:
                    duration_parts.append(f"{days}d")
                if hours > 0:
                    duration_parts.append(f"{hours}h")
                if minutes > 0 and days == 0:
                    duration_parts.append(f"{minutes}m")
                
                return " ".join(duration_parts) if duration_parts else "Less than 1 hour"
        
        return None
        
    except Exception as e:
        print(f"⚠️ Error calculating total trip duration: {e}")
        return None


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
    """Enhanced analysis with proper round-trip detection and formatting"""
    
    # Check if we have bulk search results first
    if state.get("bulk_search_results"):
        return analyze_bulk_search_results(state)
    
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
            
            # IMPORTANT: Check trip type to determine which extraction function to use
            trip_type = state.get("trip_type", "one-way")
            has_return_date = bool(state.get("return_date"))
            
            print(f"🔍 Trip type: {trip_type}, Has return date: {has_return_date}")
            
            if trip_type == "round-trip" or has_return_date:
                print(f"✅ Using round-trip extraction for {trip_type}")
                flight_details = extract_roundtrip_flight_details(cheapest_flight, state)
            else:
                print(f"✅ Using one-way extraction for {trip_type}")
                flight_details = extract_flight_details(cheapest_flight, state)
            
            state["response_text"] = format_flight_response(flight_details)
            
            print(f"✅ Found cheapest {trip_type} flight: {flight_details['currency']} {flight_details['price']}")
        else:
            state["response_text"] = "✈️ I found flights but couldn't determine pricing."
            
    except Exception as e:
        print(f"❌ Error in enhanced flight analysis: {e}")
        import traceback
        traceback.print_exc()
        state["response_text"] = "😔 Error analyzing flight results."
    
    return state


def extract_flight_details(flight_offering: Dict, state: FlightBookingState, override_date: Optional[str] = None) -> Dict:
    """Enhanced flight details extraction with ReferenceList parsing, baggage information, and layover calculation"""
    
    details = {
        "price": "N/A",
        "currency": "EUR",
        "departure_time": "N/A",
        "arrival_time": "N/A",
        "airline": "N/A",
        "stops": "N/A",
        "baggage": "Standard",
        "duration": "N/A",
        "flight_number": "N/A",
        "layover_details": [],  # New field for detailed layover info
        "total_layover_time": "N/A"  # New field for total layover time
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
        
        # Step 2: Look up flight details in ReferenceList with enhanced layover calculation
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
                
                # Sort segments by departure time for proper layover calculation
                flight_segments.sort(key=lambda x: (
                    x.get("Departure", {}).get("date", ""),
                    x.get("Departure", {}).get("time", "")
                ))
                
                # Step 3: Enhanced layover calculation and flight details extraction
                if flight_segments:
                    # Calculate layover details
                    layover_info = calculate_layover_details(flight_segments)
                    details.update(layover_info)
                    
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
                    
                    # Enhanced stops information with layover details
                    num_segments = len(flight_segments)
                    if num_segments == 1:
                        details["stops"] = "Direct flight"
                    else:
                        stops_info = f"{num_segments - 1} stop(s)"
                        
                        # Add layover cities and durations
                        if details["layover_details"]:
                            layover_summary = []
                            for layover in details["layover_details"]:
                                city = layover["city"]
                                duration = layover["duration"]
                                layover_summary.append(f"{city} ({duration})")
                            
                            stops_info += f" via {', '.join(layover_summary)}"
                        
                        details["stops"] = stops_info
                    
                    # Calculate total flight duration (including layovers)
                    total_travel_duration = calculate_total_flight_duration(flight_segments)
                    if total_travel_duration:
                        details["duration"] = total_travel_duration
                    elif total_duration_minutes > 0:
                        # Fallback to sum of individual flight durations
                        hours = total_duration_minutes // 60
                        minutes = total_duration_minutes % 60
                        if hours > 0:
                            details["duration"] = f"{hours}h {minutes}m"
                        else:
                            details["duration"] = f"{minutes}m"
                    
                    # Flight numbers
                    if flight_numbers:
                        details["flight_number"] = ", ".join(flight_numbers)
                
                print(f"✅ Enhanced flight details with layovers extracted: {details}")
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


def calculate_layover_details(flight_segments: List[Dict]) -> Dict:
    """Calculate detailed layover information between flight segments"""
    
    layover_details = []
    total_layover_minutes = 0
    
    try:
        for i in range(len(flight_segments) - 1):
            current_flight = flight_segments[i]
            next_flight = flight_segments[i + 1]
            
            # Get arrival info of current flight
            current_arrival = current_flight.get("Arrival", {})
            arrival_date = current_arrival.get("date", "")
            arrival_time = current_arrival.get("time", "")
            arrival_location = current_arrival.get("location", "")
            
            # Get departure info of next flight  
            next_departure = next_flight.get("Departure", {})
            departure_date = next_departure.get("date", "")
            departure_time = next_departure.get("time", "")
            
            if all([arrival_date, arrival_time, departure_date, departure_time]):
                layover_duration = calculate_time_difference(
                    arrival_date, arrival_time, departure_date, departure_time
                )
                
                if layover_duration and layover_duration > 0:
                    layover_details.append({
                        "city": get_city_name_enhanced(arrival_location),
                        "airport_code": arrival_location,
                        "duration": format_duration_human_readable(layover_duration),
                        "duration_minutes": layover_duration
                    })
                    total_layover_minutes += layover_duration
                    
                    print(f"✅ Layover calculated: {arrival_location} ({get_city_name_enhanced(arrival_location)}) - {format_duration_human_readable(layover_duration)}")
    
    except Exception as e:
        print(f"⚠️ Error calculating layover details: {e}")
        import traceback
        traceback.print_exc()
    
    return {
        "layover_details": layover_details,
        "total_layover_time": format_duration_human_readable(total_layover_minutes) if total_layover_minutes > 0 else "N/A"
    }


def calculate_time_difference(date1: str, time1: str, date2: str, time2: str) -> Optional[int]:
    """Calculate time difference in minutes between two datetime points"""
    
    try:
        from datetime import datetime
        
        # Parse datetime strings (format: YYYY-MM-DD HH:MM:SS)
        dt1_str = f"{date1} {time1}"
        dt2_str = f"{date2} {time2}"
        
        dt1 = datetime.strptime(dt1_str, "%Y-%m-%d %H:%M:%S")
        dt2 = datetime.strptime(dt2_str, "%Y-%m-%d %H:%M:%S")
        
        # Calculate difference
        time_diff = dt2 - dt1
        
        # Return difference in minutes (ensure it's positive)
        minutes = int(time_diff.total_seconds() / 60)
        return max(0, minutes)  # Ensure non-negative layover time
        
    except Exception as e:
        print(f"⚠️ Error calculating time difference between {date1} {time1} and {date2} {time2}: {e}")
        return None


def calculate_total_flight_duration(flight_segments: List[Dict]) -> Optional[str]:
    """Calculate total flight duration from departure to final arrival (including layovers)"""
    
    try:
        if not flight_segments:
            return None
        
        # Get first departure and last arrival
        first_flight = flight_segments[0]
        last_flight = flight_segments[-1]
        
        first_departure = first_flight.get("Departure", {})
        last_arrival = last_flight.get("Arrival", {})
        
        dep_date = first_departure.get("date", "")
        dep_time = first_departure.get("time", "")
        arr_date = last_arrival.get("date", "")
        arr_time = last_arrival.get("time", "")
        
        if all([dep_date, dep_time, arr_date, arr_time]):
            total_minutes = calculate_time_difference(dep_date, dep_time, arr_date, arr_time)
            if total_minutes and total_minutes > 0:
                return format_duration_human_readable(total_minutes)
        
        return None
        
    except Exception as e:
        print(f"⚠️ Error calculating total flight duration: {e}")
        return None


def format_duration_human_readable(minutes: int) -> str:
    """Format duration in minutes to human readable format"""
    
    try:
        if minutes <= 0:
            return "0m"
        
        if minutes < 60:
            return f"{minutes}m"
        
        hours = minutes // 60
        remaining_minutes = minutes % 60
        
        if remaining_minutes == 0:
            return f"{hours}h"
        else:
            return f"{hours}h {remaining_minutes}m"
            
    except:
        return f"{minutes}m" if minutes > 0 else "0m"


def get_city_name_enhanced(airport_code: str) -> str:
    """Enhanced city name mapping with major connection hubs"""
    
    city_mapping = {
        # Major Middle East connection hubs
        'DOH': 'Doha', 'DXB': 'Dubai', 'AUH': 'Abu Dhabi', 'SHJ': 'Sharjah',
        'MCT': 'Muscat', 'KWI': 'Kuwait City', 'BAH': 'Bahrain', 'RUH': 'Riyadh',
        'JED': 'Jeddah', 'CAI': 'Cairo', 'AMM': 'Amman', 'BEY': 'Beirut',
        
        # European connection hubs
        'IST': 'Istanbul', 'SAW': 'Istanbul Sabiha', 'FRA': 'Frankfurt', 
        'AMS': 'Amsterdam', 'LHR': 'London Heathrow', 'LGW': 'London Gatwick',
        'CDG': 'Paris Charles de Gaulle', 'ORY': 'Paris Orly', 'MUC': 'Munich',
        'ZUR': 'Zurich', 'VIE': 'Vienna', 'ATH': 'Athens', 'FCO': 'Rome',
        'MAD': 'Madrid', 'BCN': 'Barcelona', 'ARN': 'Stockholm', 'CPH': 'Copenhagen',
        'OSL': 'Oslo', 'HEL': 'Helsinki', 'WAW': 'Warsaw', 'PRG': 'Prague',
        
        # Pakistan airports
        'LHE': 'Lahore', 'KHI': 'Karachi', 'ISB': 'Islamabad', 'PEW': 'Peshawar',
        'MUX': 'Multan', 'UET': 'Quetta', 'RYK': 'Rahim Yar Khan', 'BWP': 'Bahawalpur',
        'SKT': 'Sialkot', 'ATG': 'Attock', 'CJL': 'Chitral',
        
        # Indian major airports
        'DEL': 'Delhi', 'BOM': 'Mumbai', 'MAA': 'Chennai', 'BLR': 'Bangalore',
        'HYD': 'Hyderabad', 'CCU': 'Kolkata', 'COK': 'Kochi', 'TRV': 'Trivandrum',
        'AMD': 'Ahmedabad', 'PNQ': 'Pune', 'GOI': 'Goa', 'JAI': 'Jaipur',
        
        # Asian connection hubs
        'SIN': 'Singapore', 'HKG': 'Hong Kong', 'NRT': 'Tokyo Narita', 
        'HND': 'Tokyo Haneda', 'ICN': 'Seoul Incheon', 'KUL': 'Kuala Lumpur',
        'BKK': 'Bangkok', 'TPE': 'Taipei', 'MNL': 'Manila', 'CGK': 'Jakarta',
        'PVG': 'Shanghai Pudong', 'PEK': 'Beijing Capital', 'CAN': 'Guangzhou',
        'SZX': 'Shenzhen', 'CTU': 'Chengdu', 'XIY': 'Xian',
        
        # US major airports
        'JFK': 'New York JFK', 'LGA': 'New York LaGuardia', 'EWR': 'Newark',
        'LAX': 'Los Angeles', 'ORD': 'Chicago O\'Hare', 'DFW': 'Dallas-Fort Worth',
        'MIA': 'Miami', 'SFO': 'San Francisco', 'SEA': 'Seattle', 'BOS': 'Boston',
        'ATL': 'Atlanta', 'DEN': 'Denver', 'PHX': 'Phoenix', 'LAS': 'Las Vegas',
        
        # Canadian airports
        'YYZ': 'Toronto Pearson', 'YVR': 'Vancouver', 'YUL': 'Montreal',
        'YYC': 'Calgary', 'YOW': 'Ottawa',
        
        # Australian/Oceania airports
        'SYD': 'Sydney', 'MEL': 'Melbourne', 'BNE': 'Brisbane', 'PER': 'Perth',
        'AKL': 'Auckland', 'CHC': 'Christchurch',
        
        # African airports
        'JNB': 'Johannesburg', 'CPT': 'Cape Town', 'NBO': 'Nairobi', 'ADD': 'Addis Ababa',
        'LOS': 'Lagos', 'CMN': 'Casablanca', 'TUN': 'Tunis', 'ALG': 'Algiers',
        
        # South American airports
        'GRU': 'São Paulo', 'GIG': 'Rio de Janeiro', 'EZE': 'Buenos Aires',
        'SCL': 'Santiago', 'LIM': 'Lima', 'BOG': 'Bogotá', 'UIO': 'Quito'
    }
    
    return city_mapping.get(airport_code, airport_code)


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
    """Enhanced flight response formatting with proper round-trip detection"""
    
    trip_type = details.get('trip_type', 'one-way')
    
    # Check if we have round-trip details (both outbound and return info)
    has_outbound = any(key.startswith('outbound_') for key in details.keys())
    has_return = any(key.startswith('return_') for key in details.keys())
    
    if trip_type == "round-trip" or (has_outbound and has_return):
        # Format round-trip response
        print(f"🔄 Formatting round-trip response with outbound: {has_outbound}, return: {has_return}")
        
        outbound_layover_section = ""
        if details.get("outbound_layover_details"):
            outbound_layover_section = "\n🔄 Outbound Layovers:"
            for layover in details["outbound_layover_details"]:
                outbound_layover_section += f"\n   • {layover['city']} ({layover['airport_code']}) - {layover['duration']}"
        
        return_layover_section = ""
        if details.get("return_layover_details"):
            return_layover_section = "\n🔄 Return Layovers:"
            for layover in details["return_layover_details"]:
                return_layover_section += f"\n   • {layover['city']} ({layover['airport_code']}) - {layover['duration']}"
        
        response = f"""✈️ ROUND-TRIP FLIGHT FOUND! ✈️

💰 Total Price: {details['currency']} {details['price']} (round-trip)

🛫 OUTBOUND FLIGHT:
📅 Departure: {details.get('outbound_departure_time', 'N/A')}
🛬 Arrival: {details.get('outbound_arrival_time', 'N/A')}
🏢 Airline: {details.get('outbound_airline', 'N/A')}
✈️ Flight: {details.get('outbound_flight_number', 'N/A')}
🔄 Stops: {details.get('outbound_stops', 'N/A')}{outbound_layover_section}
⏱️ Duration: {details.get('outbound_duration', 'N/A')}

🏠 RETURN FLIGHT:
📅 Departure: {details.get('return_departure_time', 'N/A')}
🛬 Arrival: {details.get('return_arrival_time', 'N/A')}
🏢 Airline: {details.get('return_airline', 'N/A')}
✈️ Flight: {details.get('return_flight_number', 'N/A')}
🔄 Stops: {details.get('return_stops', 'N/A')}{return_layover_section}
⏱️ Duration: {details.get('return_duration', 'N/A')}

🧳 Baggage: {details.get('baggage', 'Standard')}
⏰ Total Trip Duration: {details.get('total_duration', 'N/A')}

❓ Would you like me to search for more options or help you with booking?"""
        
        return response
    
    else:
        # One-way flight formatting
        print(f"➡️ Formatting one-way response")
        
        layover_section = ""
        if details.get("layover_details"):
            layover_section = "\n🔄 Layovers:"
            for layover in details["layover_details"]:
                layover_section += f"\n   • {layover['city']} ({layover['airport_code']}) - {layover['duration']}"
        
        response = f"""✈️ FLIGHT FOUND! ✈️

💰 Price: {details['currency']} {details['price']}
🛫 Departure: {details['departure_time']}
🛬 Arrival: {details['arrival_time']}
🏢 Airline: {details['airline']}
✈️ Flight: {details['flight_number']}
🔄 Stops: {details['stops']}{layover_section}
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