"""
Unified Conversation Agent for Flight Booking
Handles slot filling, language detection, IATA mapping, and action selection
"""

import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
try:
    from langdetect import detect, DetectorFactory
    # Set deterministic language detection
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    print("⚠️ langdetect not available, falling back to English")
    LANGDETECT_AVAILABLE = False
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from ..models.schemas import ConversationState, ConversationResponse
from ..payloads.flight_search import (
    build_roundtrip_flight_payload,
    build_oneway_flight_payload,
    build_multi_city_payload
)



# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Metro-area IATA codes for broad searches
METRO_AREA_CODES = {
    "london": "LON",  # LHR/LGW/STN/LCY/LTN
    "new york": "NYC",  # JFK/LGA/EWR
    "paris": "PAR",  # CDG/ORY/BVA
    "tokyo": "TYO",  # NRT/HND
    "moscow": "MOW",  # SVO/DME/VKO
    "milan": "MIL",  # MXP/LIN/BGY
    "rome": "ROM",  # FCO/CIA
    "stockholm": "STO",  # ARN/BMA/NYO
    "berlin": "BER",  # BER/SXF/TXL
    "chicago": "CHI",  # ORD/MDW
    "washington": "WAS",  # DCA/IAD/BWI
    "los angeles": "LAX",
    "san francisco": "SFO",
    "dubai": "DXB",
    "istanbul": "IST",
    "cairo": "CAI",
    "mumbai": "BOM",
    "delhi": "DEL",
    "bangkok": "BKK",
    "singapore": "SIN",
    "sydney": "SYD",
    "melbourne": "MEL",
    "toronto": "YYZ",
    "montreal": "YUL",
    "vancouver": "YVR"
}

# Specific airport mappings
AIRPORT_CODES = {
    "heathrow": "LHR",
    "gatwick": "LGW",
    "stansted": "STN",
    "luton": "LTN",
    "jfk": "JFK",
    "laguardia": "LGA",
    "lga": "LGA",
    "newark": "EWR",
    "ewr": "EWR",
    "charles de gaulle": "CDG",
    "cdg": "CDG",
    "orly": "ORY",
    "schiphol": "AMS",
    "ams": "AMS",
    "frankfurt": "FRA",
    "fra": "FRA",
    "munich": "MUC",
    "muc": "MUC",
    "zurich": "ZUR",
    "zur": "ZUR",
    "madrid": "MAD",
    "mad": "MAD",
    "barcelona": "BCN",
    "bcn": "BCN",
    "arlanda": "ARN",
    "arn": "ARN",
    "karachi": "KHI",
    "khi": "KHI",
    "lahore": "LHE",
    "lhe": "LHE",
    "islamabad": "ISB",
    "isb": "ISB",
    "peshawar": "PEW",
    "pew": "PEW",
    "quetta": "UET",
    "uet": "UET",
    "multan": "MUX",
    "mux": "MUX",
    "athens": "ATH",
    "ath": "ATH"
}


def detect_language(text: str) -> str:
    """Detect language from user text and return BCP47 code"""
    if not LANGDETECT_AVAILABLE:
        return 'en-US'  # Fallback when langdetect is not available
        
    try:
        detected = detect(text)
        # Map to BCP47 codes
        lang_mapping = {
            'en': 'en-US',
            'ur': 'ur-PK',
            'ar': 'ar-SA',
            'fr': 'fr-FR',
            'de': 'de-DE',
            'es': 'es-ES',
            'it': 'it-IT',
            'nl': 'nl-NL',
            'sv': 'sv-SE',
            'tr': 'tr-TR',
            'ru': 'ru-RU',
            'zh': 'zh-CN',
            'ja': 'ja-JP',
            'ko': 'ko-KR',
            'hi': 'hi-IN',
            'th': 'th-TH'
        }
        return lang_mapping.get(detected, 'en-US')
    except:
        return 'en-US'  # Default fallback


def normalize_city_name(city_name: str) -> Optional[str]:
    """
    Normalize city/airport name to IATA code
    Prefers metro-area codes for broader searches
    """
    if not city_name:
        return None
    
    normalized = city_name.lower().strip()
    
    # Check if it's already a valid IATA code
    if len(normalized) == 3 and normalized.isalpha():
        return normalized.upper()
    
    # Check metro-area codes first (preferred for broader searches)
    if normalized in METRO_AREA_CODES:
        return METRO_AREA_CODES[normalized]
    
    # Check specific airport codes
    if normalized in AIRPORT_CODES:
        return AIRPORT_CODES[normalized]
    
    # Check if it contains a known city name
    for city, code in METRO_AREA_CODES.items():
        if city in normalized or normalized in city:
            return code
    
    for airport, code in AIRPORT_CODES.items():
        if airport in normalized or normalized in airport:
            return code
    
    return None  # Will trigger CLARIFY action


def parse_date_natural(date_str: str, context_date: datetime = None) -> Optional[str]:
    """
    Parse natural language dates and return ISO format YYYY-MM-DD
    Handles flexible inputs like "24th August", "next Friday", "in two weeks"
    """
    if not date_str:
        return None
    
    if context_date is None:
        context_date = datetime.now()
    
    date_str = date_str.lower().strip()
    
    # Already in ISO format
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        try:
            parsed = datetime.strptime(date_str, '%Y-%m-%d')
            return parsed.strftime('%Y-%m-%d')
        except:
            return None
    
    # Use LLM for complex date parsing
    date_prompt = f"""
    Parse this date expression into ISO format (YYYY-MM-DD).
    Current date: {context_date.strftime('%Y-%m-%d')}
    Date expression: "{date_str}"
    
    Rules:
    - If year is omitted, use current year ({context_date.year})
    - If the resolved date is in the past, keep it as-is (we'll handle validation separately)
    - Handle natural expressions like "next Friday", "24th August", "in two weeks"
    
    Return ONLY the date in YYYY-MM-DD format, or "INVALID" if cannot parse.
    """
    
    try:
        response = llm.invoke([HumanMessage(content=date_prompt)])
        result = response.content.strip()
        
        if result == "INVALID" or not re.match(r'^\d{4}-\d{2}-\d{2}$', result):
            return None
            
        # Validate the date
        datetime.strptime(result, '%Y-%m-%d')
        return result
    except:
        return None


def validate_dates(depart_date: str, return_date: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate dates against current time and return trip constraints
    Returns (is_valid, error_message)
    """
    try:
        now = datetime.now()
        today = now.date()
        
        depart_dt = datetime.strptime(depart_date, '%Y-%m-%d').date()
        
        # Check if departure is in the past
        if depart_dt < today:
            return False, "That date has passed. Please give a current or future date."
        
        # Check return date if provided
        if return_date:
            return_dt = datetime.strptime(return_date, '%Y-%m-%d').date()
            if return_dt < depart_dt:
                return False, "Return date must be on or after departure date."
        
        return True, None
        
    except ValueError:
        return False, "Invalid date format. Please use a valid date."


def _simple_flight_extraction(user_message: str) -> Dict[str, Any]:
    """
    Simple fallback extraction when LLM fails
    Uses basic pattern matching for common flight terms
    """
    import re
    
    message_lower = user_message.lower()
    
    # Check for flight keywords in multiple languages
    flight_keywords = ['flight', 'fly', 'ticket', 'travel', 'trip', 'لاہور', 'دبئی', 'lahore', 'dubai', 'جانا', 'سفر']
    has_flight_intent = any(keyword in message_lower for keyword in flight_keywords)
    
    if has_flight_intent:
        # Try to extract cities using the IATA mapping
        origin_mentioned = None
        destination_mentioned = None
        
        # Check for known cities
        for city, code in METRO_AREA_CODES.items():
            if city in message_lower:
                if not origin_mentioned:
                    origin_mentioned = city
                elif city != origin_mentioned:
                    destination_mentioned = city
        
        for airport, code in AIRPORT_CODES.items():
            if airport in message_lower:
                if not origin_mentioned:
                    origin_mentioned = airport
                elif airport != origin_mentioned:
                    destination_mentioned = airport
        
        return {
            "origin_mentioned": origin_mentioned,
            "destination_mentioned": destination_mentioned,
            "departure_date_mentioned": None,  # Too complex for simple extraction
            "return_date_mentioned": None,
            "passengers_mentioned": None,
            "trip_type_mentioned": None,
            "clarification_needed": []
        }
    
    return {"clarification_needed": ["Could not understand the request"]}


def extract_flight_info(user_message: str, current_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract flight information from user message using LLM
    Updates only the slots that are mentioned or changed
    """
    
    extraction_prompt = f"""
    Extract flight booking information from this user message: "{user_message}"
    
    Current conversation state:
    - Origin: {current_state.get('origin', 'Not set')}
    - Destination: {current_state.get('destination', 'Not set')}
    - Dates: {current_state.get('dates', 'Not set')}
    - Passengers: {current_state.get('passengers', 'Not set')}
    - Trip type: {current_state.get('trip_type', 'Not set')}
    
    ONLY extract information that is explicitly mentioned or being changed in this message.
    DO NOT fill in information that isn't clearly stated.
    
    For cities/airports:
    - Extract the actual place names (e.g., "London", "New York", "Heathrow")
    - Do not convert to IATA codes yet
    - If user specifies a specific airport, note that
    
    For dates:
    - Extract natural date expressions as-is
    - Note if it's for departure, return, or both
    
    For trip type:
    - "one_way" for one-way trips
    - "return" for round trips
    - "multi_city" for complex itineraries
    
    Return ONLY a JSON object with fields that were mentioned:
    {{
        "origin_mentioned": "place name or null",
        "destination_mentioned": "place name or null",
        "departure_date_mentioned": "natural date expression or null",
        "return_date_mentioned": "natural date expression or null",
        "passengers_mentioned": number or null,
        "trip_type_mentioned": "one_way|return|multi_city or null",
        "clarification_needed": ["list of ambiguous items that need clarification"]
    }}
    """
    
    try:
        response = llm.invoke([HumanMessage(content=extraction_prompt)])
        content = response.content.strip()
        
        print(f"🤖 LLM Response for flight extraction: {content}")
        
        # Clean JSON formatting
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        if not content:
            print("⚠️ Empty response from LLM")
            return {"clarification_needed": ["Could not understand the request"]}
        
        return json.loads(content)
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"📝 Raw content: '{content}'")
        # Fallback to simple extraction
        return _simple_flight_extraction(user_message)
    except Exception as e:
        print(f"❌ Error extracting flight info: {e}")
        print("🔄 Falling back to simple extraction")
        return _simple_flight_extraction(user_message)


def build_response(
    action: str,
    language: str,
    response_mode: str,
    state_update: Dict[str, Any],
    missing_slots: List[str] = None,
    search_payload: Dict[str, Any] = None,
    notes: str = None
) -> ConversationResponse:
    """Build natural language response based on action and state"""
    
    if missing_slots is None:
        missing_slots = []
    
    # Generate natural utterance based on action
    utterance_prompts = {
        "ASK_MISSING": f"Politely ask for missing flight information: {', '.join(missing_slots)}. Be conversational and helpful.",
        "SEARCH": "Confirm that you're searching for flights with the provided details.",
        "CONFIRM_TRIP": "Ask the user to confirm their flight details before searching.",
        "CLARIFY": "Ask for clarification about ambiguous information in a friendly way.",
        "SMALL_TALK": "Respond to general conversation in a helpful, friendly manner.",
        "OTHER": "Provide a helpful response to the user's request."
    }
    
    utterance_prompt = f"""
    Generate a natural, conversational response in {language}.
    Action: {action}
    Context: {utterance_prompts.get(action, 'Respond helpfully')}
    State update: {state_update}
    Notes: {notes or 'None'}
    
    Make the response:
    - Natural and conversational
    - Appropriate for {'speech' if response_mode == 'speech' else 'text'} mode
    - Concise but friendly
    - In the detected language ({language})
    
    Return only the response text, no explanations.
    """
    
    try:
        response = llm.invoke([HumanMessage(content=utterance_prompt)])
        utterance = response.content.strip()
    except:
        # Fallback responses
        fallbacks = {
            "ASK_MISSING": f"I need a bit more information. Could you please provide: {', '.join(missing_slots)}?",
            "SEARCH": "Let me search for flights with your details.",
            "CONFIRM_TRIP": "Please confirm your flight details.",
            "CLARIFY": "Could you clarify that for me?",
            "SMALL_TALK": "How can I help you with your travel plans?",
            "OTHER": "I'm here to help with your flight booking."
        }
        utterance = fallbacks.get(action, "How can I help you?")
    
    return ConversationResponse(
        response_mode=response_mode,
        language=language,
        action=action,
        missing_slots=missing_slots,
        state_update=state_update,
        utterance=utterance,
        search_payload=search_payload,
        notes=notes
    )


def process_conversation_turn(
    user_message: str,
    current_state: ConversationState,
    user_mode: str = "text"
) -> ConversationResponse:
    """
    Main conversation processing function
    Implements the complete slot-filling and action selection logic
    """
    
    # Detect language from user message
    detected_language = detect_language(user_message)
    
    # Extract information from current message
    extracted_info = extract_flight_info(user_message, current_state)
    
    # Initialize state updates
    state_update = {
        "user_message": user_message,
        "language": detected_language,
        "response_mode": user_mode,
        "last_updated": datetime.now().isoformat(),
        "search_stale": False
    }
    
    # Process extracted information and update slots
    any_slot_changed = False
    
    # Process origin
    if extracted_info.get("origin_mentioned"):
        origin_iata = normalize_city_name(extracted_info["origin_mentioned"])
        if origin_iata:
            if current_state.get("origin") != origin_iata:
                state_update["origin"] = origin_iata
                any_slot_changed = True
        else:
            return build_response(
                action="CLARIFY",
                language=detected_language,
                response_mode=user_mode,
                state_update=state_update,
                notes=f"Unknown city/airport: {extracted_info['origin_mentioned']}"
            )
    
    # Process destination
    if extracted_info.get("destination_mentioned"):
        dest_iata = normalize_city_name(extracted_info["destination_mentioned"])
        if dest_iata:
            if current_state.get("destination") != dest_iata:
                state_update["destination"] = dest_iata
                any_slot_changed = True
        else:
            return build_response(
                action="CLARIFY",
                language=detected_language,
                response_mode=user_mode,
                state_update=state_update,
                notes=f"Unknown city/airport: {extracted_info['destination_mentioned']}"
            )
    
    # Process dates
    current_dates = current_state.get("dates", {})
    new_dates = current_dates.copy()
    
    if extracted_info.get("departure_date_mentioned"):
        depart_date = parse_date_natural(extracted_info["departure_date_mentioned"])
        if depart_date:
            if current_dates.get("depart") != depart_date:
                new_dates["depart"] = depart_date
                any_slot_changed = True
        else:
            return build_response(
                action="CLARIFY",
                language=detected_language,
                response_mode=user_mode,
                state_update=state_update,
                notes="Could not understand the date"
            )
    
    if extracted_info.get("return_date_mentioned"):
        return_date = parse_date_natural(extracted_info["return_date_mentioned"])
        if return_date:
            if current_dates.get("return") != return_date:
                new_dates["return"] = return_date
                any_slot_changed = True
        else:
            return build_response(
                action="CLARIFY",
                language=detected_language,
                response_mode=user_mode,
                state_update=state_update,
                notes="Could not understand the return date"
            )
    
    if new_dates != current_dates:
        state_update["dates"] = new_dates
    
    # Process passengers
    if extracted_info.get("passengers_mentioned"):
        passengers = extracted_info["passengers_mentioned"]
        if current_state.get("passengers") != passengers:
            state_update["passengers"] = passengers
            any_slot_changed = True
    
    # Process trip type
    if extracted_info.get("trip_type_mentioned"):
        trip_type = extracted_info["trip_type_mentioned"]
        if current_state.get("trip_type") != trip_type:
            state_update["trip_type"] = trip_type
            any_slot_changed = True
    
    # Mark search as stale if any slot changed
    if any_slot_changed:
        state_update["search_stale"] = True
    
    # Create updated state for validation
    updated_state = {**current_state, **state_update}
    
    # Validate dates if we have them
    if updated_state.get("dates", {}).get("depart"):
        depart = updated_state["dates"]["depart"]
        return_date = updated_state["dates"].get("return")
        
        is_valid, error_msg = validate_dates(depart, return_date)
        if not is_valid:
            return build_response(
                action="CLARIFY",
                language=detected_language,
                response_mode=user_mode,
                state_update=state_update,
                notes=error_msg
            )
    
    # Check for missing required slots
    required_slots = ["origin", "destination", "dates", "passengers", "trip_type"]
    missing_slots = []
    
    for slot in required_slots:
        if slot == "dates":
            if not updated_state.get("dates", {}).get("depart"):
                missing_slots.append("departure date")
        elif not updated_state.get(slot):
            missing_slots.append(slot)
    
    # Action selection logic
    if missing_slots:
        return build_response(
            action="ASK_MISSING",
            language=detected_language,
            response_mode=user_mode,
            state_update=state_update,
            missing_slots=missing_slots
        )
    
    # All slots present - build search payload
    try:
        origin = updated_state["origin"]
        destination = updated_state["destination"]
        dates = updated_state["dates"]
        passengers = updated_state["passengers"]
        trip_type = updated_state["trip_type"]
        
        if trip_type == "return":
            search_payload = build_roundtrip_flight_payload(
                from_city=origin,
                to_city=destination,
                departure_date=dates["depart"],
                return_date=dates["return"],
                passengers=passengers
            )
        elif trip_type == "one_way":
            search_payload = build_oneway_flight_payload(
                from_city=origin,
                to_city=destination,
                departure_date=dates["depart"],
                passengers=passengers
            )
        else:  # multi_city
            # For now, treat as one-way - can be enhanced later
            search_payload = build_oneway_flight_payload(
                from_city=origin,
                to_city=destination,
                departure_date=dates["depart"],
                passengers=passengers
            )
        
        state_update["search_payload"] = search_payload
        
        return build_response(
            action="SEARCH",
            language=detected_language,
            response_mode=user_mode,
            state_update=state_update,
            search_payload=search_payload,
            notes="All required information collected"
        )
        
    except Exception as e:
        return build_response(
            action="CLARIFY",
            language=detected_language,
            response_mode=user_mode,
            state_update=state_update,
            notes=f"Error building search: {str(e)}"
        ) 