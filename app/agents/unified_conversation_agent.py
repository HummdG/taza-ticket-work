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
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# LangChain imports for enhanced chaining and memory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains import LLMChain, ConversationChain
from langchain.schema import BaseOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import ChatMessageHistory

from ..models.schemas import ConversationState, ConversationResponse
from ..payloads.flight_search import (
    build_roundtrip_flight_payload,
    build_oneway_flight_payload,
    build_multi_city_payload
)



# Initialize LLMs with chaining
main_llm = ChatOpenAI(model="gpt-4o", temperature=0)
reformulation_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Initialize conversation memory for each user
user_memories = {}

def get_conversation_memory(user_id: str) -> ConversationBufferMemory:
    """Get or create conversation buffer memory for a user"""
    if user_id not in user_memories:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=2000,  # Limit memory size to prevent token overflow
            human_prefix="User",
            ai_prefix="Assistant"
        )
        user_memories[user_id] = memory
        print(f"🧠 Created new conversation memory for user: {user_id}")
    return user_memories[user_id]

# Query Reformulation Chain
reformulation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query reformulation expert for flight booking conversations. 
    Your job is to extract ONLY the essential flight booking information from user messages and conversation history.
    
    Remove noise, filler words, repetition, and non-flight-related content.
    Keep only: locations, dates, passenger count, trip type preferences, and direct flight requests.
    
    Input: Raw user message + conversation history
    Output: Clean, essential information only
    
    Examples:
    Input: "Um, well, I was thinking, you know, maybe I want to go from London to Paris next Friday for like 2 people"
    Output: "London to Paris next Friday, 2 passengers"
    
    Input: "Hello how are you? By the way I need flights from Lahore to Dubai on September 6th"  
    Output: "Lahore to Dubai September 6th"
    
    Return only the reformulated query, nothing else."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "User message: {user_message}\n\nReformulated query:")
])

# Flight Information Extraction Chain
extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a flight information extraction expert. Extract flight booking details from user messages.
    
    CRITICAL: Return ONLY valid JSON with the exact structure below. No explanations, no markdown formatting.
    
    Available trip types: "one_way", "return", "multi_city"
    Use metro-area IATA codes when possible (LON, NYC, PAR) unless user specifies exact airport.
    
    JSON structure:
    {{
        "origin_mentioned": "place name or null",
        "destination_mentioned": "place name or null", 
        "departure_date_mentioned": "natural date expression or null",
        "return_date_mentioned": "natural date expression or null",
        "passengers_mentioned": number or null,
        "trip_type_mentioned": "one_way|return|multi_city or null",
        "clarification_needed": ["list of ambiguous items or empty array"]
    }}"""),
    ("human", "Extract flight information from: {message}")
])

extraction_chain = extraction_prompt | main_llm | JsonOutputParser()

# Main Conversation Chain with Memory
conversation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are TazaTicket's flight booking assistant. You help users find and book flights.

    Current conversation state: {state_summary}
    User's language: {language}
    Response mode: {response_mode}

    RULES:
    1. Always respond in the user's detected language
    2. Be concise and natural - perfect for {response_mode} mode
    3. Never re-ask for information you already have in the conversation history
    4. When all required info is complete, confirm before searching
    5. Guide users step by step through missing information
    6. Reference previous conversation context naturally

    Required for flight search: origin, destination, departure date, passengers, trip type
    
    Action needed: {action}
    Missing info: {missing_slots}
    
    Response should be helpful, friendly, and focused on getting flight details."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_message}")
])

def create_conversation_chain(user_id: str) -> ConversationChain:
    """Create a conversation chain with memory for a specific user"""
    memory = get_conversation_memory(user_id)
    
    return ConversationChain(
        llm=main_llm,
        prompt=conversation_prompt,
        memory=memory,
        verbose=False
    )

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
        response = main_llm.invoke([HumanMessage(content=date_prompt)])
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


def extract_flight_info_with_chains(user_message: str, current_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract flight information using LangChain with query reformulation and conversation memory
    """
    try:
        user_id = current_state.get('user_id', 'unknown')
        memory = get_conversation_memory(user_id)
        
        # Step 1: Reformulate query to remove noise using conversation memory
        print(f"🔄 Reformulating query with conversation memory: {user_message}")
        
        reformulation_chain = LLMChain(
            llm=reformulation_llm,
            prompt=reformulation_prompt,
            memory=memory,
            verbose=False
        )
        
        reformulated_response = reformulation_chain.predict(user_message=user_message)
        reformulated_content = reformulated_response.strip()
        print(f"✨ Reformulated: {reformulated_content}")
        
        # Step 2: Extract flight information from clean query
        print(f"🎯 Extracting flight info from reformulated query")
        extracted = extraction_chain.invoke({"message": reformulated_content})
        
        print(f"📊 Extracted: {extracted}")
        return extracted
        
    except Exception as e:
        print(f"❌ Error in chain-based extraction: {e}")
        print("🔄 Falling back to simple extraction")
        return _simple_flight_extraction(user_message)


def _format_conversation_history(history: List[Dict]) -> str:
    """Format conversation history for reformulation context"""
    if not history:
        return "No previous conversation"
    
    recent_history = history[-3:]  # Keep last 3 exchanges
    formatted = []
    
    for exchange in recent_history:
        role = exchange.get('role', 'unknown')
        content = exchange.get('content', '')
        if role == 'user':
            formatted.append(f"User: {content}")
        elif role == 'assistant':
            formatted.append(f"Assistant: {content}")
    
    return " | ".join(formatted)


def extract_flight_info(user_message: str, current_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract flight information - now using enhanced chains
    """
    return extract_flight_info_with_chains(user_message, current_state)


def build_response_with_chain(
    action: str,
    language: str,
    response_mode: str,
    state_update: Dict[str, Any],
    conversation_history: List[Dict] = None,
    user_message: str = "",
    missing_slots: List[str] = None,
    search_payload: Dict[str, Any] = None,
    notes: str = None
) -> ConversationResponse:
    """Build natural language response using conversation chain with buffer memory"""
    
    if missing_slots is None:
        missing_slots = []
    
    user_id = state_update.get('user_id', 'unknown')
    
    try:
        # Create conversation chain with memory for this user
        chain = create_conversation_chain(user_id)
        
        # Create state summary
        state_summary = _create_state_summary(state_update)
        
        # Prepare input for chain
        chain_input = {
            "user_message": user_message,
            "action": action,
            "missing_slots": missing_slots,
            "state_summary": state_summary,
            "language": language,
            "response_mode": response_mode
        }
        
        print(f"🗣️ Generating response with conversation chain and memory")
        
        # Use the conversation chain with memory
        response = chain.predict(
            user_message=user_message,
            action=action,
            missing_slots=", ".join(missing_slots) if missing_slots else "None",
            state_summary=state_summary,
            language=language,
            response_mode=response_mode
        )
        
        utterance = response.strip()
        print(f"💬 Generated response: {utterance}")
        
    except Exception as e:
        print(f"❌ Error in conversation chain: {e}")
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


def _create_state_summary(state: Dict[str, Any]) -> str:
    """Create a readable summary of current conversation state"""
    summary_parts = []
    
    if state.get('origin'):
        summary_parts.append(f"From: {state['origin']}")
    if state.get('destination'):
        summary_parts.append(f"To: {state['destination']}")
    if state.get('dates', {}).get('depart'):
        summary_parts.append(f"Departure: {state['dates']['depart']}")
    if state.get('dates', {}).get('return'):
        summary_parts.append(f"Return: {state['dates']['return']}")
    if state.get('passengers'):
        summary_parts.append(f"Passengers: {state['passengers']}")
    if state.get('trip_type'):
        summary_parts.append(f"Type: {state['trip_type']}")
    
    return " | ".join(summary_parts) if summary_parts else "No flight details yet"


def build_response(
    action: str,
    language: str,
    response_mode: str,
    state_update: Dict[str, Any],
    missing_slots: List[str] = None,
    search_payload: Dict[str, Any] = None,
    notes: str = None
) -> ConversationResponse:
    """Legacy build_response function - redirects to chain-based version"""
    
    return build_response_with_chain(
        action=action,
        language=language,
        response_mode=response_mode,
        state_update=state_update,
        conversation_history=[],
        user_message="",
        missing_slots=missing_slots,
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
            return build_response_with_chain(
                action="CLARIFY",
                language=detected_language,
                response_mode=user_mode,
                state_update=state_update,
                conversation_history=current_state.get('conversation_history', []),
                user_message=user_message,
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
            return build_response_with_chain(
                action="CLARIFY",
                language=detected_language,
                response_mode=user_mode,
                state_update=state_update,
                conversation_history=current_state.get('conversation_history', []),
                user_message=user_message,
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
            return build_response_with_chain(
                action="CLARIFY",
                language=detected_language,
                response_mode=user_mode,
                state_update=state_update,
                conversation_history=current_state.get('conversation_history', []),
                user_message=user_message,
                notes="Could not understand the date"
            )
    
    if extracted_info.get("return_date_mentioned"):
        return_date = parse_date_natural(extracted_info["return_date_mentioned"])
        if return_date:
            if current_dates.get("return") != return_date:
                new_dates["return"] = return_date
                any_slot_changed = True
        else:
            return build_response_with_chain(
                action="CLARIFY",
                language=detected_language,
                response_mode=user_mode,
                state_update=state_update,
                conversation_history=current_state.get('conversation_history', []),
                user_message=user_message,
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
            return build_response_with_chain(
                action="CLARIFY",
                language=detected_language,
                response_mode=user_mode,
                state_update=state_update,
                conversation_history=current_state.get('conversation_history', []),
                user_message=user_message,
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
        return build_response_with_chain(
            action="ASK_MISSING",
            language=detected_language,
            response_mode=user_mode,
            state_update=state_update,
            conversation_history=current_state.get('conversation_history', []),
            user_message=user_message,
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
        
        return build_response_with_chain(
            action="SEARCH",
            language=detected_language,
            response_mode=user_mode,
            state_update=state_update,
            conversation_history=updated_state.get('conversation_history', []),
            user_message=user_message,
            search_payload=search_payload,
            notes="All required information collected"
        )
        
    except Exception as e:
        return build_response_with_chain(
            action="CLARIFY",
            language=detected_language,
            response_mode=user_mode,
            state_update=state_update,
            conversation_history=current_state.get('conversation_history', []),
            user_message=user_message,
            notes=f"Error building search: {str(e)}"
        ) 