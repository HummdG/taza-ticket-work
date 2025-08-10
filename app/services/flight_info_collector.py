"""
Flight information collection service for incomplete flight requests
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Initialize LLM for information extraction
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


class FlightInfoCollector:
    """Manages collection of flight information from partial requests"""
    
    def __init__(self):
        self.required_fields = ["from_city", "to_city", "departure_date"]
        self.optional_fields = ["return_date", "passengers", "passenger_age", "trip_type"]
    
    def extract_flight_info(self, user_message: str, conversation_context: str = "") -> Dict:
        """Extract available flight information from user message"""
        
        today = datetime.now().strftime("%Y-%m-%d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        # --- Fast-path heuristics before LLM ---
        try:
            msg = (user_message or "").strip()
            msg_lower = msg.lower()
            # Numeric-only → passengers update
            if re.fullmatch(r"\d+", msg):
                pax = int(msg)
                pax = max(1, pax)
                return {
                    "from_city": None,
                    "to_city": None,
                    "departure_date": None,
                    "return_date": None,
                    "passengers": pax,
                    "passenger_age": None,
                    "trip_type": None,
                    "flight_intent": True
                }
            # Trip type mentions
            if any(kw in msg_lower for kw in ["round trip", "round-trip", "return trip", "returning", "two way", "2-way"]):
                return {
                    "from_city": None,
                    "to_city": None,
                    "departure_date": None,
                    "return_date": None,
                    "passengers": None,
                    "passenger_age": None,
                    "trip_type": "round-trip",
                    "flight_intent": True
                }
            if any(kw in msg_lower for kw in ["one way", "one-way", "oneway"]):
                return {
                    "from_city": None,
                    "to_city": None,
                    "departure_date": None,
                    "return_date": None,
                    "passengers": None,
                    "passenger_age": None,
                    "trip_type": "one-way",
                    "flight_intent": True
                }
        except Exception:
            pass
        
        context_section = ""
        if conversation_context:
            context_section = f"\nPrevious conversation:\n{conversation_context}\n"
        
        extraction_prompt = f"""
        Today's date is {today}. Extract any available flight information from this message: "{user_message}"
        {context_section}
        Return ONLY a JSON object with these fields (use null for missing information):
        {{
            "from_city": "3-letter airport code or city name or null",
            "to_city": "3-letter airport code or city name or null",
            "departure_date": "YYYY-MM-DD format or null",
            "return_date": "YYYY-MM-DD format or null",
            "passengers": "number or null",
            "passenger_age": "age or null",
            "trip_type": "one-way or round-trip or null",
            "flight_intent": true/false
        }}
        
        Rules:
        - Set flight_intent to TRUE only if the message contains clear flight/travel booking intent
        - Greetings and general smalltalk → flight_intent: false
        - Parse relative dates: "tomorrow" = {tomorrow}, "next week" = 7 days from today
        - If user mentions "return", "round trip", or a return date, set trip_type: "round-trip"
        - If trip_type is "round-trip" but return_date is not specified, keep return_date as null
        - If passengers is not specified, leave it null (we will ask)
        - Use conversation context to remember previously mentioned information
        """
        
        try:
            print(f"🔍 Extracting flight info from: {user_message}")
            response = llm.invoke([HumanMessage(content=extraction_prompt)])
            content = response.content if isinstance(response.content, str) else str(response.content)
            
            # Clean the response
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            extracted_info = json.loads(content)
            print(f"✅ Extracted flight info: {extracted_info}")
            return extracted_info
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"❌ Error extracting flight info: {e}")
            return {"flight_intent": False}
    
    def identify_missing_info(self, flight_info: Dict) -> List[str]:
        """Identify what flight information is still missing (required fields)"""
        missing = []
        
        for field in self.required_fields:
            if not flight_info.get(field):
                missing.append(field)
        
        return missing
    
    def identify_additional_missing_info(self, flight_info: Dict) -> List[str]:
        """Identify additional preferred info to collect before searching (trip_type/return_date, passengers)."""
        missing = []
        trip_type = flight_info.get("trip_type")
        return_date = flight_info.get("return_date")
        passengers = flight_info.get("passengers")
        
        if not trip_type:
            missing.append("trip_type")
        elif trip_type == "round-trip" and not return_date:
            missing.append("return_date")
        
        if passengers in (None, "", 0):
            missing.append("passengers")
        
        return missing
    
    def generate_question_for_missing_info(self, missing_fields: List[str], current_info: Dict) -> str:
        """Generate an appropriate question to collect missing required information"""
        
        if not missing_fields:
            return ""
        
        # Prioritize the questions in a logical order
        if "from_city" in missing_fields:
            return "✈️ Great! I'd love to help you find a flight. 🌍 Which city or airport will you be departing from?"
        
        elif "to_city" in missing_fields:
            from_city = current_info.get("from_city", "")
            return f"🎯 Perfect! And where would you like to fly to{(' from ' + from_city) if from_city else ''}?"
        
        elif "departure_date" in missing_fields:
            from_city = current_info.get("from_city", "")
            to_city = current_info.get("to_city", "")
            return f"📅 Excellent! When would you like to fly{(' from ' + from_city) if from_city else ''}{(' to ' + to_city) if to_city else ''}? You can say 'tomorrow', 'next Friday', or a date like YYYY-MM-DD."
        
        else:
            # All required info collected
            return ""
    
    def generate_question_for_additional_info(self, missing_additional: List[str], current_info: Dict) -> str:
        """Generate question to collect trip type/return date/passengers."""
        if not missing_additional:
            return ""
        
        if "trip_type" in missing_additional:
            return "🔁 Is this a round-trip or one-way? If round-trip, please also share your return date."
        if "return_date" in missing_additional:
            to_city = current_info.get("to_city", "")
            return f"↩️ Noted it's round-trip. What's your return date{(' from ' + to_city) if to_city else ''}?"
        if "passengers" in missing_additional:
            return "👥 How many passengers should I search for? (e.g., 1, 2)"
        return ""
    
    def merge_flight_info(self, existing_info: Dict, new_info: Dict) -> Dict:
        """Merge new flight information with existing information"""
        merged = existing_info.copy()
        
        for key, value in new_info.items():
            if value is not None and value != "":
                merged[key] = value
        
        return merged
    
    def is_flight_info_complete(self, flight_info: Dict) -> bool:
        """Check if all required flight information is available"""
        missing = self.identify_missing_info(flight_info)
        return len(missing) == 0
    
    def format_collected_info_summary(self, flight_info: Dict) -> str:
        """Create a summary of collected information for confirmation"""
        from_city = flight_info.get("from_city", "Unknown")
        to_city = flight_info.get("to_city", "Unknown") 
        departure_date = flight_info.get("departure_date", "Unknown")
        passengers_val = flight_info.get("passengers")
        try:
            passengers = int(passengers_val) if passengers_val not in (None, "") else 1
        except Exception:
            passengers = 1
        
        summary = f"🎉 Perfect! Let me search for flights:\n"
        summary += f"✈️ From: {from_city}\n"
        summary += f"🎯 To: {to_city}\n" 
        summary += f"📅 Departure: {departure_date}\n"
        summary += f"👥 Passengers: {passengers}\n\n"
        summary += "🔍 Searching for the best options..."
        
        return summary


# Global flight info collector instance
flight_collector = FlightInfoCollector() 