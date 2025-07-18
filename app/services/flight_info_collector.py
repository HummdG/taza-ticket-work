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
        self.optional_fields = ["return_date", "passengers", "passenger_age"]
    
    def extract_flight_info(self, user_message: str, conversation_context: str = "") -> Dict:
        """Extract available flight information from user message"""
        
        today = datetime.now().strftime("%Y-%m-%d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
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
            "flight_intent": true/false
        }}
        
        Rules for flight_intent:
        - Set flight_intent to TRUE only if the message contains clear flight/travel booking intent
        - Simple greetings ("Hi", "Hello", "Hey") should have flight_intent: false
        - General questions ("How are you?", "What's up?") should have flight_intent: false
        - Only set true for messages mentioning: flights, booking, travel plans, destinations, dates in travel context
        
        Other rules:
        - Today is {today}, tomorrow is {tomorrow}
        - Parse relative dates: "tomorrow" = {tomorrow}, "next week" = 7 days from today
        - Use standard 3-letter IATA codes when possible (LHE for Lahore, ATH for Athens)
        - Use conversation context to remember previously mentioned information
        - If user just says "Paris" in context of previous flight discussion, assume it's a destination
        
        Examples:
        - "Hi" → {{"from_city": null, "to_city": null, "departure_date": null, "return_date": null, "passengers": null, "passenger_age": null, "flight_intent": false}}
        - "Hello" → {{"from_city": null, "to_city": null, "departure_date": null, "return_date": null, "passengers": null, "passenger_age": null, "flight_intent": false}}
        - "I want to go to Paris" → {{"from_city": null, "to_city": "CDG", "departure_date": null, "return_date": null, "passengers": null, "passenger_age": null, "flight_intent": true}}
        - "Tomorrow" (in flight context) → {{"from_city": null, "to_city": null, "departure_date": "{tomorrow}", "return_date": null, "passengers": null, "passenger_age": null, "flight_intent": true}}
        - "Book a flight" → {{"from_city": null, "to_city": null, "departure_date": null, "return_date": null, "passengers": null, "passenger_age": null, "flight_intent": true}}
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
        """Identify what flight information is still missing"""
        missing = []
        
        for field in self.required_fields:
            if not flight_info.get(field):
                missing.append(field)
        
        return missing
    
    def generate_question_for_missing_info(self, missing_fields: List[str], current_info: Dict) -> str:
        """Generate an appropriate question to collect missing information"""
        
        if not missing_fields:
            return ""
        
        # Prioritize the questions in a logical order
        if "from_city" in missing_fields:
            return "✈️ Great! I'd love to help you find a flight. 🌍 Which city or airport will you be departing from?"
        
        elif "to_city" in missing_fields:
            from_city = current_info.get("from_city", "")
            return f"🎯 Perfect! And where would you like to fly to from {from_city}?"
        
        elif "departure_date" in missing_fields:
            from_city = current_info.get("from_city", "")
            to_city = current_info.get("to_city", "")
            return f"📅 Excellent! When would you like to fly from {from_city} to {to_city}? You can say dates like 'tomorrow', 'next Friday', or a specific date."
        
        else:
            # All required info collected
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
        passengers = flight_info.get("passengers", 1)
        
        summary = f"🎉 Perfect! Let me search for flights:\n"
        summary += f"✈️ From: {from_city}\n"
        summary += f"🎯 To: {to_city}\n" 
        summary += f"📅 Departure: {departure_date}\n"
        summary += f"👥 Passengers: {passengers}\n\n"
        summary += "🔍 Searching for the best options..."
        
        return summary


# Global flight info collector instance
flight_collector = FlightInfoCollector() 