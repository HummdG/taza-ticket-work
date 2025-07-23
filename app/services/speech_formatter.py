"""
Fixed Natural Speech Converter for Flight Details
Replace your app/services/speech_formatter.py with this enhanced version
"""

import re
from datetime import datetime
from typing import Dict, Optional

class FlightSpeechFormatter:
    """Convert structured flight responses into natural human speech"""
    
    def __init__(self):
        # City code to full name mapping for better speech
        self.city_names = {
            'LHE': 'Lahore',
            'BER': 'Berlin',
            'MEL': 'Melbourne', 
            'ATH': 'Athens',
            'DXB': 'Dubai',
            'LHR': 'London Heathrow',
            'LGW': 'London Gatwick',
            'STN': 'London Stansted',
            'JFK': 'New York JFK',
            'LAX': 'Los Angeles',
            'CDG': 'Paris Charles de Gaulle',
            'FRA': 'Frankfurt',
            'IST': 'Istanbul',
            'DOH': 'Doha',
            'SYD': 'Sydney',
            'BOM': 'Mumbai',
            'DEL': 'Delhi',
            'KHI': 'Karachi',
            'ISB': 'Islamabad',
            'NYC': 'New York',
            'LON': 'London',
            'PAR': 'Paris',
            'AUH': 'Abu Dhabi',
            # Add more as needed
        }
        
        # Airlines full names
        self.airline_names = {
            'EK': 'Emirates',
            'PK': 'Pakistan International Airlines',
            'QR': 'Qatar Airways',
            'EY': 'Etihad Airways',
            'TK': 'Turkish Airlines',
            'BA': 'British Airways',
            'LH': 'Lufthansa',
            'AF': 'Air France',
            'KL': 'KLM',
            'SQ': 'Singapore Airlines',
            'CX': 'Cathay Pacific',
            'PC': 'Pegasus Airlines',
            '3U': 'China Eastern Airlines',
            'Various': 'multiple airlines'
        }
    
    def convert_to_natural_speech(self, flight_response: str, detected_language: str = 'en') -> str:
        """
        Convert structured flight response to natural human speech
        
        Args:
            flight_response: The structured flight response text
            detected_language: Language for appropriate phrasing
            
        Returns:
            str: Natural speech text ready for TTS
        """
        
        try:
            print(f"🗣️ Converting to natural speech: '{flight_response[:100]}...'")
            
            # Extract flight details from the response
            flight_details = self._extract_flight_details_enhanced(flight_response)
            
            if not flight_details:
                # If we can't parse it, clean it up for basic speech
                print(f"⚠️ Could not extract flight details, using basic cleanup")
                return self._clean_for_basic_speech(flight_response)
            
            print(f"✅ Extracted flight details for speech: {flight_details}")
            
            # Generate natural speech based on language
            if detected_language in ['ur', 'hi']:
                return self._generate_urdu_hindi_speech(flight_details)
            elif detected_language == 'ar':
                return self._generate_arabic_speech(flight_details)
            else:
                return self._generate_english_speech(flight_details)
                
        except Exception as e:
            print(f"⚠️ Error in natural speech conversion: {e}")
            return self._clean_for_basic_speech(flight_response)
    
    def _extract_flight_details_enhanced(self, response: str) -> Optional[Dict]:
        """Enhanced extraction of flight details from structured response"""
        
        details = {}
        
        try:
            print(f"🔍 Extracting details from response: {response}")
            
            # Extract price (multiple patterns)
            price_patterns = [
                r'💰 Price: (\w+) ([\d,\.]+)',
                r'Price: (\w+) ([\d,\.]+)',
                r'💰.*?(\w+) ([\d,\.]+)'
            ]
            
            for pattern in price_patterns:
                price_match = re.search(pattern, response)
                if price_match:
                    details['currency'] = price_match.group(1)
                    details['price'] = price_match.group(2).replace(',', '')
                    break
            
            # Extract departure info (multiple patterns)
            departure_patterns = [
                r'🛫 Departure: (.+?)(?:\n|$)',
                r'Departure: (.+?)(?:\n|$)',
                r'🛫.*?(.+?)(?:\n|$)'
            ]
            
            for pattern in departure_patterns:
                departure_match = re.search(pattern, response)
                if departure_match:
                    departure_full = departure_match.group(1).strip()
                    details['departure_info'] = departure_full
                    
                    # Try to extract components
                    # Pattern: "Dec 06 at 06:05 (LHE) Terminal M"
                    dep_pattern = r'(\w+ \d+) at (\d+:\d+) \((\w+)\)'
                    dep_match = re.search(dep_pattern, departure_full)
                    if dep_match:
                        details['departure_date'] = dep_match.group(1)
                        details['departure_time'] = dep_match.group(2)
                        details['from_city'] = dep_match.group(3)
                    break
            
            # Extract arrival info (multiple patterns)
            arrival_patterns = [
                r'🛬 Arrival: (.+?)(?:\n|$)',
                r'Arrival: (.+?)(?:\n|$)',
                r'🛬.*?(.+?)(?:\n|$)'
            ]
            
            for pattern in arrival_patterns:
                arrival_match = re.search(pattern, response)
                if arrival_match:
                    arrival_full = arrival_match.group(1).strip()
                    details['arrival_info'] = arrival_full
                    
                    # Try to extract components
                    # Pattern: "Dec 06 at 13:30 (BER)"
                    arr_pattern = r'(\w+ \d+) at (\d+:\d+) \((\w+)\)'
                    arr_match = re.search(arr_pattern, arrival_full)
                    if arr_match:
                        details['arrival_date'] = arr_match.group(1)
                        details['arrival_time'] = arr_match.group(2)
                        details['to_city'] = arr_match.group(3)
                    break
            
            # Extract airline (multiple patterns)
            airline_patterns = [
                r'🏢 Airline: (.+?)(?:\n|$)',
                r'Airline: (.+?)(?:\n|$)',
                r'🏢.*?(.+?)(?:\n|$)'
            ]
            
            for pattern in airline_patterns:
                airline_match = re.search(pattern, response)
                if airline_match:
                    details['airline'] = airline_match.group(1).strip()
                    break
            
            # Extract flight numbers
            flight_patterns = [
                r'✈️ Flight: (.+?)(?:\n|$)',
                r'Flight: (.+?)(?:\n|$)',
                r'✈️.*?(.+?)(?:\n|$)'
            ]
            
            for pattern in flight_patterns:
                flight_match = re.search(pattern, response)
                if flight_match:
                    details['flight_number'] = flight_match.group(1).strip()
                    break
            
            # Extract stops
            stops_patterns = [
                r'🔄 Stops: (.+?)(?:\n|$)',
                r'Stops: (.+?)(?:\n|$)',
                r'🔄.*?(.+?)(?:\n|$)'
            ]
            
            for pattern in stops_patterns:
                stops_match = re.search(pattern, response)
                if stops_match:
                    details['stops'] = stops_match.group(1).strip()
                    break
            
            # Extract duration
            duration_patterns = [
                r'⏱️ Duration: (.+?)(?:\n|$)',
                r'Duration: (.+?)(?:\n|$)',
                r'⏱️.*?(.+?)(?:\n|$)'
            ]
            
            for pattern in duration_patterns:
                duration_match = re.search(pattern, response)
                if duration_match:
                    details['duration'] = duration_match.group(1).strip()
                    break
            
            # Extract baggage
            baggage_patterns = [
                r'🧳 Baggage: (.+?)(?:\n|$)',
                r'Baggage: (.+?)(?:\n|$)',
                r'🧳.*?(.+?)(?:\n|$)'
            ]
            
            for pattern in baggage_patterns:
                baggage_match = re.search(pattern, response)
                if baggage_match:
                    details['baggage'] = baggage_match.group(1).strip()
                    break
            
            print(f"✅ Enhanced extraction results: {details}")
            return details if details else None
            
        except Exception as e:
            print(f"⚠️ Error in enhanced extraction: {e}")
            return None
    
    def _generate_english_speech(self, details: Dict) -> str:
        """Generate comprehensive natural English speech"""
        
        speech_parts = []
        
        # Greeting
        speech_parts.append("Great news! I found an excellent flight for you.")
        
        # Price first (most important) - keep it simple and natural
        price = details.get('price')
        currency = details.get('currency', 'EUR')
        if price:
            currency_name = self._get_currency_name(currency)
            speech_parts.append(f"Price is {price} {currency_name}.")
        
        # Route and airline
        airline = details.get('airline', '')
        if airline and airline != 'N/A':
            airline_name = self._get_airline_name(airline)
            speech_parts.append(f"Flying with {airline_name}.")
        
        # Flight numbers - keep simple
        flight_number = details.get('flight_number', '')
        if flight_number and flight_number != 'N/A':
            speech_parts.append(f"Flight {flight_number}.")
        
        # Departure details - more natural
        departure_info = details.get('departure_info') or details.get('departure_date', '')
        if departure_info:
            # Use full departure info if available
            cleaned_departure = self._clean_time_info(departure_info)
            speech_parts.append(f"Departure {cleaned_departure}.")
        
        # Arrival details - more natural
        arrival_info = details.get('arrival_info') or details.get('arrival_date', '')
        if arrival_info:
            cleaned_arrival = self._clean_time_info(arrival_info)
            speech_parts.append(f"Arrival {cleaned_arrival}.")
        
        # Flight duration - keep simple`
        duration = details.get('duration', '')
        if duration and duration != 'N/A':
            speech_parts.append(f"Flight time {duration}.")
        
        # Stops information - more natural
        stops = details.get('stops', '')
        if stops and stops != 'N/A':
            if 'direct' in stops.lower():
                speech_parts.append("Direct flight.")
            elif 'stop' in stops.lower():
                speech_parts.append(f"{stops}.")
        
        # Baggage information - simplified
        baggage = details.get('baggage', '')
        if baggage and baggage != 'Standard' and baggage != 'N/A':
            if 'fee applies' in baggage.lower():
                speech_parts.append("Baggage fees may apply.")
            elif 'included' in baggage.lower():
                speech_parts.append("First bag included.")
            else:
                speech_parts.append(f"Baggage: {baggage}.")
        
        # Closing - keep simple
        speech_parts.append("Would you like me to search for more options?")
        
        # Join with natural pauses
        return self._join_speech_parts(speech_parts)
    
    def _generate_urdu_hindi_speech(self, details: Dict) -> str:
        """Generate speech template for Urdu/Hindi (will be translated)"""
        
        # Generate comprehensive English first - AWS Translate will handle the conversion
        speech_parts = []
        
        # Greeting
        speech_parts.append("Excellent! I have found a wonderful flight for you.")
        
        # Price
        price = details.get('price')
        currency = details.get('currency', 'EUR')
        if price:
            currency_name = self._get_currency_name(currency)
            speech_parts.append(f"The ticket price is {price} {currency_name}.")
        
        # Airline
        airline = details.get('airline', '')
        if airline and airline != 'N/A':
            airline_name = self._get_airline_name(airline)
            speech_parts.append(f"You will be flying with {airline_name}.")
        
        # Route details
        departure_info = details.get('departure_info', '')
        if departure_info:
            cleaned_departure = self._clean_time_info(departure_info)
            speech_parts.append(f"Your departure is {cleaned_departure}.")
        
        arrival_info = details.get('arrival_info', '')
        if arrival_info:
            cleaned_arrival = self._clean_time_info(arrival_info)
            speech_parts.append(f"Your arrival is {cleaned_arrival}.")
        
        # Duration
        duration = details.get('duration', '')
        if duration and duration != 'N/A':
            speech_parts.append(f"The total journey time is {duration}.")
        
        # Stops
        stops = details.get('stops', '')
        if stops and stops != 'N/A':
            if 'direct' in stops.lower():
                speech_parts.append("This is a direct flight with no stops.")
            else:
                speech_parts.append(f"This flight has {stops}.")
        
        # Closing
        speech_parts.append("Should I help you book this ticket or would you like to see more flight options?")
        
        return self._join_speech_parts(speech_parts)
    
    def _generate_arabic_speech(self, details: Dict) -> str:
        """Generate speech template for Arabic (will be translated)"""
        
        # Generate comprehensive English - AWS Translate will convert to Arabic
        speech_parts = []
        
        speech_parts.append("Wonderful news! I have found an excellent flight for your journey.")
        
        # Price
        price = details.get('price')
        currency = details.get('currency', 'EUR')
        if price:
            currency_name = self._get_currency_name(currency)
            speech_parts.append(f"The cost is {price} {currency_name}.")
        
        # Airline
        airline = details.get('airline', '')
        if airline and airline != 'N/A':
            airline_name = self._get_airline_name(airline)
            speech_parts.append(f"You will travel with {airline_name} airline.")
        
        # Route
        departure_info = details.get('departure_info', '')
        if departure_info:
            cleaned_departure = self._clean_time_info(departure_info)
            speech_parts.append(f"Departure is {cleaned_departure}.")
        
        arrival_info = details.get('arrival_info', '')
        if arrival_info:
            cleaned_arrival = self._clean_time_info(arrival_info)
            speech_parts.append(f"Arrival is {cleaned_arrival}.")
        
        # Duration
        duration = details.get('duration', '')
        if duration and duration != 'N/A':
            speech_parts.append(f"Flight duration is {duration}.")
        
        speech_parts.append("Would you like me to proceed with booking or show you more flight choices?")
        
        return self._join_speech_parts(speech_parts)
    
    def _clean_time_info(self, time_info: str) -> str:
        """Clean time information for better speech"""
        # Remove emojis and clean up
        cleaned = time_info
        cleaned = re.sub(r'[🛫🛬]', '', cleaned)
        cleaned = cleaned.replace('Terminal M', 'from Terminal M')
        cleaned = cleaned.replace('Terminal ', 'Terminal ')
        return cleaned.strip()
    
    def _get_city_name(self, city_code: str) -> str:
        """Convert city code to full name"""
        return self.city_names.get(city_code, city_code)
    
    def _get_airline_name(self, airline: str) -> str:
        """Convert airline code to full name"""
        # Handle cases like "Various" or full names already
        if len(airline) == 2:
            return self.airline_names.get(airline, airline)
        return airline
    
    def _get_currency_name(self, currency: str) -> str:
        """Convert currency code to spoken name"""
        currency_names = {
            'USD': 'US Dollars',
            'EUR': 'Euros', 
            'GBP': 'British Pounds',
            'AED': 'UAE Dirhams',
            'PKR': 'Pakistani Rupees',
            'INR': 'Indian Rupees',
            'SAR': 'Saudi Riyals',
            'QAR': 'Qatari Riyals'
        }
        return currency_names.get(currency, currency)
    
    def _join_speech_parts(self, parts: list) -> str:
        """Join speech parts with natural pauses"""
        return " ".join(parts)
    
    def _clean_for_basic_speech(self, text: str) -> str:
        """Basic cleanup for unstructured text"""
        import re
        
        # Remove emojis
        emoji_pattern = re.compile("["
                                 u"\U0001F600-\U0001F64F"  # emoticons
                                 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                 u"\U0001F680-\U0001F6FF"  # transport & map
                                 u"\U0001F1E0-\U0001F1FF"  # flags
                                 u"\U00002702-\U000027B0"
                                 u"\U000024C2-\U0001F251"
                                 "]+", flags=re.UNICODE)
        
        cleaned = emoji_pattern.sub('', text)
        
        # Replace labels with natural speech
        replacements = {
            'Price:': 'The price is',
            'Departure:': 'Departing',
            'Arrival:': 'Arriving',
            'Airline:': 'Flying with',
            'Flight:': 'Flight number',
            'Stops:': 'This flight has',
            'Duration:': 'Flight time is',
            'Baggage:': 'Baggage allowance is',
            'FLIGHT FOUND!': 'I found a great flight for you!',
            'N/A': 'not specified'
        }
        
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        # Clean up extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned


# Global formatter instance
flight_speech_formatter = FlightSpeechFormatter()

# Convenience function
def format_flight_for_speech(flight_response: str, language: str = 'en') -> str:
    """Convert flight response to natural speech"""
    return flight_speech_formatter.convert_to_natural_speech(flight_response, language)