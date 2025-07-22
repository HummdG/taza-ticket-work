"""
Natural Speech Converter for Flight Details
Add this to app/services/speech_formatter.py (new file)
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
            # Extract flight details from the response
            flight_details = self._extract_flight_details(flight_response)
            
            if not flight_details:
                # If we can't parse it, clean it up for basic speech
                return self._clean_for_basic_speech(flight_response)
            
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
    
    def _extract_flight_details(self, response: str) -> Optional[Dict]:
        """Extract flight details from structured response"""
        
        details = {}
        
        try:
            # Extract price
            price_match = re.search(r'💰 Price: (\w+) ([\d,]+)', response)
            if price_match:
                details['currency'] = price_match.group(1)
                details['price'] = price_match.group(2).replace(',', '')
            
            # Extract departure info
            departure_match = re.search(r'🛫 Departure: ([\d-]+) from (\w+)', response)
            if departure_match:
                details['departure_date'] = departure_match.group(1)
                details['from_city'] = departure_match.group(2)
            
            # Extract arrival info
            arrival_match = re.search(r'🛬 Arrival: To (\w+)', response)
            if arrival_match:
                details['to_city'] = arrival_match.group(1)
            
            # Extract return date if present
            return_match = re.search(r'Return: ([\d-]+)', response)
            if return_match:
                details['return_date'] = return_match.group(1)
            
            # Extract airline
            airline_match = re.search(r'🏢 Airline: ([^\\n\\r]+)', response)
            if airline_match:
                details['airline'] = airline_match.group(1).strip()
            
            # Extract stops
            stops_match = re.search(r'🔄 Stops: ([^\\n\\r]+)', response)
            if stops_match:
                details['stops'] = stops_match.group(1).strip()
            
            # Extract baggage
            baggage_match = re.search(r'🧳 Baggage: ([^\\n\\r]+)', response)
            if baggage_match:
                details['baggage'] = baggage_match.group(1).strip()
            
            return details if details else None
            
        except Exception as e:
            print(f"⚠️ Error extracting flight details: {e}")
            return None
    
    def _generate_english_speech(self, details: Dict) -> str:
        """Generate natural English speech"""
        
        speech_parts = []
        
        # Greeting
        speech_parts.append("Great news! I found a flight for you.")
        
        # Route
        from_city = self._get_city_name(details.get('from_city', ''))
        to_city = self._get_city_name(details.get('to_city', ''))
        
        if from_city and to_city:
            speech_parts.append(f"Flying from {from_city} to {to_city}")
        
        # Date
        departure_date = details.get('departure_date')
        if departure_date:
            formatted_date = self._format_date_for_speech(departure_date)
            speech_parts.append(f"on {formatted_date}")
        
        # Price
        price = details.get('price')
        currency = details.get('currency', 'EUR')
        if price:
            currency_name = self._get_currency_name(currency)
            speech_parts.append(f"for {price} {currency_name}")
        
        # Airline
        airline = details.get('airline')
        if airline and airline != 'N/A':
            airline_name = self._get_airline_name(airline)
            speech_parts.append(f"with {airline_name}")
        
        # Flight type (stops)
        stops = details.get('stops', '').lower()
        if 'direct' in stops or stops == 'n/a':
            speech_parts.append("This is a direct flight")
        elif stops and stops != 'n/a':
            speech_parts.append(f"with {stops}")
        
        # Return flight
        return_date = details.get('return_date')
        if return_date:
            formatted_return = self._format_date_for_speech(return_date)
            speech_parts.append(f"returning on {formatted_return}")
        
        # Baggage
        baggage = details.get('baggage')
        if baggage and baggage.lower() != 'standard' and baggage != 'N/A':
            speech_parts.append(f"Baggage policy is {baggage}")
        
        # Closing
        speech_parts.append("Would you like me to help you book this flight or search for more options?")
        
        # Join with natural pauses
        return self._join_speech_parts(speech_parts)
    
    def _generate_urdu_hindi_speech(self, details: Dict) -> str:
        """Generate speech template for Urdu/Hindi (will be translated)"""
        
        # Generate in English first - AWS Translate will handle the conversion
        speech_parts = []
        
        # Greeting
        speech_parts.append("Excellent! I have found a wonderful flight for you.")
        
        # Route  
        from_city = self._get_city_name(details.get('from_city', ''))
        to_city = self._get_city_name(details.get('to_city', ''))
        
        if from_city and to_city:
            speech_parts.append(f"The flight goes from {from_city} to {to_city}")
        
        # Date
        departure_date = details.get('departure_date')
        if departure_date:
            formatted_date = self._format_date_for_speech(departure_date)
            speech_parts.append(f"departing on {formatted_date}")
        
        # Price
        price = details.get('price')
        currency = details.get('currency', 'EUR')
        if price:
            currency_name = self._get_currency_name(currency)
            speech_parts.append(f"The ticket price is {price} {currency_name}")
        
        # Airline
        airline = details.get('airline')
        if airline and airline != 'N/A':
            airline_name = self._get_airline_name(airline)
            speech_parts.append(f"You will be flying with {airline_name}")
        
        # Flight type
        stops = details.get('stops', '').lower()
        if 'direct' in stops or stops == 'n/a':
            speech_parts.append("This is a direct flight with no stops")
        
        # Closing
        speech_parts.append("Should I help you book this ticket or would you like to see more flight options?")
        
        return self._join_speech_parts(speech_parts)
    
    def _generate_arabic_speech(self, details: Dict) -> str:
        """Generate speech template for Arabic (will be translated)"""
        
        # Generate in English - AWS Translate will convert to Arabic
        speech_parts = []
        
        speech_parts.append("Wonderful news! I have found an excellent flight for your journey.")
        
        from_city = self._get_city_name(details.get('from_city', ''))
        to_city = self._get_city_name(details.get('to_city', ''))
        
        if from_city and to_city:
            speech_parts.append(f"Your journey will be from {from_city} to {to_city}")
        
        departure_date = details.get('departure_date')
        if departure_date:
            formatted_date = self._format_date_for_speech(departure_date)
            speech_parts.append(f"on {formatted_date}")
        
        price = details.get('price')
        currency = details.get('currency', 'EUR')
        if price:
            currency_name = self._get_currency_name(currency)
            speech_parts.append(f"The cost is {price} {currency_name}")
        
        airline = details.get('airline')
        if airline and airline != 'N/A':
            airline_name = self._get_airline_name(airline)
            speech_parts.append(f"with {airline_name} airline")
        
        speech_parts.append("Would you like me to proceed with booking or show you more flight choices?")
        
        return self._join_speech_parts(speech_parts)
    
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
    
    def _format_date_for_speech(self, date_str: str) -> str:
        """Format date for natural speech"""
        try:
            # Parse date (assuming YYYY-MM-DD format)
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Format for speech
            month_name = date_obj.strftime('%B')  # Full month name
            day = date_obj.day
            year = date_obj.year
            
            # Add ordinal suffix
            if 4 <= day <= 20 or 24 <= day <= 30:
                suffix = "th"
            else:
                suffix = ["st", "nd", "rd"][day % 10 - 1]
            
            return f"{month_name} {day}{suffix}, {year}"
            
        except:
            return date_str
    
    def _join_speech_parts(self, parts: list) -> str:
        """Join speech parts with natural pauses"""
        return ". ".join(parts) + "."
    
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
            'Stops:': 'Stops:',
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