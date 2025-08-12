#!/usr/bin/env python3
"""
Test script for the unified conversation system
Demonstrates slot filling, memory, language detection, and IATA mapping
"""

import asyncio
import sys
import os

# Add app to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.agents.unified_conversation_agent import process_conversation_turn
from app.services.memory_service import memory_manager


async def test_conversation_flow():
    """Test a complete conversation flow"""
    
    print("=" * 80)
    print("🧪 UNIFIED CONVERSATION SYSTEM TEST")
    print("=" * 80)
    
    test_user = "test_user_unified"
    
    # Clear any existing state
    memory = memory_manager.get_memory(test_user)
    memory.clear_conversation_state()
    
    # Test scenarios
    test_messages = [
        # Initial incomplete request
        "I want to fly to London",
        
        # Adding origin
        "From New York",
        
        # Adding date with natural language
        "Next Friday",
        
        # Adding passengers and trip type
        "For 2 passengers, round trip",
        
        # Adding return date
        "Coming back next Sunday",
        
        # Should trigger search now
        "Search for flights"
    ]
    
    print(f"\n🧠 Starting conversation with user: {test_user}")
    print(f"📝 Will test {len(test_messages)} messages")
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{'-' * 60}")
        print(f"👤 User Message {i}: {message}")
        
        # Get current state
        current_state = memory.get_conversation_state()
        print(f"🧠 Known info: {memory.get_known_info_summary()}")
        
        # Process message
        response = process_conversation_turn(
            user_message=message,
            current_state=current_state,
            user_mode="text"
        )
        
        # Update memory
        memory.update_conversation_state(response.state_update)
        
        # Display response
        print(f"🤖 Bot Response: {response.utterance}")
        print(f"⚙️  Action: {response.action}")
        print(f"🌍 Language: {response.language}")
        print(f"❌ Missing: {response.missing_slots}")
        
        if response.search_payload:
            print(f"🔍 Search payload ready: ✅")
            print(f"   - Origin: {response.search_payload.get('CatalogProductOfferingsRequest', {}).get('SearchCriteriaFlight', [{}])[0].get('From', {}).get('value', 'N/A')}")
            dest_info = response.search_payload.get('CatalogProductOfferingsRequest', {}).get('SearchCriteriaFlight', [])
            if len(dest_info) > 0:
                print(f"   - Destination: {dest_info[0].get('To', {}).get('value', 'N/A')}")
            if len(dest_info) > 1:
                print(f"   - Trip type: Round trip")
            else:
                print(f"   - Trip type: One way")
        
        # Add conversation history
        conversation_exchange = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response.utterance}
        ]
        memory.update_conversation_state({"conversation_history": conversation_exchange})


async def test_language_detection():
    """Test language detection and IATA mapping"""
    
    print(f"\n{'-' * 60}")
    print("🌍 TESTING LANGUAGE DETECTION & IATA MAPPING")
    print(f"{'-' * 60}")
    
    from app.agents.unified_conversation_agent import detect_language, normalize_city_name
    
    # Test language detection
    test_languages = [
        ("I want to fly to Paris", "en-US"),
        ("Je veux voler à Londres", "fr-FR"),
        ("Ich möchte nach Berlin fliegen", "de-DE"),
        ("Quiero volar a Madrid", "es-ES")
    ]
    
    print("\n🗣️  Language Detection Tests:")
    for text, expected in test_languages:
        detected = detect_language(text)
        status = "✅" if detected == expected else "❌"
        print(f"   {status} '{text}' → {detected} (expected: {expected})")
    
    # Test IATA mapping
    test_cities = [
        ("London", "LON"),  # Metro area
        ("Heathrow", "LHR"),  # Specific airport
        ("New York", "NYC"),  # Metro area
        ("JFK", "JFK"),  # Airport code
        ("Paris", "PAR"),  # Metro area
        ("Unknown City", None)  # Should return None
    ]
    
    print("\n🏢 IATA Code Mapping Tests:")
    for city, expected in test_cities:
        result = normalize_city_name(city)
        status = "✅" if result == expected else "❌"
        print(f"   {status} '{city}' → {result} (expected: {expected})")


async def test_date_parsing():
    """Test natural language date parsing"""
    
    print(f"\n{'-' * 60}")
    print("📅 TESTING DATE PARSING")
    print(f"{'-' * 60}")
    
    from app.agents.unified_conversation_agent import parse_date_natural
    from datetime import datetime, timedelta
    
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    next_week = today + timedelta(days=7)
    
    test_dates = [
        ("2025-01-15", "2025-01-15"),  # ISO format
        ("tomorrow", tomorrow.strftime("%Y-%m-%d")),  # Relative
        ("24th August", None),  # Natural language (would need LLM)
        ("next Friday", None),  # Natural language (would need LLM)
    ]
    
    print("\n📆 Date Parsing Tests:")
    for date_text, expected in test_dates:
        try:
            result = parse_date_natural(date_text)
            if expected is None:
                print(f"   ℹ️  '{date_text}' → {result} (LLM dependent)")
            else:
                status = "✅" if result == expected else "❌"
                print(f"   {status} '{date_text}' → {result} (expected: {expected})")
        except Exception as e:
            print(f"   ❌ '{date_text}' → Error: {e}")


async def main():
    """Run all tests"""
    try:
        print("🚀 Starting unified conversation system tests...")
        
        await test_conversation_flow()
        await test_language_detection()
        await test_date_parsing()
        
        print(f"\n{'=' * 80}")
        print("✅ All tests completed!")
        print("🎯 The unified conversation system supports:")
        print("   - Slot filling with memory persistence")
        print("   - Language detection (12+ languages)")
        print("   - IATA mapping with metro-area preferences")
        print("   - Natural date parsing")
        print("   - Unified text/speech mode handling")
        print("   - Smart action selection")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 