"""
General conversation agent for handling non-flight related queries
"""

import re
import os
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Initialize LLM for general conversation
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

# llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0)


def handle_general_conversation(user_message: str, user_id: str = "unknown", conversation_context: str = "") -> str:
    """
    Handle all non-flight conversations using natural LLM chat
    
    Args:
        user_message: The user's message
        user_id: Unique identifier for the user
        conversation_context: Previous conversation history
        
    Returns:
        str: Natural LLM response
    """
    
    # Route everything to LLM conversation with context
    return handle_llm_conversation(user_message, conversation_context)


def handle_llm_conversation(user_message: str, conversation_context: str = "") -> str:
    """Handle all general conversation using LLM with context and memory"""
    
    # Build the conversation prompt with context
    context_section = ""
    if conversation_context:
        context_section = f"\n{conversation_context}\n"
    
    conversation_prompt = f"""You are TazaTicket's friendly flight booking assistant. You can help with flight searches and bookings.
{context_section}
Current user message: "{user_message}"

Instructions:
- Use emojis appropriately to make your responses more engaging and easier to read on WhatsApp
- Use the previous conversation context to maintain continuity and remember what was discussed
- If they're asking about your capabilities or what you do, explain that you're a flight booking assistant who can search for flights worldwide
- For greetings, be warm and friendly with emojis (👋, 😊, ✈️), and offer to help with travel
- For general topics (jokes, weather, life, etc.), respond naturally and conversationally with relevant emojis
- If they ask about flights specifically, remind them you can help search for real flights using travel emojis (✈️, 🌍, 🎫)
- Keep responses concise and engaging
- Reference previous conversation when relevant (e.g., "As we were discussing...", "Following up on...")
- Don't always mention flights unless relevant - be natural
- Use emojis like: 👋 for greetings, ✈️ for flights, 🌍 for travel, 😊 for friendly responses, ❓ for questions, 💡 for suggestions, etc."""

    try:
        print(f"💬 Having LLM conversation with context: {user_message}")
        response = llm.invoke([HumanMessage(content=conversation_prompt)])
        content = response.content if isinstance(response.content, str) else str(response.content)
        
        print(f"✅ LLM conversation response generated")
        return content.strip()
        
    except Exception as e:
        print(f"❌ Error in LLM conversation: {e}")
        return """I'm TazaTicket's flight booking assistant! I can help you search for flights worldwide, but I'm also happy to chat about anything. How can I help you today?"""


# All preset responses removed - everything now goes through LLM conversation 