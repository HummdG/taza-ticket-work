"""
Memory management service for maintaining conversation context per user
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json


class ConversationMemory:
    """Manages conversation memory for a single user"""
    
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.general_history: List[Dict] = []
        self.flight_context: Dict = {}
        self.flight_collection_state: Dict = {}  # Track partial flight info collection
        self.last_activity = datetime.now()
        
    def add_message(self, user_message: str, bot_response: str, message_type: str = "general"):
        """Add a message exchange to memory"""
        message_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "bot_response": bot_response,
            "type": message_type
        }
        
        self.general_history.append(message_entry)
        
        # Keep only the most recent messages
        if len(self.general_history) > self.max_messages:
            self.general_history = self.general_history[-self.max_messages:]
            
        self.last_activity = datetime.now()
    
    def add_flight_context(self, context: Dict):
        """Add or update flight-related context"""
        self.flight_context.update(context)
        self.last_activity = datetime.now()
    
    def get_conversation_context(self, max_recent: int = 6) -> str:
        """Get formatted conversation context for LLM"""
        if not self.general_history:
            return ""
        
        # Get the most recent messages
        recent_messages = self.general_history[-max_recent:]
        
        context_lines = ["Previous conversation:"]
        for msg in recent_messages:
            context_lines.append(f"User: {msg['user_message']}")
            context_lines.append(f"Assistant: {msg['bot_response']}")
        
        return "\n".join(context_lines)
    
    def get_flight_context(self) -> Dict:
        """Get flight-related context"""
        return self.flight_context.copy()
    
    def clear_flight_context(self):
        """Clear flight context (e.g., after successful booking)"""
        self.flight_context.clear()
        self.flight_collection_state.clear()
        self.last_activity = datetime.now()
    
    def set_flight_collection_state(self, state: Dict):
        """Set the flight information collection state"""
        self.flight_collection_state = state.copy()
        self.last_activity = datetime.now()
    
    def get_flight_collection_state(self) -> Dict:
        """Get the current flight information collection state"""
        return self.flight_collection_state.copy()
    
    def is_collecting_flight_info(self) -> bool:
        """Check if currently collecting flight information"""
        return bool(self.flight_collection_state.get("collecting", False))
    
    def clear_flight_collection_state(self):
        """Clear the flight collection state"""
        self.flight_collection_state.clear()
        self.last_activity = datetime.now()
    
    def is_expired(self, hours: int = 24) -> bool:
        """Check if memory has expired"""
        return datetime.now() - self.last_activity > timedelta(hours=hours)


class MemoryManager:
    """Global memory manager for all users"""
    
    def __init__(self):
        self.user_memories: Dict[str, ConversationMemory] = {}
        self.cleanup_interval_hours = 24
    
    def get_user_memory(self, user_id: str) -> ConversationMemory:
        """Get or create memory for a user"""
        if user_id not in self.user_memories:
            self.user_memories[user_id] = ConversationMemory()
        
        return self.user_memories[user_id]
    
    def add_conversation(self, user_id: str, user_message: str, bot_response: str, message_type: str = "general"):
        """Add a conversation exchange for a user"""
        memory = self.get_user_memory(user_id)
        memory.add_message(user_message, bot_response, message_type)
    
    def get_conversation_context(self, user_id: str, max_recent: int = 6) -> str:
        """Get conversation context for a user"""
        if user_id not in self.user_memories:
            return ""
        
        return self.user_memories[user_id].get_conversation_context(max_recent)
    
    def add_flight_context(self, user_id: str, context: Dict):
        """Add flight context for a user"""
        memory = self.get_user_memory(user_id)
        memory.add_flight_context(context)
    
    def get_flight_context(self, user_id: str) -> Dict:
        """Get flight context for a user"""
        if user_id not in self.user_memories:
            return {}
        
        return self.user_memories[user_id].get_flight_context()
    
    def set_flight_collection_state(self, user_id: str, state: Dict):
        """Set flight collection state for a user"""
        memory = self.get_user_memory(user_id)
        memory.set_flight_collection_state(state)
    
    def get_flight_collection_state(self, user_id: str) -> Dict:
        """Get flight collection state for a user"""
        if user_id not in self.user_memories:
            return {}
        
        return self.user_memories[user_id].get_flight_collection_state()
    
    def is_collecting_flight_info(self, user_id: str) -> bool:
        """Check if user is currently in flight info collection mode"""
        if user_id not in self.user_memories:
            return False
        
        return self.user_memories[user_id].is_collecting_flight_info()
    
    def clear_flight_collection_state(self, user_id: str):
        """Clear flight collection state for a user"""
        if user_id in self.user_memories:
            self.user_memories[user_id].clear_flight_collection_state()
    
    def cleanup_expired_memories(self):
        """Remove expired user memories"""
        expired_users = [
            user_id for user_id, memory in self.user_memories.items()
            if memory.is_expired(self.cleanup_interval_hours)
        ]
        
        for user_id in expired_users:
            del self.user_memories[user_id]
            print(f"🧹 Cleaned up expired memory for user: {user_id}")
    
    def clear_user_memory(self, user_id: str):
        """Clear all memory for a user"""
        if user_id in self.user_memories:
            del self.user_memories[user_id]
    
    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics"""
        total_users = len(self.user_memories)
        total_messages = sum(len(memory.general_history) for memory in self.user_memories.values())
        
        return {
            "total_users": total_users,
            "total_messages": total_messages,
            "average_messages_per_user": total_messages / max(total_users, 1)
        }


# Global memory manager instance
memory_manager = MemoryManager() 