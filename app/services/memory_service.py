"""
Production-ready AWS DynamoDB memory management service for TazaTicket
Ultra-cost-effective alternative to Redis - typically 10x cheaper
Handles unlimited concurrent users with auto-scaling
Enhanced with unified conversation state management
"""

import os
import json
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
from decimal import Decimal

load_dotenv()

class DynamoDBConversationMemory:
    """DynamoDB-backed conversation memory for ultra-cheap distributed deployment"""
    
    def __init__(self, dynamodb_resource, table_name: str, user_id: str, max_messages: int = 20):
        self.dynamodb = dynamodb_resource
        self.table_name = table_name
        self.user_id = user_id
        self.max_messages = max_messages
        self.table = self.dynamodb.Table(table_name)
        
        # Update last activity
        self._update_last_activity()
    
    def add_message(self, user_message: str, bot_response: str, message_type: str = "general"):
        """Add a message exchange to DynamoDB"""
        try:
            timestamp = datetime.now()
            message_entry = {
                'user_id': self.user_id,
                'sort_key': f"message#{int(timestamp.timestamp() * 1000)}",  # Sortable timestamp
                'data_type': 'conversation',
                'timestamp': timestamp.isoformat(),
                'user_message': user_message[:1000],  # Limit message length
                'bot_response': bot_response[:2000],   # Limit response length
                'message_type': message_type,
                'ttl': int((timestamp + timedelta(hours=24)).timestamp())  # Auto-expire in 24h
            }
            
            # Add to DynamoDB
            self.table.put_item(Item=message_entry)
            
            # Clean up old messages (keep only max_messages)
            self._cleanup_old_messages()
            
            self._update_last_activity()
            
        except Exception as e:
            print(f"❌ Error adding message to DynamoDB: {e}")
    
    def add_flight_context(self, context: Dict):
        """Add or update flight-related context"""
        try:
            current_context = self.get_flight_context()
            current_context.update(context)
            
            timestamp = datetime.now()
            
            # Serialize context data
            def _json_safe(value):
                if isinstance(value, Decimal):
                    # Convert to int if integral, else float
                    return int(value) if value % 1 == 0 else float(value)
                if isinstance(value, dict):
                    return {k: _json_safe(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [_json_safe(v) for v in value]
                if isinstance(value, tuple):
                    return tuple(_json_safe(v) for v in value)
                return value

            safe_context = _json_safe(current_context)
            context_json = json.dumps(safe_context)
            if len(context_json) > 10000:  # Limit context size
                print("⚠️ Flight context too large, truncating...")
                context_json = context_json[:10000]
                safe_context = json.loads(context_json)
            
            # Store flight context
            self.table.put_item(Item={
                'user_id': self.user_id,
                'sort_key': 'flight_context',
                'data_type': 'flight_context',
                'timestamp': timestamp.isoformat(),
                'context_data': safe_context,
                'ttl': int((timestamp + timedelta(hours=24)).timestamp())
            })
            
            self._update_last_activity()
            
        except Exception as e:
            print(f"❌ Error adding flight context to DynamoDB: {e}")
    
    def get_conversation_context(self, max_recent: int = 6) -> str:
        """Get formatted conversation context"""
        try:
            # Query recent conversation messages
            response = self.table.query(
                KeyConditionExpression='user_id = :user_id AND begins_with(sort_key, :message_prefix)',
                ExpressionAttributeValues={
                    ':user_id': self.user_id,
                    ':message_prefix': 'message#'
                },
                ScanIndexForward=False,  # Get newest first
                Limit=max_recent
            )
            
            context_lines = []
            items = response.get('Items', []) if response else []
            
            if items:
                context_lines.append("Previous conversation:")
                # Reverse to show chronological order (oldest to newest)
                for item in reversed(items):
                    user_msg = item.get('user_message', '')[:200]  # Limit length
                    bot_msg = item.get('bot_response', '')[:300]   # Limit length
                    if user_msg and bot_msg:
                        context_lines.append(f"User: {user_msg}")
                        context_lines.append(f"Assistant: {bot_msg}")
            
            # Include ongoing flight info collection state for better continuity
            try:
                collection_state = self.get_flight_collection_state()
                collected_info = collection_state.get("collected_info", {}) if isinstance(collection_state, dict) else {}
                if collection_state.get("collecting") or any(collected_info.values()):
                    context_lines.append("")
                    context_lines.append("Current flight info being collected:")
                    from_city = collected_info.get("from_city") or "Unknown"
                    to_city = collected_info.get("to_city") or "Unknown"
                    dep_date = collected_info.get("departure_date") or "Unknown"
                    ret_date = collected_info.get("return_date") or "Unknown"
                    passengers = collected_info.get("passengers") or "Unknown"
                    context_lines.append(f"From: {from_city}; To: {to_city}; Departure: {dep_date}; Return: {ret_date}; Passengers: {passengers}")
            except Exception as e:
                print(f"⚠️ Error appending collection state to context: {e}")
            
            # Include recent flight context (e.g., last search) if available
            try:
                flight_ctx = self.get_flight_context()
                last_search = flight_ctx.get("last_search") if isinstance(flight_ctx, dict) else None
                if last_search:
                    context_lines.append("")
                    context_lines.append("Recent flight context:")
                    from_city = last_search.get("from_city") or ""
                    to_city = last_search.get("to_city") or ""
                    dep = last_search.get("departure_date") or ""
                    ret = last_search.get("return_date") or ""
                    pax = last_search.get("passengers") or ""
                    context_lines.append(f"Last search → From {from_city} to {to_city} on {dep}{' return ' + ret if ret else ''} for {pax} passenger(s)")
            except Exception as e:
                print(f"⚠️ Error appending flight context to context: {e}")
            
            return "\n".join(context_lines) if context_lines else ""
            
        except Exception as e:
            print(f"⚠️ Error getting conversation context from DynamoDB: {e}")
            return ""
    
    def get_flight_context(self) -> Dict:
        """Get flight-related context"""
        try:
            response = self.table.get_item(
                Key={
                    'user_id': self.user_id,
                    'sort_key': 'flight_context'
                }
            )
            
            if 'Item' in response:
                return response['Item'].get('context_data', {})
            return {}
            
        except Exception as e:
            print(f"⚠️ Error getting flight context from DynamoDB: {e}")
            return {}
    
    def clear_flight_context(self):
        """Clear flight context"""
        try:
            # Delete flight context
            self.table.delete_item(
                Key={
                    'user_id': self.user_id,
                    'sort_key': 'flight_context'
                }
            )
            
            # Delete flight collection state
            self.table.delete_item(
                Key={
                    'user_id': self.user_id,
                    'sort_key': 'flight_collection'
                }
            )
            
            self._update_last_activity()
            
        except Exception as e:
            print(f"⚠️ Error clearing flight context in DynamoDB: {e}")
    
    def set_flight_collection_state(self, state: Dict):
        """Set the flight information collection state"""
        try:
            timestamp = datetime.now()
            
            # Serialize state data
            state_json = json.dumps(state)
            if len(state_json) > 5000:  # Limit state size
                print("⚠️ Flight collection state too large, truncating...")
                state_json = state_json[:5000]
                state = json.loads(state_json)
            
            self.table.put_item(Item={
                'user_id': self.user_id,
                'sort_key': 'flight_collection',
                'data_type': 'flight_collection',
                'timestamp': timestamp.isoformat(),
                'collection_state': state,
                'ttl': int((timestamp + timedelta(hours=24)).timestamp())
            })
            
            self._update_last_activity()
            
        except Exception as e:
            print(f"❌ Error setting flight collection state in DynamoDB: {e}")
    
    def get_flight_collection_state(self) -> Dict:
        """Get the current flight information collection state"""
        try:
            response = self.table.get_item(
                Key={
                    'user_id': self.user_id,
                    'sort_key': 'flight_collection'
                }
            )
            
            if 'Item' in response:
                return response['Item'].get('collection_state', {})
            return {}
            
        except Exception as e:
            print(f"⚠️ Error getting flight collection state from DynamoDB: {e}")
            return {}
    
    def is_collecting_flight_info(self) -> bool:
        """Check if currently collecting flight information"""
        state = self.get_flight_collection_state()
        return bool(state.get("collecting", False))
    
    def clear_flight_collection_state(self):
        """Clear the flight collection state"""
        try:
            self.table.delete_item(
                Key={
                    'user_id': self.user_id,
                    'sort_key': 'flight_collection'
                }
            )
            
            self._update_last_activity()
            
        except Exception as e:
            print(f"⚠️ Error clearing flight collection state in DynamoDB: {e}")
    
    def is_expired(self, hours: int = 24) -> bool:
        """Check if memory has expired"""
        try:
            response = self.table.get_item(
                Key={
                    'user_id': self.user_id,
                    'sort_key': 'last_activity'
                }
            )
            
            if 'Item' not in response:
                return True
            
            last_activity = datetime.fromisoformat(response['Item']['timestamp'])
            return datetime.now() - last_activity > timedelta(hours=hours)
            
        except Exception:
            return True
    
    def _update_last_activity(self):
        """Update last activity timestamp"""
        try:
            timestamp = datetime.now()
            
            self.table.put_item(Item={
                'user_id': self.user_id,
                'sort_key': 'last_activity',
                'data_type': 'activity',
                'timestamp': timestamp.isoformat(),
                'ttl': int((timestamp + timedelta(hours=24)).timestamp())
            })
            
        except Exception as e:
            print(f"⚠️ Error updating last activity in DynamoDB: {e}")
    
    def _cleanup_old_messages(self):
        """Clean up old messages to maintain max_messages limit"""
        try:
            # Query all messages for this user
            response = self.table.query(
                KeyConditionExpression='user_id = :user_id AND begins_with(sort_key, :message_prefix)',
                ExpressionAttributeValues={
                    ':user_id': self.user_id,
                    ':message_prefix': 'message#'
                },
                ScanIndexForward=False  # Get newest first
            )
            
            items = response.get('Items', [])
            
            # If we have more than max_messages, delete the oldest ones
            if len(items) > self.max_messages:
                items_to_delete = items[self.max_messages:]
                
                # Delete old messages in batch (up to 25 at a time)
                for i in range(0, len(items_to_delete), 25):
                    batch = items_to_delete[i:i+25]
                    
                    with self.table.batch_writer() as writer:
                        for item in batch:
                            writer.delete_item(
                                Key={
                                    'user_id': item['user_id'],
                                    'sort_key': item['sort_key']
                                }
                            )
                
                print(f"🧹 Cleaned up {len(items_to_delete)} old messages for user {self.user_id}")
                
        except Exception as e:
            print(f"⚠️ Error cleaning up old messages: {e}")

    def set_conversation_state(self, state: Dict):
        """Set the unified conversation state"""
        try:
            timestamp = datetime.now()
            
            # Clean and serialize state data
            def _json_safe(value):
                if isinstance(value, Decimal):
                    return int(value) if value % 1 == 0 else float(value)
                if isinstance(value, dict):
                    return {k: _json_safe(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [_json_safe(v) for v in value]
                if isinstance(value, tuple):
                    return tuple(_json_safe(v) for v in value)
                return value

            safe_state = _json_safe(state)
            state_json = json.dumps(safe_state)
            
            # Limit state size
            if len(state_json) > 15000:
                print("⚠️ Conversation state too large, truncating...")
                # Keep essential fields
                essential_fields = ['user_id', 'origin', 'destination', 'dates', 'passengers', 'trip_type', 'language', 'response_mode', 'search_stale', 'last_updated']
                truncated_state = {k: v for k, v in safe_state.items() if k in essential_fields}
                safe_state = truncated_state
            
            self.table.put_item(Item={
                'user_id': self.user_id,
                'sort_key': 'conversation_state',
                'data_type': 'conversation_state',
                'timestamp': timestamp.isoformat(),
                'state_data': safe_state,
                'ttl': int((timestamp + timedelta(hours=24)).timestamp())
            })
            
            self._update_last_activity()
            
        except Exception as e:
            print(f"❌ Error setting conversation state in DynamoDB: {e}")

    def get_conversation_state(self) -> Dict:
        """Get the current unified conversation state"""
        try:
            response = self.table.get_item(
                Key={
                    'user_id': self.user_id,
                    'sort_key': 'conversation_state'
                }
            )
            
            if 'Item' in response:
                state_data = response['Item'].get('state_data', {})
                # Ensure required fields exist with defaults
                default_state = {
                    'user_id': self.user_id,
                    'user_message': '',
                    'conversation_history': [],
                    'origin': None,
                    'destination': None,
                    'dates': {},
                    'passengers': None,
                    'trip_type': None,
                    'language': 'en-US',
                    'response_mode': 'text',
                    'search_stale': False,
                    'missing_slots': [],
                    'search_payload': None,
                    'flight_results': None,
                    'last_updated': None
                }
                return {**default_state, **state_data}
            
            # Return default state for new conversations
            return {
                'user_id': self.user_id,
                'user_message': '',
                'conversation_history': [],
                'origin': None,
                'destination': None,
                'dates': {},
                'passengers': None,
                'trip_type': None,
                'language': 'en-US',
                'response_mode': 'text',
                'search_stale': False,
                'missing_slots': [],
                'search_payload': None,
                'flight_results': None,
                'last_updated': None
            }
            
        except Exception as e:
            print(f"⚠️ Error getting conversation state from DynamoDB: {e}")
            return {
                'user_id': self.user_id,
                'user_message': '',
                'conversation_history': [],
                'origin': None,
                'destination': None,
                'dates': {},
                'passengers': None,
                'trip_type': None,
                'language': 'en-US',
                'response_mode': 'text',
                'search_stale': False,
                'missing_slots': [],
                'search_payload': None,
                'flight_results': None,
                'last_updated': None
            }

    def update_conversation_state(self, updates: Dict):
        """Update specific fields in the conversation state"""
        try:
            current_state = self.get_conversation_state()
            
            # Apply updates
            for key, value in updates.items():
                if key == 'dates' and isinstance(value, dict) and isinstance(current_state.get('dates'), dict):
                    # Merge date updates
                    current_state['dates'].update(value)
                elif key == 'conversation_history' and isinstance(value, list):
                    # Append to conversation history, keeping last 10 exchanges
                    current_history = current_state.get('conversation_history', [])
                    current_history.extend(value)
                    current_state['conversation_history'] = current_history[-10:]
                else:
                    current_state[key] = value
            
            # Set timestamp
            current_state['last_updated'] = datetime.now().isoformat()
            
            # Save updated state
            self.set_conversation_state(current_state)
            
        except Exception as e:
            print(f"❌ Error updating conversation state: {e}")

    def clear_conversation_state(self):
        """Clear the conversation state"""
        try:
            self.table.delete_item(
                Key={
                    'user_id': self.user_id,
                    'sort_key': 'conversation_state'
                }
            )
            
            self._update_last_activity()
            
        except Exception as e:
            print(f"⚠️ Error clearing conversation state in DynamoDB: {e}")

    def has_active_conversation(self) -> bool:
        """Check if there's an active conversation with some filled slots"""
        try:
            state = self.get_conversation_state()
            
            # Check if any required slots are filled
            has_origin = bool(state.get('origin'))
            has_destination = bool(state.get('destination'))
            has_dates = bool(state.get('dates', {}).get('depart'))
            has_passengers = bool(state.get('passengers'))
            has_trip_type = bool(state.get('trip_type'))
            
            # Also check if conversation is recent (within 1 hour)
            last_updated = state.get('last_updated')
            if last_updated:
                try:
                    last_update_time = datetime.fromisoformat(last_updated)
                    is_recent = datetime.now() - last_update_time < timedelta(hours=1)
                except:
                    is_recent = False
            else:
                is_recent = False
            
            return (has_origin or has_destination or has_dates or has_passengers or has_trip_type) and is_recent
            
        except Exception:
            return False

    def get_known_info_summary(self) -> str:
        """Get a summary of known information to avoid re-asking"""
        try:
            state = self.get_conversation_state()
            
            summary_parts = []
            
            if state.get('origin'):
                summary_parts.append(f"Origin: {state['origin']}")
                
            if state.get('destination'):
                summary_parts.append(f"Destination: {state['destination']}")
                
            if state.get('dates', {}).get('depart'):
                dates_info = f"Departure: {state['dates']['depart']}"
                if state.get('dates', {}).get('return'):
                    dates_info += f", Return: {state['dates']['return']}"
                summary_parts.append(dates_info)
                
            if state.get('passengers'):
                summary_parts.append(f"Passengers: {state['passengers']}")
                
            if state.get('trip_type'):
                summary_parts.append(f"Trip type: {state['trip_type']}")
            
            return "; ".join(summary_parts) if summary_parts else "No previous information"
            
        except Exception:
            return "No previous information"


class DynamoDBMemoryManager:
    """Production-ready DynamoDB memory manager for unlimited scaling"""
    
    def __init__(self):
        self.dynamodb_resource = self._connect_dynamodb()
        self.table_name = os.getenv('DYNAMODB_TABLE_NAME', 'tazaticket-conversations')
        self.cleanup_interval_hours = 24
        
        # Ensure table exists
        self._ensure_table_exists()
    
    def _connect_dynamodb(self):
        """Connect to DynamoDB with proper error handling"""
        try:
            aws_region = os.getenv('AWS_REGION', 'eu-north-1')
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            
            if not aws_access_key or not aws_secret_key:
                print("❌ AWS credentials not found in environment variables")
                return MockDynamoDBResource()
            
            # Create DynamoDB resource
            dynamodb = boto3.resource(
                'dynamodb',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )
            
            # Test connection by listing tables
            list(dynamodb.tables.all())  # This will raise an exception if connection fails
            
            print(f"✅ DynamoDB connected successfully in region: {aws_region}")
            return dynamodb
            
        except NoCredentialsError:
            print("❌ AWS credentials not configured properly")
            return MockDynamoDBResource()
        except Exception as e:
            print(f"❌ DynamoDB connection failed: {e}")
            print("⚠️ Using mock client (not suitable for production)")
            return MockDynamoDBResource()
    
    def _ensure_table_exists(self):
        """Ensure the conversations table exists or create it"""
        try:
            # Try to describe the table
            table = self.dynamodb_resource.Table(self.table_name)
            table.load()  # This will raise an exception if table doesn't exist
            
            print(f"✅ DynamoDB table '{self.table_name}' exists")
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                print(f"🔧 Creating DynamoDB table: {self.table_name}")
                self._create_table()
            else:
                print(f"❌ Error checking table: {e}")
        except Exception as e:
            print(f"⚠️ Table check skipped (using mock client): {e}")
    
    def _create_table(self):
        """Create the conversations table with optimal settings"""
        try:
            table = self.dynamodb_resource.create_table(
                TableName=self.table_name,
                KeySchema=[
                    {
                        'AttributeName': 'user_id',
                        'KeyType': 'HASH'  # Partition key
                    },
                    {
                        'AttributeName': 'sort_key',
                        'KeyType': 'RANGE'  # Sort key
                    }
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'user_id',
                        'AttributeType': 'S'
                    },
                    {
                        'AttributeName': 'sort_key',
                        'AttributeType': 'S'
                    }
                ],
                BillingMode='PAY_PER_REQUEST',  # On-demand pricing (cheapest for this use case)
                StreamSpecification={
                    'StreamEnabled': False
                },
                TableClass='STANDARD',
                DeletionProtectionEnabled=False
            )
            
            # Wait for table to be created
            print("⏳ Waiting for table to be created...")
            table.wait_until_exists()
            
            # Enable TTL for automatic cleanup
            try:
                self.dynamodb_resource.meta.client.update_time_to_live(
                    TableName=self.table_name,
                    TimeToLiveSpecification={
                        'AttributeName': 'ttl',
                        'Enabled': True
                    }
                )
                print(f"✅ TTL enabled for automatic cleanup")
            except Exception as ttl_error:
                print(f"⚠️ TTL setup warning: {ttl_error}")
            
            print(f"✅ DynamoDB table '{self.table_name}' created successfully")
            
        except Exception as e:
            print(f"❌ Error creating table: {e}")
    
    def get_user_memory(self, user_id: str) -> DynamoDBConversationMemory:
        """Get or create memory for a user"""
        return DynamoDBConversationMemory(self.dynamodb_resource, self.table_name, user_id)
    
    def add_conversation(self, user_id: str, user_message: str, bot_response: str, message_type: str = "general"):
        """Add a conversation exchange for a user"""
        memory = self.get_user_memory(user_id)
        memory.add_message(user_message, bot_response, message_type)
    
    def get_conversation_context(self, user_id: str, max_recent: int = 6) -> str:
        """Get conversation context for a user"""
        memory = self.get_user_memory(user_id)
        return memory.get_conversation_context(max_recent)
    
    def add_flight_context(self, user_id: str, context: Dict):
        """Add flight context for a user"""
        memory = self.get_user_memory(user_id)
        memory.add_flight_context(context)
    
    def get_flight_context(self, user_id: str) -> Dict:
        """Get flight context for a user"""
        memory = self.get_user_memory(user_id)
        return memory.get_flight_context()
    
    def set_flight_collection_state(self, user_id: str, state: Dict):
        """Set flight collection state for a user"""
        memory = self.get_user_memory(user_id)
        memory.set_flight_collection_state(state)
    
    def get_flight_collection_state(self, user_id: str) -> Dict:
        """Get flight collection state for a user"""
        memory = self.get_user_memory(user_id)
        return memory.get_flight_collection_state()
    
    def is_collecting_flight_info(self, user_id: str) -> bool:
        """Check if user is currently in flight info collection mode"""
        memory = self.get_user_memory(user_id)
        return memory.is_collecting_flight_info()
    
    def clear_flight_collection_state(self, user_id: str):
        """Clear flight collection state for a user"""
        memory = self.get_user_memory(user_id)
        memory.clear_flight_collection_state()
    
    def clear_flight_context(self, user_id: str):
        """Clear flight context for a user"""
        memory = self.get_user_memory(user_id)
        memory.clear_flight_context()
    
    def cleanup_expired_memories(self):
        """DynamoDB TTL handles this automatically"""
        print("ℹ️ DynamoDB TTL handles automatic cleanup. Manual cleanup not needed.")
        # TTL automatically deletes expired items, so this is mostly a no-op
    
    def clear_user_memory(self, user_id: str):
        """Clear all memory for a user"""
        try:
            table = self.dynamodb_resource.Table(self.table_name)
            
            # Query all items for this user
            response = table.query(
                KeyConditionExpression='user_id = :user_id',
                ExpressionAttributeValues={
                    ':user_id': user_id
                }
            )
            
            items = response.get('Items', [])
            
            # Delete all items for this user in batches
            for i in range(0, len(items), 25):  # DynamoDB batch limit is 25
                batch = items[i:i+25]
                
                with table.batch_writer() as writer:
                    for item in batch:
                        writer.delete_item(
                            Key={
                                'user_id': item['user_id'],
                                'sort_key': item['sort_key']
                            }
                        )
            
            print(f"🧹 Cleared {len(items)} items for user: {user_id}")
            
        except Exception as e:
            print(f"⚠️ Error clearing user memory: {e}")
    
    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics"""
        try:
            table = self.dynamodb_resource.Table(self.table_name)
            
            # Get table description for item count and size
            table_description = table.meta.client.describe_table(TableName=self.table_name)
            table_info = table_description['Table']
            
            item_count = table_info.get('ItemCount', 0)
            table_size = table_info.get('TableSizeBytes', 0)
            
            # Count unique users (expensive operation, limit it)
            unique_users = "N/A"  # Disable for performance
            
            return {
                "total_users": unique_users,
                "total_items": item_count,
                "table_size_bytes": table_size,
                "table_size_mb": round(table_size / 1024 / 1024, 2) if table_size else 0,
                "average_items_per_user": "N/A",
                "dynamodb_connected": True,
                "table_name": self.table_name,
                "billing_mode": "PAY_PER_REQUEST",
                "region": os.getenv('AWS_REGION', 'unknown')
            }
            
        except Exception as e:
            print(f"⚠️ Error getting memory stats: {e}")
            return {
                "total_users": 0,
                "total_items": 0,
                "table_size_bytes": 0,
                "table_size_mb": 0,
                "average_items_per_user": 0,
                "dynamodb_connected": False,
                "error": str(e)
            }


class MockDynamoDBResource:
    """Mock DynamoDB resource for local development without AWS"""
    
    def __init__(self):
        self.data = {}
        print("⚠️ Using mock DynamoDB resource - not suitable for production!")
    
    def Table(self, table_name):
        return MockTable(self.data)
    
    def tables(self):
        return MockTables()
    
    def create_table(self, **kwargs):
        return MockTable(self.data)


class MockTables:
    """Mock tables collection"""
    
    def all(self):
        return []


class MockTable:
    """Mock DynamoDB table for local development"""
    
    def __init__(self, data_store):
        self.data = data_store
        self.meta = MockMeta()
    
    def load(self):
        pass
    
    def wait_until_exists(self):
        pass
    
    def put_item(self, Item):
        key = f"{Item['user_id']}#{Item['sort_key']}"
        self.data[key] = Item
    
    def get_item(self, Key):
        key = f"{Key['user_id']}#{Key['sort_key']}"
        if key in self.data:
            return {'Item': self.data[key]}
        return {}
    
    def query(self, **kwargs):
        # Simple mock query implementation
        items = []
        user_id = kwargs['ExpressionAttributeValues'][':user_id']
        prefix = kwargs['ExpressionAttributeValues'].get(':message_prefix', '')
        
        for key, item in self.data.items():
            if (item.get('user_id') == user_id and 
                item.get('sort_key', '').startswith(prefix.replace('#', ''))):
                items.append(item)
        
        # Sort by sort_key if needed
        items.sort(key=lambda x: x.get('sort_key', ''), 
                  reverse=kwargs.get('ScanIndexForward', True) == False)
        
        limit = kwargs.get('Limit', 100)
        return {'Items': items[:limit]}
    
    def delete_item(self, Key):
        key = f"{Key['user_id']}#{Key['sort_key']}"
        self.data.pop(key, None)
    
    def batch_writer(self):
        return MockBatchWriter(self)


class MockMeta:
    """Mock table meta for client access"""
    
    def __init__(self):
        self.client = MockClient()


class MockClient:
    """Mock DynamoDB client"""
    
    def describe_table(self, TableName):
        return {
            'Table': {
                'ItemCount': len([k for k in self.data.keys()]),
                'TableSizeBytes': 1024,
                'TableName': TableName
            }
        }
    
    def update_time_to_live(self, **kwargs):
        pass


class MockBatchWriter:
    """Mock batch writer for DynamoDB"""
    
    def __init__(self, table):
        self.table = table
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def delete_item(self, Key):
        self.table.delete_item(Key)


# Global DynamoDB memory manager instance
dynamodb_memory_manager = DynamoDBMemoryManager()

# For backward compatibility, also export as memory_manager
memory_manager = dynamodb_memory_manager