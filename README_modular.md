# WhatsApp Flight Booking Bot - Modular Architecture

This project has been refactored from a monolithic structure into a clean, modular architecture that separates concerns and makes the codebase more maintainable and extensible.

## 🏗️ Architecture Overview

```
TazaTicket/
├── app/                          # Main application package
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # FastAPI application entry point
│   ├── api/                     # External API integrations
│   │   ├── __init__.py
│   │   └── travelport.py        # Travelport API authentication & headers
│   ├── payloads/                # API payload construction
│   │   ├── __init__.py
│   │   └── flight_search.py     # Flight search payload builders
│   ├── agents/                  # LangGraph agents and workflows
│   │   ├── __init__.py
│   │   └── flight_booking_agent.py # Main flight booking workflow
│   ├── services/                # Business logic services
│   │   ├── __init__.py
│   │   └── message_handler.py   # Message processing & orchestration
│   └── models/                  # Data models and schemas
│       ├── __init__.py
│       └── schemas.py           # Pydantic models & TypedDict states
├── main.py                      # Application entry point
├── whatsapp_bot_fastapi.py      # DEPRECATED - old monolithic version
└── requirements.txt             # Dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Required environment variables in `.env` file:
  ```
  TRAVELPORT_APPLICATION_KEY=your_key
  TRAVELPORT_APPLICATION_SECRET=your_secret
  TRAVELPORT_USERNAME=your_username
  TRAVELPORT_PASSWORD=your_password
  TRAVELPORT_ACCESS_GROUP=your_access_group
  WEBHOOK_VERIFY_TOKEN=your_verify_token
  OPENAI_API_KEY=your_openai_key
  ```

### Installation

```bash
pip install -r requirements.txt
```

### Running the Application

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## 📁 Module Descriptions

### 🔐 `app/api/travelport.py`

Handles Travelport API authentication and header management.

**Key Functions:**

- `fetch_password_token()` - OAuth token generation
- `get_api_headers()` - Complete headers for API requests

### 📦 `app/payloads/flight_search.py`

Constructs API payloads for different types of flight searches.

**Key Functions:**

- `build_flight_search_payload()` - Standard one-way/round-trip searches
- `build_multi_city_payload()` - Multi-segment itineraries

### ✈️ `app/payloads/airline_codes.py`

Comprehensive reference for airline IATA codes and carrier management.

**Key Features:**

- `AIRLINE_CODES` - Complete dictionary of 60+ major airlines worldwide
- `DEFAULT_PREFERRED_CARRIERS` - Optimized list for broad search coverage
- `get_airline_name()` - Convert IATA codes to full airline names
- `get_carriers_by_region()` - Filter airlines by geographic region

### 🤖 `app/agents/flight_booking_agent.py`

Contains the LangGraph workflow for processing flight booking requests.

**Key Components:**

- `parse_travel_request()` - LLM-powered message parsing
- `search_flights()` - Travelport API integration
- `find_cheapest_flight()` - Response analysis and formatting
- `flight_booking_agent` - Compiled LangGraph workflow

### 💬 `app/agents/general_conversation_agent.py`

Handles general conversations and non-flight related queries using natural LLM chat.

**Key Components:**

- `handle_general_conversation()` - Main conversation router
- `handle_llm_conversation()` - Natural LLM chat for any topic
- `handle_greeting()` - Friendly greeting responses
- `handle_general_question()` - Bot capability explanations

### 🧠 `app/services/conversation_router.py`

Intelligent message classification to route conversations appropriately.

**Key Functions:**

- `classify_message_intent()` - Pattern-based intent classification
- `should_handle_as_flight_booking()` - Flight request detection

### 🛠️ `app/services/message_handler.py`

Orchestrates message processing and routes between conversation types.

**Key Functions:**

- `process_user_message()` - Main message routing pipeline
- `process_flight_request()` - Flight booking processing
- `format_whatsapp_response()` - WhatsApp message formatting
- `create_twiml_response()` - Twilio TwiML generation

### 📊 `app/models/schemas.py`

Defines all data models, schemas, and state structures.

**Key Models:**

- `FlightBookingState` - LangGraph state definition
- `TestMessage` - API testing model
- `FlightDetails` - Response formatting model

### 🌐 `app/main.py`

Main FastAPI application with all route handlers.

**API Endpoints:**

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /webhook` - Webhook verification
- `POST /webhook` - Message processing
- `POST /test` - Manual testing

## 🔧 Adding New Features

### Adding New Payload Types

1. Create a new function in `app/payloads/flight_search.py`
2. Import and use in the agent workflow

### Adding New API Integrations

1. Create a new file in `app/api/` (e.g., `hotel_api.py`)
2. Follow the same pattern as `travelport.py`

### Extending the Workflow

1. Add new nodes to the LangGraph workflow in `app/agents/flight_booking_agent.py`
2. Update decision functions as needed

### Adding New Message Processors

1. Create new functions in `app/services/message_handler.py`
2. Add routing logic in the main webhook handler

## 🗣️ Conversation Flow

The bot now supports two types of interactions:

### 💬 **General Conversation**

Users can chat naturally about any topic:

- **Greetings**: "Hi", "Hello", "Good morning"
- **General questions**: "How's your day?", "Tell me a joke"
- **Bot capabilities**: "What do you do?", "How can you help?"

### ✈️ **Flight Booking**

Specific flight-related requests are routed to the specialized agent:

- **Flight searches**: "I want to fly to Paris", "Find flights to Tokyo"
- **Booking requests**: "Book a flight from NYC to London"
- **Travel planning**: "Show me flights from Dubai to Bangkok tomorrow"

### 🤖 **Smart Routing**

The system automatically detects intent and routes appropriately:

```
User Message → Intent Classification →
├── Greeting → "Hi there! I'm your flight assistant..."
├── General Chat → Natural LLM conversation
├── Bot Questions → Capability explanations
└── Flight Request → LangGraph flight booking workflow
```

## 🧪 Testing

Test different conversation types using the `/test` endpoint:

**General Chat:**

```bash
curl -X POST "http://localhost:8000/test" \
     -H "Content-Type: application/json" \
     -d '{"message": "How are you today?"}'
```

**Flight Booking:**

```bash
curl -X POST "http://localhost:8000/test" \
     -H "Content-Type: application/json" \
     -d '{"message": "I want to fly from New York to London tomorrow"}'
```

## 📈 Benefits of Modular Architecture

1. **Separation of Concerns** - Each module has a single responsibility
2. **Maintainability** - Easy to find and modify specific functionality
3. **Testability** - Individual modules can be tested in isolation
4. **Extensibility** - New features can be added without affecting existing code
5. **Reusability** - Modules can be reused across different parts of the application
6. **Team Collaboration** - Different team members can work on different modules

## 🔄 Migration from Monolithic Version

The original monolithic file (`whatsapp_bot_fastapi.py`) has been preserved for reference but marked as deprecated. All functionality has been migrated to the new modular structure without any loss of features.

To migrate custom modifications:

1. Identify which module your changes belong to
2. Apply changes to the appropriate module
3. Test the functionality using the new entry point

## 🤝 Contributing

When contributing to this project:

1. Follow the modular structure
2. Add new functionality to the appropriate module
3. Create new modules if needed for new feature categories
4. Update this README when adding new modules or major features
