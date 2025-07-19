# WhatsApp Flight Booking Bot (FastAPI + Docker)

A FastAPI-based WhatsApp chatbot that integrates with Travelport API to help users find and book flights.

## 🚀 Quick Start

### 1. Environment Setup

Create a `.env` file with your credentials:

```env
# Travelport API Configuration
TRAVELPORT_APPLICATION_KEY=your_app_key_here
TRAVELPORT_APPLICATION_SECRET=your_app_secret_here
TRAVELPORT_USERNAME=your_username_here
TRAVELPORT_PASSWORD=your_password_here
TRAVELPORT_ACCESS_GROUP=your_access_group_here

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-key-here

# Twilio Configuration (for voice message support)
TWILIO_ACCOUNT_SID=your_twilio_account_sid_here
TWILIO_AUTH_TOKEN=your_twilio_auth_token_here

# Webhook Configuration
WEBHOOK_VERIFY_TOKEN=your_secure_token_here
```

### 2. Run with Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t whatsapp-flight-bot .
docker run -p 8000:8000 --env-file .env whatsapp-flight-bot
```

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Test flight search
curl -X POST http://localhost:8000/test \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to fly from Lahore to Athens on March 15th"}'
```

### 4. Connect to WhatsApp

#### Option A: Using ngrok (for testing)

```bash
# Install ngrok and expose your Docker container
ngrok http 8000

# Copy the HTTPS URL (e.g., https://abc123.ngrok-free.app)
# Set this as your Twilio webhook: https://abc123.ngrok-free.app/webhook
```

#### Option B: Deploy to production

Deploy to any cloud platform:

- **Railway**: `railway deploy`
- **Render**: Connect your GitHub repo
- **DigitalOcean Apps**: Deploy from GitHub
- **AWS/GCP/Azure**: Use container services

### 5. Configure Twilio WhatsApp

1. Go to your Twilio Console → WhatsApp Sandbox
2. Set webhook URL to: `https://your-domain.com/webhook`
3. Set HTTP method to: `POST`
4. Save configuration

## 📱 Usage

Send messages to your WhatsApp bot like:

- "I want to fly from Lahore to Athens on March 15th"
- "Book a flight from LHE to ATH next month returning in a week"
- "Find flights from Karachi to London departing tomorrow"

### 🎤 Voice Message Support

The bot now supports voice messages! Simply:

1. Hold the microphone button in WhatsApp
2. Record your flight request (e.g., "I need a flight from New York to London next week")
3. Send the voice note
4. The bot will transcribe your message and respond normally

Voice messages work for all features including flight searches, general conversation, and flight information collection.

## 🛠 API Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `GET /webhook` - WhatsApp webhook verification
- `POST /webhook` - Handle WhatsApp messages (including voice messages)
- `POST /test` - Test flight search functionality
- `POST /test-voice` - Test voice message transcription functionality

## 🔧 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python whatsapp_bot_fastapi.py

# Or with uvicorn
uvicorn whatsapp_bot_fastapi:app --reload --host 0.0.0.0 --port 8000
```

## 🐳 Docker Commands

```bash
# Build image
docker build -t whatsapp-flight-bot .

# Run container
docker run -p 8000:8000 --env-file .env whatsapp-flight-bot

# View logs
docker logs <container-id>

# Stop container
docker stop <container-id>
```

## 🔍 Troubleshooting

### Common Issues

1. **Environment variables not loaded**

   - Ensure `.env` file exists and has correct format
   - Check Docker volume mounting

2. **Travelport API errors**

   - Verify your API credentials
   - Check network connectivity

3. **OpenAI API errors**

   - Verify your OpenAI API key
   - Check API usage limits

4. **Voice message not working**

   - Ensure `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` are set in `.env`
   - Check that OpenAI API key is valid
   - Verify Twilio webhook is configured to send media

5. **Webhook not receiving messages**

   - Ensure ngrok/deployment URL is HTTPS
   - Check Twilio webhook configuration
   - Verify webhook verification token

### Debug Mode

Add to your `.env` file:

```env
DEBUG=true
```

## 📊 Features

- ✅ Natural language parsing using GPT-4
- ✅ Flight search via Travelport API
- ✅ Cheapest flight selection
- ✅ WhatsApp integration (Twilio)
- ✅ Docker containerization
- ✅ FastAPI with automatic API docs
- ✅ Health monitoring endpoints
- ✅ Error handling and logging

## 🔗 API Documentation

When running, visit: `http://localhost:8000/docs` for interactive API documentation.
