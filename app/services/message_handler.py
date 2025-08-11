"""
Complete message handling service with secure S3 voice storage
Save as: app/services/message_handler.py
"""

import os
import requests
import tempfile
import hashlib
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from datetime import datetime, timedelta
from typing import Optional, Tuple
from langdetect import detect, DetectorFactory
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from ..models.schemas import FlightBookingState
from ..agents.flight_booking_agent import flight_booking_agent
from ..agents.general_conversation_agent import handle_general_conversation
from ..services.conversation_router import should_handle_as_flight_booking, should_collect_flight_info, analyze_flight_request_completeness
from ..services.memory_service import memory_manager
from ..services.flight_info_collector import flight_collector
from dotenv import load_dotenv
from ..services.public_s3_handler import public_tazaticket_s3
# Removed AWS Translate/Polly TTS usage; Chat Completions will handle language & audio
from .speech_formatter import format_flight_for_speech

DetectorFactory.seed = 0

load_dotenv()
# Import OpenAI for Whisper and TTS APIs
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except ImportError:
    print("⚠️ OpenAI library not installed. Voice features will not work.")
    client = None

# Set seed for consistent language detection
DetectorFactory.seed = 0

# We will always use a female voice via Chat Completions (e.g., "verse")


class SecureTazaTicketS3Handler:
    """Secure S3 handler for TazaTicket voice files using presigned URLs"""
    
    def __init__(self):
        self.bucket_name = "tazaticket"
        self.region = "eu-north-1"
        self.s3_client = None
        
        if self._has_credentials():
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                    region_name=self.region
                )
                print(f"✅ Secure TazaTicket S3 client initialized")
            except Exception as e:
                print(f"❌ Failed to initialize S3 client: {e}")
                self.s3_client = None
    
    def _has_credentials(self) -> bool:
        return all([
            os.getenv('AWS_ACCESS_KEY_ID'),
            os.getenv('AWS_SECRET_ACCESS_KEY')
        ])
    
    def upload_voice_file(self, local_file_path: str, user_id: str) -> Optional[str]:
        """Upload voice file and return secure presigned URL"""
        
        if not self.is_configured():
            print("❌ Secure S3 not configured")
            return None
            
        if not os.path.exists(local_file_path):
            print(f"❌ Local file not found: {local_file_path}")
            return None
            
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_hash = self._generate_file_hash(local_file_path)[:8]
            file_extension = os.path.splitext(local_file_path)[1] or '.mp3'
            filename = f"voice/{user_id}/{timestamp}_{file_hash}{file_extension}"
            
            print(f"🔒 Uploading to secure TazaTicket S3: {filename}")
            
            # Upload file (stays private)
            self.s3_client.upload_file(
                local_file_path,
                self.bucket_name,
                filename,
                ExtraArgs={
                    'ContentType': 'audio/mpeg',
                    'CacheControl': 'max-age=3600',
                    'Metadata': {
                        'user-id': user_id,
                        'created-at': datetime.now().isoformat(),
                        'service': 'tazaticket-whatsapp-bot',
                        'type': 'voice-response'
                    }
                }
            )
            
            # Generate presigned URL (expires in 2 hours)
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': filename},
                ExpiresIn=7200  # 2 hours (7200 seconds)
            )
            
            print(f"✅ Secure presigned URL created (expires in 2h): {presigned_url[:50]}...")
            
            # Set tags for cleanup
            self._set_cleanup_tags(filename)
            
            return presigned_url
            
        except NoCredentialsError:
            print("❌ AWS credentials not found")
            return None
        except ClientError as e:
            error_code = e.response['Error']['Code']
            print(f"❌ S3 error [{error_code}]: {e.response['Error']['Message']}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
    
    def _generate_file_hash(self, file_path: str) -> str:
        """Generate hash for unique file naming"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return hashlib.md5(str(datetime.now()).encode()).hexdigest()
    
    def _set_cleanup_tags(self, filename: str):
        """Set tags for automatic cleanup"""
        try:
            self.s3_client.put_object_tagging(
                Bucket=self.bucket_name,
                Key=filename,
                Tagging={
                    'TagSet': [
                        {'Key': 'Service', 'Value': 'TazaTicket'},
                        {'Key': 'Type', 'Value': 'VoiceMessage'},
                        {'Key': 'AutoDelete', 'Value': 'true'},
                        {'Key': 'ExpiryDate', 'Value': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}
                    ]
                }
            )
        except Exception as e:
            print(f"⚠️ Could not set cleanup tags: {e}")
    
    def test_connection(self) -> dict:
        """Test secure connection"""
        if not self.is_configured():
            return {"success": False, "error": "Not configured"}
            
        try:
            # Test bucket access
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            
            # Test upload and presigned URL generation
            test_key = "voice/test/secure_test.txt"
            test_content = f"Secure TazaTicket test: {datetime.now()}"
            
            # Upload test file
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=test_key,
                Body=test_content,
                ContentType='text/plain'
            )
            
            # Generate presigned URL
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': test_key},
                ExpiresIn=300  # 5 minutes for test
            )
            
            # Test the presigned URL works
            response = requests.get(presigned_url, timeout=10)
            response.raise_for_status()
            
            # Cleanup
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=test_key)
            
            return {
                "success": True,
                "message": "Secure TazaTicket S3 working perfectly!",
                "bucket": self.bucket_name,
                "region": self.region,
                "security": "Private bucket with presigned URLs"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def is_configured(self) -> bool:
        """Check if secure S3 is configured"""
        return all([
            self._has_credentials(),
            self.s3_client is not None
        ])


# Global secure S3 handler instance
secure_tazaticket_s3 = SecureTazaTicketS3Handler()


def detect_language(text: str) -> str:
    """
    Robust language detection prioritizing script and reliable detection.
    - If Arabic script present → trust langdetect result ('ar' or 'ur').
    - If Devanagari script present → 'hi'.
    - Else → use langdetect on raw text (no keyword heuristics to avoid false positives).
    """
    try:
        import re
        cleaned_text = (text or "").strip()

        if len(cleaned_text) < 10:
            return 'en'

        # Script checks
        has_arabic = bool(re.search(r"[\u0600-\u06FF]", cleaned_text))  # Arabic script (covers Urdu/Arabic)
        has_devanagari = bool(re.search(r"[\u0900-\u097F]", cleaned_text))  # Hindi script

        if has_devanagari:
            print("🇮🇳 Detected Devanagari script → hi")
            return 'hi'

        # Use langdetect for Arabic-script and general cases
        detected_lang = detect(cleaned_text)

        if has_arabic:
            print(f"📝 Arabic script present; trusting detector: {detected_lang}")
            # langdetect may return 'ur' or 'ar'; both acceptable
            if detected_lang in ['ar', 'ur']:
                print(f"🌐 Final detected language: {detected_lang} for text: '{text[:50]}...'")
                return detected_lang
            # If Arabic script but detector returns other, default to 'ur'
            print("🔄 Overriding to 'ur' due to Arabic script")
            return 'ur'

        # For Latin/other scripts, rely solely on langdetect
        print(f"🌐 Final detected language: {detected_lang} for text: '{text[:50]}...'")
        return detected_lang

    except Exception as e:
        print(f"⚠️ Language detection failed: {e}")
        return 'en'


def generate_voice_response(text: str, language: str = 'en', user_id: str = "unknown") -> Optional[str]:
    """
    Generate voice response using natural speech formatting + OpenAI TTS (primary)
    
    Args:
        text: Text to convert to speech (will be formatted and translated)
        language: Language code for translation and voice selection
        user_id: User ID for file naming
        
    Returns:
        str: Path to generated local voice file, or None if failed
    """
    
    try:
        print(f"🎤 Generating voice response for language: {language}")
        print(f"📝 Original text: '{text[:100]}...'")
        
        # Step 1: Convert structured response to natural speech
        natural_speech_text = format_flight_for_speech(text, language)
        print(f"🗣️ Natural speech: '{natural_speech_text[:100]}...'")
        
        # Step 2: The text should already be in the correct language from ChatCompletion API
        # Just ensure it's optimized for speech
        final_text = natural_speech_text

        voice_file_path = generate_voice_response_via_chat_completion(final_text, language, user_id)
        
        if voice_file_path:
            print(f"✅ Chat Completions audio generated: {voice_file_path}")
            return voice_file_path
        else:
            print(f"❌ Chat Completions audio generation failed")
            return None
        
    except Exception as e:
        print(f"❌ Voice generation error: {e}")
        return None


def generate_voice_response_via_chat_completion(text: str, language: str = 'en', user_id: str = "unknown") -> Optional[str]:
    """
    Generate voice using OpenAI Chat Completions with audio output (female voice).
    The model is instructed to speak in the user's language and produce natural audio.
    Ensures booking references are included in voice responses.
    """
    try:
        import base64
        from openai import OpenAI
        audio_model = os.getenv("OPENAI_AUDIO_MODEL", "gpt-4o-audio-preview")
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        print(f"❌ Failed to init OpenAI client: {e}")
        return None

    try:
        # Check if there's a booking reference to include in voice
        from .memory_service import memory_manager
        try:
            flight_ctx = memory_manager.get_flight_context(user_id)
            queued_ref = None
            if isinstance(flight_ctx, dict):
                queued_ref = flight_ctx.get("broadcast_booking_reference_once")
            
            # If there's a booking reference queued, include it in the voice message
            if queued_ref:
                if "booking reference" in text.lower() or "reference" in text.lower():
                    # Add the actual reference number to the text for voice
                    text = f"{text}\n\nThe reference number is: {queued_ref}\n\nPlease call: +92 3 1 2 8 5 6 7 4 4 2"
                    # Clear the broadcast flag since we're including it in voice
                    memory_manager.add_flight_context(user_id, {"broadcast_booking_reference_once": None})
        except Exception as ref_err:
            print(f"⚠️ Could not check booking reference: {ref_err}")

        cleaned_text = clean_text_for_enhanced_tts(text)
        print(f"🎤 Using Chat Completions audio for language: {language}")

        # Prefer a clearly female-presenting voice; allow env override
        target_voice = os.getenv("OPENAI_VOICE", "shimmer")

        # Build Chat Completions request with audio output
        # The model will translate/rephrase to the detected language if needed
        completion = openai_client.chat.completions.create(
            model=audio_model,
            modalities=["text", "audio"],
            audio={"voice": target_voice, "format": "mp3"},
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"""You are TazaTicket's human-sounding travel assistant voice. 
                        Always speak in the user's language ({language}) with a warm, friendly FEMALE voice (soft, natural, expressive).
                        Sound like a real person: vary pace, add light intonation, and use short connectors appropriate to the language. 
                        Preserve all facts; don't invent details. Read times/dates naturally (THIS IS VERY IMPORTANT). Avoid listy cadence.
                        Don't be too formal with your response you are supposed to be a travel buddy and not a travel agent.
                        But you always should respond only in the {language} language. THIS IS VERY IMPORTANT WE CANNOT AFFORD FOR YOU TO SPEAK IN ANY OTHER LANGUAGE OTHER THAN {language}.
                        PLEASE DON'T BE TOO FORMAL.
                        For example in urdu/hindi you should still use stuff like "point" instead of "sharia" don't use formal urdu/hindi words.
                        If the text describes a round‑trip (mentions Outbound and Return or RETURN FLIGHT), you MUST include both legs explicitly. Do not omit the return leg.
                        If there's a booking reference number or phone number, read it clearly with pauses between digits.
                        """
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Please deliver this as a natural voice reply for WhatsApp in the user's language: \n\n" + cleaned_text
                    ),
                },
            ],
        )

        # Extract base64 audio from various possible response shapes for compatibility
        b64_audio = None
        try:
            b64_audio = completion.choices[0].message.audio.data  # new shape
        except Exception:
            pass
        if b64_audio is None:
            try:
                parts = getattr(completion.choices[0].message, "content", []) or []
                for p in parts:
                    if isinstance(p, dict):
                        audio_obj = p.get("audio") or p.get("output_audio")
                        if isinstance(audio_obj, dict) and audio_obj.get("data"):
                            b64_audio = audio_obj["data"]
                            break
            except Exception:
                pass

        if not b64_audio:
            print("❌ No audio returned from Chat Completions")
            return None

        audio_bytes = base64.b64decode(b64_audio)
        temp_filename = f"openai_chat_audio_{user_id}_{hash(cleaned_text) % 10000}.mp3"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        print(f"✅ Chat Completions audio generated: {temp_path}")
        return temp_path
    except Exception as e:
        print(f"❌ Chat Completions audio error: {e}")
        return None


def clean_text_for_enhanced_tts(text: str) -> str:
    """
    Enhanced text cleaning for better TTS output (optimized for natural speech)
    
    Args:
        text: Original text with emojis and formatting
        
    Returns:
        str: Cleaned text optimized for natural TTS
    """
    import re
    
    # Remove or replace emojis with text descriptions for natural flow
    emoji_replacements = {
        '✈️': 'flight',
        '🎯': 'destination', 
        '📅': 'date',
        '💰': 'price',
        '🛫': 'departure',
        '🛬': 'arrival', 
        '🏢': 'airline',
        '🔄': 'stops',
        '🧳': 'baggage',
        '👥': 'passengers',
        '🌍': 'worldwide',
        '🎤': '',  # Remove voice indicators
        '😊': '',
        '👋': '',
        '❓': '',
        '💡': '',
        '🔍': 'searching',
        '📊': '',
        '✅': '',
        '❌': '',
        '⚠️': 'warning',
        '🎉': 'great',
    }
    
    cleaned_text = text
    for emoji, replacement in emoji_replacements.items():
        cleaned_text = cleaned_text.replace(emoji, replacement)
    
    # Remove remaining emojis (any Unicode emoji characters)
    emoji_pattern = re.compile("["
                             u"\U0001F600-\U0001F64F"  # emoticons
                             u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                             u"\U0001F680-\U0001F6FF"  # transport & map
                             u"\U0001F1E0-\U0001F1FF"  # flags
                             u"\U00002702-\U000027B0"
                             u"\U000024C2-\U0001F251"
                             "]+", flags=re.UNICODE)
    cleaned_text = emoji_pattern.sub('', cleaned_text)
    
    # Replace problematic characters and abbreviations for natural speech
    replacements = {
        'USD': 'US Dollars',
        'EUR': 'Euros', 
        'GBP': 'British Pounds',
        'AED': 'UAE Dirhams',
        'PKR': 'Pakistani Rupees',
        'N/A': 'not available',
        '&': 'and',
        '@': 'at',
        '#': 'number',
        '%': 'percent',
        '+': 'plus',
        '=': 'equals',
        'vs': 'versus',
        'e.g.': 'for example',
        'etc.': 'and so on',
    }
    
    for old, new in replacements.items():
        cleaned_text = cleaned_text.replace(old, new)
    
    # Add natural pauses for better speech flow
    cleaned_text = re.sub(r'([.!?])\s*', r'\1 ', cleaned_text)
    
    # Clean up multiple spaces and normalize
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # Limit length for TTS (OpenAI has a 4096 character limit)
    if len(cleaned_text) > 3500:
        cleaned_text = cleaned_text[:3500] + "... Please let me know if you need more information."
    
    return cleaned_text


# Backward-compatible wrapper name
def generate_voice_response_openai_fallback(text: str, language: str = 'en', user_id: str = "unknown") -> Optional[str]:
    return generate_voice_response_via_chat_completion(text, language, user_id)

def upload_voice_file_to_accessible_url(file_path: str, user_id: str = "unknown") -> Optional[str]:
    """
    Upload voice file and return public Object URL
    
    Args:
        file_path: Local path to the voice file
        user_id: User ID for organizing files
        
    Returns:
        str: Direct Object URL for public access
    """
    
    # Priority 1: Try public TazaTicket S3 (returns direct Object URLs)
    if public_tazaticket_s3.is_configured():
        print("🌐 Uploading to public TazaTicket S3...")
        object_url = public_tazaticket_s3.upload_voice_file(file_path, user_id)
        if object_url:
            print(f"✅ Public Object URL ready: {object_url}")
            return object_url
        else:
            print("⚠️ Public S3 upload failed, falling back to local")
    else:
        print("⚠️ Public S3 not configured, using local file serving")
    
    # Priority 2: Fallback to local file serving
    try:
        filename = os.path.basename(file_path)
        
        # Add timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_part = os.path.splitext(filename)[0]
        extension = os.path.splitext(filename)[1]
        unique_filename = f"{name_part}_{timestamp}_{user_id}{extension}"
        
        # Copy file to static directory
        static_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'voice')
        os.makedirs(static_dir, exist_ok=True)
        
        static_path = os.path.join(static_dir, unique_filename)
        
        # Copy the file
        import shutil
        shutil.copy2(file_path, static_path)
        
        # Return URL
        base_url = os.getenv('BASE_URL', 'https://your-domain.com')
        file_url = f"{base_url}/static/voice/{unique_filename}"
        
        print(f"📁 Voice file accessible locally: {file_url}")
        return file_url
        
    except Exception as e:
        print(f"❌ Failed to setup voice file serving: {e}")
        return None


# Also add this function to get S3 stats for the new public handler:
def get_public_s3_stats() -> dict:
    """Get public S3 statistics for monitoring"""
    if public_tazaticket_s3.is_configured():
        return public_tazaticket_s3.test_connection()
    else:
        return {"success": False, "error": "Public S3 not configured"}


def transcribe_voice_message(media_url: str, media_content_type: str) -> str:
    """
    Transcribe a voice message using OpenAI Whisper API
    
    Args:
        media_url: URL to download the voice message
        media_content_type: MIME type of the audio file
        
    Returns:
        str: Transcribed text from the voice message
    """
    
    if not client:
        return "🎤 I received your voice message, but voice transcription is not available right now. Could you please type your message instead?"
    
    try:
        print(f"🎤 Downloading voice message from: {media_url}")
        
        # Set up authentication for Twilio media URLs
        auth = None
        if "api.twilio.com" in media_url:
            twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
            
            if not twilio_account_sid or not twilio_auth_token:
                print("⚠️ Twilio credentials not found. Trying without authentication...")
            else:
                auth = (twilio_account_sid, twilio_auth_token)
                print(f"🔐 Using Twilio authentication for media download")
        
        # Download the audio file with authentication if needed
        response = requests.get(media_url, timeout=30, auth=auth)
        response.raise_for_status()
        
        print(f"✅ Media file downloaded successfully ({len(response.content)} bytes)")
        
        # Determine file extension based on content type
        extension = ".ogg"  # Default for WhatsApp voice messages
        if "mp4" in media_content_type:
            extension = ".mp4"
        elif "mpeg" in media_content_type:
            extension = ".mp3"
        elif "wav" in media_content_type:
            extension = ".wav"
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        
        print(f"🎤 Transcribing voice message...")
        
        # Transcribe using OpenAI Whisper
        with open(temp_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        if transcript and transcript.strip():
            print(f"✅ Voice message transcribed: {transcript}")
            return transcript.strip()
        else:
            return "🎤 I couldn't understand your voice message. Could you please try again or type your message?"
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading voice message: {e}")
        if "401" in str(e):
            return "🎤 I had trouble accessing your voice message. Please check the Twilio authentication settings and try again."
        return "🎤 I had trouble downloading your voice message. Could you please try sending it again?"
        
    except Exception as e:
        print(f"❌ Error transcribing voice message: {e}")
        return "🎤 I had trouble understanding your voice message. Could you please try again or type your message instead?"


def process_user_message(user_message: str, user_id: str = "unknown", media_url: Optional[str] = None, media_content_type: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Enhanced process_user_message with natural conversation flow like ChatGPT/Claude
    
    Args:
        user_message: The user's message (could be empty for voice-only messages)
        user_id: Unique identifier for the user
        media_url: URL to media content (for voice messages, images, etc.)
        media_content_type: MIME type of the media
        
    Returns:
        Tuple[str, Optional[str]]: (text_response, voice_file_url)
    """
    
    try:
        is_voice_message = False
        detected_language = 'en'  # Default language
        original_user_message = user_message
        
        # Handle voice messages first
        if media_url and media_content_type and media_content_type.startswith('audio'):
            print(f"🎤 Processing voice message from user: {user_id}")
            is_voice_message = True
            
            # Transcribe the voice message
            transcribed_text = transcribe_voice_message(media_url, media_content_type)
            
            # Check if transcription failed
            if transcribed_text.startswith("🎤 I"):
                # Return error message directly (no voice response for errors)
                from .memory_service import memory_manager
                memory_manager.add_conversation(user_id, "[Voice Message]", transcribed_text, "voice_error")
                return transcribed_text, None
            
            # Use transcribed text as the actual message
            user_message = transcribed_text
            print(f"🎤 Transcribed: {user_message}")
        
        # ALWAYS detect language for every message (voice AND text)
        if user_message and user_message.strip():
            detected_language = detect_language(user_message)
            print(f"🌐 Detected language: {detected_language} for message: '{user_message[:50]}...'")
        
        # Handle other media types (images, documents, etc.)
        if media_url and media_content_type and not media_content_type.startswith('audio'):
            media_type = media_content_type.split('/')[0]
            response = _generate_multilingual_response(
                f"I received your {media_type}! While I can't analyze {media_type}s yet, feel free to tell me how I can help with your flight booking needs.",
                detected_language, user_id
            )
            
            from .memory_service import memory_manager
            memory_manager.add_conversation(user_id, f"[{media_type.title()}]", response, "media")
            return response, None
        
        # Get conversation context for natural flow
        from .memory_service import memory_manager
        conversation_context = memory_manager.get_conversation_context(user_id, max_recent=12)
        
        # Process the message using ChatCompletion API for intelligent routing and language handling
        response = _process_message_with_chatcompletion(user_message, user_id, conversation_context, detected_language)
        
        # Ensure we never send an empty response
        if not isinstance(response, str) or not response.strip():
            response = _generate_multilingual_response(
                "To get started, please tell me your departure city, destination, and date.",
                detected_language, user_id
            )
        
        # Generate voice response if original was a voice message
        voice_file_url = None
        if is_voice_message and response:
            print(f"🎤 Generating voice response in language: {detected_language}")
            try:
                voice_file_path = generate_voice_response(response, detected_language, user_id)
                if voice_file_path:
                    voice_file_url = upload_voice_file_to_accessible_url(voice_file_path, user_id)
                    if voice_file_url:
                        # Clean up local temp file after upload
                        try:
                            os.unlink(voice_file_path)
                            print(f"🧹 Cleaned up temporary file: {voice_file_path}")
                        except Exception as cleanup_error:
                            print(f"⚠️ Could not clean up temp file: {cleanup_error}")
                    else:
                        print("⚠️ Voice upload returned no URL; falling back to text-only reply")
                else:
                    print("⚠️ No voice file generated; responding with text only")
            except Exception as gen_err:
                print(f"❌ Voice generation error: {gen_err}")
        
        # Save the conversation to memory
        message_identifier = f"🎤 [Voice]: {original_user_message}" if is_voice_message else original_user_message
        memory_manager.add_conversation(user_id, message_identifier, response, "conversation")
        
        return response, voice_file_url
        
    except Exception as e:
        print(f"❌ Error processing message: {e}")
        error_response = _generate_multilingual_response(
            "I'm having trouble processing your request right now. Please try again later!",
            detected_language, user_id
        )
        
        # Still save error conversations to memory
        try:
            from .memory_service import memory_manager
            memory_manager.add_conversation(user_id, user_message, error_response, "error")
        except:
            pass
        
        return error_response, None


def _process_message_with_chatcompletion(user_message: str, user_id: str, conversation_context: str, detected_language: str) -> str:
    """
    Process user message using ChatCompletion API for intelligent routing and natural language handling
    This creates a ChatGPT-like conversation flow with smart flight search triggering
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        from .memory_service import memory_manager
        
        # Initialize the smart routing LLM
        routing_llm = ChatOpenAI(
            model=os.getenv("OPENAI_ROUTING_MODEL", "gpt-4o-mini"), 
            temperature=0.1
        )
        
        # Check for special commands first
        user_lower = user_message.lower().strip()
        
        # Reset/Clear commands
        if user_lower in ["new", "reset", "start over", "restart", "clear", "cancel"]:
            memory_manager.clear_flight_collection_state(user_id)
            memory_manager.clear_flight_context(user_id)
            return _generate_multilingual_response(
                "Alright, a fresh start! Where would you like to fly from?",
                detected_language, user_id
            )
        
        # Booking intent commands - support multiple languages
        booking_keywords = [
            # English
            "book", "reserve", "confirm booking", "proceed with booking",
            # Urdu/Hindi
            "بک", "بکنگ", "book", "میں بک کرنا چاہتا", "بک کر دو",
            # Arabic
            "احجز", "حجز", "أريد الحجز"
        ]
        
        has_booking_intent = any(keyword in user_lower for keyword in booking_keywords)
        
        if has_booking_intent:
            # Try to get booking reference
            flight_ctx = memory_manager.get_flight_context(user_id)
            quote_ref = None
            if isinstance(flight_ctx, dict):
                quote_ref = flight_ctx.get("last_quote_reference")
            
            if quote_ref:
                # Mark for broadcasting (both text and voice will show reference)
                memory_manager.add_flight_context(user_id, {"broadcast_booking_reference_once": quote_ref})
                return _generate_multilingual_response(
                    "Please quote the following booking reference number to the number below:",
                    detected_language, user_id
                )
            else:
                return _generate_multilingual_response(
                    "I'd be happy to help you book! First, let me search for available flights. Can you confirm your travel details?",
                    detected_language, user_id
                )
        
        # INTELLIGENT FLIGHT SEARCH DETECTION - Check if user has provided enough info to search
        flight_info = _extract_flight_info_from_conversation(user_message, conversation_context, detected_language)
        
        # Check if user is starting a completely new flight request
        if _is_new_flight_request(user_message, conversation_context, detected_language):
            print("🔄 New flight request detected - clearing previous context")
            memory_manager.clear_flight_context(user_id)
            memory_manager.clear_flight_collection_state(user_id)
        
        if _has_enough_info_to_search(flight_info):
            print("🎯 Detected complete flight info - triggering search")
            try:
                # Directly search for flights
                flight_response = _handle_flight_search(user_message, user_id, conversation_context, detected_language)
                if flight_response and any(word in flight_response.lower() for word in ["flight", "price", "eur", "usd"]):
                    return flight_response
            except Exception as e:
                print(f"⚠️ Flight search failed: {e}")
        
        # Use ChatCompletion for intelligent conversation handling
        system_prompt = f"""
        You are TazaTicket's intelligent travel assistant. Help users find flights naturally.
        
        CRITICAL INSTRUCTIONS:
        1. ALWAYS respond in the user's detected language: {detected_language}
        2. If user mentions completely different cities than before, treat it as a NEW request
        3. Be conversational and don't repeat questions
        4. ALWAYS ask about trip type (round-trip or one-way) before searching
        5. Don't mix up different flight requests
        
        CONVERSATION ANALYSIS:
        Previous conversation: {conversation_context}
        Current message: {user_message}
        
        REQUIRED INFO BEFORE SEARCHING:
        - Origin city/airport ✈️
        - Destination city/airport 🎯
        - Departure date 📅
        - Trip type (round-trip or one-way) 🔄 ← MUST ASK THIS
        - If round-trip: return date or duration
        
        IMPORTANT RULES:
        - If user says "New York to Milan" after talking about other cities, this is a NEW request
        - Don't carry over irrelevant details from previous conversations
        - ALWAYS confirm trip type before searching: "Is this a round-trip or one-way flight?"
        - If round-trip, ask for return date or duration
        - Only search when you have ALL required information
        
        RESPOND NATURALLY in {detected_language}.
        """
        
        # Create the conversation
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User said: {user_message}")
        ]
        
        # Get intelligent response
        result = routing_llm.invoke(messages)
        response_text = result.content if isinstance(result.content, str) else str(result.content)
        
        return response_text
        
    except Exception as e:
        print(f"❌ Error in ChatCompletion processing: {e}")
        return _generate_multilingual_response(
            "I'd be happy to help you find a flight! Could you tell me where you'd like to fly from, where you want to go, and when you'd like to travel?",
            detected_language, user_id
        )


def _extract_flight_info_from_conversation(user_message: str, conversation_context: str, detected_language: str) -> dict:
    """
    Extract flight information prioritizing the current message over conversation context
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        import json
        
        extractor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # First try to extract from current message only
        current_message_prompt = f"""
        Extract flight booking information from this CURRENT MESSAGE ONLY. Return a JSON object.
        
        Current message: "{user_message}"
        Language: {detected_language}
        
        Extract only what is mentioned in this CURRENT message:
        - origin_city: departure city/airport (string)
        - destination_city: arrival city/airport (string)  
        - departure_date: departure date in YYYY-MM-DD format (string)
        - return_date: return date if round-trip in YYYY-MM-DD format (string or null)
        - passengers: number of passengers (integer, default 1)
        - trip_type: "round-trip" or "one-way" (string) - look for keywords like "round trip", "return", "one way", "round-trip", etc.
        - duration_days: if user mentions duration like "20 days" (integer or null)
        
        TRIP TYPE DETECTION:
        - "round trip", "return trip", "round-trip", "واپسی", "راؤنٹریپ" = "round-trip"
        - "one way", "one-way", "single", "ایک طرفہ" = "one-way"
        - Duration mentions like "20 days", "2 weeks" usually indicate round-trip
        
        IMPORTANT: 
        - Only extract information from the current message
        - If user mentions completely different cities, use those (ignore previous context)
        - If no specific info in current message, return null for that field
        - Be smart about detecting trip type from context clues
        
        Return ONLY valid JSON.
        """
        
        messages = [
            SystemMessage(content=current_message_prompt),
            HumanMessage(content=f"Current message: {user_message}")
        ]
        
        result = extractor_llm.invoke(messages)
        response_text = result.content.strip()
        
        # Clean up response to extract JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        try:
            current_info = json.loads(response_text)
            print(f"🎯 Current message info: {current_info}")
            
            # If current message has complete new flight info, use it entirely
            if current_info.get("origin_city") and current_info.get("destination_city"):
                print("✈️ New complete flight request detected - using current message only")
                return current_info
            
            # Otherwise, merge with conversation context carefully
            if conversation_context:
                context_prompt = f"""
                Merge flight information from conversation context with current message.
                
                Conversation context: {conversation_context}
                Current message: {user_message}
                Current message info: {current_info}
                
                Rules:
                1. If current message has new cities, completely replace old cities
                2. If current message has new date, replace old date
                3. Only fill missing fields from context if current message doesn't contradict
                4. Always prioritize current message over context
                
                Return complete flight info as JSON.
                """
                
                context_messages = [
                    SystemMessage(content=context_prompt),
                    HumanMessage(content=f"Merge: {current_info}")
                ]
                
                context_result = extractor_llm.invoke(context_messages)
                context_response = context_result.content.strip()
                
                if "```json" in context_response:
                    context_response = context_response.split("```json")[1].split("```")[0]
                elif "```" in context_response:
                    context_response = context_response.split("```")[1].split("```")[0]
                
                try:
                    merged_info = json.loads(context_response)
                    print(f"🔄 Merged info: {merged_info}")
                    return merged_info
                except json.JSONDecodeError:
                    print(f"⚠️ Could not parse merged info, using current only")
                    return current_info
            
            return current_info
            
        except json.JSONDecodeError:
            print(f"⚠️ Could not parse flight info JSON: {response_text}")
            return {}
        
    except Exception as e:
        print(f"⚠️ Error extracting flight info: {e}")
        return {}


def _is_new_flight_request(user_message: str, conversation_context: str, detected_language: str) -> bool:
    """
    Detect if user is starting a completely new flight request (different cities)
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        
        detector_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        detection_prompt = f"""
        Analyze if the user is starting a COMPLETELY NEW flight request with different cities.
        
        Conversation context: {conversation_context}
        Current message: {user_message}
        
        Return "YES" if:
        - User mentions completely different origin/destination cities than before
        - User is clearly starting a new flight search
        - User mentions cities that contradict previous conversation
        
        Return "NO" if:
        - User is continuing the same conversation
        - User is adding details to existing request
        - User is asking about dates/passengers for same route
        
        Examples of NEW requests:
        - Previous: "London to Paris" → Current: "New York to Milan" = YES
        - Previous: "Lahore to Athens" → Current: "New York to Milan" = YES
        
        Examples of CONTINUING:
        - Previous: "London to Paris" → Current: "on November 5th" = NO
        - Previous: "where to go" → Current: "New York to Milan" = NO
        
        Response: YES or NO only.
        """
        
        messages = [
            SystemMessage(content=detection_prompt),
            HumanMessage(content=user_message)
        ]
        
        result = detector_llm.invoke(messages)
        response = result.content.strip().upper()
        
        return response == "YES"
        
    except Exception as e:
        print(f"⚠️ Error detecting new flight request: {e}")
        return False


def _has_enough_info_to_search(flight_info: dict) -> bool:
    """
    Check if we have enough information to search for flights
    ALWAYS requires trip type to be specified before searching
    """
    # Basic required fields
    required_fields = ["origin_city", "destination_city", "departure_date"]
    
    # Check if all required fields are present and not null/empty
    for field in required_fields:
        if not flight_info.get(field):
            return False
    
    # CRITICAL: Trip type must always be specified
    if not flight_info.get("trip_type"):
        return False
    
    # If it's round-trip, we also need return date or duration
    if flight_info.get("trip_type") == "round-trip":
        return flight_info.get("return_date") or flight_info.get("duration_days")
    
    # For one-way, we have everything we need
    return True


def _handle_flight_search(user_message: str, user_id: str, conversation_context: str, detected_language: str) -> str:
    """
    Handle the actual flight search when we have enough information
    Creates complete request with context and ensures booking reference generation
    """
    try:
        # Extract flight info to create complete request
        flight_info = _extract_flight_info_from_conversation(user_message, conversation_context, detected_language)
        
        # Build a complete flight request message
        origin = flight_info.get("origin_city", "")
        destination = flight_info.get("destination_city", "")
        departure_date = flight_info.get("departure_date", "")
        return_date = flight_info.get("return_date")
        passengers = flight_info.get("passengers", 1)
        duration_days = flight_info.get("duration_days")
        
        # If round-trip with duration but no return date, calculate it
        if flight_info.get("trip_type") == "round-trip" and duration_days and not return_date:
            try:
                from datetime import datetime, timedelta
                dep_date = datetime.strptime(departure_date, "%Y-%m-%d")
                ret_date = dep_date + timedelta(days=duration_days)
                return_date = ret_date.strftime("%Y-%m-%d")
            except Exception:
                pass
        
        # Create comprehensive flight request
        if return_date:
            complete_request = f"Round trip flight from {origin} to {destination}, departing {departure_date}, returning {return_date}, for {passengers} passenger(s)"
        else:
            complete_request = f"One way flight from {origin} to {destination} on {departure_date} for {passengers} passenger(s)"
        
        print(f"🔍 Complete flight request: {complete_request}")
        
        # Use the existing flight booking agent with complete request
        response = process_flight_request(complete_request, user_id, conversation_context)
        
        # Ensure response is in the correct language
        if response and detected_language != 'en':
            response = _translate_response_to_language(response, detected_language, user_id)
        
        # Check if booking reference was generated
        from .memory_service import memory_manager
        flight_ctx = memory_manager.get_flight_context(user_id)
        if isinstance(flight_ctx, dict) and flight_ctx.get("last_quote_reference"):
            print(f"✅ Booking reference generated: {flight_ctx.get('last_quote_reference')}")
        
        return response
        
    except Exception as e:
        print(f"❌ Error in flight search: {e}")
        return _generate_multilingual_response(
            "I'm having trouble searching for flights right now. Please try again later.",
            detected_language, user_id
        )


def _generate_multilingual_response(english_text: str, target_language: str, user_id: str) -> str:
    """
    Generate a response in the user's language using ChatCompletion
    """
    if target_language == 'en':
        return english_text
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        
        translator_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        
        translation_prompt = f"""
        Translate the following English text to {target_language}. 
        Keep the tone friendly and natural for a travel assistant.
        Preserve any emojis and maintain the conversational style.
        
        SPECIAL TERMS:
        - "round-trip" = واپسی کا ٹکٹ / راؤنڈ ٹرپ (Urdu), رحلة ذهاب وإياب (Arabic)
        - "one-way" = ایک طرفہ (Urdu), رحلة ذهاب فقط (Arabic)
        
        English text: {english_text}
        """
        
        messages = [
            SystemMessage(content=translation_prompt),
            HumanMessage(content=english_text)
        ]
        
        result = translator_llm.invoke(messages)
        translated_text = result.content if isinstance(result.content, str) else str(result.content)
        
        return translated_text.strip()
        
    except Exception as e:
        print(f"⚠️ Translation failed: {e}")
        return english_text  # Fallback to English


def _translate_response_to_language(response_text: str, target_language: str, user_id: str) -> str:
    """
    Translate a flight search response to the target language
    """
    if target_language == 'en':
        return response_text
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        
        translator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        translation_prompt = f"""
        Translate this flight search response to {target_language}.
        Preserve all flight details, dates, times, prices, and booking references exactly.
        Keep the tone friendly and natural for a travel assistant.
        Maintain any emojis and formatting.
        
        Original response: {response_text}
        """
        
        messages = [
            SystemMessage(content=translation_prompt),
            HumanMessage(content=response_text)
        ]
        
        result = translator_llm.invoke(messages)
        translated_text = result.content if isinstance(result.content, str) else str(result.content)
        
        return translated_text.strip()
        
    except Exception as e:
        print(f"⚠️ Response translation failed: {e}")
        return response_text  # Fallback to original


def _generate_trip_type_question(detected_language: str, user_id: str) -> str:
    """
    Generate a trip type question in the user's language
    """
    trip_type_questions = {
        'en': "Is this a round-trip or one-way flight?",
        'ur': "کیا یہ واپسی کا ٹکٹ ہے یا ایک طرفہ؟",
        'hi': "यह राउंड-ट्रिप है या वन-वे?",
        'ar': "هل هذه رحلة ذهاب وإياب أم ذهاب فقط؟",
        'fr': "Est-ce un aller-retour ou un aller simple?",
        'de': "Ist das Hin- und Rückflug oder nur Hinflug?",
        'es': "¿Es un vuelo de ida y vuelta o solo de ida?",
        'it': "È un volo di andata e ritorno o solo andata?",
        'pt': "É uma viagem de ida e volta ou só ida?",
        'tr': "Bu gidiş-dönüş mü yoksa tek yön mü?",
        'ru': "Это билет туда-обратно или в одну сторону?",
        'zh': "这是往返机票还是单程机票？",
        'ja': "往復航空券ですか、それとも片道ですか？",
        'ko': "왕복 항공권인가요, 편도인가요?"
    }
    
    # Return direct translation if available
    if detected_language in trip_type_questions:
        return trip_type_questions[detected_language]
    
    # Fallback to multilingual generation
    return _generate_multilingual_response(
        "Is this a round-trip or one-way flight?", 
        detected_language, 
        user_id
    )


# Legacy functions removed - now using ChatCompletion API for natural conversation flow


def process_flight_request(user_message: str, user_id: str = "unknown", conversation_context: str = "") -> str:
    """
    Process a flight booking request using the LangGraph agent with memory
    
    Args:
        user_message: The user's flight request message
        user_id: Unique identifier for the user
        conversation_context: Previous conversation history
        
    Returns:
        str: Formatted response with flight information or error message
    """
    
    try:
        # Initialize state with memory context
        initial_state: FlightBookingState = {
            "messages": [HumanMessage(content=user_message)],
            "user_message": user_message,
            "user_id": user_id,
            "from_city": None,
            "to_city": None,
            "departure_date": None,
            "return_date": None,
            "passengers": 1,
            "passenger_age": 25,
            "raw_api_response": None,
            "cheapest_flight": None,
            "response_text": "",
            "conversation_context": conversation_context
        }
        
        # Run the agent
        final_state = flight_booking_agent.invoke(initial_state)
        
        # Save flight context to memory for future reference
        if final_state.get("cheapest_flight"):
            flight_context = {
                "last_search": {
                    "from_city": final_state.get("from_city"),
                    "to_city": final_state.get("to_city"),
                    "departure_date": final_state.get("departure_date"),
                    "return_date": final_state.get("return_date"),
                    "passengers": final_state.get("passengers"),
                }
            }
            memory_manager.add_flight_context(user_id, flight_context)
        
        # Return the response
        return final_state.get("response_text", "😔 I'm sorry, I couldn't process your flight request.")
        
    except Exception as e:
        print(f"❌ Error processing flight request: {e}")
        return "I'm having trouble processing your flight request right now. Please try again later."


def format_whatsapp_response(bot_response: str) -> str:
    """
    Format bot response for WhatsApp delivery
    
    Args:
        bot_response: The bot's response text
        
    Returns:
        str: WhatsApp-formatted response
    """
    # For now, just return the response as-is
    # In the future, this could handle WhatsApp-specific formatting
    return bot_response


def create_twiml_response(message: str, media_url: Optional[str] = None) -> str:
    """
    Create TwiML response for Twilio WhatsApp integration
    
    Args:
        message: The message to send
        media_url: Optional media URL for voice/image messages
        
    Returns:
        str: TwiML formatted response
    """
    
    if media_url:
        # Send voice message with media
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>
        
        <Media>{media_url}</Media>
    </Message>
</Response>"""
    else:
        # Send regular text message
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{message}</Message>
</Response>"""


def get_s3_stats() -> dict:
    """Get S3 statistics for monitoring"""
    if secure_tazaticket_s3.is_configured():
        return secure_tazaticket_s3.test_connection()
    else:
        return {"success": False, "error": "Secure S3 not configured"}