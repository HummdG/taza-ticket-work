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
from langchain_core.messages import HumanMessage
from ..models.schemas import FlightBookingState
from ..agents.flight_booking_agent import flight_booking_agent
from ..agents.general_conversation_agent import handle_general_conversation
from ..services.conversation_router import should_handle_as_flight_booking, should_collect_flight_info, analyze_flight_request_completeness
from ..services.memory_service import memory_manager
from ..services.flight_info_collector import flight_collector
from dotenv import load_dotenv
from ..services.public_s3_handler import public_tazaticket_s3
from .aws_services import aws_translation_service, aws_polly_service, translate_to_language, generate_polly_speech
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

# Language mapping for TTS voices
LANGUAGE_VOICE_MAPPING = {
    'en': 'alloy',      # English - default voice
    'ur': 'nova',       # Urdu - closest supported voice
    'ar': 'shimmer',    # Arabic
    'hi': 'echo',       # Hindi - closest supported voice
    'es': 'fable',      # Spanish
    'fr': 'onyx',       # French
    'de': 'alloy',      # German
    'it': 'nova',       # Italian
    'pt': 'shimmer',    # Portuguese
    'ru': 'echo',       # Russian
    'ja': 'fable',      # Japanese
    'ko': 'onyx',       # Korean
    'zh': 'alloy',      # Chinese
    'default': 'alloy'  # Fallback
}


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
    Detect the language of the given text
    
    Args:
        text: Text to analyze
        
    Returns:
        str: ISO language code (e.g., 'en', 'ur', 'ar')
    """
    try:
        # Clean text for better detection
        cleaned_text = text.strip().lower()
        
        # Skip very short texts
        if len(cleaned_text) < 10:
            return 'en'  # Default to English for short texts
            
        detected_lang = detect(cleaned_text)
        print(f"🌐 Detected language: {detected_lang} for text: '{text[:50]}...'")
        return detected_lang
        
    except Exception as e:
        print(f"⚠️ Language detection failed: {e}")
        return 'en'  # Default to English


def generate_voice_response(text: str, language: str = 'en', user_id: str = "unknown") -> Optional[str]:
    """
    Generate voice response using natural speech formatting + AWS services
    
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
        
        # Step 2: Translate to target language if not English
        final_text = natural_speech_text
        if language != 'en' and aws_translation_service.is_configured():
            print(f"🌐 Translating natural speech to {language}...")
            final_text = translate_to_language(natural_speech_text, language)
            print(f"✅ Translated text: '{final_text[:100]}...'")
        elif language != 'en':
            print(f"⚠️ AWS Translate not available, using original text")
        
        # Step 3: Generate speech using AWS Polly
        if aws_polly_service.is_configured():
            print(f"🎤 Generating speech with AWS Polly for language: {language}")
            voice_file_path = generate_polly_speech(final_text, language, user_id)
            
            if voice_file_path:
                print(f"✅ AWS Polly voice response generated: {voice_file_path}")
                return voice_file_path
            else:
                print(f"❌ AWS Polly speech generation failed")
        else:
            print(f"⚠️ AWS Polly not available")
        
        # Fallback: Use OpenAI TTS if AWS Polly fails
        return generate_voice_response_openai_fallback(final_text, language, user_id)
        
    except Exception as e:
        print(f"❌ Voice generation error: {e}")
        return generate_voice_response_openai_fallback(text, language, user_id)


def generate_voice_response_openai_fallback(text: str, language: str = 'en', user_id: str = "unknown") -> Optional[str]:
    """
    Fallback voice response using OpenAI TTS (original implementation)
    """
    
    # Import OpenAI for fallback
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except ImportError:
        print("❌ OpenAI library not available for fallback TTS")
        return None
    
    if not client:
        print("❌ OpenAI client not available for fallback TTS")
        return None
    
    try:
        print(f"🔄 Using OpenAI TTS as fallback for language: {language}")
        
        # Clean text for TTS
        cleaned_text = clean_text_for_tts(text)
        
        # Voice selection for OpenAI (simplified)
        voice_mapping = {
            'en': 'alloy',
            'ur': 'nova',
            'ar': 'shimmer',
            'hi': 'echo',
            'es': 'fable',
            'fr': 'onyx',
            'de': 'alloy',
            'it': 'nova',
            'pt': 'shimmer',
            'ru': 'echo',
            'ja': 'fable',
            'ko': 'onyx',
            'zh': 'alloy',
            'default': 'alloy'
        }
        
        voice = voice_mapping.get(language, voice_mapping['default'])
        
        # Generate speech using OpenAI TTS
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=cleaned_text,
            response_format="mp3",
            speed=0.85  # Slower speech
        )
        
        # Save to temporary file
        temp_filename = f"openai_voice_{user_id}_{hash(cleaned_text) % 10000}.mp3"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        
        # Write audio data to file
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
        
        print(f"✅ OpenAI TTS fallback generated: {temp_path}")
        return temp_path
        
    except Exception as e:
        print(f"❌ OpenAI TTS fallback failed: {e}")
        return None


def clean_text_for_tts(text: str) -> str:
    """
    Clean text for better TTS output (works for both AWS Polly and OpenAI)
    
    Args:
        text: Original text with emojis and formatting
        
    Returns:
        str: Cleaned text suitable for TTS
    """
    import re
    
    # Remove or replace emojis with text descriptions
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
        '🎉': '',
    }
    
    cleaned_text = text
    for emoji, replacement in emoji_replacements.items():
        cleaned_text = cleaned_text.replace(emoji, replacement)
    
    # Remove remaining emojis (any Unicode emoji characters)
    emoji_pattern = re.compile("["
                             u"\U0001F600-\U0001F64F"  # emoticons
                             u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                             u"\U0001F680-\U0001F6FF"  # transport & map
                             u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                             u"\U00002702-\U000027B0"
                             u"\U000024C2-\U0001F251"
                             "]+", flags=re.UNICODE)
    cleaned_text = emoji_pattern.sub('', cleaned_text)
    
    # Clean up extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # Limit length for TTS
    if len(cleaned_text) > 4000:
        cleaned_text = cleaned_text[:4000] + "..."
    
    return cleaned_text

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
    Enhanced process_user_message with AWS services integration
    
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
            
            # Detect language from transcribed text
            detected_language = detect_language(transcribed_text)
            
            # Process transcribed text normally
            user_message = transcribed_text
            print(f"🌐 Voice message language detected as: {detected_language}")
        
        # Handle other media types (images, documents, etc.)
        elif media_url and media_content_type:
            media_type = media_content_type.split('/')[0]
            if media_type == 'image':
                response = "🖼️ I can see you sent an image! While I can't analyze images yet, feel free to describe what you'd like help with regarding your travel plans."
            elif media_type == 'video':
                response = "🎬 I received your video! While I can't analyze videos yet, please let me know how I can help with your flight booking needs."
            elif media_type == 'application':  # Documents, PDFs, etc.
                response = "📄 I received your document! While I can't read documents yet, feel free to tell me what information you need help with for your travel."
            else:
                response = "📎 I received your file! While I can't process files yet, please let me know how I can assist you with flight booking."
            
            from .memory_service import memory_manager
            memory_manager.add_conversation(user_id, f"[{media_type.title()}]", response, "media")
            return response, None
        
        # ... [Keep all the existing message routing logic] ...
        # Handle regular text messages (existing logic)
        from .memory_service import memory_manager
        from .conversation_router import should_handle_as_flight_booking, should_collect_flight_info, analyze_flight_request_completeness
        from ..agents.general_conversation_agent import handle_general_conversation
        
        # Get conversation context from memory
        conversation_context = memory_manager.get_conversation_context(user_id)
        
        # Check if user is currently collecting flight information
        is_collecting = memory_manager.is_collecting_flight_info(user_id)
        
        if is_collecting:
            print(f"🔄 Continuing flight info collection for user: {user_id}")
            response = handle_flight_info_collection(user_message, user_id, conversation_context)
            message_type = "flight_collection"
        
        # Check if this is a complete flight booking request
        elif should_handle_as_flight_booking(user_message, conversation_context):
            print(f"🛫 Routing to flight booking agent for user: {user_id}")
            response = process_flight_request(user_message, user_id, conversation_context)
            message_type = "flight"
        
        # Check if this is a partial flight request that needs more info
        elif should_collect_flight_info(user_message, conversation_context):
            print(f"📝 Starting flight info collection for user: {user_id}")
            response = start_flight_info_collection(user_message, user_id, conversation_context)
            message_type = "flight_collection"
        
        # Default to general conversation
        else:
            print(f"💬 Routing to general conversation agent for user: {user_id}")
            response = handle_general_conversation(user_message, user_id, conversation_context)
            message_type = "general"
        
        # Generate voice response if original was a voice message
        voice_file_url = None
        if is_voice_message and response:
            print(f"🎤 Generating voice response in language: {detected_language}")
            
            # Use AWS services for voice generation
            voice_file_path = generate_voice_response(response, detected_language, user_id)
            
            if voice_file_path:
                # Upload to accessible URL
                voice_file_url = upload_voice_file_to_accessible_url(voice_file_path, user_id)
                
                # Clean up local temp file after upload
                try:
                    os.unlink(voice_file_path)
                    print(f"🧹 Cleaned up temporary file: {voice_file_path}")
                except Exception as cleanup_error:
                    print(f"⚠️ Could not clean up temp file: {cleanup_error}")
        
        # Save the conversation to memory
        message_identifier = f"🎤 [Voice]: {user_message}" if is_voice_message else user_message
        memory_manager.add_conversation(user_id, message_identifier, response, message_type)
        
        return response, voice_file_url
            
    except Exception as e:
        print(f"❌ Error processing message: {e}")
        error_response = "😅 I'm having trouble processing your request right now. Please try again later!"
        
        # Still save error conversations to memory
        try:
            from .memory_service import memory_manager
            memory_manager.add_conversation(user_id, user_message, error_response, "error")
        except:
            pass
            
        return error_response, None


def start_flight_info_collection(user_message: str, user_id: str, conversation_context: str) -> str:
    """
    Start collecting flight information from a partial request
    
    Args:
        user_message: The user's message with partial flight info
        user_id: Unique identifier for the user
        conversation_context: Previous conversation context
        
    Returns:
        str: Question to collect missing information
    """
    
    try:
        # Extract available flight information
        has_intent, is_complete, extracted_info = analyze_flight_request_completeness(user_message, conversation_context)
        
        if not has_intent:
            # Shouldn't happen, but fallback to general conversation
            return handle_general_conversation(user_message, user_id, conversation_context)
        
        if is_complete:
            # Complete information, route to flight search
            return process_flight_request(user_message, user_id, conversation_context)
        
        # Save the collection state
        collection_state = {
            "collecting": True,
            "collected_info": extracted_info,
            "started_with": user_message
        }
        memory_manager.set_flight_collection_state(user_id, collection_state)
        
        # Generate question for missing information
        missing_fields = flight_collector.identify_missing_info(extracted_info)
        question = flight_collector.generate_question_for_missing_info(missing_fields, extracted_info)
        
        return question
        
    except Exception as e:
        print(f"❌ Error starting flight info collection: {e}")
        return "✈️ I'd be happy to help you find a flight! 😊 Could you tell me where you'd like to fly from, where you want to go, and when you'd like to travel?"


def handle_flight_info_collection(user_message: str, user_id: str, conversation_context: str) -> str:
    """
    Continue collecting flight information from user responses
    
    Args:
        user_message: The user's response with additional flight info
        user_id: Unique identifier for the user
        conversation_context: Previous conversation context
        
    Returns:
        str: Next question or flight search results
    """
    
    try:
        # Get current collection state
        collection_state = memory_manager.get_flight_collection_state(user_id)
        current_info = collection_state.get("collected_info", {})
        
        # Extract new information from user message
        new_info = flight_collector.extract_flight_info(user_message, conversation_context)
        
        # Merge with existing information
        merged_info = flight_collector.merge_flight_info(current_info, new_info)
        
        # Check if we now have complete information
        if flight_collector.is_flight_info_complete(merged_info):
            # Complete! Clear collection state and search for flights
            memory_manager.clear_flight_collection_state(user_id)
            
            # Create a summary and proceed to flight search
            summary = flight_collector.format_collected_info_summary(merged_info)
            
            # Construct a complete flight request message
            from_city = merged_info.get("from_city", "")
            to_city = merged_info.get("to_city", "")
            departure_date = merged_info.get("departure_date", "")
            
            complete_request = f"Find flights from {from_city} to {to_city} on {departure_date}"
            
            # Process as complete flight request
            flight_response = process_flight_request(complete_request, user_id, conversation_context)
            
            return f"{summary}\n\n{flight_response}"
        
        else:
            # Still missing information, update state and ask for more
            collection_state["collected_info"] = merged_info
            memory_manager.set_flight_collection_state(user_id, collection_state)
            
            # Generate next question
            missing_fields = flight_collector.identify_missing_info(merged_info)
            question = flight_collector.generate_question_for_missing_info(missing_fields, merged_info)
            
            return question
        
    except Exception as e:
        print(f"❌ Error handling flight info collection: {e}")
        # Clear collection state on error
        memory_manager.clear_flight_collection_state(user_id)
        return "😅 I had trouble processing that information. Let's start over - ✈️ could you tell me where you'd like to fly from, where you want to go, and when?"


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
        <Body>{message}</Body>
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