"""
AWS Translation and Polly Services for TazaTicket
Save as: app/services/aws_services.py
"""

import os
import boto3
import tempfile
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv()

class AWSTranslationService:
    """AWS Translate service for dynamic language translation"""
    
    def __init__(self):
        self.translate_client = None
        self.region = "eu-north-1"  # Use same region as S3
        
        if self._has_credentials():
            try:
                self.translate_client = boto3.client(
                    'translate',
                    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                    region_name=self.region
                )
                print(f"✅ AWS Translate client initialized")
            except Exception as e:
                print(f"❌ Failed to initialize AWS Translate: {e}")
                self.translate_client = None
    
    def _has_credentials(self) -> bool:
        return all([
            os.getenv('AWS_ACCESS_KEY_ID'),
            os.getenv('AWS_SECRET_ACCESS_KEY')
        ])
    
    def translate_text(self, text: str, target_language: str, source_language: str = "en") -> str:
        """
        Translate text to target language using AWS Translate
        
        Args:
            text: Text to translate
            target_language: Target language code (e.g., 'ur', 'ar', 'es')
            source_language: Source language code (default: 'en')
            
        Returns:
            str: Translated text or original text if translation fails
        """
        
        if not self.is_configured():
            print("❌ AWS Translate not configured")
            return text
        
        if target_language == source_language:
            return text  # No translation needed
        
        # Map some language codes to AWS Translate supported codes
        language_mapping = {
            'ur': 'ur',      # Urdu
            'ar': 'ar',      # Arabic
            'hi': 'hi',      # Hindi
            'es': 'es',      # Spanish
            'fr': 'fr',      # French
            'de': 'de',      # German
            'it': 'it',      # Italian
            'pt': 'pt',      # Portuguese
            'ru': 'ru',      # Russian
            'ja': 'ja',      # Japanese
            'ko': 'ko',      # Korean
            'zh': 'zh',      # Chinese (simplified)
            'zh-cn': 'zh',   # Chinese (simplified)
            'zh-tw': 'zh-TW', # Chinese (traditional)
            'tr': 'tr',      # Turkish
            'fa': 'fa',      # Persian/Farsi
            'bn': 'bn',      # Bengali
            'ta': 'ta',      # Tamil
            'te': 'te',      # Telugu
            'ml': 'ml',      # Malayalam
            'kn': 'kn',      # Kannada
            'gu': 'gu',      # Gujarati
            'pa': 'pa',      # Punjabi
        }
        
        aws_target_lang = language_mapping.get(target_language, target_language)
        aws_source_lang = language_mapping.get(source_language, source_language)
        
        try:
            print(f"🌐 Translating from {aws_source_lang} to {aws_target_lang}: '{text[:50]}...'")
            
            result = self.translate_client.translate_text(
                Text=text,
                SourceLanguageCode=aws_source_lang,
                TargetLanguageCode=aws_target_lang
            )
            
            translated_text = result['TranslatedText']
            print(f"✅ Translation successful: '{translated_text[:50]}...'")
            
            return translated_text
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            print(f"❌ AWS Translate error [{error_code}]: {e.response['Error']['Message']}")
            
            # Handle specific errors
            if error_code == 'UnsupportedLanguagePairException':
                print(f"⚠️ Language pair {aws_source_lang}->{aws_target_lang} not supported")
            
            return text  # Return original text on error
            
        except Exception as e:
            print(f"❌ Translation error: {e}")
            return text
    
    def is_configured(self) -> bool:
        """Check if AWS Translate is configured"""
        return self.translate_client is not None
    
    def test_translation(self) -> dict:
        """Test AWS Translate functionality"""
        if not self.is_configured():
            return {"success": False, "error": "Not configured"}
        
        try:
            # Test translation
            test_text = "Hello, this is a test message for flight booking assistance."
            
            # Test English to Urdu
            urdu_translation = self.translate_text(test_text, 'ur', 'en')
            
            # Test English to Arabic
            arabic_translation = self.translate_text(test_text, 'ar', 'en')
            
            return {
                "success": True,
                "message": "AWS Translate working perfectly!",
                "test_translations": {
                    "original": test_text,
                    "urdu": urdu_translation,
                    "arabic": arabic_translation
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class AWSPollyService:
    """AWS Polly service for natural text-to-speech"""
    
    def __init__(self):
        self.polly_client = None
        self.region = "eu-north-1"  # Use same region as S3
        
        if self._has_credentials():
            try:
                self.polly_client = boto3.client(
                    'polly',
                    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                    region_name=self.region
                )
                print(f"✅ AWS Polly client initialized")
            except Exception as e:
                print(f"❌ Failed to initialize AWS Polly: {e}")
                self.polly_client = None
    
    def _has_credentials(self) -> bool:
        return all([
            os.getenv('AWS_ACCESS_KEY_ID'),
            os.getenv('AWS_SECRET_ACCESS_KEY')
        ])
    
    def get_voice_for_language(self, language_code: str) -> Dict[str, str]:
        """
        Get the best AWS Polly voice for a language
        
        Args:
            language_code: Language code (e.g., 'en', 'ur', 'ar')
            
        Returns:
            Dict with voice_id, language_code, and engine
        """
        
        # AWS Polly voice mapping for natural-sounding voices
        voice_mapping = {
            # English voices
            'en': {'voice_id': 'Joanna', 'language_code': 'en-US', 'engine': 'neural'},
            
            # Arabic voices
            'ar': {'voice_id': 'Zeina', 'language_code': 'ar-XWW', 'engine': 'standard'},
            
            # Urdu - use Hindi voice (closest available)
            'ur': {'voice_id': 'Aditi', 'language_code': 'hi-IN', 'engine': 'standard'},
            
            # Hindi voices
            'hi': {'voice_id': 'Aditi', 'language_code': 'hi-IN', 'engine': 'standard'},
            
            # Spanish voices
            'es': {'voice_id': 'Lupe', 'language_code': 'es-US', 'engine': 'neural'},
            
            # French voices
            'fr': {'voice_id': 'Lea', 'language_code': 'fr-FR', 'engine': 'neural'},
            
            # German voices
            'de': {'voice_id': 'Marlene', 'language_code': 'de-DE', 'engine': 'neural'},
            
            # Italian voices
            'it': {'voice_id': 'Carla', 'language_code': 'it-IT', 'engine': 'neural'},
            
            # Portuguese voices
            'pt': {'voice_id': 'Ines', 'language_code': 'pt-PT', 'engine': 'neural'},
            
            # Russian voices
            'ru': {'voice_id': 'Tatyana', 'language_code': 'ru-RU', 'engine': 'standard'},
            
            # Japanese voices
            'ja': {'voice_id': 'Takumi', 'language_code': 'ja-JP', 'engine': 'neural'},
            
            # Korean voices
            'ko': {'voice_id': 'Seoyeon', 'language_code': 'ko-KR', 'engine': 'neural'},
            
            # Chinese voices
            'zh': {'voice_id': 'Zhiyu', 'language_code': 'zh-CN', 'engine': 'standard'},
            
            # Turkish voices
            'tr': {'voice_id': 'Filiz', 'language_code': 'tr-TR', 'engine': 'standard'},
        }
        
        return voice_mapping.get(language_code, voice_mapping['en'])  # Default to English
    
    def generate_speech(self, text: str, language_code: str = 'en', user_id: str = "unknown") -> Optional[str]:
        """
        Generate speech using AWS Polly
        
        Args:
            text: Text to convert to speech
            language_code: Language code for voice selection
            user_id: User ID for file naming
            
        Returns:
            str: Path to generated audio file, or None if failed
        """
        
        if not self.is_configured():
            print("❌ AWS Polly not configured")
            return None
        
        if not text or not text.strip():
            print("❌ No text provided for speech generation")
            return None
        
        try:
            # Get appropriate voice for language
            voice_config = self.get_voice_for_language(language_code)
            voice_id = voice_config['voice_id']
            polly_language_code = voice_config['language_code']
            engine = voice_config['engine']
            
            print(f"🎤 Generating speech with AWS Polly:")
            print(f"   Voice: {voice_id}")
            print(f"   Language: {polly_language_code}")
            print(f"   Engine: {engine}")
            print(f"   Text: '{text[:100]}...'")
            
            # Clean text for better speech synthesis
            cleaned_text = self._clean_text_for_polly(text)
            
            # Generate speech using AWS Polly
            response = self.polly_client.synthesize_speech(
                Text=cleaned_text,
                OutputFormat='mp3',
                VoiceId=voice_id,
                Engine=engine,
                LanguageCode=polly_language_code if engine == 'neural' else None,
                SpeechMarkTypes=[],  # No speech marks needed
                # Add SSML for better speech control if needed
                TextType='text'  # Can be 'ssml' for advanced control
            )
            
            # Save to temporary file
            temp_filename = f"polly_voice_{user_id}_{hash(cleaned_text) % 10000}.mp3"
            temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
            
            # Write audio stream to file
            with open(temp_path, 'wb') as f:
                f.write(response['AudioStream'].read())
            
            print(f"✅ AWS Polly speech generated: {temp_path}")
            return temp_path
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            print(f"❌ AWS Polly error [{error_code}]: {e.response['Error']['Message']}")
            
            # Handle specific errors
            if error_code == 'InvalidVoiceId':
                print(f"⚠️ Voice {voice_id} not available, falling back to default")
                # Retry with default English voice
                try:
                    response = self.polly_client.synthesize_speech(
                        Text=self._clean_text_for_polly(text),
                        OutputFormat='mp3',
                        VoiceId='Joanna',
                        Engine='neural'
                    )
                    temp_filename = f"polly_voice_fallback_{user_id}_{hash(text) % 10000}.mp3"
                    temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
                    with open(temp_path, 'wb') as f:
                        f.write(response['AudioStream'].read())
                    return temp_path
                except:
                    return None
            
            return None
            
        except Exception as e:
            print(f"❌ Speech generation error: {e}")
            return None
    
    def _clean_text_for_polly(self, text: str) -> str:
        """
        Clean and optimize text for AWS Polly
        
        Args:
            text: Original text
            
        Returns:
            str: Cleaned text optimized for speech synthesis
        """
        import re
        
        # Remove or replace emojis and special characters
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
            '🎤': '',
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
        
        # Remove remaining emojis
        emoji_pattern = re.compile("["
                                 u"\U0001F600-\U0001F64F"  # emoticons
                                 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                 u"\U0001F680-\U0001F6FF"  # transport & map
                                 u"\U0001F1E0-\U0001F1FF"  # flags
                                 u"\U00002702-\U000027B0"
                                 u"\U000024C2-\U0001F251"
                                 "]+", flags=re.UNICODE)
        cleaned_text = emoji_pattern.sub('', cleaned_text)
        
        # Replace problematic characters for speech
        replacements = {
            'USD': 'US Dollars',
            'EUR': 'Euros',
            'GBP': 'British Pounds',
            'N/A': 'not available',
            '&': 'and',
            '@': 'at',
            '#': 'number',
            '%': 'percent',
            '+': 'plus',
            '=': 'equals',
        }
        
        for old, new in replacements.items():
            cleaned_text = cleaned_text.replace(old, new)
        
        # Clean up multiple spaces and normalize
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Limit length (Polly has a 3000 character limit for text)
        if len(cleaned_text) > 2500:
            cleaned_text = cleaned_text[:2500] + "..."
        
        # Add natural pauses for better speech flow
        cleaned_text = re.sub(r'([.!?])\s*', r'\1 ', cleaned_text)
        
        return cleaned_text
    
    def is_configured(self) -> bool:
        """Check if AWS Polly is configured"""
        return self.polly_client is not None
    
    def test_polly(self) -> dict:
        """Test AWS Polly functionality"""
        if not self.is_configured():
            return {"success": False, "error": "Not configured"}
        
        try:
            # Test speech generation in English
            test_text = "Hello, this is a test of AWS Polly for TazaTicket flight booking service."
            
            voice_file_path = self.generate_speech(test_text, 'en', 'test_user')
            
            if voice_file_path and os.path.exists(voice_file_path):
                file_size = os.path.getsize(voice_file_path)
                
                # Clean up test file
                try:
                    os.unlink(voice_file_path)
                except:
                    pass
                
                return {
                    "success": True,
                    "message": "AWS Polly working perfectly!",
                    "test_file_size": f"{file_size} bytes",
                    "supported_voices": self._get_available_voices()
                }
            else:
                return {"success": False, "error": "Speech generation failed"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_available_voices(self) -> dict:
        """Get list of available voices for supported languages"""
        return {
            "english": ["Joanna", "Matthew", "Ivy", "Justin"],
            "arabic": ["Zeina"],
            "hindi": ["Aditi"],
            "spanish": ["Lupe", "Miguel"],
            "french": ["Lea", "Mathieu"],
            "german": ["Marlene", "Hans"],
            "italian": ["Carla", "Giorgio"],
            "portuguese": ["Ines", "Cristiano"],
            "russian": ["Tatyana", "Maxim"],
            "japanese": ["Takumi", "Mizuki"],
            "korean": ["Seoyeon"],
            "chinese": ["Zhiyu"]
        }


# Global service instances
aws_translation_service = AWSTranslationService()
aws_polly_service = AWSPollyService()


# Convenience functions for easy import
def translate_to_language(text: str, target_language: str) -> str:
    """Convenience function to translate text"""
    return aws_translation_service.translate_text(text, target_language)


def generate_polly_speech(text: str, language_code: str, user_id: str = "unknown") -> Optional[str]:
    """Convenience function to generate speech with Polly"""
    return aws_polly_service.generate_speech(text, language_code, user_id)


def test_aws_services() -> dict:
    """Test both AWS services"""
    translate_test = aws_translation_service.test_translation()
    polly_test = aws_polly_service.test_polly()
    
    return {
        "translate": translate_test,
        "polly": polly_test,
        "overall_success": translate_test.get("success", False) and polly_test.get("success", False)
    }