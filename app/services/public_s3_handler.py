"""
Fixed Public S3 Handler for TazaTicket - No ACL needed for public buckets
Save as: app/services/public_s3_handler.py (replace existing)
"""

import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from datetime import datetime, timedelta
from typing import Optional
import hashlib


class PublicTazaTicketS3Handler:
    """Public S3 handler that returns direct Object URLs (no ACL needed)"""
    
    def __init__(self):
        self.bucket_name = "tazaticket"
        self.region = "eu-north-1"
        self.base_url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com"
        self.s3_client = None
        
        if self._has_credentials():
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                    region_name=self.region
                )
                print(f"✅ Public TazaTicket S3 client initialized")
            except Exception as e:
                print(f"❌ Failed to initialize S3 client: {e}")
                self.s3_client = None
    
    def _has_credentials(self) -> bool:
        return all([
            os.getenv('AWS_ACCESS_KEY_ID'),
            os.getenv('AWS_SECRET_ACCESS_KEY')
        ])
    
    def upload_voice_file(self, local_file_path: str, user_id: str) -> Optional[str]:
        """Upload voice file and return direct Object URL (no ACL needed)"""
        
        if not self.is_configured():
            print("❌ Public S3 not configured")
            return None
            
        if not os.path.exists(local_file_path):
            print(f"❌ Local file not found: {local_file_path}")
            return None
            
        try:
            # Generate unique filename with safe characters for URLs
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_hash = self._generate_file_hash(local_file_path)[:8]
            file_extension = os.path.splitext(local_file_path)[1] or '.mp3'
            
            # Clean user_id for safe filename (replace special characters)
            safe_user_id = user_id.replace(":", "_").replace("+", "").replace(" ", "_")
            filename = f"voice/{safe_user_id}/{timestamp}_{file_hash}{file_extension}"
            
            print(f"🌐 Uploading to public TazaTicket S3: {filename}")
            
            # Upload file WITHOUT ACL (bucket is already public)
            self.s3_client.upload_file(
                local_file_path,
                self.bucket_name,
                filename,
                ExtraArgs={
                    'ContentType': 'audio/mpeg',
                    'CacheControl': 'max-age=3600',
                    # NO ACL parameter - bucket is already public
                    'Metadata': {
                        'user-id': safe_user_id,
                        'created-at': datetime.now().isoformat(),
                        'service': 'tazaticket-whatsapp-bot',
                        'type': 'voice-response'
                    }
                }
            )
            
            # Generate direct Object URL (no expiration needed since it's public)
            # URL encode the filename for proper handling
            import urllib.parse
            encoded_filename = urllib.parse.quote(filename, safe='/')
            object_url = f"{self.base_url}/{encoded_filename}"
            
            print(f"✅ Public Object URL created: {object_url}")
            
            # Set tags for cleanup (optional)
            self._set_cleanup_tags(filename)
            
            return object_url
            
        except NoCredentialsError:
            print("❌ AWS credentials not found")
            return None
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']
            print(f"❌ S3 error [{error_code}]: {error_msg}")
            
            # Specific handling for ACL errors
            if "AccessControlListNotSupported" in error_msg:
                print("🔧 Bucket doesn't support ACLs - this is normal for public buckets")
                print("   Files should still be accessible via bucket policy")
            
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
        """Test public connection"""
        if not self.is_configured():
            return {"success": False, "error": "Not configured"}
            
        try:
            # Test bucket access
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            
            # Test upload and direct URL access
            test_key = "voice/test/public_test.txt"
            test_content = f"Public TazaTicket test: {datetime.now()}"
            
            # Upload test file WITHOUT ACL
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=test_key,
                Body=test_content,
                ContentType='text/plain'
                # NO ACL parameter
            )
            
            # Generate direct Object URL
            import urllib.parse
            encoded_key = urllib.parse.quote(test_key, safe='/')
            object_url = f"{self.base_url}/{encoded_key}"
            
            # Test the direct URL works
            import requests
            response = requests.get(object_url, timeout=10)
            response.raise_for_status()
            
            # Cleanup
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=test_key)
            
            return {
                "success": True,
                "message": "Public TazaTicket S3 working perfectly!",
                "bucket": self.bucket_name,
                "region": self.region,
                "base_url": self.base_url,
                "security": "Public bucket with direct Object URLs (no ACL needed)",
                "test_url": object_url
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def is_configured(self) -> bool:
        """Check if public S3 is configured"""
        return all([
            self._has_credentials(),
            self.s3_client is not None
        ])


# Global public S3 handler instance
public_tazaticket_s3 = PublicTazaTicketS3Handler()