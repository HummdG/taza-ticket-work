"""Test AWS credentials"""

import boto3
import os
from dotenv import load_dotenv
from botocore.exceptions import ClientError, NoCredentialsError

load_dotenv()

def test_aws_credentials():
    print("🔐 Testing AWS Credentials...")
    print("=" * 50)
    
    # Check environment variables
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region = os.getenv('AWS_REGION', 'eu-north-1')
    bucket = os.getenv('AWS_S3_BUCKET', 'tazaticket')
    
    if not access_key:
        print("❌ AWS_ACCESS_KEY_ID not found in .env")
        return False
    
    if not secret_key:
        print("❌ AWS_SECRET_ACCESS_KEY not found in .env")
        return False
    
    print(f"✅ Access Key: {access_key}")
    print(f"✅ Secret Key: {secret_key[:4]}...{secret_key[-4:]}")
    print(f"✅ Region: {region}")
    print(f"✅ Bucket: {bucket}")
    
    # Test S3 client
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        
        print("\n🧪 Testing S3 connection...")
        
        # Test 1: List buckets (basic permission test)
        try:
            response = s3_client.list_buckets()
            print("✅ Can list buckets - credentials are valid!")
            
            # Check if our bucket exists
            bucket_names = [b['Name'] for b in response['Buckets']]
            if bucket in bucket_names:
                print(f"✅ Bucket '{bucket}' found!")
            else:
                print(f"⚠️ Bucket '{bucket}' not found in: {bucket_names}")
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            print(f"❌ Cannot list buckets: {error_code}")
            if error_code == 'SignatureDoesNotMatch':
                print("🚨 SIGNATURE MISMATCH - Wrong secret key!")
                return False
            elif error_code == 'InvalidAccessKeyId':
                print("🚨 INVALID ACCESS KEY - Wrong access key!")
                return False
        
        # Test 2: Try to access our specific bucket
        try:
            s3_client.head_bucket(Bucket=bucket)
            print(f"✅ Can access bucket '{bucket}'!")
            
            # Test 3: Try to list objects in bucket
            response = s3_client.list_objects_v2(Bucket=bucket, MaxKeys=5)
            object_count = response.get('KeyCount', 0)
            print(f"✅ Bucket contains {object_count} objects")
            
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            print(f"❌ Cannot access bucket '{bucket}': {error_code}")
            if error_code == 'NoSuchBucket':
                print(f"🚨 Bucket '{bucket}' doesn't exist!")
            elif error_code == 'SignatureDoesNotMatch':
                print("🚨 SIGNATURE MISMATCH - Check your secret key!")
            return False
            
    except NoCredentialsError:
        print("❌ AWS credentials not found")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_aws_credentials()
    
    if success:
        print("\n🎉 AWS credentials are working perfectly!")
    else:
        print("\n🔧 Fix the credential issues above and try again")
        print("\n💡 Most likely solutions:")
        print("1. Create new access keys in AWS Console")
        print("2. Update your .env file with correct keys")
        print("3. Restart your application")