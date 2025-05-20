import boto3
from dotenv import load_dotenv
import os
import mimetypes
from urllib.parse import urlparse

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")
BUCKET_NAME = "narratron"

s3_client = boto3.client(
    "s3",
    region_name='us-east-1',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def get_files(bucket_name, directory):
    try:
        if not directory.endswith("/"):
            directory += "/"
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=directory)
        files = []
        if 'Contents' in response:
            for file in response['Contents']:
                files.append(file['Key'])
        return files
    except Exception as e:
        print(f"Error: {e}")
        return []

def upload_audio_file(file_obj, file_name, bucket):
    content_type, _ = mimetypes.guess_type(file_name)
    if content_type is None:
        content_type = "application/octet-stream"
    s3_client.upload_fileobj(
        Fileobj=file_obj,
        Bucket=bucket,
        Key=file_name,
        ExtraArgs={'ContentType': content_type}
    )
    print(f"✅ Uploaded {file_name} to {bucket}")

def generate_presigned_url(object_key):
    try:
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': BUCKET_NAME,
                'Key': object_key
            },
            ExpiresIn=7 * 3600 * 24
        )
        return presigned_url
    except Exception as e:
        print(f"Error generating pre-signed URL: {e}")
        return None