import boto3
import os
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_KEY"),
    aws_secret_access_key=os.getenv("SECRET_AWS_KEY"),
    region_name='us-east-1'
)

bucket_name = "minecraft-videos-dance"
local_file = "video.mp4"
s3_key = "videos/video.mp4"

s3.upload_file(local_file, bucket_name, s3_key)
print("Upload complete!")