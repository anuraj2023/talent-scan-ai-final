from dotenv import load_dotenv
import os

load_dotenv()

OPEN_AI_KEY = os.getenv('OPEN_AI_API_KEY')
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
AWS_REGION = os.getenv('AWS_REGION')


def get_env_vars():
    return {
        'OPEN_AI_KEY': OPEN_AI_KEY,
        'AWS_ACCESS_KEY': AWS_ACCESS_KEY,
        'AWS_SECRET_KEY': AWS_SECRET_KEY,
        'S3_BUCKET_NAME': S3_BUCKET_NAME,
        'AWS_REGION' : AWS_REGION
    }