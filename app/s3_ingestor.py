import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from config import get_env_vars
from logging_config import logger
from urllib.parse import quote

env_vars = get_env_vars()
AWS_ACCESS_KEY = env_vars['AWS_ACCESS_KEY']
AWS_SECRET_KEY = env_vars['AWS_SECRET_KEY']
AWS_REGION = env_vars['AWS_REGION'] 

s3_client = boto3.client('s3', 
                         aws_access_key_id=AWS_ACCESS_KEY, 
                         aws_secret_access_key=AWS_SECRET_KEY,
                         region_name=AWS_REGION)

def upload_to_s3(file, bucket, s3_file):
    try:
        s3_client.upload_fileobj(file, bucket, s3_file)
        encoded_s3_file = quote(s3_file)
        s3_url = f"https://{bucket}.s3.{AWS_REGION}.amazonaws.com/{encoded_s3_file}"
        
        logger.info(f"Successfully uploaded {s3_file} to bucket {bucket}")
        logger.info(f"S3 URL: {s3_url}")
        
        return True, s3_url
    except NoCredentialsError:
        logger.error("AWS credentials not available")
        return False, None
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"ClientError uploading to S3: {error_code} - {error_message}")
        return False, None
    except Exception as e:
        logger.error(f"Unexpected error uploading to S3: {str(e)}")
        return False, None