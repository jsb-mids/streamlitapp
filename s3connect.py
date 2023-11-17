import os
from io import BytesIO
import boto3
import pickle

from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())


AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")


def getObject(obj):
    s3 = boto3.client(
        service_name='s3',
        region_name=AWS_DEFAULT_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    response = s3.get_object(Bucket='shop-buddy-chatbot-data', Key=obj)
    if 'pickle' in obj:
        data = response['Body'].read()
        return pickle.load(BytesIO(data))
    else:
        return response
        
# getObject('pickle/categories.pkl')