import os
import pandas as pd
from datetime import datetime
from typing import List

from shuo_clip3 import get_response
from productDetails import get_product_details

from fastapi import FastAPI
from pydantic import BaseModel

from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache

from redis import asyncio as aioredis

redis_url = os.getenv("service_url", "localhost")
redis_port = os.getenv("redis_port", "6379")
# url = f"redis://{redis_url}:{redis_port}"
url = "redis://localhost:6379"


app = FastAPI()

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url(url)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

class Message(BaseModel):
    content: str

class ProductDetails(BaseModel):
    uniq_id: str
    title: str
    url: str
    price: float
    href: str

class Response(BaseModel):
    response: str
    product_details: List[ProductDetails] = []
    

@app.post("/chat")
@cache()
async def chat_with_bot(message: Message):
    user_message = message.content
    bot_response, uuids = get_response(user_message)
    if uuids is None:
        return Response(response=bot_response, product_details=[])
    else:
        product_details_list =   get_product_details(uuids)
        product_details_objects = []
        for product_details in product_details_list:
            product_details_objects.append(ProductDetails(**product_details))
        return Response(response=bot_response, product_details=product_details_objects)


# Health endpoint
@app.get("/health")
async def health():
	return {"time": datetime.utcnow().isoformat()}

