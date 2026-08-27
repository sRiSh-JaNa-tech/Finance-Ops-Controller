import os
import json
import logging
from dotenv import load_dotenv
from finance_ops.agent.vertex_client import GeminiReconciliationClient

logging.basicConfig(level=logging.INFO)
load_dotenv()

client = GeminiReconciliationClient()
print("Model:", client.model_name)
print("API Key present:", bool(client.api_key))

res = client.call_gemini_api_native(
    messages=[{"role": "user", "parts": [{"text": "Reply with JSON: {\"status\": \"ok\"}"}]}]
)
print("API Result:", res)
