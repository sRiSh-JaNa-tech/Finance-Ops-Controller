import os
import pytest
from finance_ops.agent.vertex_client import GeminiReconciliationClient

def test_gemini_api_connectivity():
    """
    Tests whether the Gemini API is reachable and returning valid responses.
    This test will be skipped if the GEMINI_API_KEY environment variable is not set.
    """
    client = GeminiReconciliationClient()
    
    if not client.has_credentials:
        pytest.skip("GEMINI_API_KEY or VERTEX_API_KEY not set in environment. Skipping API test.")
        
    prompt = "Reply with 'API_OK' and nothing else."
    response = client.call_gemini_api(prompt)
    
    assert response is not None, "Gemini API returned None instead of a valid response."
    assert "API_OK" in response, f"Gemini API returned unexpected response: {response}"
