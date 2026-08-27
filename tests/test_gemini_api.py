import os
import pytest
from finance_ops.agent.vertex_client import GeminiReconciliationClient

def test_gemini_api_connectivity():
    """
    Tests whether the Gemini API is reachable and returning valid responses.
    This test will be skipped if the GEMINI_API_KEY environment variable is not set.
    """
    client = GeminiReconciliationClient()
    
    if not client.has_credentials or os.environ.get("RUN_LIVE_LLM_TESTS") != "1":
        pytest.skip("RUN_LIVE_LLM_TESTS not enabled. Skipping live API network test.")
        
    prompt = "Reply with 'API_OK' and nothing else."
    response = client.call_gemini_api(prompt)
    
    assert response is not None, "Gemini API returned None instead of a valid response."
    assert "API_OK" in response, f"Gemini API returned unexpected response: {response}"
