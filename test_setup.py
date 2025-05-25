"""
Test script to verify OpenAI API setup before running full analysis.
"""

import os
import asyncio
from openai import AsyncOpenAI


async def test_openai_connection():
    """Test basic OpenAI API functionality."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("\nPlease set it using:")
        print("  PowerShell: $env:OPENAI_API_KEY=\"your-api-key\"")
        print("  Command Prompt: set OPENAI_API_KEY=your-api-key")
        return False
    
    print("✓ API key found")
    
    client = AsyncOpenAI(api_key=api_key)
    
    # Test chat completion
    print("\nTesting chat completion (gpt-4o-mini)...")
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'Hello, Moneybench!'"}],
            max_tokens=10
        )
        print(f"✓ Chat response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ Chat completion failed: {e}")
        return False
    
    # Test embeddings
    print("\nTesting embeddings (text-embedding-3-large)...")
    try:
        response = await client.embeddings.create(
            model="text-embedding-3-large",
            input="Test embedding for Moneybench"
        )
        embedding_dim = len(response.data[0].embedding)
        print(f"✓ Embedding created: {embedding_dim} dimensions")
    except Exception as e:
        print(f"❌ Embedding creation failed: {e}")
        return False
    
    print("\n✅ All tests passed! Ready to run analysis.")
    return True


if __name__ == "__main__":
    asyncio.run(test_openai_connection()) 