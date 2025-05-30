#!/usr/bin/env python3
"""
Simple test script to verify OpenRouter integration is working.
"""

import asyncio
import os
from openai import AsyncOpenAI


async def test_openrouter():
    """Test OpenRouter API connection."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Please set OPENROUTER_API_KEY environment variable")
        return False
    
    print("Testing OpenRouter connection...")
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    try:
        # Test with a simple prompt
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini",  # Default model
            messages=[
                {"role": "user", "content": "Say 'OpenRouter is working!' if you can read this."}
            ],
            max_tokens=50,
            extra_headers={
                "HTTP-Referer": "https://github.com/moneybench",
                "X-Title": "Moneybench Test"
            }
        )
        
        result = response.choices[0].message.content
        print(f"✅ OpenRouter response: {result}")
        print(f"   Model used: {response.model}")
        return True
        
    except Exception as e:
        print(f"❌ OpenRouter error: {e}")
        return False


async def test_openai_embeddings():
    """Test OpenAI embeddings."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return False
    
    print("\nTesting OpenAI embeddings...")
    
    client = AsyncOpenAI(api_key=api_key)
    
    try:
        response = await client.embeddings.create(
            model="text-embedding-3-large",
            input="Test embedding"
        )
        
        embedding = response.data[0].embedding
        print(f"✅ OpenAI embeddings working! Dimension: {len(embedding)}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI error: {e}")
        return False


async def main():
    """Run all tests."""
    print("Moneybench OpenRouter Integration Test")
    print("=" * 40)
    
    # Test both services
    openrouter_ok = await test_openrouter()
    openai_ok = await test_openai_embeddings()
    
    print("\n" + "=" * 40)
    if openrouter_ok and openai_ok:
        print("✅ All tests passed! You're ready to run the analysis.")
    else:
        print("❌ Some tests failed. Please check your API keys and try again.")
        if not openrouter_ok:
            print("   - Fix OPENROUTER_API_KEY")
        if not openai_ok:
            print("   - Fix OPENAI_API_KEY")


if __name__ == "__main__":
    asyncio.run(main()) 