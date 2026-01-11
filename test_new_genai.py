import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ GOOGLE_API_KEY not found")
else:
    print(f"✅ Found GOOGLE_API_KEY: {api_key[:5]}...{api_key[-5:]}")
    
    client = genai.Client(api_key=api_key)
    
    print("\n--- Testing Generation with google-genai ---")
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Hello! Can you hear me?"
        )
        print(f"✅ SUCCESS! Response: {response.text[:100]}...")
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
