import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ GOOGLE_API_KEY not found in environment variables.")
    print("Please add GOOGLE_API_KEY=your_key_here to your .env file.")
else:
    print(f"✅ Found GOOGLE_API_KEY: {api_key[:5]}...{api_key[-5:]}")
    
    try:
        genai.configure(api_key=api_key, transport='rest')
        
        print("\n--- Listing Available Models (REST) ---")
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    print(f"FOUND MATCH: {m.name}")
        except Exception as e:
            print(f"\n❌ Error listing models: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

        print("\n--- Testing Specific Model Access ---")
        models_to_test = ["gemini-1.5-flash", "gemini-pro", "models/gemini-1.5-flash", "gemini-1.5-flash-001"]
        
        for model_name in models_to_test:
            print(f"Testing generation with: {model_name}")
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("Hello, can you hear me?")
                print(f"   ✅ SUCCESS! Response: {response.text[:50]}...")
            except Exception as e:
                print(f"   ❌ FAILED: {e}")

    except Exception as e:
        print(f"❌ Configuration failed: {e}")
