import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

from app import ai

try:
    print("Testing Gemini integration with model:", ai.MODEL)
    res = ai._generate(contents="Hello! Please reply 'OK' if you can read this.", config=None)
    print("Success! Gemini response:", res.text)
except Exception as e:
    print("Error:", str(e))
