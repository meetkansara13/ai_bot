import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(
    api_key=API_KEY,
    http_options={"api_version": "v1"}
)

print("Listing Available Models:\n")

models = client.models.list()

for m in models:
    print(m.name)
