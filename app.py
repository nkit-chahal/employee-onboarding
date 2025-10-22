from dotenv import load_dotenv
import os

load_dotenv()  # loads from .env file
azure_api_key = os.getenv("AZURE_API_KEY")

print("Loaded Azure API Key:", azure_api_key[:5] + "..." if azure_api_key else "Not found")
