import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Read Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)

# Function to call Gemini
def ask_mistral(prompt):
    model = genai.GenerativeModel("gemini-1.5-pro")   # You can use gemini-1.5-flash for cheaper & faster

    response = model.generate_content(
        [
            {"role": "system", "content": "You are a helpful financial assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.text

