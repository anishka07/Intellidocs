import os
from dotenv import load_dotenv

import google.generativeai as genai

from src.entities.intellidocs_response import IntellidocsResponse
from utils.constants import ConstantSettings

load_dotenv()

api_key = os.getenv("GOOGLE_GEMINI_API")


def gemini_response(user_query: str, context: str) -> IntellidocsResponse:
    prompt = ConstantSettings.GEMINI_PROMPT.format(user_query, context)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    r = model.generate_content(prompt)
    response = IntellidocsResponse(
        structured_response=r.text
    )
    return response.structured_response


if __name__ == "__main__":
    text = gemini_response(user_query="what is statistical learning?", context="")
    print(text)
