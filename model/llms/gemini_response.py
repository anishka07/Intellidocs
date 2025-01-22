import os
from dotenv import load_dotenv

import google.generativeai as genai

from utils.constants import ConstantSettings

load_dotenv()

api_key = os.getenv('GOOGLE_GEMINI_API_TOKEN')


def gemini_response(user_query: str, context: str = None) -> str:
    if context is not None:
        prompt = ConstantSettings.GEMINI_PROMPT.format(user_query, context)
    else:
        prompt = ConstantSettings.GEMINI_NC_PROMPT.format(user_query)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    r = model.generate_content(prompt)
    return r.text


if __name__ == '__main__':
    text = gemini_response(user_query="what is statistical learning?", context="macro nutrients")
    print(text)
