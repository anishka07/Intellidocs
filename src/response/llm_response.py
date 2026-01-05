from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from utils.constants import ConstantSettings

load_dotenv()


llm = HuggingFaceEndpoint(model="openai/gpt-oss-20b")
model = ChatHuggingFace(llm=llm)


def llm_response(query, context):
    response = model.invoke(ConstantSettings.LLM_PROMPT.format(query, context)).content
    return response
