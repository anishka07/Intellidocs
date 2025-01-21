import torch
from transformers import pipeline

from utils.constants import ConstantSettings

pipe = pipeline(
    "text-generation",
    model=ConstantSettings.LLM_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)


def truncate_context(context, max_length=1024):
    return context[:max_length]


def llama_response(query: str, context: str, p=pipe):
    context = truncate_context(context)
    messages = [
        {
            "role": "system",
            "content": "You are a Retrieval-Augmented Generation (RAG) system. Your job is to analyze the query and the retrieved context, then provide a clear, concise, and structured response.",
        },
        {
            "role": "user",
            "content": f"The user asked the query: '{query}'. Based on the context retrieved, summarize and provide a structured response. this is the context: {context}. If the context does not match the question, give your own answer.",
        },
    ]
    prompt = p.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    outputs = p(
        prompt,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.5,
        top_k=20,
        top_p=0.9
    )
    generated_text = outputs[0]["generated_text"]
    if "<|assistant|>" in generated_text:
        answer = generated_text.split("<|assistant|>")[1].strip()
    else:
        answer = generated_text.strip()
    return answer
