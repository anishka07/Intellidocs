import torch
from transformers import pipeline
import functools


def setup_pipeline(model_name: str, device: str = None):
    """
    Initialize the pipeline with optimized settings.
    Args:
        model_name: Name of the model to use
        device: Device to run the model on ('cuda' or 'cpu')
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        model_kwargs={
            "low_cpu_mem_usage": True,
            "use_cache": True
        }
    )


@functools.lru_cache(maxsize=128)
def get_chat_template(tokenizer, messages):
    """Cache the chat template results to avoid recomputation"""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def truncate_context(context: str, max_length: int = 1024) -> str:
    """Efficiently truncate context to maximum length"""
    return context[:max_length]


def llama_response(query: str, context: str, p) -> str:
    """
    Generate a response using the LLM model with optimized performance.
    Args:
        query: User query string
        context: Context string
        p: Optional pipeline instance
    Returns:
        str: Generated response
    """
    # Truncate context early to reduce memory usage
    context = truncate_context(context)

    # Prepare messages with minimal formatting
    messages = [
        {
            "role": "system",
            "content": "You are a Retrieval-Augmented Generation (RAG) system. Your job is to analyze the query and the retrieved context, then provide a clear, concise, and structured response. If the context doesn't match the question, provide your own answer."
        },
        {
            "role": "user",
            "content": f"Query: '{query}'. Context: {context}."
        }
    ]

    # Get cached template
    prompt = get_chat_template(p.tokenizer, tuple(map(str, messages)))

    # Optimize generation parameters
    outputs = p(
        prompt,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.5,
        top_k=20,
        top_p=0.9,
        pad_token_id=p.tokenizer.pad_token_id,
        eos_token_id=p.tokenizer.eos_token_id,
        use_cache=True,
        return_full_text=False  # Only return the new text
    )

    # Extract response efficiently
    generated_text = outputs[0]["generated_text"]

    # Split response using string methods instead of regex
    if "<|assistant|>" in generated_text:
        answer = generated_text.split("<|assistant|>")[1].strip()
    else:
        answer = generated_text.strip()

    return answer


# Optional: Add a class-based implementation for better resource management
class LLMResponseGenerator:
    def __init__(self, model_name: str, device: str = None):
        self.pipeline = setup_pipeline(model_name, device)

    def generate_response(self, query: str, context: str) -> str:
        return llama_response(query, context, self.pipeline)

    def __del__(self):
        # Clean up resources
        if hasattr(self, 'pipeline'):
            del self.pipeline
            torch.cuda.empty_cache()