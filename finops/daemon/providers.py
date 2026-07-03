import os

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI


def _compose_message(prompt: str, context: str) -> str:
    if context:
        return f"{context}\n\n{prompt}"
    return prompt


async def call_llm(provider: str, model: str, prompt: str, context: str) -> tuple[str, int, int]:
    message = _compose_message(prompt, context)
    if provider == "anthropic":
        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        resp = await client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": message}],
        )
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens
    if provider == "openai":
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}],
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return text, usage.prompt_tokens, usage.completion_tokens
    raise ValueError(f"unknown provider: {provider}")
