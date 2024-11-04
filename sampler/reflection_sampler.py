import base64
import time
from typing import Any

import openai
from openai import OpenAI

from type_definitions import MessageList, SamplerBase

REFLECTION_SYSTEM_MESSAGE = '''You are Tess-R1, an advanced AI that was created for complex reasoning. Given a user query, you are able to first create a Chain-of-Thought (CoT) reasoning. Once the CoT is devised, you then proceed to first think about how to answer. While doing this, you have the capability to contemplate on the thought, and also provide alternatives. Once the CoT steps have been thought through, you then respond by creating the final output.'''


class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "reflection_70b",
        system_message: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 6000,
    ):
        self.api_key_name = "test"
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = client

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    extra_body={"skip_special_tokens": False},
                    stream=False
                )
                try:
                    return response.choices[0].message.content.split("<output>")[1].split("</output>")[0]
                except:
                    return response.choices[0].message.content

            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return ""
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
