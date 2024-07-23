from openai import OpenAI
from google.generativeai import GenerativeModel
from groq import Groq
from anthropic import Anthropic
from pydantic import BaseModel
from loguru import logger

OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o"]
GOOGLE_MODELS = ["gemini-1.5-pro-latest", "gemini-1.5-flash"]
GROQ_MODELS = ["gemma2-9b-it", "llama3-groq-70b-8192-tool-use-preview", "llama3-groq-8b-8192-tool-use-preview"]
ANTHROPIC_MODELS = ["claude-3-5-sonnet-20240620"]
MODELS = OPENAI_MODELS + GOOGLE_MODELS + GROQ_MODELS + ANTHROPIC_MODELS


def get_google_messages(messages):
    google_messages = []

    for message in messages:
        if message["role"] == "user":
            google_messages.append({"role": "user", "parts": [message.content[0]["text"]]})
        elif message["role"] == "assistant":
            # TODO 画像を処理する
            google_messages.append({"role": "model", "parts": [message.content[0]["text"]]})
        else:
            raise NotImplementedError()

    return google_messages


def get_groq_messages(messages):
    groq_messages = []

    for message in messages:
        logger.debug(message)
        if isinstance(message, SystemMessage):
            groq_messages.append({"role": "system", "content": message.content})
        elif isinstance(message, HumanMessage):
            groq_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            groq_messages.append({"role": "assistant", "content": message.content})
        else:
            raise NotImplementedError()

    return groq_messages


class BaseMessage(BaseModel):
    roll: str = ""
    content: str

    def as_raw_message(self):
        return {"role": self.roll, "content": [{"type": "text", "text": self.content}]}


class HumanMessage(BaseMessage):
    roll: str = "user"
    content: str


class SystemMessage(BaseMessage):
    roll: str = "system"
    content: str


class AIMessage(BaseMessage):
    roll: str = "assistant"
    content: str


class LLM:
    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 8192, stream: bool = True):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream

    def _get_messages(self, messages):
        if self.model_name in GROQ_MODELS:
            return get_groq_messages(messages)
        else:
            return [m.as_raw_message() for m in messages]

    def chat(self, messages):
        messages = self._get_messages(messages)
        logger.debug(messages)

        if self.model_name in OPENAI_MODELS:
            client = OpenAI()
            return client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stream=self.stream,
            )

        if self.model_name in GROQ_MODELS:
            client = Groq()
            return client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stream=self.stream,
            )

        system_message = messages[0]
        if system_message["role"] != "system":
            raise ValueError("The first message must be a system message.")

        rest_messages = messages[1:]

        if self.model_name in GOOGLE_MODELS:
            client = GenerativeModel(
                self.model_name,
                generation_config={
                    "temperature": self.temperature,
                    "top_p": 0.95,
                    "top_k": 64,
                    "max_output_tokens": self.max_tokens,
                    "response_mime_type": "text/plain",
                },
                system_instruction=system_message["content"][0]["text"],
            )
            return client.generate_content(contents=get_google_messages(rest_messages), stream=self.stream)

        if self.model_name in ANTHROPIC_MODELS:
            client = Anthropic()
            return client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_message["content"][0]["text"],
                messages=rest_messages,
                stream=self.stream,
            )
        else:
            raise ValueError(f"Invalid model name: {self.model_name}")
