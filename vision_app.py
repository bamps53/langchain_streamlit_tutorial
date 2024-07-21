import base64
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()

SYSTEM_MESSAGE = "あなたはOCRが得意なAIアシスタントです。画像に写っている文字を読み取って回答してください。"
OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o"]
GOOGLE_MODELS = ["gemini-1.5-pro-latest", "gemini-1.5-flash"]
ANTHROPIC_MODELS = ["claude-3-5-sonnet-20240620"]
MODELS = OPENAI_MODELS + GOOGLE_MODELS + ANTHROPIC_MODELS


class OCROutput(BaseModel):
    """Information about an image."""

    image_description: str = Field(description="a short description of the image")
    text: str = Field(description="text extracted from the image")


parser = JsonOutputParser(pydantic_object=OCROutput)


def init_history():
    st.session_state.messages = [SystemMessage(content=SYSTEM_MESSAGE)]


def show_history():
    def show_content(content: str | list[dict]):
        if isinstance(content, str):
            st.write(content)
        elif isinstance(content, list):
            for item in content:
                if item["type"] == "text":
                    st.write(item["text"])
                elif item["type"] == "image_url":
                    st.image(item["image_url"]["url"], use_column_width=True)

    for message in st.session_state.messages:
        if isinstance(message, SystemMessage):
            with st.chat_message("system"):
                show_content(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                show_content(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                show_content(message.content)


def is_image_used_in_history():
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            for item in message.content:
                if item["type"] == "image_url":
                    return True
    return False


def build_llm(model_name, temperature):
    if model_name in OPENAI_MODELS:
        return ChatOpenAI(model=model_name, temperature=temperature, streaming=True)
    elif model_name in GOOGLE_MODELS:
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, streaming=True)
    elif model_name in ANTHROPIC_MODELS:
        return ChatAnthropic(model=model_name, temperature=temperature, streaming=True)
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def to_base64(uploaded_file):
    name = uploaded_file.name
    ext = name.split(".")[-1]
    file_buffer = uploaded_file.read()
    b64 = base64.b64encode(file_buffer).decode()
    return f"data:image/{ext};base64,{b64}"


def create_image_message(user_input: str, images: list[str]):
    content = [
        {"type": "text", "text": user_input},
    ]
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": to_base64(image)}})
    return HumanMessage(content=content)


def main():
    st.set_page_config(
        page_title="My ChatGPT",
        page_icon="🤖",
    )

    with st.sidebar:
        st.session_state.images = st.file_uploader(label=" ", accept_multiple_files=True)
        task = st.radio("Select a task", ["OCR", "Chat"])
        model_name = st.radio("Select a model", MODELS)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0)
        clear_history = st.button("Clear chat history")
        if clear_history or "messages" not in st.session_state:
            init_history()

    if "messages" not in st.session_state:
        init_history()

    show_history()

    llm = build_llm(model_name, temperature)

    if user_input := st.chat_input("Type a message..."):
        with st.chat_message("user"):
            st.write(user_input)
        if task == "OCR" and st.session_state.images is not None and not is_image_used_in_history():
            for image in st.session_state.images:
                st.image(image, use_column_width=True)
            human_message = create_image_message(user_input, st.session_state.images)
        else:
            human_message = HumanMessage(content=user_input)
        st.session_state.messages.append(human_message)

        stream = llm.stream(st.session_state.messages)
        with st.chat_message("assistant"):
            content = st.write_stream(stream)
        st.session_state.messages.append(AIMessage(content=content))


if __name__ == "__main__":
    main()
