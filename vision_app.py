import base64
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()

SYSTEM_MESSAGE = (
    "あなたは画像認識が得意なAIアシスタントです。画像に写っている内容及び文字を読み取って回答してください。"
)
OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o"]
GOOGLE_MODELS = ["gemini-1.5-pro-latest", "gemini-1.5-flash"]
ANTHROPIC_MODELS = ["claude-3-5-sonnet-20240620"]
MODELS = OPENAI_MODELS + GOOGLE_MODELS + ANTHROPIC_MODELS


class VisionOutput(BaseModel):
    """Information about an image."""

    image_description: str = Field(description="簡単な画像の内容の描画")
    text: str = Field(description="画像から抽出されたテキスト")


parser = PydanticOutputParser(pydantic_object=VisionOutput)


def init_history():
    st.session_state.messages = [SystemMessage(content=SYSTEM_MESSAGE)]


def show_history():
    def show_content(content: str | list[dict]):
        if isinstance(content, str):
            st.write(content)
        elif isinstance(content, list):
            text_contents = [item["text"] for item in content if item["type"] == "text"]
            image_contents = [item["image_url"]["url"] for item in content if item["type"] == "image_url"]
            for text_content in text_contents:
                st.write(text_content)
                break  # 2個目以降はFunction callingのインストラクションなのでスキップ
            for image_content in image_contents:
                st.image(image_content, use_column_width=True)

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
        {"type": "text", "text": parser.get_format_instructions()},
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
        task = st.radio("Select a task", ["Vision", "Chat"])
        model_name = st.radio("Select a model", MODELS)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0)
        clear_history = st.button("Clear chat history")
        if clear_history or "messages" not in st.session_state:
            init_history()

    if "messages" not in st.session_state:
        init_history()

    show_history()

    if user_input := st.chat_input("Type a message..."):
        # Input
        with st.chat_message("user"):
            st.write(user_input)
            has_image = False
            if task == "Vision" and st.session_state.images is not None and not is_image_used_in_history():
                has_image = True
                for image in st.session_state.images:
                    st.image(image, use_column_width=True)
                human_message = create_image_message(user_input, st.session_state.images)
            else:
                human_message = HumanMessage(content=user_input)
        st.session_state.messages.append(human_message)

        # Output
        with st.chat_message("assistant"):
            llm = build_llm(model_name, temperature)
            stream = llm.stream(st.session_state.messages)
            if has_image:
                raw_json_output = st.write_stream(stream)
                output = parser.parse(raw_json_output)
                content = f"画像の内容: {output.image_description}\n\n画像から抽出されたテキスト: {output.text}"
            else:
                content = st.write_stream(stream)
        st.session_state.messages.append(AIMessage(content=content))


if __name__ == "__main__":
    main()
