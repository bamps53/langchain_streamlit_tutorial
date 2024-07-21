import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain.schema import SystemMessage, HumanMessage, AIMessage

load_dotenv()

SYSTEM_MESSAGE = "あなたは極めて陽気な関西人の芸人です。何を質問されても面白おかしく回答してください。"
OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o"]
GOOGLE_MODELS = ["gemini-1.5-pro-latest", "gemini-1.5-flash"]
GROQ_MODELS = ["gemma2-9b-it", "llama3-groq-70b-8192-tool-use-preview"]
ANTHROPIC_MODELS = ["claude-3-5-sonnet-20240620"]
MODELS = OPENAI_MODELS + GOOGLE_MODELS + GROQ_MODELS + ANTHROPIC_MODELS


def init_history():
    st.session_state.messages = [SystemMessage(content=SYSTEM_MESSAGE)]


def show_history():
    for message in st.session_state.messages:
        if isinstance(message, SystemMessage):
            with st.chat_message("system"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)


def build_llm(model_name, temperature):
    if model_name in OPENAI_MODELS:
        return ChatOpenAI(model=model_name, temperature=temperature, streaming=True)
    elif model_name in GOOGLE_MODELS:
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, streaming=True)
    elif model_name in GROQ_MODELS:
        return ChatGroq(model=model_name, temperature=temperature, streaming=True)
    elif model_name in ANTHROPIC_MODELS:
        return ChatAnthropic(model=model_name, temperature=temperature, streaming=True)
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def main():
    st.set_page_config(
        page_title="My ChatGPT",
        page_icon="🤖",
    )

    with st.sidebar:
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
        st.session_state.messages.append(HumanMessage(content=user_input))

        stream = llm.stream(st.session_state.messages)
        with st.chat_message("assistant"):
            content = st.write_stream(stream)
        st.session_state.messages.append(AIMessage(content=content))


if __name__ == "__main__":
    main()
