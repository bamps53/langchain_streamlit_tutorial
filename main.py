import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

load_dotenv()

SYSTEM_MESSAGE = "あなたは極めて陽気な関西人の芸人です。何を質問されても面白おかしく回答してください。"


def init_history():
    st.session_state.messages = [SystemMessage(content=SYSTEM_MESSAGE)]


def main():
    st.set_page_config(
        page_title="ChatGPT clone",
        page_icon="🤖",
    )

    with st.sidebar:
        model_name = st.radio("Select a model", ["gpt-4o-mini", "gpt-4o"])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0)
        clear_history = st.button("Clear chat history")
        if clear_history or "messages" not in st.session_state:
            init_history()

    if "messages" not in st.session_state:
        init_history()

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

    llm = ChatOpenAI(model=model_name, temperature=temperature, streaming=True)

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
