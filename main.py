import streamlit as st
from dotenv import load_dotenv
from llm import LLM, HumanMessage, SystemMessage, AIMessage, MODELS

load_dotenv()

DEFAULT_SYSTEM_MESSAGE = "あなたはコーディングに特化したAIアシスタントです。"


def init_history(system_message):
    st.session_state.messages = [SystemMessage(content=system_message)]


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


def main():
    st.set_page_config(
        page_title="Code Writer",
        page_icon="🤖",
    )

    with st.sidebar:
        model_name = st.radio("Select a model", MODELS)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0)
        system_message = st.text_area("System message", DEFAULT_SYSTEM_MESSAGE, height=150)
        clear_history = st.button("Clear chat history")
        if clear_history or "messages" not in st.session_state:
            init_history(system_message)

    if "messages" not in st.session_state:
        init_history(system_message)

    show_history()

    llm = LLM(model_name, temperature)

    if user_input := st.chat_input("Type a message..."):
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append(HumanMessage(content=user_input))

        stream = llm.chat(st.session_state.messages)
        with st.chat_message("assistant"):
            content = st.write_stream(stream)
        st.session_state.messages.append(AIMessage(content=content))


if __name__ == "__main__":
    main()
