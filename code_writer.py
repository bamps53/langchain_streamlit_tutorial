import io
import json
import os
import sys
from loguru import logger
from matplotlib import pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from llm import LLM, HumanMessage, SystemMessage, AIMessage, MODELS, from_raw_message
from editor import get_code_editor

load_dotenv()

DEFAULT_SYSTEM_MESSAGE = "あなたはコーディングに特化したAIアシスタントです。実行可能なpythonのコードのみを'code'領域に格納してjson形式で返してください。"


def init_history(system_message):
    st.session_state.messages = [SystemMessage(content=system_message)]


def show_history():
    for message in st.session_state.messages:
        if isinstance(message, SystemMessage):
            pass
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                content = eval(message.content)
                if "code" in content:
                    get_code_editor(content["code"])
                else:
                    st.write(content)


def execute_and_capture_output(code):
    # Create a StringIO object to capture the output
    output = io.StringIO()

    # Save the current stdout
    old_stdout = sys.stdout

    try:
        # Redirect stdout to the StringIO object
        sys.stdout = output

        # Execute the code
        exec(code)

        # a hack to find out if the code contains a plot
        if "import matplotlib.pyplot as plt" in code:
            # Save the plot to a file
            plot_filename = "plot.png"
            plt.savefig(plot_filename)
            st.image(plot_filename)
            plt.clf()

        # Get the output from the StringIO object
        result = output.getvalue()

    except Exception as e:
        result = str(e)

    finally:
        # Restore the original stdout
        sys.stdout = old_stdout

    return result


def main():
    st.set_page_config(
        page_title="Code Writer",
        page_icon="🤖",
    )

    with st.sidebar:
        session_name = st.text_input("Session name", "example")
        model_name = st.radio("Select a model", MODELS)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0)
        system_message = st.text_area("System message", DEFAULT_SYSTEM_MESSAGE, height=150)
        clear_history = st.button("Clear chat history")

    # Create a directory for the session
    session_dir = f"sessions/{session_name}"
    os.makedirs(session_dir, exist_ok=True)

    # Load the chat history
    session_history_path = f"{session_dir}/history.json"
    if os.path.exists(session_history_path):
        with open(session_history_path, "r") as f:
            st.session_state.messages = [from_raw_message(m) for m in json.load(f)]

    if clear_history or "messages" not in st.session_state:
        init_history(system_message)

    show_history()

    llm = LLM(model_name, temperature, json_mode=True, stream=False)

    if user_input := st.chat_input("Type a message..."):
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append(HumanMessage(content=user_input))

        with st.spinner("AI is writing a code..."):
            content = llm.chat(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=content))

        content = eval(content)  # convert to dict
        with st.chat_message("assistant"):
            get_code_editor(content["code"])
            ret = execute_and_capture_output(content["code"])
            st.write(ret)

    # save the chat history
    with open(session_history_path, "w") as f:
        json.dump([m.as_raw_message() for m in st.session_state.messages], f)


if __name__ == "__main__":
    main()
