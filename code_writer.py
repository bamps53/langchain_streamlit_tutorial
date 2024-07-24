import io
import json
import os
import sys
import traceback
from loguru import logger
from matplotlib import pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from llm import LLM, HumanMessage, SystemMessage, AIMessage, MODELS, from_raw_message
from editor import get_code_editor

load_dotenv()

DEFAULT_SYSTEM_MESSAGE_JA = "あなたはコーディングに特化したAIアシスタントです。実行可能なpythonのコードのみを'code'領域に格納してjson形式で返してください。"
DEFAULT_SYSTEM_MESSAGE_EN = "You are an AI assistant specialized in coding. Please store executable Python code only in the 'code' field and return it in JSON format."


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

    # Save the current stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        # Redirect stdout and stderr to the StringIO object
        sys.stdout = output
        sys.stderr = output

        # Execute the code
        exec(code)

        # A hack to find out if the code contains a plot
        if "import matplotlib.pyplot as plt" in code:
            # Save the plot to a file
            plot_filename = "plot.png"
            plt.savefig(plot_filename)
            st.image(plot_filename)
            plt.clf()

        # Get the output from the StringIO object
        result = output.getvalue()
        has_error = False

    except Exception as e:
        # Capture the stack trace and append it to the result
        result = output.getvalue() + "\n" + traceback.format_exc()
        has_error = True

    finally:
        # Restore the original stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return result, has_error


def main():
    st.set_page_config(
        page_title="Code Writer",
        page_icon="🤖",
    )

    with st.sidebar:
        session_name = st.text_input("Session name", "example")
        upload_file = st.file_uploader("Upload a file")

        # model_name = st.radio("Select a model", MODELS)
        # temperature = st.slider("Temperature", 0.0, 1.0, 0.0)
        model_name = "gpt-4o-mini"  # しばらく固定
        temperature = 0.0  # しばらく固定

        language = st.radio("Select a language", ["Japanese", "English"])
        if language == "English":
            system_message = st.text_area("System message", DEFAULT_SYSTEM_MESSAGE_EN, height=150)
        else:
            system_message = st.text_area("System message", DEFAULT_SYSTEM_MESSAGE_JA, height=150)

        clear_history = st.button("Clear chat history")

    # Create a directory for the session
    session_dir = f"sessions/{session_name}"
    os.makedirs(session_dir, exist_ok=True)

    # save uploaded file
    if upload_file:
        with open(f"{session_dir}/{upload_file.name}", "wb") as f:
            f.write(upload_file.getbuffer())

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

        num_retry = 5
        i = 0
        while i < num_retry:
            with st.spinner("AI is writing a code..."):
                content = llm.chat(st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=content))

            content = eval(content)  # convert to dict
            with st.chat_message("assistant"):
                get_code_editor(content["code"])
                ret, has_error = execute_and_capture_output(content["code"])
                st.code(ret)
            
            # if there is no error, break the loop
            if not has_error:
                break
            else:
                # if there is an error, add the error message to the chat history and retry
                st.session_state.messages.append(HumanMessage(content=ret))

            i += 1

    # save the chat history
    with open(session_history_path, "w") as f:
        json.dump([m.as_raw_message() for m in st.session_state.messages], f)


if __name__ == "__main__":
    main()
