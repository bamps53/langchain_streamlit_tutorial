import io
import json
import os
import subprocess
import sys
import traceback
from loguru import logger
from matplotlib import pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from llm import LLM, EnvMessage, HumanMessage, SystemMessage, AIMessage, from_raw_message
from editor import get_code_editor

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

JSON_SCHEMA_JA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "そのまま実行可能なコード"},
        "language": {"type": "string", "description": "コードの言語 (python or sh)"},
        # "response": {"type": "string", "description": "ユーザーへの返答"},
    },
    "required": ["code", "language"],
}

JSON_SCHEMA_EN = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "Executable code"},
        "language": {"type": "string", "description": "Language of the code (python or sh)"},
        # "response": {"type": "string", "description": "Response to the user"},
    },
    "required": ["code", "language"],
}

DEFAULT_SYSTEM_MESSAGE_JA = f"あなたはコーディングに特化したAIアシスタントです。以下のスキーマに従ったJSON形式でコードを記述してください。\n\n```json\n{json.dumps(JSON_SCHEMA_JA, indent=2, ensure_ascii=False)}\n```"  # noqa: E501
DEFAULT_SYSTEM_MESSAGE_EN = f"You are an AI assistant specialized in coding. Please write the code in JSON format according to the following schema.\n\n```json\n{json.dumps(JSON_SCHEMA_EN, indent=2)}\n```"  # noqa: E501


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
                if "response" in content:
                    st.write(content["response"])
                if "code" in content:
                    get_code_editor(content["code"], content["language"])
                else:
                    st.write(content)
        elif isinstance(message, EnvMessage):
            with st.chat_message("env", avatar="🖥"):
                st.code(message.content)


def execute_and_capture_output(code, language):
    if language == "python":
        # Create a StringIO object to capture the output
        output = io.StringIO()

        # Save the current stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            # Redirect stdout and stderr to the StringIO object
            sys.stdout = output
            sys.stderr = output

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

    elif language == "sh":
        try:
            # Run the shell script
            result = subprocess.run(["bash", "-c", code], capture_output=True, text=True)

            has_error = result.returncode != 0

            # Capture standard output and standard error
            stdout = result.stdout
            stderr = result.stderr

            return stdout + stderr, has_error
        except Exception as e:
            # Capture any exception that occurs and its stack trace
            return str(e), True

    else:
        return "Unsupported language", True


def main():
    os.chdir(ROOT_DIR)  # Change the current working directory to the root directory

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

        language = st.radio("Select a language", ["English", "Japanese"])
        if language == "English":
            system_message = st.text_area("System message", DEFAULT_SYSTEM_MESSAGE_EN, height=150)
        else:
            system_message = st.text_area("System message", DEFAULT_SYSTEM_MESSAGE_JA, height=150)

        clear_history = st.button("Clear chat history")

    # Create a directory for the session
    session_dir = f"{ROOT_DIR}/sessions/{session_name}"
    os.makedirs(session_dir, exist_ok=True)
    os.chdir(session_dir)  # Change the current working directory to the session directory

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
                logger.debug(content)

            with st.chat_message("assistant"):
                if "response" in content:
                    st.write(content["response"])
                get_code_editor(content["code"], language=content["language"])

            with st.chat_message("env", avatar="🖥"):
                ret, has_error = execute_and_capture_output(content["code"], content["language"])
                st.code(ret)
                st.session_state.messages.append(EnvMessage(content=ret))

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
