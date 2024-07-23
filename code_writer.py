import io
import sys
from loguru import logger
from matplotlib import pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from llm import LLM, HumanMessage, SystemMessage, AIMessage, MODELS
from editor import get_code_editor

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
    st.markdown("""
                ## Code Writerの動作
                1. 人間が自然言語で仕様を入力する
                2. LLMがテストを書く
                3. 人間がテストをチェックして承認する
                4. LLMがコードを生成する
                5. テストを実行する
                6. テストが通ったら終了、通らなかったら修正する
                7. 結果を報告する
                """)

    response_dict = get_code_editor("# write a pseudo code here\n")
    logger.debug(response_dict)

    ret = execute_and_capture_output(response_dict["text"])
    st.write(ret)

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

    llm = LLM(model_name, temperature, json_mode=True)

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
