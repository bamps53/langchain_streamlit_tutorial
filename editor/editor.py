import os
import json
from code_editor import code_editor

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(f"{ROOT_DIR}/resources/example_custom_buttons_bar_alt.json") as json_button_file_alt:
    custom_buttons_alt = json.load(json_button_file_alt)

# Load Info bar CSS from JSON file
with open(f"{ROOT_DIR}/resources/example_info_bar.json") as json_info_file:
    info_bar = json.load(json_info_file)


def get_code_editor(default_code, language="python"):
    info_bar["info"][0]["name"] = language
    return code_editor(
        default_code,
        height=[1, 22],
        lang=language,
        theme="contrast",
        shortcuts="vscode",
        focus=False,
        buttons=custom_buttons_alt,
        info=info_bar,
        props={"style": {"borderRadius": "0px 0px 8px 8px"}},
        response_mode="submit",
        options={"wrap": True},
    )
