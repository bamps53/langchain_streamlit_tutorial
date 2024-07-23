import json
from code_editor import code_editor


with open("editor/resources/example_custom_buttons_bar_alt.json") as json_button_file_alt:
    custom_buttons_alt = json.load(json_button_file_alt)

# Load Info bar CSS from JSON file
with open("editor/resources/example_info_bar.json") as json_info_file:
    info_bar = json.load(json_info_file)


def get_code_editor(default_code):
    return code_editor(
        default_code,
        height=[19, 22],
        lang="python",
        theme="contrast",
        shortcuts="vscode",
        focus=True,
        buttons=custom_buttons_alt,
        info=info_bar,
        props={"style": {"borderRadius": "0px 0px 8px 8px"}},
        response_mode="submit",
        options={"wrap": True},
    )
