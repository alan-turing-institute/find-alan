import os

import gradio as gr
from PIL import Image

from find_alan.pipeline_stub import generate_image, improve_image


def run_generate(prompt: str):
    if not prompt or not prompt.strip():
        gr.Warning("Please enter a prompt.")
        return gr.update(), gr.update()
    paths = generate_image(prompt.strip())
    return [Image.open(p) for p in paths], paths


def on_select(evt: gr.SelectData, paths: list):
    if paths and 0 <= evt.index < len(paths):
        return paths[evt.index], f"Selected: image {evt.index + 1} of {len(paths)}"
    return None, "No image selected"


def run_improve(selected_path: str):
    if not selected_path:
        raise gr.Error("Please select an image from the gallery first.")
    result = Image.open(improve_image(selected_path))
    return (
        result,
        gr.update(visible=False),   # gallery_col
        gr.update(visible=True),    # improved_col
        gr.update(visible=False),   # generate_col
        gr.update(visible=False),   # selection_label
        gr.update(visible=False),   # improve_btn
    )


def run_restart():
    return (
        gr.update(visible=True),                              # gallery_col
        gr.update(visible=False),                             # improved_col
        [],                                                   # gallery_output cleared
        [],                                                   # paths_state cleared
        None,                                                 # selected_path_state cleared
        gr.update(visible=True),                              # generate_col
        gr.update(value="No image selected", visible=True),  # selection_label
        gr.update(visible=True),                              # improve_btn
    )


def main():
    with gr.Blocks(title="find-alan") as demo:
        gr.Markdown("# Find Alan")

        paths_state = gr.State([])
        selected_path_state = gr.State(None)

        # Image area — toggled between stages
        with gr.Column(visible=True) as gallery_col:
            gallery_output = gr.Gallery(columns=2, rows=2, show_label=False, object_fit="contain")

        with gr.Column(visible=False) as improved_col:
            improved_output = gr.Image(show_label=False)
            restart_btn = gr.Button("Start Again", variant="secondary")

        # Stage 1 controls
        with gr.Column() as generate_col:
            prompt_input = gr.Textbox(placeholder="Enter a prompt...", show_label=False)
            generate_btn = gr.Button("Generate", variant="primary")

        # Stage 2 controls
        selection_label = gr.Textbox(value="No image selected", interactive=False, show_label=False)
        improve_btn = gr.Button("Improve Selected", variant="secondary")

        generate_btn.click(fn=run_generate, inputs=prompt_input, outputs=[gallery_output, paths_state])
        prompt_input.submit(fn=run_generate, inputs=prompt_input, outputs=[gallery_output, paths_state])
        gallery_output.select(fn=on_select, inputs=paths_state, outputs=[selected_path_state, selection_label])
        improve_btn.click(
            fn=run_improve,
            inputs=selected_path_state,
            outputs=[improved_output, gallery_col, improved_col, generate_col, selection_label, improve_btn],
        )
        restart_btn.click(
            fn=run_restart,
            outputs=[gallery_col, improved_col, gallery_output, paths_state, selected_path_state, generate_col, selection_label, improve_btn],
        )

    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GUI_PORT", 7860)), css="footer { display: none !important; }")


if __name__ == "__main__":
    main()
