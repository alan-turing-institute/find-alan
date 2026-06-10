import os

import gradio as gr
from PIL import Image

from find_alan.pipeline_stub import generate_image, improve_image


def run_generate(prompt: str):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    paths = generate_image(prompt.strip())
    return [Image.open(p) for p in paths], paths


def on_select(evt: gr.SelectData, paths: list):
    if paths and 0 <= evt.index < len(paths):
        return paths[evt.index], f"Selected: image {evt.index + 1} of {len(paths)}"
    return None, "No image selected"


def run_improve(selected_path: str):
    if not selected_path:
        raise gr.Error("Please select an image from the gallery first.")
    return Image.open(improve_image(selected_path))


def main():
    with gr.Blocks(title="find-alan") as demo:
        gr.Markdown("# Find Alan")

        paths_state = gr.State([])
        selected_path_state = gr.State(None)

        # Stage 1 — generate
        gallery_output = gr.Gallery(columns=2, rows=2, show_label=False, object_fit="contain")
        prompt_input = gr.Textbox(placeholder="Enter a prompt...", show_label=False)
        generate_btn = gr.Button("Generate", variant="primary")

        generate_btn.click(fn=run_generate, inputs=prompt_input, outputs=[gallery_output, paths_state])
        prompt_input.submit(fn=run_generate, inputs=prompt_input, outputs=[gallery_output, paths_state])

        # Stage 2 — improve
        selection_label = gr.Textbox(value="No image selected", interactive=False, show_label=False)
        improve_btn = gr.Button("Improve Selected", variant="secondary")
        improved_output = gr.Image(show_label=False)

        gallery_output.select(fn=on_select, inputs=paths_state, outputs=[selected_path_state, selection_label])
        improve_btn.click(fn=run_improve, inputs=selected_path_state, outputs=improved_output)

    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GUI_PORT", 7860)), css="footer { display: none !important; }")


if __name__ == "__main__":
    main()
