import gradio as gr
from PIL import Image

from find_alan.pipeline_stub import generate_image


def run_pipeline(prompt: str):
    if not prompt or not prompt.strip():
        return None
    paths = generate_image(prompt.strip())
    return [Image.open(p) for p in paths]


def main():
    with gr.Blocks(title="find-alan", css="footer { display: none !important; }") as demo:
        gr.Markdown("# Find Alan")
        image_output = gr.Gallery(columns=2, rows=2, show_label=False, object_fit="contain")
        prompt_input = gr.Textbox(
            placeholder="Enter a prompt...",
            label="Prompt",
            show_label=False,
        )
        generate_btn = gr.Button("Generate", variant="primary")
        generate_btn.click(fn=run_pipeline, inputs=prompt_input, outputs=image_output)
        prompt_input.submit(fn=run_pipeline, inputs=prompt_input, outputs=image_output)

    demo.launch()


if __name__ == "__main__":
    main()
