import gradio as gr

from find_alan.pipeline import generate_image


def run_pipeline(prompt: str):
    if not prompt or not prompt.strip():
        return None
    return generate_image(prompt.strip())


def main():
    with gr.Blocks(title="find-alan") as demo:
        image_output = gr.Image(label="Output", show_label=False)
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
