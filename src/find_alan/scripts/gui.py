import json
import os
import urllib.error
import urllib.request

import gradio as gr
from PIL import Image

from find_alan.scene_generate import (
    DEFAULT_LLM_MODEL,
    SceneGenerationConfig,
    generate_scene_stream,
    resolve_tile_prompts,
)

MODELS: dict[str, dict] = {
    "FLUX.2-klein-4B (fast)": dict(
        model_id="black-forest-labs/FLUX.2-klein-4B",
        default_steps=4,
        default_guidance=3.5,
    ),
    "FLUX.2-klein-9B (balanced)": dict(
        model_id="black-forest-labs/FLUX.2-klein-9B",
        default_steps=8,
        default_guidance=3.5,
    ),
    "FLUX.2-dev-Turbo (fast+quality)": dict(
        model_id="black-forest-labs/FLUX.2-dev",
        default_steps=8,
        default_guidance=2.5,
        lora_weights="fal/FLUX.2-dev-Turbo",
        lora_weight_name="flux.2-turbo-lora.safetensors",
        custom_sigmas=(1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031),
    ),
    "FLUX.2-dev (quality)": dict(
        model_id="black-forest-labs/FLUX.2-dev",
        default_steps=28,
        default_guidance=3.5,
    ),
}
DEFAULT_MODEL = "FLUX.2-klein-4B (fast)"
DEFAULT_LLM_URL = "http://localhost:8000/v1"


def _n_to_grid(n: int) -> tuple[int, int]:
    if n <= 3:
        return n, 1
    return 2, 2


def on_model_change(model_label: str):
    return gr.update(value=MODELS[model_label]["default_steps"])


def check_llm_status(llm_url: str) -> str:
    url = llm_url.strip()
    if not url:
        return "No URL set"
    try:
        req = urllib.request.Request(
            url.rstrip("/") + "/models",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            body = json.loads(resp.read().decode())
        models = [m["id"] for m in body.get("data", [])]
        label = ", ".join(models) if models else "no models listed"
        return f"✓ Connected — {label}"
    except urllib.error.URLError as exc:
        return f"✗ Unavailable — {exc.reason}"
    except Exception as exc:
        return f"✗ Unavailable — {exc}"


def on_mode_change(mode: str, llm_url: str):
    visible = mode == "LLM"
    status = check_llm_status(llm_url) if visible else ""
    return gr.update(visible=visible), gr.update(value=status, visible=visible)


def run_generate(
    prompt: str,
    n_scenes: int,
    model_label: str,
    steps: int,
    mode: str,
    llm_url: str,
):
    if not prompt or not prompt.strip():
        gr.Warning("Please enter a prompt.")
        return

    cols, rows = _n_to_grid(n_scenes)
    total = cols * rows
    model_cfg = MODELS[model_label]

    # ── Step 1: resolve prompts and display them ────────────────────────────
    yield [], [], "Resolving prompts…", gr.update(visible=False)

    llm_base_url = llm_url.strip() if mode == "LLM" else None
    raw_prompts, source = resolve_tile_prompts(
        theme=prompt.strip(),
        cols=cols,
        rows=rows,
        llm_base_url=llm_base_url,
        llm_model=DEFAULT_LLM_MODEL,
        llm_timeout=60.0,
    )
    prompts_rows = [[f"Tile {i + 1}", p] for i, p in enumerate(raw_prompts)]
    source_label = "conference example" if source == "fallback" else source
    yield (
        [],
        [],
        f"Prompts ready ({source_label}). Loading model…",
        gr.update(value=prompts_rows, visible=True),
    )

    # ── Step 2: generate tiles ───────────────────────────────────────────────
    cfg = SceneGenerationConfig(
        theme=prompt.strip(),
        cols=cols,
        rows=rows,
        save_tiles=True,
        flux_model_id=model_cfg["model_id"],
        steps=steps,
        guidance_scale=model_cfg.get("default_guidance", 3.5),
        tile_prompts=tuple(raw_prompts),
        lora_weights=model_cfg.get("lora_weights"),
        lora_weight_name=model_cfg.get("lora_weight_name"),
        custom_sigmas=model_cfg.get("custom_sigmas"),
    )

    tile_paths: list[str] = []
    for idx, value in generate_scene_stream(cfg):
        if idx is None:
            result = value
            yield (
                [Image.open(p) for p in result.tile_paths],
                [str(p) for p in result.tile_paths],
                f"Done — {total} sub-scene{'s' if total != 1 else ''} generated.",
                gr.update(visible=True),
            )
        else:
            if value is not None:
                tile_paths.append(str(value))
            yield (
                [Image.open(p) for p in tile_paths],
                tile_paths,
                f"Tile {idx + 1}/{total} complete",
                gr.update(visible=True),
            )


def on_select(evt: gr.SelectData, paths: list):
    if paths and 0 <= evt.index < len(paths):
        return paths[evt.index], f"Selected: image {evt.index + 1} of {len(paths)}"
    return None, "No image selected"


def run_improve(selected_path: str):
    if not selected_path:
        raise gr.Error("Please select an image from the gallery first.")
    # Placeholder — upscale → refine → inpaint will be wired here
    result = Image.open(selected_path)
    return (
        result,
        gr.update(visible=False),   # gallery_col
        gr.update(visible=True),    # improved_col
        gr.update(visible=False),   # generate_col
        gr.update(visible=False),   # prompts_col
        gr.update(visible=False),   # status_label
        gr.update(visible=False),   # selection_label
        gr.update(visible=False),   # improve_btn
    )


def run_restart():
    return (
        gr.update(visible=True),                              # gallery_col
        gr.update(visible=False),                             # improved_col
        [],                                                   # gallery_output
        [],                                                   # paths_state
        None,                                                 # selected_path_state
        gr.update(visible=True),                              # generate_col
        gr.update(visible=False),                             # prompts_col
        gr.update(value="", visible=True),                    # status_label
        gr.update(value="No image selected", visible=True),  # selection_label
        gr.update(visible=True),                              # improve_btn
    )


def main():
    with gr.Blocks(title="find-alan") as demo:
        gr.Markdown("# Find Alan")

        paths_state = gr.State([])
        selected_path_state = gr.State(None)

        # ── Image area ───────────────────────────────────────────────────────
        with gr.Column(visible=True) as gallery_col:
            gallery_output = gr.Gallery(
                columns=2, rows=2, show_label=False, object_fit="contain"
            )

        with gr.Column(visible=False) as improved_col:
            improved_output = gr.Image(show_label=False)
            restart_btn = gr.Button("Start Again", variant="secondary")

        # ── Stage 1: generation controls ────────────────────────────────────
        with gr.Column() as generate_col:
            prompt_input = gr.Textbox(
                placeholder="Enter a theme or prompt…", show_label=False
            )
            with gr.Row():
                mode_input = gr.Radio(
                    choices=["Example", "LLM"],
                    value="Example",
                    label="Prompt source",
                )
                with gr.Column(visible=False) as llm_col:
                    with gr.Row():
                        llm_url_input = gr.Textbox(
                            value=DEFAULT_LLM_URL,
                            label="LLM endpoint URL",
                        )
                        check_llm_btn = gr.Button("Check", size="sm", scale=0)
                    llm_status_label = gr.Textbox(
                        value="", interactive=False, show_label=False
                    )
            with gr.Row():
                model_input = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value=DEFAULT_MODEL,
                    label="Model",
                )
                steps_input = gr.Slider(
                    minimum=1, maximum=28, step=1,
                    value=MODELS[DEFAULT_MODEL]["default_steps"],
                    label="Steps",
                )
                n_scenes_input = gr.Slider(
                    minimum=1, maximum=4, step=1, value=4,
                    label="Sub-scenes",
                )
            generate_btn = gr.Button("Generate", variant="primary")

        # ── Prompts display (shown after resolve, hidden at start/restart) ──
        with gr.Column(visible=False) as prompts_col:
            gr.Markdown("#### Sub-scene prompts")
            prompts_display = gr.Dataframe(
                headers=["Tile", "Prompt"],
                col_count=(2, "fixed"),
                interactive=False,
                wrap=True,
            )

        # ── Stage 2: improve controls ────────────────────────────────────────
        status_label = gr.Textbox(value="", interactive=False, show_label=False)
        selection_label = gr.Textbox(
            value="No image selected", interactive=False, show_label=False
        )
        improve_btn = gr.Button("Improve Selected", variant="secondary")

        # ── Event wiring ─────────────────────────────────────────────────────
        mode_input.change(
            fn=on_mode_change,
            inputs=[mode_input, llm_url_input],
            outputs=[llm_col, llm_status_label],
        )
        check_llm_btn.click(
            fn=check_llm_status,
            inputs=llm_url_input,
            outputs=llm_status_label,
        )
        model_input.change(fn=on_model_change, inputs=model_input, outputs=steps_input)

        generate_inputs = [
            prompt_input, n_scenes_input, model_input, steps_input,
            mode_input, llm_url_input,
        ]
        generate_outputs = [gallery_output, paths_state, status_label, prompts_col]

        generate_btn.click(
            fn=run_generate,
            inputs=generate_inputs,
            outputs=generate_outputs,
        )
        prompt_input.submit(
            fn=run_generate,
            inputs=generate_inputs,
            outputs=generate_outputs,
        )
        gallery_output.select(
            fn=on_select,
            inputs=paths_state,
            outputs=[selected_path_state, selection_label],
        )
        improve_btn.click(
            fn=run_improve,
            inputs=selected_path_state,
            outputs=[
                improved_output,
                gallery_col, improved_col, generate_col, prompts_col,
                status_label, selection_label, improve_btn,
            ],
        )
        restart_btn.click(
            fn=run_restart,
            outputs=[
                gallery_col, improved_col,
                gallery_output, paths_state, selected_path_state,
                generate_col, prompts_col,
                status_label, selection_label, improve_btn,
            ],
        )

    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GUI_PORT", 7860)),
        css="footer { display: none !important; }",
        show_error=True,
    )


if __name__ == "__main__":
    main()
