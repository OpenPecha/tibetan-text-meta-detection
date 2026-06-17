"""Gradio Space: Tibetan title/author span highlighting."""

import gradio as gr

from pipeline.inference import (
    DEFAULT_MODEL_ID,
    highlight_spans,
    load_model_and_tokenizer,
    predict_segment,
)

print(f"Loading model: {DEFAULT_MODEL_ID}")
MODEL, TOKENIZER, DEVICE = load_model_and_tokenizer(DEFAULT_MODEL_ID)


def detect_metadata(text: str) -> tuple[str, list[dict]]:
    if not text or not text.strip():
        return "<p><em>Paste a Tibetan text segment above.</em></p>", []
    spans = predict_segment(MODEL, TOKENIZER, text, device=DEVICE)
    return highlight_spans(text, spans), spans


DESCRIPTION = """
Paste a **Tibetan text segment**. The model runs the same sliding-window pipeline
used in training (512-token windows, stride 256, up to 15 begin + 15 end slides),
then highlights detected **title** (gold) and **author** (blue) spans.
"""

with gr.Blocks(title="Tibetan Metadata Highlight") as demo:
    gr.Markdown("# Tibetan Title & Author Detector")
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            text_in = gr.Textbox(
                label="Tibetan segment",
                lines=12,
                placeholder="Paste Tibetan text here…",
            )
            run_btn = gr.Button("Detect", variant="primary")
        with gr.Column():
            html_out = gr.HTML(label="Highlighted text")
            json_out = gr.JSON(label="Detected spans")

    run_btn.click(fn=detect_metadata, inputs=text_in, outputs=[html_out, json_out])
    text_in.submit(fn=detect_metadata, inputs=text_in, outputs=[html_out, json_out])

    gr.Markdown(
        "Model: [ganga4364/tibetan-metadata-koichi-ner]"
        "(https://huggingface.co/ganga4364/tibetan-metadata-koichi-ner) "
        "(Koichi tokenizer; window F1 12.3%) · "
        "Baseline: [ganga4364/tibetan-metadata-roberta-ner]"
        "(https://huggingface.co/ganga4364/tibetan-metadata-roberta-ner)"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
