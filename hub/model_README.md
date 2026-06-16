---
license: apache-2.0
language:
- bo
tags:
- tibetan
- token-classification
- ner
- metadata
- roberta
library_name: transformers
base_model: spsither/tibetan_RoBERTa_S_e3
pipeline_tag: token-classification
---

# Tibetan Metadata RoBERTa NER

Fine-tuned [`spsither/tibetan_RoBERTa_S_e3`](https://huggingface.co/spsither/tibetan_RoBERTa_S_e3) for **title** and **author** span detection in Tibetan text segments.

## Labels (BIO)

| ID | Label |
|----|-------|
| 0 | O |
| 1 | B-TITLE |
| 2 | I-TITLE |
| 3 | B-AUTHOR |
| 4 | I-AUTHOR |

## Training

- Dataset: [ganga4364/tibetan-metadata-detector](https://huggingface.co/datasets/ganga4364/tibetan-metadata-detector)
- 3 epochs, batch size 64, lr 2e-5
- Sliding-window examples: 512 tokens, stride 256, 15B+15E overlap-aware

## Test metrics (balanced windows, 30,357 test windows)

| Metric | Value |
|--------|-------|
| span F1 | 3.1% |
| span precision | 1.8% |
| span recall | 13.9% |
| title F1 | 7.4% |
| author F1 | 1.0% |

Trained on balanced fixed-label windows (274k examples after O-only cap + author oversample).

## Segment-level test metrics (6,492 segments, merged inference)

| Metric | Value |
|--------|-------|
| exact span F1 | 8.0% |
| exact span precision | 5.4% |
| exact span recall | 15.3% |
| exact title F1 | 12.7% |
| exact author F1 | 0.7% |

## Inference

**Important:** Do not pass entire long segments in one forward pass. Use the same sliding-window pipeline as training (15 begin + 15 end, stride 256), merge overlapping predictions, then extract spans.

```python
from pipeline.inference import load_model_and_tokenizer, predict_segment, highlight_spans

model, tokenizer, device = load_model_and_tokenizer("ganga4364/tibetan-metadata-roberta-ner")
spans = predict_segment(model, tokenizer, your_tibetan_text, device=device)
print(highlight_spans(your_tibetan_text, spans))
```

## Demo

Try the Gradio Space: [ganga4364/tibetan-metadata-highlight](https://huggingface.co/spaces/ganga4364/tibetan-metadata-highlight)
