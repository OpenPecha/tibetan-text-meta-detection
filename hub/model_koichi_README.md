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
base_model: KoichiYasuoka/roberta-base-tibetan
pipeline_tag: token-classification
---

# Tibetan Metadata Koichi RoBERTa NER

Fine-tuned [`KoichiYasuoka/roberta-base-tibetan`](https://huggingface.co/KoichiYasuoka/roberta-base-tibetan) for **title** and **author** span detection in Tibetan text segments.

Comparison run using the same data, balance settings, and training hyperparameters as the [Spsither baseline](https://huggingface.co/ganga4364/tibetan-metadata-roberta-ner).

## Labels (BIO)

| ID | Label |
|----|-------|
| 0 | O |
| 1 | B-TITLE |
| 2 | I-TITLE |
| 3 | B-AUTHOR |
| 4 | I-AUTHOR |

## Training

- Source: 3,794 extracted docs, Koichi tokenizer windows (934,690 raw → 222,320 balanced train)
- 3 epochs, batch size 64, lr 2e-5, entity weight 10
- Sliding-window examples: 512 tokens, stride 256, 15B+15E overlap-aware
- Split: 89% / 1% / 10%

## Window-level test metrics (28,497 test windows)

| Metric | Koichi | Spsither baseline |
|--------|--------|-------------------|
| span F1 | 12.3% | 3.1% |
| title F1 | 15.8% | 7.4% |
| author F1 | 9.4% | 1.0% |

## Segment-level test metrics (6,683 segments, exact span match)

| Metric | Koichi | Spsither baseline |
|--------|--------|-------------------|
| span F1 | 7.4% | 8.0% |
| title F1 | **15.2%** | 12.7% |
| author F1 | 0.4% | 0.7% |

## Inference

Use the same sliding-window pipeline as training (15 begin + 15 end, stride 256), merge overlapping predictions, then extract spans.

```python
from pipeline.inference import load_model_and_tokenizer, predict_segment, highlight_spans

model, tokenizer, device = load_model_and_tokenizer("ganga4364/tibetan-metadata-koichi-ner")
spans = predict_segment(model, tokenizer, your_tibetan_text, device=device)
print(highlight_spans(your_tibetan_text, spans))
```

## Demo

Try the Gradio Space: [ganga4364/tibetan-metadata-highlight](https://huggingface.co/spaces/ganga4364/tibetan-metadata-highlight)
