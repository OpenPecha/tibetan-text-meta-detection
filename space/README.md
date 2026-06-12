---
title: Tibetan Metadata Highlight
emoji: 📜
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
python_version: 3.10
short_description: Highlight title and author spans in Tibetan text segments
---

# Tibetan Metadata Highlight

Detect **title** and **author** metadata in Tibetan text using a fine-tuned RoBERTa model
with the same sliding-window inference pipeline as training.

- **Title** spans are highlighted in gold
- **Author** spans are highlighted in blue

Model: [ganga4364/tibetan-metadata-roberta-ner](https://huggingface.co/ganga4364/tibetan-metadata-roberta-ner)
