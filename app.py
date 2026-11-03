from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Sequence

import gradio as gr


ROOT_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_LINES = 100


def load_model(filename: str) -> Any:
    model_path = ROOT_DIR / "models" / filename
    if not model_path.is_file():
        raise FileNotFoundError(f"Required model file not found: {model_path}")
    with model_path.open("rb") as model_file:
        return pickle.load(model_file)


SEGMENTATION_MODEL = load_model("mm_word_seg_crf.pkl")
POS_MODEL = load_model("crf_pos.pkl")


def character_features(chars: Sequence[str], index: int) -> dict[str, object]:
    if not 0 <= index < len(chars):
        raise IndexError(
            f"Character index {index} is outside a sequence of length {len(chars)}"
        )

    features: dict[str, object] = {"bias": 1.0, "char": chars[index]}
    for offset in (-3, -2, -1, 1, 2, 3):
        neighbor = index + offset
        if 0 <= neighbor < len(chars):
            features[f"char[{offset}]"] = chars[neighbor]
        elif neighbor < 0:
            features[f"char[{offset}]"] = "<BOS>"
        else:
            features[f"char[{offset}]"] = "<EOS>"

    for width in (2, 3, 4):
        first_start = max(0, index - width + 1)
        last_start = min(index, len(chars) - width)
        for start in range(first_start, last_start + 1):
            end = start + width
            features[f"ngram[{start - index}:{end - index - 1}]"] = "".join(
                chars[start:end]
            )

    if index == 0:
        features["BOS"] = True
    if index == len(chars) - 1:
        features["EOS"] = True
    return features


def segment_unspaced(text: str) -> str:
    chars = list(text)
    if not chars:
        return ""

    labels = SEGMENTATION_MODEL.predict_single(
        [character_features(chars, index) for index in range(len(chars))]
    )
    if len(labels) != len(chars):
        raise RuntimeError("Prediction length does not match input character count")

    output: list[str] = []
    for char, label in zip(chars, labels):
        if label == "B" and output:
            output.append(" ")
        output.append(char)
    return "".join(output)


def segment_text(text: str) -> str:
    pieces: list[str] = []
    span: list[str] = []
    for char in text:
        if char.isspace():
            if span:
                pieces.append(segment_unspaced("".join(span)))
                span.clear()
            pieces.append(char)
        else:
            span.append(char)
    if span:
        pieces.append(segment_unspaced("".join(span)))
    return "".join(pieces)


def pos_features(words: Sequence[str], index: int) -> list[str]:
    word_count = len(words)
    features: list[str] = []

    for offset in (-2, -1, 0, 1, 2):
        neighbor = index + offset
        if 0 <= neighbor < word_count:
            features.append(f"w[{offset}]={words[neighbor]}")

    if index - 1 >= 0:
        features.append(f"w[-1]|w[0]={words[index - 1]}|{words[index]}")
    if index + 1 < word_count:
        features.append(f"w[0]|w[1]={words[index]}|{words[index + 1]}")
    if index == 0:
        features.append("__BOS__")
    if index == word_count - 1:
        features.append("__EOS__")
    return features


def tag_segmented_text(text: str) -> str:
    tagged_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.endswith("\r\n"):
            content, line_ending = line[:-2], "\r\n"
        elif line.endswith(("\r", "\n")):
            content, line_ending = line[:-1], line[-1:]
        else:
            content, line_ending = line, ""

        words = content.split()
        if words:
            features = [pos_features(words, index) for index in range(len(words))]
            tags = POS_MODEL.predict([features])[0]
            if len(tags) != len(words):
                raise RuntimeError("POS prediction length does not match token count")
            tagged_lines.append(
                " ".join(f"{word}/{tag}" for word, tag in zip(words, tags))
                + line_ending
            )
        else:
            tagged_lines.append(line_ending)

    return "".join(tagged_lines)


def load_uploaded_text(file_path: str | None) -> str:
    if not file_path:
        return ""

    lines: list[str] = []
    try:
        with Path(file_path).open(encoding="utf-8-sig") as uploaded_file:
            for line in uploaded_file:
                if len(lines) >= MAX_UPLOAD_LINES:
                    raise gr.Error(
                        f"Uploaded files are limited to {MAX_UPLOAD_LINES} lines."
                    )
                lines.append(line)
    except UnicodeDecodeError as error:
        raise gr.Error("The uploaded file must be UTF-8 encoded.") from error
    except OSError as error:
        raise gr.Error(f"Could not read the uploaded file: {error}") from error

    return "".join(lines)


def create_download_file(text: str, prefix: str) -> str:
    file_descriptor, file_path = tempfile.mkstemp(prefix=prefix, suffix=".txt")
    with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as output_file:
        output_file.write(text)
    return file_path


def process_text(text: str) -> tuple[str, str, str, str]:
    segmented_text = segment_text(text)
    tagged_text = tag_segmented_text(segmented_text)
    segmented_file = create_download_file(segmented_text, "burmese_segmented_")
    tagged_file = create_download_file(tagged_text, "burmese_pos_tagged_")
    return segmented_text, tagged_text, segmented_file, tagged_file


with gr.Blocks(title="Burmese Word Segmentation and POS Tagging") as demo:
    gr.Markdown(
        "# Burmese Word Segmentation and POS Tagging\n"
        "Upload a UTF-8 `.txt` file (up to 100 lines) or enter Burmese text below."
    )

    with gr.Tab("Tagger"):
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                file_input = gr.File(
                    label="Upload text file (maximum 100 lines)",
                    file_types=[".txt"],
                    file_count="single",
                    type="filepath",
                    height=190,
                )
            with gr.Column(scale=1, min_width=320):
                input_text = gr.Textbox(
                    label="Burmese input text",
                    lines=6,
                    placeholder="Enter Burmese text or upload a .txt file",
                )
                process_button = gr.Button("Segment and POS tag", variant="primary")

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                segmented_output = gr.Textbox(
                    label="Segmented text",
                    lines=6,
                    buttons=["copy"],
                )
                segmented_download = gr.DownloadButton(
                    label="Download segmented text",
                    value=None,
                )
            with gr.Column(scale=1, min_width=320):
                tagged_output = gr.Textbox(
                    label="Segmented text with POS tags",
                    lines=6,
                    buttons=["copy"],
                )
                tagged_download = gr.DownloadButton(
                    label="Download segmented and tagged text",
                    value=None,
                )

        file_input.change(
            fn=load_uploaded_text,
            inputs=file_input,
            outputs=input_text,
        )
        process_button.click(
            fn=process_text,
            inputs=input_text,
            outputs=[
                segmented_output,
                tagged_output,
                segmented_download,
                tagged_download,
            ],
        )

    with gr.Tab("Model scores"):
        gr.Markdown(
            "Scores below are the recorded held-out test results from the project "
            "notebooks. The segmentation scores measure character-level boundaries; "
            "the POS scores measure word tags and exact sentence matches. The POS "
            "settings were selected using this same test set, so those results may "
            "be optimistic."
        )
        gr.Dataframe(
            headers=["Segmentation model metric", "Score"],
            datatype=["str", "number"],
            value=[
                ["Boundary precision", 0.9698],
                ["Boundary recall", 0.9779],
                ["Boundary F1", 0.9738],
                ["Character accuracy", 0.9874],
            ],
            type="array",
            interactive=False,
            label="Word Segmentator Model",
        )
        gr.Dataframe(
            headers=["POS model metric", "Score"],
            datatype=["str", "number"],
            value=[
                ["Macro-F1", 0.9976],
                ["Token accuracy", 0.9967],
                ["Exact-sentence accuracy", 0.959],
            ],
            type="array",
            interactive=False,
            label="Tagging Model",
        )


if __name__ == "__main__":
    demo.launch()
