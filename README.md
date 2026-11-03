# Burmese Word Segmentation and POS Tagging

This project experiments with Burmese (Myanmar) word segmentation and part-of-speech (POS) tagging. Burmese text does not consistently mark word boundaries with spaces, so the application first predicts word boundaries and then assigns a POS tag to each segmented word.

The project includes:

- A Gradio web application for entering Burmese text or uploading a UTF-8 `.txt` file.
- A character-feature CRF word segmenter and a feature-based CRF POS tagger used by the application.
- HMM, CRF, and BiLSTM-CRF POS experiments, plus a CRF segmentation experiment, in Jupyter notebooks.
- Pre-trained model files in `models/` and a script for downloading the corpus into `dataset/`.

The app's upload is limited to 100 lines. It displays segmented text and tagged text, with download buttons for both results. Model files are loaded from `models/` when `app.py` starts.

## POS tagset

The corpus uses the following 15 tags. Meanings and examples follow the [MY-POS corpus tagset](https://github.com/ye-kyaw-thu/myPOS#pos-tags).

| Tag | Meaning | Example |
|---|---|---|
| `abb` | Abbreviation | `အထက` |
| `adj` | Adjective | `ရဲရင့်` |
| `adv` | Adverb | `နည်းနည်း`  |
| `conj` | Conjunction | `နှင့်` |
| `fw` | Foreign word | `BBC`, `Google` |
| `int` | Interjection | `အမလေး` |
| `n` | Noun | `ကျောင်း`  |
| `num` | Number written with numeric characters | `၁`, `၁၀` |
| `part` | Particle, including grammatical and auxiliary particles | `များ`, `ခဲ့` |
| `ppm` | Post-positional marker | `က`, `ကို`, `မှာ` |
| `pron` | Pronoun | `ကျွန်တော်`, `သူ` |
| `punc` | Punctuation | `။`, `၊` |
| `sb` | Symbol | `#`, `+`, `÷` |
| `tn` | Number written as a word | `တစ်`, `နှစ်`, `တစ်ရာ` |
| `v` | Verb | `ကူညီ` |

The corpus distinguishes numeric characters (`num`) from number words (`tn`).

## Requirements

- Python 3.10 or later is recommended.
- The model files in `models/` must be present to run the application.
- The full dependency list is in [`requirements.txt`](requirements.txt).

## Install and run

Run these commands from this project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Gradio prints a local URL in the terminal (usually `http://127.0.0.1:7860`). Open it in a browser to use the tagger.

If PowerShell prevents environment activation, either allow activation for the current terminal according to your organization's policy, or run the virtual environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

### Download the corpus and run the notebooks

The corpus files are excluded from Git by `.gitignore`. From the project directory, download them with:

```powershell
python scripts\download_dataset.py
```

Then start Jupyter:

```powershell
jupyter lab
```

Open a notebook under `notebooks/`. The notebooks document data preparation, model experiments, and evaluation. Use a fresh validation split for model selection rather than tuning on the test corpus.

## Project analysis

### Recorded test results

The notebooks report the following results on the 1,000-sentence test corpus:

| Task / model | Metric | Recorded score |
|---|---|---:|
| Word segmentation CRF | Boundary precision | 96.98% |
| Word segmentation CRF | Boundary recall | 97.79% |
| Word segmentation CRF | Boundary F1 | 97.38% |
| Word segmentation CRF | Character accuracy | 98.74% |
| POS HMM baseline | Token accuracy / macro-F1 | 94.94% / 90.07% |
| POS HMM tuned | Token accuracy / macro-F1 | 95.37% / 91.43% |
| POS feature CRF baseline | Token accuracy / macro-F1 | 97.04% / 94.85% |
| POS feature CRF tuned | Token accuracy / macro-F1 | 99.67% / 99.76% |
| POS BiLSTM-CRF | Token accuracy / macro-F1 | 98.56% / 97.88% |

The application uses the CRF models. The reported tuned feature-CRF POS result is substantially higher than the baseline and the BiLSTM-CRF result in these notebook runs. However, the notebooks selected HMM/CRF hyperparameters using the same 1,000-sentence test corpus used to report the tuned scores. Those tuned results are therefore not independent estimates of generalization and should be treated as optimistic. A separate validation set and an untouched test set are needed for a fair comparison.

The training file contains 43,196 sentences and 564,517 tokens; the test file contains 1,000 sentences and 13,468 tokens. The tag frequencies are uneven (for example, nouns and particles are common while symbols and abbreviations are rare), so macro-F1 and per-tag results are important alongside overall token accuracy. Sentence-level accuracy is also useful because a sentence can contain an error even when most of its tokens are correct.

## Limitations

- **Low-resource coverage:** The corpus is modest relative to high-resource language datasets and may not cover enough vocabulary, topics, domains, dialects, or writing styles.
- **Segmentation ambiguity:** Burmese word boundaries are not always explicitly marked by spaces. Different valid tokenization choices can affect both segmentation and downstream POS scores.
- **Label imbalance and rare words:** Infrequent tags and out-of-vocabulary words have fewer examples, making their predictions less reliable.
- **Domain and orthographic variation:** Text from social media, modern usage, regional varieties, code-switching, or non-standard spelling may differ from the corpus.
- **Evaluation leakage risk:** The tuned HMM and CRF settings were selected on the reported test corpus. The scores should not be interpreted as unbiased final-test performance.
- **Pipeline error propagation:** Incorrect segmentation changes the tokens seen by the POS tagger and can cause downstream tagging errors.
- **Model scope:** The application uses the feature-based CRF models; the HMM and BiLSTM-CRF models are notebook experiments and are not selectable in the current UI.
- **Data licensing:** The MY-POS corpus is distributed under CC BY-NC-SA 4.0 according to its source repository. Check that license and its conditions before redistributing or using the downloaded corpus. The repository's MIT license does not replace the corpus license.

## References

1. Ye Kyaw Thu et al., [MY-POS: Myanmar Part-of-Speech Corpus](https://github.com/ye-kyaw-thu/myPOS), including the version 3.0 corpus and tag definitions.
2. John Lafferty, Andrew McCallum, and Fernando Pereira. [“Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data.”](https://repository.upenn.edu/cis_papers/159/) ICML, 2001.
3. Zhiheng Huang, Wei Xu, and Kai Yu. [“Bidirectional LSTM-CRF Models for Sequence Tagging.”](https://arxiv.org/abs/1508.01991), 2015.
4. [Gradio documentation](https://www.gradio.app/docs) for the application interface.
