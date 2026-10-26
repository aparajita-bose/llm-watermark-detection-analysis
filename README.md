# LLM Watermark Detection Analysis

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aparajita-bose/llm-watermark-detection-analysis/blob/main/notebooks/watermark_detection_analysis.ipynb)

[Read the accompanying report (PDF)](report.pdf)

I built this project to explore whether a text watermark can be detected in summaries produced by `google-t5/t5-small`. I applied the [Kirchenbauer et al. (2023)](https://proceedings.mlr.press/v202/kirchenbauer23a.html) watermark implementation during generation, then compared the detector's predictions on original news articles and watermarked summaries from CNN/DailyMail and XSum.

> **Environment note:** I developed and ran the notebook in Google Colab (Python 3.11; T4 GPU selected in its metadata). I have not verified local execution, so I cannot guarantee that it will work outside Colab.

## Research objective and background

Text watermarks are meant to make generated text statistically identifiable without noticeably changing it. I wanted to see how the reference green-list detector behaves with an encoder-decoder summarization model, using the original news documents as non-watermarked controls and the generated summaries as watermarked examples.

## Watermarking algorithm and model

I used the official [`jwkirchenbauer/lm-watermarking`](https://github.com/jwkirchenbauer/lm-watermarking) implementation, specifically its `WatermarkLogitsProcessor` and `WatermarkDetector`. During generation, the logits processor promotes a seeded subset of vocabulary tokens; the detector then applies the corresponding statistical test. Here is the configuration used in the notebook:

| Setting | Value |
| --- | --- |
| Model | `google-t5/t5-small` |
| Watermark parameters | `gamma=0.25`, `delta=2.0`, `seeding_scheme="simple_1"` |
| Detection threshold | z-score `4.0` |
| Token selection | `select_green_tokens=True` |
| Generation | `max_length=300`, `num_beams=1` |

I load T5 through Hugging Face Transformers; see the [T5 Small model card](https://huggingface.co/google-t5/t5-small) for the checkpoint.

## Datasets and experimental setup

I use the training split of two summarization datasets:

- **CNN/DailyMail** via `load_dataset("cnn_dailymail", "3.0.0")`. The notebook evaluates the first 200 training articles.
- **XSum** via `load_dataset("xsum")`. The notebook shuffles the training split with seed 70 and selects 200 documents.

For each input document, I run the detector on the original article or document (label 0) and on a watermark-processed T5 summary (label 1). That gives 200 controls and 200 generated summaries for each dataset in the reported results.

## Methodology and evaluation

I combine the two classes for each dataset and report a confusion matrix, precision, recall, F1 score, and accuracy with scikit-learn. “Watermarked” is the positive class. I did not evaluate summary quality or watermark robustness in this notebook.

## Results

These are the results saved in the notebook. I did not rerun the experiment while preparing this repository.

| Dataset | Accuracy | Non-watermarked F1 | Watermarked precision | Watermarked recall | Watermarked F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN/DailyMail | 0.66 | 0.74 | 1.00 | 0.31 | 0.47 |
| XSum | 0.62 | 0.73 | 1.00 | 0.24 | 0.39 |

### Confusion matrices

Rows are ground-truth labels and columns are detector predictions. The counts come from the notebook's saved prediction outputs.

**CNN/DailyMail**

| Actual \ Predicted | Non-watermarked | Watermarked |
| --- | ---: | ---: |
| Non-watermarked | 200 | 0 |
| Watermarked | 138 | 62 |

**XSum**

| Actual \ Predicted | Non-watermarked | Watermarked |
| --- | ---: | ---: |
| Non-watermarked | 200 | 0 |
| Watermarked | 151 | 49 |

The [notebook](notebooks/watermark_detection_analysis.ipynb) also includes the rendered confusion matrices and accuracy comparison.

## Key findings and limitations

- In these runs, the detector made no false-positive watermark predictions on the 200 source documents in either dataset.
- The detector found only 31% of watermarked CNN/DailyMail summaries and 24% of watermarked XSum summaries. In other words, the reported accuracy partly reflects correctly identifying controls rather than reliably recovering the watermark from summaries.
- I see this as a small exploratory study, not a broad benchmark: it uses training examples, one model checkpoint, one decoding setup, and 200 examples per class. It does not tell us how the method performs after paraphrasing or editing, with other decoding settings, or on other models.
- I did not set the model to evaluation mode, seed generation, move model inputs to the selected device, or record package and implementation revisions. Those are reproducibility limitations of the original notebook.
- The notebook clones the watermark repository at its then-current default revision, so the exact workflow could change as that dependency evolves.

## Repository structure

```text
.
├── LICENSE
├── README.md
├── requirements.txt
└── notebooks/
    └── watermark_detection_analysis.ipynb
```

## Run in Google Colab

1. Select the **Open in Colab** badge above, or open [the notebook](notebooks/watermark_detection_analysis.ipynb) on GitHub and use Colab.
2. Choose a GPU runtime if available; the original notebook metadata selects a T4 GPU.
3. Run the setup cells. They install `torchvision`, `scikit-learn`, and `datasets`, then clone `jwkirchenbauer/lm-watermarking` into the Colab working directory.
4. Run the remaining cells to download the model and datasets and reproduce the workflow.

The notebook imports `transformers`, `torch`, and `matplotlib` without installing them explicitly, relying on the Colab environment. It also installs `torchvision`, although it does not import it. Running the cells downloads external datasets and model weights and performs generation; I did not run them while updating this repository.

## Dependencies

I listed the notebook's direct Python imports in `requirements.txt`. I left versions unpinned because the notebook records its Colab Python version but not compatible versions for these packages. I have not tested these requirements locally. The notebook fetches the watermark implementation with its Git clone cell rather than through `requirements.txt`.

## References and acknowledgments

- Kirchenbauer, J. et al. (2023). [*A Watermark for Large Language Models*](https://proceedings.mlr.press/v202/kirchenbauer23a.html). ICML / PMLR 202. I used the authors' [official implementation](https://github.com/jwkirchenbauer/lm-watermarking).
- Raffel, C. et al. (2020). [*Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*](https://www.jmlr.org/papers/v21/20-074.html). The experiment uses the [T5 Small checkpoint](https://huggingface.co/google-t5/t5-small).
- Hermann, K. M. et al. (2015). [*Teaching Machines to Read and Comprehend*](https://arxiv.org/abs/1506.03340). CNN/DailyMail is loaded through the [Hugging Face dataset repository](https://huggingface.co/datasets/abisee/cnn_dailymail).
- Narayan, S., Cohen, S. B., & Lapata, M. (2018). [*Don't Give Me the Details, Just the Summary!*](https://aclanthology.org/D18-1206/). XSum is available on [Hugging Face](https://huggingface.co/datasets/EdinburghNLP/xsum).

## License

This project is released under the [MIT License](LICENSE).
