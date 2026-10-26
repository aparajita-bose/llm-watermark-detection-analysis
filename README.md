# LLM Watermark Detection Analysis

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aparajita-bose/llm-watermark-detection-analysis/blob/main/notebooks/watermark_detection_analysis.ipynb)

An exploratory analysis of watermark detection for summaries generated with `google-t5/t5-small`. The experiment applies the [Kirchenbauer et al. (2023)](https://proceedings.mlr.press/v202/kirchenbauer23a.html) watermark implementation during generation, then tests whether its detector distinguishes source news articles from watermarked model summaries on CNN/DailyMail and XSum.

> **Environment note:** The notebook was developed and executed in Google Colab (Python 3.11; T4 GPU selected in its metadata). Local execution has not been verified and is not guaranteed to work.

## Research objective and background

Text watermarks aim to make generated text statistically identifiable without visibly changing it. This project examines how the reference green-list watermark detector behaves when watermarking an encoder-decoder summarization model, using original news documents as non-watermarked controls and generated summaries as watermarked examples.

## Watermarking algorithm and model

The notebook clones the official [`jwkirchenbauer/lm-watermarking`](https://github.com/jwkirchenbauer/lm-watermarking) repository and imports its `WatermarkLogitsProcessor` and `WatermarkDetector`. At each generation step, the logits processor promotes a seeded subset of vocabulary tokens; the detector uses the corresponding statistical test. The executed configuration is:

| Setting | Value |
| --- | --- |
| Model | `google-t5/t5-small` |
| Watermark parameters | `gamma=0.25`, `delta=2.0`, `seeding_scheme="simple_1"` |
| Detection threshold | z-score `4.0` |
| Token selection | `select_green_tokens=True` |
| Generation | `max_length=300`, `num_beams=1` |

The notebook loads T5 through Hugging Face Transformers; see the [T5 Small model card](https://huggingface.co/google-t5/t5-small) for the checkpoint.

## Datasets and experimental setup

The analysis uses the training split of two summarization datasets:

- **CNN/DailyMail** via `load_dataset("cnn_dailymail", "3.0.0")`. The notebook evaluates the first 200 training articles.
- **XSum** via `load_dataset("xsum")`. The notebook shuffles the training split with seed 70 and selects 200 documents.

For each input document, the detector is applied to the original article/document (label 0) and to a watermark-processed T5 summary (label 1). Each dataset therefore contributes 200 controls and 200 generated summaries to the reported classification results.

## Methodology and evaluation

The notebook combines both classes per dataset and reports a confusion matrix, precision, recall, F1 score, and accuracy using scikit-learn. “Watermarked” is the positive class. No summary-quality metric or robustness test is performed.

## Results

These values are transcribed from the saved notebook outputs; no experiment was rerun for this repository update.

| Dataset | Accuracy | Non-watermarked F1 | Watermarked precision | Watermarked recall | Watermarked F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN/DailyMail | 0.66 | 0.74 | 1.00 | 0.31 | 0.47 |
| XSum | 0.62 | 0.73 | 1.00 | 0.24 | 0.39 |

### Confusion matrices

Rows are ground-truth labels and columns are detector predictions. Counts are transcribed from the notebook's saved prediction outputs.

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

The preserved rendered figures, including the accuracy comparison, remain available in the [notebook](notebooks/watermark_detection_analysis.ipynb).

## Key findings and limitations

- In the recorded runs, the detector made no false-positive watermark predictions on the 200 source documents in either dataset.
- It detected only 31% of watermarked CNN/DailyMail summaries and 24% of watermarked XSum summaries. The observed accuracy is therefore driven partly by correctly classifying controls, rather than strong recovery of the watermark in summaries.
- This is a small, single-configuration exploratory evaluation: it uses training examples, one model checkpoint, one decoding configuration, and 200 examples per class. It does not establish performance under paraphrasing, editing, alternative decoding settings, or other models.
- The notebook does not set model evaluation mode, seed generation, move model inputs to the selected device, or record package/implementation revisions. These are reproducibility limitations, not changes made here.
- The notebook clones the watermark repository at its then-current default revision, so an exact rerun may differ as that dependency evolves.

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

The notebook imports `transformers`, `torch`, and `matplotlib` without installing them explicitly, relying on the Colab environment. It also installs `torchvision`, although the notebook does not import it. Running cells downloads external datasets/model weights and performs generation; this repository update did not run them.

## Dependencies

`requirements.txt` lists the direct Python imports used by the notebook. Versions are intentionally not pinned because the notebook only records its Colab Python version and does not record compatible versions for these packages. The requirements have not been tested locally. The watermark implementation itself is fetched by the notebook’s Git clone cell rather than installed from `requirements.txt`.

## References and acknowledgments

- Kirchenbauer, J. et al. (2023). [*A Watermark for Large Language Models*](https://proceedings.mlr.press/v202/kirchenbauer23a.html). ICML / PMLR 202. This project uses the authors’ [official implementation](https://github.com/jwkirchenbauer/lm-watermarking).
- Raffel, C. et al. (2020). [*Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*](https://www.jmlr.org/papers/v21/20-074.html). The experiment uses the [T5 Small checkpoint](https://huggingface.co/google-t5/t5-small).
- Hermann, K. M. et al. (2015). [*Teaching Machines to Read and Comprehend*](https://arxiv.org/abs/1506.03340). CNN/DailyMail is loaded through the [Hugging Face dataset repository](https://huggingface.co/datasets/abisee/cnn_dailymail).
- Narayan, S., Cohen, S. B., & Lapata, M. (2018). [*Don’t Give Me the Details, Just the Summary!*](https://aclanthology.org/D18-1206/). XSum is available on [Hugging Face](https://huggingface.co/datasets/EdinburghNLP/xsum).

## License

This project is released under the [MIT License](LICENSE).
