# Phase 8 - Model Evaluation

Implemented in `src/models/evaluate.py`, `src/analysis/error_analysis.py` and `model_analysis.py`;
run with `python scripts/evaluate_model.py`. The model was evaluated **once** on **1,161 unseen articles**.

## Metrics (Linear SVM)
| Metric | Value |
|---|---|
| Accuracy | 97.42% (95% confidence interval 96.34% to 98.18%) |
| Precision (REAL = positive class) | 97.02% |
| Recall | 98.00% |
| F1-score | 97.51% |
| Macro F1 | 97.41% |
| ROC-AUC | 0.9969 |
| Training accuracy | 100.00% |
| Majority-class baseline accuracy | 51.59% |
| p-value: model vs baseline (one-sided binomial test) | < 1e-100 |

## Confusion matrix (rows = actual, columns = predicted)
|  | Predicted FAKE | Predicted REAL |
|---|---|---|
| **Actual FAKE** | 544 | 18 |
| **Actual REAL** | 12 | 587 |

## Classification report
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| FAKE | 0.9784 | 0.9680 | 0.9732 | 562 |
| REAL | 0.9702 | 0.9800 | 0.9751 | 599 |

## Sanity checks
* **Label-shuffle control:** the same model trained on randomly shuffled labels scores [0.5159, 0.5159, 0.5159] on the test set, which is just the baseline. The real model's score therefore comes from patterns, not from the pipeline.
* **Overfitting:** training accuracy 100.00% versus test accuracy 97.42% (gap 2.58%).
* **Most influential terms towards FAKE:** video, president trump, just, gop, image, read, featured image, featured, hillary, breaking
* **Most influential terms towards REAL:** president donald, wednesday, republican, thursday, tuesday, said statement, friday, monday, presidential, sunday

## Error analysis
30 of 1,161 test articles were misclassified (2.58%):
18 fake articles called real, 12 real articles called fake.
Mean confidence was 97.81% when correct and 74.94% when wrong,
so low-confidence predictions deserve extra caution. Error rate by article length: {'short': 0.0257, 'medium': 0.0258, 'long': 0.026}.
Error rate by subject (used here for analysis only): {'Government News': 0.1143, 'Middle-east': 0.0, 'News': 0.0043, 'US_News': 0.0, 'left-news': 0.0465, 'politics': 0.0539, 'politicsNews': 0.0308, 'worldnews': 0.0098}.

## Interpretation and limits
The test articles come from the **same dataset** as the training articles. The score therefore measures performance on
*similar* news and is probably higher than what would be seen on articles from other publishers, countries or years
(see document 09). Some influential words (for example "featured image", "getty images", "video") are habits of the
particular fake-news sources in this dataset rather than universal signs of fakeness.
