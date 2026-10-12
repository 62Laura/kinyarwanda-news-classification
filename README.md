# Kinyarwanda News Topic Classification

Classifying Kinyarwanda news articles into 14 topics with three models (TF-IDF + Logistic Regression, BiLSTM, fine-tuned KinyaBERT-small) and a deployed web app.

| Resource | Link |
|---|---|
| Live app | https://kinyarwanda-news-classification-hhzv4dtv6xbesu9vkfgckr.streamlit.app/ |
| Fine-tuned model (Hugging Face) | https://huggingface.co/lauraCeline/kinyarwanda-news-classification |

## 1. Problem

Kinyarwanda is the national language of Rwanda, yet it has far fewer NLP tools than high-resource languages. News topic classification is a practical first task: it can power search, recommendation and archiving for Rwandan media. The goal of this project is to build and compare classifiers for Kinyarwanda news, analyse where they fail, and deploy the best-suited model so that anyone can try it.

## 2. Dataset

**KINNEWS** (Niyongabo et al., 2020): 21,268 Kinyarwanda news articles collected from Rwandan news websites and newspapers, labelled with 14 topics: politics, sport, economy, health, entertainment, history, technology, tourism, culture, fashion, religion, environment, education, relationship.

- Version used: the `cleaned` CSVs (special characters, URLs and stop words removed by the dataset authors), downloaded from `https://github.com/saradhix/kinnews_kirnews`.
- Original split: 17,014 train and 4,254 test. Five training rows with empty text were dropped, leaving 17,009.
- Our split (fixed, `random_state=42`): train 14,457, validation 2,552 (15% of train, stratified), test 4,254 (the original test set, used once per model).
- Model input for every model: lower-cased title followed by the cleaned article text.
- Class imbalance is strong: 3,929 politics articles against 152 education articles in train+validation.

**Duplicates warning.** Many articles appear more than once in this copy of KINNEWS. Only 7,867 of the 14,457 training texts are unique, and 2,230 of the 4,254 test articles (52.4%) have an exact text match in the training or validation data. Scores on the full test set are therefore optimistic. The report gives both the full-test scores and scores on the 2,024 test articles with no exact copy in training or validation.

## 3. Methods

| Model | Description |
|---|---|
| TF-IDF + Logistic Regression (baseline) | 100,000 uni- and bigram features, sublinear TF, `min_df=2`; Logistic Regression with balanced class weights; `C` chosen on validation macro-F1 from {0.5, 1, 5, 10, 20} (best: 20). |
| BiLSTM | PyTorch; 50k vocabulary, 200-d embeddings trained from scratch, 128-unit bidirectional LSTM, mean and max pooling, class-weighted cross-entropy, Adam (1e-3), early stopping on validation macro-F1 (best epoch 12), 10.3M parameters. |
| KinyaBERT-small (fine-tuned) | `jean-paul/KinyaBERT-small`, a community 6-layer BERT pretrained on Kinyarwanda text, fine-tuned for 14-way classification. AdamW, learning rate 5e-5, batch 32, max length 256, fp16, 10% warm-up; best epoch 3; 66.6M parameters. |

Note: this community model is a plain BERT. It is not the morphology-aware KinyaBERT of Nzeyimana and Rubungo (2022).

For deployment the fine-tuned model was exported to ONNX and int8-quantized (266.4 MB to 67.0 MB) so it runs on a free CPU server without PyTorch.

## 4. Results

Test set (4,254 articles). The primary metric is macro-F1 because the classes are imbalanced.

| Model | Accuracy | Macro-F1 | Precision (macro) | Recall (macro) | Train time |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.8902 | 0.8308 | 0.8361 | 0.8290 | 62 s |
| BiLSTM | 0.8366 | 0.7299 | 0.7068 | 0.7812 | 213 s |
| KinyaBERT-small (fine-tuned) | 0.8778 | 0.8052 | 0.7998 | 0.8149 | 530 s |

Only the 2,024 test articles with no exact copy in train or validation:

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 0.7994 | 0.7315 |
| BiLSTM | 0.6927 | 0.5868 |
| KinyaBERT-small (fine-tuned) | 0.7722 | 0.6926 |

Main findings:

- The simple TF-IDF baseline scored highest, with and without the duplicate articles. The more complex models did not beat it.
- Scores drop by about 10 macro-F1 points once duplicates are excluded, so the full-test numbers overstate how well the models handle unseen articles.
- The hardest classes are history, education and environment (small classes, with 34 to 65 test articles each). Common confusions are economy, politics and health.
- int8 quantization changed accuracy on a 500-article sample from 0.864 to 0.862 (macro-F1 0.8071 to 0.8034), with 99.2% prediction agreement.

Details, error analysis and discussion are in the report.

## 5. Repository structure

```
notebooks/            01 to 05, run in this order (Google Colab)
data/processed/       train.csv, val.csv, test.csv (the fixed split)
figures/              EDA and training-curve figures
results/              metrics_summary.csv, confusion matrices, error files, top words
streamlit_app/        the deployed app (streamlit_app.py, onnx_classifier.py, examples.txt, requirements.txt)
.streamlit/           theme configuration
```

## 6. How to reproduce

The notebooks were written for Google Colab. Each one clones this repository, does its work and pushes its outputs back.

1. `01_data_exploration.ipynb` downloads KINNEWS, explores it and writes the fixed split to `data/processed/`.
2. `02_baseline_tfidf_logreg.ipynb` trains the baseline.
3. `03_bilstm.ipynb` trains the BiLSTM (use a GPU runtime).
4. `04_kinyabert_finetune.ipynb` fine-tunes KinyaBERT-small (use a GPU runtime) and can upload the model to the Hugging Face Hub.
5. `05_deploy_app.ipynb` exports the model to ONNX, quantizes it, checks that predictions still match, uploads the file to the Hub and writes the Streamlit app.

You need a GitHub personal access token with `repo` scope, and for steps 4 and 5 a Hugging Face token with write access. Notebooks ask for tokens at run time and never store them.

To run the app locally:

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/streamlit_app.py
```

The app downloads the quantized model from `lauraCeline/kinyarwanda-news-classification` on first start. Set the `MODEL_ID` environment variable to use a different Hugging Face repository or a local folder.

## 7. Limitations

- Duplicate articles in the dataset (see Section 2) inflate full-test scores, and the validation set used for tuning overlaps with training.
- Results come from one split and one training run per model; no confidence intervals are reported.
- The deployed model is KinyaBERT-small, which scored below the TF-IDF baseline on this test set.
- The model works on news text and always returns one of the 14 topics, even for text that fits none.
- Whether the pretraining data of KinyaBERT-small overlaps with the KINNEWS test articles could not be checked.

## 8. Existing resources and own work

Existing resources used: the KINNEWS dataset, the `jean-paul/KinyaBERT-small` model, and the libraries scikit-learn, PyTorch, Hugging Face Transformers and Tokenizers, ONNX Runtime and Streamlit.

Implemented in this project: data cleaning and the fixed split, the three classification pipelines, hyper-parameter selection, evaluation and error analysis, the ONNX export and quantization, and the web app.

## 9. References

- Niyongabo, R. A., Hong, Q., Kreutzer, J., and Huang, L. (2020). KINNEWS and KIRNEWS: Benchmarking cross-lingual text classification for Kinyarwanda and Kirundi. *COLING 2020*, 5507-5521. https://aclanthology.org/2020.coling-main.480
- Nzeyimana, A. and Rubungo, A. N. (2022). KinyaBERT: a morphology-aware Kinyarwanda language model. *ACL 2022*, 5347-5363. https://aclanthology.org/2022.acl-long.367
- Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT 2019*, 4171-4186.
- Model card: https://huggingface.co/jean-paul/KinyaBERT-small

The full reference list is in the report.
