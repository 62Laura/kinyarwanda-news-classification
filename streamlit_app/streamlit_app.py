import html
import os

import pandas as pd
import streamlit as st

from onnx_classifier import Classifier

GITHUB_URL = "https://github.com/62Laura/kinyarwanda-news-classification"
MODEL_URL = "https://huggingface.co/lauraCeline/kinyarwanda-news-classification"

# Test-set scores from results/metrics_summary.csv (KINNEWS held-out test set)
RESULTS = pd.DataFrame(
    [
        ("TF-IDF + Logistic Regression", 0.8902, 0.8308, 0.8361, 0.8290),
        ("BiLSTM", 0.8366, 0.7299, 0.7068, 0.7812),
        ("KinyaBERT-small (fine-tuned)", 0.8778, 0.8052, 0.7998, 0.8149),
    ],
    columns=["Model", "Accuracy", "Macro-F1", "Precision (macro)", "Recall (macro)"],
)

st.set_page_config(
    page_title="Kinyarwanda News Topic Classifier",
    page_icon=os.path.join(os.path.dirname(os.path.abspath(__file__)), "favicon.png"),
    layout="wide",
)

CSS = """
<style>
html, body, [class*="css"], .stMarkdown, button, textarea, input { font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 4.2rem; max-width: 1150px; }

.hero { padding: 2.1rem 2rem 1.7rem 2rem; border-radius: 6px; margin-bottom: 1.4rem; background: #14452D; color: #FBF8F1; }
.stripe { display: flex; height: 6px; margin: -2.1rem -2rem 1.6rem -2rem; border-radius: 6px 6px 0 0; overflow: hidden; }
.stripe span { flex: 1; }
.hero h1 { font-family: Georgia, 'Times New Roman', serif; font-weight: 400; font-size: 2.4rem; margin: 0 0 0.5rem 0; color: #FBF8F1; }
.hero p { margin: 0; font-size: 1.05rem; max-width: 46rem; color: #E6EFE8; line-height: 1.5; }
.tag { display: inline-block; font-size: 0.78rem; letter-spacing: .05em; text-transform: uppercase;
       padding: 2px 8px; border-radius: 3px; border: 1px solid rgba(251,248,241,.45); margin-bottom: .8rem; }

.card { background: #FFFFFF; border: 1px solid #D9D1BC; border-radius: 6px; padding: 1.3rem 1.4rem; }
.card.empty { background: #F6F1E4; border-style: dashed; color: #5C6B61; text-align: center; padding: 2.6rem 1.4rem; }
.label-small { font-size: .78rem; letter-spacing: .06em; text-transform: uppercase; color: #5C6B61; margin-bottom: 2px; }
.top-label { font-family: Georgia, 'Times New Roman', serif; font-size: 2.5rem; line-height: 1.1; color: #14452D; margin: 0; text-transform: capitalize; }
.top-conf { font-size: 1.05rem; color: #3E5246; margin: 2px 0 14px 0; }
.bar-row { display: flex; align-items: center; margin: 7px 0; font-size: .95rem; }
.bar-name { width: 7.8rem; text-transform: capitalize; color: #2B3A31; }
.bar-lbl { flex: 0 0 330px; font-size: .88rem; padding-right: 10px; color: #2B3A31; }
.bar-lbl small { color: #5C6B61; }
.bar-track { flex: 1; height: 10px; border-radius: 2px; background: #EFE9D9; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 2px; }
.bar-pct { width: 3.6rem; text-align: right; color: #4B5D52; font-variant-numeric: tabular-nums; }
.warn { margin-top: 12px; padding: 9px 12px; border-radius: 4px; background: #FFF4D6; color: #5E4400; font-size: .9rem; }
@media (max-width: 700px) { .bar-lbl { flex-basis: 130px; font-size: .8rem; } }

.chip { display: inline-block; padding: 3px 9px; margin: 3px 4px 3px 0; border-radius: 3px; font-size: .88rem;
        background: #F1ECE0; border: 1px solid #D9D1BC; color: #2B3A31; text-transform: capitalize; }
.stat { background: #FFFFFF; border: 1px solid #D9D1BC; border-radius: 6px; padding: 1rem 1.2rem; }
.stat .n { font-family: Georgia, 'Times New Roman', serif; font-size: 2rem; color: #14452D; line-height: 1.1; }
.stat .t { font-size: .85rem; color: #5C6B61; }

div.stButton > button[kind="primary"] { background: #1E6B45; border: 0; border-radius: 4px; padding: .55rem 1.6rem; font-weight: 700; }
div.stButton > button[kind="primary"]:hover { background: #14452D; }
.stTabs [data-baseweb="tab"] { font-weight: 700; }
textarea { border-radius: 4px !important; }
.legal h2 { font-family: Georgia, 'Times New Roman', serif; font-weight: 400; margin-top: 1.6rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the model (first visit only)...")
def get_classifier():
    return Classifier()


def load_examples():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples.txt")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def fill_example():
    idx = st.session_state.get("example_choice")
    if idx is not None:
        st.session_state["text"] = load_examples()[idx]


def result_card(probs):
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    top, p = ranked[0]
    rows = ""
    for i, (label, q) in enumerate(ranked[:5]):
        colour = "#1E6B45" if i == 0 else "#7FB3CF"
        rows += (
            f'<div class="bar-row"><div class="bar-name">{html.escape(label)}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{q * 100:.1f}%;background:{colour}"></div></div>'
            f'<div class="bar-pct">{q:.1%}</div></div>'
        )
    warn = ""
    if p < 0.5:
        warn = ('<div class="warn">Low confidence: the model is unsure. The text may be short, ambiguous, '
                "or unlike typical news writing.</div>")
    return (
        '<div class="card"><div class="label-small">Predicted topic</div>'
        f'<p class="top-label">{html.escape(top)}</p>'
        f'<div class="top-conf">{p:.1%} confidence</div>'
        '<div class="label-small" style="margin-bottom:6px">Top 5 topics</div>'
        f"{rows}{warn}</div>"
    )


def classifier_page():
    clf = get_classifier()
    EXAMPLES = load_examples()
    st.markdown(
        '<div class="hero"><div class="stripe"><span style="background:#0E7FB8"></span><span style="background:#E5B800"></span>'
        '<span style="background:#20603A"></span></div>'
        '<span class="tag">Kinyarwanda &middot; 14 topics</span>'
        "<h1>Kinyarwanda News Topic Classifier</h1>"
        "<p>Paste a Kinyarwanda news article or headline. The model assigns it to one of 14 topics and shows how confident it is.</p></div>",
        unsafe_allow_html=True,
    )

    tab_try, tab_results, tab_about = st.tabs(["Try it", "How well does it work?", "About"])

    with tab_try:
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            if EXAMPLES:
                st.selectbox(
                    "Start from a real article from the test set",
                    options=list(range(len(EXAMPLES))),
                    index=None,
                    format_func=lambda i: f"Example {i + 1}: " + " ".join(EXAMPLES[i].split()[:7]) + "...",
                    placeholder="Choose an example (optional)",
                    key="example_choice",
                    on_change=fill_example,
                )
            text = st.text_area("Kinyarwanda news text", key="text", height=230,
                                placeholder="Paste or type Kinyarwanda news text here...")
            go = st.button("Classify", type="primary")
        with right:
            if go:
                if len(text.split()) < 3:
                    st.warning("Please enter at least a few words of Kinyarwanda news text.")
                else:
                    st.markdown(result_card(clf.predict_proba(text)), unsafe_allow_html=True)
            else:
                st.markdown('<div class="card empty">The predicted topic and the model\'s confidence appear here.</div>',
                            unsafe_allow_html=True)
        st.markdown('<div class="label-small" style="margin-top:1.4rem">The 14 topics it can predict</div>'
                    + "".join(f'<span class="chip">{html.escape(l)}</span>' for l in clf.labels),
                    unsafe_allow_html=True)

    with tab_results:
        st.markdown("Three models were trained on the same data split and scored on the same held-out test set. "
                    "**Macro-F1** averages the F1 score of the 14 topics equally, so small topics count as much as big ones.")
        best = RESULTS.loc[RESULTS["Macro-F1"].idxmax()]
        c1, c2, c3 = st.columns(3)
        for col, (_, r) in zip((c1, c2, c3), RESULTS.iterrows()):
            col.markdown(f'<div class="stat"><div class="n">{r["Macro-F1"]:.3f}</div>'
                         f'<div class="t">{html.escape(r["Model"])}<br>macro-F1</div></div>', unsafe_allow_html=True)
        st.write("")
        st.dataframe(RESULTS.style.format({c: "{:.4f}" for c in RESULTS.columns[1:]}), hide_index=True)
        rows = ""
        for _, r in RESULTS.iterrows():
            for metric, color in (("Macro-F1", "#1E6B45"), ("Accuracy", "#7FB3CF")):
                v = float(r[metric])
                rows += (f'<div class="bar-row"><div class="bar-lbl">{html.escape(r["Model"])} <small>{metric}</small></div>'
                         f'<div class="bar-track"><div class="bar-fill" style="width:{v*100:.1f}%;background:{color}"></div></div>'
                         f'<div style="width:52px;text-align:right">{v:.3f}</div></div>')
        st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)
        st.caption(f"Best macro-F1 on this test set: {best['Model']} ({best['Macro-F1']:.3f}). "
                   "This app serves the fine-tuned KinyaBERT-small model.")

    with tab_about:
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("#### Data")
            st.markdown("The **KINNEWS** dataset: Kinyarwanda news articles, each labelled with one of 14 topics "
                        "(Niyongabo et al., 2020). Articles were cleaned, split into train, validation and test sets, "
                        "and used to train the three models.")
            st.markdown("#### Limits")
            st.markdown("- Trained on news text only: less reliable on social media, chat or very short phrases.\n"
                        "- It always picks one of the 14 topics, even for text that fits none.\n"
                        "- Long articles are cut at 256 word pieces.")
        with b:
            st.markdown("#### Model")
            st.markdown("**KinyaBERT-small**, a 6-layer BERT pretrained on Kinyarwanda text, fine-tuned to classify topics. "
                        "For this app it is converted to ONNX and int8-quantized, shrinking it from 266 MB to 67 MB "
                        "so it runs on a free server without PyTorch.")
            st.markdown("#### Links")
            st.markdown(f"- [Source code and experiments on GitHub]({GITHUB_URL})\n- [Fine-tuned model on Hugging Face]({MODEL_URL})")


LAST_UPDATED = "12 October 2026"
ISSUES_URL = GITHUB_URL + "/issues"


def privacy_page():
    st.markdown(f"""
<div class="legal">

## Privacy Policy

Last updated: {LAST_UPDATED}

This page explains what happens to the text you enter in the Kinyarwanda News Topic Classifier. The app is a student
project built to demonstrate a text classification model. It has no accounts, no sign-in and no payments.

## What the app processes

When you press Classify, the text you typed or pasted is sent to the app's server, converted into a prediction by the
model, and the result is shown back to you. The app does not write your text to a database or a file, and it does not
keep it after the prediction is returned. The app does not ask for your name, email address or any other personal detail.

Please do not enter personal, confidential or sensitive information. The model is designed for public news text.

## What the app does not do

The app does not use advertising, does not build visitor profiles and does not add analytics or tracking scripts of its own.
The app's own pages do not load fonts or scripts from third-party services.

## Hosting provider

The app is hosted on Streamlit Community Cloud. Like any web host, the hosting platform may process technical information
such as your IP address and the pages requested, in order to deliver and secure the service. That processing is governed by the
provider's own notice: [Streamlit privacy policy](https://streamlit.io/privacy-policy).

## Your choices and contact

Because the app does not store what you enter, there is no stored content to access or delete. If you have a question about
this policy, open an issue at [the project repository]({ISSUES_URL}).

This policy may be updated as the project changes. The date at the top shows the latest revision.

</div>
""", unsafe_allow_html=True)


def terms_page():
    st.markdown(f"""
<div class="legal">

## Terms and Conditions

Last updated: {LAST_UPDATED}

By using the Kinyarwanda News Topic Classifier you agree to these terms. If you do not agree, please do not use the app.

## Purpose

The app is an educational demonstration of a machine learning model that assigns Kinyarwanda news text to one of 14 topics.
It is not a commercial service.

## No guarantee of accuracy

Predictions are produced by a statistical model and can be wrong. On the held-out test set the model served here reached a
macro-F1 of 0.805, which means a meaningful share of articles are placed in the wrong topic. The model was trained on news
articles and may perform worse on other kinds of text. Do not rely on its output for decisions that affect people.

## Acceptable use

Use the app lawfully and in good faith. Do not enter personal or confidential information, do not send automated or very
high volumes of requests, and do not try to disrupt or gain unauthorised access to the service.

## Third-party resources

The project builds on the KINNEWS dataset and the KinyaBERT-small model, each published by its own authors under its own
licence. Details and references are in [the project repository]({GITHUB_URL}). Those resources remain the property of their
respective owners.

## Availability and changes

The app is provided as is and as available, without warranties of any kind. It may be changed, interrupted or taken offline
at any time. To the extent permitted by law, the author is not liable for any loss arising from use of the app.
These terms may be updated, and the date above shows the latest revision.

## Contact

Questions can be raised by opening an issue at [the project repository]({ISSUES_URL}).

</div>
""", unsafe_allow_html=True)


pages = [
    st.Page(classifier_page, title="Classifier", url_path="classifier", default=True),
    st.Page(privacy_page, title="Privacy Policy", url_path="privacy"),
    st.Page(terms_page, title="Terms and Conditions", url_path="terms"),
]
st.navigation(pages, position="top").run()
