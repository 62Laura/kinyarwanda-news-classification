import os

import pandas as pd
import streamlit as st

from onnx_classifier import Classifier

st.set_page_config(page_title="Kinyarwanda News Classifier", page_icon="📰")


@st.cache_resource(show_spinner="Loading the model (first visit only)...")
def get_classifier():
    return Classifier()


def load_examples():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples.txt")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def use_example(text):
    st.session_state["text"] = text


clf = get_classifier()

st.title("Kinyarwanda News Topic Classifier")
st.write(
    "Paste a Kinyarwanda news article or headline and the model predicts its topic. "
    "It is KinyaBERT-small fine-tuned on the KINNEWS dataset, covering 14 categories: "
    + ", ".join(clf.labels)
    + ". It was trained on news text only, so it is less reliable on other kinds of text "
    "(social media, chat, very short phrases), and it always picks one of the 14 topics, "
    "even for text that fits none of them."
)

examples = load_examples()
if examples:
    with st.expander("Try a real example from the test set"):
        for i, ex in enumerate(examples):
            st.caption(ex[:140] + ("..." if len(ex) > 140 else ""))
            st.button(f"Use example {i + 1}", key=f"ex{i}", on_click=use_example, args=(ex,))

text = st.text_area("Kinyarwanda news text", key="text", height=200)

if st.button("Classify", type="primary"):
    if len(text.split()) < 3:
        st.warning("Please enter at least a few words of Kinyarwanda news text.")
    else:
        probs = clf.predict_proba(text)
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        st.success(f"Predicted category: **{ranked[0][0]}** ({ranked[0][1]:.1%})")
        st.write("Top 5 categories")
        for label, p in ranked[:5]:
            st.progress(min(max(p, 0.0), 1.0), text=f"{label}: {p:.1%}")
        with st.expander("All 14 probabilities"):
            st.bar_chart(pd.Series(probs, name="probability"))
