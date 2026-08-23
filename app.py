"""
Streamlit front-end for the sarcasm detector.

Run locally:   streamlit run app.py
Deployed on:   Streamlit Community Cloud (see README)

Model resolution order is handled by Code/pkg/model_training/transformer.py; this file
only supplies the Streamlit-specific source (st.secrets) on top of it.
"""
import json
from pathlib import Path

import streamlit as st

from Code.pkg.analysis.attention_html import render_attention_html
from Code.pkg.model_training.transformer import (NON_SARCASTIC_THRESHOLD, SARCASTIC_THRESHOLD,
                                                 load_model, predict)

REPO_ROOT = Path(__file__).resolve().parent
METRICS_PATH = REPO_ROOT / 'metrics.json'

EXAMPLES = [
    'thirtysomething scientists unveil doomsday clock of hair loss',
    'dem rep. totally nails why congress is falling short on gender, racial equality',
    'area man passionate defender of what he imagines constitution to be',
    "j.k. rowling wishes snape happy birthday in the most magical way",
]

st.set_page_config(page_title='Sarcasm Detection', page_icon=':performing_arts:',
                   layout='centered')


def secret_model_id():
    """st.secrets raises if no secrets file exists at all, so this stays defensive."""
    try:
        return st.secrets.get('SARCASM_MODEL_ID')
    except Exception:
        return None


@st.cache_resource(show_spinner='Loading model...')
def get_model(model_id):
    # Cached as a resource so the weights load once per server process rather than on
    # every rerun -- Streamlit reruns the whole script on each keystroke/interaction.
    return load_model(model_id)


@st.cache_data
def get_metrics():
    if METRICS_PATH.is_file():
        with open(METRICS_PATH, encoding='utf-8') as f:
            return json.load(f)
    return None


def render_setup_help(error):
    st.error('No trained model is available yet.')
    st.markdown("""
Train one, then point the app at it:

1. Open `notebooks/train_distilbert.ipynb` in Google Colab and run it on a T4 GPU
   (about 5 minutes).
2. Either **unzip** the resulting model into `Code/pkg/trained_models/distilbert-sarcasm/`,
   **or** push it to the HuggingFace Hub and set `SARCASM_MODEL_ID`.

On Streamlit Community Cloud, set `SARCASM_MODEL_ID` under *App settings -> Secrets*.
Locally you can instead export it as an environment variable.
""")
    with st.expander('Error detail'):
        st.code(str(error))


st.title('Sarcasm Detection')
st.caption('Fine-tuned DistilBERT with an attention heat-map, on the News Headlines corpus.')

try:
    tokenizer, model, resolved_id = get_model(secret_model_id())
except Exception as error:  # no model trained/configured yet
    render_setup_help(error)
    st.stop()

metrics = get_metrics()

with st.sidebar:
    st.subheader('Model')
    st.code(resolved_id, language=None)

    if metrics:
        st.subheader('Test-set performance')
        scores = metrics.get('distilbert', {})
        st.metric('Accuracy', '{:.1%}'.format(scores.get('accuracy', 0)))
        st.metric('F1', '{:.3f}'.format(scores.get('f1', 0)))
        baseline = metrics.get('baseline_tfidf_logreg', {})
        if baseline:
            st.caption('TF-IDF + LogReg baseline: {:.1%} accuracy'.format(
                baseline.get('accuracy', 0)))
        st.caption('Held out from {:,} de-duplicated headlines.'.format(
            metrics.get('n_total', 0)))

    st.subheader('How to read this')
    st.markdown("""
Scores above **{high}** are called Sarcastic, below **{low}** Non-Sarcastic, and anything
between the two is reported as Neutral rather than forced to a side.
""".format(high=SARCASTIC_THRESHOLD, low=NON_SARCASTIC_THRESHOLD))

st.warning("""
**Two honest caveats.**
In the training corpus every sarcastic headline comes from *The Onion* and every
non-sarcastic one from *HuffPost*, so part of what the model recognises is house style,
not sarcasm as such. And the heat-map shows where the model *attended*, which is an
interpretability aid, **not** proof of what drove the decision.
""")

if 'text' not in st.session_state:
    st.session_state.text = EXAMPLES[0]

def use_example(example):
    # Must be an on_click callback rather than an `if st.button(...)` body: callbacks run
    # before the rerun, so this assignment lands before the text_area widget reclaims its
    # key. Assigning in the script body is silently overridden by the widget's own value.
    st.session_state.text = example


st.markdown('**Try an example**')
columns = st.columns(len(EXAMPLES))
for index, (column, example) in enumerate(zip(columns, EXAMPLES)):
    label = example[:22] + ('...' if len(example) > 22 else '')
    column.button(label, help=example, key='example_' + str(index),
                  on_click=use_example, args=(example,), use_container_width=True)

text = st.text_area('Enter text', key='text', height=100)

if not text.strip():
    st.info('Enter some text above to classify it.')
    st.stop()

result = predict(text, tokenizer, model)

label_colour = {'Sarcastic': 'red', 'Non-Sarcastic': 'green', 'Neutral': 'orange'}
left, right = st.columns([1, 1])
left.markdown('### :{}[{}]'.format(label_colour[result['label']], result['label']))
right.metric('Sarcasm score', '{:.3f}'.format(result['score']))

st.markdown('#### Attention')
st.markdown(render_attention_html(result['tokens'], result['weights'], result['score']),
            unsafe_allow_html=True)

if result['clean_text'] != text.strip().lower():
    with st.expander('Preprocessed text actually fed to the model'):
        st.code(result['clean_text'], language=None)
