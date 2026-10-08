from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from hf_dataset_browser_lib import (
    DEFAULT_REPO_ID,
    DEFAULT_REPO_TYPE,
    DEFAULT_SCAN_LIMIT,
    build_generation_sentence_labels,
    build_sentence_spans,
    build_stats,
    compute_deceptive_sentence_idx,
    flatten_history,
    generation_bucket,
    list_environments,
    list_localization_repo_paths,
    list_models_for_environment,
    load_repo_record,
    normalize_history,
    parse_repo_path,
    plot_sentence_localization,
    render_prefix_generation_with_sentence_indices_html,
    resolve_sentence_span,
    sentence_selector_label,
    strip_code_fences,
)


st.set_page_config(page_title="HF Sentence Localization Explorer", layout="wide")
st.title("Sentence-level Deception Localization Dashboard")
st.caption("Hugging Face browser for `anonymous-neurips-2026-ED/deception-localization`.")


if "hf_result_context" not in st.session_state:
    st.session_state.hf_result_context = None


st.sidebar.header("Hugging Face")
default_token = ""
try:
    default_token = st.secrets.get("HF_TOKEN", "")
except Exception:
    default_token = ""
if not default_token:
    default_token = os.environ.get("HF_TOKEN", "")

repo_id = st.sidebar.text_input("Dataset repo", value=DEFAULT_REPO_ID)
hf_token = st.sidebar.text_input("HF token (optional)", value=default_token, type="password")
scan_limit = int(
    st.sidebar.number_input(
        "Max examples to list",
        min_value=50,
        max_value=10000,
        value=DEFAULT_SCAN_LIMIT,
        step=50,
        help="Only list the first N example files under the selected environment/model.",
    )
)
filename_filter = st.sidebar.text_input(
    "Filename filter",
    value="",
    help="Optional substring filter applied within the selected environment/model/localization folder.",
)

try:
    environments = list_environments(repo_id=repo_id, repo_type=DEFAULT_REPO_TYPE, token=hf_token or None)
except Exception as exc:
    st.sidebar.error("Could not list dataset environments.")
    st.sidebar.code(type(exc).__name__ + ": " + str(exc))
    st.stop()

if not environments:
    st.sidebar.error("No environments found in the dataset repo.")
    st.stop()

selected_environment = st.sidebar.selectbox("Environment", environments, index=0)

try:
    models = list_models_for_environment(
        selected_environment,
        repo_id=repo_id,
        repo_type=DEFAULT_REPO_TYPE,
        token=hf_token or None,
    )
except Exception as exc:
    st.sidebar.error("Could not list models for the selected environment.")
    st.sidebar.code(type(exc).__name__ + ": " + str(exc))
    st.stop()

if not models:
    st.sidebar.error("No models found for the selected environment.")
    st.stop()

selected_model = st.sidebar.selectbox("Model", models, index=0)

try:
    repo_paths = list_localization_repo_paths(
        selected_environment,
        selected_model,
        repo_id=repo_id,
        repo_type=DEFAULT_REPO_TYPE,
        token=hf_token or None,
        limit=scan_limit,
        filename_filter=filename_filter,
    )
except Exception as exc:
    st.sidebar.error("Could not list localization files for the selected environment/model.")
    st.sidebar.code(type(exc).__name__ + ": " + str(exc))
    st.stop()

st.sidebar.caption(
    f"Showing {len(repo_paths):,} example references from "
    f"`{selected_environment}/{selected_model}/localization`."
)

if not repo_paths:
    st.warning("No localization files matched the current environment/model/filter/limit.")
    st.stop()

example_options = {path.split("/")[-1]: path for path in repo_paths}
selected_example_name = st.sidebar.selectbox("Example", list(example_options.keys()), index=0)
selected_repo_path = example_options[selected_example_name]

try:
    result = load_repo_record(
        selected_repo_path,
        repo_id=repo_id,
        repo_type=DEFAULT_REPO_TYPE,
        token=hf_token or None,
    )
except Exception as exc:
    st.error("Could not load the selected example from Hugging Face.")
    st.code(type(exc).__name__ + ": " + str(exc))
    st.stop()

raw_text = result.get("raw_text") or ""
history = result.get("history") or []
if not history:
    st.error("No history found in this example.")
    st.stop()

history_norm = normalize_history(history)
df_plot = flatten_history(history_norm, raw_text)
df_stats = build_stats(history_norm)
example_id = str(result.get("example_id") or "")
right_sentence_end_idx = result.get("right_sentence_end_idx")
deceptive_sentence_idx = compute_deceptive_sentence_idx(right_sentence_end_idx, df_stats)
sentence_spans, sentence_span_map = build_sentence_spans(raw_text)
parsed_path = parse_repo_path(selected_repo_path)

current_result_context: Tuple[str, str] = (selected_repo_path, example_id)
if st.session_state.hf_result_context != current_result_context:
    st.session_state.hf_result_context = current_result_context
    for key in (
        "selected_sentence_idx",
        "selected_probe_step",
        "truthful_sel",
        "deceptive_sel",
        "sample_selection_context",
    ):
        if key in st.session_state:
            del st.session_state[key]


st.subheader("Result Summary")
summary_lines = [
    f"Environment: {parsed_path['environment']}",
    f"Model: {parsed_path['model']}",
    f"Example ID: {example_id}",
    f"Repo path: {selected_repo_path}",
    f"Probes: {len(history_norm)}",
]
if deceptive_sentence_idx is not None:
    summary_lines.append(f"Deceptive sentence idx: {deceptive_sentence_idx}")
if right_sentence_end_idx is not None:
    summary_lines.append(f"Stored right sentence end idx: {right_sentence_end_idx}")
st.markdown("\n".join(f"- {line}" for line in summary_lines))

with st.expander("Raw prompt"):
    prompt = result.get("prompt") or ""
    if prompt:
        st.text(prompt)
    else:
        st.info("No prompt stored in this example.")


st.subheader("Deception Rate vs Sentence Index")
if len(df_stats) > 0:
    fig1, df_stats = plot_sentence_localization(
        df_stats,
        deceptive_sentence_idx=deceptive_sentence_idx,
        right_sentence_end_idx=right_sentence_end_idx,
    )
    st.pyplot(fig1)
    with st.expander("Show probe statistics"):
        st.dataframe(df_stats, use_container_width=True)
else:
    st.info("No stats available to plot.")


st.subheader("Prefix Selector")
st.caption(
    "Choose where the fixed prefix should end. The dashboard holds the text fixed up to "
    "that sentence, then shows continuations sampled from that point."
)

available_idxs = sorted(int(value) for value in df_stats["sentence_idx"].dropna().unique()) if len(df_stats) else []
if not available_idxs:
    st.info("No sentence indices available for selection.")
    st.stop()

if st.session_state.get("selected_sentence_idx") not in available_idxs and "selected_sentence_idx" in st.session_state:
    del st.session_state["selected_sentence_idx"]

selected_sentence_idx = st.selectbox(
    "Sentence index",
    available_idxs,
    key="selected_sentence_idx",
    format_func=lambda idx: sentence_selector_label(idx, sentence_span_map),
)

probe_rows = [
    (idx, probe)
    for idx, probe in enumerate(history_norm)
    if probe.get("sentence_idx") == selected_sentence_idx
    or probe.get("sentence_end_idx") == selected_sentence_idx + 1
]
if not probe_rows:
    st.info("No probe found for this sentence.")
    st.stop()

probe = probe_rows[0][1]
if len(probe_rows) > 1:
    step_options = [idx for idx, _ in probe_rows]
    if st.session_state.get("selected_probe_step") not in step_options and "selected_probe_step" in st.session_state:
        del st.session_state["selected_probe_step"]
    selected_step = st.selectbox(
        "Probe step",
        step_options,
        key="selected_probe_step",
        format_func=lambda idx: f"step {idx}",
    )
    probe = dict(history_norm[selected_step])

resolved_sentence_text, resolved_start, resolved_end = resolve_sentence_span(
    selected_sentence_idx,
    sentence_span_map,
    probe=probe,
)
resolved_sentence_idx = int(selected_sentence_idx)

# Prefer the sentence recovered from raw_text so this matches the dropdown label.
# Some stored probe["sentence_text"] values can be shifted by one.
sentence_text = (resolved_sentence_text or probe.get("sentence_text") or "").strip()

selected_span = (
    (resolved_start, resolved_end)
    if resolved_start is not None and resolved_end is not None
    else probe.get("char_span")
)

st.markdown(
    f"Sentence idx: {resolved_sentence_idx} | Deception rate: {probe.get('deception_rate')} | "
    f"Valid samples: {probe.get('num_valid')}"
)
if sentence_text:
    st.markdown(f"Sentence: `{sentence_text}`")

bucket_counts = (
    pd.Series(
        [generation_bucket(generation) for generation in (probe.get("generations") or [])],
        name="count",
    )
    .value_counts()
    .reindex(["truthful", "deceptive", "invalid"], fill_value=0)
)
st.dataframe(bucket_counts.to_frame(), use_container_width=False)


st.subheader("Sample Selector")
st.caption(
    "Select one continuation sampled from the chosen prefix. Truthful and deceptive "
    "continuations are listed separately so you can compare how the same fixed prefix "
    "can lead to different outcomes."
)

subset = df_plot[
    (df_plot["sentence_idx"] == selected_sentence_idx)
    | (df_plot["sentence_end_idx"] == selected_sentence_idx + 1)
]
if subset.empty:
    st.info("No generations available for this probe.")
    st.stop()

sample_selection_context = (selected_repo_path, int(selected_sentence_idx))
if st.session_state.get("sample_selection_context") != sample_selection_context:
    st.session_state.sample_selection_context = sample_selection_context
    st.session_state.truthful_sel = "None"
    st.session_state.deceptive_sel = "None"


def on_truthful_change():
    st.session_state.deceptive_sel = "None"


def on_deceptive_change():
    st.session_state.truthful_sel = "None"


truthful = subset[subset["is_truthful"] == True]
deceptive = subset[subset["is_truthful"] == False]
invalid_count = int((subset["is_truthful"].isna()).sum())

truthful_opts: Dict[str, int] = {
    f"Truthful generation {row.sample_id}": int(row.sample_id)
    for _, row in truthful.iterrows()
}
deceptive_opts: Dict[str, int] = {
    f"Deceptive generation {row.sample_id}": int(row.sample_id)
    for _, row in deceptive.iterrows()
}

col1, col2 = st.columns(2)
with col1:
    st.markdown("Truthful samples")
    st.selectbox(
        "Select truthful sample",
        ["None"] + list(truthful_opts.keys()),
        key="truthful_sel",
        on_change=on_truthful_change,
    )
with col2:
    st.markdown("Deceptive samples")
    st.selectbox(
        "Select deceptive sample",
        ["None"] + list(deceptive_opts.keys()),
        key="deceptive_sel",
        on_change=on_deceptive_change,
    )

if invalid_count:
    st.caption(f"Invalid generations for this sentence: {invalid_count}")

selected_sample_id: Optional[int] = None
if st.session_state.truthful_sel != "None":
    selected_sample_id = truthful_opts.get(st.session_state.truthful_sel)
elif st.session_state.deceptive_sel != "None":
    selected_sample_id = deceptive_opts.get(st.session_state.deceptive_sel)


st.subheader("Generation Viewer")
st.caption(
    "This view shows the fixed prefix together with the selected continuation. Blue text "
    "comes before the final sentence in the fixed prefix, black text is the final sentence "
    "in the fixed prefix, and green text is the sampled continuation."
)

if selected_sample_id is None:
    st.info("Select a truthful or deceptive sample above to view the generation.")
    st.stop()

selected_rows = subset[subset["sample_id"] == selected_sample_id]
if selected_rows.empty:
    st.info("Selected sample is not available for the current sentence.")
    st.stop()

row = selected_rows.iloc[0]
gen_text = strip_code_fences(row.get("gen_text") or "")

# Always show the full fixed prefix ending at the selected sentence.
if resolved_end is not None:
    prefix_text = raw_text[:resolved_end]
else:
    prefix_text = sentence_text or ""

prefix_mode_key = "full"

display_selected_idx, _ = build_generation_sentence_labels(
    prefix_text,
    prefix_mode_key,
    resolved_sentence_idx,
)

# Hide sentence-index labels like ^S_12 in the rendered text.
generation_sentence_labels = {}

st.markdown(
    f"Truthful: {row.get('is_truthful')} | Parse error: {row.get('parse_error')} | "
    f"Selected prefix ends at sentence {resolved_sentence_idx + 1}"
)

st.markdown(
    render_prefix_generation_with_sentence_indices_html(
        prefix_text,
        gen_text,
        display_selected_idx,
        sentence_labels=generation_sentence_labels,
    ),
    unsafe_allow_html=True,
)

with st.expander("Full generation text", expanded=False):
    full_generation_text = row.get("full_generation_text") or ""
    if full_generation_text:
        st.text(full_generation_text)
    else:
        st.info("No full generation text stored.")

with st.expander("Parsed output", expanded=False):
    parsed = row.get("parsed")
    if parsed:
        st.json(parsed, expanded=False)
    else:
        st.info("No parsed output stored.")

with st.expander("Evaluation", expanded=False):
    evaluation = row.get("evaluation")
    if evaluation:
        st.json(evaluation, expanded=False)
    else:
        st.info("No evaluation block stored.")