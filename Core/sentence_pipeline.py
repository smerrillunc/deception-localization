from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional


SENTENCE_BOUNDARY_CHARS = ".!?"
SENTENCE_TRAILING_CLOSERS = "\"')}]"
COMMON_ABBREVIATIONS = {
    "dr",
    "etc",
    "fig",
    "jr",
    "mr",
    "mrs",
    "ms",
    "prof",
    "sr",
    "st",
    "vs",
}


def _is_decimal_point(text: str, idx: int) -> bool:
    return (
        text[idx] == "."
        and idx > 0
        and idx + 1 < len(text)
        and text[idx - 1].isdigit()
        and text[idx + 1].isdigit()
    )


def _is_common_abbreviation(text: str, idx: int) -> bool:
    if text[idx] != ".":
        return False
    start = idx
    while start > 0 and text[start - 1].isalpha():
        start -= 1
    token = text[start:idx].lower()
    return token in COMMON_ABBREVIATIONS


def _iter_sentence_bounds(text: str) -> Iterator[tuple[int, int]]:
    n_chars = len(text)
    start = 0
    while start < n_chars and text[start].isspace():
        start += 1
    if start >= n_chars:
        return

    idx = start
    while idx < n_chars:
        char = text[idx]
        if char in SENTENCE_BOUNDARY_CHARS and not _is_decimal_point(text, idx) and not _is_common_abbreviation(text, idx):
            end = idx + 1
            while end < n_chars and text[end] in SENTENCE_TRAILING_CLOSERS:
                end += 1
            yield start, end
            start = end
            while start < n_chars and text[start].isspace():
                start += 1
            idx = start
            continue
        idx += 1

    if start < n_chars:
        end = n_chars
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            yield start, end


def split_sentences(text: Any) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    cleaned = " ".join(text.strip().split())
    return [span["text"] for span in split_sentence_spans(cleaned)]


def split_sentence_spans(text: Any) -> List[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return []
    spans = []
    for start, end in _iter_sentence_bounds(text):
        spans.append(
            {
                "start": start,
                "end": end,
                "text": text[start:end],
            }
        )
    return spans


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


@dataclass
class TaxonomyLabel:
    id: str
    name: str
    description: str
    examples: Optional[List[str]] = None


@dataclass
class SentenceTaxonomy:
    name: str
    version: str
    labels: List[TaxonomyLabel]
    instructions: Optional[str] = None

    @staticmethod
    def from_json(path: str | Path) -> "SentenceTaxonomy":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        labels = [
            TaxonomyLabel(
                id=item["id"],
                name=item["name"],
                description=item["description"],
                examples=item.get("examples"),
            )
            for item in data["labels"]
        ]
        return SentenceTaxonomy(
            name=data.get("name", "taxonomy"),
            version=data.get("version", "v1"),
            labels=labels,
            instructions=data.get("instructions"),
        )

    def to_prompt_block(self) -> str:
        lines = []
        lines.append(f"Taxonomy: {self.name} ({self.version})")
        for idx, label in enumerate(self.labels, 1):
            lines.append(f"{idx}. {label.name}: {label.description}")
            if label.examples:
                lines.append(f"Examples: {' | '.join(label.examples)}")
        if self.instructions:
            lines.append(f"Instructions: {self.instructions}")
        return "\n".join(lines)


def make_sentence_id(example_id: str, sentence_idx: int) -> str:
    return f"{example_id}/sent_{sentence_idx:04d}"


def build_sentence_records(
    examples: Iterable[Dict[str, Any]],
    *,
    text_field: str = "action_reasoning",
    example_id_field: str = "record_id",
    include_example_fields: Optional[List[str]] = None,
) -> Iterator[Dict[str, Any]]:
    include_example_fields = include_example_fields or []

    for ex in examples:
        example_id = ex.get(example_id_field) or ex.get("example_id")
        if not example_id:
            continue
        text = ex.get(text_field)
        sentences = split_sentence_spans(text)
        for idx, sent in enumerate(sentences):
            rec = {
                "sentence_id": make_sentence_id(example_id, idx),
                "example_id": example_id,
                "source_field": text_field,
                "sentence_idx": idx,
                "sentence_text": sent["text"],
                "start": sent["start"],
                "end": sent["end"],
            }
            for field in include_example_fields:
                if field in ex:
                    rec[field] = ex[field]
            yield rec


def build_sentence_prompt(sentence: str, taxonomy: SentenceTaxonomy) -> str:
    return (
        "Label the sentence with exactly one taxonomy category.\n"
        "Return JSON only: {\"label_id\": \"...\", \"label_name\": \"...\", \"confidence\": 0-1}\n\n"
        f"{taxonomy.to_prompt_block()}\n\n"
        f"Sentence:\n{sentence}"
    )


TaggerFn = Callable[[str, SentenceTaxonomy], Dict[str, Any]]


def tag_sentences(
    sentences: Iterable[Dict[str, Any]],
    taxonomy: SentenceTaxonomy,
    *,
    tagger: TaggerFn,
    sentence_text_field: str = "sentence_text",
    cache: Optional[Dict[str, Dict[str, Any]]] = None,
    cache_key_field: str = "sentence_id",
) -> Iterator[Dict[str, Any]]:
    for rec in sentences:
        sentence = rec.get(sentence_text_field, "")
        cache_key = rec.get(cache_key_field)
        if cache is not None and cache_key in cache:
            tag = cache[cache_key]
        else:
            tag = tagger(sentence, taxonomy)
            if cache is not None and cache_key is not None:
                cache[cache_key] = tag
        yield {
            "sentence_id": rec["sentence_id"],
            "example_id": rec["example_id"],
            **tag,
        }


def build_localization_input(
    *,
    examples_path: str | Path,
    sentences_path: str | Path,
    tags_path: str | Path,
    out_path: str | Path,
    build_record_fn: Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
) -> None:
    """
    Join examples + sentences + tags and build localization input records.
    build_record_fn(example, sentence, tag) -> dict
    """
    examples = {ex["example_id"]: ex for ex in read_jsonl(examples_path)}
    sentences = {s["sentence_id"]: s for s in read_jsonl(sentences_path)}

    def _records():
        for tag in read_jsonl(tags_path):
            sentence = sentences.get(tag["sentence_id"])
            if not sentence:
                continue
            example = examples.get(sentence["example_id"])
            if not example:
                continue
            yield build_record_fn(example, sentence, tag)

    write_jsonl(_records(), out_path)
