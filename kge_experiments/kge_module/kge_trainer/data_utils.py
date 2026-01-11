"""Helpers for reading project datasets into triple form."""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class TripleExample:
    head: str
    relation: str
    tail: str


def detect_triple_format(sample_line: str) -> str:
    sample = sample_line.strip()
    if not sample:
        return "unknown"
    if "(" in sample and ")" in sample:
        return "prolog"
    if sample.count("\t") >= 2:
        return "tsv"
    if sample.count(",") >= 2:
        return "csv"
    return "unknown"


def _normalize_token(token: str) -> str:
    token = token.strip().strip("'\"").strip()
    return token


def _parse_prolog_fact(line: str) -> Optional[TripleExample]:
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None
    if raw.endswith("."):
        raw = raw[:-1]
    if "(" not in raw or ")" not in raw:
        return None
    predicate, remainder = raw.split("(", 1)
    args = remainder.split(")", 1)[0]
    terms = [_normalize_token(a) for a in args.split(",") if a.strip()]
    if len(terms) != 2:
        raise ValueError(f"Expected binary predicate, got '{line.strip()}'")
    return TripleExample(head=terms[0], relation=_normalize_token(predicate), tail=terms[1])


def _iter_triples_from_file(path: str, format_hint: str = "auto") -> Iterable[TripleExample]:
    if format_hint == "auto":
        format_hint = "unknown"
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    format_hint = detect_triple_format(line)
                    break
    if format_hint == "prolog":
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                triple = _parse_prolog_fact(line)
                if triple is not None:
                    yield triple
        return

    delimiter = "\t" if format_hint == "tsv" else ","
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            cleaned = [_normalize_token(item) for item in row if item.strip()]
            if len(cleaned) < 3:
                continue
            yield TripleExample(head=cleaned[0], relation=cleaned[1], tail=cleaned[2])


def load_triples(path: str, format_hint: str = "auto") -> List[TripleExample]:
    return list(_iter_triples_from_file(path, format_hint=format_hint))


def load_triples_with_mappings(
    path: str,
    format_hint: str = "auto",
) -> Tuple[List[Tuple[int, int, int]], Dict[str, int], Dict[str, int]]:
    triples = load_triples(path, format_hint)
    entity2id: Dict[str, int] = {}
    relation2id: Dict[str, int] = {}
    next_e = 0
    next_r = 0
    triple_ids: List[Tuple[int, int, int]] = []
    for triple in triples:
        if triple.head not in entity2id:
            entity2id[triple.head] = next_e
            next_e += 1
        if triple.tail not in entity2id:
            entity2id[triple.tail] = next_e
            next_e += 1
        if triple.relation not in relation2id:
            relation2id[triple.relation] = next_r
            next_r += 1
        triple_ids.append(
            (
                entity2id[triple.head],
                relation2id[triple.relation],
                entity2id[triple.tail],
            )
        )
    return triple_ids, entity2id, relation2id


def load_dataset_split(
    data_root: str,
    dataset_name: str,
    split_filename: str,
) -> str:
    path = os.path.join(data_root, dataset_name, split_filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Could not find split '{split_filename}' for dataset '{dataset_name}' at {path}")
    return path
