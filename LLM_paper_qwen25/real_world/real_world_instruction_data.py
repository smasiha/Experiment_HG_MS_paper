from __future__ import annotations

import json
import os
import random
import sys
import time
import math
from pathlib import Path
from typing import Dict, Iterable, List

from datasets import load_dataset


INSTRUCTION_SOURCE_SPECS: List[Dict[str, str]] = [
    {"name": "AlpacaGPT4", "dataset": "ScaleBiO/ScaleBiO-Train-alpaca_gpt4"},
    {"name": "ShareGPT4", "dataset": "ScaleBiO/ScaleBiO-Train-share_gpt4"},
    {"name": "SlimOrca", "dataset": "ScaleBiO/ScaleBiO-Train-slimorca"},
    {"name": "AlpacaChat", "dataset": "ScaleBiO/ScaleBiO-Train-alpaca_chat_turn2"},
    {"name": "OpenOrcaGPT4", "dataset": "ScaleBiO/ScaleBiO-Train-Open-Orca-1million-gpt-4"},
    {"name": "WildChat", "dataset": "ScaleBiO/ScaleBiO-Train-allenai-WildChat-1m"},
    {"name": "LMSYS-Chat", "dataset": "ScaleBiO/ScaleBiO-Train-lmsys-chat-1m"},
    {"name": "GPTeacher", "dataset": "ScaleBiO/ScaleBiO-Train-teknium-GPTeacher-General-Instruct"},
    {"name": "Airoboros", "dataset": "ScaleBiO/ScaleBiO-Train-jondurbin-airoboros-3"},
    {"name": "Alpaca-es", "dataset": "ScaleBiO/ScaleBiO-Train-alpaca_es_51942"},
    {"name": "Alpaca-de", "dataset": "ScaleBiO/ScaleBiO-Train-alpaca_de_49963"},
    {"name": "Alpaca-ja", "dataset": "ScaleBiO/ScaleBiO-Train-alpaca_ja_51999"},
    {"name": "Alpaca-ko", "dataset": "ScaleBiO/ScaleBiO-Train-alpaca_ko_49620"},
    {"name": "Alpaca-ru", "dataset": "ScaleBiO/ScaleBiO-Train-alpaca_ru_29822"},
    {"name": "Alpaca-it", "dataset": "ScaleBiO/ScaleBiO-Train-alpaca_it_51710"},
    {"name": "Alpaca-fr", "dataset": "ScaleBiO/ScaleBiO-Train-alpaca_fr_55178"},
    {"name": "Alpaca-zh", "dataset": "ScaleBiO/ScaleBiO-Train-alpaca_zh_48818"},
    {"name": "Alpaca-pt", "dataset": "ScaleBiO/ScaleBiO-Train-alpaca_pt_51759"},
]

VALIDATION_DATASET = "ScaleBiO/ScaleBiO-Validation-MTBench"
TUNE_DATASET = "pxyyy/ScaleBio-MTBench-val"
TEST_DATASET = "ScaleBiO/ScaleBiO-Test-MTBench"


def _dataset_cache_dir() -> str | None:
    cache_dir = os.environ.get("HF_DATASETS_CACHE_DIR")
    if cache_dir:
        return cache_dir
    scratch_dir = Path("/scratch/masiha/hypergrad_instruction_cache/hf_datasets")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    return str(scratch_dir)


def _load_streaming_dataset(dataset_name: str):
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            return load_dataset(dataset_name, split="train", streaming=True, cache_dir=_dataset_cache_dir())
        except Exception as exc:  # pragma: no cover - retry path
            last_error = exc
            message = str(exc)
            if "429" not in message and "Too Many Requests" not in message:
                raise
            if attempt == 5:
                raise
            sleep_seconds = min(120, 15 * (2**attempt))
            print(
                json.dumps(
                    {
                        "stage": "hf_retry",
                        "dataset": dataset_name,
                        "attempt": attempt + 1,
                        "sleep_seconds": sleep_seconds,
                    }
                ),
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)
    assert last_error is not None
    raise last_error


def _clean_items(items: Iterable[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        role = str(item.get("from") or item.get("role") or "").lower()
        content = str(item.get("value") or item.get("content") or "").strip()
        if not content:
            continue
        if role in {"human", "user"}:
            cleaned.append({"role": "user", "content": content})
        elif role in {"gpt", "assistant"}:
            cleaned.append({"role": "assistant", "content": content})
    return cleaned


def _row_to_record(row: dict, source_name: str, source_id: int = -1) -> dict | None:
    items = _clean_items(row.get("items") or row.get("messages") or [])
    if len(items) < 2:
        return None
    has_user = any(item["role"] == "user" for item in items)
    has_assistant = any(item["role"] == "assistant" for item in items)
    if not (has_user and has_assistant):
        return None
    return {
        "items": items,
        "source_name": source_name,
        "source_id": source_id,
        "category": row.get("category"),
    }


def _sample_streaming_rows(
    dataset_name: str,
    *,
    source_name: str,
    source_id: int,
    needed: int,
    seed: int,
    require_full: bool = True,
) -> list[dict]:
    dataset = _load_streaming_dataset(dataset_name)
    rng = random.Random(seed)
    reservoir: list[dict] = []
    seen = 0
    max_seen = max(needed * 20, needed)
    for row in dataset:
        record = _row_to_record(row, source_name=source_name, source_id=source_id)
        if record is None:
            continue
        seen += 1
        if len(reservoir) < needed:
            reservoir.append(record)
        else:
            idx = rng.randrange(seen)
            if idx < needed:
                reservoir[idx] = record
        if seen >= max_seen and len(reservoir) >= needed:
            break
    if len(reservoir) < needed and require_full:
        raise RuntimeError(f"Only collected {len(reservoir)} items from {dataset_name}, need {needed}.")
    if len(reservoir) < needed:
        print(
            json.dumps(
                {
                    "stage": "raw_shortfall",
                    "dataset": dataset_name,
                    "available": len(reservoir),
                    "requested": needed,
                }
            ),
            file=sys.stderr,
        )
    return reservoir


def iter_source_records(*, source_id: int) -> Iterable[dict]:
    spec = INSTRUCTION_SOURCE_SPECS[source_id]
    for row in _load_streaming_dataset(spec["dataset"]):
        record = _row_to_record(row, source_name=spec["name"], source_id=source_id)
        if record is not None:
            yield record


def _collect_all_rows(dataset_name: str, *, source_name: str, source_id: int) -> list[dict]:
    rows: list[dict] = []
    for row in _load_streaming_dataset(dataset_name):
        record = _row_to_record(row, source_name=source_name, source_id=source_id)
        if record is not None:
            rows.append(record)
    return rows


def load_or_prepare_instruction_splits(
    *,
    cache_path: str,
    reweight_train_per_source: int,
    reweight_train_raw_oversample: float,
    seed: int,
) -> Dict[str, object]:
    cache_file = Path(cache_path)
    if cache_file.exists():
        try:
            with cache_file.open() as f:
                cached = json.load(f)
            required_keys = {"source_names", "reweight_train", "val_opt", "val_tune", "test"}
            if required_keys.issubset(cached.keys()):
                return cached
        except Exception:
            pass
        cache_file.unlink(missing_ok=True)

    source_names = [spec["name"] for spec in INSTRUCTION_SOURCE_SPECS]
    reweight_train: list[list[dict]] = []
    raw_target = max(reweight_train_per_source, math.ceil(reweight_train_per_source * reweight_train_raw_oversample))
    for source_id, spec in enumerate(INSTRUCTION_SOURCE_SPECS):
        sampled = _sample_streaming_rows(
            spec["dataset"],
            source_name=spec["name"],
            source_id=source_id,
            needed=raw_target,
            seed=seed + 97 * (source_id + 1),
            require_full=False,
        )
        reweight_train.append(sampled)

    val_opt = _collect_all_rows(VALIDATION_DATASET, source_name="MTBenchVal", source_id=-1)
    val_tune = _collect_all_rows(TUNE_DATASET, source_name="MTBenchTune", source_id=-1)
    test = _collect_all_rows(TEST_DATASET, source_name="MTBenchTest", source_id=-1)

    payload = {
        "source_names": source_names,
        "reweight_train": reweight_train,
        "val_opt": val_opt,
        "val_tune": val_tune,
        "test": test,
    }
    if cache_path:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with cache_file.open("w") as f:
                json.dump(payload, f)
        except OSError as exc:
            print(
                json.dumps(
                    {
                        "stage": "cache_write_skipped",
                        "cache_path": str(cache_file),
                        "error": str(exc),
                    }
                ),
                file=sys.stderr,
            )
            cache_file.unlink(missing_ok=True)
    return payload


def sample_final_train_records(
    *,
    source_counts: List[int],
    raw_by_source: List[List[dict]] | None,
    raw_oversample: float,
    seed: int,
) -> List[dict]:
    final_records: List[dict] = []
    for source_id, (count, spec) in enumerate(zip(source_counts, INSTRUCTION_SOURCE_SPECS)):
        if count <= 0:
            continue
        needed = max(count, math.ceil(count * raw_oversample))
        if raw_by_source is not None and source_id < len(raw_by_source) and len(raw_by_source[source_id]) >= needed:
            rng = random.Random(seed + 10_000 + 131 * (source_id + 1))
            sampled = rng.sample(list(raw_by_source[source_id]), needed)
        else:
            sampled = _sample_streaming_rows(
                spec["dataset"],
                source_name=spec["name"],
                source_id=source_id,
                needed=needed,
                seed=seed + 10_000 + 131 * (source_id + 1),
                require_full=False,
            )
        final_records.extend(sampled)
    random.Random(seed + 77_777).shuffle(final_records)
    return final_records
