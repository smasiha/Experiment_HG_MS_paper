from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset

from exact_math_reasoning_data import Example, format_instruction_prompt, tokenize_examples


def instruction_example(record: Dict[str, object], source_name: str, source_id: int) -> Example:
    return Example(
        prompt=format_instruction_prompt(str(record.get("instruction") or ""), str(record.get("input") or "")),
        response=str(record.get("output") or "").strip(),
        source_name=source_name,
        source_id=source_id,
    )


def _serialize_examples(examples: List[Example]) -> List[Dict[str, object]]:
    return [asdict(example) for example in examples]


def _deserialize_examples(items: List[Dict[str, object]]) -> List[Example]:
    return [Example(**item) for item in items]


def _load_alpaca_clean(seed: int) -> List[Dict[str, object]]:
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    dataset = dataset.shuffle(seed=seed)
    return [dataset[idx] for idx in range(len(dataset))]


def _load_alpaca_gpt4_en(seed: int) -> List[Dict[str, object]]:
    dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")
    dataset = dataset.shuffle(seed=seed)
    return [dataset[idx] for idx in range(len(dataset))]


def _load_alpaca_gpt4_zh(seed: int) -> List[Dict[str, object]]:
    dataset = load_dataset("llm-wizard/alpaca-gpt4-data-zh", split="train")
    dataset = dataset.shuffle(seed=seed)
    return [dataset[idx] for idx in range(len(dataset))]


def build_small_scale_splits(task: str, *, seed: int) -> Dict[str, object]:
    if task == "denoising":
        rows = _load_alpaca_clean(seed)
        clean_train = [instruction_example(row, "Clean", 0) for row in rows[:1000]]
        noisy_base = rows[1000:10000]
        noisy_train = []
        for row in noisy_base:
            example = instruction_example(row, "Noisy", 1)
            example.response = "."
            noisy_train.append(example)
        val = [instruction_example(row, "Validation-Clean", 0) for row in rows[10000:11000]]
        test = [instruction_example(row, "Test-Clean", 0) for row in rows[11000:12000]]
        return {
            "task": task,
            "source_names": ["Clean", "Noisy"],
            "target_lambda": [1.0, 0.0],
            "reweight_train": [clean_train, noisy_train],
            "val": val,
            "test": test,
            "test_by_source": {"clean": test},
        }

    if task == "multilingual":
        zh_rows = _load_alpaca_gpt4_zh(seed)
        en_rows = _load_alpaca_gpt4_en(seed + 1)
        zh_train = [instruction_example(row, "ZH", 0) for row in zh_rows[:40000]]
        en_train = [instruction_example(row, "EN", 1) for row in en_rows[:40000]]
        val_zh = [instruction_example(row, "Val-ZH", 0) for row in zh_rows[40000:40600]]
        val_en = [instruction_example(row, "Val-EN", 1) for row in en_rows[40000:40400]]
        test_zh = [instruction_example(row, "Test-ZH", 0) for row in zh_rows[40600:41200]]
        test_en = [instruction_example(row, "Test-EN", 1) for row in en_rows[40400:40800]]
        val = val_zh + val_en
        test = test_zh + test_en
        random.Random(seed + 99).shuffle(val)
        random.Random(seed + 199).shuffle(test)
        return {
            "task": task,
            "source_names": ["ZH", "EN"],
            "target_lambda": [0.6, 0.4],
            "reweight_train": [zh_train, en_train],
            "val": val,
            "test": test,
            "test_by_source": {"zh": test_zh, "en": test_en},
        }

    raise ValueError(f"Unsupported task {task}")


def load_or_prepare_small_scale_splits(
    *,
    task: str,
    seed: int,
    cache_path: str | None,
) -> Dict[str, object]:
    if cache_path and Path(cache_path).exists():
        with open(cache_path) as f:
            payload = json.load(f)
        return {
            "task": payload["task"],
            "source_names": payload["source_names"],
            "target_lambda": payload["target_lambda"],
            "reweight_train": [_deserialize_examples(items) for items in payload["reweight_train"]],
            "val": _deserialize_examples(payload["val"]),
            "test": _deserialize_examples(payload["test"]),
            "test_by_source": {name: _deserialize_examples(items) for name, items in payload["test_by_source"].items()},
        }

    splits = build_small_scale_splits(task, seed=seed)
    if cache_path:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(
                {
                    "task": splits["task"],
                    "source_names": splits["source_names"],
                    "target_lambda": splits["target_lambda"],
                    "reweight_train": [_serialize_examples(items) for items in splits["reweight_train"]],
                    "val": _serialize_examples(splits["val"]),
                    "test": _serialize_examples(splits["test"]),
                    "test_by_source": {
                        name: _serialize_examples(items) for name, items in splits["test_by_source"].items()
                    },
                },
                f,
                indent=2,
            )
    return splits


def tokenize_small_scale_splits(
    *,
    tokenizer,
    splits: Dict[str, object],
    max_length: int,
    tokenized_cache_path: str | None,
) -> Dict[str, object]:
    if tokenized_cache_path and Path(tokenized_cache_path).exists():
        return torch.load(tokenized_cache_path, map_location="cpu")

    payload = {
        "task": splits["task"],
        "source_names": splits["source_names"],
        "target_lambda": splits["target_lambda"],
        "reweight_train": [tokenize_examples(tokenizer, items, max_length=max_length) for items in splits["reweight_train"]],
        "val": tokenize_examples(tokenizer, splits["val"], max_length=max_length),
        "test": tokenize_examples(tokenizer, splits["test"], max_length=max_length),
        "test_by_source": {
            name: tokenize_examples(tokenizer, items, max_length=max_length) for name, items in splits["test_by_source"].items()
        },
    }
    if tokenized_cache_path:
        cache_file = Path(tokenized_cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cache_file)
    return payload
