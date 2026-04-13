from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import torch
from datasets import IterableDataset, concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download


@dataclass
class Example:
    prompt: str
    response: str
    source_name: str
    source_id: int = -1
    eval_kind: str | None = None
    target_answer: str | None = None


TRAIN_SOURCE_SPECS: List[Dict[str, str | None]] = [
    {"name": "DART-Math", "dataset": "hkust-nlp/dart-math-uniform", "config": None, "split": "train", "kind": "dart_math"},
    {"name": "SlimOrca", "dataset": "Open-Orca/SlimOrca", "config": "default", "split": "train", "kind": "conversation"},
    {"name": "UltraInteract", "dataset": "openbmb/UltraInteract_sft", "config": "default", "split": "train", "kind": "instruction_response"},
    {"name": "MathInstruct", "dataset": "TIGER-Lab/MathInstruct", "config": "default", "split": "train", "kind": "math_instruct"},
    {"name": "Orca-Math", "dataset": "microsoft/orca-math-word-problems-200k", "config": "default", "split": "train", "kind": "orca_math"},
    {"name": "WizardLM", "dataset": "WizardLMTeam/WizardLM_evol_instruct_V2_196k", "config": "default", "split": "train", "kind": "conversation"},
    {"name": "Magicoder", "dataset": "ise-uiuc/Magicoder-Evol-Instruct-110K", "config": "default", "split": "train", "kind": "instruction_response"},
    {"name": "ShareGPT-Vicuna", "dataset": "anon8231489123/ShareGPT_Vicuna_unfiltered", "config": None, "split": "train", "kind": "sharegpt_json"},
    {"name": "GPTeacher", "dataset": "teknium/GPTeacher-General-Instruct", "config": "default", "split": "train", "kind": "inst_input_response"},
    {"name": "GPT4-LLM-Cleaned", "dataset": "teknium/GPT4-LLM-Cleaned", "config": "default", "split": "train", "kind": "inst_input_output"},
]

MATH_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

VAL_SOURCE_SPECS: List[Dict[str, str | None]] = [
    {"name": "GSM8K-val", "dataset": "openai/gsm8k", "config": "main", "split": "train", "kind": "gsm8k"},
    {"name": "CompetitionMath-val", "dataset": "EleutherAI/hendrycks_math", "config": None, "split": "train", "kind": "hendrycks_math"},
    {"name": "Self-OSS-val", "dataset": "bigcode/self-oss-instruct-sc2-exec-filter-50k", "config": "default", "split": "train", "kind": "instruction_response"},
    {"name": "MMLU-val", "dataset": "cais/mmlu", "config": "all", "split": "validation", "kind": "mmlu"},
    {"name": "ARC-Easy-val", "dataset": "allenai/ai2_arc", "config": "ARC-Easy", "split": "validation", "kind": "arc"},
    {"name": "ARC-Challenge-val", "dataset": "allenai/ai2_arc", "config": "ARC-Challenge", "split": "validation", "kind": "arc"},
]


def _strip(text: str | None) -> str:
    return (text or "").strip()


def format_instruction_prompt(instruction: str, input_text: str = "") -> str:
    instruction = _strip(instruction)
    input_text = _strip(input_text)
    if input_text:
        return f"Instruction:\n{instruction}\n\nInput:\n{input_text}\n\nResponse:\n"
    return f"Instruction:\n{instruction}\n\nResponse:\n"


def format_math_prompt(problem: str) -> str:
    return f"Solve the following problem and provide the final answer clearly.\n\nProblem:\n{_strip(problem)}\n\nAnswer:\n"


def format_mc_prompt(question: str, choices: Sequence[str]) -> str:
    choice_lines = []
    for idx, choice in enumerate(choices):
        choice_lines.append(f"{chr(ord('A') + idx)}. {_strip(choice)}")
    choice_block = "\n".join(choice_lines)
    return (
        "Answer the multiple-choice question. State the correct option and answer.\n\n"
        f"Question:\n{_strip(question)}\n\nChoices:\n{choice_block}\n\nAnswer:\n"
    )


def format_conversation_prompt(messages: Sequence[Dict[str, str]]) -> str:
    parts: List[str] = []
    for message in messages:
        role = (message.get("role") or message.get("from") or "").lower()
        content = _strip(message.get("content") or message.get("value"))
        if not content:
            continue
        if role in {"system"}:
            parts.append(f"System: {content}")
        elif role in {"assistant", "gpt"}:
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    if not parts:
        return ""
    return "\n\n".join(parts) + "\n\nAssistant:\n"


def canonicalize_numeric_answer(text: str) -> str:
    text = text.strip().replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return text.strip().lower()
    number = matches[-1]
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return number


def extract_gsm8k_answer(text: str) -> str:
    if "####" in text:
        text = text.split("####")[-1]
    return canonicalize_numeric_answer(text)


def _extract_last_boxed(text: str) -> str:
    marker_positions = [text.rfind("\\boxed{"), text.rfind("\\fbox{")]
    start = max(marker_positions)
    if start < 0:
        return ""
    brace_start = text.find("{", start)
    if brace_start < 0:
        return ""
    depth = 0
    pieces: List[str] = []
    for char in text[brace_start + 1 :]:
        if char == "{":
            depth += 1
            pieces.append(char)
        elif char == "}":
            if depth == 0:
                return "".join(pieces).strip()
            depth -= 1
            pieces.append(char)
        else:
            pieces.append(char)
    return ""


def normalize_math_answer(text: str) -> str:
    boxed = _extract_last_boxed(text)
    if boxed:
        text = boxed
    text = text.strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("$", "").replace(" ", "")
    text = text.replace("\\!", "").replace("\\,", "")
    text = text.replace("\\%", "%")
    if text.startswith("="):
        text = text[1:]
    numeric = canonicalize_numeric_answer(text)
    return numeric if re.fullmatch(r"-?\d+(?:\.\d+)?", numeric) else text.lower()


def _normalize_conversation_record(record: Dict[str, object], source_name: str) -> Example | None:
    messages = record.get("conversations")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    last_assistant = None
    prompt_messages = None
    for idx in range(len(messages) - 1, -1, -1):
        message = messages[idx]
        if not isinstance(message, dict):
            continue
        role = (message.get("role") or message.get("from") or "").lower()
        if role in {"assistant", "gpt"}:
            last_assistant = message
            prompt_messages = messages[:idx]
            break
    if last_assistant is None or prompt_messages is None:
        return None
    prompt = format_conversation_prompt(prompt_messages)
    response = _strip(last_assistant.get("content") or last_assistant.get("value"))
    if not prompt or not response:
        return None
    return Example(prompt=prompt, response=response, source_name=source_name)


def _normalize_train_record(record: Dict[str, object], spec: Dict[str, str | None]) -> Example | None:
    kind = spec["kind"]
    source_name = spec["name"] or "source"
    if kind == "dart_math":
        prompt = format_math_prompt(str(record.get("query") or ""))
        response = _strip(record.get("response"))
    elif kind == "instruction_response":
        prompt = format_instruction_prompt(str(record.get("instruction") or ""))
        response = _strip(record.get("response"))
    elif kind == "math_instruct":
        prompt = format_math_prompt(str(record.get("instruction") or ""))
        response = _strip(record.get("output"))
    elif kind == "orca_math":
        prompt = format_math_prompt(str(record.get("question") or ""))
        response = _strip(record.get("answer"))
    elif kind == "conversation":
        return _normalize_conversation_record(record, source_name)
    elif kind == "sharegpt_json":
        return _normalize_conversation_record(record, source_name)
    elif kind == "inst_input_response":
        prompt = format_instruction_prompt(str(record.get("instruction") or ""), str(record.get("input") or ""))
        response = _strip(record.get("response"))
    elif kind == "inst_input_output":
        prompt = format_instruction_prompt(str(record.get("instruction") or ""), str(record.get("input") or ""))
        response = _strip(record.get("output"))
    else:
        raise ValueError(f"Unsupported train kind {kind}")
    if not prompt or not response:
        return None
    return Example(prompt=prompt, response=response, source_name=source_name)


def _normalize_val_record(record: Dict[str, object], spec: Dict[str, str | None]) -> Example | None:
    kind = spec["kind"]
    source_name = spec["name"] or "source"
    if kind == "gsm8k":
        response = _strip(record.get("answer"))
        return Example(
            prompt=format_math_prompt(str(record.get("question") or "")),
            response=response,
            source_name=source_name,
            eval_kind="gsm8k",
            target_answer=extract_gsm8k_answer(response),
        )
    if kind == "hendrycks_math":
        response = _strip(record.get("solution"))
        return Example(
            prompt=format_math_prompt(str(record.get("problem") or "")),
            response=response,
            source_name=source_name,
            eval_kind="math",
            target_answer=normalize_math_answer(response),
        )
    if kind == "instruction_response":
        return Example(
            prompt=format_instruction_prompt(str(record.get("instruction") or "")),
            response=_strip(record.get("response")),
            source_name=source_name,
        )
    if kind == "mmlu":
        choices = [str(choice) for choice in record.get("choices") or []]
        answer_idx = int(record.get("answer"))
        answer_text = choices[answer_idx]
        response = f"{chr(ord('A') + answer_idx)}. {answer_text}"
        return Example(
            prompt=format_mc_prompt(str(record.get("question") or ""), choices),
            response=response,
            source_name=source_name,
        )
    if kind == "arc":
        choices_obj = record.get("choices") or {}
        labels = list(choices_obj.get("label") or [])
        texts = list(choices_obj.get("text") or [])
        mapping = {label: text for label, text in zip(labels, texts)}
        ordered_texts = [mapping[label] for label in labels]
        answer_key = str(record.get("answerKey") or "").strip()
        answer_text = mapping.get(answer_key, answer_key)
        response = f"{answer_key}. {answer_text}" if answer_key else answer_text
        return Example(
            prompt=format_mc_prompt(str(record.get("question") or ""), ordered_texts),
            response=response,
            source_name=source_name,
        )
    raise ValueError(f"Unsupported val kind {kind}")


def _iter_sharegpt_rows() -> Iterator[Dict[str, object]]:
    json_path = hf_hub_download(
        repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
        filename="ShareGPT_V3_unfiltered_cleaned_split.json",
        repo_type="dataset",
    )
    with open(json_path) as f:
        rows = json.load(f)
    yield from rows


def _iter_gpt4_cleaned_rows() -> Iterator[Dict[str, object]]:
    for filename in ["alpaca_gpt4_data_unfiltered.json", "unnatural_instructions_unfiltered.json"]:
        json_path = hf_hub_download(
            repo_id="teknium/GPT4-LLM-Cleaned",
            filename=filename,
            repo_type="dataset",
        )
        with open(json_path) as f:
            rows = json.load(f)
        for row in rows:
            yield row


def _iter_hendrycks_math_rows(split: str) -> Iterator[Dict[str, object]]:
    datasets = [load_dataset("EleutherAI/hendrycks_math", cfg, split=split) for cfg in MATH_CONFIGS]
    combined = concatenate_datasets(datasets)
    for row in combined:
        yield row


def reservoir_sample_examples(
    spec: Dict[str, str | None],
    *,
    sample_size: int,
    seed: int,
    normalize_kind: str,
    require_full: bool = True,
) -> List[Example]:
    rng = random.Random(seed)
    if spec["kind"] == "sharegpt_json":
        rows = list(_iter_sharegpt_rows())
        rng.shuffle(rows)
        sampled: List[Example] = []
        for row in rows:
            example = _normalize_train_record(row, spec)
            if example is None or not example.prompt or not example.response:
                continue
            sampled.append(example)
            if len(sampled) >= sample_size:
                break
        if require_full and len(sampled) < sample_size:
            raise RuntimeError(f"Only sampled {len(sampled)} examples from {spec['name']} but needed {sample_size}.")
        return sampled

    if spec["dataset"] == "teknium/GPT4-LLM-Cleaned":
        rows = list(_iter_gpt4_cleaned_rows())
        rng.shuffle(rows)
        sampled = []
        for row in rows:
            example = (
                _normalize_train_record(row, spec)
                if normalize_kind == "train"
                else _normalize_val_record(row, spec)
            )
            if example is None or not example.prompt or not example.response:
                continue
            sampled.append(example)
            if len(sampled) >= sample_size:
                break
        if require_full and len(sampled) < sample_size:
            raise RuntimeError(f"Only sampled {len(sampled)} examples from {spec['name']} but needed {sample_size}.")
        return sampled

    if spec["kind"] == "hendrycks_math":
        combined = concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", cfg, split=str(spec["split"])) for cfg in MATH_CONFIGS])
        combined = combined.shuffle(seed=seed)
        sampled = []
        for row in combined.select(range(min(sample_size * 2, len(combined)))):
            example = _normalize_val_record(row, spec)
            if example is None or not example.prompt or not example.response:
                continue
            sampled.append(example)
            if len(sampled) >= sample_size:
                break
        if require_full and len(sampled) < sample_size:
            raise RuntimeError(f"Only sampled {len(sampled)} examples from {spec['name']} but needed {sample_size}.")
        return sampled

    dataset = load_dataset(
        str(spec["dataset"]),
        spec["config"],
        split=str(spec["split"]),
        streaming=True,
    )
    if isinstance(dataset, IterableDataset):
        dataset = dataset.shuffle(seed=seed, buffer_size=max(sample_size * 50, 1000))
        iterator = iter(dataset)
    else:
        dataset = dataset.shuffle(seed=seed)
        iterator = iter(dataset.select(range(min(sample_size * 3, len(dataset)))))

    sampled = []
    for row in iterator:
        example = (
            _normalize_train_record(row, spec)
            if normalize_kind == "train"
            else _normalize_val_record(row, spec)
        )
        if example is None or not example.prompt or not example.response:
            continue
        sampled.append(example)
        if len(sampled) >= sample_size:
            break
    if require_full and len(sampled) < sample_size:
        raise RuntimeError(f"Only sampled {len(sampled)} examples from {spec['name']} but needed {sample_size}.")
    return sampled


def _val_split_sizes(val_per_source: int) -> tuple[list[int], list[int]]:
    tune_counts = [max(1, int(math.floor(val_per_source / 6.0))) for _ in VAL_SOURCE_SPECS]
    total_tune = sum(tune_counts)
    target_tune = max(1, val_per_source)
    while total_tune < target_tune:
        for idx in range(len(tune_counts)):
            if total_tune == target_tune:
                break
            tune_counts[idx] += 1
            total_tune += 1
    opt_sizes = [val_per_source - tune for tune in tune_counts]
    tune_sizes = tune_counts
    return opt_sizes, tune_sizes


def load_or_prepare_exact_splits(
    *,
    cache_path: str | None,
    reweight_train_per_source: int,
    train_raw_oversample: float,
    val_per_source: int,
    val_raw_oversample: float,
    seed: int,
) -> Dict[str, object]:
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path) as f:
            payload = json.load(f)
        return {
            "source_names": payload["source_names"],
            "reweight_train": [
                [Example(**item) for item in source_examples] for source_examples in payload["reweight_train"]
            ],
            "val_opt": [Example(**item) for item in payload["val_opt"]],
            "val_tune": [Example(**item) for item in payload["val_tune"]],
        }

    source_names = [str(spec["name"]) for spec in TRAIN_SOURCE_SPECS]
    reweight_train: List[List[Example]] = []
    raw_train_size = math.ceil(reweight_train_per_source * train_raw_oversample)
    for source_id, spec in enumerate(TRAIN_SOURCE_SPECS):
        print(json.dumps({"stage": "sample_train_source", "source": spec["name"], "raw_target": raw_train_size}))
        sampled = reservoir_sample_examples(spec, sample_size=raw_train_size, seed=seed + 101 * source_id, normalize_kind="train")
        for example in sampled:
            example.source_id = source_id
        reweight_train.append(sampled)
        print(json.dumps({"stage": "sample_train_source_done", "source": spec["name"], "count": len(sampled)}))

    val_opt: List[Example] = []
    val_tune: List[Example] = []
    opt_sizes, tune_sizes = _val_split_sizes(val_per_source)
    raw_val_size = max(val_per_source, math.ceil(val_per_source * val_raw_oversample))
    for source_id, spec in enumerate(VAL_SOURCE_SPECS):
        print(json.dumps({"stage": "sample_val_source", "source": spec["name"], "raw_target": raw_val_size}))
        sampled = reservoir_sample_examples(
            spec,
            sample_size=raw_val_size,
            seed=seed + 10_000 + 101 * source_id,
            normalize_kind="val",
            require_full=False,
        )
        rng = random.Random(seed + 20_000 + source_id)
        rng.shuffle(sampled)
        keep = sampled[:val_per_source]
        if len(keep) < val_per_source:
            raise RuntimeError(f"Only {len(keep)} validation examples available for {spec['name']}; need {val_per_source}.")
        for example in keep:
            example.source_id = source_id
        opt_count = opt_sizes[source_id]
        tune_count = tune_sizes[source_id]
        val_opt.extend(keep[:opt_count])
        val_tune.extend(keep[opt_count : opt_count + tune_count])
        print(json.dumps({"stage": "sample_val_source_done", "source": spec["name"], "count": len(keep)}))

    payload = {
        "source_names": source_names,
        "reweight_train": [[asdict(example) for example in source_examples] for source_examples in reweight_train],
        "val_opt": [asdict(example) for example in val_opt],
        "val_tune": [asdict(example) for example in val_tune],
    }
    if cache_path is not None:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(payload, f, indent=2)
    return {
        "source_names": source_names,
        "reweight_train": reweight_train,
        "val_opt": val_opt,
        "val_tune": val_tune,
    }


def sample_final_train_examples(
    *,
    source_counts: Sequence[int],
    train_raw_oversample: float,
    seed: int,
) -> List[Example]:
    final_examples: List[Example] = []
    for source_id, (count, spec) in enumerate(zip(source_counts, TRAIN_SOURCE_SPECS)):
        if count <= 0:
            continue
        raw_count = max(count, math.ceil(count * train_raw_oversample))
        sampled = reservoir_sample_examples(
            spec,
            sample_size=raw_count,
            seed=seed + 50_000 + 131 * source_id,
            normalize_kind="train",
        )
        for example in sampled:
            example.source_id = source_id
            final_examples.append(example)
    random.Random(seed + 77_777).shuffle(final_examples)
    return final_examples


def load_eval_examples(*, include_math: bool = True) -> Dict[str, List[Example]]:
    gsm8k = reservoir_sample_examples(
        {"name": "GSM8K-test", "dataset": "openai/gsm8k", "config": "main", "split": "test", "kind": "gsm8k"},
        sample_size=1319,
        seed=123,
        normalize_kind="val",
    )
    math_examples: List[Example] = []
    if include_math:
        for cfg in MATH_CONFIGS:
            ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="test")
            for row in ds:
                example = Example(
                    prompt=format_math_prompt(str(row.get("problem") or "")),
                    response=_strip(row.get("solution")),
                    source_name="CompetitionMath-test",
                    eval_kind="math",
                    target_answer=normalize_math_answer(_strip(row.get("solution"))),
                )
                math_examples.append(example)
    return {"gsm8k": gsm8k, "math": math_examples}


def encode_example(tokenizer, example: Example, max_length: int) -> Dict[str, object] | None:
    prompt_ids = tokenizer(
        example.prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length + 1,
    )["input_ids"]
    response_ids = tokenizer(
        example.response + tokenizer.eos_token,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length + 1,
    )["input_ids"]
    if len(prompt_ids) + len(response_ids) > max_length:
        return None
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "source_id": example.source_id,
        "prompt": example.prompt,
        "target_answer": example.target_answer,
        "eval_kind": example.eval_kind,
    }


def tokenize_examples(tokenizer, examples: Sequence[Example], max_length: int) -> List[Dict[str, object]]:
    tokenized: List[Dict[str, object]] = []
    for example in examples:
        encoded = encode_example(tokenizer, example, max_length=max_length)
        if encoded is not None:
            tokenized.append(encoded)
    return tokenized


def collate_batch(
    batch: Sequence[Dict[str, object]],
    *,
    device: torch.device,
    pad_token_id: int,
) -> Dict[str, object]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    attention_mask = []
    labels = []
    source_ids = []
    prompts = []
    target_answers = []
    eval_kinds = []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad)
        attention_mask.append(item["attention_mask"] + [0] * pad)
        labels.append(item["labels"] + [-100] * pad)
        source_ids.append(int(item.get("source_id", -1)))
        prompts.append(item.get("prompt"))
        target_answers.append(item.get("target_answer"))
        eval_kinds.append(item.get("eval_kind"))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "source_ids": torch.tensor(source_ids, dtype=torch.long, device=device),
        "prompts": prompts,
        "target_answers": target_answers,
        "eval_kinds": eval_kinds,
    }


def sample_balanced_batch(
    tokenized_by_source: Sequence[Sequence[Dict[str, object]]],
    *,
    per_source_batch: int,
    rng: random.Random,
    device: torch.device,
    pad_token_id: int,
) -> Dict[str, object]:
    batch: List[Dict[str, object]] = []
    for source_examples in tokenized_by_source:
        for _ in range(per_source_batch):
            batch.append(source_examples[rng.randrange(len(source_examples))])
    rng.shuffle(batch)
    return collate_batch(batch, device=device, pad_token_id=pad_token_id)


def sample_batch(
    tokenized_examples: Sequence[Dict[str, object]],
    *,
    batch_size: int,
    rng: random.Random,
    device: torch.device,
    pad_token_id: int,
) -> Dict[str, object]:
    batch = [tokenized_examples[rng.randrange(len(tokenized_examples))] for _ in range(batch_size)]
    return collate_batch(batch, device=device, pad_token_id=pad_token_id)


def iter_batches(
    tokenized_examples: Sequence[Dict[str, object]],
    *,
    batch_size: int,
    device: torch.device,
    pad_token_id: int,
) -> Iterable[Dict[str, object]]:
    for start in range(0, len(tokenized_examples), batch_size):
        batch = tokenized_examples[start : start + batch_size]
        yield collate_batch(batch, device=device, pad_token_id=pad_token_id)
