from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import requests


DEFAULT_FASTCHAT_QUESTION_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)

JUDGE_INSTRUCTIONS = (
    "You are an impartial MT-Bench-style judge. Score the assistant's answer for each turn on a 1-10 scale. "
    "Judge helpfulness, relevance, correctness, depth, clarity, and turn-to-turn coherence. "
    "If reference answers are provided, use them as guidance for correctness, but do not require exact wording. "
    "Return JSON only with keys turn_scores, average_score, and rationale. "
    "turn_scores must be a list of objects with keys turn and score."
)


@dataclass(frozen=True)
class MTBenchQuestion:
    question_id: str
    category: str
    turns: List[str]
    reference: List[str]


def _jsonl_load(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open() as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _jsonl_append(path: Path, row: Dict[str, object]) -> None:
    with path.open("a") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _default_model_id(model_name: str) -> str:
    return model_name.rsplit("/", 1)[-1]


def _download_to_path(url: str, path: Path, timeout_seconds: float = 60.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    path.write_text(response.text)
    return path


def _load_fastchat_questions(path: Path) -> List[MTBenchQuestion]:
    questions: List[MTBenchQuestion] = []
    with path.open() as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            turns = [str(item).strip() for item in row.get("turns", []) if str(item).strip()]
            if not turns:
                continue
            reference = [str(item).strip() for item in row.get("reference", []) if str(item).strip()]
            questions.append(
                MTBenchQuestion(
                    question_id=str(row["question_id"]),
                    category=str(row.get("category") or "unknown"),
                    turns=turns,
                    reference=reference,
                )
            )
    return questions


def _build_questions_from_scalebio_records(records: Sequence[Dict[str, object]]) -> List[MTBenchQuestion]:
    questions: List[MTBenchQuestion] = []
    for index, record in enumerate(records):
        turns: List[str] = []
        reference: List[str] = []
        for item in record.get("items", []):
            role = str(item.get("role") or "").lower()
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                turns.append(content)
            elif role == "assistant":
                reference.append(content)
        if not turns:
            continue
        questions.append(
            MTBenchQuestion(
                question_id=f"scalebio_test_{index + 1}",
                category=str(record.get("category") or "unknown"),
                turns=turns,
                reference=reference,
            )
        )
    return questions


def load_mtbench_questions(
    *,
    source: str,
    question_file: str,
    question_url: str,
    question_limit: int,
    output_dir: Path,
    scale_bio_test_records: Sequence[Dict[str, object]],
) -> tuple[List[MTBenchQuestion], Dict[str, object]]:
    if source == "fastchat":
        if question_file:
            path = Path(question_file)
        else:
            path = output_dir / "mtbench_fastchat_questions.jsonl"
            if not path.exists():
                _download_to_path(question_url, path)
        questions = _load_fastchat_questions(path)
        metadata = {
            "source": "fastchat",
            "question_file": str(path),
            "question_url": question_url,
        }
    elif source == "scalebio_test":
        questions = _build_questions_from_scalebio_records(scale_bio_test_records)
        metadata = {
            "source": "scalebio_test",
            "question_file": None,
            "question_url": None,
        }
    else:
        raise ValueError(f"Unsupported MT-Bench question source: {source}")

    if question_limit > 0:
        questions = questions[:question_limit]
        metadata["question_limit"] = question_limit
    return questions, metadata


def _render_training_text_prompt(user_turns: Sequence[str], assistant_turns: Sequence[str]) -> str:
    parts: List[str] = []
    for index, user_turn in enumerate(user_turns):
        parts.append(f"User: {user_turn.strip()}\n\n")
        if index < len(assistant_turns):
            parts.append(f"Assistant: {assistant_turns[index].strip()}\n\n")
        else:
            parts.append("Assistant:")
    return "".join(parts)


def _render_chat_template_prompt(tokenizer, user_turns: Sequence[str], assistant_turns: Sequence[str]) -> str:
    messages: List[Dict[str, str]] = []
    for index, user_turn in enumerate(user_turns):
        messages.append({"role": "user", "content": user_turn.strip()})
        if index < len(assistant_turns):
            messages.append({"role": "assistant", "content": assistant_turns[index].strip()})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _render_prompt(
    *,
    tokenizer,
    prompt_style: str,
    user_turns: Sequence[str],
    assistant_turns: Sequence[str],
) -> str:
    if prompt_style == "chat_template":
        if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
            raise RuntimeError("Tokenizer does not expose a chat template, but mtbench_prompt_style=chat_template.")
        return _render_chat_template_prompt(tokenizer, user_turns, assistant_turns)
    return _render_training_text_prompt(user_turns, assistant_turns)


def generate_mtbench_answers(
    *,
    args,
    adapter_state: Dict[str, object],
    output_dir: Path,
    questions: Sequence[MTBenchQuestion],
    device,
) -> Dict[str, object]:
    from exact_math_reasoning import (
        assign_trainable_state,
        generate_batches,
        load_model_and_tokenizer,
    )

    answers_path = output_dir / "mtbench_model_answer.jsonl"
    model, tokenizer = load_model_and_tokenizer(args, device)
    assign_trainable_state(model, adapter_state)
    model.eval()

    answer_turns: Dict[str, List[str]] = {question.question_id: [] for question in questions}
    max_turn_count = max((len(question.turns) for question in questions), default=0)
    for turn_index in range(max_turn_count):
        active_questions: List[MTBenchQuestion] = []
        prompts: List[str] = []
        for question in questions:
            if turn_index >= len(question.turns):
                continue
            prior_answers = answer_turns[question.question_id]
            prompts.append(
                _render_prompt(
                    tokenizer=tokenizer,
                    prompt_style=args.mtbench_prompt_style,
                    user_turns=question.turns[: turn_index + 1],
                    assistant_turns=prior_answers,
                )
            )
            active_questions.append(question)
        if not prompts:
            continue
        outputs = generate_batches(
            model,
            tokenizer,
            prompts,
            max_input_length=args.mtbench_max_input_length,
            max_new_tokens=args.mtbench_max_new_tokens,
            batch_size=args.mtbench_generation_batch_size,
            device=device,
        )
        for question, output_text in zip(active_questions, outputs):
            answer_turns[question.question_id].append(output_text.strip())

    del model
    if device.type == "cuda":
        import torch

        torch.cuda.empty_cache()

    model_id = args.mtbench_model_id or _default_model_id(args.model_name)
    with answers_path.open("w") as file:
        for question in questions:
            row = {
                "question_id": question.question_id,
                "answer_id": uuid.uuid4().hex,
                "model_id": model_id,
                "category": question.category,
                "choices": [{"index": 0, "turns": answer_turns[question.question_id]}],
                "reference": question.reference,
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "answer_file": str(answers_path),
        "model_id": model_id,
        "question_count": len(questions),
        "turn_count": sum(len(item.turns) for item in questions),
    }


def _responses_output_text(payload: Dict[str, object]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    texts: List[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text_value = content.get("text")
            if isinstance(text_value, str):
                texts.append(text_value)
            elif isinstance(text_value, dict):
                maybe_value = text_value.get("value")
                if isinstance(maybe_value, str):
                    texts.append(maybe_value)
    return "\n".join(texts).strip()


def _extract_json_object(text: str) -> Dict[str, object]:
    def _sanitize(candidate: str) -> str:
        return "".join(
            char if (ord(char) >= 32 or char in "\n\r\t") else " "
            for char in candidate
        )

    text = text.strip()
    if not text:
        raise ValueError("Empty judge response.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return json.loads(_sanitize(text))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return json.loads(_sanitize(candidate))
    raise ValueError(f"Judge response does not contain a JSON object: {text[:200]}")


def _judge_user_prompt(
    *,
    question: MTBenchQuestion,
    answer_turns: Sequence[str],
    use_reference: bool,
) -> str:
    payload: Dict[str, object] = {
        "category": question.category,
        "question_id": question.question_id,
        "conversation": [
            {
                "turn": index + 1,
                "user": user_turn,
                "assistant": answer_turns[index] if index < len(answer_turns) else "",
            }
            for index, user_turn in enumerate(question.turns)
        ],
        "rubric": {
            "score_range": "1-10",
            "criteria": [
                "helpfulness",
                "relevance",
                "correctness",
                "depth",
                "clarity",
                "coherence with previous turns",
            ],
        },
    }
    if use_reference and question.reference:
        payload["reference"] = [
            {
                "turn": index + 1,
                "assistant": reference_turn,
            }
            for index, reference_turn in enumerate(question.reference)
        ]
    return (
        "Evaluate the following assistant conversation using MT-Bench-style single-answer grading. "
        "Return JSON only.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _post_openai_responses(
    *,
    api_base: str,
    api_key: str,
    judge_model: str,
    prompt: str,
    timeout_seconds: float,
) -> str:
    response = requests.post(
        api_base.rstrip("/") + "/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": judge_model,
            "instructions": JUDGE_INSTRUCTIONS,
            "input": prompt,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return _responses_output_text(response.json())


def _post_openai_chat_completions(
    *,
    api_base: str,
    api_key: str,
    judge_model: str,
    prompt: str,
    timeout_seconds: float,
) -> str:
    response = requests.post(
        api_base.rstrip("/") + "/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": judge_model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": JUDGE_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices", [])
    if not choices:
        raise RuntimeError(f"Empty chat completion response: {payload}")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Chat completion did not return text content: {payload}")
    return content


def _score_question_with_retries(
    *,
    api_base: str,
    api_key: str,
    judge_model: str,
    prompt: str,
    timeout_seconds: float,
    max_retries: int,
    sleep_base_seconds: float,
) -> Dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            try:
                output_text = _post_openai_responses(
                    api_base=api_base,
                    api_key=api_key,
                    judge_model=judge_model,
                    prompt=prompt,
                    timeout_seconds=timeout_seconds,
                )
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code in {400, 404, 405}:
                    output_text = _post_openai_chat_completions(
                        api_base=api_base,
                        api_key=api_key,
                        judge_model=judge_model,
                        prompt=prompt,
                        timeout_seconds=timeout_seconds,
                    )
                else:
                    raise
            return _extract_json_object(output_text)
        except Exception as exc:  # pragma: no cover - network retry path
            last_error = exc
            if attempt + 1 == max_retries:
                break
            sleep_seconds = sleep_base_seconds * (2**attempt)
            if isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code == 429:
                retry_after = exc.response.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_seconds = max(sleep_seconds, float(retry_after))
                    except ValueError:
                        pass
                sleep_seconds = max(sleep_seconds, 15.0)
            time.sleep(sleep_seconds)
    assert last_error is not None
    raise last_error


def summarize_mtbench_judgments(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    turn_scores: List[float] = []
    category_to_scores: Dict[str, List[float]] = {}
    question_scores: List[float] = []
    for row in rows:
        category = str(row.get("category") or "unknown")
        question_score = float(row.get("average_score") or 0.0)
        question_scores.append(question_score)
        category_to_scores.setdefault(category, []).append(question_score)
        for item in row.get("turn_scores", []):
            if isinstance(item, dict) and "score" in item:
                turn_scores.append(float(item["score"]))
    return {
        "question_count": len(rows),
        "turn_count": len(turn_scores),
        "mean_question_score": sum(question_scores) / max(1, len(question_scores)),
        "mean_turn_score": sum(turn_scores) / max(1, len(turn_scores)),
        "category_scores": {
            category: sum(scores) / len(scores)
            for category, scores in sorted(category_to_scores.items())
            if scores
        },
    }


def run_mtbench_judging(
    *,
    args,
    output_dir: Path,
    questions: Sequence[MTBenchQuestion],
    answer_file: str,
) -> Dict[str, object]:
    api_key = os.environ.get(args.mtbench_api_key_env, "").strip()
    if not api_key:
        return {
            "status": "skipped_missing_api_key",
            "judge_model": args.mtbench_judge_model,
            "api_key_env": args.mtbench_api_key_env,
        }

    answer_rows = _jsonl_load(Path(answer_file))
    answer_by_question = {
        str(row["question_id"]): row
        for row in answer_rows
    }
    question_by_id = {question.question_id: question for question in questions}
    judgments_path = output_dir / "mtbench_judgment.jsonl"
    existing_rows = _jsonl_load(judgments_path)
    judged_ids = {str(row["question_id"]) for row in existing_rows}

    for question_id, answer_row in answer_by_question.items():
        if question_id in judged_ids:
            continue
        question = question_by_id.get(question_id)
        if question is None:
            continue
        choices = answer_row.get("choices", [])
        if not choices:
            continue
        answer_turns = list(choices[0].get("turns", []))
        prompt = _judge_user_prompt(
            question=question,
            answer_turns=answer_turns,
            use_reference=args.mtbench_use_reference,
        )
        judge_payload = _score_question_with_retries(
            api_base=args.mtbench_api_base,
            api_key=api_key,
            judge_model=args.mtbench_judge_model,
            prompt=prompt,
            timeout_seconds=args.mtbench_judge_timeout_seconds,
            max_retries=args.mtbench_judge_max_retries,
            sleep_base_seconds=args.mtbench_judge_retry_base_seconds,
        )
        turn_scores = []
        for item in judge_payload.get("turn_scores", []):
            if isinstance(item, dict):
                try:
                    turn_scores.append(
                        {
                            "turn": int(item.get("turn") or len(turn_scores) + 1),
                            "score": float(item.get("score")),
                        }
                    )
                except (TypeError, ValueError):
                    continue
        average_score = judge_payload.get("average_score")
        if average_score is None:
            average_score = sum(item["score"] for item in turn_scores) / max(1, len(turn_scores))
        row = {
            "question_id": question.question_id,
            "model_id": answer_row.get("model_id"),
            "judge_model": args.mtbench_judge_model,
            "category": question.category,
            "turn_scores": turn_scores,
            "average_score": float(average_score),
            "rationale": str(judge_payload.get("rationale") or ""),
            "reference_used": bool(args.mtbench_use_reference and question.reference),
        }
        _jsonl_append(judgments_path, row)
        existing_rows.append(row)

    summary = summarize_mtbench_judgments(existing_rows)
    summary.update(
        {
            "status": "completed",
            "judge_model": args.mtbench_judge_model,
            "judgment_file": str(judgments_path),
        }
    )
    summary_path = output_dir / "mtbench_summary.json"
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)
    summary["summary_file"] = str(summary_path)
    return summary


def maybe_run_mtbench_evaluation(
    *,
    args,
    adapter_state: Dict[str, object],
    splits: Dict[str, object],
    output_dir: Path,
    device,
) -> Dict[str, object]:
    if not args.run_mtbench_eval and not args.run_mtbench_judge:
        return {"status": "disabled"}

    mtbench_dir = output_dir / "mtbench_eval"
    mtbench_dir.mkdir(parents=True, exist_ok=True)
    questions, question_meta = load_mtbench_questions(
        source=args.mtbench_question_source,
        question_file=args.mtbench_question_file,
        question_url=args.mtbench_question_url,
        question_limit=args.mtbench_question_limit,
        output_dir=mtbench_dir,
        scale_bio_test_records=splits["test"],
    )
    generation_summary = generate_mtbench_answers(
        args=args,
        adapter_state=adapter_state,
        output_dir=mtbench_dir,
        questions=questions,
        device=device,
    )
    for stale_path in [mtbench_dir / "mtbench_judgment.jsonl", mtbench_dir / "mtbench_summary.json"]:
        stale_path.unlink(missing_ok=True)
    result = {
        "status": "generated",
        "question_source": question_meta,
        "generation": generation_summary,
    }
    if args.run_mtbench_judge:
        result["judging"] = run_mtbench_judging(
            args=args,
            output_dir=mtbench_dir,
            questions=questions,
            answer_file=generation_summary["answer_file"],
        )
    return result
