"""Text-classification benchmark adapter from Meta-Harness.

The benchmark surface follows the Meta-Harness reference example, while the
runner, model client, result schema, and Pareto output use OptiHarness code.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from memomemo.model import DEFAULT_BASE_URL, DEFAULT_MODEL, LocalModelClient
from memomemo.pareto import ParetoPoint, save_frontier
from memomemo.schemas import CandidateResult
from memomemo.utils.text import estimate_tokens


ALL_TEXT_CLASSIFICATION_TASKS = (
    "USPTO",
    "USPTO50K",
    "Symptom2Disease",
    "S2D",
    "LawBench",
    "Law",
    "USPTO_IPC_SECTION",
)
DEFAULT_TEXT_CLASSIFICATION_DATASETS = ("USPTO", "Symptom2Disease", "LawBench")
DEFAULT_TEXT_CLASSIFICATION_MEMORY_SYSTEMS = ("no_memory", "fewshot_all")
DEFAULT_TEXT_CLASSIFICATION_BASELINES = DEFAULT_TEXT_CLASSIFICATION_MEMORY_SYSTEMS
DEFAULT_TEXT_CLASSIFICATION_SEEDS = (42,)
DEFAULT_TEXT_CLASSIFICATION_SPLITS = {
    "num_train": 50,
    "num_val": 30,
    "num_test": 100,
}
DEFAULT_TEXT_CLASSIFICATION_SPLITS_BY_DATASET = {
    "USPTO": {"num_train": 50, "num_val": 30, "num_test": 100},
    "SYMPTOM2DISEASE": {"num_train": 200, "num_val": 50, "num_test": 212},
    "LAWBENCH": {"num_train": 200, "num_val": 50, "num_test": 100},
    "USPTO_IPC_SECTION": DEFAULT_TEXT_CLASSIFICATION_SPLITS,
}

USPTO_IPC_DATASET_ID = "ufukhaman/uspto_balanced_200k_ipc_classification"
USPTO50K_DATA_ENV = "MEMOMEMO_USPTO50K_DATA_DIR"
USPTO50K_VENDOR_DATA = Path("references/vendor/mce-artifact/env/uspto/data")
SYMPTOM2DISEASE_DATA_ENV = "MEMOMEMO_SYMPTOM2DISEASE_DATA_DIR"
SYMPTOM2DISEASE_VENDOR_DATA = Path(
    "references/vendor/mce-artifact/env/symptom_diagnosis/data"
)
LAWBENCH_DATA_ENV = "MEMOMEMO_LAWBENCH_DATA_DIR"
LAWBENCH_VENDOR_DATA = Path("references/vendor/mce-artifact/env/crime_prediction/data")
USPTO_MAX_TEXT_CHARS = 6000
IPC_SECTION_DESCRIPTIONS = {
    "A": "Human Necessities",
    "B": "Performing Operations; Transporting",
    "C": "Chemistry; Metallurgy",
    "D": "Textiles; Paper",
    "E": "Fixed Constructions",
    "F": "Mechanical Engineering; Lighting; Heating; Weapons; Blasting",
    "G": "Physics",
    "H": "Electricity",
}


@dataclass(frozen=True)
class ClassificationExample:
    """One text-classification example."""

    task_id: str
    input: str
    target: str
    raw_question: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_batch_result(
        self,
        *,
        prediction: str,
        was_correct: bool,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "input": self.input,
            "prediction": prediction,
            "ground_truth": self.target,
            "was_correct": was_correct,
            "metadata": metadata or {},
        }
        if self.raw_question:
            payload["raw_question"] = self.raw_question
        payload.update(self.metadata)
        return payload


@dataclass(frozen=True)
class TextClassificationSplits:
    """Train/validation/test splits plus evaluator."""

    train: list[ClassificationExample]
    val: list[ClassificationExample]
    test: list[ClassificationExample]
    evaluator: Callable[..., bool]


class PromptLLM:
    """Callable prompt wrapper over OptiHarness's OpenAI-compatible client."""

    def __init__(
        self,
        *,
        client: LocalModelClient,
        dry_run: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self.client = client
        self.dry_run = dry_run
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._lock = threading.Lock()

    def __call__(self, prompt: str) -> str:
        if self.dry_run:
            content = json.dumps(
                {
                    "reasoning": "dry_run",
                    "final_answer": _dry_run_classification_answer(prompt),
                }
            )
            prompt_tokens = estimate_tokens(prompt)
            completion_tokens = estimate_tokens(content)
        else:
            response = self.client.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            content = response.content
            prompt_tokens = response.prompt_tokens
            completion_tokens = response.completion_tokens

        with self._lock:
            self.total_calls += 1
            self.total_input_tokens += int(prompt_tokens)
            self.total_output_tokens += int(completion_tokens)
        return content

    def get_usage(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self.total_calls,
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
            }


def extract_json_field(text: str, field: str, default: str = "") -> str:
    """Extract a JSON field from direct JSON, fenced JSON, or loose text."""

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return str(data.get(field, default))
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE):
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return str(data.get(field, default))
        except json.JSONDecodeError:
            pass

    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 1
        pos = start + 1
        in_str = False
        while pos < len(text) and depth > 0:
            current = text[pos]
            if current == '"' and (pos == 0 or text[pos - 1] != "\\"):
                in_str = not in_str
            elif not in_str:
                if current == "{":
                    depth += 1
                elif current == "}":
                    depth -= 1
            pos += 1
        if depth == 0:
            candidate = re.sub(r",\s*([\]}])", r"\1", text[start:pos])
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return str(data.get(field, default))
            except json.JSONDecodeError:
                pass

    matches = re.findall(rf'"{re.escape(field)}"\s*:\s*"([^"]*)"', text)
    return matches[-1] if matches else default


class ClassificationMemorySystem:
    """Meta-Harness-style memory system interface."""

    def __init__(self, llm: PromptLLM) -> None:
        self._llm = llm
        self._prompt_local = threading.local()

    def call_llm(self, prompt: str) -> str:
        self._prompt_local.last_prompt_len = len(prompt)
        self._prompt_local.last_prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        self._prompt_local.last_prompt_text = prompt
        return self._llm(prompt)

    def get_last_prompt_info(self) -> dict[str, Any]:
        return {
            "prompt_len": getattr(self._prompt_local, "last_prompt_len", None),
            "prompt_hash": getattr(self._prompt_local, "last_prompt_hash", None),
            "prompt_text": getattr(self._prompt_local, "last_prompt_text", None),
        }

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    def get_context_length(self) -> int:
        return len(self.get_state())

    def get_state(self) -> str:
        raise NotImplementedError

    def set_state(self, state: str) -> None:
        raise NotImplementedError


NO_MEMORY_PROMPT = """Answer the following question.

{input}

Respond in this exact JSON format:
{{"reasoning": "[short reasoning]", "final_answer": "[answer]"}}"""


class NoMemoryClassification(ClassificationMemorySystem):
    """No-memory baseline from the Meta-Harness text-classification example."""

    name = "no_memory"

    def __init__(self, llm: PromptLLM) -> None:
        super().__init__(llm)
        self._state = "{}"

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        response = self.call_llm(NO_MEMORY_PROMPT.format(input=input))
        answer = extract_json_field(response, "final_answer")
        return answer, {"full_response": response}

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        return None

    def get_state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        self._state = state


FEWSHOT_PROMPT = """Solve the classification problem below based on the examples provided.

{examples_section}

Problem:
{input}

Instructions:
- Follow the patterns shown in the examples above.
- Respond in JSON format.

{{"reasoning": "[short reasoning]", "final_answer": "[answer]"}}"""


class FewShotClassification(ClassificationMemorySystem):
    """Few-shot memory baseline using accumulated labeled examples."""

    name = "fewshot"

    def __init__(
        self,
        llm: PromptLLM,
        *,
        max_examples: int = 50,
        max_chars: int = 30000,
    ) -> None:
        super().__init__(llm)
        self.max_examples = int(max_examples)
        self.max_chars = int(max_chars)
        self.examples: list[dict[str, str]] = []

    def _format_examples_section(self, *, seed: int | None = None) -> str:
        if not self.examples:
            return ""
        if seed is not None and len(self.examples) > self.max_examples:
            rng = random.Random(seed)
            selected = rng.sample(self.examples, self.max_examples)
        else:
            selected = self.examples[-self.max_examples :]
            if seed is not None:
                rng = random.Random(seed)
                selected = list(selected)
                rng.shuffle(selected)

        parts: list[str] = []
        total_chars = 0
        for example in selected:
            question = example.get("raw_question") or example["input"]
            part = f"Q: {question}\nA: {example['target']}"
            if total_chars + len(part) > self.max_chars:
                break
            parts.append(part)
            total_chars += len(part) + 2
        return "\n\n".join(parts)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        seed = int(hashlib.sha256(input.encode("utf-8")).hexdigest()[:8], 16)
        examples_section = self._format_examples_section(seed=seed)
        response = self.call_llm(
            FEWSHOT_PROMPT.format(examples_section=examples_section, input=input)
        )
        answer = extract_json_field(response, "final_answer")
        return answer, {
            "full_response": response,
            "num_examples": len(self.examples),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for result in batch_results:
            example = {"input": result["input"], "target": result["ground_truth"]}
            if "raw_question" in result:
                example["raw_question"] = result["raw_question"]
            self.examples.append(example)

    def get_context_length(self) -> int:
        return len(self._format_examples_section())

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples}, indent=2)

    def set_state(self, state: str) -> None:
        data = json.loads(state)
        self.examples = list(data.get("examples", []))


class FewShotAllClassification(FewShotClassification):
    """Few-shot baseline using all training examples up to the context cap."""

    name = "fewshot_all"

    def __init__(self, llm: PromptLLM) -> None:
        super().__init__(llm, max_examples=9999)


TEXT_CLASSIFICATION_MEMORY_REGISTRY = {
    NoMemoryClassification.name: NoMemoryClassification,
    FewShotClassification.name: FewShotClassification,
    FewShotAllClassification.name: FewShotAllClassification,
}


def available_text_classification_memories() -> tuple[str, ...]:
    return tuple(sorted(TEXT_CLASSIFICATION_MEMORY_REGISTRY))


def build_text_classification_memory(
    name: str,
    llm: PromptLLM,
) -> ClassificationMemorySystem:
    try:
        cls = TEXT_CLASSIFICATION_MEMORY_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(available_text_classification_memories())
        raise ValueError(f"unknown text-classification memory {name!r}; available: {available}") from exc
    return cls(llm)


def load_text_classification_splits(
    dataset: str,
    *,
    num_train: int | None = None,
    num_val: int | None = None,
    num_test: int | None = None,
    shuffle_seed: int = 42,
) -> TextClassificationSplits:
    """Load deterministic splits for a supported text-classification dataset."""

    normalized = dataset.strip().upper().replace("-", "_")
    num_train, num_val, num_test = _resolve_split_sizes(
        normalized,
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
    )
    if normalized in {"USPTO", "USPTO50K", "USPTO_50K"}:
        return _load_uspto50k_splits(
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            shuffle_seed=shuffle_seed,
        )
    if normalized in {"S2D", "SYMPTOM2DISEASE", "SYMPTOM_2_DISEASE"}:
        return _load_symptom2disease_splits(
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            shuffle_seed=shuffle_seed,
        )
    if normalized in {"LAW", "LAWBENCH", "CRIME_PREDICTION"}:
        return _load_lawbench_splits(
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            shuffle_seed=shuffle_seed,
        )
    if normalized in {"USPTO_IPC", "USPTO_IPC_SECTION", "USPTO_PATENT_IPC"}:
        return _load_uspto_ipc_splits(
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            shuffle_seed=shuffle_seed,
        )
    raise ValueError(
        f"Dataset {dataset!r} is not available. "
        f"Available: {', '.join(ALL_TEXT_CLASSIFICATION_TASKS)}."
    )


def _canonical_dataset_key(normalized: str) -> str:
    if normalized in {"USPTO", "USPTO50K", "USPTO_50K"}:
        return "USPTO"
    if normalized in {"S2D", "SYMPTOM2DISEASE", "SYMPTOM_2_DISEASE"}:
        return "SYMPTOM2DISEASE"
    if normalized in {"LAW", "LAWBENCH", "CRIME_PREDICTION"}:
        return "LAWBENCH"
    if normalized in {"USPTO_IPC", "USPTO_IPC_SECTION", "USPTO_PATENT_IPC"}:
        return "USPTO_IPC_SECTION"
    return normalized


def _resolve_split_sizes(
    normalized_dataset: str,
    *,
    num_train: int | None,
    num_val: int | None,
    num_test: int | None,
) -> tuple[int, int, int]:
    defaults = DEFAULT_TEXT_CLASSIFICATION_SPLITS_BY_DATASET.get(
        _canonical_dataset_key(normalized_dataset),
        DEFAULT_TEXT_CLASSIFICATION_SPLITS,
    )
    return (
        defaults["num_train"] if num_train is None else int(num_train),
        defaults["num_val"] if num_val is None else int(num_val),
        defaults["num_test"] if num_test is None else int(num_test),
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _uspto50k_data_dir() -> Path:
    env_value = os.environ.get(USPTO50K_DATA_ENV)
    if env_value:
        return Path(env_value).expanduser()
    return _project_root() / USPTO50K_VENDOR_DATA


def _artifact_data_dir(env_var: str, default_rel: Path) -> Path:
    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value).expanduser()
    return _project_root() / default_rel


def _require_jsonl_data_dir(data_dir: Path, *, dataset_name: str, env_var: str) -> None:
    missing = [
        name
        for name in ("train.jsonl", "val.jsonl", "test.jsonl")
        if not (data_dir / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{dataset_name} data is missing. Fetch the MCE artifact with "
            "`scripts/fetch_reference_repos.sh`, or set "
            f"`{env_var}` to a directory containing train.jsonl, val.jsonl, "
            f"and test.jsonl. Looked in: {data_dir}"
        )


def _read_jsonl_classification_split(
    *,
    data_dir: Path,
    filename: str,
    split: str,
    limit: int,
    dataset_tag: str,
    input_key: str,
    target_key: str,
    format_input: Callable[[dict[str, Any]], str],
    metadata: dict[str, Any],
) -> list[ClassificationExample]:
    if limit < 0:
        raise ValueError("split sizes must be non-negative")
    examples: list[ClassificationExample] = []
    with (data_dir / filename).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if len(examples) >= limit:
                break
            row = json.loads(line)
            question = format_input(row)
            examples.append(
                ClassificationExample(
                    task_id=f"{dataset_tag}::{split}::{index}",
                    input=question,
                    target=str(row[target_key]),
                    raw_question=str(row[input_key]),
                    metadata=dict(metadata),
                )
            )
    if len(examples) < limit:
        raise ValueError(
            f"requested {limit} {split} examples from {dataset_tag}, "
            f"but only found {len(examples)} in {data_dir / filename}"
        )
    return examples


def _load_uspto50k_splits(
    *,
    num_train: int,
    num_val: int,
    num_test: int,
    shuffle_seed: int,
) -> TextClassificationSplits:
    del shuffle_seed  # The MCE artifact ships fixed train/val/test JSONL splits.

    data_dir = _uspto50k_data_dir()
    _require_jsonl_data_dir(
        data_dir,
        dataset_name="USPTO-50k",
        env_var=USPTO50K_DATA_ENV,
    )

    def read_split(filename: str, split: str, limit: int) -> list[ClassificationExample]:
        return _read_jsonl_classification_split(
            data_dir=data_dir,
            filename=filename,
            split=split,
            limit=limit,
            dataset_tag="USPTO50K",
            input_key="question",
            target_key="target",
            format_input=lambda row: str(row["question"]),
            metadata={"source": "mce-artifact", "task": "retrosynthesis"},
        )

    return TextClassificationSplits(
        train=read_split("train.jsonl", "train", num_train),
        val=read_split("val.jsonl", "val", num_val),
        test=read_split("test.jsonl", "test", num_test),
        evaluator=_uspto50k_evaluator,
    )


def _load_symptom2disease_splits(
    *,
    num_train: int,
    num_val: int,
    num_test: int,
    shuffle_seed: int,
) -> TextClassificationSplits:
    del shuffle_seed
    data_dir = _artifact_data_dir(
        SYMPTOM2DISEASE_DATA_ENV,
        SYMPTOM2DISEASE_VENDOR_DATA,
    )
    _require_jsonl_data_dir(
        data_dir,
        dataset_name="Symptom2Disease",
        env_var=SYMPTOM2DISEASE_DATA_ENV,
    )

    instruction = (
        "You are an expert medical diagnostician. Based on the patient's symptoms, "
        "provide the most likely diagnosis.\n\n"
        "Possible diagnoses include: drug reaction, allergy, chicken pox, diabetes, "
        "psoriasis, hypertension, cervical spondylosis, bronchial asthma, varicose "
        "veins, malaria, dengue, arthritis, impetigo, fungal infection, common cold, "
        "gastroesophageal reflux disease, urinary tract infection, typhoid, pneumonia, "
        "peptic ulcer disease, jaundice, migraine.\n\n"
        "Please respond with the diagnosis name."
    )

    def format_input(row: dict[str, Any]) -> str:
        return f"{instruction}\n\nPatient symptoms:\n{row['question']}\n\nAnswer:"

    def read_split(filename: str, split: str, limit: int) -> list[ClassificationExample]:
        return _read_jsonl_classification_split(
            data_dir=data_dir,
            filename=filename,
            split=split,
            limit=limit,
            dataset_tag="Symptom2Disease",
            input_key="question",
            target_key="answer",
            format_input=format_input,
            metadata={"source": "mce-artifact", "task": "symptom_diagnosis"},
        )

    return TextClassificationSplits(
        train=read_split("train.jsonl", "train", num_train),
        val=read_split("val.jsonl", "val", num_val),
        test=read_split("test.jsonl", "test", num_test),
        evaluator=_symptom2disease_evaluator,
    )


def _load_lawbench_splits(
    *,
    num_train: int,
    num_val: int,
    num_test: int,
    shuffle_seed: int,
) -> TextClassificationSplits:
    del shuffle_seed
    data_dir = _artifact_data_dir(LAWBENCH_DATA_ENV, LAWBENCH_VENDOR_DATA)
    _require_jsonl_data_dir(
        data_dir,
        dataset_name="LawBench crime prediction",
        env_var=LAWBENCH_DATA_ENV,
    )

    def format_input(row: dict[str, Any]) -> str:
        instruction = str(
            row.get(
                "instruction",
                "请你模拟法官依据下面事实给出罪名，将答案写在[罪名]和<eoa>之间。",
            )
        )
        return f"{instruction}\n\n案件事实:\n{row['question']}\n\n答案:"

    def read_split(filename: str, split: str, limit: int) -> list[ClassificationExample]:
        return _read_jsonl_classification_split(
            data_dir=data_dir,
            filename=filename,
            split=split,
            limit=limit,
            dataset_tag="LawBench",
            input_key="question",
            target_key="answer",
            format_input=format_input,
            metadata={"source": "mce-artifact", "task": "crime_prediction"},
        )

    return TextClassificationSplits(
        train=read_split("train.jsonl", "train", num_train),
        val=read_split("val.jsonl", "val", num_val),
        test=read_split("test.jsonl", "test", num_test),
        evaluator=_lawbench_evaluator,
    )


def _load_uspto_ipc_splits(
    *,
    num_train: int,
    num_val: int,
    num_test: int,
    shuffle_seed: int,
) -> TextClassificationSplits:
    return _load_uspto_ipc_hf_splits(
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        shuffle_seed=shuffle_seed,
    )


def run_text_classification_benchmark(
    *,
    out_dir: Path,
    datasets: Iterable[str] = DEFAULT_TEXT_CLASSIFICATION_DATASETS,
    memory_systems: Iterable[str] = DEFAULT_TEXT_CLASSIFICATION_MEMORY_SYSTEMS,
    seeds: Iterable[int] = DEFAULT_TEXT_CLASSIFICATION_SEEDS,
    num_train: int | None = None,
    num_val: int | None = None,
    num_test: int | None = None,
    mode: str = "offline",
    num_epochs: int = 1,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = "EMPTY",
    timeout_s: int = 300,
    dry_run: bool = False,
    temperature: float = 0.0,
    max_eval_workers: int = 1,
    force: bool = False,
    split_loader: Callable[..., TextClassificationSplits] = load_text_classification_splits,
    pareto_quality_threshold: float = 0.0,
) -> dict[str, Any]:
    """Run text-classification benchmark rows and aggregate OptiHarness results."""

    if mode not in {"online", "offline"}:
        raise ValueError("mode must be 'online' or 'offline'")

    out_dir.mkdir(parents=True, exist_ok=True)
    row_dir = out_dir / "rows"
    row_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = out_dir / "candidate_results"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    selected_datasets = [str(item) for item in datasets]
    selected_memories = [str(item) for item in memory_systems]
    selected_seeds = [int(item) for item in seeds]
    started = time.time()
    rows: list[dict[str, Any]] = []

    for dataset in selected_datasets:
        for seed in selected_seeds:
            splits = split_loader(
                dataset,
                num_train=num_train,
                num_val=num_val,
                num_test=num_test,
                shuffle_seed=seed,
            )
            for memory_name in selected_memories:
                result_path = _row_result_path(
                    row_dir=row_dir,
                    dataset=dataset,
                    memory_name=memory_name,
                    model=model,
                    seed=seed,
                )
                if result_path.exists() and not force:
                    rows.append(json.loads(result_path.read_text(encoding="utf-8")))
                    continue

                client = LocalModelClient(
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    timeout_s=timeout_s,
                )
                llm = PromptLLM(
                    client=client,
                    dry_run=dry_run,
                    temperature=temperature,
                )
                memory = build_text_classification_memory(memory_name, llm)
                row = _run_one_text_classification_row(
                    dataset=dataset,
                    seed=seed,
                    memory_name=memory_name,
                    memory=memory,
                    llm=llm,
                    splits=splits,
                    mode=mode,
                    num_epochs=num_epochs,
                    model=model,
                    base_url=base_url,
                    dry_run=dry_run,
                    max_eval_workers=max_eval_workers,
                )
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
                rows.append(row)

    candidates = _aggregate_text_classification_candidates(
        rows,
        candidate_dir=candidate_dir,
        metric="test_accuracy" if (num_test is None or num_test > 0) else "val_accuracy",
    )
    frontier_path = out_dir / "pareto_frontier.json"
    save_frontier(
        frontier_path,
        [
            ParetoPoint(
                candidate_id=item.candidate_id,
                scaffold_name=item.scaffold_name,
                passrate=item.passrate,
                token_consuming=item.token_consuming,
                avg_token_consuming=item.avg_token_consuming,
                average_score=item.average_score,
                result_path=item.result_path,
                config=item.config,
            )
            for item in candidates
        ],
        quality_gap_threshold=pareto_quality_threshold,
    )

    summary = {
        "benchmark": "text_classification",
        "out_dir": str(out_dir),
        "datasets": selected_datasets,
        "memory_systems": selected_memories,
        "seeds": selected_seeds,
        "num_train": num_train,
        "num_val": num_val,
        "num_test": num_test,
        "split_defaults": DEFAULT_TEXT_CLASSIFICATION_SPLITS_BY_DATASET,
        "mode": mode,
        "num_epochs": num_epochs if mode == "offline" else None,
        "dry_run": dry_run,
        "model": model,
        "base_url": base_url,
        "max_eval_workers": max_eval_workers,
        "duration_s": time.time() - started,
        "row_count": len(rows),
        "candidate_count": len(candidates),
        "rows": rows,
        "candidates": [candidate.to_dict() for candidate in candidates],
        "pareto_frontier_path": str(frontier_path),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def evaluate_text_classification_memory(
    *,
    memory: ClassificationMemorySystem,
    llm: PromptLLM,
    splits: TextClassificationSplits,
    dataset: str,
    seed: int,
    memory_name: str,
    mode: str,
    num_epochs: int,
    model: str,
    base_url: str,
    dry_run: bool,
    max_eval_workers: int,
) -> dict[str, Any]:
    """Evaluate one already-instantiated text-classification memory system."""

    return _run_one_text_classification_row(
        dataset=dataset,
        seed=seed,
        memory_name=memory_name,
        memory=memory,
        llm=llm,
        splits=splits,
        mode=mode,
        num_epochs=num_epochs,
        model=model,
        base_url=base_url,
        dry_run=dry_run,
        max_eval_workers=max_eval_workers,
    )


def _run_one_text_classification_row(
    *,
    dataset: str,
    seed: int,
    memory_name: str,
    memory: ClassificationMemorySystem,
    llm: PromptLLM,
    splits: TextClassificationSplits,
    mode: str,
    num_epochs: int,
    model: str,
    base_url: str,
    dry_run: bool,
    max_eval_workers: int,
) -> dict[str, Any]:
    started = time.time()
    train_result = _train_memory(
        memory,
        examples=splits.train,
        evaluator=splits.evaluator,
        mode=mode,
        num_epochs=num_epochs,
        max_workers=max_eval_workers,
    )
    val_result = _evaluate_memory(
        memory,
        examples=splits.val,
        evaluator=splits.evaluator,
        max_workers=max_eval_workers,
    )
    test_result = _evaluate_memory(
        memory,
        examples=splits.test,
        evaluator=splits.evaluator,
        max_workers=max_eval_workers,
    )
    usage = llm.get_usage()
    context_values = [
        item["context_len"]
        for item in val_result["predictions"] + test_result["predictions"]
    ]
    avg_context_chars = int(sum(context_values) / len(context_values)) if context_values else 0
    return {
        "benchmark": "text_classification",
        "dataset": dataset,
        "memory": memory_name,
        "model": model,
        "base_url": base_url,
        "seed": seed,
        "mode": mode,
        "num_epochs": num_epochs if mode == "offline" else None,
        "dry_run": dry_run,
        "timestamp": datetime.now().isoformat(),
        "runtime_seconds": round(time.time() - started, 2),
        "train_accuracy": train_result["accuracy"],
        "train_correct": train_result["correct"],
        "train_total": train_result["total"],
        "val_accuracy": val_result["accuracy"],
        "val_correct": val_result["correct"],
        "val_total": val_result["total"],
        "test_accuracy": test_result["accuracy"],
        "test_correct": test_result["correct"],
        "test_total": test_result["total"],
        "memory_context_chars": avg_context_chars,
        "llm_calls": usage["calls"],
        "llm_input_tokens": usage["input_tokens"],
        "llm_output_tokens": usage["output_tokens"],
        "llm_total_tokens": usage["total_tokens"],
        "memory_state": memory.get_state(),
    }


def _train_memory(
    memory: ClassificationMemorySystem,
    *,
    examples: list[ClassificationExample],
    evaluator: Callable[..., bool],
    mode: str,
    num_epochs: int,
    max_workers: int,
) -> dict[str, Any]:
    if mode == "offline":
        for _epoch in range(max(1, int(num_epochs))):
            for example in examples:
                memory.learn_from_batch(
                    [
                        example.to_batch_result(
                            prediction=example.target,
                            was_correct=True,
                        )
                    ]
                )
        return {
            "accuracy": None,
            "correct": None,
            "total": len(examples),
        }

    correct = 0
    for example in examples:
        prediction, metadata = memory.predict(example.input)
        ok = bool(evaluator(prediction, example.target, **example.metadata))
        correct += int(ok)
        memory.learn_from_batch(
            [
                example.to_batch_result(
                    prediction=prediction,
                    was_correct=ok,
                    metadata=metadata,
                )
            ]
        )
    total = len(examples)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
    }


def _evaluate_memory(
    memory: ClassificationMemorySystem,
    *,
    examples: list[ClassificationExample],
    evaluator: Callable[..., bool],
    max_workers: int,
) -> dict[str, Any]:
    if not examples:
        return {"accuracy": 0.0, "correct": 0, "total": 0, "predictions": []}

    def predict_one(index: int, example: ClassificationExample) -> tuple[int, dict[str, Any]]:
        prediction, metadata = memory.predict(example.input)
        prompt_info = memory.get_last_prompt_info()
        prompt_len = int(prompt_info.get("prompt_len") or 0)
        ok = bool(evaluator(prediction, example.target, **example.metadata))
        return index, {
            "task_id": example.task_id,
            "prediction": prediction,
            "target": example.target,
            "was_correct": ok,
            "prompt_len": prompt_len,
            "context_len": max(0, prompt_len - len(example.input)) if prompt_len else 0,
            "prompt_hash": prompt_info.get("prompt_hash"),
            "metadata": metadata,
        }

    results: list[dict[str, Any] | None] = [None] * len(examples)
    workers = min(max(1, int(max_workers)), len(examples))
    if workers == 1:
        for index, example in enumerate(examples):
            _, result = predict_one(index, example)
            results[index] = result
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(predict_one, index, example): index
                for index, example in enumerate(examples)
            }
            for future in as_completed(futures):
                index, result = future.result()
                results[index] = result

    predictions = [item for item in results if item is not None]
    correct = sum(1 for item in predictions if item["was_correct"])
    return {
        "accuracy": correct / len(predictions) if predictions else 0.0,
        "correct": correct,
        "total": len(predictions),
        "predictions": predictions,
    }


def _aggregate_text_classification_candidates(
    rows: list[dict[str, Any]],
    *,
    candidate_dir: Path,
    metric: str,
) -> list[CandidateResult]:
    by_memory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_memory[str(row["memory"])].append(row)

    candidates: list[CandidateResult] = []
    for memory_name, memory_rows in sorted(by_memory.items()):
        accuracies = [float(row.get(metric, 0.0) or 0.0) for row in memory_rows]
        counts = [int(row.get("test_total" if metric.startswith("test") else "val_total", 0)) for row in memory_rows]
        total_tokens = sum(int(row.get("llm_total_tokens", 0) or 0) for row in memory_rows)
        total_count = sum(counts)
        passrate = sum(accuracies) / len(accuracies) if accuracies else 0.0
        candidate = CandidateResult(
            candidate_id=f"text_classification_{memory_name}",
            scaffold_name=memory_name,
            passrate=passrate,
            average_score=passrate,
            token_consuming=total_tokens,
            avg_token_consuming=(total_tokens / total_count if total_count else 0.0),
            avg_prompt_tokens=(
                sum(int(row.get("llm_input_tokens", 0) or 0) for row in memory_rows) / total_count
                if total_count
                else 0.0
            ),
            avg_completion_tokens=(
                sum(int(row.get("llm_output_tokens", 0) or 0) for row in memory_rows) / total_count
                if total_count
                else 0.0
            ),
            count=total_count,
            config={
                "benchmark": "text_classification",
                "memory_system": memory_name,
                "metric": metric,
                "rows": len(memory_rows),
            },
            result_path=str(candidate_dir / f"{memory_name}.json"),
        )
        payload = {
            "candidate": candidate.to_dict(),
            "rows": memory_rows,
        }
        Path(candidate.result_path).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        candidates.append(candidate)
    return candidates


def _load_uspto_ipc_hf_splits(
    *,
    num_train: int,
    num_val: int,
    num_test: int,
    shuffle_seed: int,
) -> TextClassificationSplits:
    try:
        from datasets import ClassLabel, Dataset, load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise ImportError(
            "The text-classification benchmark needs Hugging Face datasets. "
            "Install with `python -m pip install -e '.[benchmark]'`."
        ) from exc

    ds = load_dataset(USPTO_IPC_DATASET_ID)
    train_source = ds["train"]
    test_source = ds["test"] if "test" in ds else ds["train"]

    def select(source: Dataset, n: int, seed: int) -> Dataset:
        if n < 0:
            raise ValueError("split sizes must be non-negative")
        if n > len(source):
            raise ValueError(f"requested {n} examples, but split only has {len(source)}")
        return source.shuffle(seed=seed).select(range(n)) if n else source.select([])

    train_val = select(train_source, num_train + num_val, shuffle_seed)
    train_rows = train_val.select(range(num_train)) if num_train else train_val.select([])
    val_rows = (
        train_val.select(range(num_train, num_train + num_val))
        if num_val
        else train_val.select([])
    )
    test_rows = select(test_source, num_test, shuffle_seed + 1)
    features = train_source.features

    def label_name(label: Any) -> str:
        feature = features.get("label")
        if isinstance(feature, ClassLabel):
            return feature.int2str(int(label))
        return str(label)

    def format_row(row: dict[str, Any], split: str, index: int) -> ClassificationExample:
        target = label_name(row["label"]).strip().upper()
        label_lines = "\n".join(
            f"{letter}: {description}"
            for letter, description in IPC_SECTION_DESCRIPTIONS.items()
        )
        patent_text = _truncate_text(str(row["text"]))
        raw_question = (
            "Classify this patent into one IPC section letter (A-H).\n\n"
            f"Patent text:\n{patent_text}"
        )
        return ClassificationExample(
            task_id=f"USPTO::{split}::{shuffle_seed}::{index}",
            input=(
                "You are classifying patents by IPC section.\n\n"
                "Choose exactly one section letter from these options:\n"
                f"{label_lines}\n\n"
                f"Patent text:\n{patent_text}\n\n"
                "Answer with only the IPC section letter."
            ),
            target=target,
            raw_question=raw_question,
            metadata={
                "ipc_class": str(row.get("ipc_class", "")),
                "subclass": str(row.get("subclass", "")),
            },
        )

    return TextClassificationSplits(
        train=[format_row(row, "train", idx) for idx, row in enumerate(train_rows)],
        val=[format_row(row, "val", idx) for idx, row in enumerate(val_rows)],
        test=[format_row(row, "test", idx) for idx, row in enumerate(test_rows)],
        evaluator=_uspto_evaluator,
    )


def _parse_uspto50k_reactants(text: str) -> set[str]:
    answer = extract_json_field(text, "final_answer", default=text)
    answer = re.sub(r"```(?:JSON)?|```", "", answer, flags=re.IGNORECASE).strip()
    return {part.strip().lower() for part in answer.split(".") if part.strip()}


def _uspto50k_evaluator(prediction: str, target: str, **_: Any) -> bool:
    return _parse_uspto50k_reactants(prediction) == _parse_uspto50k_reactants(target)


def _extract_symptom_diagnosis(text: str) -> str:
    answer = extract_json_field(text, "final_answer", default="")
    if not answer:
        match = re.search(r"\[DIAGNOSIS\](.*?)\[/DIAGNOSIS\]", text, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1)
    if not answer:
        match = re.search(r"(?:diagnosis|final diagnosis|conclusion)[:：]\s*([^\n]+)", text, re.IGNORECASE)
        if match:
            answer = match.group(1)
    if not answer:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        answer = lines[-1] if lines else text
    return re.sub(r"[.!?]+$", "", answer.lower().strip())


def _symptom2disease_evaluator(prediction: str, target: str, **_: Any) -> bool:
    pred = re.sub(r"\s+", " ", _extract_symptom_diagnosis(str(prediction)))
    expected = re.sub(r"\s+", " ", str(target).lower().strip())
    expected = re.sub(r"[.!?]+$", "", expected)
    return pred == expected


def _extract_lawbench_crimes(text: str) -> set[str]:
    answer = extract_json_field(text, "final_answer", default="")
    source = answer or text
    match = re.search(r"\[罪名\](.*?)(?:<eoa>|$)", source, re.DOTALL)
    if match:
        crimes = match.group(1).strip()
    else:
        match = re.search(r"罪名[:：](.*?)(?:\n|$)", source, re.DOTALL)
        crimes = match.group(1).strip() if match else source.strip()
    crimes = re.sub(r"<eoa>.*", "", crimes, flags=re.DOTALL).strip()
    return {item.strip() for item in re.split(r"[;；,，、]", crimes) if item.strip()}


def _lawbench_evaluator(prediction: str, target: str, **_: Any) -> bool:
    return _extract_lawbench_crimes(str(prediction)) == _extract_lawbench_crimes(str(target))


def _uspto_evaluator(prediction: str, target: str, **_: Any) -> bool:
    target = str(target).strip().upper()
    pred = str(prediction).strip().upper()
    pred = re.sub(r"```(?:JSON)?|```", "", pred).strip()
    if pred == target:
        return True
    match = re.search(r"\b(?:IPC\s+)?(?:SECTION|CLASS)?\s*([A-H])\b", pred)
    if match:
        return match.group(1) == target
    if pred.isdigit():
        names = list(IPC_SECTION_DESCRIPTIONS)
        index = int(pred)
        if 0 <= index < len(names):
            return names[index] == target
    return False


def _truncate_text(text: str, max_chars: int = USPTO_MAX_TEXT_CHARS) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " [truncated]"


def _dry_run_classification_answer(prompt: str) -> str:
    match = re.search(r"Choose exactly one section letter from these options:\s*([A-H]):", prompt)
    if match:
        return match.group(1)
    return "A"


def _row_result_path(
    *,
    row_dir: Path,
    dataset: str,
    memory_name: str,
    model: str,
    seed: int,
) -> Path:
    model_leaf = re.sub(r"[^\w\-.]", "_", model.split("/")[-1] or model)
    return row_dir / dataset / memory_name / f"{model_leaf}_seed{seed}.json"
