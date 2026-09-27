"""Load NuminaMath-TIR and build conversational GRPO prompts."""

from datasets import Dataset, load_dataset

DATASET_ID = "AI-MO/NuminaMath-TIR"

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def make_conversation(example: dict) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }


def load_train_test() -> tuple[Dataset, Dataset]:
    """train[:5%] for training, test[:5%] held out (HF tutorial splits)."""
    train_dataset, test_dataset = load_dataset(
        DATASET_ID,
        split=["train[:5%]", "test[:5%]"],
    )
    train_dataset = train_dataset.map(make_conversation)
    test_dataset = test_dataset.map(make_conversation)
    drop_cols = [c for c in ("messages", "problem") if c in train_dataset.column_names]
    train_dataset = train_dataset.remove_columns(drop_cols)
    test_dataset = test_dataset.remove_columns(drop_cols)
    return train_dataset, test_dataset


def load_smoke_train(n: int = 2) -> Dataset:
    """Minimal train slice for smoke test (does not load test split or train[:5%])."""
    n = max(1, n)
    train_dataset = load_dataset(DATASET_ID, split=f"train[:{n}]")
    train_dataset = train_dataset.map(make_conversation)
    drop_cols = [c for c in ("messages", "problem") if c in train_dataset.column_names]
    return train_dataset.remove_columns(drop_cols)
