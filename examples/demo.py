"""Demo with a FAKE deterministic "LLM" (no model, no network). Run: sh examples/regenerate.sh  (output saved in examples/demo_output.txt)."""
import os
import sys
import tempfile
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from agent_semcache import SemCache, cached, similarity  # noqa: E402

CALLS = []


def fake_llm(prompt: str) -> str:
    """Pretend model call: deterministic, and we count how often it is really invoked."""
    CALLS.append(prompt)
    return f"answer-{zlib.crc32(prompt.encode()):08x}"


PAIRS = [
    ("How do I reverse a list in Python?", "how do i reverse a list in python"),
    ("How do I reverse a list in Python?", "How can I reverse a Python list?"),
    ("How do I reverse a list in Python?", "reverse list python"),
    ("What is the capital of France?", "What is the capital of Spain?"),
    ("Summarize this log file", "Summarise this log file"),
    ("dog bites man", "man bites dog"),
    ("What is a car?", "What is an automobile?"),
]
THRESHOLDS = (0.5, 0.7, 0.85)


def main() -> None:
    print("== similarity scores (char3-jaccard) and what each threshold would do ==")
    print(f"{'first prompt':<36} {'second prompt':<36} {'score':>5}  " + "  ".join(f"t={t}" for t in THRESHOLDS))
    for a, b in PAIRS:
        s = similarity(a, b)
        print(f"{a:<36} {b:<36} {s:5.2f}  " + "  ".join(f"{'HIT ' if s >= t else 'miss'} " for t in THRESHOLDS))

    print("\n== the @cached decorator over a fake LLM, threshold 0.7 ==")
    workload = ["How do I reverse a list in Python?", "how do i reverse a list in python", "How do I reverse a list in Python ?!",
                "What is the capital of France?", "What's the capital of France?", "What is the capital of Spain?",
                "Explain TCP slow start", "explain tcp slow-start", "Explain UDP"]
    with tempfile.TemporaryDirectory() as td:
        wrapped = cached(threshold=0.7, path=os.path.join(td, "demo.sqlite3"), namespace="demo")(fake_llm)
        for p in workload:
            before = len(CALLS)
            out = wrapped(p)
            print(f"{p:<40} -> {out}  {'(model called)' if len(CALLS) > before else '(served from cache)'}")
        st = wrapped.cache_info()
        print(f"\nfake model calls: {len(CALLS)} of {len(workload)} prompts")
        print(f"stats: hits={st['hits']} misses={st['misses']} hit_rate={st['hit_rate']} entries={st['entries']} stores={st['stores']}")
        wrapped.cache.close()


if __name__ == "__main__":
    main()
