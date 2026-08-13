"""Evaluate the RAG pipeline with DeepEval, judged by a local Ollama model.

    python -m eval.run_eval                 # everything
    python -m eval.run_eval --retrieval     # retrieval only, no judge, seconds
    python -m eval.run_eval --limit 3       # first 3 goldens, for a quick loop

DeepEval's metrics are LLM-as-judge and default to OpenAI, which would mean an
API key and shipping document text off the machine. The judge here is Ollama,
so evaluation stays as local as the pipeline it measures.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
CORPUS = BASE / "corpus"

# DeepEval reports usage analytics by default. This project's premise is that
# nothing leaves the machine, so opt out before it is imported anywhere.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("ERROR_REPORTING", "NO")

# Point the store at a scratch directory *before* app.config is imported, so
# evaluating never touches documents the user has indexed for real.
EVAL_STORE = BASE / ".eval_store"
os.environ.setdefault("STORE_DIR", str(EVAL_STORE))

sys.path.insert(0, str(BASE.parent))

from app import store  # noqa: E402
from app.chunking import chunk_text, extract_text  # noqa: E402
from app.config import CHAT_MODEL, OLLAMA_HOST, TOP_K  # noqa: E402
from app.rag import stream_answer  # noqa: E402
from eval.dataset import GOLDENS, Golden  # noqa: E402

JUDGE_MODEL = os.getenv("JUDGE_MODEL", CHAT_MODEL)


# ── pipeline under test ────────────────────────────────────────────────────

def index_corpus() -> int:
    """Rebuild the eval index from scratch so runs are comparable."""
    if EVAL_STORE.exists():
        shutil.rmtree(EVAL_STORE)
    store.reset()

    total = 0
    for path in sorted(CORPUS.iterdir()):
        if path.suffix.lower() not in (".pdf", ".txt", ".md"):
            continue
        chunks = chunk_text(extract_text(path.name, path.read_bytes()))
        total += store.add_document(path.name, chunks)
    return total


def answer_question(question: str, top_k: int) -> tuple[str, list[dict]]:
    """Run one question through retrieval and generation, without streaming."""
    hits = store.search(question, top_k=top_k)
    text = "".join(stream_answer(question, hits))
    return text, hits


# ── retrieval check (deterministic, no judge) ──────────────────────────────

def retrieval_hit(golden: Golden, hits: list[dict]) -> bool | None:
    """Did the chunk carrying the answer make it into the context?

    Returns None when the golden has nothing to retrieve — the unanswerable
    case, where an empty or irrelevant context is the correct outcome.
    """
    if not golden.must_retrieve:
        return None
    needle = " ".join(golden.must_retrieve.lower().split())
    return any(needle in " ".join(h["text"].lower().split()) for h in hits)


def run_retrieval_only(goldens: list[Golden], top_k: int) -> list[dict]:
    rows = []
    for golden in goldens:
        started = time.perf_counter()
        hits = store.search(golden.question, top_k=top_k)
        rows.append(
            {
                "question": golden.question,
                "tags": list(golden.tags),
                "hit": retrieval_hit(golden, hits),
                "top_score": hits[0]["score"] if hits else None,
                "seconds": round(time.perf_counter() - started, 2),
                "retrieved": [h["text"][:90] for h in hits],
            }
        )
    return rows


# ── full evaluation ────────────────────────────────────────────────────────

def build_metrics(threshold: float):
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )
    from deepeval.models import OllamaModel

    judge = OllamaModel(model=JUDGE_MODEL, base_url=OLLAMA_HOST, temperature=0)

    # async_mode off: concurrent requests to one local Ollama just queue, and
    # serial runs keep the progress output readable.
    common = dict(model=judge, threshold=threshold, async_mode=False)
    return [
        # Generation: is the answer on topic, and does it stick to the context?
        AnswerRelevancyMetric(**common),
        FaithfulnessMetric(**common),
        # Retrieval: did the right chunks come back, ranked sensibly?
        ContextualPrecisionMetric(**common),
        ContextualRecallMetric(**common),
        ContextualRelevancyMetric(**common),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", action="store_true",
                        help="retrieval hit-rate only; no LLM judge")
    parser.add_argument("--limit", type=int, help="only the first N goldens")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--json", type=Path, help="write results here")
    args = parser.parse_args()

    goldens = GOLDENS[: args.limit] if args.limit else GOLDENS

    print(f"Indexing {CORPUS}…")
    try:
        chunks = index_corpus()
    except Exception as exc:
        print(f"\nCould not index — is Ollama running?\n  {exc}")
        return 1
    print(f"  {chunks} chunks\n")

    # With top_k >= chunks every question retrieves the whole corpus, so the
    # hit rate is 100% whether or not search works. Say so rather than
    # reporting a number that cannot fail.
    if chunks <= args.top_k:
        print(
            f"  WARNING: {chunks} chunks with --top-k {args.top_k} means every\n"
            f"  question retrieves everything. Retrieval scores are\n"
            f"  meaningless here — add more documents to eval/corpus/ or\n"
            f"  lower --top-k.\n"
        )

    if args.retrieval:
        rows = run_retrieval_only(goldens, args.top_k)
        scored = [r for r in rows if r["hit"] is not None]
        hits = sum(r["hit"] for r in scored)
        for row in rows:
            mark = "—" if row["hit"] is None else ("PASS" if row["hit"] else "FAIL")
            print(f"  [{mark:4}] {row['question']}")
            if row["hit"] is False:
                print(f"          got: {row['retrieved'][0][:80]}…")
        print(f"\nRetrieval hit rate: {hits}/{len(scored)}")
        if args.json:
            args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return 0 if hits == len(scored) else 1

    from deepeval import evaluate
    from deepeval.test_case import LLMTestCase

    cases, extras = [], []
    for i, golden in enumerate(goldens, 1):
        print(f"[{i}/{len(goldens)}] {golden.question}")
        started = time.perf_counter()
        actual, hits = answer_question(golden.question, args.top_k)
        elapsed = time.perf_counter() - started
        print(f"        {actual[:100].strip()}…  ({elapsed:.1f}s)")

        cases.append(
            LLMTestCase(
                input=golden.question,
                actual_output=actual,
                expected_output=golden.expected_output,
                retrieval_context=[h["text"] for h in hits],
            )
        )
        extras.append(
            {
                "question": golden.question,
                "tags": list(golden.tags),
                "retrieval_hit": retrieval_hit(golden, hits),
                "seconds": round(elapsed, 1),
            }
        )

    scored = [e for e in extras if e["retrieval_hit"] is not None]
    hits = sum(e["retrieval_hit"] for e in scored)
    print(f"\nRetrieval hit rate: {hits}/{len(scored)}")
    print(f"Judging with {JUDGE_MODEL} — this is the slow part.\n")

    results = evaluate(test_cases=cases, metrics=build_metrics(args.threshold))

    report = collect(results, extras)
    print_report(report, args.threshold)

    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")

    # Non-zero when anything failed, so this can gate a commit or CI run.
    failed = sum(1 for q in report["questions"] if not q["passed"])
    return 1 if failed else 0


# ── reporting ──────────────────────────────────────────────────────────────

def collect(results, extras: list[dict]) -> dict:
    """Flatten DeepEval's result object into plain data we can print or store."""
    questions, by_metric = [], {}

    for i, test in enumerate(results.test_results):
        metrics = []
        for m in test.metrics_data or []:
            metrics.append(
                {
                    "metric": m.name,
                    "score": None if m.score is None else round(m.score, 3),
                    "passed": bool(m.success),
                    "reason": m.reason,
                    "error": m.error,
                }
            )
            by_metric.setdefault(m.name, []).append(m.score)

        extra = extras[i] if i < len(extras) else {}
        questions.append(
            {
                "question": test.input,
                "tags": extra.get("tags", []),
                "answer": test.actual_output,
                "expected": test.expected_output,
                "retrieval_hit": extra.get("retrieval_hit"),
                "seconds": extra.get("seconds"),
                "passed": bool(test.success),
                "metrics": metrics,
            }
        )

    averages = {
        name: round(sum(s for s in scores if s is not None) / max(1, len(scores)), 3)
        for name, scores in by_metric.items()
    }
    return {"judge": JUDGE_MODEL, "averages": averages, "questions": questions}


def print_report(report: dict, threshold: float) -> None:
    print("\n" + "=" * 78)
    print(f"Per-question detail (threshold {threshold}, judge {report['judge']})")
    print("=" * 78)

    for q in report["questions"]:
        tags = f"  [{', '.join(q['tags'])}]" if q["tags"] else ""
        print(f"\n{'PASS' if q['passed'] else 'FAIL'}  {q['question']}{tags}")
        print(f"      answer:   {q['answer'][:150].strip()}")
        if q["retrieval_hit"] is False:
            print("      NOTE: the chunk holding the answer was never retrieved —")
            print("            treat the generation scores below as unearned.")
        for m in q["metrics"]:
            mark = "ok  " if m["passed"] else "FAIL"
            print(f"      {mark} {m['metric']:<22} {m['score']}")
            # The reason is the useful part; a bare number says nothing about
            # what to change.
            if not m["passed"] and m["reason"]:
                print(f"           ↳ {m['reason'][:300]}")
            if m["error"]:
                print(f"           ↳ judge error: {m['error'][:200]}")

    print("\n" + "=" * 78)
    print("Averages")
    for name, avg in report["averages"].items():
        bar = "█" * int(avg * 24)
        print(f"  {name:<24} {avg:<6} {bar}")
    passed = sum(1 for q in report["questions"] if q["passed"])
    print(f"\n  {passed}/{len(report['questions'])} questions passed every metric")


if __name__ == "__main__":
    raise SystemExit(main())
