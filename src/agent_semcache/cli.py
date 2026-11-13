"""Command line: lookup / store / stats / clear. Output is JSON on stdout. Exit codes: 0 ok or hit, 1 miss, 2 usage error."""
from __future__ import annotations

import argparse
import json
import sys

from .cache import DEFAULT_MAX_ENTRIES, DEFAULT_THRESHOLD, SemCache, default_db_path
from .similarity import DEFAULT_MEASURE, MEASURES


def _text(value: str) -> str:
    return sys.stdin.read() if value == "-" else value


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="agent-semcache", description="Local lexical prompt cache (n-gram / token similarity, SQLite, no embeddings).")
    ap.add_argument("--db", default=default_db_path(), help="SQLite file (default: $AGENT_SEMCACHE_DB or ~/.cache/agent-semcache/cache.sqlite3)")
    ap.add_argument("--namespace", default="default")
    ap.add_argument("--measure", default=DEFAULT_MEASURE, choices=sorted(MEASURES))
    ap.add_argument("--ttl", type=float, default=None, help="seconds an entry stays usable (default: forever)")
    ap.add_argument("--max-entries", type=int, default=DEFAULT_MAX_ENTRIES, help="least recently used entries are evicted past this (default %(default)s)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lk = sub.add_parser("lookup", help="print the stored answer for a similar prompt; exit 1 on a miss")
    lk.add_argument("prompt", help="prompt text, or - to read stdin")
    lk.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    lk.add_argument("--context", default="", help="entries only match inside the same context string")
    st = sub.add_parser("store", help="save an answer for a prompt")
    st.add_argument("prompt", help="prompt text, or - to read stdin")
    st.add_argument("response", help="answer text, or - to read stdin")
    st.add_argument("--context", default="")
    sub.add_parser("stats", help="hit / miss counters and entry count")
    cl = sub.add_parser("clear", help="delete every entry of the namespace")
    cl.add_argument("--reset-stats", action="store_true", help="also zero the hit / miss counters")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cache = SemCache(args.db, threshold=getattr(args, "threshold", DEFAULT_THRESHOLD), measure=args.measure, ttl=args.ttl,
                         max_entries=args.max_entries, namespace=args.namespace)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    with cache:
        if args.cmd == "lookup":
            prompt = _text(args.prompt)
            hit = cache.lookup(prompt, context=args.context)
            if hit is None:
                near = cache.nearest(prompt, context=args.context)
                print(json.dumps({"hit": False, "threshold": args.threshold, "best_score": near.score if near else None,
                                  "best_prompt": near.prompt if near else None}))
                return 1
            print(json.dumps({"hit": True, "score": hit.score, "matched_prompt": hit.prompt, "response": hit.response,
                              "age_s": round(hit.age_s, 3), "entry_hits": hit.hits}))
            return 0
        if args.cmd == "store":
            if args.prompt == "-" and args.response == "-":
                print("error: only one of prompt / response can read stdin", file=sys.stderr)
                return 2
            cache.store(_text(args.prompt), _text(args.response), context=args.context)
            print(json.dumps({"stored": True, "entries": cache.stats()["entries"]}))
            return 0
        if args.cmd == "stats":
            print(json.dumps(cache.stats(), indent=2))
            return 0
        if args.cmd == "clear":
            print(json.dumps({"cleared": cache.clear(reset_stats=args.reset_stats)}))
            return 0
    return 2
