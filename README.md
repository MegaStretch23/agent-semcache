# agent-semcache

A small **local prompt cache** for LLM agents. Before you pay for (or wait on) a model call, ask the cache whether you already have an answer for a *similar* prompt. If a stored prompt scores above your threshold, you get its stored answer back; if not, you call the model and store the result.

- Python 3.9+ declared, **standard library only** (SQLite, `re`, `unicodedata`). No network, no embedding API, no model. Tests pass on Python 3.9, 3.12 and 3.13 (checked on Ubuntu 24.04; I did not test other platforms).
- `@cached(threshold=...)` decorator for any `fn(prompt) -> str`, plus a CLI with `lookup`, `store`, `stats` and `clear`.
- TTL, a max-entries limit (least recently used goes first), and **real hit / miss counters** kept in the cache file.

> **The similarity is lexical, not semantic.** It compares the characters (3-character n-grams, by default) or the words two prompts share after lowercasing and stripping punctuation. It does **not** understand meaning: "What is a car?" and "What is an automobile?" score 0.31 (a miss), while "What is the capital of France?" and "What is the capital of Spain?" score 0.63, and "dog bites man" and "man bites dog" score 0.73 (word order is mostly ignored). A hit can therefore return a wrong answer for a prompt that merely looks alike. Pick a threshold you have checked against your own prompts, and do not use this where a wrong cached answer is costly. If you need real semantic matching you need an embedding model, which this tool deliberately does not use.

## Quickstart

```bash
git clone https://github.com/MegaStretch23/agent-semcache.git
cd agent-semcache
pip install .

agent-semcache --db demo.sqlite3 store "How do I undo a git commit?" "git reset HEAD~1"
agent-semcache --db demo.sqlite3 lookup "how do i undo a git commit" --threshold 0.8   # hit, exit code 0
agent-semcache --db demo.sqlite3 lookup "bake a sourdough loaf" --threshold 0.8        # miss, exit code 1
agent-semcache --db demo.sqlite3 stats
```

(`PYTHONPATH=src python3 -m agent_semcache ...` works without installing.) Output is JSON. `lookup` exits 0 on a hit and 1 on a miss; on a miss it prints the best score it found so you can tune the threshold. Pass `-` as the prompt to read it from stdin.

In Python:

```python
from agent_semcache import cached

@cached(threshold=0.8, ttl=3600, max_entries=500)      # file: $AGENT_SEMCACHE_DB or ~/.cache/agent-semcache/cache.sqlite3
def ask_model(prompt: str) -> str:
    return call_your_llm(prompt)                        # your real call goes here

ask_model("Explain TCP slow start")        # miss: calls the model, stores the answer
ask_model("explain tcp slow-start")        # hit: same words after normalization, no model call
print(ask_model.cache_info())              # hits, misses, hit_rate, entries, evictions, expired, ...
```

Any extra arguments (`ask_model(prompt, temperature=0)`) must be identical for a stored answer to be reused. The function must take the prompt as its first argument and return a `str`. Exceptions are not cached.

The lower-level API is `SemCache(path).lookup(prompt)`, `.store(prompt, response)`, `.nearest(prompt)` (closest stored prompt whatever its score, changes no counter), `.stats()` and `.clear()`.

## Saved demo run

This is a **saved demo run** of `sh examples/regenerate.sh`, kept in `examples/demo_output.txt`. The "LLM" is a fake deterministic function (`examples/demo.py`: it returns a hash of the prompt and counts how often it was called), so there is no model, no network, and the output is identical on every run. The first table is the threshold trade-off with the default measure; the second is the decorator at threshold 0.7.

```text
== similarity scores (char3-jaccard) and what each threshold would do ==
first prompt                         second prompt                        score  t=0.5  t=0.7  t=0.85
How do I reverse a list in Python?   how do i reverse a list in python     1.00  HIT    HIT    HIT  
How do I reverse a list in Python?   How can I reverse a Python list?      0.60  HIT    miss   miss 
How do I reverse a list in Python?   reverse list python                   0.49  miss   miss   miss 
What is the capital of France?       What is the capital of Spain?         0.63  HIT    miss   miss 
Summarize this log file              Summarise this log file               0.77  HIT    HIT    miss 
dog bites man                        man bites dog                         0.73  HIT    HIT    miss 
What is a car?                       What is an automobile?                0.31  miss   miss   miss 

== the @cached decorator over a fake LLM, threshold 0.7 ==
How do I reverse a list in Python?       -> answer-b510a6b3  (model called)
how do i reverse a list in python        -> answer-b510a6b3  (served from cache)
How do I reverse a list in Python ?!     -> answer-b510a6b3  (served from cache)
What is the capital of France?           -> answer-1e54b09f  (model called)
What's the capital of France?            -> answer-1e54b09f  (served from cache)
What is the capital of Spain?            -> answer-48ac97e1  (model called)
Explain TCP slow start                   -> answer-b647f4e2  (model called)
explain tcp slow-start                   -> answer-b647f4e2  (served from cache)
Explain UDP                              -> answer-4fe2ff73  (model called)

fake model calls: 5 of 9 prompts
stats: hits=4 misses=5 hit_rate=0.4444 entries=5 stores=5
```

Read it as a warning as much as a feature: at 0.5 the France / Spain pair is a hit and would return the *France* answer for a *Spain* question; at 0.7 the dog / man pair is a hit; at 0.85 only the near-identical prompts match.

## Threshold trade-off

| threshold | effect |
|---|---|
| high (0.9 and up) | Only prompts that differ by case, punctuation or a letter or two match. Few wrong answers, few hits. |
| middle (0.6 to 0.85) | Catches spelling variants and small rewordings (in the saved run, "Summarise" for "Summarize" scores 0.77). Also starts matching prompts that differ in one important word. |
| low (below 0.6) | Matches many prompts that merely share common words. Expect wrong answers. |

There is no universally right value; it depends on your prompts. Use `lookup` (it prints `best_score` on a miss) or `SemCache.nearest()` on a sample of your own prompts before you trust a threshold.

## Measures

`--measure` (CLI) or `measure=` (library): `char3-jaccard` (default), `char3-cosine`, `token-jaccard`, `token-cosine`. Character n-grams tolerate typos and small spelling changes; token measures compare word sets and ignore word order completely (`token-jaccard` scores "dog bites man" against "man bites dog" at exactly 1.0). Identical normalized prompts always score exactly 1.0.

## Behaviour details

- **Normalization:** Unicode NFKC, casefold, every run of punctuation / whitespace becomes one space. Digits are kept, so "python 3.9" and "python 3.12" are not identical.
- **Storage:** one SQLite file. A lookup is a linear scan of the entries in the namespace, so keep `max_entries` modest (default 1000). I have not benchmarked large caches.
- **TTL:** an entry older than `ttl` seconds is deleted on the next lookup, store or stats call. A cache hit does not extend its life; storing the same prompt again restarts it. The TTL belongs to the process that is reading, not to the stored row.
- **Max entries:** past the limit the least recently *used* entries (per namespace) are evicted and counted in `stats`.
- **Namespaces and context:** entries match only inside the same namespace (the decorator uses the function's name) and the same context string.
- **Stats:** `hits`, `misses`, `lookups`, `hit_rate` (`null` before the first lookup), `stores`, `evictions`, `expired`, `entries`. Counters live in the cache file and survive restarts; `clear --reset-stats` zeroes them.
- The cache stores your prompts and answers in plain text in the SQLite file. Put the file somewhere with the permissions you want.

## Tests

```bash
pip install pytest
python -m pytest -q        # 62 tests
```

The tests use a real SQLite file and the real similarity code (no mocks of the cache). The only injected object is a clock function so TTL and eviction order can be checked without sleeping; two tests also use real `time.sleep`. The CLI tests run the real command in a subprocess.

## Also by the author

If you need data rather than a cache, I run pay-per-event tools on Apify, for example the [SEC Form 4 insider cluster scanner](https://apify.com/johnveds15/sec-form4-insider-clusters): $0.0004 per Form 4 filing parsed and $0.02 per insider cluster detected (price read from the Apify API on 2026-11-13). Source-code packs for agent work are on Gumroad: the [Agent-Turn Delta-Debugger CLI](https://vedovino.gumroad.com/l/zdwyic) ($25, finds which turns break an agent run) and the [Agent log policy checker](https://vedovino.gumroad.com/l/agent-log-policy-checker) ($19, finds rule violations in an agent's JSONL log), prices as listed on 2026-11-13. They are separate products and unrelated to this tool; there is no paid version of agent-semcache.

MIT licensed.
