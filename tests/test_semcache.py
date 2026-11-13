"""Real tests: every cache here is a real SQLite file (or :memory:) and the real similarity code. The only injected thing is a
clock callable, so TTL and eviction order can be tested without sleeping; one test also uses real time.sleep."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from agent_semcache import MEASURES, SemCache, cached, normalize, similarity

ROOT = Path(__file__).resolve().parent.parent


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "c.sqlite3")


# ---- normalize / similarity -----------------------------------------------------------------------------------
def test_normalize_case_and_punctuation():
    assert normalize("  Hello,   WORLD!! ") == "hello world"


def test_normalize_unicode_nfkc():
    assert normalize("ｆｕｌｌｗｉｄｔｈ  Ⅻ") == "fullwidth xii"


def test_normalize_punctuation_only_prompts_stay_distinct():
    assert normalize("?!") != normalize("...")
    assert similarity("?!", "...") == 0.0


def test_normalize_rejects_non_str():
    with pytest.raises(TypeError):
        normalize(5)


def test_identical_after_normalization_scores_exactly_one():
    for m in MEASURES:
        assert similarity("What is TCP?", "what is   tcp", m) == 1.0


def test_disjoint_prompts_score_zero():
    assert similarity("aaaa", "zzzz", "char3-jaccard") == 0.0
    assert similarity("alpha beta", "gamma delta", "token-jaccard") == 0.0


def test_token_jaccard_known_value():
    # {a,b,c} vs {b,c,d}: intersection 2, union 4
    assert similarity("a b c", "b c d", "token-jaccard") == 0.5


def test_token_cosine_known_value():
    # vectors (1,1,0) and (0,1,1): dot 1, norms sqrt2 each -> 0.5
    assert similarity("a b", "b c", "token-cosine") == 0.5


def test_char3_jaccard_known_value():
    # " ab " -> {" ab","ab "}, " abc " -> {" ab","abc","bc "}: 1 shared of 4 -> 0.25
    assert similarity("ab", "abc", "char3-jaccard") == 0.25


def test_similarity_is_symmetric_for_every_measure():
    a, b = "reverse a list in python", "python list reversal"
    for m in MEASURES:
        assert similarity(a, b, m) == similarity(b, a, m)


def test_scores_stay_between_zero_and_one():
    for m in MEASURES:
        for a, b in [("x", "y"), ("", "a"), ("same", "same"), ("one two", "one three")]:
            assert 0.0 <= similarity(a, b, m) <= 1.0


def test_unknown_measure_rejected():
    with pytest.raises(ValueError):
        similarity("a", "b", "embedding")


def test_lexical_not_semantic_synonyms_do_not_match():
    assert similarity("What is a car?", "What is an automobile?") < 0.5


def test_word_order_is_mostly_ignored_documented_hazard():
    assert similarity("dog bites man", "man bites dog", "token-jaccard") == 1.0


# ---- SemCache core --------------------------------------------------------------------------------------------
def test_exact_prompt_hits(db):
    c = SemCache(db)
    c.store("hello there", "hi")
    h = c.lookup("hello there")
    assert h and h.response == "hi" and h.score == 1.0


def test_near_prompt_hits_above_threshold(db):
    c = SemCache(db, threshold=0.7)
    c.store("Summarize this log file", "S")
    h = c.lookup("Summarise this log file")
    assert h and h.response == "S" and 0.7 <= h.score < 1.0


def test_near_prompt_misses_below_threshold(db):
    c = SemCache(db, threshold=0.9)
    c.store("Summarize this log file", "S")
    assert c.lookup("Summarise this log file") is None


def test_per_call_threshold_overrides_default(db):
    c = SemCache(db, threshold=0.99)
    c.store("Summarize this log file", "S")
    assert c.lookup("Summarise this log file") is None
    assert c.lookup("Summarise this log file", threshold=0.5) is not None


def test_best_of_several_wins(db):
    c = SemCache(db, threshold=0.3)
    c.store("explain tcp slow start", "TCP")
    c.store("explain udp checksums", "UDP")
    assert c.lookup("explain tcp slow-start please").response == "TCP"


def test_empty_cache_miss(db):
    assert SemCache(db).lookup("anything") is None


def test_store_same_normalized_prompt_overwrites(db):
    c = SemCache(db)
    c.store("Hello?", "one")
    c.store("hello", "two")
    assert c.stats()["entries"] == 1
    assert c.lookup("HELLO").response == "two"


def test_response_must_be_str(db):
    with pytest.raises(TypeError):
        SemCache(db).store("p", 5)


def test_threshold_validation(db):
    for bad in (-0.1, 1.5, "0.8", None, True):
        with pytest.raises(ValueError):
            SemCache(db, threshold=bad)
    with pytest.raises(ValueError):
        SemCache(db).lookup("x", threshold=2)


def test_ttl_and_max_entries_validation(db):
    with pytest.raises(ValueError):
        SemCache(db, ttl=0)
    with pytest.raises(ValueError):
        SemCache(db, max_entries=0)


def test_namespaces_do_not_share_entries(db):
    a, b = SemCache(db, namespace="a"), SemCache(db, namespace="b")
    a.store("same prompt", "from a")
    assert b.lookup("same prompt") is None
    assert a.lookup("same prompt").response == "from a"


def test_context_must_match(db):
    c = SemCache(db)
    c.store("same prompt", "ctx1", context="model=x")
    assert c.lookup("same prompt", context="model=y") is None
    assert c.lookup("same prompt", context="model=x").response == "ctx1"


def test_persists_across_instances(db):
    SemCache(db).store("persist me", "yes")
    assert SemCache(db).lookup("persist me").response == "yes"


def test_in_memory_cache_works():
    c = SemCache(":memory:")
    c.store("a b c", "x")
    assert c.lookup("a b c").response == "x"


def test_nearest_changes_no_counter(db):
    c = SemCache(db)
    c.store("reverse a list", "R")
    n = c.nearest("reverse the list")
    assert n and 0 < n.score < 1
    assert c.stats()["lookups"] == 0


# ---- TTL ------------------------------------------------------------------------------------------------------
def test_ttl_expires_entry(db):
    clk = Clock()
    c = SemCache(db, ttl=60, clock=clk)
    c.store("short lived", "x")
    clk.t += 59
    assert c.lookup("short lived") is not None
    clk.t += 2   # stored at 1000, now 1061: older than 60 s
    assert c.lookup("short lived") is None
    s = c.stats()
    assert s["expired"] == 1 and s["entries"] == 0


def test_ttl_restarts_when_overwritten(db):
    clk = Clock()
    c = SemCache(db, ttl=60, clock=clk)
    c.store("p", "old")
    clk.t += 50
    c.store("p", "new")
    clk.t += 50
    assert c.lookup("p").response == "new"


def test_hit_does_not_extend_ttl(db):
    clk = Clock()
    c = SemCache(db, ttl=60, clock=clk)
    c.store("p", "x")
    clk.t += 40
    assert c.lookup("p")
    clk.t += 25
    assert c.lookup("p") is None


def test_ttl_with_real_time(db):
    c = SemCache(db, ttl=0.3)
    c.store("real clock", "x")
    assert c.lookup("real clock")
    time.sleep(0.45)
    assert c.lookup("real clock") is None


def test_hit_reports_age(db):
    clk = Clock()
    c = SemCache(db, clock=clk)
    c.store("p", "x")
    clk.t += 12.5
    assert c.lookup("p").age_s == 12.5


# ---- max entries ----------------------------------------------------------------------------------------------
def test_max_entries_evicts_oldest_unused(db):
    clk = Clock()
    c = SemCache(db, max_entries=3, clock=clk)
    for i, w in enumerate(["alpha", "bravo", "charlie", "delta"]):
        clk.t += 1
        c.store(f"unique {w} prompt text", w)
    s = c.stats()
    assert s["entries"] == 3 and s["evictions"] == 1
    assert c.lookup("unique alpha prompt text") is None
    assert c.lookup("unique delta prompt text") is not None


def test_eviction_is_least_recently_used_not_oldest(db):
    clk = Clock()
    c = SemCache(db, max_entries=2, clock=clk)
    c.store("first entry text", "1"); clk.t += 1
    c.store("second entry text", "2"); clk.t += 1
    assert c.lookup("first entry text")      # touch the first one
    clk.t += 1
    c.store("third entry text", "3")
    assert c.lookup("first entry text") is not None
    assert c.lookup("second entry text") is None


def test_max_entries_is_per_namespace(db):
    a, b = SemCache(db, namespace="a", max_entries=1), SemCache(db, namespace="b", max_entries=1)
    a.store("one", "1"); b.store("one", "1"); a.store("two", "2")
    assert a.stats()["entries"] == 1 and b.stats()["entries"] == 1


def test_no_limit_when_max_entries_none(db):
    c = SemCache(db, max_entries=None)
    for i in range(30):
        c.store(f"prompt number {i} xyz", str(i))
    assert c.stats()["entries"] == 30


# ---- stats / clear --------------------------------------------------------------------------------------------
def test_stats_count_real_hits_and_misses(db):
    c = SemCache(db)
    assert c.stats()["hit_rate"] is None and c.stats()["lookups"] == 0
    c.store("alpha beta gamma", "x")
    c.lookup("alpha beta gamma"); c.lookup("alpha beta gamma"); c.lookup("zzz qqq www")
    s = c.stats()
    assert (s["hits"], s["misses"], s["lookups"], s["stores"]) == (2, 1, 3, 1)
    assert s["hit_rate"] == round(2 / 3, 4)


def test_stats_persist_across_instances(db):
    c = SemCache(db); c.store("p", "x"); c.lookup("p"); c.lookup("nope nope")
    s = SemCache(db).stats()
    assert (s["hits"], s["misses"]) == (1, 1)


def test_clear_removes_entries_keeps_counters_by_default(db):
    c = SemCache(db); c.store("p", "x"); c.lookup("p")
    assert c.clear() == 1
    s = c.stats()
    assert s["entries"] == 0 and s["hits"] == 1


def test_clear_reset_stats(db):
    c = SemCache(db); c.store("p", "x"); c.lookup("p")
    c.clear(reset_stats=True)
    s = c.stats()
    assert s["hits"] == 0 and s["misses"] == 0 and s["stores"] == 0


def test_clear_only_touches_own_namespace(db):
    a, b = SemCache(db, namespace="a"), SemCache(db, namespace="b")
    a.store("p", "1"); b.store("p", "2")
    a.clear()
    assert b.lookup("p").response == "2"


def test_entry_hit_counter(db):
    c = SemCache(db)
    c.store("p", "x")
    assert c.lookup("p").hits == 1 and c.lookup("p").hits == 2


def test_thread_safety_counts_add_up(db):
    c = SemCache(db)
    c.store("shared prompt", "x")

    def work():
        for _ in range(25):
            c.lookup("shared prompt")
    ts = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert c.stats()["hits"] == 100


# ---- decorator ------------------------------------------------------------------------------------------------
def test_decorator_calls_function_once_for_similar_prompts(db):
    calls = []

    @cached(threshold=0.8, path=db)
    def llm(prompt):
        calls.append(prompt)
        return "A:" + prompt.lower()
    assert llm("Explain TCP slow start") == "A:explain tcp slow start"
    assert llm("explain tcp slow start!") == "A:explain tcp slow start"
    assert len(calls) == 1
    s = llm.cache_info()
    assert (s["hits"], s["misses"]) == (1, 1)


def test_decorator_calls_again_for_different_prompt(db):
    calls = []

    @cached(threshold=0.9, path=db)
    def llm(prompt):
        calls.append(prompt)
        return str(len(calls))
    llm("write a poem about the sea"); llm("fix this sql query please")
    assert len(calls) == 2


def test_bare_decorator_form(db, monkeypatch):
    monkeypatch.setenv("AGENT_SEMCACHE_DB", db)
    n = []

    @cached
    def llm(prompt):
        n.append(1)
        return "x"
    llm("p q r"); llm("p q r")
    assert len(n) == 1


def test_decorator_extra_args_are_part_of_the_key(db):
    @cached(threshold=0.5, path=db)
    def llm(prompt, temperature=0):
        return f"{prompt}@{temperature}"
    assert llm("same words", temperature=0) == "same words@0"
    assert llm("same words", temperature=1) == "same words@1"
    assert llm("same words", temperature=1) == "same words@1"
    assert llm.cache_info()["hits"] == 1


def test_decorator_prompt_keyword_and_type_checks(db):
    @cached(path=db)
    def llm(prompt):
        return "ok"
    assert llm(prompt="hello world") == "ok"
    with pytest.raises(TypeError):
        llm(5)
    with pytest.raises(TypeError):
        llm()


def test_decorator_requires_str_result(db):
    @cached(path=db)
    def llm(prompt):
        return 5
    with pytest.raises(TypeError):
        llm("p")


def test_decorator_does_not_cache_exceptions(db):
    n = []

    @cached(path=db)
    def llm(prompt):
        n.append(1)
        if len(n) == 1:
            raise RuntimeError("boom")
        return "fine"
    with pytest.raises(RuntimeError):
        llm("p")
    assert llm("p") == "fine"


def test_decorator_ttl_and_max_entries_passthrough(db):
    clk = Clock()

    @cached(threshold=1.0, path=db, ttl=10, max_entries=2, clock=clk)
    def llm(prompt):
        return prompt
    llm("one"); llm("two"); llm("three")
    s = llm.cache_info()
    assert s["entries"] == 2 and s["evictions"] == 1 and s["ttl_s"] == 10
    clk.t += 11
    assert llm.cache_info()["entries"] == 0


def test_decorator_invalid_threshold():
    with pytest.raises(ValueError):
        cached(threshold=3)


# ---- CLI (real subprocess) ------------------------------------------------------------------------------------
def cli(db, *args, stdin=None):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    return subprocess.run([sys.executable, "-m", "agent_semcache", "--db", db, *args], capture_output=True, text=True, env=env, input=stdin)


def test_cli_store_lookup_hit(db):
    assert cli(db, "store", "How do I undo a git commit?", "git reset HEAD~1").returncode == 0
    r = cli(db, "lookup", "how do i undo a git commit", "--threshold", "0.8")
    assert r.returncode == 0
    out = json.loads(r.stdout)
    assert out["hit"] is True and out["response"] == "git reset HEAD~1" and out["score"] == 1.0


def test_cli_lookup_miss_exit_1_with_best_score(db):
    cli(db, "store", "reverse a list in python", "x")
    r = cli(db, "lookup", "completely different words", "--threshold", "0.9")
    assert r.returncode == 1
    out = json.loads(r.stdout)
    assert out["hit"] is False and out["best_score"] is not None and out["best_score"] < 0.9


def test_cli_stats_counts_across_invocations(db):
    cli(db, "store", "p q r", "x"); cli(db, "lookup", "p q r"); cli(db, "lookup", "nothing like it")
    s = json.loads(cli(db, "stats").stdout)
    assert (s["hits"], s["misses"], s["entries"]) == (1, 1, 1)


def test_cli_clear(db):
    cli(db, "store", "p q r", "x")
    assert json.loads(cli(db, "clear").stdout) == {"cleared": 1}
    assert json.loads(cli(db, "stats").stdout)["entries"] == 0


def test_cli_stdin_prompt(db):
    cli(db, "store", "from stdin prompt", "ans")
    r = cli(db, "lookup", "-", stdin="from stdin prompt\n")
    assert r.returncode == 0 and json.loads(r.stdout)["response"] == "ans"


def test_cli_bad_threshold_exit_2(db):
    r = cli(db, "lookup", "x", "--threshold", "7")
    assert r.returncode == 2 and "threshold" in r.stderr


def test_cli_ttl_flag_expires(db):
    cli(db, "store", "p q r", "x")
    time.sleep(0.4)
    assert cli(db, "--ttl", "0.2", "lookup", "p q r").returncode == 1


def test_cli_max_entries_flag(db):
    cli(db, "--max-entries", "1", "store", "first prompt here", "1")
    cli(db, "--max-entries", "1", "store", "second prompt here", "2")
    assert json.loads(cli(db, "stats").stdout)["entries"] == 1
