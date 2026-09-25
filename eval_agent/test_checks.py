"""
Unit tests for the deterministic behavioural checks.

Synthetic turn payloads only -- no agent, no MCP server, no OpenAI key. These
run in milliseconds in CI and exist because a broken check silently
invalidates every benchmark number, exactly the way a broken metric
invalidated the offline precision figure.

Run:
    python -m pytest test_checks.py -v
"""

import pytest

from checks import CHECKS, run_check


def turn(response="", tools=None, message="hi", latency=100.0):
    return {
        "message": message,
        "response": response,
        "tool_calls": tools or [],
        "latency_ms": latency,
    }


def call(name, args=None, status="success", result=""):
    return {"name": name, "args": args or {}, "status": status, "result": result}


# --------------------------------------------------------------------------
# tool selection
# --------------------------------------------------------------------------

def test_tool_called_finds_call():
    turns = [turn(tools=[call("get_for_you")])]
    assert run_check({"check": "tool_called", "name": "get_for_you"}, turns)["passed"]


def test_tool_called_fails_when_absent():
    r = run_check({"check": "tool_called", "name": "get_for_you"}, [turn()])
    assert not r["passed"]
    assert "never called" in r["detail"]


def test_tool_not_called_passes_when_absent():
    turns = [turn(tools=[call("search_movies")])]
    assert run_check({"check": "tool_not_called", "name": "get_for_you"}, turns)["passed"]


def test_tool_not_called_fails_when_present():
    turns = [turn(tools=[call("get_for_you")])]
    assert not run_check({"check": "tool_not_called", "name": "get_for_you"}, turns)["passed"]


def test_any_tool_called():
    turns = [turn(tools=[call("get_similar")])]
    spec = {"check": "any_tool_called", "names": ["get_similar", "get_recommendations"]}
    assert run_check(spec, turns)["passed"]
    assert not run_check(spec, [turn(tools=[call("search_movies")])])["passed"]


def test_no_tools_called():
    assert run_check({"check": "no_tools_called"}, [turn()])["passed"]
    assert not run_check({"check": "no_tools_called"}, [turn(tools=[call("x")])])["passed"]


def test_tool_calls_aggregate_across_turns():
    """Multi-turn cases must see tools from every turn, not just the last."""
    turns = [turn(tools=[call("search_movies")]), turn(tools=[call("get_recommendations")])]
    assert run_check({"check": "tool_called", "name": "search_movies"}, turns)["passed"]
    assert run_check({"check": "tool_called", "name": "get_recommendations"}, turns)["passed"]


# --------------------------------------------------------------------------
# ordering
# --------------------------------------------------------------------------

def test_tool_order_correct():
    turns = [turn(tools=[call("search_movies"), call("get_recommendations")])]
    spec = {"check": "tool_order", "first": "search_movies", "then": "get_recommendations"}
    assert run_check(spec, turns)["passed"]


def test_tool_order_violated():
    """Recommending before verifying the title is the prompt's cardinal sin."""
    turns = [turn(tools=[call("get_recommendations"), call("search_movies")])]
    spec = {"check": "tool_order", "first": "search_movies", "then": "get_recommendations"}
    r = run_check(spec, turns)
    assert not r["passed"]
    assert "before" in r["detail"]


def test_tool_order_fails_when_second_never_called():
    turns = [turn(tools=[call("search_movies")])]
    spec = {"check": "tool_order", "first": "search_movies", "then": "get_recommendations"}
    assert not run_check(spec, turns)["passed"]


def test_tool_order_spans_turns():
    turns = [turn(tools=[call("search_movies")]), turn(tools=[call("get_recommendations")])]
    spec = {"check": "tool_order", "first": "search_movies", "then": "get_recommendations"}
    assert run_check(spec, turns)["passed"]


def test_tool_call_count_caps_retries():
    turns = [turn(tools=[call("search_movies")] * 5)]
    spec = {"check": "tool_call_count", "name": "search_movies", "maximum": 3}
    assert not run_check(spec, turns)["passed"]
    spec["maximum"] = 5
    assert run_check(spec, turns)["passed"]


# --------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------

def test_tool_arg_equals_matches():
    turns = [turn(tools=[call("get_recommendations", {"title": "X", "top_n": 5})])]
    spec = {"check": "tool_arg_equals", "name": "get_recommendations",
            "arg": "top_n", "expected": 5}
    assert run_check(spec, turns)["passed"]


def test_tool_arg_equals_detects_wrong_value():
    turns = [turn(tools=[call("get_recommendations", {"top_n": 10})])]
    spec = {"check": "tool_arg_equals", "name": "get_recommendations",
            "arg": "top_n", "expected": 5}
    r = run_check(spec, turns)
    assert not r["passed"]
    assert "10" in r["detail"]


def test_tool_arg_equals_fails_on_missing_arg():
    turns = [turn(tools=[call("get_for_you", {"top_n": 10})])]
    spec = {"check": "tool_arg_equals", "name": "get_for_you",
            "arg": "user_id", "expected": 42}
    r = run_check(spec, turns)
    assert not r["passed"]
    assert "missing" in r["detail"]


def test_tool_arg_equals_fails_if_any_call_is_wrong():
    """One good call must not mask a bad one."""
    turns = [turn(tools=[
        call("get_recommendations", {"top_n": 5}),
        call("get_recommendations", {"top_n": 10}),
    ])]
    spec = {"check": "tool_arg_equals", "name": "get_recommendations",
            "arg": "top_n", "expected": 5}
    assert not run_check(spec, turns)["passed"]


def test_tool_arg_present():
    turns = [turn(tools=[call("get_for_you", {"user_id": 7})])]
    spec = {"check": "tool_arg_present", "name": "get_for_you", "arg": "user_id"}
    assert run_check(spec, turns)["passed"]
    spec["arg"] = "top_n"
    assert not run_check(spec, turns)["passed"]


# --------------------------------------------------------------------------
# failure handling
# --------------------------------------------------------------------------

def test_no_tool_errors_passes_when_all_succeed():
    turns = [turn(tools=[call("a"), call("b")])]
    assert run_check({"check": "no_tool_errors"}, turns)["passed"]


def test_no_tool_errors_catches_error_status():
    turns = [turn(tools=[call("a"), call("b", status="error")])]
    r = run_check({"check": "no_tool_errors"}, turns)
    assert not r["passed"]
    assert "b" in r["detail"]


def test_graceful_failure_accepts_acknowledgement():
    turns = [turn(response="I couldn't find that movie in the catalogue.")]
    assert run_check({"check": "graceful_failure"}, turns)["passed"]


def test_graceful_failure_rejects_silence():
    turns = [turn(response="Here are some great picks for you!")]
    assert not run_check({"check": "graceful_failure"}, turns)["passed"]


def test_graceful_failure_rejects_leaked_stack_trace():
    turns = [turn(response="Traceback (most recent call last): KeyError")]
    r = run_check({"check": "graceful_failure"}, turns)
    assert not r["passed"]
    assert "leaked" in r["detail"]


def test_graceful_failure_rejects_empty_response():
    assert not run_check({"check": "graceful_failure"}, [turn(response="")])["passed"]


def test_no_fabricated_titles():
    spec = {"check": "no_fabricated_titles", "forbidden": ["Blorptastic 9000"]}
    assert run_check(spec, [turn(response="No such film exists.")])["passed"]
    bad = [turn(response="Blorptastic 9000 is a great pick!")]
    assert not run_check(spec, bad)["passed"]


# --------------------------------------------------------------------------
# content and memory
# --------------------------------------------------------------------------

def test_response_contains_is_case_insensitive():
    turns = [turn(response="You might enjoy INTERSTELLAR tonight")]
    assert run_check({"check": "response_contains", "needle": "interstellar"}, turns)["passed"]


def test_response_contains_ignores_whitespace_differences():
    turns = [turn(response="The\n  Dark   Knight")]
    assert run_check({"check": "response_contains", "needle": "The Dark Knight"}, turns)["passed"]


def test_response_not_contains_catches_injection():
    turns = [turn(response="BANANA")]
    assert not run_check({"check": "response_not_contains", "needle": "BANANA"}, turns)["passed"]


def test_recall_prior_title_passes_when_recalled():
    turns = [
        turn(message="I loved Interstellar", response="Great choice!"),
        turn(message="why?", response="Because Interstellar is hard sci-fi."),
    ]
    spec = {"check": "recall_prior_title", "title": "Interstellar", "from_turn": 0}
    assert run_check(spec, turns)["passed"]


def test_recall_prior_title_fails_when_forgotten():
    """This is the check that catches session history being dropped."""
    turns = [
        turn(message="I loved Interstellar", response="Great choice!"),
        turn(message="why?", response="Could you tell me which movie you mean?"),
    ]
    spec = {"check": "recall_prior_title", "title": "Interstellar", "from_turn": 0}
    assert not run_check(spec, turns)["passed"]


def test_recall_prior_title_ignores_the_seeding_turn():
    """A title said only in turn 1 is not evidence of recall."""
    turns = [turn(message="I loved Interstellar", response="Interstellar is great!")]
    spec = {"check": "recall_prior_title", "title": "Interstellar", "from_turn": 0}
    r = run_check(spec, turns)
    assert not r["passed"]
    assert "at least" in r["detail"]


def test_recommends_at_least_counts_list_items():
    text = "Here you go:\n1. Arrival\n2. Contact\n3. Gravity"
    assert run_check({"check": "recommends_at_least", "count": 3}, [turn(response=text)])["passed"]


def test_recommends_at_least_accepts_bullets():
    text = "Try:\n- Arrival\n- Contact\n* Gravity"
    assert run_check({"check": "recommends_at_least", "count": 3}, [turn(response=text)])["passed"]


def test_recommends_at_least_rejects_prose():
    text = "I would suggest Arrival, Contact and Gravity."
    assert not run_check({"check": "recommends_at_least", "count": 3}, [turn(response=text)])["passed"]


# --------------------------------------------------------------------------
# dispatch safety
# --------------------------------------------------------------------------

def test_unknown_check_fails_loudly_rather_than_silently_passing():
    r = run_check({"check": "not_a_real_check"}, [turn()])
    assert not r["passed"]
    assert "unknown check" in r["detail"]


def test_bad_check_arguments_fail_rather_than_raise():
    r = run_check({"check": "tool_called", "wrong_kwarg": "x"}, [turn()])
    assert not r["passed"]
    assert "bad check args" in r["detail"]


def test_every_registered_check_is_callable():
    assert all(callable(fn) for fn in CHECKS.values())


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_no_check_raises_on_empty_turns(name):
    """A transport failure yields empty turns; no check may explode on it."""
    spec = {"check": name}
    defaults = {
        "tool_called": {"name": "x"},
        "tool_not_called": {"name": "x"},
        "any_tool_called": {"names": ["x"]},
        "tool_order": {"first": "a", "then": "b"},
        "tool_call_count": {"name": "x", "maximum": 1},
        "tool_arg_equals": {"name": "x", "arg": "a", "expected": 1},
        "tool_arg_present": {"name": "x", "arg": "a"},
        "no_fabricated_titles": {"forbidden": ["x"]},
        "response_contains": {"needle": "x"},
        "response_contains_any": {"needles": ["x"]},
        "response_not_contains": {"needle": "x"},
        "recall_prior_title": {"title": "x"},
        "recommends_at_least": {"count": 1},
    }
    spec.update(defaults.get(name, {}))
    result = run_check(spec, [])
    assert isinstance(result["passed"], bool)
