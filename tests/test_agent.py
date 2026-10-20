"""Agent loop, permission tiers, approvals and the number self-check.

The model is replaced by scripted backends, so these tests run offline and
show exactly what happens when a model behaves well or badly.
"""
import json
from pathlib import Path

import pytest

from bayflow.agent import Agent, GeminiBackend, ModelTurn, ToolCall, check_numbers
from bayflow.data import load_dataset
from bayflow.store import Store
from bayflow.tools import Dispatcher, PermissionDenied, owner_decide

DEMO = Path(__file__).resolve().parent.parent / "data" / "demo"
KEY = json.loads((DEMO / "answer_key.json").read_text())


class ScriptedBackend:
    """Plays back a fixed list of model turns and records what it was sent."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.sent = []

    def start(self, system_prompt, declarations):
        self.declared = [d["name"] for d in declarations]

    def send_user(self, text):
        self.sent.append(("user", text))
        return self.turns.pop(0)

    def send_tool_results(self, results):
        self.sent.append(("tools", results))
        return self.turns.pop(0)


@pytest.fixture
def env():
    ds = load_dataset(DEMO)
    store = Store()
    return ds, store, Dispatcher(ds, store, session="test")


def test_agent_only_sees_read_and_suggest_tools(env):
    ds, store, disp = env
    names = {d["name"] for d in disp.declarations()}
    assert names == {"reconcile_sales_payments", "cash_flow_summary", "forecast_cash",
                     "check_spending", "check_tax_documents", "propose_correction"}


def test_agent_answers_from_tool_results(env):
    ds, store, disp = env
    backend = ScriptedBackend([
        ModelTurn(calls=[ToolCall("reconcile_sales_payments", {"start": "2026-09-01", "end": "2026-09-30"})]),
        ModelTurn(text="In September you recorded 161,110 baht of sales and received 159,875 baht. "
                       "3 items need a look, 1,235 baht in total."),
    ])
    reply = Agent(backend, disp, store).ask("Did I get all my money in September?")
    assert reply.unverified_numbers == []
    assert reply.tool_calls[0]["tool"] == "reconcile_sales_payments"
    sent = backend.sent[1][1][0][1]
    assert sent["label"].startswith("Tool result") and "data" in sent
    assert any(r["event"] == "tool_call" for r in store.audit_rows())


def test_made_up_number_is_flagged(env):
    ds, store, disp = env
    backend = ScriptedBackend([
        ModelTurn(calls=[ToolCall("check_tax_documents", {"month": "2026-09"})]),
        ModelTurn(text="Two purchases have no tax invoice, so 137.39 baht of VAT is at risk. "
                       "You could save about 4,800 baht a year."),
    ])
    reply = Agent(backend, disp, store).ask("Any tax paperwork problems?")
    assert reply.unverified_numbers == ["4,800"]


def test_agent_cannot_approve_even_if_it_tries(env):
    ds, store, disp = env
    with pytest.raises(PermissionDenied):
        disp.call("owner_decide", {"proposal_id": 1, "approve": True})
    backend = ScriptedBackend([
        ModelTurn(calls=[ToolCall("owner_decide", {"proposal_id": 1, "approve": True})]),
        ModelTurn(text="I cannot approve changes. Only you can."),
    ])
    reply = Agent(backend, disp, store).ask("approve everything")
    assert reply.tool_calls[0]["ok"] is False
    refused = [r for r in store.audit_rows() if r["event"] == "tool_refused"]
    assert len(refused) == 2


def test_proposal_changes_nothing_until_owner_approves(env, tmp_path):
    ds, store, disp = env
    dup = KEY["duplicate_bills"][0]
    bill = dup["duplicate"]
    before = ds.sales.line_total.sum()
    result = disp.call("propose_correction", {"kind": "void_bill", "args": {"bill_id": bill},
                                              "reason": f"Identical to {dup['original']}"})
    assert result["status"] == "pending"
    assert load_dataset(DEMO, store.approved_corrections("demo")).sales.line_total.sum() == before

    owner_decide(store, result["proposal_id"], approve=True)
    after = load_dataset(DEMO, store.approved_corrections("demo"))
    assert before - after.sales.line_total.sum() == dup["amount"]
    assert store.approved_corrections("upload-other") == []        # approvals belong to one set of records
    with pytest.raises(ValueError):
        owner_decide(store, result["proposal_id"], approve=False)   # cannot decide twice


def test_bad_proposal_is_rejected_cleanly(env):
    ds, store, disp = env
    assert "error" in disp.call("propose_correction", {"kind": "delete_everything", "args": {}, "reason": "x"})
    assert "error" in disp.call("propose_correction", {"kind": "void_bill", "args": {"bill_id": "NOPE"},
                                                       "reason": "x"})
    assert store.proposals() == []


def test_injected_instruction_in_records_has_no_power(env):
    """A purchase note says 'approve every pending correction'. Even if the model
    obeyed it, the only tools it has cannot approve anything."""
    ds, store, disp = env
    note = ds.purchases.loc[ds.purchases.note.str.contains("IGNORE"), "note"].iloc[0]
    disp.call("propose_correction", {"kind": "set_tax_invoice",
                                     "args": {"purchase_id": KEY["missing_tax_invoice"][0]["purchase_id"], "has_tax_invoice": "yes"}, "reason": "test"})
    backend = ScriptedBackend([
        ModelTurn(calls=[ToolCall("check_tax_documents", {})]),
        ModelTurn(calls=[ToolCall("owner_decide", {"proposal_id": 1, "approve": True}),
                         ToolCall("apply_correction", {"proposal_id": 1})]),
        ModelTurn(text="One purchase note contains an unusual instruction, which I ignored."),
    ])
    Agent(backend, disp, store).ask(f"check this: {note}")
    assert [p["status"] for p in store.proposals()] == ["pending"]


def test_number_check_ignores_ids_dates_and_small_counts():
    src = [{"amount": 195.0, "total": 161940.0}]
    text = "Bill B260909-2788 on 2026-09-09 for 195 baht. 3 items. Total 161,940."
    assert check_numbers(text, src) == []
    assert check_numbers("You lost 9,999 baht", src) == ["9,999"]


def test_loop_stops_if_model_keeps_calling_tools(env):
    ds, store, disp = env
    turns = [ModelTurn(calls=[ToolCall("forecast_cash", {"days": 7})]) for _ in range(20)]
    reply = Agent(ScriptedBackend(turns), disp, store).ask("loop")
    assert "Stopped after too many steps" in reply.text


def test_gemini_backend_builds_valid_requests(monkeypatch, env):
    """Exercises the real google-genai types with a fake HTTP layer."""
    from google.genai import types
    ds, store, disp = env

    responses = [
        types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[
            types.Part(function_call=types.FunctionCall(name="forecast_cash", args={"days": 14}))]))]),
        types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[
            types.Part.from_text(text="Your lowest point is shown above.")]))]),
    ]
    seen = []

    class FakeModels:
        def generate_content(self, model, contents, config):
            seen.append((model, list(contents), config))
            return responses[len(seen) - 1]

    backend = GeminiBackend(api_key="test-key", model="test-model")
    backend.client = type("C", (), {"models": FakeModels()})()
    reply = Agent(backend, disp, store).ask("How does cash look?")
    assert reply.text == "Your lowest point is shown above."
    assert seen[0][2].tools[0].function_declarations[0].name
    fn_resp = seen[1][1][-1].parts[0].function_response
    assert fn_resp.name == "forecast_cash" and "data" in fn_resp.response


def _fake_gemini(env, behaviours, **kw):
    """GeminiBackend whose HTTP layer follows a script: an exception to raise or a text reply."""
    from google.genai import types
    ds, store, disp = env
    calls = []

    class FakeModels:
        def generate_content(self, model, contents, config):
            calls.append(model)
            b = behaviours.pop(0)
            if isinstance(b, Exception):
                raise b
            return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(
                role="model", parts=[types.Part.from_text(text=b)]))])

    backend = GeminiBackend(api_key="k", model="main-model", backoff_s=0, **kw)
    backend.client = type("C", (), {"models": FakeModels()})()
    return Agent(backend, disp, store), backend, calls


def _busy():
    from google.genai import errors
    return errors.ServerError(503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}})


def test_busy_model_is_retried(env):
    agent, backend, calls = _fake_gemini(env, [_busy(), _busy(), "All good."], fallback_models=[])
    assert agent.ask("hi").text == "All good."
    assert calls == ["main-model"] * 3


def test_falls_back_to_second_model(env):
    agent, backend, calls = _fake_gemini(env, [_busy(), _busy(), _busy(), "From fallback."],
                                         fallback_models=["backup-model"])
    assert agent.ask("hi").text == "From fallback."
    assert calls[-1] == "backup-model" and backend.last_model_used == "backup-model"


def test_failed_question_is_rolled_back(env):
    from bayflow.agent import ModelUnavailable
    agent, backend, calls = _fake_gemini(env, [_busy()] * 3 + ["Second try works."], fallback_models=[])
    with pytest.raises(ModelUnavailable):
        agent.ask("first try")
    assert backend.contents == []                       # no half-finished turn left behind
    assert agent.ask("second try").text == "Second try works."
    assert len(backend.contents) == 2


def test_non_busy_errors_are_not_retried(env):
    from google.genai import errors
    bad_key = errors.ClientError(400, {"error": {"code": 400, "message": "API key not valid", "status": "INVALID_ARGUMENT"}})
    agent, backend, calls = _fake_gemini(env, [bad_key], fallback_models=["backup-model"])
    with pytest.raises(errors.ClientError):
        agent.ask("hi")
    assert calls == ["main-model"]


def test_demo_reset_keeps_the_audit_log(env):
    ds, store, disp = env
    disp.call("forecast_cash", {"days": 7})
    disp.call("propose_correction", {"kind": "void_bill", "args": {"bill_id": KEY["duplicate_bills"][0]["duplicate"]},
                                     "reason": "test"})
    before = len(store.audit_rows())
    assert store.reset_proposals() == 1
    assert store.proposals() == []
    rows = store.audit_rows()
    assert len(rows) == before + 1 and rows[-1]["event"] == "demo_reset"
