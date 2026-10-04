"""Orchestrator: Claude Agent SDK session with 4 sub-agents, 3 stdio MCP servers + 1 in-process server, 1 skill.

  python -m agents.orchestrator --cutoff 2020-09-22                  # stops before the board (review first)
  python -m agents.orchestrator --cutoff 2020-09-22 --compose-board  # full run incl. final_board.png
  python -m agents.orchestrator --season summer                      # bonus: summer cutoff (2020-05-26)
  python -m agents.orchestrator --mock                               # dry run: mock images, outputs/mock_run/

Auth: uses ANTHROPIC_API_KEY if a real one is set, otherwise the local Claude Code login.
Every tool call (which agent, tool, args, duration, ok) is logged to outputs/runs/<run_id>/trace.jsonl
and copied to outputs/trace.jsonl. Image generation is capped in code (4 new images, max 1 revision per run).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from claude_agent_sdk import (AgentDefinition, AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, HookMatcher,
                              PermissionResultAllow, PermissionResultDeny, ResultMessage, TextBlock, ToolUseBlock)

import config
from agents import runtools

ROOT = config.ROOT
PROMPTS = ROOT / "agents" / "prompts"
SKILL = ROOT / ".claude" / "skills" / "style-dna-brief" / "SKILL.md"
SEASON_CUTOFFS = {"summer": "2020-05-26", "autumn": "2020-09-22"}
RUN_IMAGE_BUDGET, RUN_MAX_REVISIONS = 4, 1
IDLE_S, MAX_NUDGES = 60, 6  # harness: wait for async sub-agents; nudge at most 6 times


def _mcp(name: str) -> list[str]:
    """Fully-qualified tool names for one of our servers."""
    tools = {
        "retail": ["get_style_attributes", "get_sales_curve", "get_reference_images", "season_summary"],
        "forecast": ["predict_top_k", "explain_style", "write_evidence", "evaluation_report"],
        "image": ["list_concepts", "generate_concept", "similarity_score", "novelty_check", "compose_board"],
    }[name]
    return [f"mcp__{name}__{t}" for t in tools]


def agents() -> dict[str, AgentDefinition]:
    """The four specialist sub-agents with their tool whitelists."""
    p = lambda n: (PROMPTS / f"{n}.md").read_text(encoding="utf-8")  # noqa: E731
    return {
        "forecaster": AgentDefinition(
            description="Forecasts the diversified top-3 styles for a cutoff, explains them (SHAP) and writes their "
                        "evidence packs. Use first.",
            prompt=p("forecaster"), model="sonnet", background=False,
            tools=[*_mcp("forecast"), "mcp__retail__season_summary"]),
        "style_analyst": AgentDefinition(
            description="Turns ONE winning style into a validated KEEP/CHANGE design brief + image prompt using the "
                        "style-dna-brief skill. Call once per winner.",
            prompt=p("analyst") + "\n\n---\n\n" + SKILL.read_text(encoding="utf-8"), model="sonnet", background=False,
            tools=[*_mcp("retail"), "Read", "Skill", "mcp__run__validate_brief"], skills=["style-dna-brief"]),
        "designer": AgentDefinition(
            description="Generates (or reuses) ONE concept image for a validated brief; also applies a single "
                        "critic revision note.",
            prompt=p("designer"), model="sonnet", background=False,
            tools=["mcp__image__list_concepts", "mcp__image__generate_concept"]),
        "critic": AgentDefinition(
            description="Judges concepts (CLIP novelty + visual check), approves or gives ONE revision note, and "
                        "records the decision.",
            prompt=p("critic"), model="sonnet", background=False,
            tools=["mcp__image__novelty_check", "mcp__image__similarity_score", "Read", "mcp__run__record_critique"]),
    }


class Tracer:
    """Hook callbacks that log every tool call and sub-agent start/stop to a JSONL trace."""

    def __init__(self, path: Path) -> None:
        self.path, self.t0, self.start = path, time.time(), {}
        self.counts: dict[str, int] = {}
        self.running: set[str] = set()  # sub-agent ids started but not yet stopped

    def _write(self, rec: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _short(obj, n: int = 400) -> str:
        s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)
        return s if len(s) <= n else s[:n] + " …"

    async def pre(self, inp, tool_use_id, ctx) -> dict:
        """PreToolUse: remember the start time."""
        self.start[inp.get("tool_use_id") or tool_use_id] = time.time()
        return {}

    async def post(self, inp, tool_use_id, ctx) -> dict:
        """PostToolUse / PostToolUseFailure: log the call with its duration."""
        tid = inp.get("tool_use_id") or tool_use_id
        ok = inp["hook_event_name"] == "PostToolUse"
        agent = inp.get("agent_type") or "orchestrator"
        self.counts[agent] = self.counts.get(agent, 0) + 1
        self._write({"t": round(time.time() - self.t0, 2), "event": "tool_call", "agent": agent,
                     "tool": inp["tool_name"], "args": self._short(inp.get("tool_input")),
                     "duration_s": round(time.time() - self.start.pop(tid, time.time()), 2), "ok": ok,
                     "result": self._short(inp.get("tool_response") if ok else inp.get("error"), 300)})
        return {}

    async def subagent(self, inp, tool_use_id, ctx) -> dict:
        """SubagentStart / SubagentStop events (also track which sub-agents are still running)."""
        if inp["hook_event_name"] == "SubagentStart":
            self.running.add(inp.get("agent_id"))
        else:
            self.running.discard(inp.get("agent_id"))
        self._write({"t": round(time.time() - self.t0, 2), "event": inp["hook_event_name"],
                     "agent": inp.get("agent_type"), "agent_id": inp.get("agent_id")})
        return {}


def make_permission_gate(allowed: list[str], tracer: Tracer):
    """can_use_tool callback: every permission decision is made in code (never an unanswerable prompt).

    Allows the whitelist; Write only under outputs/; everything else is denied with a reason (and traced).
    """
    allowed_set = set(allowed)

    async def gate(tool_name: str, tool_input: dict, context) -> PermissionResultAllow | PermissionResultDeny:
        reason = None
        if tool_name not in allowed_set:
            reason = f"{tool_name} is not on this run's tool whitelist."
        elif tool_name == "Write":
            target = Path(tool_input.get("file_path", "")).resolve()
            if config.OUT_DIR.resolve() not in target.parents:
                reason = "Writes are only allowed under outputs/."
        if reason:
            tracer._write({"t": round(time.time() - tracer.t0, 2), "event": "permission_denied",
                           "tool": tool_name, "reason": reason})
            return PermissionResultDeny(message=reason)
        return PermissionResultAllow()

    return gate


def build_options(run_dir: Path, run_id: str, tracer: Tracer, model: str, max_turns: int,
                  max_budget: float, mock: bool = False) -> ClaudeAgentOptions:
    """ClaudeAgentOptions: MCP servers, sub-agents, skills, tool whitelist, hooks, budget."""
    py = sys.executable
    base_env = {"PYTHONIOENCODING": "utf-8"}
    if mock:
        base_env.update(IMAGE_BACKEND="mock", HM_EVIDENCE_DIR=str(config.EVIDENCE_DIR))
    stdio = lambda script, env=None: {"type": "stdio", "command": py,  # noqa: E731
                                      "args": [str(ROOT / "mcp_servers" / script)], "env": {**base_env, **(env or {})}}
    image_env = {"IMAGE_RUN_ID": run_id, "IMAGE_RUN_BUDGET": str(RUN_IMAGE_BUDGET),
                 "IMAGE_RUN_MAX_REVISIONS": str(RUN_MAX_REVISIONS)}
    allowed = ["Task", "Agent", "Read", "Write", "Skill", *_mcp("retail"), *_mcp("forecast"), *_mcp("image"),
               *runtools.TOOLS]
    return ClaudeAgentOptions(
        system_prompt=(PROMPTS / "orchestrator.md").read_text(encoding="utf-8"),
        mcp_servers={"retail": stdio("retail_data.py"), "forecast": stdio("forecast.py"),
                     "image": stdio("image_gen.py", image_env), "run": runtools.SERVER},
        agents=agents(),
        setting_sources=["project"],  # loads .claude/skills (style-dna-brief)
        tools=["Task", "Agent", "Read", "Write", "Skill"],
        # no allowed_tools on purpose: an allow-list entry would auto-approve and bypass the gate below
        disallowed_tools=["Bash", "Edit", "WebFetch", "WebSearch", "NotebookEdit"],
        can_use_tool=make_permission_gate(allowed, tracer), permission_mode="default",
        model=model, max_turns=max_turns, max_budget_usd=max_budget, cwd=str(ROOT),
        hooks={"PreToolUse": [HookMatcher(hooks=[tracer.pre])],
               "PostToolUse": [HookMatcher(hooks=[tracer.post])],
               "PostToolUseFailure": [HookMatcher(hooks=[tracer.post])],
               "SubagentStart": [HookMatcher(hooks=[tracer.subagent])],
               "SubagentStop": [HookMatcher(hooks=[tracer.subagent])]},
    )


async def run(cutoff: str, compose_board: bool, model: str, max_turns: int, max_budget: float,
              mock: bool = False) -> Path:
    """Run one orchestrated session; return the run directory."""
    if mock:  # everything (evidence, board, runs) goes under outputs/mock_run/
        config.EVIDENCE_DIR = config.OUT_DIR / "mock_run" / "evidence"
        config.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + ("-mock" if mock else "")
    run_dir = (config.EVIDENCE_DIR.parent if mock else config.OUT_DIR) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    runtools.RUN.update(run_id=run_id, cutoff=cutoff)
    tracer = Tracer(run_dir / "trace.jsonl")
    opts = build_options(run_dir, run_id, tracer, model, max_turns, max_budget, mock)
    prompt = (f"Run parameters: CUTOFF={cutoff} RUN_ID={run_id} RUN_DIR={run_dir.as_posix()} "
              f"COMPOSE_BOARD={'yes' if compose_board else 'no'}. Execute the plan.")
    print(f"run {run_id} | cutoff {cutoff} | model {model} | auth: "
          f"{'ANTHROPIC_API_KEY' if config.ANTHROPIC_API_KEY else 'Claude Code login'}")
    t, result = time.time(), None
    # Sub-agents run asynchronously in this CLI version: the orchestrator's turn can end while they work, and
    # their results arrive as later turns. So keep the session open until a turn ends with none running;
    # if all have finished but the orchestrator went quiet, nudge it once per idle period.
    async with ClaudeSDKClient(options=opts) as client:  # streaming session: needed for can_use_tool
        await client.query(prompt)
        it = client.receive_messages().__aiter__()
        nudges = 0
        while True:
            try:
                msg = await asyncio.wait_for(it.__anext__(), timeout=IDLE_S)
            except asyncio.TimeoutError:
                if tracer.running or nudges >= MAX_NUDGES:
                    if not tracer.running:
                        break
                    continue
                nudges += 1
                print("[harness] sub-agents finished — nudging the orchestrator to continue")
                await client.query("All launched sub-agents have finished and returned their results. "
                                   "Continue with the plan.")
                continue
            except StopAsyncIteration:
                break
            if isinstance(msg, AssistantMessage):
                who = "  [sub]" if msg.parent_tool_use_id else "[orch]"
                for b in msg.content:
                    if isinstance(b, TextBlock) and b.text.strip():
                        print(f"{who} {b.text.strip()[:300]}")
                    elif isinstance(b, ToolUseBlock):
                        print(f"{who} → {b.name}")
            elif isinstance(msg, ResultMessage):
                result = msg
                if not tracer.running and (run_dir / "run_summary.md").exists():
                    break
    summary = {"run_id": run_id, "cutoff": cutoff, "seconds": round(time.time() - t, 1),
               "turns": result.num_turns if result else None, "is_error": result.is_error if result else True,
               "cost_usd": result.total_cost_usd if result else None, "tool_calls_by_agent": tracer.counts,
               "permission_denials": [str(x) for x in (result.permission_denials or [])] if result else None,
               "final_message": result.result if result else None}
    (run_dir / "result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if not mock:
        shutil.copy(run_dir / "trace.jsonl", config.OUT_DIR / "trace.jsonl")
    print(f"\ndone in {summary['seconds']}s | turns {summary['turns']} | cost ${summary['cost_usd'] or 0:.2f} "
          f"(nominal; billed to the Claude Code plan when using the login) | tool calls {tracer.counts}")
    return run_dir


async def critique_only(run_id: str, code: str, concept: str, reference: str, context: str = "") -> dict:
    """Run ONLY the critic sub-agent on one concept, recording the verdict under an existing run id.

    Used for manual follow-ups (e.g. a user-approved extra revision) so they get the same judge,
    tools, permission gate and trace as the main run.
    """
    run_dir = config.OUT_DIR / "runs" / run_id
    meta = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    runtools.RUN.update(run_id=run_id, cutoff=meta["cutoff"])
    tracer = Tracer(run_dir / "trace.jsonl")
    critic = agents()["critic"]
    py = sys.executable
    opts = ClaudeAgentOptions(
        system_prompt=critic.prompt, model="claude-sonnet-5", max_turns=15, cwd=str(ROOT),
        mcp_servers={"image": {"type": "stdio", "command": py, "args": [str(ROOT / "mcp_servers" / "image_gen.py")],
                               "env": {"PYTHONIOENCODING": "utf-8"}}, "run": runtools.SERVER},
        tools=["Read"], can_use_tool=make_permission_gate(critic.tools, tracer), permission_mode="default",
        hooks={"PreToolUse": [HookMatcher(hooks=[tracer.pre])], "PostToolUse": [HookMatcher(hooks=[tracer.post])],
               "PostToolUseFailure": [HookMatcher(hooks=[tracer.post])]})
    brief = json.loads((config.EVIDENCE_DIR / code / "brief.json").read_text(encoding="utf-8"))
    prompt = (f"Judge ONE concept. product_code={code}; concept_path={concept}; reference_image={reference}.\n"
              f"Brief KEEP: {json.dumps(brief['keep'], ensure_ascii=False)}\n"
              f"Brief CHANGE: {json.dumps(brief['change'], ensure_ascii=False)}\n{context}")
    tracer._write({"t": 0, "event": "manual_critique_start", "agent": "critic", "concept": concept})
    text = ""
    async with ClaudeSDKClient(options=opts) as client:
        await client.query(prompt)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                text += "".join(b.text for b in msg.content if isinstance(b, TextBlock))
    for line in (config.EVIDENCE_DIR / code / "critic.jsonl").read_text(encoding="utf-8").splitlines()[::-1]:
        rec = json.loads(line)
        if Path(rec["concept_path"]).resolve() == Path(concept).resolve() and rec["run_id"] == run_id:
            return rec
    raise RuntimeError(f"Critic did not record a verdict. Its reply: {text[:500]}")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", default=None, help="last day of data to use (default 2020-09-22)")
    ap.add_argument("--season", choices=list(SEASON_CUTOFFS), help="shortcut for a seasonal cutoff")
    ap.add_argument("--compose-board", action="store_true", help="also compose outputs/final_board.png")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--critique", nargs=3, metavar=("RUN_ID", "CODE", "CONCEPT"),
                    help="only run the critic on one concept (manual follow-up), recorded under RUN_ID")
    ap.add_argument("--context", default="", help="extra context for --critique (e.g. the prompt used)")
    ap.add_argument("--mock", action="store_true", help="dry run with the mock image generator")
    ap.add_argument("--max-budget", type=float, default=15.0, help="stop if the (nominal) cost exceeds this, USD")
    a = ap.parse_args()
    if a.critique:
        from data_science import select
        run_id, code, concept = a.critique
        rec = asyncio.run(critique_only(run_id, code, concept, str(select.reference_images(code)[0]), a.context))
        print(json.dumps({k: rec[k] for k in ("decision", "note")}, ensure_ascii=False, indent=1),
              "| CLIP", rec["novelty"].get("max_sim_to_refs"), rec["novelty"].get("verdict"))
        return
    cutoff = a.cutoff or SEASON_CUTOFFS.get(a.season or "autumn")
    asyncio.run(run(cutoff, a.compose_board, a.model, a.max_turns, a.max_budget, a.mock))


if __name__ == "__main__":
    main()
