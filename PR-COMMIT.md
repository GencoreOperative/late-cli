feat(session): minimal deterministic auto-compaction — light up the executor's context-exhaustion safeguard

PR #148 landed the context-exhaustion safeguard (typed detection,
no-retry-waste, compact+retry rounds) but left the compactor itself as a
stub that always returned ErrCompactionUnavailable — so a request over the
model's window died with "auto-compaction ran but freed nothing" instead
of recovering. This wires a working, dependency-free compactor into the
seam #148 already created; the executor-facing API is unchanged and PR
#139 can still replace the installed closure later.

Session.CompactContext (internal/session/compact.go) runs two
deterministic, shape-safe passes:

- Span pass: elides one contiguous block of old history into a short
  digest (elision notice + previews of up to 2 user requests + tools
  invoked). Never touches history[0] (the task) or the recent tail
  (max(n/4, 4) messages); cut boundaries slide until no assistant
  tool_calls group is split from its results.
- Tool-result pass: replaces large old tool results (>= 192 tokens) with
  short placeholders in place — message and tool-call pairing stay, only
  content shrinks; the newest tool group survives when it runs to the end
  of history. This pass handles histories that are short in message count
  but huge in bytes, where a span cut has nothing to bite on.

Both passes no-op honestly when they cannot net a saving, so the executor
never spends a retry on a pass that freed nothing. Tokens are measured
with common.EstimateMessageTokens — the same BPE counter the predictive
guard uses.

Per-session compactor registry (internal/executor/context_guard.go): the
guard resolves the compactor for the orchestrator actually running via
SetSessionContextCompactor(id, c), keyed by the orchestrator ID each
Execute injects into its ctx, falling back to the process-wide
SetContextCompactor. Without this, a subagent hitting the limit would
compact the ROOT's history and retry its own still-oversized request.
The root registers under "main"; each subagent registers over its own
session at spawn time (internal/agent/agent.go).

Feedback: every successful compaction announces itself as a TUI warning
toast via the existing diagnostics pipeline ("context auto-compacted:
saved ~N tokens (M messages)", prefixed "subagent <id>: " for children) —
covering the predictive pass (silent before the first request) and
subagents whose status line is only visible while focused. Reactive-path
status line and recovery toast are unchanged and now actually occur.

On by default, no config: the pass only fires at (or 95% toward) a real
limit, never touches the task or working set, and no-ops when it cannot
win. Trade stated plainly: no shadow log, no expand/undo — an elided
span or tool result is genuinely gone; the model's recourse is re-running
the tool (the placeholder says so).

Tests: new internal/session/compact_test.go (8 subtests: span invariants,
tool-group integrity, no-ops, cancellation, elision cases);
TestSessionContextCompactorResolvesByOrchestratorID pins per-ID
resolution, fallback, and removal; all pre-existing guard tests pass
unchanged against the new registry. Full go test ./... -race clean.

Verified live: a run that previously failed with "auto-compaction ran but
freed nothing" (20,160-token request vs 16,128 window) now recovers via
the tool-result pass and completes.

Co-Authored-By: Claude Code <noreply@anthropic.com>
