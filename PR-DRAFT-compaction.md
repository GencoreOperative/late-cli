## The problem

PR #148 landed the context-exhaustion safeguard — typed detection, no-retry-waste,
compact+retry rounds — but left the compactor itself as a stub:

```go
// cmd/late/main.go (before)
executor.SetContextCompactor(func(ctx context.Context) (executor.CompactionReport, error) {
    return executor.CompactionReport{}, executor.ErrCompactionUnavailable
})
```

So on context exhaustion the run always ends in:

```
**Context Limit Exceeded**
stream error: the conversation exceeded this model's context window: late could not
recover it automatically — start a new session with /new … : auto-compaction ran but
freed nothing
```

Observed live: a run failed with a 20,160-token request against a 16,128-token window —
the machinery around the stub (detection, retry tiers, TUI surfaces) all works, but
nothing ever shrinks history. The stub's comments point at an unmerged "compaction PR"
(#139), which remains open, conflicted with main, ~23k lines, and dependent on an
external scoring service.

## The fix

A minimal, dependency-free compactor wired into the seam #148 already created. The
executor-facing API is unchanged — #139 can still replace the installed closure later.

### `Session.CompactContext` (`internal/session/compact.go`, new)

Two deterministic, shape-safe passes over session history:

1. **Span pass** — elides one contiguous block of old history into a short digest:
   `[context auto-compacted: N earlier messages (~T tokens) were elided …]` plus previews
   of up to 2 user requests and the tools invoked. Never touches `history[0]` (the task)
   or the recent tail (max(n/4, 4) messages); cut boundaries slide until no
   assistant-`tool_calls` group is split from its results.
2. **Tool-result pass** — replaces large old tool results (≥ 192 tokens) with short
   placeholders in place; the message and its tool-call pairing stay, only the content
   shrinks: `[tool output elided by auto-compaction (~T tokens): <tool> output was
   removed to fit the context window. Re-run the tool if you need this result.]`

This second pass is what fixed the live failure above: agent histories are often short in
message count but huge in bytes (large tool outputs), where a span cut has nothing to
bite on.

Both passes no-op honestly (return zero) when they cannot net a saving, so the executor
never spends a retry on a pass that freed nothing. Tokens are measured with
`common.EstimateMessageTokens` — the same BPE counter the predictive guard uses.

### Per-session compactor registry (`internal/executor/context_guard.go`)

The guard now resolves the compactor for the orchestrator actually running:
`SetSessionContextCompactor(id, c)` keyed by orchestrator ID, resolved via the ID the
orchestrator injects into its ctx at `Execute` time, falling back to the process-wide
`SetContextCompactor`. Without this, a subagent hitting the limit would compact the
ROOT's history and retry its own still-oversized request — recovering nothing. The root
registers under `main`; each subagent registers over its own session at spawn time
(`internal/agent/agent.go`).

### User feedback

Every successful compaction announces itself as a TUI warning toast via the existing
diagnostics pipeline — `context auto-compacted: saved ~N tokens (M messages)`, prefixed
`subagent <id>: …` for children. This covers the two otherwise-invisible cases: the
predictive pass (runs silently before the first request) and a subagent whose status
line is only visible while focused. The reactive path's existing status line
("context limit hit — auto-compacted (saved ~N tokens, M messages compacted) —
retrying") and recovery toast are unchanged and now actually occur.

### Trigger paths (all pre-existing, unchanged)

- **Predictive**: compacts once before a request estimated above 95% of a known window
  (llama-server `/props` discovery or `context-size-tokens` config).
- **Reactive**: on 413 / `finish_reason=length` exhaustion, compact + retry, up to 2
  rounds. Failed attempts commit nothing, so the retry is a clean re-request.

**No config, on by default.** Deliberate: the pass only fires at (or 95% toward) a real
limit, never touches the task or working set, and no-ops when it cannot win — there is
no state where a user would prefer the old hard failure.

## Relationship to #139

Independent and compatible. #139's executor-facing seam (`ContextCompactor` closure) is
untouched; if it lands, it replaces the installed closure and the recovery path lights
up with its scorer without further executor changes. The trade stated plainly: this has
no shadow log and no `expand`/undo — an elided span or tool result is genuinely gone;
the model's recourse is re-running the tool (the placeholder says so).

## Testing

`go build ./...`, `go vet ./...`, `gofmt` clean. Full `go test ./... -count=1` green.

- New `internal/session/compact_test.go` (8 subtests): span compaction with
  head/tail/digest/persistence invariants; tool-group integrity under compaction;
  small-history and digest-too-big no-ops; canceled-context safety; tool-result elision
  (newest group kept, all-eligible case, small results never touched).
- `internal/executor/context_guard_test.go`: all pre-existing guard tests pass unchanged
  against the new registry; new `TestSessionContextCompactorResolvesByOrchestratorID`
  pins per-ID resolution, fallback, and removal.
- Verified live: a run that previously died with `auto-compaction ran but freed nothing`
  (20,160 vs 16,128 tokens) now recovers via the tool-result pass and completes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
