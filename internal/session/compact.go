package session

import (
	"context"
	"fmt"
	"late/internal/client"
	"late/internal/common"
	"sort"
	"strings"
	"time"
)

// Minimal, deterministic history compaction for the executor's
// context-exhaustion safeguard (internal/executor/context_guard.go): when a
// request is rejected (or would be) for exceeding the model's context window,
// the installed compactor frees room with two shape-safe passes and the
// executor retries the request over the shrunken conversation:
//
//  1. Span pass: elide a contiguous span of old history into a short digest.
//  2. Tool-result pass: replace large old tool results in place with short
//     placeholders. This is the pass that saves histories which are short in
//     message count but huge in bytes (e.g. a subagent whose opener carries
//     context files and whose value sits in a few very large tool outputs) —
//     where a span cut has nothing to bite on.
//
// Design constraints, in priority order:
//   - Never corrupt the conversation shape: an assistant message carrying
//     tool_calls is never separated from its tool results (elision keeps the
//     message, only shrinking its content), and no cut leaves a dangling
//     tool result.
//   - Never elide the original task (history[0], the first user message) or
//     the recent working set (the newest tool group, the recent tail).
//   - Deterministic and dependency-free: no scorer service, no shadow log.
//     The same history always yields the same cuts.

const (
	// minMessagesForCompaction bounds the span pass below: a history this
	// small has no meaningful old span to digest (the tool-result pass
	// still runs).
	minMessagesForCompaction = 8
	// minTailKeep is the floor on the untouched recent tail (n/4, at least
	// this many messages) for the span pass: the model's current working set.
	minTailKeep = 4
	// digestPreviewRunes caps each user-message preview embedded in the
	// digest, so the replacement can never approach the size of the span
	// it replaces.
	digestPreviewRunes = 200
	// maxDigestPreviews bounds how many mid-conversation user requests the
	// digest quotes. The first user message is always kept verbatim (index
	// 0), so these are follow-up instructions from the elided span.
	maxDigestPreviews = 2
	// toolResultElideMinTokens: tool results smaller than this are never
	// elided — the placeholder would approach the original's size, and small
	// outputs are often load-bearing (paths, exit codes).
	toolResultElideMinTokens = 192
	// toolResultElisionPlaceholder is the content left in place of an elided
	// result. It names the tool so the model can re-run it if the output
	// matters.
	toolResultElisionPlaceholder = "[tool output elided by auto-compaction (~%d tokens): %s output was removed to fit the context window. Re-run the tool if you need this result.]"
)

// hasToolCalls reports whether m is an assistant message carrying tool_calls:
// a group head whose tool results must never be separated from it by a cut.
func hasToolCalls(m client.ChatMessage) bool {
	return m.Role == "assistant" && len(m.ToolCalls) > 0
}

// isToolResult reports whether m is a tool result message.
func isToolResult(m client.ChatMessage) bool {
	return m.Role == "tool"
}

// CompactContext shrinks history until a rejected request fits the model's
// window: first the span pass, then the tool-result pass. It returns the
// estimated token delta and the number of messages removed or rewritten
// (0, 0, nil when nothing could be freed). The context is honored for early
// cancellation. On success the change is persisted; a persistence failure is
// swallowed when the in-memory history DID shrink (self-healing: the next
// appendMessage re-persists everything) and surfaced when it did not.
//
// When the passes free nothing, HistoryProfile exposes why — the callers'
// diagnostics sink should surface it (see main.go / agent.go compactor
// closures).
func (s *Session) CompactContext(ctx context.Context) (tokensSaved, messagesCompacted int, err error) {
	if err := ctx.Err(); err != nil {
		return 0, 0, err
	}

	// Pass 1: span digest over the old history.
	tokens := tokenEstimates(s.History)
	history, freed, removed := compactSpan(s.History, tokens)

	// Pass 2: elide large old tool results in place (on the pass-1 result).
	tokens = tokenEstimates(history)
	history, elidedTokens, elidedCount := elideLargeToolResults(history, tokens)

	totalFreed := freed + elidedTokens
	if totalFreed <= 0 {
		return 0, 0, nil
	}

	s.History = history
	if persistErr := s.saveAndNotify(); persistErr != nil {
		// Persistence failed: surface the error only when the pass freed
		// nothing, because then there is no recovery to report. When the
		// in-memory history DID shrink, stay silent — the executor's retry
		// should proceed, and the next appendMessage re-persists the full
		// shrunken history anyway (a failed persist here is self-healing,
		// not fatal).
		if freed <= 0 && elidedTokens <= 0 {
			return 0, 0, persistErr
		}
	}
	return totalFreed, removed + elidedCount, nil
}

// tokenEstimates returns per-message token estimates using the same BPE
// counter the predictive guard uses, so "freed" means what the context bar
// and the provider both see.
func tokenEstimates(history []client.ChatMessage) []int {
	tokens := make([]int, len(history))
	for idx, m := range history {
		tokens[idx] = common.EstimateMessageTokens(m)
	}
	return tokens
}

// compactSpan elides a contiguous span of old history and replaces it with a
// short digest user message. The cut [i, j) is chosen group-safe: after
// boundary adjustment, history[i-1] never carries unanswered tool_calls into
// the cut and the suffix never opens on an orphaned tool result, so the
// compacted history is shape-identical to one produced by a well-formed
// shorter conversation. Pure: returns the new slice (or the input unchanged
// when no cut pays for itself).
func compactSpan(history []client.ChatMessage, tokens []int) ([]client.ChatMessage, int, int) {
	n := len(history)
	if n < minMessagesForCompaction {
		return history, 0, 0
	}

	// The recent tail is the live working set: never elided. jMax is the
	// exclusive upper bound of the cut range.
	tailKeep := n / 4
	if tailKeep < minTailKeep {
		tailKeep = minTailKeep
	}
	jMax := n - tailKeep
	if jMax <= 1 {
		return history, 0, 0
	}

	// Start boundary: advance i past every position that would split a tool
	// group — an assistant with tool_calls kept at i-1 whose results would
	// fall inside the cut, or a tool result at i whose call would stay
	// behind. The loop lands i on a group boundary (a user or plain
	// assistant message following a closed group).
	i := 1
	for i < jMax && (hasToolCalls(history[i-1]) || isToolResult(history[i])) {
		i++
	}

	// End boundary: pull j back so the cut closes cleanly — no tool result
	// at j whose call would be elided, and no assistant-with-tool_calls at
	// j-1 whose results would survive as orphans in the suffix.
	j := jMax
	for j > i && ((j < n && isToolResult(history[j])) || hasToolCalls(history[j-1])) {
		j--
	}
	if j <= i {
		return history, 0, 0
	}

	// The digest must be smaller than what it replaces, or the pass freed
	// nothing and cannot earn the executor's retry.
	elapsedTokens := 0
	for idx := i; idx < j; idx++ {
		elapsedTokens += tokens[idx]
	}
	digestText := buildDigestText(history[i:j], elapsedTokens)
	digestTokens := common.EstimateMessageTokens(client.ChatMessage{Role: "user", Content: client.TextContent(digestText)})
	freed := elapsedTokens - digestTokens
	if freed <= 0 {
		return history, 0, 0
	}

	compacted := make([]client.ChatMessage, 0, n-(j-i)+1)
	compacted = append(compacted, history[:i]...)
	compacted = append(compacted, client.ChatMessage{
		Role:      "user",
		Content:   client.TextContent(digestText),
		Timestamp: time.Now().Format(time.RFC3339),
	})
	compacted = append(compacted, history[j:]...)
	return compacted, freed, j - i
}

// elideLargeToolResults replaces large tool results — except those of the
// MOST RECENT tool group — with short placeholders, in place. The message
// itself stays, so the assistant tool_calls / tool result pairing is never
// broken; only the content shrinks. Results are elided largest-first
// (deterministic, tie-break by index) and only when they clear
// toolResultElideMinTokens, so the placeholder never approaches the
// original's size. Pure: returns the input unchanged when nothing is worth
// eliding.
func elideLargeToolResults(history []client.ChatMessage, tokens []int) ([]client.ChatMessage, int, int) {
	// protectFrom is the index from which tool results are the live working
	// set — never elided. When the MOST RECENT tool group's results extend
	// to the end of history, only its LAST result is live (the output the
	// model is about to consume); the group's older results are already
	// stale working set and elidable. A batched call that read seven files
	// once produced a single group holding ~34k tokens — protecting all of
	// it defeated every pass (observed via HistoryProfile). A group
	// followed by later messages is old history entirely: nothing protected.
	lastGroupHead := -1
	for idx := range history {
		if hasToolCalls(history[idx]) {
			lastGroupHead = idx
		}
	}
	protectFrom := len(history)
	if lastGroupHead >= 0 {
		end := lastGroupHead + 1
		for end < len(history) && isToolResult(history[end]) {
			end++
		}
		if end == len(history) {
			// Group runs to the end: the live working set. With multiple
			// results, only the LAST is live (the output the model is about
			// to consume) — the group's older results are stale and
			// elidable. A single-result group stays fully protected.
			if end-1 > lastGroupHead+1 {
				protectFrom = end - 1
			} else {
				protectFrom = lastGroupHead + 1
			}
		}
	}

	// Eligible: tool results before the protected region that clear the
	// size floor.
	type candidate struct {
		idx    int
		tokens int
	}
	var candidates []candidate
	for idx, m := range history {
		if idx >= protectFrom || !isToolResult(m) || tokens[idx] < toolResultElideMinTokens {
			continue
		}
		candidates = append(candidates, candidate{idx: idx, tokens: tokens[idx]})
	}
	if len(candidates) == 0 {
		return history, 0, 0
	}
	sort.Slice(candidates, func(a, b int) bool {
		if candidates[a].tokens != candidates[b].tokens {
			return candidates[a].tokens > candidates[b].tokens
		}
		return candidates[a].idx < candidates[b].idx
	})

	// toolNames maps tool_call IDs to names by walking the assistant group
	// heads once, so placeholders can name the tool that produced them.
	toolNames := make(map[string]string)
	for _, m := range history {
		if !hasToolCalls(m) {
			continue
		}
		for _, tc := range m.ToolCalls {
			if tc.ID != "" {
				toolNames[tc.ID] = tc.Function.Name
			}
		}
	}

	out := make([]client.ChatMessage, len(history))
	copy(out, history)
	freed, count := 0, 0
	for _, c := range candidates {
		toolName := toolNames[history[c.idx].ToolCallID]
		if toolName == "" {
			toolName = "tool"
		}
		placeholder := client.TextContent(fmt.Sprintf(toolResultElisionPlaceholder, c.tokens, toolName))
		placeholderTokens := common.EstimateMessageTokens(client.ChatMessage{Role: "tool", Content: placeholder})
		net := c.tokens - placeholderTokens
		if net <= 0 {
			continue
		}
		out[c.idx] = client.ChatMessage{
			Role:       "tool",
			ToolCallID: history[c.idx].ToolCallID,
			Content:    placeholder,
			Timestamp:  history[c.idx].Timestamp,
		}
		freed += net
		count++
	}
	if count == 0 {
		return history, 0, 0
	}
	return out, freed, count
}

// buildDigestText renders the digest that replaces the elided span: the
// elision notice, bounded previews of the user requests inside the span, and
// the distinct tools invoked there — enough for the model to keep its
// bearings, a tiny fraction of the span it replaces.
func buildDigestText(elided []client.ChatMessage, elapsedTokens int) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "[context auto-compacted: %d earlier messages (~%d tokens) were elided to fit the model's context window. Work continues from the messages below.]", len(elided), elapsedTokens)

	previews := 0
	toolNames := make(map[string]struct{})
	for _, m := range elided {
		if m.Role == "user" && previews < maxDigestPreviews {
			previews++
			fmt.Fprintf(&sb, "\nEarlier user request: %s", truncateUTF8(m.Content.String(), digestPreviewRunes))
		}
		for _, tc := range m.ToolCalls {
			toolNames[tc.Function.Name] = struct{}{}
		}
	}
	if len(toolNames) > 0 {
		names := make([]string, 0, len(toolNames))
		for name := range toolNames {
			names = append(names, name)
		}
		sort.Strings(names)
		fmt.Fprintf(&sb, "\nTools invoked in the elided span: %s", strings.Join(names, ", "))
	}
	return sb.String()
}

// HistoryProfile renders a one-line per-message breakdown of the session's
// history: role, index, and token estimate for each message. It exists for
// the compaction-safeguard's failure path: when the passes free nothing, the
// surfaced "freed nothing" message alone is undiagnosable — this profile
// shows which messages hold the context (a huge opener, a tail-sized tool
// result, many medium ones) so the protection rules can be reasoned about
// from the field.
//
// Single line by contract: the TUI's DiagnosticMsg handler truncates toast
// text to the terminal width. Message content is never included — only
// sizes, so the line also stays free of any sensitive file content.
func (s *Session) HistoryProfile() string {
	tokens := tokenEstimates(s.History)
	total := 0
	for _, t := range tokens {
		total += t
	}
	var sb strings.Builder
	fmt.Fprintf(&sb, "compaction freed nothing; history: %d msgs, ~%d tokens:", len(s.History), total)
	for idx, m := range s.History {
		if idx > 0 {
			sb.WriteString(" |")
		}
		role := m.Role
		tag := ""
		switch {
		case hasToolCalls(m):
			tag = "+calls"
		case isToolResult(m):
			tag = "-result"
		}
		if tag != "" {
			role += tag
		}
		fmt.Fprintf(&sb, " %d:%s~%d", idx, role, tokens[idx])
	}
	return sb.String()
}
