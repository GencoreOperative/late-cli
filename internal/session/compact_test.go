package session

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"late/internal/client"
)

// compactTestSession builds a session with a persistent history file in a
// temp dir, mirroring the session_pop_test.go conventions (stubbed
// SessionDir, t.TempDir()).
func compactTestSession(t *testing.T, msgs []client.ChatMessage) *Session {
	t.Helper()
	tmpDir := t.TempDir()
	oldSessionDir := SessionDir
	SessionDir = func() (string, error) { return tmpDir, nil }
	t.Cleanup(func() { SessionDir = oldSessionDir })
	historyPath := filepath.Join(tmpDir, "session-compact.json")
	return New(nil, historyPath, msgs, "sp", true)
}

// fillHistory produces a deterministic message list: one leading user task,
// then fillers of role round-robin, each with distinct padding so token
// counts differ per message.
func fillHistory(n int) []client.ChatMessage {
	msgs := make([]client.ChatMessage, 0, n)
	msgs = append(msgs, client.ChatMessage{Role: "user", Content: client.TextContent("original task: build the feature")})
	roles := []string{"assistant", "user", "assistant", "user", "assistant", "user"}
	for idx := 1; idx < n; idx++ {
		msgs = append(msgs, client.ChatMessage{
			Role:    roles[idx%len(roles)],
			Content: client.TextContent(strings.Repeat("filler content for compaction testing ", 20) + string(rune('a'+idx%26))),
		})
	}
	return msgs
}

func TestCompactContext(t *testing.T) {
	t.Run("compacts old span, keeps head and tail, persists", func(t *testing.T) {
		s := compactTestSession(t, fillHistory(20))
		original := len(s.History)

		freed, compacted, err := s.CompactContext(context.Background())
		if err != nil {
			t.Fatalf("CompactContext returned error: %v", err)
		}
		if compacted == 0 || freed <= 0 {
			t.Fatalf("CompactContext freed nothing: tokens=%d messages=%d", freed, compacted)
		}
		if got := len(s.History); got >= original {
			t.Errorf("history shrank to %d from %d", got, original)
		}
		// Head preserved: the original task message is intact.
		if s.History[0].Role != "user" || !strings.Contains(s.History[0].Content.String(), "original task") {
			t.Errorf("first message changed: %+v", s.History[0])
		}
		// Tail preserved: the last original message survives verbatim.
		if got, want := s.History[len(s.History)-1].Content.String(),
			fillHistory(20)[19].Content.String(); got != want {
			t.Errorf("last message changed:\n got %q\nwant %q", got, want)
		}
		// Exactly one digest message was inserted, recognizable by its text.
		digests := 0
		for _, m := range s.History {
			if strings.Contains(m.Content.String(), "[context auto-compacted:") {
				digests++
			}
		}
		if digests != 1 {
			t.Errorf("got %d digest messages, want exactly 1", digests)
		}
		// The change persisted.
		data, err := os.ReadFile(s.HistoryPath)
		if err != nil {
			t.Fatalf("history file missing after compaction: %v", err)
		}
		if !strings.Contains(string(data), "[context auto-compacted:") {
			t.Errorf("persisted history does not contain the digest")
		}
	})

	t.Run("never splits an assistant tool_calls group", func(t *testing.T) {
		// Build a history whose midpoint region is a tool group: assistant
		// with tool_calls followed by several tool results. A cut that keeps
		// the assistant but elides its results — or elides the assistant and
		// keeps a result — would corrupt the conversation shape.
		msgs := fillHistory(20)
		msgs[9] = client.ChatMessage{
			Role:      "assistant",
			ToolCalls: []client.ToolCall{{ID: "call-1", Function: client.FunctionCall{Name: "bash", Arguments: "{}"}}},
		}
		msgs[10] = client.ChatMessage{Role: "tool", ToolCallID: "call-1", Content: client.TextContent("tool output one")}
		msgs[11] = client.ChatMessage{Role: "tool", ToolCallID: "call-1", Content: client.TextContent("tool output two")}
		s := compactTestSession(t, msgs)

		if _, _, err := s.CompactContext(context.Background()); err != nil {
			t.Fatalf("CompactContext returned error: %v", err)
		}

		// Invariants over the compacted history: every tool result's call
		// survives, and no assistant tool-call is orphaned from its results.
		callIDs := make(map[string]int) // tool_call_id -> results seen
		for _, m := range s.History {
			for _, tc := range m.ToolCalls {
				if tc.ID != "" {
					callIDs[tc.ID] = 0
				}
			}
			if m.Role == "tool" {
				callIDs[m.ToolCallID]++
			}
		}
		for id, results := range callIDs {
			if results != 1 {
				t.Errorf("tool call %q has %d results in compacted history, want 1", id, results)
			}
		}
	})

	t.Run("no-op on small history", func(t *testing.T) {
		s := compactTestSession(t, fillHistory(4))
		freed, compacted, err := s.CompactContext(context.Background())
		if err != nil {
			t.Fatalf("CompactContext returned error: %v", err)
		}
		if freed != 0 || compacted != 0 {
			t.Errorf("expected a no-op, got tokens=%d messages=%d", freed, compacted)
		}
		if len(s.History) != 4 {
			t.Errorf("history changed: %d messages", len(s.History))
		}
	})

	t.Run("no-op when span is smaller than its digest", func(t *testing.T) {
		// Very short messages: the span's tokens barely exceed the digest.
		msgs := make([]client.ChatMessage, 0, 12)
		msgs = append(msgs, client.ChatMessage{Role: "user", Content: client.TextContent("task")})
		for idx := 1; idx < 12; idx++ {
			msgs = append(msgs, client.ChatMessage{Role: "user", Content: client.TextContent("hi")})
		}
		s := compactTestSession(t, msgs)
		freed, compacted, err := s.CompactContext(context.Background())
		if err != nil {
			t.Fatalf("CompactContext returned error: %v", err)
		}
		if freed < 0 || compacted < 0 {
			t.Errorf("negative report: tokens=%d messages=%d", freed, compacted)
		}
		if len(s.History) != 12 {
			t.Errorf("history changed: %d messages", len(s.History))
		}
	})

	t.Run("canceled context returns error without touching history", func(t *testing.T) {
		s := compactTestSession(t, fillHistory(20))
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, _, err := s.CompactContext(ctx); err == nil {
			t.Fatalf("expected context error, got nil")
		}
		if len(s.History) != 20 {
			t.Errorf("history changed: %d messages", len(s.History))
		}
	})

	t.Run("elides large old tool results, keeps newest group intact", func(t *testing.T) {
		// The real-world failure shape: a history short in message count but
		// huge in bytes — several enormous tool results from earlier groups
		// plus one recent group. The span pass cannot bite (protected tail);
		// the tool-result pass must.
		msgs := fillHistory(6)
		big := strings.Repeat("tool output line ", 4000) // ~68k chars
		msgs[2] = client.ChatMessage{Role: "assistant", ToolCalls: []client.ToolCall{
			{ID: "old-1", Function: client.FunctionCall{Name: "read_file", Arguments: "{}"}},
		}}
		msgs[3] = client.ChatMessage{Role: "tool", ToolCallID: "old-1", Content: client.TextContent(big)}
		msgs[4] = client.ChatMessage{Role: "assistant", ToolCalls: []client.ToolCall{
			{ID: "new-1", Function: client.FunctionCall{Name: "bash", Arguments: "{}"}},
		}}
		msgs[5] = client.ChatMessage{Role: "tool", ToolCallID: "new-1", Content: client.TextContent(big)}
		s := compactTestSession(t, msgs)

		freed, compacted, err := s.CompactContext(context.Background())
		if err != nil {
			t.Fatalf("CompactContext returned error: %v", err)
		}
		if freed <= 0 || compacted == 0 {
			t.Fatalf("freed nothing: tokens=%d messages=%d", freed, compacted)
		}
		if len(s.History) != 6 {
			t.Fatalf("history length changed: %d, want 6 (elision keeps messages in place)", len(s.History))
		}
		// The OLD result was replaced by the placeholder...
		if !strings.Contains(s.History[3].Content.String(), "[tool output elided by auto-compaction") {
			t.Errorf("old large tool result was not elided: %q", truncateUTF8(s.History[3].Content.String(), 120))
		}
		// ...the NEWEST group's result is untouched (live working set)...
		if s.History[5].Content.String() != big {
			t.Errorf("newest tool group result was elided but must survive")
		}
		// ...and the pairing is intact.
		if s.History[2].ToolCalls[0].ID != "old-1" || s.History[3].ToolCallID != "old-1" {
			t.Errorf("tool pairing broken after elision")
		}
	})

	t.Run("elides older results inside the newest batched group", func(t *testing.T) {
		// The live-observed failure shape (HistoryProfile): one assistant
		// message makes seven batched read_file calls; the group's results
		// run to the end of history and hold ~34k tokens. The old rule
		// protected the WHOLE group and every pass freed nothing. Now only
		// the group's LAST result is protected.
		msgs := fillHistory(3) // 0:user 1:assistant 2:user
		msgs = append(msgs, client.ChatMessage{Role: "assistant", ToolCalls: []client.ToolCall{
			{ID: "b1", Function: client.FunctionCall{Name: "read_file", Arguments: "{}"}},
			{ID: "b2", Function: client.FunctionCall{Name: "read_file", Arguments: "{}"}},
			{ID: "b3", Function: client.FunctionCall{Name: "read_file", Arguments: "{}"}},
		}})
		big := strings.Repeat("file content ", 3000) // ~39k chars each
		msgs = append(msgs,
			client.ChatMessage{Role: "tool", ToolCallID: "b1", Content: client.TextContent(big)},
			client.ChatMessage{Role: "tool", ToolCallID: "b2", Content: client.TextContent(big)},
			client.ChatMessage{Role: "tool", ToolCallID: "b3", Content: client.TextContent(big)},
		)
		s := compactTestSession(t, msgs)

		freed, compacted, err := s.CompactContext(context.Background())
		if err != nil {
			t.Fatalf("CompactContext returned error: %v", err)
		}
		if freed <= 0 || compacted == 0 {
			t.Fatalf("freed nothing: tokens=%d messages=%d", freed, compacted)
		}
		if len(s.History) != 7 {
			t.Fatalf("history length changed: %d, want 7", len(s.History))
		}
		// The two OLDER results in the group are elided...
		if !strings.Contains(s.History[4].Content.String(), "[tool output elided by auto-compaction") {
			t.Errorf("first batched result not elided: %q", truncateUTF8(s.History[4].Content.String(), 120))
		}
		if !strings.Contains(s.History[5].Content.String(), "[tool output elided by auto-compaction") {
			t.Errorf("second batched result not elided: %q", truncateUTF8(s.History[5].Content.String(), 120))
		}
		// ...the LAST result survives untouched...
		if s.History[6].Content.String() != big {
			t.Errorf("last result of the newest group was elided but must survive")
		}
		// ...and every tool_call still has exactly one result (pairing intact).
		seen := make(map[string]int)
		for _, m := range s.History {
			for _, tc := range m.ToolCalls {
				seen[tc.ID] = 0
			}
			if m.Role == "tool" {
				seen[m.ToolCallID]++
			}
		}
		for id, results := range seen {
			if results != 1 {
				t.Errorf("tool call %q has %d results, want 1", id, results)
			}
		}
	})

	t.Run("elides every large result when no tool group follows", func(t *testing.T) {
		// A history whose last message is not a tool group: no group is the
		// "newest", so all large results are eligible. Six messages keeps the
		// span pass below its threshold, isolating the elide pass.
		msgs := fillHistory(6)
		big := strings.Repeat("verbose tool dump ", 3000)
		msgs[1] = client.ChatMessage{Role: "assistant", ToolCalls: []client.ToolCall{
			{ID: "g1", Function: client.FunctionCall{Name: "bash", Arguments: "{}"}},
		}}
		msgs[2] = client.ChatMessage{Role: "tool", ToolCallID: "g1", Content: client.TextContent(big)}
		s := compactTestSession(t, msgs)

		freed, compacted, err := s.CompactContext(context.Background())
		if err != nil {
			t.Fatalf("CompactContext returned error: %v", err)
		}
		if freed <= 0 || compacted == 0 {
			t.Fatalf("freed nothing: tokens=%d messages=%d", freed, compacted)
		}
		if !strings.Contains(s.History[2].Content.String(), "[tool output elided by auto-compaction") {
			t.Errorf("large tool result not elided: %q", truncateUTF8(s.History[2].Content.String(), 120))
		}
		if !strings.Contains(s.History[2].Content.String(), "bash") {
			t.Errorf("placeholder does not name the tool")
		}
	})

	t.Run("small tool results are never elided", func(t *testing.T) {
		// Six messages keeps the span pass below its threshold, isolating
		// the elide pass.
		msgs := fillHistory(6)
		msgs[1] = client.ChatMessage{Role: "assistant", ToolCalls: []client.ToolCall{
			{ID: "g1", Function: client.FunctionCall{Name: "bash", Arguments: "{}"}},
		}}
		msgs[2] = client.ChatMessage{Role: "tool", ToolCallID: "g1", Content: client.TextContent("exit 0")}
		s := compactTestSession(t, msgs)

		if _, _, err := s.CompactContext(context.Background()); err != nil {
			t.Fatalf("CompactContext returned error: %v", err)
		}
		if s.History[2].Content.String() != "exit 0" {
			t.Errorf("small tool result was elided: %q", s.History[2].Content.String())
		}
	})
}

func TestHistoryProfile(t *testing.T) {
	msgs := fillHistory(6)
	msgs[2] = client.ChatMessage{Role: "assistant", ToolCalls: []client.ToolCall{
		{ID: "g1", Function: client.FunctionCall{Name: "read_file", Arguments: "{}"}},
	}}
	msgs[3] = client.ChatMessage{Role: "tool", ToolCallID: "g1", Content: client.TextContent(strings.Repeat("x", 5000))}
	s := compactTestSession(t, msgs)

	profile := s.HistoryProfile()
	if !strings.HasPrefix(profile, "compaction freed nothing; history: 6 msgs, ~") {
		t.Fatalf("unexpected profile header: %q", profile)
	}
	// One line only — the TUI toast handler truncates to terminal width.
	if strings.Contains(profile, "\n") {
		t.Errorf("profile contains newlines, want a single line: %q", profile)
	}
	// Roles and tags present, tool result tagged and sized.
	for _, want := range []string{"0:user~", "2:assistant+calls~", "3:tool-result~"} {
		if !strings.Contains(profile, want) {
			t.Errorf("profile missing %q: %q", want, profile)
		}
	}
	// No message content leaks into the profile.
	if strings.Contains(profile, "filler content") || strings.Contains(profile, strings.Repeat("x", 100)) {
		t.Errorf("profile leaks message content: %q", profile)
	}
}
