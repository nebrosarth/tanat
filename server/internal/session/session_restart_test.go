package session

import (
	"path/filepath"
	"testing"
)

// TestSessionKeySurvivesRestart: a client that was already logged in keeps its key across
// a server restart. Before this, sessions lived only in memory, so a restart silently
// invalidated the key the client still held -- its next broadcast-server (mpd) login was
// rejected and the player got «Не удалось подключиться к серверу рассылок» plus a
// reconnect box on a server that was up and working.
func TestSessionKeySurvivesRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tanat.db")

	s := NewPersistentStore(path)
	u, key, ok := s.LoginOrRegister("a@b.c", "pw")
	if !ok {
		t.Fatal("login failed")
	}
	uid := u.ID
	if err := s.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	s2 := NewPersistentStore(path) // "restart"
	defer s2.Close()
	got, valid := s2.BySessKey(key)
	if !valid {
		t.Fatal("the session key did not survive the restart -- mpd would reject the client")
	}
	if got.ID != uid {
		t.Fatalf("key resolved to user %d, want %d", got.ID, uid)
	}

	// Revoking must reach the DB too, or a restart would bring a dead key back.
	s2.InvalidateSession(key)
	s3 := NewPersistentStore(path)
	defer s3.Close()
	if _, valid := s3.BySessKey(key); valid {
		t.Fatal("a revoked key came back after a restart")
	}
}
