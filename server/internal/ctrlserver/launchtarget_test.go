package ctrlserver

import "testing"

// fakeLauncher records Launch calls and returns a fixed target.
type fakeLauncher struct {
	host             string
	port             int32
	calls            int
	lastMap, lastRoom int32
}

func (f *fakeLauncher) Launch(mapID, room int32) (string, int32, error) {
	f.calls++
	f.lastMap, f.lastRoom = mapID, room
	return f.host, f.port, nil
}

// TestLaunchTargetFallsBackToShared: with no per-match launcher, launches point at
// the shared BattleHost/BattlePorts (the pre-existing single-server behaviour).
func TestLaunchTargetFallsBackToShared(t *testing.T) {
	s := New()
	ip, ports := s.launchTarget(101, 500)
	if ip != s.BattleHost {
		t.Fatalf("ip = %q, want shared %q", ip, s.BattleHost)
	}
	if len(ports.Dense) != 1 || ports.Dense[0].(int32) != s.BattlePorts[0] {
		t.Fatalf("ports = %v, want shared %v", ports.Dense, s.BattlePorts)
	}
}

// TestLaunchTargetUsesMatchLauncher: with a launcher set, launches point at the
// room's dedicated per-match server, and the map/room are threaded through.
func TestLaunchTargetUsesMatchLauncher(t *testing.T) {
	s := New()
	fl := &fakeLauncher{host: "10.0.0.9", port: 55123}
	s.MatchLauncher = fl

	ip, ports := s.launchTarget(101, 500)
	if ip != "10.0.0.9" {
		t.Fatalf("ip = %q, want launcher host 10.0.0.9", ip)
	}
	if len(ports.Dense) != 1 || ports.Dense[0].(int32) != 55123 {
		t.Fatalf("ports = %v, want [55123]", ports.Dense)
	}
	if fl.calls != 1 || fl.lastMap != 101 || fl.lastRoom != 500 {
		t.Fatalf("launcher call wrong: calls=%d map=%d room=%d, want 1/101/500", fl.calls, fl.lastMap, fl.lastRoom)
	}
}
