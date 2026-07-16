// Command ctrlserver runs the Tanat Online "Ctrl" HTTP server, i.e. the
// entry_point.php replacement the client talks to for login, hero,
// inventory, chat, clan etc., plus the raw-TCP "Battle" server that powers
// combat and the non-combat central-square hub. See internal/ctrlserver and
// internal/battleserver for coverage.
//
// The real client's default port is 80 (see the embedded production config
// at _decompiled/embedded_tanat_config.xml); we default to 8080 here so this
// doesn't require admin/root to run during development. To point the real
// client at this server, launch Tanat.exe with `-c <path-to-config.xml>`
// pointing control_server/host+port at this process (see local_test_config.xml).
package main

import (
	"flag"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"

	"tanatserver/internal/battleserver"
	"tanatserver/internal/ctrlserver"
	"tanatserver/internal/mpd"
	"tanatserver/internal/session"
)

func main() {
	addr := flag.String("addr", ":8080", "listen address for the Ctrl HTTP server")
	battleAddr := flag.String("battle-addr", ":9339", "listen address for the Battle TCP server")
	battleHost := flag.String("battle-host", "127.0.0.1", "host advertised to the client for the Battle server (in area_conf)")
	mpdAddr := flag.String("mpd-addr", ":9340", "listen address for the MPD push server (chat/party/presence)")
	logPath := flag.String("log", defaultLogPath(), "also write logs to this file (blank = stdout only)")
	accountsPath := flag.String("accounts", defaultAccountsPath(), "JSON file to persist accounts/heroes (blank = in-memory only)")
	dotaPlayers := flag.Int("dota-players", 1, "«Штурм» (DOTA) match size: players a match waits for before it starts (1-10; 1 = solo instant-match)")
	flag.Parse()

	if *logPath != "" {
		if f, err := os.Create(*logPath); err != nil {
			log.Printf("could not open log file %s: %v (stdout only)", *logPath, err)
		} else {
			defer f.Close()
			log.SetOutput(io.MultiWriter(os.Stdout, f))
			log.Printf("logging to %s", *logPath)
		}
	}

	srv := ctrlserver.New()
	if *accountsPath != "" {
		srv.Store = session.NewPersistentStore(*accountsPath)
	}
	if applied := srv.SetDotaMatchSize(int32(*dotaPlayers)); applied != int32(*dotaPlayers) {
		log.Printf("«Штурм» match size %d out of range, clamped to %d", *dotaPlayers, applied)
	}

	// The Battle server shares the Ctrl server's session store (CONNECT arrives
	// with the user's id; later the self-player/avatar chain will need the hero).
	battle := battleserver.New(srv.Store)
	if _, portStr, err := splitHostPort(*battleAddr); err == nil {
		if port, err := strconv.Atoi(portStr); err == nil {
			srv.BattleHost = *battleHost
			srv.BattlePorts = []int32{int32(port)}
		}
	}

	// The MPD push hub (chat, party, presence) shares the same store. It is
	// advertised to the client in chat|conf under the same host as the Battle server
	// (same machine); the Battle server mirrors square occupancy into it for area
	// chat.
	hub := mpd.NewHub(srv.Store)
	srv.MPD = hub
	battle.MPD = hub
	// Party co-members get online/offline pushes as a user's MPD socket comes and goes.
	hub.OnConnect = srv.NotifyOnline
	hub.OnDisconnect = srv.NotifyOffline
	if _, portStr, err := splitHostPort(*mpdAddr); err == nil {
		if port, err := strconv.Atoi(portStr); err == nil {
			srv.MPDHost = *battleHost
			srv.MPDPorts = []int32{int32(port)}
		}
	}

	go func() {
		if err := battle.ListenAndServe(*battleAddr); err != nil {
			log.Fatalf("battle server: %v", err)
		}
	}()
	go func() {
		if err := hub.ListenAndServe(*mpdAddr); err != nil {
			log.Fatalf("mpd server: %v", err)
		}
	}()

	log.Printf("ctrlserver listening on %s (POST /entry_point.php); battle on %s advertised as %s:%v; mpd on %s; «Штурм» match size %d",
		*addr, *battleAddr, srv.BattleHost, srv.BattlePorts, *mpdAddr, srv.DotaMatchSize)
	if err := http.ListenAndServe(*addr, srv.Handler()); err != nil {
		log.Fatal(err)
	}
}

// defaultLogPath puts ctrlserver.log next to the executable so it is easy to
// find regardless of the working directory the client launcher uses.
func defaultLogPath() string {
	return besideExe("ctrlserver.log")
}

// defaultAccountsPath persists accounts next to the executable.
func defaultAccountsPath() string {
	return besideExe("accounts.json")
}

func besideExe(name string) string {
	exe, err := os.Executable()
	if err != nil {
		return name
	}
	return filepath.Join(filepath.Dir(exe), name)
}

// splitHostPort tolerates a bare ":9339" (host empty) form.
func splitHostPort(addr string) (host, port string, err error) {
	for i := len(addr) - 1; i >= 0; i-- {
		if addr[i] == ':' {
			return addr[:i], addr[i+1:], nil
		}
	}
	return "", addr, nil
}
