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
	"crypto/rand"
	"encoding/hex"
	"flag"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"

	"tanatserver/internal/adminserver"
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
	dbPath := flag.String("db", defaultDBPath(), "SQLite database file for accounts/heroes (blank = in-memory only; a sibling accounts.json is imported once on a fresh DB)")
	dotaPlayers := flag.Int("dota-players", 1, "«Штурм» (DOTA) match size: players a match waits for before it starts (1-10; 1 = solo instant-match)")
	adminAddr := flag.String("admin-addr", ":8090", "listen address for the web admin panel (blank = disabled)")
	adminPass := flag.String("admin-pass", "", "admin panel password (blank = generate a random one and print it to the log)")
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
	if *dbPath != "" {
		srv.Store = session.NewPersistentStore(*dbPath)
	}
	// Apply any admin-panel gameplay settings persisted from a previous run, on top
	// of gamedata's authored defaults, before anything serves.
	adminserver.LoadSettings(srv.Store)
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
	// Штурм/Охота launches get a dedicated per-match Battle server (its own clock)
	// so the in-battle timer counts from match start, not process uptime. It shares
	// the session store (pending-battle handoff) and MPD hub with the main server,
	// and is advertised under the same host; the central square stays on the main
	// battle listener above.
	srv.MatchLauncher = battleserver.NewMatchHost(srv.Store, hub, *battleHost)
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

	// The web admin panel is a fourth listener in this process, sharing the session
	// store. It is disabled with a blank -admin-addr; with no -admin-pass a random
	// password is generated and printed so it is never left unprotected.
	if *adminAddr != "" {
		pass := *adminPass
		if pass == "" {
			pass = randomAdminPassword()
			log.Printf("admin panel: no -admin-pass set, generated password: %s", pass)
		}
		admin := adminserver.New(srv.Store, pass)
		go func() {
			log.Printf("admin panel on %s (open http://<this-host>%s/ in a browser)", *adminAddr, *adminAddr)
			if err := admin.ListenAndServe(*adminAddr); err != nil {
				log.Fatalf("admin server: %v", err)
			}
		}()
	}

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

// defaultDBPath keeps the SQLite database next to the executable.
func defaultDBPath() string {
	return besideExe("tanat.db")
}

// randomAdminPassword mints a short, URL-safe hex password used when the operator
// starts the admin panel without -admin-pass, so it is never left open.
func randomAdminPassword() string {
	b := make([]byte, 6)
	if _, err := rand.Read(b); err != nil {
		return "changeme"
	}
	return hex.EncodeToString(b)
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
