package battleserver

import (
	"bufio"
	"encoding/json"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// TestBotTelemetryDirGating pins the TANAT_BOT_TELEMETRY convention: unset/empty is
// off, "1" is the default relative directory, anything else is that literal path --
// matching HUNT_DEBUG/TANAT_DOTA_BOTS's established env-var-gate style.
func TestBotTelemetryDirGating(t *testing.T) {
	t.Setenv("TANAT_BOT_TELEMETRY", "")
	if got := botTelemetryDir(); got != "" {
		t.Errorf("botTelemetryDir() with unset env = %q, want empty (disabled)", got)
	}
	t.Setenv("TANAT_BOT_TELEMETRY", "1")
	if got := botTelemetryDir(); got != "telemetry" {
		t.Errorf("botTelemetryDir() with \"1\" = %q, want \"telemetry\"", got)
	}
	t.Setenv("TANAT_BOT_TELEMETRY", "C:/tmp/my-telemetry")
	if got := botTelemetryDir(); got != "C:/tmp/my-telemetry" {
		t.Errorf("botTelemetryDir() with a path = %q, want that literal path", got)
	}
}

// TestTelemetryRecorderNilSafe: every method on a nil *telemetryRecorder must be a
// harmless no-op, so every call site in the engine can invoke it unconditionally.
func TestTelemetryRecorderNilSafe(t *testing.T) {
	var r *telemetryRecorder
	r.record(telemetryMatchStart{telemetryEvent: newTelemetryEvent("match_start", 0), MapID: 1})
	r.close() // must not panic on a nil channel
}

// TestTelemetryBotTeleportLifecycleJSONL records both a cancelled and a successful
// bot channel, then parses the actual JSONL output. The refresh must carry a new
// target-FX UID, cancellation must preserve its reason, and success must expose a
// distinct arrival FX UID alongside the target kind/ID and destination.
func TestTelemetryBotTeleportLifecycleJSONL(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	dir := t.TempDir()
	rec := newTelemetryRecorder(dir, inst.dota.m.ID)
	if rec == nil {
		t.Fatal("newTelemetryRecorder returned nil against a writable temp dir")
	}
	inst.dota.telemetry = rec
	now := float64(s.battleTime())

	s.telemetryRecordBotTeleportLocked(bot, "bot_teleport_start", now,
		61001, "creep", 12, -3, 101, 102, 0, "")
	s.telemetryRecordBotTeleportLocked(bot, "bot_teleport_destination_refresh", now+2,
		61001, "creep", 13, -2, 101, 103, 0, "")
	s.telemetryRecordBotTeleportLocked(bot, "bot_teleport_cancel", now+2.1,
		61001, "creep", 13, -2, 101, 103, 0, "target_invalid")
	s.telemetryRecordBotTeleportLocked(bot, "bot_teleport_start", now+3,
		50022, "structure", 80, 40, 201, 202, 0, "")
	s.telemetryRecordBotTeleportLocked(bot, "bot_teleport_success", now+5,
		50022, "structure", 81, 41, 201, 202, 203, "")
	inst.dota.telemetry = nil
	rec.close()

	got := readTelemetryLines(t, dir)
	if len(got) != 5 {
		t.Fatalf("got %d teleport lifecycle lines, want 5", len(got))
	}
	wantTypes := []string{
		"bot_teleport_start",
		"bot_teleport_destination_refresh",
		"bot_teleport_cancel",
		"bot_teleport_start",
		"bot_teleport_success",
	}
	for i, want := range wantTypes {
		if got[i]["type"] != want {
			t.Errorf("line %d type = %v, want %q", i, got[i]["type"], want)
		}
		if got[i]["bot_id"] != float64(bot.objID) {
			t.Errorf("line %d bot_id = %v, want %d", i, got[i]["bot_id"], bot.objID)
		}
	}

	start := got[0]
	if start["target_id"] != float64(61001) || start["target_kind"] != "creep" ||
		start["destination_x"] != float64(12) || start["destination_y"] != float64(-3) ||
		start["marker_fx"] != float64(101) || start["target_fx"] != float64(102) || start["arrival_fx"] != float64(0) {
		t.Errorf("start = %+v, want creep target/destination and initial FX UIDs", start)
	}
	refresh := got[1]
	if refresh["target_id"] != float64(61001) || refresh["target_kind"] != "creep" ||
		refresh["destination_x"] != float64(13) || refresh["destination_y"] != float64(-2) ||
		refresh["marker_fx"] != float64(101) || refresh["target_fx"] != float64(103) ||
		refresh["target_fx"] == start["target_fx"] {
		t.Errorf("refresh = %+v, want changed target FX UID and destination", refresh)
	}
	cancel := got[2]
	if cancel["target_id"] != float64(61001) || cancel["target_kind"] != "creep" ||
		cancel["cancel_reason"] != "target_invalid" {
		t.Errorf("cancel = %+v, want creep target and reason=target_invalid", cancel)
	}
	success := got[4]
	if success["target_id"] != float64(50022) || success["target_kind"] != "structure" ||
		success["destination_x"] != float64(81) || success["destination_y"] != float64(41) ||
		success["marker_fx"] != float64(201) || success["target_fx"] != float64(202) ||
		success["arrival_fx"] != float64(203) || success["arrival_fx"] == success["target_fx"] {
		t.Errorf("success = %+v, want structure target/destination and distinct arrival FX UID", success)
	}

	// The production helper must also be safe when telemetry is disabled, not only
	// when callers hold a nil recorder directly.
	s.telemetryRecordBotTeleportLocked(bot, "bot_teleport_cancel", now+6,
		50022, "structure", 81, 41, 201, 202, 0, "telemetry_disabled")
}

func TestTelemetryBotPurchaseJSONL(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()

	const botID = botIDBase + 1
	bot.objID = botID
	bot.selfPlayerID = botID
	s.Store.CreateBotHero(botID, "purchase-test-bot")
	bot.huntState.level = 4 // the first catalog root selected below is tier 2
	const startingMoney int32 = 1000000
	if !s.Store.SetHeroMoney(botID, startingMoney, 0) {
		t.Fatal("setup: failed to create bot wallet")
	}

	var item gamedata.AvatarItem
	for _, candidate := range gamedata.AvatarItems() {
		if len(candidate.Parents) == 0 {
			item = candidate
			break
		}
	}
	if item.ArticleID == 0 {
		t.Fatal("setup: avatar item catalog has no root item")
	}

	dir := t.TempDir()
	rec := newTelemetryRecorder(dir, inst.dota.m.ID)
	if rec == nil {
		t.Fatal("newTelemetryRecorder returned nil")
	}
	inst.dota.telemetry = rec

	now := float64(s.battleTime())
	bot.lock()
	ok := s.buyItemLocked(bot, item.ArticleID)
	bot.unlock()
	if !ok {
		t.Fatalf("buyItemLocked(%d) failed for affordable root item", item.ArticleID)
	}
	inst.dota.telemetry = nil
	rec.close()

	got := readTelemetryLines(t, dir)
	if len(got) != 1 {
		t.Fatalf("got %d telemetry lines, want one purchase line", len(got))
	}
	line := got[0]
	if line["type"] != "bot_item_purchase" || line["bot_id"] != float64(botID) ||
		line["article_id"] != float64(item.ArticleID) || line["tree_id"] != float64(item.TreeID) ||
		line["stage"] != float64(item.Stage) || line["price"] != float64(item.Price) ||
		line["money_before"] != float64(startingMoney) || line["money_after"] != float64(startingMoney-item.Price) {
		t.Fatalf("purchase telemetry = %+v, want item=%d price=%d money=%d->%d", line, item.ArticleID, item.Price, startingMoney, startingMoney-item.Price)
	}
	if line["t"] != float64(now-inst.dota.startedAt) {
		t.Errorf("purchase telemetry t = %v, want match time near %v", line["t"], now-inst.dota.startedAt)
	}
}

func TestTelemetryBotStructureEscapeJSONLAndNilSafe(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	dir := t.TempDir()
	rec := newTelemetryRecorder(dir, inst.dota.m.ID)
	if rec == nil {
		t.Fatal("newTelemetryRecorder returned nil")
	}
	inst.dota.telemetry = rec
	now := float64(s.battleTime())
	threat := structOfSide(inst, gamedata.DotaCreepTower, dotaTeamElf)
	if threat == nil {
		t.Fatal("setup: no enemy tower")
	}
	bot.huntState.hp = bot.huntState.maxHPLocked(now) * 0.42
	s.telemetryRecordBotStructureEscapeLocked(bot, threat, 8, 3, "retreat_detour", now)
	inst.dota.telemetry = nil
	// Nil recorder and nil threat must both remain harmless.
	s.telemetryRecordBotStructureEscapeLocked(bot, nil, 0, 0, "nil-threat", now+1)
	rec.close()

	got := readTelemetryLines(t, dir)
	if len(got) != 1 {
		t.Fatalf("got %d telemetry lines, want one structure escape line", len(got))
	}
	line := got[0]
	if line["type"] != "bot_structure_escape" || line["bot_id"] != float64(bot.objID) || line["threat_id"] != float64(threat.id) {
		t.Fatalf("structure escape identity = %+v", line)
	}
	if line["destination_x"] != float64(8) || line["destination_y"] != float64(3) || line["reason"] != "retreat_detour" {
		t.Errorf("structure escape destination/reason = %+v", line)
	}
	if line["hp_fraction"] != 0.42 {
		t.Errorf("structure escape hp_fraction = %v, want 0.42", line["hp_fraction"])
	}
}

func TestTelemetryBotRetreatUtilityJSONFieldsAndNilSafe(t *testing.T) {
	s, bot, inst, cleanup := newDotaConn(t, "Avtr_HK_Astarot")
	defer cleanup()
	dir := t.TempDir()
	rec := newTelemetryRecorder(dir, inst.dota.m.ID)
	if rec == nil {
		t.Fatal("newTelemetryRecorder returned nil")
	}
	inst.dota.telemetry = rec
	now := float64(s.battleTime())
	s.telemetryRecordBotRetreatUtilityLocked(bot, 2, 0, "retreat_home_dash", "point_dash", now, 4, -6, true)
	s.telemetryRecordBotRetreatUtilityLocked(bot, 3, 7001, "retreat_pursuer_control", "nearby_control", now+1, 0, 0, false)
	var nilConn *conn
	s.telemetryRecordBotRetreatUtilityLocked(nilConn, 1, 0, "nil", "nil", now, 0, 0, false)
	inst.dota.telemetry = nil
	rec.close()

	got := readTelemetryLines(t, dir)
	if len(got) != 2 {
		t.Fatalf("got %d retreat utility lines, want 2", len(got))
	}
	first := got[0]
	if first["type"] != "bot_retreat_utility" || first["bot_id"] != float64(bot.objID) ||
		first["slot"] != float64(2) || first["target_id"] != float64(0) ||
		first["reason"] != "retreat_home_dash" || first["category"] != "point_dash" {
		t.Fatalf("retreat utility identity = %+v", first)
	}
	destination, ok := first["destination"].(map[string]any)
	if !ok || destination["x"] != float64(4) || destination["y"] != float64(-6) {
		t.Fatalf("retreat utility destination = %#v, want x=4 y=-6", first["destination"])
	}
	second := got[1]
	if second["target_id"] != float64(7001) || second["reason"] != "retreat_pursuer_control" || second["category"] != "nearby_control" {
		t.Fatalf("pursuer-control telemetry = %+v", second)
	}
	if _, ok := second["destination"]; ok {
		t.Fatalf("nil destination was not omitted: %+v", second)
	}
}

// TestTelemetryRecorderWritesJSONL drives the real writer goroutine end to end:
// created against a temp dir, fed a couple of events, closed, then the file is read
// back and each line must parse as the exact JSON shape recorded.
func TestTelemetryRecorderWritesJSONL(t *testing.T) {
	dir := t.TempDir()
	rec := newTelemetryRecorder(dir, 101)
	if rec == nil {
		t.Fatal("newTelemetryRecorder returned nil against a writable temp dir")
	}
	rec.record(telemetryMatchStart{telemetryEvent: newTelemetryEvent("match_start", 0), MapID: 101})
	rec.record(telemetrySnapshot{
		telemetryEvent: newTelemetryEvent("snapshot", 1.2),
		ID:             1000, Name: "Bot1", Avatar: "Avtr_Tank_Rognar", IsBot: true, Team: 1,
		X: 5, Y: -3, HPFrac: 0.8, ManaFrac: 0.5, Level: 2, Phase: "lane",
	})
	rec.close()

	matches, err := filepath.Glob(filepath.Join(dir, "101-*.jsonl"))
	if err != nil || len(matches) != 1 {
		t.Fatalf("expected exactly one 101-*.jsonl file in %s, got %v (err=%v)", dir, matches, err)
	}

	f, err := os.Open(matches[0])
	if err != nil {
		t.Fatalf("open %s: %v", matches[0], err)
	}
	defer f.Close()

	var lines []map[string]any
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var m map[string]any
		if err := json.Unmarshal(sc.Bytes(), &m); err != nil {
			t.Fatalf("line %q did not parse as JSON: %v", sc.Text(), err)
		}
		lines = append(lines, m)
	}
	if len(lines) != 2 {
		t.Fatalf("got %d JSONL lines, want 2", len(lines))
	}
	if lines[0]["type"] != "match_start" || lines[0]["map_id"] != float64(101) {
		t.Errorf("line 0 = %+v, want type=match_start map_id=101", lines[0])
	}
	if lines[1]["type"] != "snapshot" || lines[1]["is_bot"] != true || lines[1]["avatar"] != "Avtr_Tank_Rognar" {
		t.Errorf("line 1 = %+v, want type=snapshot is_bot=true avatar=Avtr_Tank_Rognar", lines[1])
	}
	if lines[1]["hp_frac"] != 0.8 {
		t.Errorf("line 1 hp_frac = %v, want 0.8", lines[1]["hp_frac"])
	}
}

// TestTelemetrySnapshotLocked drives the real snapshot recorder against a live «Штурм»
// bot member and checks the recorded fields, including the bot-only phase/lane state.
func TestTelemetrySnapshotLocked(t *testing.T) {
	s := New(session.NewStore())
	dm := gamedata.DotaMaps()[0]
	inst := newDotaInstance(s, dm.ID, dm.ID)
	now := float64(s.battleTime())

	bot := dotaPlayerConn(t, s, inst, 900001, dotaTeamHuman, 5, -3)
	bot.huntState.hp, bot.huntState.mana = 80, 50
	inst.bots[bot.objID] = &botBrain{c: bot, phase: botPhaseRoam, lane: 1, engageTarget: 42}

	dir := t.TempDir()
	rec := newTelemetryRecorder(dir, dm.ID)

	bot.mvMu.Lock()
	s.telemetrySnapshotLocked(rec, bot, 12.5, now)
	bot.mvMu.Unlock()

	rec.close()
	got := readTelemetryLines(t, dir)
	if len(got) != 1 {
		t.Fatalf("got %d recorded lines, want 1", len(got))
	}
	snap := got[0]
	if snap["type"] != "snapshot" || snap["id"] != float64(bot.objID) {
		t.Fatalf("snapshot = %+v, want type=snapshot id=%d", snap, bot.objID)
	}
	if snap["is_bot"] != true {
		t.Error("snapshot is_bot = false, want true (objID is in the bot id range)")
	}
	if snap["phase"] != "roam" || snap["lane"] != float64(1) || snap["engage_target"] != float64(42) {
		t.Errorf("snapshot bot fields = %+v, want phase=roam lane=1 engage_target=42", snap)
	}
	if snap["x"] != float64(5) || snap["y"] != float64(-3) {
		t.Errorf("snapshot position = (%v,%v), want (5,-3)", snap["x"], snap["y"])
	}
}

func TestTelemetrySnapshotRetreatModes(t *testing.T) {
	s := New(session.NewStore())
	dm := gamedata.DotaMaps()[0]
	inst := newDotaInstance(s, dm.ID, dm.ID)

	disengaging := dotaPlayerConn(t, s, inst, 900001, dotaTeamHuman, 5, -3)
	recovering := dotaPlayerConn(t, s, inst, 900002, dotaTeamHuman, 6, -3)
	cleared := dotaPlayerConn(t, s, inst, 900003, dotaTeamHuman, 7, -3)
	inst.bots[disengaging.objID] = &botBrain{c: disengaging, retreating: true, retreatMode: botRetreatModeDisengage}
	inst.bots[recovering.objID] = &botBrain{c: recovering, retreating: true, retreatMode: botRetreatModeRecovery}
	// An inactive brain may retain a stale mode value internally, but snapshots must
	// omit retreat_mode unless retreating is currently active.
	inst.bots[cleared.objID] = &botBrain{c: cleared, retreatMode: botRetreatModeDisengage}

	dir := t.TempDir()
	rec := newTelemetryRecorder(dir, dm.ID)
	now := float64(s.battleTime())
	for _, bot := range []*conn{disengaging, recovering, cleared} {
		bot.mvMu.Lock()
		s.telemetrySnapshotLocked(rec, bot, 12.5, now)
		bot.mvMu.Unlock()
	}
	rec.close()

	got := readTelemetryLines(t, dir)
	if len(got) != 3 {
		t.Fatalf("got %d snapshots, want 3", len(got))
	}
	wantModes := map[float64]string{
		float64(disengaging.objID): "disengage",
		float64(recovering.objID):  "recovery",
	}
	for _, snap := range got {
		id, ok := snap["id"].(float64)
		if !ok {
			t.Fatalf("snapshot has no numeric id: %+v", snap)
		}
		if want, active := wantModes[id]; active {
			if snap["retreating"] != true || snap["retreat_mode"] != want {
				t.Errorf("active retreat snapshot for id %.0f = %+v, want retreating=true mode=%q", id, snap, want)
			}
			continue
		}
		if _, present := snap["retreat_mode"]; present {
			t.Errorf("inactive retreat_mode was serialized for id %.0f: %+v", id, snap)
		}
	}
}

// TestTelemetryRecordDamageAndDeath: a fatal hit must record BOTH a damage line and a
// death line, with attacker/victim is_bot correctly resolved for a hero-vs-hero kill.
func TestTelemetryRecordDamageAndDeath(t *testing.T) {
	s := New(session.NewStore())
	dm := gamedata.DotaMaps()[0]
	inst := newDotaInstance(s, dm.ID, dm.ID)

	victim := dotaPlayerConn(t, s, inst, 1000, dotaTeamHuman, 0, 0)
	killer := dotaPlayerConn(t, s, inst, 900001, dotaTeamElf, 1, 0) // bot id range

	dir := t.TempDir()
	rec := newTelemetryRecorder(dir, dm.ID)

	victim.mvMu.Lock()
	s.telemetryRecordDamageLocked(rec, victim, killer.objID, killer, 500, true, 30.0, float64(s.battleTime()))
	victim.mvMu.Unlock()
	rec.close()

	got := readTelemetryLines(t, dir)
	if len(got) != 2 {
		t.Fatalf("got %d lines, want 2 (damage + death)", len(got))
	}
	dmg := got[0]
	if dmg["type"] != "damage" || dmg["victim_id"] != float64(victim.objID) || dmg["fatal"] != true {
		t.Errorf("damage line = %+v, want type=damage victim_id=%d fatal=true", dmg, victim.objID)
	}
	if dmg["attacker_is_hero"] != true || dmg["attacker_is_bot"] != true {
		t.Errorf("damage line attacker flags = %+v, want attacker_is_hero=true attacker_is_bot=true", dmg)
	}
	death := got[1]
	if death["type"] != "death" || death["victim_id"] != float64(victim.objID) || death["killer_id"] != float64(killer.objID) {
		t.Errorf("death line = %+v, want type=death victim_id=%d killer_id=%d", death, victim.objID, killer.objID)
	}
	if death["killer_is_bot"] != true {
		t.Errorf("death line killer_is_bot = %v, want true", death["killer_is_bot"])
	}
}

// TestNewDotaInstanceStartsTelemetryWhenEnabled: newDotaInstance must create a live
// recorder (and write match_start) exactly when TANAT_BOT_TELEMETRY is set, and leave
// it nil (no file, no goroutine) when it is not -- the default, zero-overhead case.
func TestNewDotaInstanceStartsTelemetryWhenEnabled(t *testing.T) {
	dm := gamedata.DotaMaps()[0]

	t.Setenv("TANAT_BOT_TELEMETRY", "")
	s := New(session.NewStore())
	instOff := newDotaInstance(s, dm.ID, dm.ID)
	if instOff.dota.telemetry != nil {
		t.Fatal("telemetry recorder created despite TANAT_BOT_TELEMETRY being unset")
	}

	dir := t.TempDir()
	t.Setenv("TANAT_BOT_TELEMETRY", dir)
	instOn := newDotaInstance(s, dm.ID+1000, dm.ID)
	if instOn.dota.telemetry == nil {
		t.Fatal("no telemetry recorder created despite TANAT_BOT_TELEMETRY being set")
	}
	instOn.dota.telemetry.close()

	got := readTelemetryLines(t, dir)
	if len(got) != 1 || got[0]["type"] != "match_start" || got[0]["map_id"] != float64(dm.ID) {
		t.Errorf("expected exactly one match_start line, got %+v", got)
	}
}

// TestDotaEndLockedWritesMatchEndAndCloses: dotaEndLocked must record a match_end line
// carrying the winner and every member's final K/D/A/level, then close the recorder.
func TestDotaEndLockedWritesMatchEndAndCloses(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("TANAT_BOT_TELEMETRY", dir)

	s := New(session.NewStore())
	dm := gamedata.DotaMaps()[0]
	inst := newDotaInstance(s, dm.ID, dm.ID)
	winner := dotaPlayerConn(t, s, inst, 1000, dotaTeamHuman, 0, 0)
	winner.huntState.frags, winner.huntState.deaths, winner.huntState.assists = 3, 1, 2
	loser := dotaPlayerConn(t, s, inst, 900001, dotaTeamElf, 1, 0)

	winner.lock()
	now := float64(s.battleTime())
	s.dotaEndLocked(winner, dotaTeamHuman, now)
	winner.unlock()

	got := readTelemetryLines(t, dir)
	var end map[string]any
	for _, l := range got {
		if l["type"] == "match_end" {
			end = l
		}
	}
	if end == nil {
		t.Fatalf("no match_end line among %d recorded lines", len(got))
	}
	if end["winner"] != float64(dotaTeamHuman) {
		t.Errorf("match_end winner = %v, want %d", end["winner"], dotaTeamHuman)
	}
	final, ok := end["final"].([]any)
	if !ok || len(final) != 2 {
		t.Fatalf("match_end final = %v, want 2 entries", end["final"])
	}
	_ = loser
}

// TestBotTelemetryRecordsRealMatch uses a real TCP CONNECT/READY launch and
// asynchronous JSONL writer, but advances the Assault clock explicitly. It
// verifies the live telemetry path without waiting for real server ticks.
func TestBotTelemetryRecordsRealMatch(t *testing.T) {
	if testing.Short() {
		t.Skip("real-clock smoke test skipped in -short")
	}
	dir := t.TempDir()
	t.Setenv("TANAT_BOT_TELEMETRY", dir)
	t.Setenv("TANAT_DOTA_BOTS", "6")

	store := session.NewStore()
	s := New(store)
	clock := newManualBattleClock()
	s.clock = clock
	instanceStarted := make(chan *huntInstance, 1)
	s.instanceTickerStarter = func(*huntInstance) {}
	s.instanceReadyHook = func(inst *huntInstance) { instanceStarted <- inst }
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()
	go s.Serve(ln)

	const userID int32 = 1
	dm := gamedata.DotaMaps()[0]
	av, _ := gamedata.AvatarByID(botRosterAvatarIDs[0])
	store.SetPendingBattle(userID, session.PendingBattle{
		MapID: dm.ID, AvatarID: av.ID, Passwd: "pw", Scene: dm.Scene, Room: dm.ID + 778,
	})

	cl, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer cl.Close()
	_ = cl.SetDeadline(time.Now().Add(15 * time.Second))
	r := battleproto.NewReader(cl)

	if err := battleproto.Write(cl, battleproto.Packet{
		Cmd: battleproto.CmdConnect, RequestID: 1, Status: true,
		Args: amf.NewArray().Set("clientId", userID).Set("pass", "pw"),
	}); err != nil {
		t.Fatalf("send CONNECT: %v", err)
	}
	if _, err := r.Read(); err != nil {
		t.Fatalf("read CONNECT reply: %v", err)
	}
	if err := battleproto.Write(cl, battleproto.Packet{
		Cmd: battleproto.CmdReady, RequestID: 2, Status: true, Args: amf.NewArray(),
	}); err != nil {
		t.Fatalf("send READY: %v", err)
	}
	drainDone := make(chan struct{})
	go func() {
		defer close(drainDone)
		for {
			if _, err := r.Read(); err != nil {
				return
			}
		}
	}()

	var inst *huntInstance
	select {
	case inst = <-instanceStarted:
	case <-time.After(3 * time.Second):
		t.Fatal("no Assault instance appeared after CONNECT/READY")
	}
	driver := &manualDotaBots{server: s, inst: inst, clock: clock}
	defer driver.close()
	for tick := 0; tick < int((3*time.Second)/AssaultTick); tick++ {
		driver.step()
	}

	cl.Close()
	<-drainDone

	// The match itself never ends in this short window, so its telemetry recorder's
	// writer goroutine is still holding the file open -- close it explicitly (matching
	// what dotaEndLocked does for real) before reading the file back, or t.TempDir()'s
	// cleanup races the still-open handle on Windows.
	s.mu.Lock()
	for _, i := range s.insts {
		if i.dota != nil && i.dota.telemetry != nil {
			i.mu.Lock()
			i.dota.telemetry.close()
			i.dota.telemetry = nil
			i.mu.Unlock()
		}
	}
	s.mu.Unlock()

	got := readTelemetryLines(t, dir)
	if len(got) == 0 {
		t.Fatal("no telemetry lines recorded during a 3s simulated match window")
	}
	var starts, snaps int
	positions := map[int32]bool{}
	for _, l := range got {
		switch l["type"] {
		case "match_start":
			starts++
		case "snapshot":
			snaps++
			if id, ok := l["id"].(float64); ok {
				positions[int32(id)] = true
			}
		}
	}
	if starts != 1 {
		t.Errorf("got %d match_start lines, want exactly 1", starts)
	}
	if snaps < 10 {
		t.Errorf("got only %d snapshots in 3 simulated seconds, want a healthy stream", snaps)
	}
	if len(positions) < 2 {
		t.Errorf("snapshots only ever covered %d distinct hero ids, want the player + several bots", len(positions))
	}
}

// readTelemetryLines reads every *.jsonl file in dir (glob, since the filename embeds
// a timestamp/sequence) and parses every line as a generic JSON object, in file-then-
// line order.
func readTelemetryLines(t *testing.T, dir string) []map[string]any {
	t.Helper()
	matches, err := filepath.Glob(filepath.Join(dir, "*.jsonl"))
	if err != nil {
		t.Fatalf("glob %s: %v", dir, err)
	}
	var out []map[string]any
	for _, path := range matches {
		f, err := os.Open(path)
		if err != nil {
			t.Fatalf("open %s: %v", path, err)
		}
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			var m map[string]any
			if err := json.Unmarshal(sc.Bytes(), &m); err != nil {
				f.Close()
				t.Fatalf("%s: line %q did not parse as JSON: %v", path, sc.Text(), err)
			}
			out = append(out, m)
		}
		f.Close()
	}
	return out
}
