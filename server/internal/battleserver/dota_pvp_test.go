package battleserver

import (
	"net"
	"testing"
	"time"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// «Штурм» true PvP: two sides of REAL players (Human/«Собор» = team 1, Elf/«Изгнанники» =
// team 2), not the old co-op-vs-bots. This file pins the absolute-team model, the
// per-player side assignment, cross-side hostility, and that a hero can strike the enemy
// side's structures/heroes from EITHER base (the old hostile() was wired to team 1 and
// read backwards for an Elf-side player).

// dotaPlayerConn wires a second «Штурм» player into an existing instance on the given
// team at (x,y). Mirrors arenaConn but for a DOTA world. Pushes drain into a pipe reader.
func dotaPlayerConn(t *testing.T, s *Server, inst *huntInstance, objID, team int32, x, y float32) *conn {
	t.Helper()
	srv, cli := net.Pipe()
	t.Cleanup(func() { srv.Close(); cli.Close() })
	r := battleproto.NewReader(cli)
	go func() {
		for {
			if _, err := r.Read(); err != nil {
				return
			}
		}
	}()
	av := avatarByPrefab(t, "Avtr_Tank_Velial")
	c := &conn{Conn: srv}
	c.objID = objID
	c.selfPlayerID = objID
	c.x, c.y, c.snapT = x, y, s.battleTime()
	hs := &huntState{
		av: av, kit: gamedata.SkillsFor(av),
		mobs: inst.mobs, summons: map[int32]*summonState{},
		summonProtos: map[string]int32{},
		hp:           av.Health, mana: av.Mana, team: team,
		worldReady: true,
	}
	hs.tr.add(objID)
	hs.inst = inst
	c.huntState = hs
	c.inst = inst
	c.nav = inst.nav
	c.lk = &inst.mu
	inst.members[objID] = c
	t.Cleanup(func() { c.lock(); hs.closed = true; c.unlock() })
	return c
}

// structOfSide returns a live structure of the given role on the given side's team.
func structOfSide(inst *huntInstance, role gamedata.DotaRole, team int32) *mobState {
	for _, m := range inst.mobs {
		if m.structure && m.dotaRole == role && m.team == team && !m.dead {
			return m
		}
	}
	return nil
}

// TestDotaAbsoluteTeams: every seeded structure carries an ABSOLUTE side team -- Human
// on team 1, Elf on team 2 -- and nothing is on the old relative "enemy" team (-1). Two
// distinct positive teams are what makes the two bases hostile on the client without a
// per-viewer rewrite.
func TestDotaAbsoluteTeams(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	_ = s
	var human, elf int
	for _, m := range inst.mobs {
		switch m.team {
		case dotaTeamHuman:
			human++
		case dotaTeamElf:
			elf++
		default:
			t.Fatalf("structure %d on team %d, want an absolute side team (1 Human / 2 Elf)", m.id, m.team)
		}
	}
	if human == 0 || elf == 0 {
		t.Fatalf("expected structures on both sides, got human=%d elf=%d", human, elf)
	}
	if altarOf(inst, dotaTeamHuman) == nil || altarOf(inst, dotaTeamElf) == nil {
		t.Fatal("both sides must have an altar")
	}
}

// TestDotaSideModel pins the pure side<->team mapping and the joiner alternation.
func TestDotaSideModel(t *testing.T) {
	if teamForSide(gamedata.DotaSideHuman) != dotaTeamHuman ||
		teamForSide(gamedata.DotaSideElf) != dotaTeamElf {
		t.Fatal("teamForSide must map Human->1, Elf->2")
	}
	if sideForTeam(dotaTeamHuman) != gamedata.DotaSideHuman ||
		sideForTeam(dotaTeamElf) != gamedata.DotaSideElf {
		t.Fatal("sideForTeam must invert teamForSide")
	}
	d := &dotaState{nextSide: gamedata.DotaSideHuman}
	got := []gamedata.DotaSide{d.assignSide(), d.assignSide(), d.assignSide(), d.assignSide()}
	want := []gamedata.DotaSide{
		gamedata.DotaSideHuman, gamedata.DotaSideElf,
		gamedata.DotaSideHuman, gamedata.DotaSideElf,
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("assignSide join %d = %v, want %v (bases must fill evenly)", i, got[i], want[i])
		}
	}
}

// TestDotaPlayersHostileAcrossSides: a Human hero and an Elf hero are enemies (legal PvP
// targets) and resolve as members of the same PvP world; two on one side are not.
func TestDotaPlayersHostileAcrossSides(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial") // team unset -> Human (team 1)
	defer cleanup()
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, human.x+3, human.y)
	humanAlly := dotaPlayerConn(t, s, inst, 1002, dotaTeamHuman, human.x-3, human.y)

	if human.playerTeam() != dotaTeamHuman || elf.playerTeam() != dotaTeamElf {
		t.Fatalf("player teams: human=%d elf=%d, want 1 and 2", human.playerTeam(), elf.playerTeam())
	}
	if !arenaEnemies(human, elf) {
		t.Error("Human and Elf heroes are not enemies -- PvP is off")
	}
	if arenaEnemies(human, humanAlly) {
		t.Error("two Human heroes are enemies -- friendly fire is on")
	}
	// The PvP attack engine must resolve the enemy hero as a live member of this DOTA world
	// (pvpMember returned nil for anything but «Арена» before, so a click on an enemy hero in
	// «Штурм» was silently dropped).
	if human.pvpMember(elf.objID) != elf {
		t.Error("pvpMember did not resolve the enemy hero in a «Штурм» world")
	}
}

// TestDotaElfPlayerCanAttackHumanStructures is the core hostile()-fix: an Elf-side hero
// (team 2) may attack Human structures (team 1) but NOT its own side's -- the mirror of a
// Human hero attacking Elf structures. The old hostile() was hard-wired to team 1, so an
// Elf player could attack only their OWN buildings and none of the enemy's.
func TestDotaElfPlayerCanAttackHumanStructures(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	humanGun := structOfSide(inst, gamedata.DotaGun, dotaTeamHuman)
	elfGun := structOfSide(inst, gamedata.DotaGun, dotaTeamElf)
	if humanGun == nil || elfGun == nil {
		t.Fatal("need a gun on each side")
	}
	// An Elf-side hero standing among the structures.
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, humanGun.x, humanGun.y)
	act := attackProtoID(elf.huntState.av)

	elf.lock()
	defer elf.unlock()
	// Enemy (Human) gun: the order is accepted -> the hero locks onto it.
	s.doActionLocked(elf, -1, act, humanGun.id, 0, 0, false)
	if elf.huntState.attackTarget != humanGun.id {
		t.Fatalf("Elf hero could not attack the enemy Human gun (attackTarget=%d, want %d)",
			elf.huntState.attackTarget, humanGun.id)
	}
	elf.huntState.attackTarget = 0
	s.stopAttackLocked(elf, false)
	// Own (Elf) gun: friendly fire -> the order is refused, no target set.
	s.doActionLocked(elf, -1, act, elfGun.id, 0, 0, false)
	if elf.huntState.attackTarget != 0 {
		t.Fatalf("Elf hero attacked its OWN side's gun (attackTarget=%d) -- friendly fire is on",
			elf.huntState.attackTarget)
	}
	elf.huntState.closed = true // neutralise the armed swing timer before cleanup
}

// TestDotaEnemyCreepHuntsEnemyHero: a creep targets a hero of the OPPOSING side and never
// one of its own -- absolute-team, so an Elf creep hunts the Human hero while ignoring the
// Elf hero, and vice-versa.
func TestDotaEnemyCreepHuntsEnemyHero(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial") // Human hero (team 1)
	defer cleanup()
	// Both heroes stand together in open ground, away from base structures.
	human.x, human.y = 0, 0
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, 1, 0)

	human.lock()
	defer human.unlock()

	// An Elf-side creep on top of them must pick the HUMAN hero (its enemy), not the Elf one.
	elfCreep := &mobState{
		id: 60101, mobIdx: inst.dota.m.ElfCreepMelee, mob: gamedata.Mobs()[inst.dota.m.ElfCreepMelee],
		x: 0, y: 0, hp: 500, maxHP: 500, team: dotaTeamElf, lastSync: float64(s.battleTime()),
	}
	inst.mobs[elfCreep.id] = elfCreep
	tgt := s.dotaAcquireTargetLocked(human, elfCreep, 30, float64(s.battleTime()))
	if tgt == nil || tgt.player == nil {
		t.Fatalf("Elf creep acquired no enemy hero (got %+v)", tgt)
	}
	if tgt.player != human {
		t.Fatalf("Elf creep targeted the Elf hero (same side); it must hunt the Human hero")
	}
	delete(inst.mobs, elfCreep.id)

	// Symmetric: a Human-side creep must pick the ELF hero.
	humanCreep := &mobState{
		id: 60102, mobIdx: inst.dota.m.HumanCreepMelee, mob: gamedata.Mobs()[inst.dota.m.HumanCreepMelee],
		x: 1, y: 0, hp: 500, maxHP: 500, team: dotaTeamHuman, lastSync: float64(s.battleTime()),
	}
	inst.mobs[humanCreep.id] = humanCreep
	tgt2 := s.dotaAcquireTargetLocked(human, humanCreep, 30, float64(s.battleTime()))
	if tgt2 == nil || tgt2.player != elf {
		t.Fatalf("Human creep must hunt the Elf hero, got %+v", tgt2)
	}
}

// TestDotaWinIsAbsolute: whichever base altar falls, the OTHER side wins, and the winner
// team is the same no matter which member drives the tick (an Elf player sees the same
// BATTLE_END winner as a Human one -- no player-relative flip).
func TestDotaWinIsAbsolute(t *testing.T) {
	// Elf altar falls -> Human (team 1) wins, driven by an ELF member.
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, 0, 0)
	now := float64(s.battleTime())
	elf.lock()
	altarOf(inst, dotaTeamElf).dead = true
	s.dotaTickLocked(elf, now)
	elf.unlock()
	if !inst.dota.ended || inst.dota.winner != dotaTeamHuman {
		t.Fatalf("Elf altar fell: ended=%v winner=%d, want ended=true winner=%d (Human)",
			inst.dota.ended, inst.dota.winner, dotaTeamHuman)
	}

	// Human altar falls -> Elf (team 2) wins.
	s2, human2, inst2, cleanup2 := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup2()
	now2 := float64(s2.battleTime())
	human2.lock()
	altarOf(inst2, dotaTeamHuman).dead = true
	s2.dotaTickLocked(human2, now2)
	human2.unlock()
	if !inst2.dota.ended || inst2.dota.winner != dotaTeamElf {
		t.Fatalf("Human altar fell: winner=%d, want %d (Elf)", inst2.dota.winner, dotaTeamElf)
	}
}

// TestDotaPlayerKillDoesNotEndMatch: killing an enemy HERO in «Штурм» respawns them but
// never ends the match (only an altar does) -- the «Арена» frag/win path must stay a no-op
// here, or a single duel would end a lane-push game.
func TestDotaPlayerKillDoesNotEndMatch(t *testing.T) {
	s, human, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, human.x+1, human.y)
	now := float64(s.battleTime())

	human.lock()
	defer human.unlock()
	elf.huntState.hp = 60
	for i := 0; i < 50 && elf.huntState.deadUntil == 0; i++ {
		s.hitPlayerFromLocked(elf, human.objID, 20, now, nil, human)
	}
	if elf.huntState.deadUntil == 0 {
		t.Fatal("the enemy hero never died despite repeated lethal blows")
	}
	if inst.dota.ended {
		t.Fatal("a hero kill ended the «Штурм» match -- only an altar may end it")
	}
}

// TestDotaElfSummonFightsForElfSide pins the summon-team fix: a summon takes its OWNER's
// absolute side, so an Elf-side (team 2) player's non-pet summon renders as team 2 and
// attacks the enemy Human side -- never its own Elf allies. Before the fix summons were
// hardcoded to team 1, so an Elf summon rendered as a Human unit and beat up its own side.
func TestDotaElfSummonFightsForElfSide(t *testing.T) {
	s, _, inst, cleanup := newDotaConn(t, "Avtr_Tank_Velial")
	defer cleanup()
	elf := dotaPlayerConn(t, s, inst, 1001, dotaTeamElf, 0, 0)
	now := float64(s.battleTime())

	elf.lock()
	defer elf.unlock()
	// Register a summon prototype and cast one non-pet summon at the caster.
	elf.huntState.summonProtos["Mob_ZombieCrawl_01"] = 800
	op := gamedata.Op{
		Kind: gamedata.OpSummon, Unit: "Mob_ZombieCrawl_01",
		Count: gamedata.PerLevel{1}, HP: gamedata.PerLevel{300},
		Dmg: gamedata.PerLevel{40}, Lifetime: gamedata.PerLevel{60},
	}
	s.summonLocked(elf, op, opCtx{slot: 2, level: 1}, now)
	if len(elf.huntState.summons) != 1 {
		t.Fatalf("expected 1 summon, got %d", len(elf.huntState.summons))
	}
	var sm *summonState
	for _, x := range elf.huntState.summons {
		sm = x
	}
	if sm.team != dotaTeamElf {
		t.Fatalf("Elf player's summon has team %d, want %d (its owner's side)", sm.team, dotaTeamElf)
	}
	// Sit the summon between an enemy Human creep and a friendly Elf creep, both adjacent.
	sm.x, sm.y = 0, 0
	sm.nextSwing = now
	humanCreep := &mobState{
		id: 60201, mobIdx: inst.dota.m.HumanCreepMelee, mob: gamedata.Mobs()[inst.dota.m.HumanCreepMelee],
		x: 0.5, y: 0, hp: 800, maxHP: 800, team: dotaTeamHuman, lastSync: now, active: true, shown: true,
	}
	elfCreep := &mobState{
		id: 60202, mobIdx: inst.dota.m.ElfCreepMelee, mob: gamedata.Mobs()[inst.dota.m.ElfCreepMelee],
		x: -0.5, y: 0, hp: 800, maxHP: 800, team: dotaTeamElf, lastSync: now, active: true, shown: true,
	}
	inst.mobs[humanCreep.id] = humanCreep
	inst.mobs[elfCreep.id] = elfCreep

	for bt := now; bt <= now+3.0; bt += 0.2 {
		s.tickSummonsLocked(elf, bt)
	}
	if humanCreep.hp >= 800 {
		t.Errorf("Elf summon never hit the enemy Human creep (hp %g) -- it should fight the enemy side", humanCreep.hp)
	}
	if elfCreep.hp != 800 {
		t.Errorf("Elf summon damaged its OWN Elf creep (hp %g) -- friendly fire; team gate inverted", elfCreep.hp)
	}
}

// TestDotaElfLaunchAssignsElfSide drives the real CONNECT/READY launch with a matchmaker-
// assigned Elf side (PendingBattle.Team) and checks the self PLAYER_REG carries team 2 --
// end-to-end proof that a player can actually enter «Штурм» on the Elf base.
func TestDotaElfLaunchAssignsElfSide(t *testing.T) {
	store := session.NewStore()
	s := New(store)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()
	go s.Serve(ln)

	const userID int32 = 1
	dm := gamedata.DotaMaps()[0]
	av := avatarByPrefab(t, "Avtr_Tank_Velial")
	store.SetPendingBattle(userID, session.PendingBattle{
		MapID: dm.ID, AvatarID: av.ID, Passwd: "pw", Scene: dm.Scene, Room: dm.ID,
		Team: dotaTeamElf,
	})

	cl, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer cl.Close()
	_ = cl.SetDeadline(time.Now().Add(5 * time.Second))
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

	// The self PLAYER_REG (id == userID) must put the player on the Elf side (team 2).
	for {
		p, err := r.Read()
		if err != nil {
			t.Fatalf("never saw the self PLAYER_REG on the Elf side: %v", err)
		}
		if p.Cmd != battleproto.CmdPlayerReg || p.Args == nil {
			continue
		}
		if id, _ := p.Args.GetInt("id"); id != userID {
			continue
		}
		team, _ := p.Args.GetInt("team")
		if team != dotaTeamElf {
			t.Fatalf("self PLAYER_REG team = %d, want %d (Elf side, from PendingBattle.Team)", team, dotaTeamElf)
		}
		return
	}
}
