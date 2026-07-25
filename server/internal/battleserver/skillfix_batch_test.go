package battleserver

import (
	"net"
	"strings"
	"sync"
	"testing"

	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

// newAvatarConn builds a bare, world-ready conn for any avatar id (team 1) with a draining
// client reader so pushes to the net.Pipe never block. Mirrors newUrgConn.
func newAvatarConn(t *testing.T, avatarID int32) (*Server, *conn) {
	t.Helper()
	s := New(session.NewStore())
	srv, cli := net.Pipe()
	t.Cleanup(func() { srv.Close(); cli.Close() })

	var mu sync.Mutex
	pkts := &[]battleproto.Packet{}
	r := battleproto.NewReader(cli)
	go func() {
		for {
			p, err := r.Read()
			if err != nil {
				return
			}
			mu.Lock()
			*pkts = append(*pkts, p)
			mu.Unlock()
		}
	}()

	av, ok := gamedata.AvatarByID(avatarID)
	if !ok {
		t.Fatalf("avatar %d missing", avatarID)
	}
	c := &conn{Conn: srv}
	c.objID = 1000
	c.x, c.y, c.snapT = 0, 0, s.battleTime()
	hs := &huntState{
		av: av, kit: gamedata.SkillsFor(av), team: 1, worldReady: true,
		mobs: map[int32]*mobState{}, summons: map[int32]*summonState{},
		hp: 500, mana: 100,
	}
	for i := range hs.skillLevel {
		hs.skillLevel[i] = 1
	}
	hs.tr.add(c.objID)
	c.huntState = hs
	t.Cleanup(func() { c.mvMu.Lock(); hs.closed = true; c.mvMu.Unlock() })
	return s, c
}

// TestDualCastMaskRewrite: a friend-or-foe skill authored "ENEMY+NOT_BUILDING+FRIEND"
// (unsatisfiable under the client's AND-combined TargetValidator) is emitted to the client
// as a satisfiable NOT_BUILDING mask, so a click on an enemy, ally, or self is accepted.
func TestDualCastMaskRewrite(t *testing.T) {
	frost, _ := gamedata.AvatarByID(34)
	s3 := gamedata.SkillsFor(frost).Skills[2] // «Гробница холода»
	if !strings.Contains(s3.Target, "ENEMY") || !strings.Contains(s3.Target, "FRIEND") {
		t.Fatalf("Frost s3 authored target = %q, want a dual ENEMY+FRIEND mask", s3.Target)
	}
	desc := activeChildDesc(frost, s3)
	if !strings.Contains(desc, `name="target" value="NOT_BUILDING"`) {
		t.Errorf("emitted target not rewritten to NOT_BUILDING:\n%s", desc)
	}
	// The unsatisfiable combination must NOT reach the client.
	if strings.Contains(desc, "ENEMY+NOT_BUILDING+FRIEND") {
		t.Error("client still receives the unsatisfiable ENEMY+…+FRIEND mask")
	}
}

// TestKionaManaOnKill: «Вдохновение» restores current mana on a kill (was a passive
// mana-regen buff that never fired on kill).
func TestKionaManaOnKill(t *testing.T) {
	s, c := newAvatarConn(t, 6) // Kiona
	hs := c.huntState
	hs.manaOnKillSlot = 3
	hs.mana = 10
	// Sanity: s3 really carries the on-kill mana op now.
	if hs.kit.Skills[2].Ops[0].Kind != gamedata.OpManaOnKill {
		t.Fatalf("Kiona s3 op[0] = %q, want mana_on_kill", hs.kit.Skills[2].Ops[0].Kind)
	}
	before := hs.mana
	s.applyManaOnKillLocked(c, float64(s.battleTime()))
	if hs.mana <= before {
		t.Errorf("mana not restored on kill: %g -> %g", before, hs.mana)
	}
}

// TestNeirofimManaScaledGate: «Паралич воли» always deals its flat base damage («наносит
// магический урон и замедляет цель, в зависимости от количества маны у цели» -- the client
// never says a mana-less target takes NO damage), but only a target that actually carries
// mana also takes the missing-mana bonus and the slow (pass-8 fix: the prior encoding gated
// the WHOLE effect, including the flat base, behind maxMana>0 -- a mana-less melee mob was
// fully immune to Neirofim's signature nuke, not just immune to its mana-scaled extras).
func TestNeirofimManaScaledGate(t *testing.T) {
	s, c := newAvatarConn(t, 36) // Neirofim
	hs := c.huntState
	op := hs.kit.Skills[0].Ops[0]
	if op.Kind != gamedata.OpManaScaledDamage {
		t.Fatalf("Neirofim s1 op[0] = %q, want mana_scaled_damage", op.Kind)
	}
	mob := gamedata.Mobs()[2]

	manaMob := &mobState{id: 501, mobIdx: 2, mob: mob, x: 1, y: 0, hp: 900, maxMana: 100, mana: 40, homed: true}
	hs.mobs[manaMob.id] = manaMob
	s.applyOpsLocked(c, []gamedata.Op{op}, opCtx{slot: 1, level: 1, target: manaMob}, float64(s.battleTime()))
	dmgWithMana := 900 - manaMob.hp
	if dmgWithMana <= 0 {
		t.Errorf("mana mob took no damage: hp=%g", manaMob.hp)
	}
	if manaMob.st.slowUntil == 0 {
		t.Error("mana mob must be slowed")
	}

	meleeMob := &mobState{id: 502, mobIdx: 2, mob: mob, x: 1, y: 0, hp: 900, maxMana: 0, mana: 0, homed: true}
	hs.mobs[meleeMob.id] = meleeMob
	s.applyOpsLocked(c, []gamedata.Op{op}, opCtx{slot: 1, level: 1, target: meleeMob}, float64(s.battleTime()))
	dmgMeleeless := 900 - meleeMob.hp
	if dmgMeleeless <= 0 {
		t.Errorf("mana-less mob must still take the flat base damage, but hp stayed %g", meleeMob.hp)
	}
	if dmgMeleeless >= dmgWithMana {
		t.Errorf("mana-less mob (no missing-mana bonus) took %g, want less than the mana mob's %g", dmgMeleeless, dmgWithMana)
	}
	if meleeMob.st.slowUntil != 0 {
		t.Error("mana-less mob must not be slowed (nothing to scale the slow from)")
	}
}

// TestGellarSoulCounter: the «Порабощение» passive icon reports the live banked soul count.
func TestGellarSoulCounter(t *testing.T) {
	_, c := newAvatarConn(t, 29) // Gellar
	hs := c.huntState
	hs.soulSlot = 2
	hs.soulStacks = 7
	cnt, ok := hs.passiveBuffCounterLocked(2, 1, 0)
	if !ok || cnt != 7 {
		t.Errorf("soul counter = %d,%v, want 7,true", cnt, ok)
	}
}

// TestVelialWillToWinCardBonus: Velial's «Воля к победе» (s3) adds its missing-HP bonus to
// the displayed attack damage — zero at full HP, positive when hurt, scaling with the gap.
func TestVelialWillToWinCardBonus(t *testing.T) {
	_, c := newAvatarConn(t, 13) // Velial
	hs := c.huntState
	now := float64(0)
	maxHP := hs.maxHPLocked(now)

	hs.hp = maxHP
	if b := hs.casterMissingHPBonusLocked(now); b != 0 {
		t.Errorf("full-HP bonus = %g, want 0", b)
	}
	hs.hp = maxHP * 0.5
	half := hs.casterMissingHPBonusLocked(now)
	if half <= 0 {
		t.Fatalf("half-HP bonus = %g, want > 0", half)
	}
	hs.hp = maxHP * 0.1
	low := hs.casterMissingHPBonusLocked(now)
	if low <= half {
		t.Errorf("bonus must grow as HP drops: half=%g low=%g", half, low)
	}
}

// TestMorlokaiTotemStationary: the «Грозовой тотем» summon is flagged stationary and never
// receives a move order (it holds its spawn point).
func TestMorlokaiTotemStationary(t *testing.T) {
	morlo, _ := gamedata.AvatarByID(33)
	s4 := gamedata.SkillsFor(morlo).Skills[3]
	var summonOp *gamedata.Op
	for i := range s4.Ops {
		if s4.Ops[i].Kind == gamedata.OpSummon {
			summonOp = &s4.Ops[i]
		}
	}
	if summonOp == nil {
		t.Fatal("Morlokai s4 has no summon op")
	}
	if !summonOp.Stationary {
		t.Error("Morlokai s4 totem summon must be Stationary")
	}
}
