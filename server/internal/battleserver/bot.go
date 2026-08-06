package battleserver

// Server-controlled «Штурм» bots: a bot is an ordinary *conn/*huntState wired up with a
// net.Pipe() standing in for the TCP socket (the same construction dota_pvp_test.go's
// dotaPlayerConn uses for a second test player, here used for real runtime), so every
// existing combat/economy/order entry point (startAttackLocked, startSkillOrderLocked,
// upgradeSkillLocked, buyItemLocked, moveToLocked...) treats it exactly like a live
// player -- a bot never bypasses validation a real client would have to pass. The
// "thinking" happens in bot_brain.go, ticked from runInstanceTicker alongside the normal
// per-member upkeep.

import (
	"fmt"
	"log"
	"net"
	"os"
	"strconv"

	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
)

// botIDBase is the reserved objID/selfPlayerID space for bots, clear of every other id
// space a connection touches: real Ctrl-issued userIDs (small, sequential from 1), avatar
// objIDs (1000+userID), Hunt mobs (2000+), «Штурм» structures (50000+), creeps (60000+),
// the boss guard (70000), summons (300000+), trap anchors (400000+), loot drops. A bot
// never collides with a real player's session even at large matchmaking scale.
const botIDBase int32 = 900000

// botRosterAvatarIDs are the 10 balanced avatars (see the PVP redesign: full stat +
// ability economy passes) bots are drawn from. 11 avatars came out of that redesign with
// fully-designed kits (Astarot, Mihalych, Sigilion, Velial, Teridin, Tangren, Abominator,
// BlackDragon, PlusMinus, Neirofim, Ariana); Mihalych is dropped here purely to trim the
// role spread down to exactly 10 -- 6 of the 11 are the Killer archetype, and cutting one
// of the Killers (arbitrarily, Mihalych) leaves 2 Warrior / 5 Killer / 1 Mage / 2 Support,
// a materially more varied 5-a-side draw than 2/6/1/2.
var botRosterAvatarIDs = []int32{7, 10, 13, 17, 18, 22, 23, 35, 36, 38}

// botAvatarRoster resolves botRosterAvatarIDs to their gamedata.Avatar each call (cheap:
// gamedata.AvatarByID is an in-memory slice scan), so a live admin stat override is picked
// up the same way a real player's avatar selection already is.
func botAvatarRoster() []gamedata.Avatar {
	out := make([]gamedata.Avatar, 0, len(botRosterAvatarIDs))
	for _, id := range botRosterAvatarIDs {
		if a, ok := gamedata.AvatarByID(id); ok {
			out = append(out, a)
		}
	}
	return out
}

// dotaBotFillTarget reads TANAT_DOTA_BOTS: unset, empty, or non-positive = bots disabled
// (default, unchanged behaviour for every existing launch path); an integer 2..10 is the
// target TOTAL headcount (real players + bots) a fresh «Штурм» match is topped up to.
// Gated exactly like the existing TANAT_CASTLE_TEST_MAP dev-only switch.
func dotaBotFillTarget() int {
	v := os.Getenv("TANAT_DOTA_BOTS")
	if v == "" {
		return 0
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 2 {
		return 0
	}
	if n > 10 {
		n = 10
	}
	return n
}

// maybeFillDotaBotsLocked backfills a fresh «Штурм» match with bots up to
// dotaBotFillTarget's headcount, once per instance (dotaState.botsFilled). Called after a
// real player's world state is built (so their side/team is already assigned and the
// alternating side counter is where a genuine second joiner would find it). A no-op
// whenever bots are disabled, the instance isn't «Штурм», or it was already filled.
func (s *Server) maybeFillDotaBotsLocked(c *conn) {
	inst := c.inst
	if inst == nil || inst.dota == nil {
		return
	}
	target := dotaBotFillTarget()
	if target < 2 {
		return
	}
	inst.mu.Lock()
	defer inst.mu.Unlock()
	if inst.dota.botsFilled {
		return
	}
	inst.dota.botsFilled = true
	s.spawnDotaBotsLocked(inst, target)
}

// spawnDotaBotsLocked adds bot heroes until the match reaches target headcount, keeping
// the two sides as even as possible (balancedDotaSideLocked), cycling the balanced avatar
// roster. Caller holds inst.mu.
func (s *Server) spawnDotaBotsLocked(inst *huntInstance, target int) {
	roster := botAvatarRoster()
	if len(roster) == 0 {
		log.Printf("battle: «Штурм» bot fill requested but the avatar roster is empty, skipping")
		return
	}
	for slot := 0; len(inst.members) < target; slot++ {
		side := s.balancedDotaSideLocked(inst)
		av := roster[slot%len(roster)]
		s.newBotConnLocked(inst, slot, side, av)
	}
}

// balancedDotaSideLocked returns whichever side currently has fewer members, counted
// directly off inst.members rather than trusting dotaState.nextSide's own alternation.
// That counter is NOT authoritative: a real matchmade player carries an explicit
// PendingBattle.Team and sendHuntWorldState assigns it straight from sideForTeam,
// bypassing assignSide() entirely -- so nextSide never advances for them. Filling bots by
// blindly consuming nextSide after such a player joined double-counted their side (seen
// live: a 10-headcount fill split 6/4 instead of 5/5). Counting actual membership is
// correct regardless of which path assigned anyone's side. Ties fall back to assignSide()
// (which still advances it, so a LATER real joiner with no pre-assigned team keeps
// alternating sensibly relative to the bots already here).
func (s *Server) balancedDotaSideLocked(inst *huntInstance) gamedata.DotaSide {
	humanN, elfN := 0, 0
	for _, mem := range inst.members {
		if mem.playerTeam() == dotaTeamHuman {
			humanN++
		} else {
			elfN++
		}
	}
	switch {
	case humanN < elfN:
		return gamedata.DotaSideHuman
	case elfN < humanN:
		return gamedata.DotaSideElf
	default:
		return inst.dota.assignSide()
	}
}

// botName is the display nick a bot's PLAYER_REG carries.
func botName(slot int) string { return fmt.Sprintf("Bot_%d", slot+1) }

// newBotConnLocked builds a full «Штурм» participant with no real network socket behind
// it and inserts it into inst.members, mirroring sendHuntWorldState's real-player setup
// (side/spawn, unlearned skills at rank 0 with 1 free point, hasProjectile) closely enough
// that every shared combat/order path (startAttackLocked, startSkillOrderLocked,
// upgradeSkillLocked, buyItemLocked...) behaves identically for a bot and a live player.
// Caller holds inst.mu.
func (s *Server) newBotConnLocked(inst *huntInstance, slot int, side gamedata.DotaSide, av gamedata.Avatar) *conn {
	id := botIDBase + int32(slot)
	srv, cli := net.Pipe()
	r := battleproto.NewReader(cli)
	go func() {
		for {
			if _, err := r.Read(); err != nil {
				return
			}
		}
	}()
	s.Store.CreateBotHero(id, botName(slot))

	now := float64(s.battleTime())
	team := teamForSide(side)
	// How many teammates (real or bot) are already on this side -- decides the lane
	// this bot defaults to (see assignBotLane), so a full 5-a-side team spreads
	// 2/1/2 across the three lanes instead of piling onto one.
	sideOrdinal := 0
	for _, mem := range inst.members {
		if mem.playerTeam() == team {
			sideOrdinal++
		}
	}
	sx, sy := inst.dota.sideSpawn(side)
	// Fan bots on the same side out a little so a full 5-bot side doesn't spawn stacked
	// on one point, mirroring spawnCreepWaveLocked's own fan-out.
	off := float32(slot/2) * 1.2
	px, py := float32(sx)+off, float32(sy)-off

	kit := gamedata.SkillsFor(av)
	c := &conn{Conn: srv}
	c.objID, c.selfPlayerID = id, id
	c.x, c.y, c.snapT = px, py, s.battleTime()
	c.nav = inst.nav
	c.name = botName(slot)

	hs := &huntState{
		av: av, kit: kit,
		mobs: inst.mobs, summons: map[int32]*summonState{}, summonProtos: map[string]int32{},
		hp: av.Health, mana: av.Mana, team: team, worldReady: true,
		hasProjectile: kit.AttackProjectile,
		points:        1,
	}
	for i := range hs.skillLevel {
		hs.skillLevel[i] = int32(skillStartRank(i + 1))
	}
	hs.respawnX, hs.respawnY = px, py
	hs.rebornIdx = -1
	hs.tr.add(id)
	hs.inst = inst
	c.huntState = hs
	c.inst = inst
	c.lk = &inst.mu

	inst.members[id] = c
	s.dotaWorldSetupLocked(c, now)
	s.introduceMemberLocked(c, now)
	s.sendInitialMoneyLocked(c)

	inst.bots[id] = newBotBrain(c, slot, sideOrdinal)

	log.Printf("battle: bot %d (%s, %s) joined «Штурм» room=%d team=%d at (%.1f,%.1f)",
		id, av.Prefab, botName(slot), inst.id, team, px, py)
	return c
}
