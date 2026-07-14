package battleserver

import (
	"testing"

	"tanatserver/internal/gamedata"
)

// TestBossSkillCastsAndHits: a boss with abilities, given a target in range,
// starts a cast (schedules a hit + sets the cooldown), and the hit damages the
// player when the wind-up lands.
func TestBossSkillCastsAndHits(t *testing.T) {
	s, c, _, sx, sy := newNavConn(t)
	hs := c.huntState
	hs.hp = 5000 // survive the hit

	// Pick the first mob that has a skill kit (the bosses).
	idx, mb := -1, gamedata.Mob{}
	for i, m := range gamedata.Mobs() {
		if len(m.Skills) > 0 {
			idx, mb = i, m
			break
		}
	}
	if idx < 0 {
		t.Fatal("no boss mob defines skills")
	}

	now := float64(s.battleTime())
	boss := &mobState{id: 2000, mobIdx: idx, mob: mb, x: sx + 3, y: sy, hp: mb.Health, aggro: true}
	boss.skillReady = make([]float64, len(mb.Skills))
	hs.mobs[boss.id] = boss
	hs.tr.add(boss.id)

	c.mvMu.Lock()
	defer c.mvMu.Unlock()

	if !s.tryBossSkillLocked(c, boss, c.members(), now) {
		t.Fatal("boss did not cast a skill with the target in range and LoS clear")
	}
	if boss.skillHitAt <= now {
		t.Fatalf("cast scheduled no hit (skillHitAt=%.2f, now=%.2f)", boss.skillHitAt, now)
	}
	if boss.skillReady[0] <= now {
		t.Fatal("skill cooldown was not armed after casting")
	}

	// The hit lands only after the wind-up.
	hpBefore := hs.hp
	impact := boss.skillHitAt
	boss.skillHitAt = 0
	s.landBossSkillLocked(c, boss, c.members(), impact)
	if hs.hp >= hpBefore {
		t.Fatalf("boss skill dealt no damage (hp %.0f -> %.0f)", hpBefore, hs.hp)
	}

	// A mob with no skills never casts.
	plain := &mobState{id: 2001, mobIdx: 0, mob: gamedata.Mobs()[0], x: sx + 2, y: sy, aggro: true}
	if s.tryBossSkillLocked(c, plain, c.members(), now) {
		t.Fatal("a skill-less mob cast a boss skill")
	}
}
