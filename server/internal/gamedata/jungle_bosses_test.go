package gamedata

import (
	"math"
	"testing"
)

// TestJungleSpawnOverride: map_4_2's battle-start point is the in-game-confirmed
// pocket (35.32,30.23), NOT navGrid42's seed marker -- SpawnAt must win over Nav.
func TestJungleSpawnOverride(t *testing.T) {
	m, ok := HuntMapByID(42)
	if !ok {
		t.Fatal("map_4_2 (id 42) missing")
	}
	sx, sy := m.Spawn()
	if sx != 35.32 || sy != 30.23 {
		t.Fatalf("map_4_2 spawn = (%.2f,%.2f), want the overridden (35.32,30.23)", sx, sy)
	}
	if !navGrid42.Walkable(sx, sy) {
		t.Fatalf("map_4_2 start (%.2f,%.2f) is not walkable", sx, sy)
	}
}

func isJungleBoss(mob int) bool {
	return mob == mobBossGrimlok || mob == mobBossTitanid || mob == mobBossFairy || mob == mobBossAnhel
}

// TestJungleBossesPinned: all four bosses are present, pinned at absolute arenas on
// walkable floor, reachable from spawn by pathfinding, carry no Skills (v1 basic
// attack), and are the LAST four spawns in the ladder order Grimlok..Anhel.
func TestJungleBossesPinned(t *testing.T) {
	m, _ := HuntMapByID(42)
	sx, sy := m.Spawn()
	order := []int{mobBossGrimlok, mobBossFairy, mobBossTitanid, mobBossAnhel}

	var bossSpawns []MobSpawn
	for _, sp := range m.Spawns {
		if isJungleBoss(sp.Mob) {
			bossSpawns = append(bossSpawns, sp)
		}
	}
	if len(bossSpawns) != 4 {
		t.Fatalf("map_4_2 has %d bosses, want 4", len(bossSpawns))
	}
	// The four bosses are the tail of the spawn list, in ladder order.
	tail := m.Spawns[len(m.Spawns)-4:]
	for i, sp := range tail {
		if sp.Mob != order[i] {
			t.Errorf("boss tail[%d] = Mob %d, want %d", i, sp.Mob, order[i])
		}
		if !sp.Abs {
			t.Errorf("boss Mob %d is not Abs", sp.Mob)
		}
		if !navGrid42.Walkable(sp.DX, sp.DY) {
			t.Errorf("boss Mob %d arena (%.2f,%.2f) not walkable", sp.Mob, sp.DX, sp.DY)
		}
		if p := navGrid42.Path(sx, sy, sp.DX, sp.DY); len(p) == 0 {
			t.Errorf("boss Mob %d arena (%.2f,%.2f) not reachable from spawn", sp.Mob, sp.DX, sp.DY)
		}
		if len(Mobs()[sp.Mob].Skills) != 0 {
			t.Errorf("boss Mob %d has Skills -- v1 jungle bosses are basic-attack only", sp.Mob)
		}
	}
}

// TestJungleBossLadder: HP / damage / XP / coins rise strictly along the ladder.
func TestJungleBossLadder(t *testing.T) {
	order := []int{mobBossGrimlok, mobBossFairy, mobBossTitanid, mobBossAnhel}
	names := []string{"Grimlok", "Fairy", "Titanid", "Anhel"}
	for i := 1; i < len(order); i++ {
		lo, hi := Mobs()[order[i-1]], Mobs()[order[i]]
		if hi.Health <= lo.Health {
			t.Errorf("%s HP %.0f <= %s HP %.0f", names[i], hi.Health, names[i-1], lo.Health)
		}
		if hi.DmgMax <= lo.DmgMax {
			t.Errorf("%s dmg %d <= %s dmg %d", names[i], hi.DmgMax, names[i-1], lo.DmgMax)
		}
		if hi.XP <= lo.XP {
			t.Errorf("%s XP %.0f <= %s XP %.0f", names[i], hi.XP, names[i-1], lo.XP)
		}
		if hi.Coins <= lo.Coins {
			t.Errorf("%s coins %d <= %s coins %d", names[i], hi.Coins, names[i-1], lo.Coins)
		}
	}
}

// TestJungleTrashGenerated: the generator produced a healthy number of trash packs,
// every generated mob sits on walkable floor (=> reachable, the grid is the spawn-
// connected component), outside the start safe ring, and is a jungle-roster creature.
func TestJungleTrashGenerated(t *testing.T) {
	m, _ := HuntMapByID(42)
	sx, sy := m.Spawn()
	var trash int
	for _, sp := range m.Spawns {
		if isJungleBoss(sp.Mob) {
			continue
		}
		trash++
		if !sp.Abs {
			t.Errorf("trash Mob %d not Abs", sp.Mob)
		}
		if !navGrid42.Walkable(sp.DX, sp.DY) {
			t.Errorf("trash Mob %d at (%.1f,%.1f) not walkable", sp.Mob, sp.DX, sp.DY)
		}
		if d := math.Hypot(sp.DX-sx, sp.DY-sy); d < dungeonSpawnClear {
			t.Errorf("trash Mob %d is %.1fm from spawn -- inside the safe ring (%.0f)", sp.Mob, d, dungeonSpawnClear)
		}
		// The jungle roster is mobSpider..mobGolemElite, PLUS two deliberate cross-roster
		// pins: the «Деревенский пожар» burning skeleton (jungleBurntVillage) and the
		// hunter-flower forced into the user-marked western corridor (jungleFlowerPool) --
		// still walkable and spawn-clear (checked above), just not native jungle creatures.
		if sp.Mob != mobSkeletonBurning && sp.Mob != mobFlowerHunter && (sp.Mob < mobSpider || sp.Mob > mobGolemElite) {
			t.Errorf("spawn Mob %d is not a jungle-roster creature (or a deliberate pin)", sp.Mob)
		}
	}
	if trash < 20 {
		t.Errorf("only %d jungle trash mobs generated -- expected a populated map", trash)
	}
}

// TestBossArenasClearOnAllMaps: on EVERY map that pins bosses, no trash mob spawns
// within the boss's mob-free ring -- bosses are engaged in a clean arena, the same way
// respawn points are kept clear. Pack centres are >= dungeonBossClear from each boss;
// members spread up to dungeonPackRadius, so the guaranteed member floor is the
// difference (with a rounding margin).
func TestBossArenasClearOnAllMaps(t *testing.T) {
	floor := dungeonBossClear - dungeonPackRadius - 0.3
	check := func(name string, spawns []MobSpawn, bosses [][2]float64, isBoss func(int) bool) {
		for _, sp := range spawns {
			if isBoss(sp.Mob) {
				continue
			}
			mx, my := sp.DX, sp.DY // all trash here is Abs
			for _, b := range bosses {
				if d := math.Hypot(mx-b[0], my-b[1]); d < floor {
					t.Errorf("%s: trash Mob %d at (%.1f,%.1f) is %.1fm from a boss (%.1f,%.1f) -- inside the %.0fm arena ring",
						name, sp.Mob, mx, my, d, b[0], b[1], dungeonBossClear)
				}
			}
		}
	}
	cb := make([][2]float64, len(dungeonBosses))
	for i, b := range dungeonBosses {
		cb[i] = [2]float64{b.x, b.y}
	}
	isCryptBoss := func(m int) bool {
		return m == mobBossElgorm || m == mobBossVelial || m == mobBossCerber || m == mobBossHekata
	}
	check("crypt map_4_0", dungeonPack40, cb, isCryptBoss)

	jb := make([][2]float64, len(jungleBosses))
	for i, b := range jungleBosses {
		jb[i] = [2]float64{b.x, b.y}
	}
	check("jungle map_4_2", junglePack, jb, isJungleBoss)
}

// TestBurntVillageInJungle: the «Деревенский пожар» quest (Map_4_2 Stage2_2) is now completable --
// the standard pack generator fills the burnt village with burning-skeleton packs (no hand-placed
// pins), every one on walkable floor, reachable, clear of the spawn/boss rings, and within the
// burned-house cluster. Pins the user's request that this creature populate map_4_2's village.
func TestBurntVillageInJungle(t *testing.T) {
	m, _ := HuntMapByID(42)
	sx, sy := m.Spawn()
	// The ACTUAL burnt village -- the baked House_*_Burned / Skeleton_burned prop cluster
	// centred at jungleBurntVillageCenter (extracted from map_4_2.unity3d).
	n := 0
	for _, sp := range m.Spawns {
		if sp.Mob != mobSkeletonBurning {
			continue
		}
		n++
		if !navGrid42.Walkable(sp.DX, sp.DY) {
			t.Errorf("burnt-village skeleton at (%.1f,%.1f) is not walkable", sp.DX, sp.DY)
		}
		if p := navGrid42.Path(sx, sy, sp.DX, sp.DY); len(p) == 0 {
			t.Errorf("burnt-village skeleton at (%.1f,%.1f) is not reachable from spawn", sp.DX, sp.DY)
		}
		if d := math.Hypot(sp.DX-sx, sp.DY-sy); d < dungeonSpawnClear {
			t.Errorf("burnt-village skeleton is %.1fm from spawn -- inside the safe ring", d)
		}
		for _, b := range jungleBosses {
			if d := math.Hypot(sp.DX-b.x, sp.DY-b.y); d < dungeonBossClear {
				t.Errorf("burnt-village skeleton at (%.1f,%.1f) is %.1fm from a boss -- inside its arena", sp.DX, sp.DY, d)
			}
		}
		if d := math.Hypot(sp.DX-jungleBurntVillageCenter.X, sp.DY-jungleBurntVillageCenter.Y); d > jungleBurntVillageRadius {
			t.Errorf("burnt-village skeleton at (%.1f,%.1f) is %.1fm from the burned-house cluster -- outside the village", sp.DX, sp.DY, d)
		}
	}
	if n < 3 {
		t.Errorf("expected the burnt village populated with burning-skeleton packs, got %d", n)
	}
}

// TestBurntVillageOnlySkeletons: the burnt village is a skeletons-only zone -- the standard
// generator fills it, but every mob inside its radius is a burning skeleton; no regular jungle
// pack (spider/tribe/dino/...) draws there, so the ruins read as a burned-out village.
func TestBurntVillageOnlySkeletons(t *testing.T) {
	vc := jungleBurntVillageCenter
	inside := 0
	for _, sp := range junglePack {
		if math.Hypot(sp.DX-vc.X, sp.DY-vc.Y) > jungleBurntVillageRadius {
			continue
		}
		inside++
		if sp.Mob != mobSkeletonBurning {
			t.Errorf("mob %d at (%.1f,%.1f) is inside the burnt village but not a burning skeleton",
				sp.Mob, sp.DX, sp.DY)
		}
	}
	if inside < 3 {
		t.Fatalf("burnt village holds too few mobs (%d) -- the generator should fill it with skeleton packs", inside)
	}
}

// TestGolemGating: golems spawn ONLY on the Titanid trail (golemAllowed), never toward
// the other bosses; and the gate is not vacuous (at least one golem was placed).
func TestGolemGating(t *testing.T) {
	m, _ := HuntMapByID(42)
	golems := 0
	for _, sp := range m.Spawns {
		if !isGolem(sp.Mob) {
			continue
		}
		golems++
		if !golemAllowed(sp.DX, sp.DY) {
			t.Errorf("golem Mob %d at (%.1f,%.1f) violates the Titanid-trail gate (x<=%.1f)",
				sp.Mob, sp.DX, sp.DY, golemGate.X+46)
		}
	}
	if golems == 0 {
		t.Error("no golems generated -- the Titanid-trail golem zone is empty")
	}
	t.Logf("generated %d golems, all west of x=%.1f (Titanid trail)", golems, golemGate.X+46)
}
