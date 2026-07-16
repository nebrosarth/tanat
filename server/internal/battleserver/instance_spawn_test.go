package battleserver

import "testing"

// map_4_1's nav grid is minimap-reconstructed and its roster is now bosses + elite packs
// placed walkable-by-construction (invasionPack41). This asserts the real invariant:
// every mob (trash and boss) spawns on the walkable floor and is reachable from the
// player spawn, with newHuntInstance's off-floor clamp as the backstop.
func TestMap41MobsSnapToWalkableFloor(t *testing.T) {
	inst := newHuntInstance(&Server{}, 1, 41)
	if inst.nav == nil {
		t.Fatal("map_4_1 instance has no nav grid")
	}
	sx, sy := inst.nav.Spawn()
	if len(inst.mobs) == 0 {
		t.Fatal("map_4_1 instance spawned no mobs")
	}
	for id, m := range inst.mobs {
		if !inst.nav.Walkable(float64(m.x), float64(m.y)) {
			t.Errorf("mob %d spawned off the walkable floor at (%.1f,%.1f)", id, m.x, m.y)
		}
		if p := inst.nav.Path(sx, sy, float64(m.x), float64(m.y)); len(p) == 0 {
			t.Errorf("mob %d at (%.1f,%.1f) is not reachable from spawn", id, m.x, m.y)
		}
	}
}
