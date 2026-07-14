package gamedata

import (
	"bufio"
	"os"
	"strings"
	"testing"
)

// validPrefabs loads the exact prefab names the client's AssetLoader can resolve,
// dumped from the client's data/resources.xml (<Prefab name="..."/>). The client
// keys assets on this name; a name that is not present resolves to null and the
// object spawns invisible and untargetable.
func validPrefabs(t *testing.T) map[string]bool {
	t.Helper()
	f, err := os.Open("testdata/valid_prefabs.txt")
	if err != nil {
		t.Fatalf("open valid_prefabs.txt: %v", err)
	}
	defer f.Close()
	set := map[string]bool{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if s := strings.TrimSpace(sc.Text()); s != "" {
			set[s] = true
		}
	}
	return set
}

// TestMobPrefabsResolvable guards against the invisible-mob bug: every mob's
// Prefab must be a real prefab name the client can load, not just a bundle name.
func TestMobPrefabsResolvable(t *testing.T) {
	valid := validPrefabs(t)
	for i, m := range Mobs() {
		if !valid[m.Prefab] {
			t.Errorf("mob %d (%s) prefab %q is not a loadable prefab name (would spawn invisible)", i, m.NameKey, m.Prefab)
		}
	}
}

// TestSummonPrefabsResolvable guards the same failure for summoned units.
func TestSummonPrefabsResolvable(t *testing.T) {
	valid := validPrefabs(t)
	seen := map[string]bool{}
	for _, ks := range skillsByPrefab {
		for _, sk := range ks.Skills {
			for _, op := range sk.Ops {
				collectSummonUnits(op, seen)
			}
		}
	}
	for unit := range seen {
		if !valid[unit] {
			t.Errorf("summon unit prefab %q is not a loadable prefab name (would spawn invisible)", unit)
		}
	}
}

func collectSummonUnits(op Op, seen map[string]bool) {
	if op.Kind == OpSummon && op.Unit != "" {
		seen[op.Unit] = true
	}
	for _, child := range op.Ops {
		collectSummonUnits(child, seen)
	}
}
