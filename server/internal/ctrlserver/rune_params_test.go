package ctrlserver

import (
	"testing"

	"tanatserver/internal/amf"
	"tanatserver/internal/gamedata"
)

// findRune returns the catalog rune with the given infix (e.g. "Health_S4_Grey").
func findRune(t *testing.T, infix string) gamedata.Rune {
	t.Helper()
	for _, r := range gamedata.Runes() {
		if r.Infix == infix {
			return r
		}
	}
	t.Fatalf("rune %q not in catalog", infix)
	return gamedata.Rune{}
}

// TestRuneParamsAMFShape checks the items.amf tooltip params the Ctrl catalog attaches to a rune:
// a placeholder rune (S4+) ships one {skill_id,impact:0,value>0} entry per LongDesc placeholder so
// the client substitutes "{Health}" with a number; a literal-desc rune (S1-S3) ships none.
func TestRuneParamsAMFShape(t *testing.T) {
	// Literal-desc rune: no params (its card already reads the literal from the text).
	if got := runeParamsAMF(findRune(t, "Health_S1_Grey")); got != nil {
		t.Errorf("Health_S1_Grey: want nil params (literal desc), got %d entries", len(got.Dense))
	}

	// Placeholder Health rune: exactly {Health, HealthRegen}, both ADD with a positive value.
	params := runeParamsAMF(findRune(t, "Health_S4_Grey"))
	if params == nil {
		t.Fatal("Health_S4_Grey: want params, got nil")
	}
	seen := map[string]float64{}
	for _, v := range params.Dense {
		e, ok := v.(*amf.MixedArray)
		if !ok {
			t.Fatalf("param entry is %T, want *amf.MixedArray", v)
		}
		key, _ := e.GetString("skill_id")
		impact, _ := e.GetInt("impact")
		val, _ := e.GetFloat("value")
		if impact != 0 {
			t.Errorf("param %s impact=%d, want 0 (ADD)", key, impact)
		}
		if val <= 0 {
			t.Errorf("param %s value=%v, want > 0", key, val)
		}
		seen[key] = val
	}
	if len(seen) != 2 || seen["Health"] == 0 {
		t.Errorf("Health_S4_Grey params = %v, want {Health,HealthRegen} both set", seen)
	}
	if _, ok := seen["HealthRegen"]; !ok {
		t.Errorf("Health_S4_Grey missing HealthRegen param (has %v)", seen)
	}
	// The primary Health value must equal the applied buff (display == effect).
	if r := findRune(t, "Health_S4_Grey"); seen["Health"] != r.EffectiveValue() {
		t.Errorf("Health param %v != EffectiveValue %v", seen["Health"], r.EffectiveValue())
	}

	// AttackPower rune has a single placeholder ({DamageMin}), no regen line.
	ap := runeParamsAMF(findRune(t, "AttackPower_S4_Grey"))
	if ap == nil || len(ap.Dense) != 1 {
		t.Fatalf("AttackPower_S4_Grey: want 1 param ({DamageMin}), got %v", ap)
	}
	if k, _ := ap.Dense[0].(*amf.MixedArray).GetString("skill_id"); k != "DamageMin" {
		t.Errorf("AttackPower param key = %q, want DamageMin", k)
	}
}
