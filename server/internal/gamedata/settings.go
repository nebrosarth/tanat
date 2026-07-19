package gamedata

// RUNTIME SETTINGS. Almost all of this package is authored, compile-time content
// (the avatar/mob rosters, per-class stat templates, the scaling formulas). The
// admin panel needs to tune a curated slice of that WHILE THE SERVER RUNS, so the
// tunable knobs live here in a single immutable Settings snapshot swapped
// atomically: readers (combat scaling, unit spawn, fog) load the current pointer
// lock-free on the hot path; the admin server publishes a new snapshot with
// Apply/Update. The authored defaults reproduce the exact numbers the code used
// before this layer existed, so with an untouched Settings nothing changes.
//
// Persistence lives OUTSIDE this package: the admin server marshals a Snapshot to
// JSON and stores it in the SQLite meta table, and on boot the process loads that
// blob and calls ApplyJSON. gamedata therefore never imports session/storage.

import (
	"encoding/json"
	"sync/atomic"
)

// Settings is the full set of runtime-tunable knobs. It is treated as immutable
// once stored: writers always build a fresh value (with fresh override maps) and
// swap the pointer, so concurrent readers never see a half-updated map.
type Settings struct {
	// FogOfWarEnabled governs the «Штурм» war-fog: false reveals the whole map
	// (units broadcast a reveal-all vision radius instead of their normal one). Only
	// the «Штурм» map_1_0 and map_0_0 scenes bake the war-fog plane the client needs,
	// so this knob is inert on «Охота» -- use HuntFogEnabled for that.
	FogOfWarEnabled bool `json:"fog_of_war"`

	// HuntFogEnabled governs «Охота» concealment, which is a DIFFERENT mechanism:
	// Hunt scenes bake no war-fog plane, so the server hand-rolls it -- a mob is only
	// created on the client once a player draws near, and rendered translucent in the
	// outer ring. false disables that: every mob on the map is revealed and simulated
	// from the start (no reveal-on-approach, no shade), i.e. the whole map is visible.
	HuntFogEnabled bool `json:"hunt_fog"`

	// Mob per-level scaling slopes: the fraction added per placement level above 1
	// (level-1 is always identity). Defaults 0.15/0.10/0.40/0.15 reproduce the old
	// MobHPMul/MobDmgMul/MobXPMul/MobCoinMul constants.
	MobHPPerLevel   float64 `json:"mob_hp_per_level"`
	MobDmgPerLevel  float64 `json:"mob_dmg_per_level"`
	MobXPPerLevel   float64 `json:"mob_xp_per_level"`
	MobCoinPerLevel float64 `json:"mob_coin_per_level"`

	// Hero per-level growth slopes (damage/heal/spell-power, and max health/mana).
	// Defaults 0.06/0.06 reproduce the old levelCombatGrowth/levelHealthGrowth.
	HeroPowerPerLevel  float64 `json:"hero_power_per_level"`
	HeroHealthPerLevel float64 `json:"hero_health_per_level"`

	// Global reward multipliers applied on TOP of per-level scaling -- the "2x coin
	// weekend" event knob. 1.0 = no change.
	XPMultiplier   float64 `json:"xp_multiplier"`
	CoinMultiplier float64 `json:"coin_multiplier"`

	// New-hero starting wallet (was hard-coded 1000/100 in CreateHero).
	NewHeroMoney   int32 `json:"new_hero_money"`
	NewHeroDiamond int32 `json:"new_hero_diamond"`

	// Per-entity base-stat overrides, applied when a unit is instantiated into a
	// battle (so a change takes effect for newly-spawned units without a restart).
	// AvatarOverrides is keyed by avatar id (see AvatarByID); MobOverrides by mob
	// roster index (see MobByIndex). The inner map is stat-name -> new value; only
	// listed stats are overridden, everything else keeps its authored value. Empty
	// maps (the default) mean "authored stats verbatim". Integer JSON keys are
	// encoded as quoted strings by encoding/json, which round-trips fine.
	AvatarOverrides map[int32]map[string]float64 `json:"avatar_overrides,omitempty"`
	MobOverrides    map[int32]map[string]float64 `json:"mob_overrides,omitempty"`
}

// settingsBox holds the live Settings pointer. Readers Load() it lock-free.
var settingsBox atomic.Pointer[Settings]

func init() { settingsBox.Store(defaultSettings()) }

// defaultSettings returns the authored baseline -- every value chosen so the game
// behaves EXACTLY as it did before this layer existed.
func defaultSettings() *Settings {
	return &Settings{
		FogOfWarEnabled:    true,
		HuntFogEnabled:     true,
		MobHPPerLevel:      0.15,
		MobDmgPerLevel:     0.10,
		MobXPPerLevel:      0.40,
		MobCoinPerLevel:    0.15,
		HeroPowerPerLevel:  levelCombatGrowth,
		HeroHealthPerLevel: levelHealthGrowth,
		XPMultiplier:       1.0,
		CoinMultiplier:     1.0,
		NewHeroMoney:       1000,
		NewHeroDiamond:     100,
		AvatarOverrides:    map[int32]map[string]float64{},
		MobOverrides:       map[int32]map[string]float64{},
	}
}

// settings returns the current immutable snapshot pointer. Callers MUST treat it
// (and its maps) as read-only -- mutating it is a data race with other readers.
func settings() *Settings { return settingsBox.Load() }

// Snapshot returns a deep COPY the caller may freely read or mutate (e.g. the
// admin server before re-Applying). Override maps are cloned.
func Snapshot() Settings {
	s := *settings()
	s.AvatarOverrides = cloneOverrides(s.AvatarOverrides)
	s.MobOverrides = cloneOverrides(s.MobOverrides)
	return s
}

// Apply publishes s as the new live settings. It deep-copies the override maps
// (so a later mutation of the caller's maps can't alias the live snapshot) and
// clamps the values that would break the game if left non-positive.
func Apply(s Settings) {
	s.AvatarOverrides = cloneOverrides(s.AvatarOverrides)
	s.MobOverrides = cloneOverrides(s.MobOverrides)
	normalize(&s)
	settingsBox.Store(&s)
}

// Update reads the current settings, lets fn mutate a copy, and publishes it.
// Returns the published snapshot. Concurrent Updates are last-writer-wins (the
// admin server serializes them anyway).
func Update(fn func(*Settings)) Settings {
	s := Snapshot()
	fn(&s)
	Apply(s)
	return Snapshot()
}

// MarshalSettings serializes the live settings for persistence.
func MarshalSettings() ([]byte, error) { s := Snapshot(); return json.Marshal(s) }

// ApplyJSON parses a persisted settings blob and publishes it. It starts from the
// authored defaults and overlays only the fields present in the JSON, so a blob
// written by an older build (missing a knob added later) keeps that knob's
// default instead of zeroing it.
func ApplyJSON(b []byte) error {
	s := *defaultSettings()
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	Apply(s)
	return nil
}

// normalize keeps the published settings sane: non-nil maps, and no negative
// multipliers/wallets (a negative would produce nonsense combat/economy). Slopes
// may be zero (flat scaling) or negative in principle, so they are left alone.
func normalize(s *Settings) {
	if s.AvatarOverrides == nil {
		s.AvatarOverrides = map[int32]map[string]float64{}
	}
	if s.MobOverrides == nil {
		s.MobOverrides = map[int32]map[string]float64{}
	}
	if s.XPMultiplier < 0 {
		s.XPMultiplier = 0
	}
	if s.CoinMultiplier < 0 {
		s.CoinMultiplier = 0
	}
	if s.NewHeroMoney < 0 {
		s.NewHeroMoney = 0
	}
	if s.NewHeroDiamond < 0 {
		s.NewHeroDiamond = 0
	}
}

func cloneOverrides(m map[int32]map[string]float64) map[int32]map[string]float64 {
	out := make(map[int32]map[string]float64, len(m))
	for k, v := range m {
		inner := make(map[string]float64, len(v))
		for kk, vv := range v {
			inner[kk] = vv
		}
		out[k] = inner
	}
	return out
}

// FogOfWar reports whether the battle server should send normal (fogged) vision
// radii on «Штурм». False means reveal-all.
func FogOfWar() bool { return settings().FogOfWarEnabled }

// HuntFog reports whether «Охота» concealment is active. False means every mob is
// revealed and simulated from the start (no reveal-on-approach, no shade overlay).
func HuntFog() bool { return settings().HuntFogEnabled }

// NewHeroWallet returns the starting money/diamond for a freshly created hero.
func NewHeroWallet() (money, diamond int32) {
	s := settings()
	return s.NewHeroMoney, s.NewHeroDiamond
}

// ---- per-entity stat overrides ----

// AvatarStatFields / MobStatFields are the stat names the admin panel may
// override, in display order. Kept in sync with applyAvatarMods/applyMobMods and
// AvatarStat/MobStat below.
func AvatarStatFields() []string {
	return []string{"Health", "Mana", "HealthRegen", "ManaRegen", "AttackSpeed",
		"AttackRange", "DmgMin", "DmgMax", "PhysArmor", "MagicArmor", "SpellPower"}
}

func MobStatFields() []string {
	return []string{"Health", "Mana", "DmgMin", "DmgMax", "AttackSpeed", "Speed",
		"XP", "Coins", "PhysArmor", "AttackRange"}
}

// AvatarStat reads one stat off an avatar by field name (the inverse of an
// override). Unknown field -> 0.
func AvatarStat(a Avatar, field string) float64 {
	switch field {
	case "Health":
		return a.Health
	case "Mana":
		return a.Mana
	case "HealthRegen":
		return a.HealthRegen
	case "ManaRegen":
		return a.ManaRegen
	case "AttackSpeed":
		return a.AttackSpeed
	case "AttackRange":
		return a.AttackRange
	case "DmgMin":
		return float64(a.DmgMin)
	case "DmgMax":
		return float64(a.DmgMax)
	case "PhysArmor":
		return a.PhysArmor
	case "MagicArmor":
		return a.MagicArmor
	case "SpellPower":
		return a.SpellPower
	}
	return 0
}

// MobStat reads one stat off a mob by field name. Unknown field -> 0.
func MobStat(m Mob, field string) float64 {
	switch field {
	case "Health":
		return m.Health
	case "Mana":
		return m.Mana
	case "DmgMin":
		return float64(m.DmgMin)
	case "DmgMax":
		return float64(m.DmgMax)
	case "AttackSpeed":
		return m.AttackSpeed
	case "Speed":
		return m.Speed
	case "XP":
		return m.XP
	case "Coins":
		return float64(m.Coins)
	case "PhysArmor":
		return m.PhysArmor
	case "AttackRange":
		return m.AttackRange
	}
	return 0
}

func applyAvatarMods(a *Avatar, mods map[string]float64) {
	for k, v := range mods {
		switch k {
		case "Health":
			a.Health = v
		case "Mana":
			a.Mana = v
		case "HealthRegen":
			a.HealthRegen = v
		case "ManaRegen":
			a.ManaRegen = v
		case "AttackSpeed":
			a.AttackSpeed = v
		case "AttackRange":
			a.AttackRange = v
		case "DmgMin":
			a.DmgMin = int32(v)
		case "DmgMax":
			a.DmgMax = int32(v)
		case "PhysArmor":
			a.PhysArmor = v
		case "MagicArmor":
			a.MagicArmor = v
		case "SpellPower":
			a.SpellPower = v
		}
	}
}

func applyMobMods(m *Mob, mods map[string]float64) {
	for k, v := range mods {
		switch k {
		case "Health":
			m.Health = v
		case "Mana":
			m.Mana = v
		case "DmgMin":
			m.DmgMin = int32(v)
		case "DmgMax":
			m.DmgMax = int32(v)
		case "AttackSpeed":
			m.AttackSpeed = v
		case "Speed":
			m.Speed = v
		case "XP":
			m.XP = v
		case "Coins":
			m.Coins = int32(v)
		case "PhysArmor":
			m.PhysArmor = v
		case "AttackRange":
			m.AttackRange = v
		}
	}
}
