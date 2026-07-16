package battleserver

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
)

// debugStartLevelOverride, when > 0, forces every hunt spawn to start at that
// character level (1-based) instead of 1. Tests set it directly; in production it
// is populated from the TANAT_HUNT_START_LEVEL env var. Purely a testing aid --
// it lets a tester spawn at, say, level 5 to reach the level-gated skill ranks
// (the ult unlocks at level 5) without grinding XP.
var debugStartLevelOverride int32

// huntStartLevel returns the character level (1-based) a hunt avatar should spawn
// at: the test override if set, else TANAT_HUNT_START_LEVEL, else 1.
func huntStartLevel() int32 {
	if debugStartLevelOverride > 0 {
		return debugStartLevelOverride
	}
	if v := os.Getenv("TANAT_HUNT_START_LEVEL"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return int32(n)
		}
	}
	return 1
}

// debugWTFMode, when true, is "WTF MODE": AVATAR skills cost no mana and never go
// on cooldown. Tests set it directly; production reads TANAT_WTF_MODE. Mobs are
// untouched -- they have no skill mana/cooldown, and their auto-attack timing is a
// separate path. A pure testing aid for exercising skills without resource limits.
var debugWTFMode bool

// wtfMode reports whether WTF MODE is on (test override or TANAT_WTF_MODE set to a
// truthy value; any non-empty value except "0"/"false" counts as on).
func wtfMode() bool {
	if debugWTFMode {
		return true
	}
	v := os.Getenv("TANAT_WTF_MODE")
	return v != "" && v != "0" && !strings.EqualFold(v, "false")
}

// skillManaCost / skillCooldown gate an avatar skill's mana cost and cooldown
// through WTF MODE: both collapse to 0 when it is on, so casts are free and instant.
func skillManaCost(cost float64) float64 {
	if wtfMode() {
		return 0
	}
	return cost
}

func skillCooldown(cd float64) float64 {
	if wtfMode() {
		return 0
	}
	return cd
}

// Hunt-mode battle. The registration chain (GAME_DATA -> PROTOTYPE_INFO ->
// PLAYER_REG -> CREATE_OBJECT -> SET_AVATAR -> first POSITION sync) fires
// SelfPlayer.Init -> BattleScreen.ShowGui exactly as in the lobby; on top of it
// this file builds the authored per-avatar skill kit (effectors, buff protos,
// summon protos, mob attack protos) and dispatches DO_ACTION to the effect
// engine (effects.go), while the combat tick (mobai.go) drives mobs, statuses,
// summons, traps, channels, death and regen.

// SyncType bits (TanatKernel.SyncType). POSITION carries 5 floats, POS_ANGLE
// 3; TEAM/SILENCE/immunities are int32; every other type is a single float.
const (
	syncPosition    uint64 = 0x1
	syncHealth      uint64 = 0x2
	syncMana        uint64 = 0x4
	syncExperience  uint64 = 0x8
	syncDmgMin      uint64 = 0x20
	syncDmgMax      uint64 = 0x40
	syncAttackSpeed uint64 = 0x80
	syncAttackRange uint64 = 0x800
	syncMaxHealth   uint64 = 0x1000
	syncHealthRegen uint64 = 0x2000
	syncSilence     uint64 = 0x8000000
	syncPhysArmor   uint64 = 0x40000
	syncMagicArmor  uint64 = 0x80000
	syncMaxMana     uint64 = 0x100000
	syncManaRegen   uint64 = 0x200000
	syncSpeed       uint64 = 0x400000
	syncTeam        uint64 = 0x1000000
	syncCastCost    uint64 = 0x100000000
	syncCastStr     uint64 = 0x200000000
	syncCastCd      uint64 = 0x400000000
	syncRadius      uint64 = 0x800000000 // SyncType.RADIUS: body/collision radius
	syncSpellPower  uint64 = 0x1000000000
)

// ---- battle prototype ids ----
//
// One id space per battle connection: the avatar object prototype (100+),
// mob prototypes (500+), mob/summon attack effect prototypes (699/700+),
// summon unit prototypes (800+) and the avatar's effect prototypes (1000+).

func avatarProtoID(avatarID int32) int32 { return 100 + avatarID }
func mobProtoID(mobIdx int) int32        { return 500 + int32(mobIdx) }

const summonAttackProtoID int32 = 699

func mobAttackProtoID(mobIdx int) int32 { return 700 + int32(mobIdx) }
func summonProtoID(i int) int32         { return 800 + int32(i) }

// Summon unit prototypes are PARTY-WIDE stable ids, assigned once per distinct
// summon prefab across ALL avatars -- not per connection. A teammate renders the
// owner's summoned unit, so the model's prototype id must mean the same prefab on
// every client (the old per-conn 800+i counter mapped id 800 to whichever prefab a
// given player's kit summoned first, so two different avatars' summons collided).
// This mirrors mob prototype ids, which are globally stable via the gamedata index.
type summonProto struct {
	id     int32
	prefab string
	desc   string
}

var (
	summonProtoOnce  sync.Once
	summonProtoList  []summonProto
	summonProtoByKey map[string]int32
)

func buildSummonProtos() {
	summonProtoByKey = map[string]int32{}
	for _, a := range gamedata.Avatars() {
		for _, sk := range gamedata.SkillsFor(a).Skills {
			for _, op := range collectSummonOps(sk.Ops) {
				if _, seen := summonProtoByKey[op.Unit]; seen {
					continue
				}
				id := summonProtoID(len(summonProtoList))
				summonProtoByKey[op.Unit] = id
				summonProtoList = append(summonProtoList, summonProto{
					id: id, prefab: op.Unit, desc: summonUnitProtoDesc(op.Unit, op.HP.At(1)),
				})
			}
		}
	}
}

// summonProtos returns every distinct summon-unit prototype (stable order/ids), so
// every member's world-state chain can register the WHOLE set -- like mob protos.
func summonProtos() []summonProto {
	summonProtoOnce.Do(buildSummonProtos)
	return summonProtoList
}

// summonProtoIDFor maps a summon prefab to its party-wide prototype id.
func summonProtoIDFor(prefab string) (int32, bool) {
	summonProtoOnce.Do(buildSummonProtos)
	id, ok := summonProtoByKey[prefab]
	return id, ok
}

func effBase(a gamedata.Avatar) int32               { return 1000 + a.ID*100 }
func skillProtoID(a gamedata.Avatar, i int) int32   { return effBase(a) + int32(i) }      // i = 1..4
func activeProtoID(a gamedata.Avatar, i int) int32  { return effBase(a) + 10 + int32(i) } // i = 1..4
func paramsProtoID(a gamedata.Avatar) int32         { return effBase(a) + 20 }
func attackProtoID(a gamedata.Avatar) int32         { return effBase(a) + 30 }
func buffProtoID(a gamedata.Avatar, slot int) int32 { return effBase(a) + 40 + int32(slot) }

// ---- shared packet builders ----

// protoInfoPkt builds a PROTOTYPE_INFO packet (id + XML desc). Registered up front
// so the client knows an object/effect prototype before it is instantiated.
func protoInfoPkt(id int32, desc string) battleproto.Packet {
	return battleproto.Packet{Cmd: battleproto.CmdPrototypeInfo, Args: amf.NewArray().
		Set("id", id).Set("desc", desc)}
}

// addEffectorArgs builds the ADD_EFFECTOR arg map (the six-key wire contract:
// id/proto/owner/parent/start/args). A nil args defaults to an empty array.
func addEffectorArgs(id, proto, owner, parent int32, start float64, args *amf.MixedArray) *amf.MixedArray {
	if args == nil {
		args = amf.NewArray()
	}
	return amf.NewArray().
		Set("id", id).Set("proto", proto).Set("owner", owner).
		Set("parent", parent).Set("start", start).Set("args", args)
}

// newActionArgs builds the ACTION packet shared by avatar swings/casts and
// mob/summon/boss swings: acting object, action proto, target, start time,
// item=false, and a targetPos point. Callers build targetPos (a cast point, or
// {0,0} for a plain swing) and pick the emit path (pushAvatarAllLocked for avatars,
// broadcastObjLocked for mobs/summons).
func newActionArgs(objID, action, targetObj int32, start float64, targetPos *amf.MixedArray) *amf.MixedArray {
	return amf.NewArray().
		Set("id", objID).Set("action", action).Set("targetObj", targetObj).
		Set("start", start).Set("item", false).Set("targetPos", targetPos)
}

// skillSlotByProto maps a DO_ACTION/UPGRADE_SKILL action id back to the skill
// slot (1..4), or 0 if it is not one of the avatar's skill prototypes.
func skillSlotByProto(a gamedata.Avatar, proto int32) int {
	d := proto - effBase(a)
	if d >= 1 && d <= 4 {
		return int(d)
	}
	return 0
}

// paramsLevelsAttr gates the PARAMS/boost effector (the non-skill stat-boost
// button): 4 ranks, one per early level. Unchanged by the 5-rank skill work.
const paramsLevelsAttr = "-1;1;2;3"

// isUltSlot reports whether a 1-based skill slot is the ult (slot 4). The ult has
// 4 ranks gated to avatar levels 5/10/15/20; slots 1-3 have 5 ranks. NOTHING is
// granted for free -- every skill starts UNLEARNED at rank 0 and must be bought
// with a skill point (the player begins with 1 point at level 1).
func isUltSlot(slot int) bool { return slot == 4 }

// skillLevelsAttr builds the client `levels` gating array: entry N = 0-based
// avatar level required to buy rank N+1 (no -1 entries => no free ranks). Slots
// 1-3: rank r needs 0-based level r-1 (so rank 1 is buyable from the start with a
// point). The ult needs 5r-1 (0-based 4/9/14/19) for rank r.
func skillLevelsAttr(slot, maxRank int) string {
	parts := make([]string, maxRank)
	for r := 1; r <= maxRank; r++ {
		if isUltSlot(slot) {
			parts[r-1] = itoa(5*r - 1)
		} else {
			parts[r-1] = itoa(r - 1)
		}
	}
	return strings.Join(parts, ";")
}

// skillStartRank is the rank a skill begins the battle at (= the count of leading
// -1 entries in its levels array). No skill has a free rank now, so all start at 0.
func skillStartRank(slot int) int { return 0 }

// skillReqLevel is the 0-based avatar level required to raise a skill from curRank
// to curRank+1. Mirrors skillLevelsAttr (server-side gate). No -1 => a point is
// always needed; slots 1-3 rank 1 (curRank 0) needs level 0 (buyable from start).
func skillReqLevel(slot, curRank int) int {
	if isUltSlot(slot) {
		return 5*(curRank+1) - 1
	}
	return curRank
}

// ---- prototype desc XML builders ----

func xmlEsc(s string) string {
	r := strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;")
	return r.Replace(s)
}

func ftoa(f float64) string { return fmt.Sprintf("%g", f) }
func itoa(i int) string     { return fmt.Sprintf("%d", i) }

// xpAttr renders the PExperiencer XP value the client reads. The client
// (GameData.GetExp) indexes this array by the CURRENT level to get the NEXT
// level's XP requirement: nextLvlExp = mLevelsXP[level]. So entry 0 must be the
// XP to reach level 1, entry 1 the XP to reach level 2, etc. -- i.e. the
// cumulative thresholds WITHOUT AvatarXPLevels' leading 0 (that leading 0 made
// the client show "xp/0" at level 0). The server's own level-up logic keeps the
// leading-0 array (grantXPLocked indexes levels[level+1]); only the wire value drops it.
func xpAttr() string {
	levels := gamedata.AvatarXPLevels[1:]
	parts := make([]string, len(levels))
	for i, v := range levels {
		parts[i] = ftoa(v)
	}
	return strings.Join(parts, ";")
}

func avatarProtoDesc(a gamedata.Avatar) string {
	return `<Proto>` +
		`<PPrefab value="` + xmlEsc(a.Prefab) + `"/>` +
		`<PAvatar value="true"/>` +
		`<PDesc>` +
		`<Name value="` + xmlEsc(a.Name()) + `"/>` +
		`<Short value="` + xmlEsc(a.Short()) + `"/>` +
		`<Long value="` + xmlEsc(a.Long()) + `"/>` +
		`<Icon value="` + xmlEsc(a.Icon()) + `"/>` +
		`</PDesc>` +
		`<PDestructible><Health value="` + ftoa(a.Health) + `"/></PDestructible>` +
		`<PCaster><Mana value="` + ftoa(a.Mana) + `"/></PCaster>` +
		`<PExperiencer><XP value="` + xpAttr() + `"/></PExperiencer>` +
		`</Proto>`
}

func mobProtoDesc(m gamedata.Mob) string {
	return `<Proto>` +
		`<PPrefab value="` + xmlEsc(m.Prefab) + `"/>` +
		`<PDesc>` +
		`<Name value="` + xmlEsc(m.NameKey) + `"/>` +
		`<Short value=""/><Long value=""/>` +
		`<Icon value="` + xmlEsc(m.Icon) + `"/>` +
		`</PDesc>` +
		`<PDestructible><Health value="` + ftoa(m.Health) + `"/></PDestructible>` +
		`</Proto>`
}

// summonUnitProtoDesc builds an attackable allied-unit prototype for a summon.
// The summon prefab is a mob model (e.g. Elgorm's guls reuse Mob_ZombieCrawl_01),
// so it borrows that mob's shipped name + enemy-card icon; without them the unit
// card renders blank (the reported "no icon" bug).
func summonUnitProtoDesc(prefab string, hp float64) string {
	name, icon := "", ""
	for _, m := range gamedata.Mobs() {
		if m.Prefab == prefab {
			name, icon = m.NameKey, m.Icon
			break
		}
	}
	return `<Proto>` +
		`<PPrefab value="` + xmlEsc(prefab) + `"/>` +
		`<PDesc><Name value="` + xmlEsc(name) + `"/><Short value=""/><Long value=""/><Icon value="` + xmlEsc(icon) + `"/></PDesc>` +
		`<PDestructible><Health value="` + ftoa(hp) + `"/></PDestructible>` +
		`</Proto>`
}

func effectProtoDesc(nameKey, descKey, icon, skillType, attribs string) string {
	return `<Proto><PEffectDesc>` +
		`<Desc>` +
		`<name value="` + xmlEsc(nameKey) + `"/>` +
		`<short value="` + xmlEsc(descKey) + `"/>` +
		`<long value="` + xmlEsc(descKey) + `"/>` +
		`<icon value="` + xmlEsc(icon) + `"/>` +
		`<type value="` + skillType + `"/>` +
		`</Desc>` +
		`<Attribs>` + attribs + `</Attribs>` +
		`</PEffectDesc></Proto>`
}

func attrItem(name, value string) string {
	return `<item name="` + name + `" value="` + value + `"/>`
}

func attrEnum(name, value string) string {
	return `<enum name="` + name + `" value="` + value + `"/>`
}

// perLevelInts renders a per-rank int array as a ';'-joined string (one entry
// per rank; the client parses these leveled attribs to int and throws on
// fractions). The array LENGTH must match the skill's `levels` array length.
func perLevelInts(vals []int) string {
	parts := make([]string, len(vals))
	for i, v := range vals {
		parts[i] = itoa(v)
	}
	return strings.Join(parts, ";")
}

// skillProtoDesc is the parent SKILL prototype (the visible panel button). Its
// `levels` array (per-slot: 5 ranks for slots 1-3, the level-gated 4-rank ult)
// is what the client's upgrade UI reads to lock/unlock the rank-up button.
func skillProtoDesc(a gamedata.Avatar, sk gamedata.Skill, slot int) string {
	return effectProtoDesc(
		a.SkillTitle(slot), a.SkillDesc(slot), a.SkillIconPath(slot), "SKILL",
		attrItem("levels", skillLevelsAttr(slot, sk.MaxRank())))
}

// activeChildDesc is the castable child bound to a SKILL parent (ACTIVE) or a
// TOGGLE. It carries the targeting rules and per-level costs the client reads.
func activeChildDesc(a gamedata.Avatar, sk gamedata.Skill) string {
	typ := "ACTIVE"
	if sk.Type == "TOGGLE" {
		typ = "TOGGLE"
	} else if sk.Type == "PASSIVE" {
		typ = "PASSIVE"
	}
	target := sk.Target
	if target == "" {
		target = "SELF"
	}
	// `targeting` is parsed client-side against the SkillTargeting enum
	// (NONE=0, SELF=1, TARGET=2, CLAMP=4, NOCLAMP=8) -- a DIFFERENT enum from
	// `target` (SkillTarget). For a POINT skill PlayerControl.ActivateAbility only
	// shows the ground-AoE cursor (mSkillZoneAoe, placed at the clicked point) when
	// TargetingMask == 0; any non-zero value drops it into the line-zone branch that
	// is ATTACHED TO THE AVATAR (mSkillZoneLine.AttachToObj) -- i.e. the AoE follows
	// the caster instead of landing where clicked. So an unspecified targeting must
	// map to NONE(0), NOT SELF(1). (Line/swath skills carry an explicit Targeting;
	// and for AoEWidth>0 the client re-adds TARGET regardless, so this is safe.)
	targeting := sk.Targeting
	if targeting == "" {
		targeting = "NONE"
	}
	// A pure self-cast must fire instantly on keypress with NO target/point
	// selection. The client (PlayerControl.ActivateAbility) only takes the
	// instant-cast branch when TargetValidator.IsNoneTarget(mask) is true, i.e.
	// the target mask is exactly 0 -- a SELF flag (0x200) is treated as "needs a
	// unit target" and drops the client into target-selection. So a self-cast
	// must emit an EMPTY target enum (mask 0), not "SELF". We also zero
	// aoeRadius/aoeWidth (a non-zero value there arms a ground-target cursor); the
	// real AoE lands server-side from the op/skill radius. POINT and unit-target
	// skills keep their target flag and indicator geometry.
	selfCast := target == "SELF"
	aoeRadius, aoeWidth := sk.AoERadius, sk.AoEWidth
	if selfCast {
		target = ""
		aoeRadius, aoeWidth = 0, 0
	}
	attribs := attrEnum("target", target) +
		attrEnum("targeting", targeting) +
		attrItem("distance", itoa(sk.Distance)) +
		attrItem("aoeRadius", itoa(aoeRadius)) +
		attrItem("aoeWidth", itoa(aoeWidth)) +
		attrItem("cooldown", perLevelInts(sk.Cooldown)) +
		attrItem("manacost", perLevelInts(sk.ManaCost)) +
		attrItem("levels", skillLevelsAttr(sk.Slot, sk.MaxRank()))
	return effectProtoDesc(a.SkillTitle(sk.Slot), a.SkillDesc(sk.Slot),
		a.SkillIconPath(sk.Slot), typ, attribs)
}

func paramsProtoDesc(a gamedata.Avatar) string {
	return effectProtoDesc(a.Name(), a.Short(), "", "PARAMS", attrItem("levels", paramsLevelsAttr))
}

// buffProtoDesc is a BUFF-type effector prototype whose icon appears on the
// self buff bar while a self-buff skill is active.
func buffProtoDesc(a gamedata.Avatar, slot int) string {
	return effectProtoDesc(a.SkillTitle(slot), a.SkillDesc(slot),
		a.SkillIconPath(slot), "BUFF", "")
}

func attackProtoDesc(a gamedata.Avatar) string {
	return effectProtoDesc(a.Name(), "", "", "ATTACK",
		attrEnum("target", "ENEMY")+
			attrEnum("targeting", "TARGET")+
			attrItem("distance", ftoa(a.AttackRange)))
}

// unitAttackProtoDesc is a bare ATTACK effector so a mob/summon plays its
// swing animation when the server pushes its ACTION.
func unitAttackProtoDesc() string {
	return effectProtoDesc("", "", "", "ATTACK",
		attrEnum("target", "ENEMY")+attrEnum("targeting", "TARGET")+attrItem("distance", "2"))
}

// ---- hunt battle state (per connection, guarded by conn.mvMu) ----

type huntState struct {
	av   gamedata.Avatar
	m    gamedata.HuntMap
	kit  *gamedata.AvatarSkills
	inst *huntInstance // the shared world this player is a member of
	tr   tracker
	mobs map[int32]*mobState // aliases inst.mobs (shared authoritative set)

	hp, mana float64
	xp       float64
	level    int32 // 0-based (client displays level+1)
	points   int32

	skillLevel    [4]int32   // 1-based skill levels, index = slot-1
	cooldownUntil [4]float64 // battle-time when each skill is ready again
	parentEff     [4]int32   // effector instance ids of the SKILL parents
	childEff      [4]int32   // effector instance ids of the ACTIVE children
	nextEffID     int32
	nextFxUID     int32

	toggleOn        [4]bool
	toggleFx        [4]int32
	toggleNextPulse [4]float64

	// growFx[slot-1] is the live EFFECT_START uid of a passive's per-level model-
	// grow VFX (Titanid "Гигантизм"); ended and restarted a size up on upgrade.
	growFx [4]int32

	// passiveBuffEff[slot-1] is the effector id of a learned PASSIVE's permanent
	// buff-bar icon (skills flagged BuffIcon, e.g. Velial's «Воля к победе»), or 0
	// if none/unlearned. Unlike active buffs, a passive is never cast, so its icon
	// is instantiated at world-build (and re-sent on respawn / refreshed on upgrade)
	// with no "duration" arg, so it shows permanently without a countdown.
	passiveBuffEff [4]int32
	// passiveBuffCount[slot-1] is the last "counter" value pushed on that buff icon
	// (the number drawn beside it -- e.g. Velial's live missing-HP bonus damage).
	// -1 = the passive has no dynamic counter / none sent yet; used to re-send the
	// effector only when the displayed integer actually changes.
	passiveBuffCount [4]int32

	hasProjectile bool
	procs         []procState // rolled when the avatar HITS (basic attack)
	defenseProcs  []procState // rolled when the avatar is STRUCK (Titanid «Каменная кожа»)

	// reviveSlot / ccImmuneSlot are the 1-based slots of a learned PASSIVE carrying an
	// OpRevive / OpImmune op (0 = none). Set at world-build. reviveReadyAt /
	// ccImmuneReadyAt are the battle-times the mechanic is available again after being
	// consumed (0 = ready now).
	reviveSlot      int
	reviveReadyAt   float64
	ccImmuneSlot    int
	ccImmuneReadyAt float64
	// healOnKillSlot is the 1-based slot of a learned PASSIVE carrying an OpHealOnKill
	// (Cerber's «Кровавый пир»), 0 = none. Honored in hitMobLocked's death branch.
	healOnKillSlot int

	st unitStatus // self-avatar status effects

	// auto-attack session (player -> mob)
	attackTarget int32
	attackSeq    int
	// autoResumed marks the current auto-attack as SERVER-driven: it was started by
	// resumeAutoAttackLocked after a cast, when the client has left its DEFENCE state
	// and will NOT re-pick the next enemy on a kill. In this mode the server must
	// drive the whole retarget chain (resume onto the next mob each kill). A
	// client-issued attack (the DEFENCE loop is live) clears it, so normal combat
	// keeps relying on the client's own retargeting and we don't double up.
	autoResumed bool

	// castLockUntil roots the avatar while a skill's cast animation plays: manual
	// MOVE_PLAYER orders are rejected until this battle-time so the player cannot
	// walk out of the cast (the client does not self-lock movement during a cast).
	castLockUntil float64

	// scheduled work processed by the combat tick
	order        *pendingCast
	payloads     []payload
	actionDones  []actionDone
	fxEnds       []fxEnd
	traps        []trapState
	channels     []channelState
	bounces      []bounceState
	summons      map[int32]*summonState
	nextSummonID int32
	summonProtos map[string]int32
	// anchorEnds defers deletion of a trap-fx anchor object until its trigger fx
	// has played out (the anchor must outlive the SELF-mode trigger fx parented to it).
	anchorEnds   []anchorEnd
	nextAnchorID int32 // solo/bare-conn fallback id space (instance uses inst.nextAnchorID)

	// drops (loot chests): solo/bare-conn fallback id space + set (instance uses
	// inst.drops/nextDropID/nextDropItemID).
	drops          map[int32]*dropState
	nextDropID     int32
	nextDropItemID int32

	// Consumable bag: this session's mirror of the hero's persistent Bag
	// (session.Store), loaded once at world-build (sendInitialBagLocked) and kept
	// in sync as potions are drunk/looted.
	bag            map[int32]int32 // articleID -> count
	bagItemID      map[int32]int32 // articleID -> wire inventory "id" (ADD_TO_INVENTORY)
	bagArticleByID map[int32]int32 // reverse of bagItemID
	nextBagID      int32
	sentItemProtos map[int32]bool // article ids already PROTOTYPE_INFO'd this session
	sentBuffProtos map[int32]bool // itemBuffProtoID ids already PROTOTYPE_INFO'd this session
	// itemCooldownUntil gates re-use of EACH specific article id independently
	// -- real per-item cooldowns (gamedata.Item.Cooldown) range 30-40s for
	// Health/Mana potions (tapering by level bracket) up to a flat 150s for
	// every buff-type potion, replacing an earlier simplification that shared
	// one flat 15s cooldown across every potion kind.
	itemCooldownUntil map[int32]float64

	// invisibleUntil / invisFxUID: an active Invisibility potion. While live, mobs
	// ignore this avatar as a target candidate (mobTargetLocked) -- it suppresses
	// NEW aggro, but doesn't retroactively break a mob already mid-chase (a
	// deliberately simple v1 rule, not a full "vanish").
	invisibleUntil float64
	invisFxUID     int32
	invisBuffEffID int32

	// revealInvisibleUntil / revealBuffEffID: an active Revelation potion.
	// Currently a no-op beyond its own buff icon/timer -- Hunt is a co-op PvE
	// mode with no enemy avatars to reveal, and mobs carry no invisibility of
	// their own. Forward-compatible state for whenever PvP or invisible mobs
	// exist. (AntiPhysArmor's phys_armor_pen, by contrast, is now LIVE: mobs
	// carry real physical armor and hitMobFlagsLocked penetrates it. AntiMagicArmor's
	// magic_armor_pen stays inert until mobs get magic armor.)
	revealInvisibleUntil float64
	revealBuffEffID      int32

	// self dash + death/respawn
	dashUntil, dashSpeed float64
	deadUntil, diedAt    float64
	corpseHidden         bool

	// Active respawn checkpoint (m.Reborn). Death returns here; walking near a
	// different Reborn_point moves it. rebornIdx = index of the active checkpoint
	// in m.Reborn (-1 = none/using Spawn()).
	respawnX, respawnY float32
	rebornIdx          int

	// worldReady flips true once the whole world-state chain (game data, self
	// avatar, effectors, cross-player intro) has been sent. Until then this member
	// is excluded from the shared tick and every broadcast fan-out, so the
	// concurrently-running instance ticker can't push mob/teammate packets to a
	// client that is still building its scene.
	worldReady bool

	closed bool
}

type mobState struct {
	id        int32
	mobIdx    int
	mob       gamedata.Mob
	x, y      float32
	vx, vy    float32
	spawnX    float32 // authored spawn point -- respawn returns the mob here
	spawnY    float32
	respawnAt float64 // battle-time to revive a dead mob (0 = not scheduled)
	hp        float64
	dead      bool

	// Level-scaled per-instance stats, computed at spawn from the mob's level-1
	// base and MobSpawn.Level (bosses use their authored values unscaled). Zero
	// means "not spawned through the scaling path" (a directly-constructed test
	// mobState), and the accessors below fall back to the raw Mob fields.
	level  int
	maxHP  float64
	dmgMin float64
	dmgMax float64
	xp     float64
	coins  int32

	st         unitStatus
	aggro      bool
	returning  bool  // leashed: walking home to spawn, regenerating, ignoring targets
	shown      bool  // created on the client (within fog-of-war reveal range)
	shaded     bool  // rendered translucent (outer fog ring) via mobShadeFx
	shadeFxUID int32 // live InvisibilityEffect uid while shaded (0 = none)
	lastSync   float64

	nextSwing   float64
	hitAt       float64
	hitDmg      float64
	hitTarget   int32
	swingDoneAt float64 // when to send ACTION_DONE for the last swing (0 = none pending)
	// projLaunchAt is when a ranged mob (skeleton archer, shooter plant, caster) lets
	// its projectile fly -- a point late in the attack wind-up, NOT its start, so the
	// arrow leaves the bow near the release frame and streaks the last stretch to the
	// target rather than drifting across the whole animation. 0 = none pending.
	projLaunchAt float64
	projTarget   int32

	// boss abilities (Mob.Skills): per-skill next-ready time + the in-flight cast.
	skillReady     []float64
	skillHitAt     float64 // when the casting skill's damage lands (0 = not casting)
	skillDmg       float64
	skillRadius    float64
	skillCX        float32 // AoE centre (where the target was when the cast started)
	skillCY        float32
	skillTargetObj int32 // objID a single-target boss skill was aimed at

	pf pathState // routed chase waypoints when the straight line is wall-blocked

	// «Штурм» (DOTA) fields. team is the object's in-battle team: 1 = the player's
	// side (allies), -1 = enemies (Hunt's convention; teamVal falls back to -1 so a
	// plain Hunt mob keeps hostile-to-player behaviour). structure marks a static
	// building rendered from a Fn_* building prefab (dotaPrefab) instead of a mob
	// model; altar marks the win object; dotaRole classifies it. Creeps are ordinary
	// mobs with a team + a lane to march (lane waypoints, laneIdx = next waypoint,
	// laneFwd = walk direction). dtarget is the object the creep/cannon is currently
	// engaging.
	team      int32
	structure bool
	altar     bool
	dotaRole  gamedata.DotaRole
	dotaPrefab string
	lane      []gamedata.Vec2
	laneIdx   int
	laneFwd   bool
	dtarget   int32
}

// teamVal is the object's sync TEAM value: the explicit DOTA team, or -1 (enemy of
// the player) for a Hunt mob that never set one. Never returns 0 (client NEUTRAL).
func (m *mobState) teamVal() int32 {
	if m.team != 0 {
		return m.team
	}
	return -1
}

// maxHealth / rollDamage / xpReward / coinReward return the mob's effective
// (level-scaled) stats, falling back to the raw Mob fields when maxHP is unset --
// i.e. for test-constructed mobStates that never went through spawn scaling.
func (m *mobState) maxHealth() float64 {
	if m.maxHP > 0 {
		return m.maxHP
	}
	return m.mob.Health
}

func (m *mobState) dmgRange() (lo, hi float64) {
	if m.maxHP > 0 {
		return m.dmgMin, m.dmgMax
	}
	return float64(m.mob.DmgMin), float64(m.mob.DmgMax)
}

func (m *mobState) rollDamage() float64 {
	lo, hi := m.dmgRange()
	return lo + rand.Float64()*(hi-lo)
}

func (m *mobState) xpReward() float64 {
	if m.maxHP > 0 {
		return m.xp
	}
	return m.mob.XP
}

func (m *mobState) coinReward() int32 {
	if m.maxHP > 0 {
		return m.coins
	}
	return m.mob.Coins
}

// physArmor is the mob's effective physical armor at `now`: its authored flat base
// (Mob.PhysArmor -- NOT level-scaled, since mitigation is already a percentage curve)
// plus any active phys_armor status mods, times the armor_pct multiplier. Velial's
// ult «Трибунал» appends a NEGATIVE phys_armor mod here, so a debuffed mob's armor
// drops (and can go below 0 = armor broken, which armorMitigation turns into a damage
// amplifier). Consumed on the incoming-damage path (hitMobFlagsLocked).
func (m *mobState) physArmor(now float64) float64 {
	return (m.mob.PhysArmor + m.st.modSum(now, "phys_armor")) * m.st.modMul(now, "armor_pct")
}

func (hs *huntState) newEffID() int32                  { hs.nextEffID++; return hs.nextEffID }
func (hs *huntState) skillDef(slot int) gamedata.Skill { return hs.kit.Skills[slot-1] }

// sendHuntWorldState pushes the hunt battle's initial state and starts the
// combat tick.
func (s *Server) sendHuntWorldState(c *conn, name string) {
	pb := c.hunt
	a, ok := gamedata.AvatarByID(pb.AvatarID)
	if !ok {
		log.Printf("battle: %s hunt avatar %d not in roster, using fallback", c.RemoteAddr(), pb.AvatarID)
		a = gamedata.Avatars()[0]
	}
	m, _ := gamedata.HuntMapByID(pb.MapID)
	kit := gamedata.SkillsFor(a)

	hs := &huntState{
		av:  a,
		m:   m,
		kit: kit,
		// Share the instance's authoritative mob set: every member of the world sees
		// the same mobs (one HP pool, killable by anyone). Go maps are references, so
		// this alias keeps all the existing hs.mobs[...] code working unchanged.
		mobs:          c.inst.mobs,
		summons:       map[int32]*summonState{},
		summonProtos:  map[string]int32{},
		hp:            a.Health,
		mana:          a.Mana,
		hasProjectile: kit.AttackProjectile,
	}
	hs.inst = c.inst
	// Every skill starts UNLEARNED at rank 0; the player begins with 1 skill point
	// (level 1) so exactly one skill can be raised to rank 1 right away. The ult
	// still can't be ranked until avatar level 5.
	for i := range hs.skillLevel {
		hs.skillLevel[i] = int32(skillStartRank(i + 1))
	}
	hs.points = 1
	now := float64(s.battleTime())

	// Debug testing aid: optionally spawn at a higher level so the level-gated
	// ranks (e.g. the ult at level 5) are reachable immediately. Set hs.level
	// FIRST -- maxHP/maxMana scale with it -- then top the pools up and hand out
	// the skill points that leveling would have granted (1 at level 1, +1 each
	// level after).
	if lvl := huntStartLevel(); lvl > 1 {
		levels := gamedata.AvatarXPLevels
		if int(lvl) > len(levels) {
			lvl = int32(len(levels))
		}
		hs.level = lvl - 1       // wire/internal level is 0-based
		hs.xp = levels[hs.level] // XP floor of that level (bar starts empty into the next)
		hs.points = lvl          // 1 initial + one per level gained
		hs.hp = hs.maxHPLocked(now)
		hs.mana = hs.maxManaLocked(now)
		log.Printf("battle: %s DEBUG spawn level %d (points %d)", c.RemoteAddr(), lvl, hs.points)
	}

	self := c.selfPlayerID
	objID := c.objID
	protoID := avatarProtoID(a.ID)
	bt := s.battleTime()
	c.lock()
	sx, sy := c.x, c.y
	// Start checkpoint = the battle-start Reborn_point (the spawn). The first tick
	// re-activates whichever Reborn the player is standing on.
	hs.respawnX, hs.respawnY = c.x, c.y
	hs.rebornIdx = -1
	c.huntState = hs
	c.unlock()

	// 1. Static data + prototypes.
	pkts := []battleproto.Packet{
		{Cmd: battleproto.CmdGameData, Args: amf.NewArray().
			Set("data", `<root><battle time_limit="0" frag_limit="0"/></root>`).
			Set("relics", amf.NewArray())},
		protoInfoPkt(protoID, avatarProtoDesc(a)),
		protoInfoPkt(paramsProtoID(a), paramsProtoDesc(a)),
	}
	for i := 1; i <= 4; i++ {
		sk := kit.Skills[i-1]
		pkts = append(pkts,
			protoInfoPkt(skillProtoID(a, i), skillProtoDesc(a, sk, i)),
			protoInfoPkt(activeProtoID(a, i), activeChildDesc(a, sk)))
		if sk.BuffIcon {
			pkts = append(pkts, protoInfoPkt(buffProtoID(a, i), buffProtoDesc(a, i)))
		}
	}
	pkts = append(pkts, protoInfoPkt(attackProtoID(a), attackProtoDesc(a)))

	// Mob prototypes + a shared attack-effector proto per mob type.
	mobTypes := map[int]bool{}
	for _, sp := range m.Spawns {
		if !mobTypes[sp.Mob] {
			mobTypes[sp.Mob] = true
			pkts = append(pkts,
				protoInfoPkt(mobProtoID(sp.Mob), mobProtoDesc(gamedata.Mobs()[sp.Mob])),
				protoInfoPkt(mobAttackProtoID(sp.Mob), unitAttackProtoDesc()))
		}
	}

	// Summon unit + attack prototypes. Summon models are shared: a teammate renders
	// the owner's summon, so every member registers the WHOLE party-wide set with the
	// same stable ids (like mob protos), not just its own avatar's units.
	pkts = append(pkts, protoInfoPkt(summonAttackProtoID, unitAttackProtoDesc()))
	for _, sp := range summonProtos() {
		hs.summonProtos[sp.prefab] = sp.id
		pkts = append(pkts, protoInfoPkt(sp.id, sp.desc))
	}

	// Invisible trap-fx anchor prototype: a stationary object a SELF-mode ground fx
	// parents to so it holds the cast point (see trap_anchor.go). One fixed proto.
	pkts = append(pkts, protoInfoPkt(trapAnchorProtoID, trapAnchorProtoDesc()))
	// Ground-loot chest prototype (see drops.go). One fixed proto, spawned wherever
	// a mob happens to drop loot.
	pkts = append(pkts, protoInfoPkt(dropChestProtoID, dropChestProtoDesc()))
	// Potion buff-bar icons are no longer eagerly registered here: each item
	// now gets its OWN dedicated buff prototype, lazily PROTOTYPE_INFO'd the
	// first time it's actually drunk (ensureItemBuffProtoLocked in
	// consumables.go) -- see itemBuffProtoID's doc for why (per-tier icon,
	// not one shared placeholder per Kind).
	// Shared consumable-use action prototype every item's <PTool> references
	// (see itemProtoDesc's doc) -- without it, clicking any bag item is a
	// total client-side no-op: no DO_ACTION packet is ever sent.
	pkts = append(pkts, protoInfoPkt(itemUseActionProtoID, itemUseActionProtoDesc()))

	// 2. Self player + avatar object. SET_AVATAR level is 0-based on the wire.
	c.lock()
	avatarIdx := hs.tr.add(objID)
	c.unlock()
	pkts = append(pkts,
		battleproto.Packet{Cmd: battleproto.CmdPlayerReg, Args: amf.NewArray().
			Set("id", self).Set("name", name).Set("team", int32(1)).Set("avatar", a.ID)},
		battleproto.Packet{Cmd: battleproto.CmdCreateObject, Args: amf.NewArray().
			Set("id", objID).Set("proto", protoID)},
		battleproto.Packet{Cmd: battleproto.CmdSetAvatar, Args: amf.NewArray().
			Set("playerID", self).Set("avatarID", objID).Set("level", hs.level).Set("points", hs.points)},
		battleproto.Packet{Cmd: battleproto.CmdSync, Args: amf.NewArray().
			Set("data", newSyncBlob(bt).addObject(objID).
				position(avatarIdx, sx, sy, 0, 0, bt).build(1))},
		battleproto.Packet{Cmd: battleproto.CmdSync, Args: amf.NewArray().
			Set("data", newSyncBlob(bt).
				setFloats(syncHealth, avatarIdx, 1.0).
				setFloats(syncMana, avatarIdx, 1.0).
				setFloats(syncExperience, avatarIdx, float32(hs.xp)).
				setFloats(syncDmgMin, avatarIdx, float32(a.DmgMin)).
				setFloats(syncDmgMax, avatarIdx, float32(a.DmgMax)).
				setFloats(syncAttackSpeed, avatarIdx, float32(a.AttackSpeed)).
				setFloats(syncAttackRange, avatarIdx, float32(a.AttackRange)).
				setFloats(syncMaxHealth, avatarIdx, float32(a.Health)).
				setFloats(syncHealthRegen, avatarIdx, float32(a.HealthRegen)).
				setFloats(syncPhysArmor, avatarIdx, float32(a.PhysArmor)).
				setFloats(syncMagicArmor, avatarIdx, float32(a.MagicArmor)).
				setFloats(syncMaxMana, avatarIdx, float32(a.Mana)).
				setFloats(syncManaRegen, avatarIdx, float32(a.ManaRegen)).
				setFloats(syncSpeed, avatarIdx, lobbyMoveSpeed).
				setFloats(syncRadius, avatarIdx, float32(a.Radius())).
				setFloats(syncSpellPower, avatarIdx, float32(a.SpellPower)).
				setFloats(syncCastCost, avatarIdx, 1.0).
				setFloats(syncCastStr, avatarIdx, 1.0).
				setFloats(syncCastCd, avatarIdx, 1.0).
				build(1))},
	)

	// 3. Effectors: PARAMS + 4 (SKILL parent + ACTIVE/TOGGLE/PASSIVE child) +
	// ATTACK. Tooltip tokens ride the child's args so the panel tips resolve.
	c.lock()
	paramsEff := hs.newEffID()
	type skillPair struct{ parent, child int32 }
	var pairs [4]skillPair
	for i := 0; i < 4; i++ {
		pairs[i] = skillPair{hs.newEffID(), hs.newEffID()}
		hs.parentEff[i] = pairs[i].parent
		hs.childEff[i] = pairs[i].child
	}
	attackEff := hs.newEffID()
	c.unlock()

	addEff := func(id, proto, parent int32, args *amf.MixedArray) {
		pkts = append(pkts, battleproto.Packet{Cmd: battleproto.CmdAddEffector,
			Args: addEffectorArgs(id, proto, objID, parent, now, args)})
	}
	addEff(paramsEff, paramsProtoID(a), -1, nil)
	for i := 0; i < 4; i++ {
		addEff(pairs[i].parent, skillProtoID(a, i+1), -1, nil)
		// Skills ship UNLEARNED at their start rank (0) -- the player buys rank 1 with
		// the starting skill point; the client shows them as rank 0 / upgradeable.
		addEff(pairs[i].child, activeProtoID(a, i+1), pairs[i].parent, childArgs(kit.Skills[i], int32(skillStartRank(i+1))))
	}
	addEff(attackEff, attackProtoID(a), -1, nil)

	// 4. Mobs live on the shared instance (seeded in newHuntInstance); this member
	// just aliases that set (hs.mobs above). The combat tick's fog-of-war pass
	// creates each mob on THIS client lazily once the player comes within
	// mobRevealRadius (revealMobLocked), so distant enemies aren't visible.
	c.lock()
	// Register on-hit passives (procs) for the tick/attack engine, plus the two
	// special passive mechanics with no op-execution path of their own: an
	// auto-revive (OpRevive) and a CC-immunity (OpImmune), honored in the death /
	// player-CC gates by their remembered slot.
	for i, sk := range kit.Skills {
		if sk.Type != "PASSIVE" {
			continue
		}
		for _, op := range sk.Ops {
			switch op.Kind {
			case gamedata.OpProc:
				pr := procState{slot: i + 1, chance: op.Chance, ops: op.Ops}
				// Most procs fire when the avatar HITS (runProcsLocked, basic-attack path).
				// A defensive proc like Titanid's «Каменная кожа» instead hardens when the
				// avatar is STRUCK, so it is rolled from the incoming-damage path.
				if procOnDamaged(hs.av.Prefab, i+1) {
					hs.defenseProcs = append(hs.defenseProcs, pr)
				} else {
					hs.procs = append(hs.procs, pr)
				}
			case gamedata.OpRevive:
				hs.reviveSlot = i + 1
			case gamedata.OpImmune:
				hs.ccImmuneSlot = i + 1
			case gamedata.OpHealOnKill:
				hs.healOnKillSlot = i + 1
			}
		}
	}
	// Apply permanent passive self-buffs immediately. Values track the skill's
	// current level (upgrades re-derive them in reapplyPassiveLocked). A passive
	// with a GrowFx (Titanid "Гигантизм") also gets its per-level model-grow VFX
	// started here -- appended to pkts (not pushed) so it ships AFTER the avatar's
	// CREATE_OBJECT, in order, giving the client an object to attach+scale.
	for i, sk := range kit.Skills {
		if sk.Type != "PASSIVE" {
			continue
		}
		level := int(hs.skillLevel[i])
		if level < 1 { // unlearned passive (rank-0 ult slot): apply nothing yet
			continue
		}
		for _, op := range sk.Ops {
			if op.Kind == gamedata.OpBuffStat && op.On != "target" && op.Dur.At(1) == 0 {
				hs.st.mods = append(hs.st.mods, statMod{
					stat: op.Stat, value: op.Value.At(level), until: 0, src: "passive" + itoa(i+1)})
			}
		}
		// A learned passive flagged BuffIcon shows a permanent status-effect icon.
		// The buff proto was registered above; instantiate its effector now (appended
		// after CREATE_OBJECT so the client has the avatar to attach it to).
		if sk.BuffIcon {
			hs.passiveBuffEff[i] = hs.newEffID()
			hs.passiveBuffCount[i] = passiveBuffCountOrNone(hs, i+1, level, now)
			pkts = append(pkts, battleproto.Packet{Cmd: battleproto.CmdAddEffector,
				Args: addEffectorArgs(hs.passiveBuffEff[i], buffProtoID(a, i+1), objID, -1, now,
					hs.passiveBuffArgs(i+1, level, now))})
		}
		if sk.GrowFx != "" {
			// Allocate from the WORLD fx space in an instance: this EFFECT_START is
			// pushed only to the self batch (self isn't in memberList() until worldReady
			// below), but its uid is later ended via fxEndLocked->worldFxEndLocked
			// (reapplyPassiveLocked on UPGRADE_SKILL), which broadcasts EFFECT_END to
			// every member. A per-conn uid (small, e.g. 1) would collide with a
			// teammate's own low uid and tear down THEIR fx; a world uid (>=1<<20) is
			// globally unique so the broadcast end is a harmless no-op on teammates.
			if c.inst != nil {
				c.inst.nextFxUID++
				hs.growFx[i] = c.inst.nextFxUID
			} else {
				hs.nextFxUID++
				hs.growFx[i] = hs.nextFxUID
			}
			pkts = append(pkts, battleproto.Packet{Cmd: battleproto.CmdEffectStart, Args: amf.NewArray().
				Set("effect", hs.growFx[i]).
				Set("owner", objID).
				Set("fx", sk.GrowFx+itoa(level)).
				Set("args", amf.NewArray())})
		}
	}
	c.unlock()

	log.Printf("battle: %s sending HUNT world state (self=%d obj=%d avatar=%s map=%d mobs=%d proj=%v)",
		c.RemoteAddr(), self, objID, a.Prefab, pb.MapID, len(m.Spawns), hs.hasProjectile)
	s.sendSeq(c, pkts)

	// Consumable bag: register this hero's persisted potions as Battle-channel
	// inventory so a hunt session starts with the same bag the lobby shows.
	c.lock()
	s.sendInitialBagLocked(c)
	c.unlock()

	// Cross-player rendering: show the existing members to this newcomer and this
	// newcomer to them, so a shared world actually looks shared. The instance
	// ticker (started at instance creation) drives the shared simulation -- no
	// per-connection combat goroutine any more.
	c.lock()
	s.introduceMemberLocked(c, float64(s.battleTime()))
	// «Штурм»: register the structure/creep prototypes and render every base structure
	// (altars, cannons, towers, generators) on this member's client. Must run before
	// worldReady so the first DOTA tick's creep syncs don't race the scene build.
	if c.inst != nil && c.inst.dota != nil {
		s.dotaWorldSetupLocked(c, float64(s.battleTime()))
	}
	// The world is fully built: join the shared tick + broadcast set now, so no
	// mob/teammate packet could have raced ahead of this member's scene load.
	hs.worldReady = true
	c.unlock()
}

// collectSummonOps gathers every summon op (including nested in proc/trap/etc).
func collectSummonOps(ops []gamedata.Op) []gamedata.Op {
	var out []gamedata.Op
	for _, op := range ops {
		if op.Kind == gamedata.OpSummon {
			out = append(out, op)
		}
		out = append(out, collectSummonOps(op.Ops)...)
	}
	return out
}

// childArgs builds an ACTIVE child's ADD_EFFECTOR args: the skill level plus the
// tooltip tokens as level-indexed arrays (5 entries: the client indexes leveled
// token arrays by the 1-based level, so pad the front to avoid an off-by-one).
func childArgs(sk gamedata.Skill, level int32) *amf.MixedArray {
	args := amf.NewArray().Set("level", level)
	maxRank := sk.MaxRank()
	for k, v := range sk.TipArgs {
		arr := amf.NewArray()
		arr.Add(v.At(1)) // index 0 (unused by the 1-based lookup)
		for l := 1; l <= maxRank; l++ {
			arr.Add(v.At(l))
		}
		args.Set(k, arr)
	}
	return args
}

// passiveBuffArgs builds the ADD_EFFECTOR args for a permanent passive buff-bar
// icon: the current level, the skill's tooltip tokens at that level (so the hover
// tip resolves), and -- for a passive whose payload scales live (Velial's «Воля к
// победе») -- a "counter" holding the current bonus, drawn as the number beside the
// icon. Deliberately omits "duration": the client only draws a countdown/radial when
// a duration is present, and a passive never expires.
func (hs *huntState) passiveBuffArgs(slot, level int, now float64) *amf.MixedArray {
	sk := hs.kit.Skills[slot-1]
	args := amf.NewArray().Set("level", int32(level))
	for k, v := range sk.TipArgs {
		args.Set(k, v.At(level))
	}
	if cnt, ok := hs.passiveBuffCounterLocked(slot, level, now); ok {
		args.Set("counter", cnt)
	}
	return args
}

// passiveBuffCounterLocked returns the live number to draw beside a passive's buff
// icon and whether the passive has a dynamic one. Currently only «Воля к победе» (a
// CasterMissingHP proc): the current bonus damage = coeff × Velial's missing-HP
// fraction, matching the value it actually adds on the next hit.
func (hs *huntState) passiveBuffCounterLocked(slot, level int, now float64) (int32, bool) {
	coeff := passiveCasterMissingHP(hs.kit.Skills[slot-1], level)
	if coeff <= 0 {
		return 0, false
	}
	maxHP := hs.maxHPLocked(now)
	if maxHP <= 0 {
		return 0, true
	}
	missing := 1 - hs.hp/maxHP
	if missing < 0 {
		missing = 0
	}
	return int32(math.Round(coeff * missing)), true
}

// passiveBuffCountOrNone returns the current counter for the slot's buff icon, or -1
// if the passive has no dynamic counter -- the sentinel stored in passiveBuffCount so
// the live refresh only re-sends the effector when a real, changed number exists.
func passiveBuffCountOrNone(hs *huntState, slot, level int, now float64) int32 {
	if cnt, ok := hs.passiveBuffCounterLocked(slot, level, now); ok {
		return cnt
	}
	return -1
}

// passiveCasterMissingHP finds a passive's CasterMissingHP coefficient at a level by
// scanning its ops (a proc wraps the damage op), or 0 if it has none.
func passiveCasterMissingHP(sk gamedata.Skill, level int) float64 {
	var scan func(ops []gamedata.Op) float64
	scan = func(ops []gamedata.Op) float64 {
		for _, op := range ops {
			if v := op.CasterMissingHP.At(level); v > 0 {
				return v
			}
			if v := scan(op.Ops); v > 0 {
				return v
			}
		}
		return 0
	}
	return scan(sk.Ops)
}

// sendEffectorsLocked re-sends the whole effector set (used on respawn, since
// the client drops all effectors of an object that left relevance).
func (s *Server) sendEffectorsLocked(c *conn, now float64) {
	hs := c.huntState
	a := hs.av
	add := func(id, proto, parent int32, args *amf.MixedArray) {
		s.push(c, battleproto.CmdAddEffector, addEffectorArgs(id, proto, c.objID, parent, now, args))
	}
	paramsEff := hs.newEffID()
	add(paramsEff, paramsProtoID(a), -1, nil)
	for i := 0; i < 4; i++ {
		p := hs.newEffID()
		ch := hs.newEffID()
		hs.parentEff[i] = p
		hs.childEff[i] = ch
		add(p, skillProtoID(a, i+1), -1, nil)
		add(ch, activeProtoID(a, i+1), p, childArgs(hs.kit.Skills[i], hs.skillLevel[i]))
		// Re-instantiate a learned passive's permanent buff-bar icon (dropped with the
		// rest of the object's effectors when it left relevance).
		sk := hs.kit.Skills[i]
		if sk.Type == "PASSIVE" && sk.BuffIcon && hs.skillLevel[i] >= 1 {
			lvl := int(hs.skillLevel[i])
			hs.passiveBuffEff[i] = hs.newEffID()
			hs.passiveBuffCount[i] = passiveBuffCountOrNone(hs, i+1, lvl, now)
			add(hs.passiveBuffEff[i], buffProtoID(a, i+1), -1, hs.passiveBuffArgs(i+1, lvl, now))
		} else {
			hs.passiveBuffEff[i] = 0
		}
	}
	add(hs.newEffID(), attackProtoID(a), -1, nil)
}

// ---- combat pushes ----

func (s *Server) push(c *conn, cmd battleproto.CmdID, args *amf.MixedArray) {
	if err := c.send(battleproto.Packet{Cmd: cmd, Args: args, RequestID: -1, Status: true}); err != nil {
		log.Printf("battle: %s push %s error: %v", c.RemoteAddr(), cmd.Name(), err)
	}
}

func (s *Server) syncMobHealthLocked(c *conn, ms *mobState) {
	frac := float32(ms.hp / ms.maxHealth())
	if frac < 0 {
		frac = 0
	}
	s.broadcastStatLocked(c, ms.id, syncHealth, frac, s.battleTime())
}

// syncSelfLocked pushes fraction/absolute stats of the self avatar.
func (s *Server) syncSelfLocked(c *conn, types ...uint64) {
	hs := c.huntState
	idx := hs.tr.index(c.objID)
	if idx < 0 {
		return
	}
	now := s.battleTime()
	nowf := float64(now)
	hpFrac := float32(hs.hp / hs.maxHPLocked(nowf))
	b := newSyncBlob(now)
	for _, typ := range types {
		switch typ {
		case syncHealth:
			b.setFloats(syncHealth, idx, hpFrac)
		case syncMana:
			b.setFloats(syncMana, idx, float32(hs.mana/hs.maxManaLocked(nowf)))
		case syncExperience:
			b.setFloats(syncExperience, idx, float32(hs.xp))
		}
	}
	s.push(c, battleproto.CmdSync, amf.NewArray().Set("data", b.build(hs.tr.count())))

	// Health is the only self stat teammates render (the ally HP bar); mana/XP are
	// private HUD. Fan the health fraction out to the other members, each with its
	// own tracking index for this avatar (self already got the full blob above).
	for _, typ := range types {
		if typ != syncHealth {
			continue
		}
		c.mobViewersLocked(c.objID, func(mem *conn, oidx, count int) {
			if mem == c {
				return
			}
			s.push(mem, battleproto.CmdSync, amf.NewArray().Set("data",
				newSyncBlob(now).setFloats(syncHealth, oidx, hpFrac).build(count)))
		})
		break
	}
}

// syncSelfIntLocked pushes one int32-encoded self sync (SILENCE etc.).
func (s *Server) syncSelfIntLocked(c *conn, typ uint64, v int32) {
	hs := c.huntState
	idx := hs.tr.index(c.objID)
	if idx < 0 {
		return
	}
	now := s.battleTime()
	s.push(c, battleproto.CmdSync, amf.NewArray().
		Set("data", newSyncBlob(now).setInt(typ, idx, v).build(hs.tr.count())))
}

// ---- DO_ACTION: auto-attack, skills, toggles ----

func (s *Server) handleDoAction(c *conn, p battleproto.Packet) {
	if c.hunt == nil || c.huntState == nil {
		s.ack(c, p)
		return
	}
	itemID := p.Args.IntOr("id", -1)
	action := p.Args.IntOr("action", -1)
	target := p.Args.IntOr("target", -1)
	var px, py float32
	hasPos := false
	if tp, ok := p.Args.GetArray("targetPos"); ok {
		x, _ := tp.GetFloat("x")
		y, _ := tp.GetFloat("y")
		px, py, hasPos = float32(x), float32(y), true
	}
	s.ack(c, p) // DoActionArgParser reads id/action/target from its own request

	c.lock()
	defer c.unlock()
	hs := c.huntState
	a := hs.av

	switch {
	case action == -1:
		// SelfPlayer.UseItem sends {id: itemObjId, action: -1, target: avatarObjId}:
		// this is a consumable use, not an attack/skill (those carry a real action
		// proto id). itemID indexes hs.bagArticleByID, not any object/proto space.
		s.useItemLocked(c, itemID, float64(s.battleTime()))
	case action == attackProtoID(a):
		ms := hs.mobs[target]
		if ms == nil || ms.dead {
			return
		}
		// «Штурм» friendly fire: never attack an ally (own-team creep/structure). Hunt
		// mobs are team -1 (teamVal), so this never blocks a PvE target.
		if ms.teamVal() == 1 {
			return
		}
		// A client-issued attack means its DEFENCE loop is live and will re-pick the
		// next enemy itself: hand retargeting back to the client.
		hs.autoResumed = false
		s.startAttackLocked(c, ms)
	case skillSlotByProto(a, action) != 0:
		s.startSkillOrderLocked(c, skillSlotByProto(a, action), target, px, py, hasPos)
	default:
		log.Printf("battle: %s DO_ACTION unknown action=%d target=%d", c.RemoteAddr(), action, target)
	}
}

// ---- player auto-attack (with optional projectile) ----

func (s *Server) startAttackLocked(c *conn, ms *mobState) {
	hs := c.huntState
	if hs.deadUntil > 0 {
		return
	}
	s.cancelOrderLocked(c)
	hs.attackTarget = ms.id
	c.resetChaseLocked() // new pursuit: the first out-of-range check paths at once
	hs.attackSeq++
	s.armAttackTimer(c, hs.attackSeq, 0, time.Duration(float64(time.Second)/s.attackPeriodLocked(hs)))
}

// resumeAutoAttackLocked makes the avatar keep fighting after an ability finishes,
// the same way it rolls onto the next mob after a kill: if it isn't already
// auto-attacking, has no pending approach-cast, and isn't mid manual-move, it
// re-engages the preferred target (the cast's victim) or, failing that, the nearest
// enemy within aggro range. With no enemy nearby it does nothing.
func (s *Server) resumeAutoAttackLocked(c *conn, now float64, preferred int32) {
	hs := c.huntState
	if hs == nil || hs.deadUntil > now || hs.st.stunned(now) {
		return
	}
	// Already swinging, chasing a cast, or walking somewhere by hand: leave it be.
	if hs.attackTarget != 0 || hs.order != nil || c.hasDest {
		return
	}
	ms := hs.mobs[preferred]
	if ms == nil || ms.dead {
		ms = s.nearestAttackableMobLocked(c, now, mobAggroRange)
	}
	if ms == nil {
		return
	}
	s.startAttackLocked(c, ms)
	// This attack is server-driven: the client won't self-retarget on a kill (it left
	// DEFENCE for the cast), so mark it so hitMobLocked chains onto the next mob.
	hs.autoResumed = true
}

// nearestAttackableMobLocked returns the closest living mob whose body is within r
// of the avatar (nil if none) -- the target it auto-acquires when it goes idle.
func (s *Server) nearestAttackableMobLocked(c *conn, now float64, r float64) *mobState {
	hs := c.huntState
	cx, cy := c.posAtLocked(float32(now))
	var best *mobState
	bestD := r
	for _, m := range hs.mobs {
		if m.dead {
			continue
		}
		d := math.Hypot(float64(m.x-cx), float64(m.y-cy)) - m.mob.Radius()
		if d < bestD {
			bestD, best = d, m
		}
	}
	return best
}

// attackPeriodLocked returns the effective attacks-per-second.
func (s *Server) attackPeriodLocked(hs *huntState) float64 {
	now := float64(s.battleTime())
	sp := hs.av.AttackSpeed * hs.st.attackFactor(now)
	if sp < 0.1 {
		sp = 0.1
	}
	return sp
}

func (s *Server) armAttackTimer(c *conn, seq int, delay, interval time.Duration) {
	time.AfterFunc(delay, func() {
		c.lock()
		defer c.unlock()
		hs := c.huntState
		if hs == nil || hs.closed || hs.attackSeq != seq || hs.deadUntil > 0 {
			return
		}
		ms := hs.mobs[hs.attackTarget]
		if ms == nil || ms.dead {
			s.stopAttackLocked(c, false)
			return
		}
		now := s.battleTime()
		cx, cy := c.posAtLocked(now)
		dist := math.Hypot(float64(ms.x-cx), float64(ms.y-cy))
		// Reach = attack range + both body radii (matches the client's own
		// AvatarAI reach math), so a big-bodied boss is hit from farther than a
		// small mob instead of a flat pad.
		if dist > hs.effAttackRangeLocked(float64(now))+hs.av.Radius()+ms.mob.Radius() {
			c.chaseMoveLocked(s, ms.x, ms.y) // throttled: re-paths on >1m drift or 1/s when idle
			s.armAttackTimer(c, seq, 250*time.Millisecond, interval)
			return
		}
		if c.vx != 0 || c.vy != 0 {
			c.stopArrivalLocked()
			c.hasDest = false
			c.x, c.y, c.vx, c.vy, c.snapT = cx, cy, 0, 0, now
			c.sendPosLocked(s, cx, cy, 0, 0, now)
		}
		// The avatar's basic-attack ACTION goes to this player AND to every teammate
		// that renders this avatar: renderAvatarForLocked put the matching ATTACK
		// effector (attackProtoID) on the remote object, so a teammate resolves the
		// swing exactly the way it resolves a mob's. The struck mob is always revealed
		// to the whole party (the attacker is well inside the co-op reveal radius), so
		// targetObj is never dangling on a teammate's client.
		actionArgs := newActionArgs(c.objID, attackProtoID(hs.av), ms.id, float64(now),
			amf.NewArray().Set("x", 0.0).Set("y", 0.0))
		s.pushAvatarAllLocked(c, battleproto.CmdAction, actionArgs)
		if hs.hasProjectile {
			// Release the projectile after the attack wind-up: 0 fires it now (a snap
			// shot, the default) so it flies immediately and the hit lands on arrival;
			// a caster like Elgorm whose bolt leaves at the END of the throw uses a
			// high AttackWindup, so the launch (and hit) are delayed to late in the
			// swing. The bolt's flight is recomputed at release from the live gap.
			release := time.Duration(hs.av.AttackWindup * float64(interval))
			s.scheduleProjectileLocked(c, seq, ms.id, release)
		} else {
			s.scheduleHitLocked(c, seq, ms.id, interval/2)
		}
		s.scheduleSwingDone(c, seq, interval)
		s.armAttackTimer(c, seq, interval, interval)
	})
}

// scheduleProjectileLocked launches the basic-attack projectile after the swing
// wind-up (release), then lands the hit when it arrives. release<=0 fires inline
// this tick -- the snap-shot path every ranged hero but Elgorm takes. The flight is
// computed from the caster→target gap at the moment of release, so a target that
// closed or fled during a long wind-up gets a correctly-timed bolt.
func (s *Server) scheduleProjectileLocked(c *conn, seq int, targetID int32, release time.Duration) {
	launch := func() {
		hs := c.huntState
		if hs == nil || hs.closed || hs.attackSeq != seq || hs.deadUntil > 0 {
			return
		}
		ms := hs.mobs[targetID]
		if ms == nil || ms.dead {
			return
		}
		now := s.battleTime()
		cx, cy := c.posAtLocked(now)
		flight := math.Hypot(float64(ms.x-cx), float64(ms.y-cy))/24 + 0.1
		s.pushAvatarAllLocked(c, battleproto.CmdSetProjectile, amf.NewArray().
			Set("source", c.objID).
			Set("target", ms.id).
			Set("hit_at", float64(now)+flight))
		// The bolt is now in flight (SET_PROJECTILE sent, client animates it landing on
		// the mob). Its hit is COMMITTED: it must land on arrival even if the player
		// cancels or retargets the attack mid-flight (which bumps attackSeq). Gating the
		// arrival hit on seq -- as a melee swing is -- would make a visibly-connecting
		// arrow deal no damage.
		s.scheduleProjectileHitLocked(c, ms.id, time.Duration(flight*float64(time.Second)))
	}
	if release <= 0 {
		launch() // inline: caller holds the lock (snap shot, byte-identical to before)
		return
	}
	time.AfterFunc(release, func() {
		c.lock()
		defer c.unlock()
		launch()
	})
}

// scheduleSwingDone closes each basic-attack swing on TEAMMATES' clients with an
// ACTION_DONE, a hair before the next swing. A remote avatar is animated purely by
// the server's ACTION/ACTION_DONE (no local AvatarAI, unlike the owner's own
// avatar): the attack clip is WrapMode.Once and only re-triggers when the action
// leaves and re-enters, but a repeat ACTION with the same id is rejected as a
// duplicate -- so without a DONE between swings a teammate sees the avatar swing
// once then freeze (the exact bug the summon path already works around). Sent to
// OTHERS only: the owner's self animation is AvatarAI-driven and must not change.
func (s *Server) scheduleSwingDone(c *conn, seq int, interval time.Duration) {
	// Fire at 85% of the swing interval: after this swing's ACTION, before the next.
	time.AfterFunc(interval*85/100, func() {
		c.lock()
		defer c.unlock()
		hs := c.huntState
		if hs == nil || hs.closed || hs.attackSeq != seq {
			return // attack stopped/retargeted; stopAttackLocked closes the final swing
		}
		s.broadcastAvatarToOthersLocked(c, battleproto.CmdActionDone, amf.NewArray().
			Set("id", c.objID).
			Set("action", attackProtoID(hs.av)).
			Set("item", false).
			Set("cooldown", float64(s.battleTime())))
	})
}

// scheduleHitLocked lands a basic-attack hit after windup, gated on the attack still
// being live (attackSeq unchanged) -- a melee swing interrupted before it connects
// deals no damage.
func (s *Server) scheduleHitLocked(c *conn, seq int, targetID int32, windup time.Duration) {
	s.scheduleHitAfterLocked(c, seq, targetID, windup, false)
}

// scheduleProjectileHitLocked lands the hit for a bolt already in flight: it is
// COMMITTED (seq is not checked), so a projectile that visibly reaches the mob still
// deals its damage even if the player cancels or retargets during the flight.
func (s *Server) scheduleProjectileHitLocked(c *conn, targetID int32, windup time.Duration) {
	s.scheduleHitAfterLocked(c, 0, targetID, windup, true)
}

func (s *Server) scheduleHitAfterLocked(c *conn, seq int, targetID int32, windup time.Duration, committed bool) {
	time.AfterFunc(windup, func() {
		c.lock()
		defer c.unlock()
		hs := c.huntState
		if hs == nil || hs.closed {
			return
		}
		if !committed && hs.attackSeq != seq {
			return
		}
		av := hs.av
		ms := hs.mobs[targetID]
		if ms == nil || ms.dead {
			return
		}
		now := s.battleTime()
		cx, cy := c.posAtLocked(now)
		if !hs.hasProjectile && math.Hypot(float64(ms.x-cx), float64(ms.y-cy)) > hs.effAttackRangeLocked(float64(now))+av.Radius()+ms.mob.Radius()+0.3 {
			return
		}
		dmg := (float64(av.DmgMin) + rand.Float64()*float64(av.DmgMax-av.DmgMin)) * hs.st.modMul(float64(now), "dmg_pct") * hs.powerMul()
		// Crit: crit_pct chance to strike for 1.5× base (+ crit_dmg_pct bonus magnitude),
		// flagged 2 on the RECEIVE_HIT so the client plays its CritStrikeEffect. (Skill
		// damage does not crit -- only the basic attack.)
		var hitFlags int32
		if crit := hs.st.modSum(float64(now), "crit_pct"); crit > 0 && rand.Float64() < crit {
			dmg *= 1.5 + hs.st.modSum(float64(now), "crit_dmg_pct")
			hitFlags = 2
		}
		s.hitMobFlagsLocked(c, ms, dmg, c.objID, hitFlags)
		// Lifesteal on basic attacks.
		if ls := hs.st.modSum(float64(now), "lifesteal_pct"); ls > 0 {
			s.healPlayerLocked(c, dmg*ls)
		}
		// On-hit passive procs.
		s.runProcsLocked(c, ms, float64(now))
	})
}

// runProcsLocked rolls each registered on-hit passive against a struck mob.
func (s *Server) runProcsLocked(c *conn, ms *mobState, now float64) {
	hs := c.huntState
	for _, pr := range hs.procs {
		level := int(hs.skillLevel[pr.slot-1])
		if level < 1 { // an unlearned passive (rank-0 ult slot) does not proc
			continue
		}
		if rand.Float64() >= pr.chance.At(level) {
			continue
		}
		ctx := opCtx{slot: pr.slot, level: level, target: ms, px: ms.x, py: ms.y, hasPos: true}
		s.applyOpsLocked(c, pr.ops, ctx, now)
	}
}

// runDefenseProcsLocked rolls each ON-DAMAGED passive after the avatar takes a hit --
// Titanid's «Каменная кожа» hardens (stacks +phys_armor) when he is STRUCK, not when
// he strikes. attacker is the mob that hit; the ops (a self armor buff) ignore it, but
// it feeds the ctx like a struck-target would for runProcsLocked.
func (s *Server) runDefenseProcsLocked(c *conn, attacker *mobState, now float64) {
	hs := c.huntState
	if len(hs.defenseProcs) == 0 {
		return
	}
	px, py := c.posAtLocked(float32(now))
	for _, pr := range hs.defenseProcs {
		level := int(hs.skillLevel[pr.slot-1])
		if level < 1 {
			continue
		}
		if rand.Float64() >= pr.chance.At(level) {
			continue
		}
		ctx := opCtx{slot: pr.slot, level: level, target: attacker, px: px, py: py, hasPos: true}
		s.applyOpsLocked(c, pr.ops, ctx, now)
	}
}

// procOnDamaged reports whether a passive's OpProc fires on taking damage (rather than
// on hitting). Hand-maintained by prefab+slot -- the trigger mode is a semantic of the
// skill, not present in the data. Titanid's «Каменная кожа» (slot 3) is the one such
// passive: it stacks armor as he is struck.
func procOnDamaged(prefab string, slot int) bool {
	return prefab == "Avtr_Tank_Titanid" && slot == 3
}

func (s *Server) stopAttackLocked(c *conn, silent bool) {
	hs := c.huntState
	if hs.attackTarget == 0 {
		return
	}
	hs.attackTarget = 0
	hs.attackSeq++
	if !silent {
		doneArgs := amf.NewArray().
			Set("id", c.objID).
			Set("action", attackProtoID(hs.av)).
			Set("item", false).
			Set("cooldown", float64(s.battleTime()))
		s.pushAvatarAllLocked(c, battleproto.CmdActionDone, doneArgs)
	}
}

// creditConnLocked resolves the player who should get kill credit for a hit whose
// visual source is `damager`: the member whose avatar it is, or the owner of the
// summon that dealt it. Falls back to c (the acting/rep connection) when the
// damager can't be matched (e.g. its owner has since left).
func (s *Server) creditConnLocked(c *conn, damager int32) *conn {
	for _, mem := range c.members() {
		hs := mem.huntState
		if hs == nil {
			continue
		}
		if mem.objID == damager {
			return mem
		}
		if _, ok := hs.summons[damager]; ok {
			return mem
		}
	}
	return c
}

// attackerArmorPenLocked returns the physical armor penetration of whoever owns the
// damager object -- a player who quaffed an AntiPhysArmor potion (phys_armor_pen), or
// the owner of the summon that landed the hit. It resolves the owner through
// creditConnLocked (players + their summons). 0 when there is no live owning avatar
// (e.g. environmental damage), leaving the mob's armor untouched.
func (s *Server) attackerArmorPenLocked(c *conn, damager int32, now float64) float64 {
	owner := s.creditConnLocked(c, damager)
	if owner == nil || owner.huntState == nil {
		return 0
	}
	return owner.huntState.st.modSum(now, "phys_armor_pen")
}

// hitMobLocked applies damage: RECEIVE_HIT + HEALTH sync (to every viewer), then
// the death sequence (ON_KILL -> HEALTH 0 -> XP/level -> delayed DELETE_OBJECT).
// damager is the RECEIVE_HIT source (the client resolves the impact VFX from it):
// a player's avatar for attacks/skills/DoTs, or a summon id for summon hits. Kill
// credit (ON_KILL/XP/coins) goes to the player who owns the damager, so in a party
// XP is awarded to whoever landed the killing blow -- not always the tick driver.
func (s *Server) hitMobLocked(c *conn, ms *mobState, dmg float64, damager int32) {
	s.hitMobFlagsLocked(c, ms, dmg, damager, 0)
}

// hitMobFlagsLocked is hitMobLocked with an explicit RECEIVE_HIT flags value (2 =
// crit, so the client plays its CritStrikeEffect over the impact).
func (s *Server) hitMobFlagsLocked(c *conn, ms *mobState, dmg float64, damager int32, flags int32) {
	if ms.dead {
		return
	}
	// «Штурм»: the altar (Fortress Crystal) shrugs off all damage until its side's
	// cannons are destroyed -- the core push rule. No effect on Hunt (ms.altar false).
	if ms.altar && c.inst != nil && c.inst.dota != nil && !c.inst.dota.altarVulnerableLocked(ms) {
		return
	}
	// Armor mitigation. The mob's physical armor softens the blow via the shared
	// armor/(armor+50) curve; the attacker's armor penetration (an AntiPhysArmor potion's
	// phys_armor_pen) chips POSITIVE armor toward 0 first. Velial's ult «Трибунал» applies
	// a negative phys_armor mod, so a stripped mob's armor can fall below 0 and it then
	// takes AMPLIFIED damage (armorMitigation > 1). Zero-armor mobs (most trash) get a 1.0
	// multiplier -- damage lands in full, exactly as before armor existed.
	now := float64(s.battleTime())
	armor := ms.physArmor(now)
	if armor > 0 {
		if pen := s.attackerArmorPenLocked(c, damager, now); pen > 0 {
			armor = math.Max(0, armor-pen)
		}
	}
	dmg *= armorMitigation(armor)
	ms.aggro = true
	ms.hp -= dmg
	s.broadcastObjLocked(c, ms.id, battleproto.CmdReceiveHit, amf.NewArray().
		Set("object", ms.id).
		Set("damager", damager).
		Set("flags", flags).
		Set("damage", dmg))
	if ms.hp > 0 {
		s.syncMobHealthLocked(c, ms)
		return
	}
	// Death. Credit goes to the damager's owner.
	killer := s.creditConnLocked(c, damager)
	ms.hp = 0
	ms.dead = true
	s.broadcastObjLocked(c, ms.id, battleproto.CmdOnKill, amf.NewArray().
		Set("killer", killer.objID).Set("id", ms.id))
	s.syncMobHealthLocked(c, ms)
	// Clear any status fx on the corpse (world-scoped -> ended on every viewer).
	for _, uid := range []int32{ms.st.stunFx, ms.st.rootFx, ms.st.slowFx, ms.st.atkSlowFx, ms.st.silenceFx, ms.st.dotFx} {
		s.worldFxEndLocked(c, uid)
	}
	// A ground-anchored barrier parented to this body must keep its anchor: hold the
	// corpse until the barrier expires (read before st is wiped).
	anchorUntil := ms.st.anchorFxUntil
	ms.st = unitStatus{}
	// End the killer's auto-attack session onto this mob (their client stops
	// swinging), and resume onto the next enemy if the attack was server-driven.
	if kh := killer.huntState; kh != nil && kh.attackTarget == ms.id {
		serverDriven := kh.autoResumed
		s.stopAttackLocked(killer, false)
		if serverDriven {
			s.resumeAutoAttackLocked(killer, float64(s.battleTime()), 0)
		}
	}
	s.grantXPLocked(killer, ms.xpReward())
	s.awardCoinsLocked(killer, ms.id, ms.coinReward())
	// On-kill heal (Cerber's «Кровавый пир»): the killer restores a fraction of the
	// slain enemy's max HP, capped.
	s.applyHealOnKillLocked(killer, ms, float64(s.battleTime()))
	// Loot: a ground chest with a single random consumable, shared party-wide
	// loot rights. Bosses always drop; trash rolls a flat 1-in-N chance.
	if s.rollDropLocked(ms) {
		s.spawnDropLocked(c, ms.x, ms.y, float64(s.battleTime()))
	}

	// Schedule the revive: every creature -- trash and boss alike -- respawns
	// mobRespawnDelay (5 min) after death, so the location can be farmed. The
	// combat tick revives it at its authored spawn point once the timer elapses.
	ms.respawnAt = float64(s.battleTime()) + mobRespawnDelay

	// After the death animation, remove the corpse from every client but KEEP the
	// mobState (dead) so the tick can respawn it later. If a barrier is anchored to
	// this body, delay the removal past the barrier's expiry (+0.5s so its EFFECT_END
	// fires first) so the SELF-mode VFX never orphans onto the caster.
	mobID := ms.id
	corpseDelay := corpseDeleteDelay
	if extra := time.Duration((anchorUntil - float64(s.battleTime()) + 0.5) * float64(time.Second)); extra > corpseDelay {
		corpseDelay = extra
	}
	time.AfterFunc(corpseDelay, func() {
		c.lock()
		defer c.unlock()
		hs := c.huntState
		if hs == nil {
			return
		}
		// In a shared world the corpse must be cleared from the REMAINING members
		// even if the killer disconnected within the corpse window; only give up if
		// the whole instance is gone (removeMobFromClientsLocked fans over the live
		// members via c.members()). A solo/bare conn keeps the old hs.closed guard.
		if c.inst != nil {
			if c.inst.closed {
				return
			}
		} else if hs.closed {
			return
		}
		m := hs.mobs[mobID]
		if m == nil || !m.dead {
			return // already respawned (or gone)
		}
		s.removeMobFromClientsLocked(c, m, float64(s.battleTime()))
	})
}

// applyHealOnKillLocked heals the killer for a fraction of the slain enemy's max HP
// (capped), when the killer has a learned OpHealOnKill passive (Cerber's «Кровавый
// пир»). Value = coefficient of the victim's max HP; Value2 = heal cap.
func (s *Server) applyHealOnKillLocked(c *conn, victim *mobState, now float64) {
	hs := c.huntState
	if hs == nil || hs.healOnKillSlot == 0 {
		return
	}
	slot := hs.healOnKillSlot
	level := int(hs.skillLevel[slot-1])
	if level < 1 {
		return
	}
	for _, op := range hs.skillDef(slot).Ops {
		if op.Kind != gamedata.OpHealOnKill {
			continue
		}
		heal := op.Value.At(level) * victim.maxHealth()
		if cap := op.Value2.At(level); cap > 0 && heal > cap {
			heal = cap
		}
		s.healPlayerLocked(c, heal)
		return
	}
}

// awardCoinsLocked pays a mob-kill bronze-coin bounty into the player's persistent
// hero money and pushes SET_MONEY so the client credits it and floats a "+N" over
// the corpse (fromObj). Money is a single integer the client splits into
// gold/silver/bronze (1 bronze = 1 unit), so coins add directly.
func (s *Server) awardCoinsLocked(c *conn, fromObj, coins int32) {
	if coins <= 0 {
		return
	}
	money, diamonds, ok := s.Store.AddHeroMoney(c.selfPlayerID, coins)
	if !ok {
		return // no hero bound to this battle connection
	}
	s.push(c, battleproto.CmdSetMoney, amf.NewArray().
		Set("from", fromObj).
		Set("money", amf.NewArray().Set("v", money).Set("r", diamonds)).
		Set("delta", amf.NewArray().Set("v", coins).Set("r", int32(0))))
}

// heroExpShare is the fraction of in-hunt XP that also feeds the account's
// PERSISTENT character level (session.Hero.Level/Exp), independent of the
// ephemeral hunt-instance level below. By design the hunt-instance level/XP
// itself is never persisted (it's a per-session battle-power scaler, reset
// every hunt); this is the one deliberate exception -- a slice of the XP
// earned still grows the permanent character across sessions.
const heroExpShare = 0.10

// grantXPLocked adds experience and processes level-ups. Each level gained grows
// the avatar's power via LevelPowerMul/LevelHealthMul (see gamedata): the max
// HP/mana pools rise and the hero is topped up by exactly the gained amount, then
// every level-scaled stat is re-synced so the HUD and the damage/HP math track.
func (s *Server) grantXPLocked(c *conn, xp float64) {
	hs := c.huntState
	hs.xp += xp
	s.syncSelfLocked(c, syncExperience)
	if charXP := int32(xp * heroExpShare); charXP > 0 {
		s.Store.AddHeroExp(c.selfPlayerID, charXP)
	}
	levels := gamedata.AvatarXPLevels
	now := float64(s.battleTime())
	leveled := false
	for int(hs.level)+1 < len(levels) && hs.xp >= levels[hs.level+1] {
		oldMaxHP := hs.maxHPLocked(now)
		oldMaxMana := hs.maxManaLocked(now)
		hs.level++
		hs.points++
		leveled = true
		// Grant the newly unlocked HP/mana so a level-up is an immediate boost, not
		// just a bigger empty bar.
		hs.hp += hs.maxHPLocked(now) - oldMaxHP
		hs.mana += hs.maxManaLocked(now) - oldMaxMana
		s.push(c, battleproto.CmdLevelUp, amf.NewArray().
			Set("id", c.objID).
			Set("level", hs.level).
			Set("points", hs.points))
		log.Printf("battle: %s LEVEL_UP -> %d (points %d)", c.RemoteAddr(), hs.level, hs.points)
	}
	if leveled {
		s.pushPlayerStatsLocked(c, now) // re-sync level-scaled dmg / maxHP / maxMana / spellpower
	}
}

// ---- UPGRADE_SKILL ----

func (s *Server) handleUpgradeSkill(c *conn, p battleproto.Packet) {
	if c.hunt == nil || c.huntState == nil {
		s.ack(c, p)
		return
	}
	proto := p.Args.IntOr("id", -1)
	c.lock()
	defer c.unlock()
	hs := c.huntState
	slot := skillSlotByProto(hs.av, proto)
	// Reject if: not a skill; no points; already at the skill's max rank; or the
	// avatar level does not yet meet the gate for the next rank (server-side mirror
	// of the client's `levels` array -- e.g. the ult's rank 1 needs level 5).
	cannot := slot == 0
	if !cannot {
		cur := int(hs.skillLevel[slot-1])
		req := skillReqLevel(slot, cur)
		cannot = hs.points <= 0 ||
			cur >= hs.kit.Skills[slot-1].MaxRank() ||
			(req >= 0 && int(hs.level) < req)
	}
	if cannot {
		if err := c.send(battleproto.Packet{Cmd: p.Cmd, Args: amf.NewArray(),
			RequestID: p.RequestID, Status: false, Error: "cannot upgrade"}); err != nil {
			log.Printf("battle: %s UPGRADE_SKILL reply error: %v", c.RemoteAddr(), err)
		}
		return
	}
	hs.points--
	hs.skillLevel[slot-1]++
	s.ack(c, p)

	old := hs.childEff[slot-1]
	s.push(c, battleproto.CmdRemEffector, amf.NewArray().Set("id", old))
	nid := hs.newEffID()
	hs.childEff[slot-1] = nid
	s.push(c, battleproto.CmdAddEffector, addEffectorArgs(nid, activeProtoID(hs.av, slot),
		c.objID, hs.parentEff[slot-1], float64(s.battleTime()),
		childArgs(hs.kit.Skills[slot-1], hs.skillLevel[slot-1])))
	// Permanent passives re-derive their self-buff stats and step their model-grow
	// VFX to the new level (the child rebuild above only refreshes tooltips).
	s.reapplyPassiveLocked(c, slot, float64(s.battleTime()))
	log.Printf("battle: %s skill %d upgraded to L%d (points left %d)",
		c.RemoteAddr(), slot, hs.skillLevel[slot-1], hs.points)
}

// reapplyPassiveLocked re-derives a permanent passive's self-buff stat mods and
// its per-level model-grow VFX at the current skill level (UPGRADE_SKILL). Safe
// only after world-build, since it pushes packets immediately.
func (s *Server) reapplyPassiveLocked(c *conn, slot int, now float64) {
	hs := c.huntState
	sk := hs.kit.Skills[slot-1]
	if sk.Type != "PASSIVE" {
		return
	}
	level := int(hs.skillLevel[slot-1])
	s.removeModsBySrcLocked(c, "passive"+itoa(slot), now)
	for _, op := range sk.Ops {
		if op.Kind == gamedata.OpBuffStat && op.On != "target" && op.Dur.At(1) == 0 {
			hs.st.mods = append(hs.st.mods, statMod{
				stat: op.Stat, value: op.Value.At(level), until: 0, src: "passive" + itoa(slot)})
		}
	}
	if sk.GrowFx != "" {
		// End the old size, attach the next-larger grow prefab (a brief scale pop).
		s.fxEndLocked(c, hs.growFx[slot-1])
		hs.growFx[slot-1] = s.fxStartLocked(c, sk.GrowFx+itoa(level), c.objID, 0, false, 0, 0)
	}
	// Refresh the permanent buff-bar icon so its hover tip tracks the new rank; this
	// also lights it up the first time a rank-0 (ult-slot) passive is learned.
	if sk.BuffIcon && level >= 1 {
		if hs.passiveBuffEff[slot-1] != 0 {
			s.push(c, battleproto.CmdRemEffector, amf.NewArray().Set("id", hs.passiveBuffEff[slot-1]))
		}
		hs.passiveBuffEff[slot-1] = hs.newEffID()
		hs.passiveBuffCount[slot-1] = passiveBuffCountOrNone(hs, slot, level, now)
		s.push(c, battleproto.CmdAddEffector, addEffectorArgs(hs.passiveBuffEff[slot-1],
			buffProtoID(hs.av, slot), c.objID, -1, now, hs.passiveBuffArgs(slot, level, now)))
	}
	s.pushPlayerStatsLocked(c, now)
}

// refreshPassiveBuffCountersLocked keeps the number beside a passive's buff icon in
// sync with its live value (Velial's «Воля к победе» bonus tracks missing HP). The
// client reads a battle buff's counter only when the effector is (re)added, so on a
// change we REM the old icon and ADD it back with the new counter -- cheap because it
// fires only when the displayed integer actually moves, not every tick.
func (s *Server) refreshPassiveBuffCountersLocked(c *conn, now float64) {
	hs := c.huntState
	if hs.deadUntil > 0 {
		return // icon is dropped while dead; sendEffectorsLocked re-adds it on respawn
	}
	for i := range hs.kit.Skills {
		if hs.passiveBuffEff[i] == 0 { // no live buff icon in this slot
			continue
		}
		slot := i + 1
		level := int(hs.skillLevel[i])
		cnt, ok := hs.passiveBuffCounterLocked(slot, level, now)
		if !ok || cnt == hs.passiveBuffCount[i] {
			continue // static passive, or the displayed integer hasn't changed
		}
		s.push(c, battleproto.CmdRemEffector, amf.NewArray().Set("id", hs.passiveBuffEff[i]))
		hs.passiveBuffEff[i] = hs.newEffID()
		hs.passiveBuffCount[i] = cnt
		s.push(c, battleproto.CmdAddEffector, addEffectorArgs(hs.passiveBuffEff[i],
			buffProtoID(hs.av, slot), c.objID, -1, now, hs.passiveBuffArgs(slot, level, now)))
	}
}

// closeHunt tears the hunt state down on disconnect (invalidates timers) and
// removes the player from its shared world, telling the remaining members to drop
// its avatar. The instance disposes itself once its last member leaves.
func (c *conn) closeHunt() {
	c.lock()
	defer c.unlock()
	// Invalidate any in-flight movement leg: it bumps moveGen so a fired-but-pending
	// arrival closure no-ops, and Stops the timer. Without this a leftover leg could
	// fire after the same user reconnects (objID is reused) into a still-alive shared
	// world and push stale coordinates for the new avatar to every teammate.
	c.stopArrivalLocked()
	if c.huntState != nil {
		c.huntState.closed = true
		c.huntState.attackSeq++
	}
	if c.inst != nil {
		c.inst.leaveInstanceLocked(c)
	}
	// A central-square occupant: drop its avatar from every remaining occupant.
	if c.linst != nil {
		c.linst.leaveLobbyInstanceLocked(c)
	}
}
