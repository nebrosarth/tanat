package battleserver

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"

	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

const (
	AssaultHeroCount       = 10
	AssaultMaxEntities     = 96
	AssaultHeroFeatureSize = 32
	AssaultEntityFeatures  = 16
	AssaultGlobalFeatures  = 32
	AssaultActionKinds     = 8
	AssaultTick            = 200 * time.Millisecond
)

const assaultSchemaV1 = "tanat.assault.v1|hero=10x32|entities=10x96x16|global=10x32|action=kind,target,direction,distance|mask=8+96+4x96"
const assaultRewardSchemaV2 = "tanat.assault.reward.v2|xp=.002|money_gain=.04|money_spend=.004|hp=2|mana=.75|death=-1|hero_kill=-.6|creep_last_hit=-.16|structure=two_thirds_damage+one_third_destroy|win=5|zero_sum=1|team_spirit=.2"

var AssaultSchemaHashV1 = sha256.Sum256([]byte(assaultSchemaV1))
var AssaultRewardHashV2 = sha256.Sum256([]byte(assaultRewardSchemaV2))

type AssaultControllerV1 uint8

const (
	AssaultControllerExternal AssaultControllerV1 = iota
	AssaultControllerAI20
	AssaultControllerAI30
	// AssaultControllerAI40 is driven by the external action batch just like
	// AssaultControllerExternal, but explicitly marks a hero as a member of the
	// shared AI-40 self-play policy.  Keeping the identity in the reset contract
	// prevents a training run from silently falling back to a scripted opponent.
	AssaultControllerAI40
)

type AssaultActionKindV1 uint8

const (
	AssaultActionWait AssaultActionKindV1 = iota
	AssaultActionMove
	AssaultActionAttack
	AssaultActionSkill1
	AssaultActionSkill2
	AssaultActionSkill3
	AssaultActionSkill4
	AssaultActionTeleport // reserved; masked in V1
)

// HeroActionV1 is deliberately factorized for policy heads. Target is an
// observation entity slot, never an internal server object id. Direction is a
// 16-way compass bin; Distance is 0/1/2 for 4/8/12 world units.
type HeroActionV1 struct {
	Kind      AssaultActionKindV1
	Target    uint16
	Direction uint8
	Distance  uint8
}

type ActionMaskV1 struct {
	Kinds       [AssaultActionKinds]uint8
	Targets     [AssaultMaxEntities]uint8
	SkillTarget [4][AssaultMaxEntities]uint8
}

type AssaultObservationV1 struct {
	Hero       [AssaultHeroFeatureSize]float32
	Entities   [AssaultMaxEntities][AssaultEntityFeatures]float32
	Global     [AssaultGlobalFeatures]float32
	EntityMask [AssaultMaxEntities]uint8
	ActionMask ActionMaskV1
}

type StepResultV1 struct {
	SchemaHash   [32]byte
	RewardHash   [32]byte
	Step         uint32
	Elapsed      float32
	Done         bool
	Winner       int32
	Invalid      [AssaultHeroCount]uint8
	Reward       [AssaultHeroCount]float32
	Observations [AssaultHeroCount]AssaultObservationV1
}

type AssaultResetV1 struct {
	Seed        int64
	Roster      [AssaultHeroCount]int32
	Controllers [AssaultHeroCount]AssaultControllerV1
	MaxSteps    uint32
}

type assaultRewardSnapshot struct {
	xp, hp, mana, objective float64
	money                   int32
	frags, creepKills       int32
	dead                    bool
}

// AssaultEnv is a synchronous, network-client-free wrapper around the same
// authoritative DOTA simulation used by live matches. A Reset owns an isolated
// in-memory session store, so training can never mutate account economy.
type AssaultEnv struct {
	server           *Server
	inst             *huntInstance
	clock            *manualBattleClock
	heroes           [AssaultHeroCount]*conn
	brains           [AssaultHeroCount]*botBrain
	controllers      [AssaultHeroCount]AssaultControllerV1
	entityIDs        [AssaultHeroCount][AssaultMaxEntities]int32
	previous         [AssaultHeroCount]assaultRewardSnapshot
	entityScratch    [AssaultHeroCount][]assaultEntity
	terminalRewarded bool
	step             uint32
	maxSteps         uint32
	closed           bool
}

func NewAssaultEnv() *AssaultEnv { return &AssaultEnv{} }

func (e *AssaultEnv) Reset(cfg AssaultResetV1) (StepResultV1, error) {
	e.Close()
	if cfg.MaxSteps == 0 {
		cfg.MaxSteps = 30 * 60 * 5
	}
	for i, controller := range cfg.Controllers {
		if controller > AssaultControllerAI40 {
			return StepResultV1{}, fmt.Errorf("assault: controller slot %d has unsupported value %d", i, controller)
		}
	}
	dm, ok := gamedata.DotaMapByID(101)
	if !ok {
		maps := gamedata.DotaMaps()
		if len(maps) == 0 {
			return StepResultV1{}, errors.New("assault: no DOTA map configured")
		}
		dm = maps[0]
	}
	clock := newManualBattleClock()
	store := session.NewStore()
	s := New(store)
	s.clock = clock
	s.rng = rand.New(rand.NewSource(cfg.Seed))
	inst := newDotaInstance(s, 1, dm.ID)
	inst.dota.telemetry = nil
	inst.dota.botAIVersionByTeam[dotaTeamHuman] = 20
	inst.dota.botAIVersionByTeam[dotaTeamElf] = 20
	e.server, e.inst, e.clock, e.closed = s, inst, clock, false

	roster := botAvatarRoster()
	if len(roster) < AssaultHeroCount {
		return StepResultV1{}, fmt.Errorf("assault: bot roster has %d avatars, need %d", len(roster), AssaultHeroCount)
	}
	for i := 0; i < AssaultHeroCount; i++ {
		avatar := roster[i]
		if cfg.Roster[i] != 0 {
			var found bool
			avatar, found = gamedata.AvatarByID(cfg.Roster[i])
			if !found {
				e.Close()
				return StepResultV1{}, fmt.Errorf("assault: roster slot %d has unknown avatar %d", i, cfg.Roster[i])
			}
		}
		side := gamedata.DotaSideHuman
		if i >= AssaultHeroCount/2 {
			side = gamedata.DotaSideElf
		}
		c := s.newHeadlessBotConnLocked(inst, i, side, avatar)
		e.heroes[i] = c
		e.brains[i] = inst.bots[c.objID]
		e.controllers[i] = cfg.Controllers[i]
		switch cfg.Controllers[i] {
		case AssaultControllerExternal:
			delete(inst.bots, c.objID)
		case AssaultControllerAI20:
			e.brains[i].aiVersion, e.brains[i].aiVersionSet = 20, true
		case AssaultControllerAI30:
			e.brains[i].aiVersion, e.brains[i].aiVersionSet = 30, true
		case AssaultControllerAI40:
			e.brains[i].aiVersion, e.brains[i].aiVersionSet = 40, true
			// Headless AI-40 actions arrive synchronously in Step.  Do not leave
			// the brain in inst.bots: that would also start the live ONNX sidecar
			// and give one hero two independent controllers.
			delete(inst.bots, c.objID)
		}
	}
	// The team orchestrator uses the strongest scripted controller present on
	// that side, while each hero retains its explicitly requested local profile.
	for i, controller := range cfg.Controllers {
		if controller == AssaultControllerAI30 {
			team := dotaTeamHuman
			if i >= AssaultHeroCount/2 {
				team = dotaTeamElf
			}
			inst.dota.botAIVersionByTeam[team] = 30
		} else if controller == AssaultControllerAI40 {
			team := dotaTeamHuman
			if i >= AssaultHeroCount/2 {
				team = dotaTeamElf
			}
			inst.dota.botAIVersionByTeam[team] = 40
		}
	}
	e.step, e.maxSteps, e.closed, e.terminalRewarded = 0, cfg.MaxSteps, false, false

	inst.mu.Lock()
	now := clock.Now()
	for i := range e.heroes {
		e.previous[i] = e.rewardSnapshotLocked(i, now)
	}
	result := e.resultLocked(nil, now)
	inst.mu.Unlock()
	return result, nil
}

func (e *AssaultEnv) Step(actions [AssaultHeroCount]HeroActionV1) (StepResultV1, error) {
	if e.closed || e.server == nil || e.inst == nil || e.clock == nil {
		return StepResultV1{}, errors.New("assault: Step called before Reset or after Close")
	}
	if e.inst.dota.ended || e.step >= e.maxSteps {
		e.inst.mu.Lock()
		result := e.resultLocked(nil, e.clock.Now())
		e.inst.mu.Unlock()
		return result, nil
	}

	var invalid [AssaultHeroCount]uint8
	e.inst.mu.Lock()
	for i := 0; i < AssaultHeroCount; i++ {
		if !assaultControllerUsesActions(e.controllers[i]) {
			continue
		}
		if !e.applyActionLocked(i, actions[i]) {
			invalid[i] = 1
		}
	}
	e.inst.mu.Unlock()

	// Execute callbacks (movement arrival, swings, projectiles, corpses) outside
	// the instance lock; each callback acquires that lock through conn.lock.
	e.clock.Advance(AssaultTick)
	now := float64(e.server.battleTime())

	e.inst.mu.Lock()
	if !e.inst.dota.ended {
		e.server.botRebalanceLanesLocked(e.inst, now)
		e.server.botPlanTeamsLocked(e.inst, now)
		members := assaultOrderedMembers(e.inst)
		var rep *conn
		for _, c := range members {
			if c.huntState == nil || c.huntState.closed {
				continue
			}
			e.server.memberTickLocked(c, now)
			rep = c
			if brain := e.inst.bots[c.objID]; brain != nil && botAIVersionForBrain(brain) != 40 {
				e.server.botTickLocked(brain, now)
			}
		}
		e.server.botAI40BatchTickLocked(e.inst, now)
		// External policies still use the exact scripted purchase/skill rules.
		for i, brain := range e.brains {
			if assaultControllerUsesActions(e.controllers[i]) && brain != nil {
				e.server.botSpendSkillPointLocked(brain)
				e.server.botBuyItemsLocked(brain, now)
			}
		}
		if rep != nil {
			e.server.dotaTickLocked(rep, now)
		}
	}
	e.step++
	result := e.resultLocked(&invalid, now)
	e.inst.mu.Unlock()
	return result, nil
}

func assaultControllerUsesActions(controller AssaultControllerV1) bool {
	return controller == AssaultControllerExternal || controller == AssaultControllerAI40
}

func (e *AssaultEnv) Close() {
	if e.closed {
		return
	}
	e.closed = true
	if e.inst != nil {
		e.inst.mu.Lock()
		e.inst.closed = true
		var ai40Runtime ai40PolicyRuntime
		if e.inst.dota != nil {
			ai40Runtime = e.inst.dota.ai40Runtime
			e.inst.dota.ai40Runtime = nil
		}
		for _, c := range e.heroes {
			if c != nil && c.Conn != nil {
				_ = c.Conn.Close()
			}
		}
		e.inst.mu.Unlock()
		if ai40Runtime != nil {
			_ = ai40Runtime.Close()
		}
	}
	e.server, e.inst, e.clock = nil, nil, nil
}

func assaultOrderedMembers(inst *huntInstance) []*conn {
	ids := make([]int32, 0, len(inst.members))
	for id := range inst.members {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	out := make([]*conn, 0, len(ids))
	for _, id := range ids {
		out = append(out, inst.members[id])
	}
	return out
}

func (e *AssaultEnv) applyActionLocked(slot int, action HeroActionV1) bool {
	c := e.heroes[slot]
	return e.applyConnActionLocked(c, &e.entityIDs[slot], action)
}

// applyConnActionLocked is shared by headless external policies and the live
// AI-40 profile. Caller holds the instance lock.
func (e *AssaultEnv) applyConnActionLocked(c *conn, entityIDs *[AssaultMaxEntities]int32, action HeroActionV1) bool {
	if c == nil || c.huntState == nil || int(action.Kind) >= AssaultActionKinds {
		return false
	}
	now := float64(e.server.battleTime())
	mask := e.observationForConnLocked(c, entityIDs, now).ActionMask
	if mask.Kinds[action.Kind] == 0 {
		return action.Kind == AssaultActionWait
	}
	hs := c.huntState
	switch action.Kind {
	case AssaultActionWait:
		return true
	case AssaultActionMove:
		direction := int(action.Direction) % 16
		distance := []float64{4, 8, 12}[minInt(int(action.Distance), 2)]
		a := float64(direction) * 2 * math.Pi / 16
		x, y := c.posAtLocked(float32(now))
		c.moveToLocked(e.server, x+float32(math.Cos(a)*distance), y+float32(math.Sin(a)*distance))
		return true
	case AssaultActionAttack:
		id, ok := assaultActionTargetID(entityIDs, action.Target)
		if !ok || mask.Targets[action.Target] == 0 {
			return false
		}
		if mob := e.inst.mobs[id]; mob != nil && !mob.dead && mob.enemyOf(c.playerTeam()) {
			e.server.startAttackLocked(c, mob)
			return true
		}
		if victim := e.inst.members[id]; victim != nil && victim.huntState != nil && arenaEnemies(c, victim) {
			e.server.startPvpAttackLocked(c, victim)
			return true
		}
		return false
	case AssaultActionSkill1, AssaultActionSkill2, AssaultActionSkill3, AssaultActionSkill4:
		skill := int(action.Kind-AssaultActionSkill1) + 1
		def := hs.skillDef(skill)
		id := int32(0)
		if skillHasTargetFlag(def, "ENEMY") || skillHasTargetFlag(def, "FRIEND") {
			if int(action.Target) >= AssaultMaxEntities || mask.SkillTarget[skill-1][action.Target] == 0 {
				return false
			}
			var ok bool
			id, ok = assaultActionTargetID(entityIDs, action.Target)
			if !ok {
				return false
			}
		}
		direction := int(action.Direction) % 16
		distance := []float64{4, 8, 12}[minInt(int(action.Distance), 2)]
		a := float64(direction) * 2 * math.Pi / 16
		x, y := c.posAtLocked(float32(now))
		return e.server.startSkillOrderLocked(c, skill, id,
			x+float32(math.Cos(a)*distance), y+float32(math.Sin(a)*distance), true)
	default:
		_ = hs
		return false
	}
}

func (e *AssaultEnv) actionTargetID(slot int, target uint16) (int32, bool) {
	return assaultActionTargetID(&e.entityIDs[slot], target)
}

func assaultActionTargetID(entityIDs *[AssaultMaxEntities]int32, target uint16) (int32, bool) {
	if entityIDs == nil || int(target) >= AssaultMaxEntities {
		return 0, false
	}
	id := entityIDs[target]
	return id, id != 0
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

type assaultEntity struct {
	id       int32
	kind     int
	distance float64
	player   *conn
	mob      *mobState
}

func (e *AssaultEnv) observationLocked(slot int, now float64) AssaultObservationV1 {
	c := e.heroes[slot]
	if c == nil {
		return AssaultObservationV1{}
	}
	sources := dotaTeamVisionSourcesLocked(e.inst, c.playerTeam(), now)
	return e.observationForConnWithScratchLocked(
		c, &e.entityIDs[slot], now, sources, &e.entityScratch[slot],
	)
}

func (e *AssaultEnv) observationForConnLocked(c *conn, entityIDs *[AssaultMaxEntities]int32, now float64) AssaultObservationV1 {
	sources := dotaTeamVisionSourcesLocked(e.inst, c.playerTeam(), now)
	var scratch []assaultEntity
	return e.observationForConnWithScratchLocked(c, entityIDs, now, sources, &scratch)
}

func (e *AssaultEnv) observationForConnWithScratchLocked(
	c *conn, entityIDs *[AssaultMaxEntities]int32, now float64,
	sources []dotaVisionSource, scratch *[]assaultEntity,
) AssaultObservationV1 {
	var o AssaultObservationV1
	if c == nil || c.huntState == nil {
		return o
	}
	hs := c.huntState
	team := c.playerTeam()
	x, y := c.posAtLocked(float32(now))
	maxHP, maxMana := hs.maxHPLocked(now), hs.maxManaLocked(now)
	money, _, _ := e.server.Store.HeroMoney(c.selfPlayerID)
	o.Hero[0] = float32(hs.av.ID) / 100
	o.Hero[1] = float32(team)
	o.Hero[2], o.Hero[3] = x/160, y/160
	o.Hero[4] = safeFrac(hs.hp, maxHP)
	o.Hero[5] = safeFrac(hs.mana, maxMana)
	o.Hero[6] = float32(hs.level) / 20
	o.Hero[7] = float32(hs.xp) / 10000
	o.Hero[8] = float32(money) / 100000
	o.Hero[9] = boolFloat(hs.deadUntil > now)
	o.Hero[10] = boolFloat(hs.st.stunned(now))
	o.Hero[11] = boolFloat(hs.st.silenced(now))
	o.Hero[12] = boolFloat(hs.st.rooted(now))
	o.Hero[13] = boolFloat(now < hs.invisibleUntil)
	o.Hero[14] = boolFloat(hs.attackTarget != 0 || hs.pvpTarget != 0)
	for i := 0; i < 4; i++ {
		o.Hero[15+i] = float32(hs.skillLevel[i]) / 5
		remaining := math.Max(0, hs.cooldownUntil[i]-now)
		o.Hero[19+i] = float32(remaining / 60)
		o.Hero[23+i] = boolFloat(hs.toggleOn[i])
	}
	o.Hero[27] = float32(hs.points) / 10
	o.Hero[28] = float32(c.vx) / 10
	o.Hero[29] = float32(c.vy) / 10
	o.Hero[30] = float32(now) / 3600

	entities := (*scratch)[:0]
	if cap(entities) < len(e.inst.members)+len(e.inst.mobs) {
		entities = make([]assaultEntity, 0, len(e.inst.members)+len(e.inst.mobs))
	}
	for _, other := range e.inst.members {
		if other.huntState == nil {
			continue
		}
		visible := other.playerTeam() == team || dotaVisibleEnemyMemberLocked(e.inst, team, other, now, sources, false)
		if !visible {
			continue
		}
		ox, oy := other.posAtLocked(float32(now))
		entities = append(entities, assaultEntity{id: other.objID, kind: 1,
			distance: math.Hypot(float64(ox-x), float64(oy-y)), player: other})
	}
	for _, mob := range e.inst.mobs {
		if mob.dead || (mob.teamVal() != team && !botVisibleEnemyMobLocked(team, mob, sources)) {
			continue
		}
		kind := 2
		if mob.structure {
			kind = 3
		}
		entities = append(entities, assaultEntity{id: mob.id, kind: kind,
			distance: math.Hypot(float64(mob.x-x), float64(mob.y-y)), mob: mob})
	}
	*scratch = entities
	sort.Slice(entities, func(i, j int) bool {
		priority := func(kind int) int {
			switch kind {
			case 1: // heroes always survive truncation
				return 0
			case 3: // strategic structures precede a large late-game creep set
				return 1
			default:
				return 2
			}
		}
		pi, pj := priority(entities[i].kind), priority(entities[j].kind)
		if pi != pj {
			return pi < pj
		}
		if entities[i].distance != entities[j].distance {
			return entities[i].distance < entities[j].distance
		}
		return entities[i].id < entities[j].id
	})
	for i := range entityIDs {
		entityIDs[i] = 0
	}
	for i := 0; i < len(entities) && i < AssaultMaxEntities; i++ {
		ent := entities[i]
		entityIDs[i] = ent.id
		o.EntityMask[i] = 1
		f := &o.Entities[i]
		f[0] = float32(ent.kind) / 3
		f[1] = float32(ent.id) / 1000000
		f[6] = float32(ent.distance) / 200
		if ent.player != nil {
			other := ent.player
			oh := other.huntState
			ox, oy := other.posAtLocked(float32(now))
			f[2] = boolFloat(other.playerTeam() == team)
			f[3], f[4] = (ox-x)/200, (oy-y)/200
			f[5] = safeFrac(oh.hp, oh.maxHPLocked(now))
			f[7] = safeFrac(oh.mana, oh.maxManaLocked(now))
			f[8] = float32(oh.level) / 20
			f[9] = boolFloat(oh.deadUntil <= now)
			f[15] = float32(oh.av.ID) / 100
		} else {
			mob := ent.mob
			f[2] = boolFloat(mob.teamVal() == team)
			f[3], f[4] = (mob.x-x)/200, (mob.y-y)/200
			f[5] = safeFrac(mob.hp, mob.maxHealth())
			f[9] = 1
			f[10] = boolFloat(mob.structure)
			f[11] = boolFloat(mob.altar)
			f[12] = boolFloat(mob.structure && mob.dotaRole == gamedata.DotaGun)
			f[13] = boolFloat(mob.structure && mob.dotaRole == gamedata.DotaCreepTower)
			f[14] = boolFloat(mob.siege)
		}
	}

	o.ActionMask.Kinds[AssaultActionWait] = 1
	canAct := hs.deadUntil <= now && !hs.st.stunned(now)
	if canAct && !hs.st.rooted(now) {
		o.ActionMask.Kinds[AssaultActionMove] = 1
	}
	for i := 0; i < AssaultMaxEntities && o.EntityMask[i] != 0; i++ {
		id := entityIDs[i]
		enemy := false
		if mob := e.inst.mobs[id]; mob != nil {
			enemy = mob.enemyOf(team) && !(mob.altar && !e.inst.dota.altarVulnerableLocked(mob))
		} else if member := e.inst.members[id]; member != nil {
			enemy = member.playerTeam() != team && member.huntState.deadUntil <= now
		}
		if enemy {
			o.ActionMask.Targets[i] = 1
		}
	}
	if canAct {
		for i := 0; i < AssaultMaxEntities; i++ {
			if o.ActionMask.Targets[i] != 0 {
				o.ActionMask.Kinds[AssaultActionAttack] = 1
				break
			}
		}
		if !hs.st.silenced(now) {
			for skill := 0; skill < 4; skill++ {
				level := int(hs.skillLevel[skill])
				def := hs.skillDef(skill + 1)
				if level < 1 || def.Type == "PASSIVE" || now < hs.cooldownUntil[skill] ||
					hs.mana < skillManaCost(float64(def.ManaCost[level-1])) {
					continue
				}
				unitTarget := skillHasTargetFlag(def, "ENEMY") || skillHasTargetFlag(def, "FRIEND")
				if !unitTarget {
					// The target head still needs one legal category for self/point skills;
					// applyActionLocked deliberately converts it to object id 0.
					o.ActionMask.SkillTarget[skill][0] = 1
					o.ActionMask.Kinds[int(AssaultActionSkill1)+skill] = 1
					continue
				}
				for i := 0; i < AssaultMaxEntities && o.EntityMask[i] != 0; i++ {
					id := entityIDs[i]
					valid := false
					if mob := e.inst.mobs[id]; mob != nil {
						valid = (mob.enemyOf(team) && skillHasTargetFlag(def, "ENEMY")) ||
							(!mob.enemyOf(team) && skillHasTargetFlag(def, "FRIEND"))
						if valid && mob.structure && skillHasTargetFlag(def, "NOT_BUILDING") {
							valid = false
						}
					} else if member := e.inst.members[id]; member != nil && member.huntState.deadUntil <= now {
						valid = (member.playerTeam() != team && skillHasTargetFlag(def, "ENEMY")) ||
							(member.playerTeam() == team && skillHasTargetFlag(def, "FRIEND"))
					}
					if valid {
						o.ActionMask.SkillTarget[skill][i] = 1
						o.ActionMask.Kinds[int(AssaultActionSkill1)+skill] = 1
					}
				}
			}
		}
	}

	o.Global[0] = float32(now) / 3600
	o.Global[1] = float32(team)
	o.Global[2] = boolFloat(e.inst.dota.ended)
	o.Global[3] = float32(e.inst.dota.winner)
	for _, member := range e.inst.members {
		if member.huntState == nil || member.huntState.deadUntil > now {
			continue
		}
		if member.playerTeam() == team {
			o.Global[4]++
		} else if dotaVisibleEnemyMemberLocked(e.inst, team, member, now, sources, false) {
			o.Global[5]++
		}
	}
	for _, mob := range e.inst.mobs {
		if mob.dead || !mob.structure {
			continue
		}
		idx := 6
		if mob.teamVal() != team {
			idx = 7
		}
		o.Global[idx] += safeFrac(mob.hp, mob.maxHealth())
	}
	return o
}

func (e *AssaultEnv) rewardSnapshotLocked(slot int, now float64) assaultRewardSnapshot {
	c := e.heroes[slot]
	if c == nil || c.huntState == nil {
		return assaultRewardSnapshot{}
	}
	money, _, _ := e.server.Store.HeroMoney(c.selfPlayerID)
	return assaultRewardSnapshot{
		xp:         c.huntState.xp,
		hp:         float64(safeFrac(c.huntState.hp, c.huntState.maxHPLocked(now))),
		mana:       float64(safeFrac(c.huntState.mana, c.huntState.maxManaLocked(now))),
		objective:  e.assaultObjectivePotentialLocked(c.playerTeam()),
		money:      money,
		frags:      c.huntState.frags,
		creepKills: c.huntState.assaultCreepKills,
		dead:       c.huntState.deadUntil > now,
	}
}

func (e *AssaultEnv) resultLocked(invalid *[AssaultHeroCount]uint8, now float64) StepResultV1 {
	r := StepResultV1{SchemaHash: AssaultSchemaHashV1, RewardHash: AssaultRewardHashV2, Step: e.step, Elapsed: float32(now)}
	r.Done = e.inst.dota.ended || e.step >= e.maxSteps
	r.Winner = e.inst.dota.winner
	if invalid != nil {
		r.Invalid = *invalid
	}
	var raw [AssaultHeroCount]float64
	vision := [2][]dotaVisionSource{
		dotaTeamVisionSourcesLocked(e.inst, dotaTeamHuman, now),
		dotaTeamVisionSourcesLocked(e.inst, dotaTeamElf, now),
	}
	for i := 0; i < AssaultHeroCount; i++ {
		cur := e.rewardSnapshotLocked(i, now)
		prev := e.previous[i]
		reward := (cur.xp - prev.xp) * 0.002
		if cur.money > prev.money {
			reward += float64(cur.money-prev.money) * 0.04
		} else if cur.money < prev.money {
			reward += float64(prev.money-cur.money) * 0.004
		}
		reward += (cur.hp - prev.hp) * 2
		reward += (cur.mana - prev.mana) * 0.75
		reward += cur.objective - prev.objective
		reward -= float64(cur.frags-prev.frags) * 0.6
		reward -= float64(cur.creepKills-prev.creepKills) * 0.16
		if cur.dead && !prev.dead {
			reward -= 1
		}
		if invalid != nil && invalid[i] != 0 {
			reward -= 0.01
		}
		if r.Done && r.Winner != 0 && e.heroes[i] != nil && !e.terminalRewarded {
			if e.heroes[i].playerTeam() == r.Winner {
				reward += 5
			}
		}
		raw[i] = reward
		teamIndex := i / (AssaultHeroCount / 2)
		r.Observations[i] = e.observationForConnWithScratchLocked(
			e.heroes[i], &e.entityIDs[i], now, vision[teamIndex], &e.entityScratch[i],
		)
		e.previous[i] = cur
	}
	// OpenAI-Five-style zero-sum correction removes incentives that merely inflate
	// both teams' shaped score. Team spirit then shares 20% of the corrected team
	// average while preserving 80% individual credit assignment.
	var teamMean [2]float64
	for i := 0; i < AssaultHeroCount; i++ {
		teamMean[i/(AssaultHeroCount/2)] += raw[i] / (AssaultHeroCount / 2)
	}
	var corrected [AssaultHeroCount]float64
	var correctedMean [2]float64
	for i := 0; i < AssaultHeroCount; i++ {
		team := i / (AssaultHeroCount / 2)
		corrected[i] = raw[i] - teamMean[1-team]
		correctedMean[team] += corrected[i] / (AssaultHeroCount / 2)
	}
	for i := 0; i < AssaultHeroCount; i++ {
		team := i / (AssaultHeroCount / 2)
		r.Reward[i] = float32(0.8*corrected[i] + 0.2*correctedMean[team])
	}
	if r.Done && e.inst.dota.ended {
		e.terminalRewarded = true
	}
	return r
}

func (e *AssaultEnv) assaultObjectivePotentialLocked(team int32) float64 {
	var potential float64
	for _, structure := range e.inst.dota.m.Structures {
		mob := e.inst.mobs[dotaStructIDBase+structure.ID]
		weight := 2.25
		switch structure.Role {
		case gamedata.DotaCreepTower:
			weight = 6
		case gamedata.DotaAltar:
			weight = 5
		}
		lost, dead := 1.0, true
		if mob != nil {
			lost = 1 - float64(safeFrac(mob.hp, mob.maxHealth()))
			dead = mob.dead
		}
		value := weight * (2.0 / 3.0) * lost
		if dead {
			value += weight / 3.0
		}
		structureTeam := dotaTeamHuman
		if structure.Side == gamedata.DotaSideElf {
			structureTeam = dotaTeamElf
		}
		if structureTeam != team {
			potential += value
		}
	}
	return potential
}

func safeFrac(v, max float64) float32 {
	if max <= 0 {
		return 0
	}
	return float32(math.Max(0, math.Min(1, v/max)))
}

func boolFloat(v bool) float32 {
	if v {
		return 1
	}
	return 0
}
