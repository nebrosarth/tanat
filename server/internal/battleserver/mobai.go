package battleserver

import (
	"cmp"
	"log"
	"math"
	"slices"
	"time"

	"tanatserver/internal/amf"
	"tanatserver/internal/battleproto"
	"tanatserver/internal/gamedata"
)

// Combat tick: a single 200ms loop per hunt connection drives everything
// timed — mob AI, DoT/HoT streams, buff expiry, toggles/auras, traps,
// channels, summons, regen and the player's death/respawn. One loop instead
// of scattered AfterFuncs keeps ordering deterministic under mvMu.

const (
	tickInterval = 200 * time.Millisecond

	// dotTickInterval is how often a DoT (poison) deals its slice of damage. It
	// matches the combat tick (5x/sec) so the health bar bleeds down smoothly
	// instead of lurching once per second. Damage numbers are throttled separately
	// (see the DoT stream in tickMobsLocked) so the fine cadence doesn't spam them.
	dotTickInterval = 0.2

	mobAggroRange = 9.0  // mobs turn hostile inside this radius
	mobLeashRange = 22.0 // and give up beyond this distance from the player

	// Spawn leash: a homed mob may not stray more than this from its OWN spawn point while
	// chasing (independent of the player-relative mobLeashRange -- whichever trips first
	// wins). A boss is pinned to its arena so it cannot be kited away (2m, user spec «у
	// боссов радиус 2 метра»); regular trash may roam roughly a pack's width before it
	// turns back. Past it the mob gives up and walks home, and does not re-aggro until it
	// is back within the radius, so it does not thrash at the edge.
	bossSpawnLeash = 2.0
	mobSpawnLeash  = 12.0

	// The leash-home HP-regen rate (fraction of max HP per second while walking home) is
	// the admin knob gamedata.MobHPRegenFrac(); its default 0.4 (~40%/s) tops a mob off
	// over the walk back from leash range so it arrives (and re-reveals to a returning
	// player) at full health. This is NOT in-combat regen -- a mob never heals while it
	// is still fighting.
	//
	// mobHomeEpsilon: a returning mob within this of its spawn is considered home.
	mobHomeEpsilon = 0.3

	// respawnEvictRange: on player respawn, any mob within this of the checkpoint is
	// sent home. Leash is player-relative, so a pack chased onto a checkpoint would
	// otherwise freeze there and re-aggro the fresh-respawned player (a spawn-camp
	// death loop). Pack spawns sit >~10m from a checkpoint by construction, so a mob
	// sent home lands outside mobAggroRange -- the checkpoint stays the safe rally
	// point the mob-free spawn ring (dungeonRebornClear) intends.
	respawnEvictRange = 14.0

	// meleeReach is the base edge-to-edge gap a melee swing crosses; the real
	// reach adds BOTH bodies' radii (attacker + target), matching the client's
	// AvatarAI (reach = range + attacker.mRadius + target.mRadius). A ranged
	// mob's own AttackRange replaces this base.
	meleeReach = 0.6

	summonSeek            = 12.0 // summons pick fights inside this radius
	stationaryTurretRange = 8.0  // a stationary summon (Morlokai's totem) zaps enemies inside this
	summonRing            = 2.8  // escort radius: summons hold a ring slot AROUND the follow point
	summonSpawnRadius     = 1.2  // tighter ring the burst spawns on, around its cast point
	summonSlotTol         = 0.6  // deadband: don't re-issue a follow until this far off the slot
	summonRadius          = 0.5  // body radius of a summoned crawler (has no gamedata.Mob entry)
	// summonSpeed is the summon's run speed. Deliberately above the avatar's
	// lobbyMoveSpeed (4.0) and on par with a normal mob (~4.2-4.6) so a pet can
	// actually catch up to and keep pace with its moving owner and its prey.
	summonSpeed = 4.4

	// mobSepRange is the desired minimum centre-to-centre spacing: mobs closer
	// than this push apart so a pack surrounds the target instead of stacking on
	// one spot. Kept below the effective melee reach so a spread pack still
	// reaches the target. mobSepWeight scales the push against the chase direction.
	//
	// It is a FLOOR, not the whole rule -- see bodySeparation, which widens it to the
	// two bodies' radii. Flat spacing was fine while every body was a same-sized Hunt
	// mob; it is not, once an altar is 3.0m across.
	mobSepRange  = 1.8
	mobSepWeight = 0.9
	// sepMargin is the breathing room added on top of two bodies' radii: the gap they
	// settle at when nothing but each other acts on them. Deliberately small -- this is
	// a collision rule ("do not stand inside each other"), not a formation.
	sepMargin = 0.35
	// mobSepStep is the deadband on the in-place sidestep: a unit already engaging a
	// target sidesteps a crowding neighbour only once the push clears this, then holds.
	// Without the deadband it would jitter at the tick rate forever; with it, a spread
	// pack falls under threshold and goes silent. mobSidestepFrac is the reduced speed of
	// that shuffle -- it reads as a shuffle, not a charge, and does not outrun the strike.
	// One rule, shared by both drivers (Hunt's in-range arm and «Штурм»'s).
	mobSepStep      = 0.5
	mobSidestepFrac = 0.6

	// mobArrowSpeed is a ranged mob's projectile speed (world units/sec) used to size
	// the arrow's flight from the release point to the target. Fast, like a real arrow.
	// minArrowFlight/maxArrowFlight bound it so a point-blank shot still animates a
	// visible streak and a long shot never dawdles; the flight is additionally capped
	// to land before the next swing so the attack cycle stays clean.
	mobArrowSpeed  = 26.0
	minArrowFlight = 0.12
	maxArrowFlight = 0.35

	// Fog of war: a mob is only created on the client while the player is within
	// mobRevealRadius, and removed again past mobHideRadius (hysteresis avoids
	// flicker at the boundary). Both exceed mobLeashRange (22) so a mob is always
	// visible before it can aggro and until it has walked home. Server-side so it
	// works without the client's scene-driven fog plane.
	mobRevealRadius = 28.0
	mobHideRadius   = 34.0

	// Graded client fog: a revealed mob is rendered translucent while it sits in
	// the outer ring of the reveal range (a "DeathShader" 50%-alpha overlay via
	// the InvisibilityEffect fx) and snaps to full opacity as the player closes
	// in. The two radii give the shade toggle its own hysteresis, and both are
	// below mobRevealRadius so a freshly-revealed distant mob always appears
	// shaded first. Purely visual; server-driven, no client change.
	mobShadeRadius   = 24.0 // become shaded at/beyond this distance
	mobUnshadeRadius = 22.0 // clear shade once nearer than this
	mobShadeFx       = "InvisibilityEffect"

	// Level-one value retained for tests that construct a generic dead avatar
	// directly. Player deaths use dotaHeroRespawnDelay, which follows Dota's
	// level table below rather than this fixed value.
	respawnDelay = 6.0
	corpseHide   = 3.0 // seconds from death to SYNC-remove (death anim shows)

	// Fountain regen: a LIVING player standing near their respawn point (base/checkpoint)
	// recovers fast, ON TOP of the normal trickle. «Штурм» (DOTA) heals a big fraction of
	// max per second so a player who walks back to base is topped off in ~2s (a MOBA
	// fountain); «Охота» gives a flat +10 HP/s, +5 mana/s at a checkpoint. NOT applied in
	// «Арена» -- its respawn points are transient combat spawns, not a safe base.
	fountainRegenRadius    = 8.0  // within this of the respawn point to get the bonus
	dotaFountainFracPerSec = 0.5  // «Штурм»: +50% of max HP/mana per second (~2s to full)
	huntFountainHPPerSec   = 10.0 // «Охота»: +10 HP/sec
	huntFountainManaPerSec = 5.0  // «Охота»: +5 mana/sec

	// Mob respawn: every creature (trash and boss) revives mobRespawnDelay after
	// death at its authored spawn point, so the location can be farmed. The corpse
	// is removed from the client corpseDeleteDelay after death (death anim plays),
	// then the mobState waits, hidden, until the respawn timer elapses.
	mobRespawnDelay   = 300.0 // 5 minutes
	corpseDeleteDelay = 4 * time.Second
)

// trapState is one armed ground trap.
type trapState struct {
	x, y      float32
	radius    float64
	until     float64
	slot      int
	level     int
	ops       []gamedata.Op
	fx        int32
	triggerFx string
	// anchor is the invisible stationary object the SELF-mode trap fx is parented to
	// (0 = the fx is owned by the caster directly). Deleted when the trap ends.
	anchor int32
}

// channelState is one running channeled skill.
type channelState struct {
	slot, level int
	until       float64
	interval    float64
	nextPulse   float64
	target      int32
	px, py      float32
	hasPos      bool
	ops         []gamedata.Op
	// interruptible marks a ground-anchored channel the caster actively sustains
	// (Elgorm's «Стрелы Аркана») rather than a planted fire-and-forget ground
	// effect (Titanid's quake): it breaks the moment the caster acts again -- moves,
	// is stunned, or casts another skill. See channelInterruptible.
	interruptible bool
	// breakDist > 0 makes a UNIT channel a LEASH (Inshari's «Изъятие сущности» siphon):
	// if the target ever gets further than breakDist from the caster the contact snaps and
	// the target is stunned for stunOnBreak seconds. Running to full duration ends it with
	// NO stun. 0 = no leash.
	breakDist   float64
	stunOnBreak float64
	// growth/stunGrowth/radiusGrowth/pulseCount implement Op.Growth/Op.RadiusGrowth
	// (Miriam's «Убийственный залп» ramps damage; Titanid's «Землетрясение» ramps stun
	// duration and widens the AoE): each pulse adds growth×pulseCount bonus damage,
	// stunGrowth×pulseCount bonus stun seconds, and radiusGrowth×pulseCount extra AoE
	// radius before firing, then increments pulseCount. All-zero is the flat (no ramp)
	// back-compat default.
	growth       float64
	growthSP     float64 // Op.GrowthPerSP: growth×pulseCount also gets a growthSP×pulseCount×spellPower term
	stunGrowth   float64
	radiusGrowth float64
	pulseCount   int
	// growthPerEnemy/growthRadius implement Op.GrowthPerEnemy (Avrora's «Освященное
	// место»): each pulse adds growthPerEnemy × the live enemy count within growthRadius
	// of the pulse's centre. 0 = no occupancy scaling (back-compat default).
	growthPerEnemy float64
	growthRadius   float64
	// pulsesLeft implements Op.Count on a channel: an exact number of pulses (Einzenhaim's
	// «Пальба навскидку» -- the client's own tooltip counts them, «{shots} выстрелов»).
	// Counting is what the skill IS; deriving the count from Dur/Interval leaves it at the
	// mercy of where the 0.2s tick grid happens to fall. 0 = duration-bounded (the default
	// every other channel keeps).
	pulsesLeft int
	// fxUIDs are the live client effects this channel owns (its cast fx and payload fx).
	// A channel's visual does NOT self-terminate: the client's baked prefabs for these
	// skills carry mLoopAnimation and a looping VisualEffect (Einzenhaim's Cast02 plus a
	// PrefabTimeSpawn that keeps spawning shots; Morlokai's Cast02 plus the SELF_TO_TARGET
	// beam), so they run until an EFFECT_END arrives. When a channel BREAKS early -- the
	// caster walks off or is stunned -- the server must send those ends, or the animation
	// and the shots keep playing with no damage behind them.
	fxUIDs []int32
	// holdsTarget marks a channel whose ROOT/SILENCE on its target is the channel itself
	// (Morlokai's «Кабала»: «Удерживает цель с помощью магии... в течение {duration} секунд»).
	// Those are applied once at cast with their full duration, so without this an
	// interrupted hold kept the victim pinned for the rest of the nominal window while the
	// caster had already walked away. Breaking the channel releases the grip.
	holdsTarget bool
}

// summonState is one allied summoned unit. In a shared world it is rendered by
// every ready member (owner + teammates), each in its own tracker index, so proto
// (the party-wide prototype id) is stored to rebuild the model on a newcomer's
// client.
// alive is THE liveness test for a summon: not reaped, not at zero HP, not expired.
// tickSummonsLocked's own reap uses exactly these three, and `dead` alone is not enough
// -- it is set lazily on the owner's next tick, so a pet can be at 0 HP or past its
// expiry and still read dead==false for the rest of the pass. Every scan that can pick a
// summon must agree, or one of them targets a corpse.
func (sm *summonState) alive(now float64) bool {
	return !sm.dead && sm.hp > 0 && sm.until >= now
}

type summonState struct {
	id    int32
	proto int32
	// team is the ABSOLUTE side this summon fights for: its owner's side, fixed at
	// creation (c.playerTeam()). It is what the summon renders with (syncTeam) so every
	// client resolves friend/foe from the owner's side -- an Elf-side («Штурм» team 2)
	// player's summon must show as team 2, ENEMY to the Human side, FRIEND to its owner --
	// and the side it must NOT attack. Hunt/solo/Human owners are team 1, unchanged.
	team        int32
	hp, maxHP   float64
	dmg         float64
	until       float64
	x, y        float32
	vx, vy      float32
	fang        float64 // fixed formation angle: this summon's slot on the ring around its anchor
	nextSwing   float64
	swingDoneAt float64 // battle-time to send ACTION_DONE closing the current swing
	dead        bool
	// posSyncAt is when this summon's velocity was last broadcast. Separation nudges the
	// heading a little every tick, so syncing on any change at all would put one POSITION
	// per pet per tick on the wire; this bounds the client's dead-reckoning drift instead.
	posSyncAt float64

	// A PET (gamedata.Op.Pet -- Grimlok's dinosaur) takes ORDERS from its owner instead
	// of seeking the nearest enemy and escorting them. slot is the skill that summoned
	// it, which is also its identity: re-casting that slot replaces it.
	//
	// The order is latched HERE rather than read off the owner each tick, because the
	// owner's own order state does not survive: hs.attackTarget clears the moment their
	// target dies and c.hasDest clears the moment THEY arrive -- while the pet, slower
	// and further back, is still walking. The pet owns its orders.
	pet        bool
	slot       int
	ordTarget  int32 // enemy the owner ordered it onto (0 = none)
	ordX, ordY float32
	ordMove    bool // walking to an ordered ground point

	// stationary (Morlokai's «Грозовой тотем»): a turret that never moves. It holds its
	// spawn point and zaps the nearest enemy within stationaryRange instead of walking into
	// melee, so tickSummonsLocked skips all of its seek/escort/separation movement.
	stationary bool

	// atkSpeedMul/atkSpeedMulUntil implement an OpBuffStat On:"own_summons" attack-speed
	// buff on this summon (Anhel's «Гнев океана»: «себе, и всем своим клонам»). 0/expired
	// = the normal 1-swing/sec cadence.
	atkSpeedMul      float64
	atkSpeedMulUntil float64
	// moveSpeedMul/moveSpeedMulUntil is the same idea for MOVE speed (Grimlok's «Дикость»:
	// the same buff also speeds up his live dinosaur). 0/expired = summonSpeed unmodified.
	moveSpeedMul      float64
	moveSpeedMulUntil float64

	// onHitOps fire on every landing basic attack this summon lands (Frost's ice elemental:
	// «Каждая атака элементаля замедляет врага... стан, если атакует врага под ознобом»).
	// Copied from the spawning Op.Ops at cast time; nil = a plain damage-only summon.
	onHitOps []gamedata.Op

	// ranged marks a summon whose CLIENT prefab carries a projectile, so its attack must be
	// announced with SET_PROJECTILE (VisualBattle.OnSetProjectile -> VisualEffectOptions.
	// ShotProjectile) instead of resolving silently. Without it the model plays its swing
	// and the victim just loses health -- the «электро шар не имеет анимации атаки» report:
	// the totem's bolt, and its sound, live entirely on that projectile prefab.
	// projHitAt/projTarget are the committed arrival of a bolt already in flight.
	ranged     bool
	projHitAt  float64
	projTarget int32
	// swingHitAt/swingTarget are the COMMITTED end of a wind-up: a summon used to connect
	// the instant its swing started, so the victim lost health while the model was still
	// raising its arm. The strike now lands (or, for a shooter, the bolt is loosed) near
	// the end of the swing, the way a mob's melee connects mid-swing rather than at once.
	swingHitAt  float64
	swingTarget int32

	pf pathState // routed chase waypoints when the straight line is wall-blocked
}

// summonRangedPrefabs are the summonable unit prefabs whose VisualEffectOptions declares a
// non-empty mProjectiles array -- read straight out of the shipped bundles, not guessed:
//
//	Avtr_Dsb_Morlokay_Skill4_prop01  LIGHTNING2      Avtr_Dsb_Morlokay_projectile_prop02
//	Avtr_Dsb_Frost_Elemental         PhysiqueDamage  Avtr_Dsb_Frost_projectile_prop01
//	Avtr_Psh_Anhel                   LIGHTNING       VFX_Avtr_Psh_Anhel_Attack_prop01
//
// Every other summonable prefab (Urg's totem and tree, the dinosaur, the zombie crawler,
// Lirvein's trap) has an empty projectile list and stays a silent melee striker.
var summonRangedPrefabs = map[string]bool{
	"Avtr_Dsb_Morlokay_Skill4_prop01": true,
	"Avtr_Dsb_Frost_Elemental":        true,
	"Avtr_Psh_Anhel":                  true,
}

// summonBoltSpeed is how fast a summon's projectile flies (world units/sec), matching the
// avatar bolt cadence in scheduleProjectileLocked (gap/24 + 0.1). The client needs the hit
// time to be strictly in the FUTURE -- OnSetProjectile drops the shot when
// mHitTime <= battle time -- so the floor is what guarantees the bolt is ever drawn.
const (
	summonBoltSpeed     = 24.0
	summonBoltMinFlight = 0.12
	// summonStrikeFrac is where in the swing CYCLE the blow actually connects, as a
	// fraction of the period. 0.7 puts it late in the animation -- «в конце анимации» --
	// while still landing before the ACTION_DONE that closes the swing at 0.9s, so the
	// next swing can re-trigger cleanly.
	summonStrikeFrac = 0.7
)

// pathState caches an A* route a tick-driven chaser (mob/summon) is following:
// the remaining waypoints, the cursor into them, the anchor of the current leg
// (fromX/Y — the point we last steered from, for pass-detection), the goal the
// route was computed for (to detect the target drifting) and when it was
// computed (to bound staleness). Used only while the straight line is blocked.
type pathState struct {
	pts          []gamedata.Vec2
	idx          int
	fromX, fromY float32
	goalX, goalY float32
	at           float64
	tried        bool // a route has been computed at least once (gates the first attempt)
}

// aimAlong returns the point a chaser at (fx,fy) should steer toward to reach
// (tx,ty). With a clear line (or no nav) it heads straight at the target and
// drops any cached route. While blocked it (re)computes an A* route — when the
// cache is empty, exhausted, stale (>1s), or the target has drifted >2m — then
// advances through the waypoints as they are reached, returning the current one.
//
// A waypoint is retired when the chaser is within 0.6m of it OR has moved at/past
// it along the travel direction (a dot-product test). The projection test makes
// advancement independent of per-tick step length, so a fast/hasted unit that
// overshoots a waypoint still advances instead of oscillating around it, without
// loosening the corner radius (which would let it cut into a wall and stick).
func (c *conn) aimAlong(ps *pathState, fx, fy, tx, ty float32, blocked bool, now float64) (float32, float32) {
	if c.nav == nil || !blocked {
		ps.pts = nil
		return tx, ty
	}
	// (Re)compute the route when it is exhausted, stale (>1s), or the target has
	// drifted >2m. Crucially, an EMPTY route (Path returned nil for a target with
	// no walkable route — e.g. walled off within leash range) is gated by the
	// staleness throttle as well, so it retries at most once per second instead of
	// re-running a worst-case component-exhausting A* on every 200ms tick.
	stale := now-ps.at > 1.0
	drift := math.Hypot(float64(tx-ps.goalX), float64(ty-ps.goalY)) > 2.0
	recompute := !ps.tried || stale // first attempt is unconditional; empty retries gated by staleness
	if len(ps.pts) > 0 {
		recompute = ps.idx >= len(ps.pts) || stale || drift
	}
	if recompute {
		ps.pts = c.nav.Path(float64(fx), float64(fy), float64(tx), float64(ty))
		ps.idx = 0
		ps.fromX, ps.fromY = fx, fy
		ps.goalX, ps.goalY = tx, ty
		ps.at = now
		ps.tried = true
	}
	for ps.idx < len(ps.pts) {
		wp := ps.pts[ps.idx]
		dxw := float64(fx) - wp.X
		dyw := float64(fy) - wp.Y
		reached := dxw*dxw+dyw*dyw < 0.36 // within 0.6m
		if !reached {
			// Passed it? travel dir = wp - from; (chaser - wp)·dir >= 0 means at/beyond wp.
			tdx := wp.X - float64(ps.fromX)
			tdy := wp.Y - float64(ps.fromY)
			if tdx*tdx+tdy*tdy > 1e-6 && dxw*tdx+dyw*tdy >= 0 {
				reached = true
			}
		}
		if reached {
			ps.fromX, ps.fromY = float32(wp.X), float32(wp.Y)
			ps.idx++
			continue
		}
		return float32(wp.X), float32(wp.Y)
	}
	return tx, ty // route failed or fully consumed — head straight (clamp keeps it legal)
}

// aimAlongAStar is the strict route-following variant used by «Штурм» creeps.
// Unlike the generic Hunt chaser, a lane creep must not fall back to a direct
// heading when its target is behind geometry: that fallback is exactly how a
// creep could enter a forest pocket and then grind against the exit wall.
//
// Nav.Path still has its own clear-line fast path, so an unobstructed leg is one
// waypoint, but every leg is validated by the map path service and a failed
// route means "hold", never "walk straight through the obstacle".
func (c *conn) aimAlongAStar(ps *pathState, fx, fy, tx, ty float32, now float64) (float32, float32, bool) {
	if c.nav == nil {
		return 0, 0, false
	}
	stale := now-ps.at > 1.0
	drift := math.Hypot(float64(tx-ps.goalX), float64(ty-ps.goalY)) > 2.0
	recompute := !ps.tried || stale
	if len(ps.pts) > 0 {
		recompute = ps.idx >= len(ps.pts) || stale || drift
	}
	if recompute {
		ps.pts = c.nav.Path(float64(fx), float64(fy), float64(tx), float64(ty))
		ps.idx = 0
		ps.fromX, ps.fromY = fx, fy
		ps.goalX, ps.goalY = tx, ty
		ps.at = now
		ps.tried = true
	}
	for ps.idx < len(ps.pts) {
		wp := ps.pts[ps.idx]
		dxw := float64(fx) - wp.X
		dyw := float64(fy) - wp.Y
		reached := dxw*dxw+dyw*dyw < 0.36
		if !reached {
			tdx := wp.X - float64(ps.fromX)
			tdy := wp.Y - float64(ps.fromY)
			if tdx*tdx+tdy*tdy > 1e-6 && dxw*tdx+dyw*tdy >= 0 {
				reached = true
			}
		}
		if reached {
			ps.fromX, ps.fromY = float32(wp.X), float32(wp.Y)
			ps.idx++
			continue
		}
		return float32(wp.X), float32(wp.Y), true
	}
	// A route can be consumed while the creep is already inside the goal cell.
	// Holding here is safe; the next movement decision will recompute if the goal
	// has moved or the route has become stale.
	return fx, fy, true
}

// procState is a registered passive on-hit proc.
type procState struct {
	slot   int
	chance gamedata.PerLevel
	ops    []gamedata.Op
	// cd (from the skill's TipArgs["cooldown"]) is the passive's INTERNAL cooldown: after a
	// successful proc it cannot fire again for cd seconds, and the client greys the skill
	// button until readyAt (the missing «уходит на перезарядку» feedback). nil/0 = no cd.
	cd      gamedata.PerLevel
	readyAt float64
	// meleeOnly implements Op.MeleeOnly on a defense (OnDamaged) proc: it rolls only when
	// the striking mob is melee (BlackDragon's «Кровь дракона»).
	meleeOnly bool
	// basicAttackOnly/skillOnly implement Op.BasicAttackOnly/Op.SkillOnly on a defense
	// (OnDamaged) proc: they roll only for a basic-attack hit (Gektor's «Реванш») or only
	// for a mob/boss SKILL hit (Neirofim's «Обращение энергии»), read from
	// huntState.lastDamageWasSkill at roll time.
	basicAttackOnly bool
	skillOnly       bool
}

// memberTickLocked runs one 200ms step of a single player's own upkeep: timed
// fx/payloads, statuses, toggles, traps, channels, summons, death/respawn and
// regen. The shared mob simulation is NOT here -- the instance ticker runs it
// exactly once per world (tickMobsLocked) after stepping every member, so mobs
// aren't double-simulated by a multi-player party. Caller holds the world lock.
func (s *Server) memberTickLocked(c *conn, now float64) {
	hs := c.huntState

	// 1. Timed fx ends and scheduled packets.
	var fxKeep []fxEnd
	for _, f := range hs.fxEnds {
		if f.at > now {
			fxKeep = append(fxKeep, f)
			continue
		}
		s.fxEndLocked(c, f.uid)
		// What it turns into when it lapses (Frost's ice block shattering, with its own
		// Frost_skill3_break sound), on the same owner and at the same instant.
		if f.then != "" && f.thenOwner != 0 {
			endUID := s.fxStartLocked(c, f.then, f.thenOwner, 0, false, 0, 0)
			hs.scheduleFxEnd(endUID, now+2.0)
		}
	}
	hs.fxEnds = fxKeep

	s.runDuePayloadsLocked(c, now)

	var adKeep []actionDone
	for _, ad := range hs.actionDones {
		if ad.at > now {
			adKeep = append(adKeep, ad)
			continue
		}
		doneArgs := amf.NewArray().
			Set("id", c.objID).
			Set("action", ad.action).
			Set("item", false).
			Set("cooldown", ad.cooldown)
		// Close the cast action on this player AND on teammates (mirrors execCast) so
		// the remote avatar's DoingAction clears -- otherwise the skill action lingers
		// in the teammate's mCurActions.
		s.pushAvatarAllLocked(c, battleproto.CmdActionDone, doneArgs)
		if ad.order {
			s.orderDoneLocked(c, ad.action)
			// A finished ability flows straight back into auto-attack, mirroring how
			// a kill rolls the avatar onto the next mob -- unless it started a channel
			// the caster is meant to keep sustaining (noResume).
			if !ad.noResume {
				s.resumeAutoAttackLocked(c, now, ad.resumeTarget)
			}
		}
	}
	hs.actionDones = adKeep

	// 2. Pending approach-cast.
	s.tickOrderLocked(c, now)

	// 3. Player statuses.
	s.tickPlayerStatusLocked(c, now)
	// 3a. Move-charge stack (Sandariel's «Острие странника»): bank a charge every
	// chargeDist units walked, up to the rank's cap; consumed on the next basic attack.
	s.tickMoveChargeLocked(c, now)

	// 3b. Respawn checkpoints: activate the Reborn_point the player walks onto.
	s.tickRebornLocked(c, now)

	// 4. Toggles: mana drain + aura pulses.
	s.tickTogglesLocked(c, now)
	// 4b. Always-on passive auras (e.g. BlackDragon's «Крылья тьмы»).
	s.tickPassiveAurasLocked(c, now)
	// 4c. Keep the number on a dynamic passive's buff icon current (Velial's «Воля к
	// победе» bonus tracks missing HP).
	s.refreshPassiveBuffCountersLocked(c, now)

	// 5. Traps (+ deferred trap-anchor cleanup).
	s.tickTrapsLocked(c, now)
	s.tickAnchorEndsLocked(c, now)
	// 5b. Vision wards (Urg's «Росток»): expire past-lifetime props + reveal stealthed
	// enemies within range.
	s.tickWardsLocked(c, now)

	// 6. Channels.
	s.tickChannelsLocked(c, now)

	// 6b. Bounces (chaining skull).
	s.tickBouncesLocked(c, now)

	// 7. Summons.
	s.tickSummonsLocked(c, now)

	// 8. Death/respawn.
	if hs.deadUntil > 0 {
		if !hs.corpseHidden && now >= hs.diedAt+corpseHide {
			hs.corpseHidden = true
			if idx := hs.tr.remove(c.objID); idx >= 0 {
				s.push(c, battleproto.CmdSync, amf.NewArray().Set("data",
					newSyncBlob(float32(now)).removeIndex(idx).build(hs.tr.count())))
			}
			// Drop this corpse from every teammate's client too, so a dead player
			// doesn't linger as a frozen body on their screen.
			for _, other := range c.members() {
				if other != c {
					s.removeAvatarForLocked(other, c)
				}
			}
		}
		if now >= hs.deadUntil {
			s.respawnPlayerLocked(c, now)
		}
	}

	// 10. Regen -- every combat tick, proportional to elapsed time, so the
	// bar creeps up smoothly instead of jumping once every couple seconds
	// (matches the mob leash-return regen below). A living player standing on their
	// respawn point (base/checkpoint) adds a FOUNTAIN bonus on top -- fast in «Штурм»,
	// a flat +10 HP/s +5 mana/s in «Охота».
	if hs.deadUntil == 0 {
		dt := tickInterval.Seconds()
		var types []uint64
		maxHP := hs.maxHPLocked(now)
		maxMana := hs.maxManaLocked(now)
		hpRate := hs.av.HealthRegen + hs.st.modSum(now, "hp_regen")
		manaRate := hs.av.ManaRegen + hs.st.modSum(now, "mana_regen")
		if s.atRespawnFountainLocked(c, now) {
			switch {
			case c.inst != nil && c.inst.dota != nil: // «Штурм» base fountain: ~2s to full
				hpRate += maxHP * dotaFountainFracPerSec
				manaRate += maxMana * dotaFountainFracPerSec
			case c.inst == nil || c.inst.arena == nil: // «Охота» checkpoint (never «Арена»)
				hpRate += huntFountainHPPerSec
				manaRate += huntFountainManaPerSec
			}
		}
		if hs.hp < maxHP {
			hs.hp = math.Min(maxHP, hs.hp+dt*hpRate)
			types = append(types, syncHealth)
		}
		if hs.mana < maxMana {
			hs.mana = math.Min(maxMana, hs.mana+dt*manaRate)
			types = append(types, syncMana)
		}
		if len(types) > 0 {
			s.syncSelfLocked(c, types...)
		}
	}
}

// atRespawnFountainLocked reports whether a LIVING player is standing near their respawn
// point -- the active checkpoint / team base (hs.respawnX/Y, kept current by
// tickRebornLocked), falling back to the map spawn before any checkpoint is reached.
// Drives the fountain regen bonus.
func (s *Server) atRespawnFountainLocked(c *conn, now float64) bool {
	hs := c.huntState
	hx, hy := hs.respawnX, hs.respawnY
	if hx == 0 && hy == 0 {
		sx, sy := hs.m.Spawn()
		hx, hy = float32(sx), float32(sy)
	}
	px, py := c.posAtLocked(float32(now))
	return math.Hypot(float64(px-hx), float64(py-hy)) <= fountainRegenRadius
}

// ---- player status upkeep ----

func (s *Server) tickPlayerStatusLocked(c *conn, now float64) {
	hs := c.huntState
	st := &hs.st

	// Expired mods: reverse icons/fx and re-sync stats.
	var keep []statMod
	changed := false
	for _, m := range st.mods {
		if m.until == 0 || m.until > now {
			keep = append(keep, m)
			continue
		}
		changed = true
		if m.buffEffID != 0 {
			s.push(c, battleproto.CmdRemEffector, amf.NewArray().Set("id", m.buffEffID))
		}
		s.fxEndLocked(c, m.fxUID)
	}
	st.mods = keep
	if changed {
		s.pushPlayerStatsLocked(c, now)
	}

	if st.shieldFx != 0 && (now >= st.shieldUntil || st.shield <= 0) {
		s.fxEndLocked(c, st.shieldFx)
		st.shieldFx = 0
	}
	if st.silenceFx != 0 && now >= st.silenceUntil {
		s.fxEndLocked(c, st.silenceFx)
		st.silenceFx = 0
		s.syncSelfIntLocked(c, syncSilence, 0)
	}
	if st.slowFx != 0 && now >= st.slowUntil {
		s.fxEndLocked(c, st.slowFx)
		st.slowFx = 0
		s.pushPlayerStatsLocked(c, now)
	}
	// Invisibility potion expiry: end the shared shade fx + buff icon.
	if hs.invisFxUID != 0 && now >= hs.invisibleUntil {
		s.worldFxEndLocked(c, hs.invisFxUID)
		hs.invisFxUID = 0
		if hs.invisBuffEffID != 0 {
			s.push(c, battleproto.CmdRemEffector, amf.NewArray().Set("id", hs.invisBuffEffID))
			hs.invisBuffEffID = 0
		}
	}
	// Urg's tree camouflage that ran its full duration without the avatar moving/acting still
	// leaves tree form (and detonates its reveal burst) on expiry.
	if hs.treeFormBurst != nil && hs.treeFormUntil > 0 && now >= hs.treeFormUntil {
		s.fireTreeFormBurstLocked(c, now)
	}
	// Wilfang's «Засада» that ran its full duration unbroken still detonates its reveal
	// burst on expiry, mirroring Urg's tree form above.
	if hs.stealthBurst != nil && now >= hs.invisibleUntil {
		s.fireStealthBurstLocked(c, now)
	}
	// Revelation potion expiry: end the buff icon (no fx to end, see the doc
	// on huntState.revealInvisibleUntil for why this potion has no other
	// visible effect yet).
	if hs.revealBuffEffID != 0 && now >= hs.revealInvisibleUntil {
		s.push(c, battleproto.CmdRemEffector, amf.NewArray().Set("id", hs.revealBuffEffID))
		hs.revealBuffEffID = 0
	}

	// HoT streams.
	var hotKeep []overTime
	healed := false
	for i := range st.hots {
		h := st.hots[i]
		for h.nextTick <= now && h.nextTick <= h.until {
			hs.hp = math.Min(hs.maxHPLocked(now), hs.hp+h.perSec)
			healed = true
			h.nextTick++
		}
		if h.until > now {
			hotKeep = append(hotKeep, h)
		}
	}
	st.hots = hotKeep
	if healed {
		s.syncSelfLocked(c, syncHealth)
	}
}

// ---- toggles / auras ----

func (s *Server) tickTogglesLocked(c *conn, now float64) {
	hs := c.huntState
	dt := tickInterval.Seconds()
	for slot := 1; slot <= 4; slot++ {
		if !hs.toggleOn[slot-1] {
			continue
		}
		def := hs.skillDef(slot)
		level := int(hs.skillLevel[slot-1])
		if hs.toggleStealthSlot == slot {
			// Astarot's «Слуга тьмы»: keep re-arming stealth every tick so it never lapses
			// while the toggle (and its mana) hold out; toggle-off breaks it immediately.
			hs.invisibleUntil = now + 1.0
		}
		for _, op := range def.Ops {
			if op.Kind != gamedata.OpAura {
				continue
			}
			hs.mana -= skillManaCost(op.TickCost.At(level) * dt)
			if hs.mana <= 0 {
				hs.mana = 0
				s.syncSelfLocked(c, syncMana)
				s.toggleOffLocked(c, slot, now, false)
				break
			}
			if now >= hs.toggleNextPulse[slot-1] {
				hs.toggleNextPulse[slot-1] = now + math.Max(op.Interval, 0.5)
				cx, cy := c.posAtLocked(float32(now))
				for _, m := range c.mobsWithinLocked(cx, cy, op.Radius) {
					ctx := opCtx{slot: slot, level: level, target: m, toggle: true}
					s.applyOpsLocked(c, op.Ops, ctx, now)
				}
				s.syncSelfLocked(c, syncMana)
			}
		}
	}
}

// tickPassiveAurasLocked pulses always-on PASSIVE auras (e.g. BlackDragon's «Крылья
// тьмы» enemy attack-speed aura). Unlike a toggle aura it has no mana cost and no
// on/off state -- it runs whenever the passive is learned and the player is alive.
// Pulse cadence reuses toggleNextPulse[slot-1] (a slot is never both a toggle and a
// passive, so the field is free on a passive slot).
func (s *Server) tickPassiveAurasLocked(c *conn, now float64) {
	hs := c.huntState
	if hs.deadUntil > 0 {
		return
	}
	for slot := 1; slot <= 4; slot++ {
		def := hs.skillDef(slot)
		if def.Type != "PASSIVE" {
			continue
		}
		level := int(hs.skillLevel[slot-1])
		if level < 1 {
			continue
		}
		for _, op := range def.Ops {
			if op.Kind != gamedata.OpAura {
				continue
			}
			if now < hs.toggleNextPulse[slot-1] {
				continue
			}
			hs.toggleNextPulse[slot-1] = now + math.Max(op.Interval, 0.5)
			// An ally-oriented aura (Arianna's «Аура стойкости»: nested On=="allies") buffs
			// every nearby friendly avatar via allyTargetsLocked instead of scanning mobs.
			if len(op.Ops) > 0 && (op.Ops[0].On == "ally" || op.Ops[0].On == "allies") {
				ctx := opCtx{slot: slot, level: level}
				s.applyOpsLocked(c, op.Ops, ctx, now)
				continue
			}
			cx, cy := c.posAtLocked(float32(now))
			for _, m := range c.mobsWithinLocked(cx, cy, op.Radius) {
				ctx := opCtx{slot: slot, level: level, target: m}
				s.applyOpsLocked(c, op.Ops, ctx, now)
			}
		}
	}
}

// ---- traps ----

func (s *Server) tickTrapsLocked(c *conn, now float64) {
	hs := c.huntState
	var keep []trapState
	for _, t := range hs.traps {
		if now > t.until {
			s.fxEndLocked(c, t.fx)
			// Lifetime elapsed with no trigger: no trigger fx follows, so the anchor
			// can be removed at once.
			s.removeTrapAnchorLocked(c, t.anchor, now)
			continue
		}
		var victim *mobState
		for _, m := range c.mobsWithinLocked(t.x, t.y, t.radius) {
			victim = m
			break
		}
		if victim == nil {
			keep = append(keep, t)
			continue
		}
		s.fxEndLocked(c, t.fx)
		// The trigger fx is also SELF-mode, so own it to the same anchor (if any) so it
		// erupts at the trap point, and keep the anchor alive until it plays out.
		fxOwner := c.objID
		if t.anchor != 0 {
			fxOwner = t.anchor
		}
		uid := s.fxStartLocked(c, t.triggerFx, fxOwner, 0, true, t.x, t.y)
		hs.scheduleFxEnd(uid, now+3)
		if t.anchor != 0 {
			hs.anchorEnds = append(hs.anchorEnds, anchorEnd{id: t.anchor, at: now + 3})
		}
		ctx := opCtx{slot: t.slot, level: t.level, target: victim,
			px: t.x, py: t.y, hasPos: true}
		s.applyOpsLocked(c, t.ops, ctx, now)
	}
	hs.traps = keep
}

// ---- channels ----

func (s *Server) tickChannelsLocked(c *conn, now float64) {
	hs := c.huntState
	var keep []channelState
	for _, ch := range hs.channels {
		// A ground-anchored channel (POINT cast: a target-less effect pinned to a
		// world point, e.g. Titanid's «Землетрясение» 3-wave quake or Elgorm's
		// arcane-arrow rain) is fire-and-forget -- it keeps erupting at its point
		// while the caster walks off or is stunned; only death or its own duration
		// ends it. A self/unit channel (e.g. Abominator's «Пожирание» drain) needs
		// the caster to keep channeling, so moving/stun/death breaks it. hasDest is
		// the truthful "is-moving" flag (c.arrival lingers non-nil after a leg).
		groundAnchored := ch.hasPos && ch.target == 0
		broken := hs.deadUntil > 0
		// A self/unit channel, OR an interruptible ground channel (a caster-sustained
		// one like Elgorm's arrow rain), ends the moment the caster moves or is
		// stunned. A plain fire-and-forget ground channel (Titanid's quake) keeps
		// erupting regardless -- and so does a self-buff pulse explicitly marked as one
		// (BlackDragon's «Взмах погибели»: channelSustainsThroughDisruption).
		if (!groundAnchored || ch.interruptible) && !channelSustainsThroughDisruption(hs.av.Prefab, ch.slot) {
			broken = broken || hs.st.stunned(now) || c.hasDest
		}
		// A channel held ON a unit ends when that unit dies. Nothing is being channelled
		// into a corpse -- and for a grip like «Короткое замыкание» or «Кабала» the beam
		// would otherwise keep firing at a body for the rest of its nominal duration.
		var target *mobState
		if ch.target > 0 {
			target = hs.mobs[ch.target]
			if target == nil || target.dead {
				target = s.dotaEnemyHeroShadowLocked(c, ch.target, now)
			}
			if target == nil {
				broken = true
			}
		}
		// Leash break (Inshari's siphon): if the tethered target strays past breakDist the
		// contact snaps, stunning it -- «при сильном отдалении контакт разорвётся, оглушая
		// цель». Only fires on a distance break, never on the natural full-duration end.
		if !broken && ch.breakDist > 0 && ch.target > 0 {
			if target != nil {
				cx, cy := c.posAtLocked(float32(now))
				if math.Hypot(float64(target.x-cx), float64(target.y-cy)) > ch.breakDist {
					if ch.stunOnBreak > 0 {
						if target.heroOwner != nil {
							s.dotaStunHeroLocked(c, target.heroOwner, now, ch.stunOnBreak)
						} else {
							s.stunMobLocked(c, target, now, ch.stunOnBreak)
						}
					}
					continue // drop the channel: contact broken
				}
			}
		}
		if broken || now > ch.until {
			// An early BREAK kills the visual with the mechanic. A natural end does not:
			// the fx already has its own scheduled end (fxLife covers the whole channel),
			// and cutting it here would clip the last spawned shot/arrow mid-flight.
			if broken {
				for _, uid := range ch.fxUIDs {
					s.fxEndLocked(c, uid)
				}
				// ...and lets go of a held victim. The status tick ends the matching
				// StunEffect/SilenceEffect visuals on the next pass, once the timers read
				// expired, so no fx teardown is needed here.
				if ch.holdsTarget && ch.target > 0 {
					if target != nil {
						if target.heroOwner != nil {
							st := &target.heroOwner.huntState.st
							st.stunUntil = math.Min(st.stunUntil, now)
							st.rootUntil = math.Min(st.rootUntil, now)
							st.silenceUntil = math.Min(st.silenceUntil, now)
							st.atkSlowUntil = math.Min(st.atkSlowUntil, now)
						} else {
							target.st.stunUntil = math.Min(target.st.stunUntil, now)
							target.st.rootUntil = math.Min(target.st.rootUntil, now)
							target.st.silenceUntil = math.Min(target.st.silenceUntil, now)
							target.st.atkSlowUntil = math.Min(target.st.atkSlowUntil, now)
						}
					}
				}
			}
			continue
		}
		if now >= ch.nextPulse {
			// Advance the schedule by the interval (accumulate), NOT now+interval.
			// The tick runs on a coarse 0.2s grid; recomputing from the snapped `now`
			// rounds every gap UP to the next tick, so a 0.46s cadence drifts to ~0.6s
			// and loses pulses (Elgorm's arrow rain dropped from 9 ticks to ~7).
			// Accumulating keeps the ideal schedule (0.2, 0.66, 1.12, ...) so each pulse
			// fires on the first tick at/after its ideal time -> the full 9 ticks land.
			//
			// The step is the AUTHORED interval. It used to be floored at 0.4s, which
			// silently rewrote every faster cadence in the specs -- Einzenhaim's 0.25s
			// volley became 0.4s and landed 2 hits instead of 3-5, and three other
			// channels (0.3/0.3/0.55) were quietly stretched too. Accumulating already
			// makes a sub-tick cadence work out: a 0.3s step on the 0.2s grid fires at
			// 0, 0.4, 0.6, 1.0, 1.2 -- individual gaps alternate, but no pulse is lost.
			// Only a non-positive interval needs a floor (it would re-fire every tick).
			step := ch.interval
			if step <= 0 {
				step = 0.4
			}
			ch.nextPulse += step
			dmgBonus := ch.growth * float64(ch.pulseCount)
			if ch.growthSP > 0 {
				dmgBonus += ch.growthSP * float64(ch.pulseCount) * hs.spellPowerLocked(now)
			}
			if ch.growthPerEnemy > 0 {
				cx, cy := ch.px, ch.py
				if !ch.hasPos {
					cx, cy = c.posAtLocked(float32(now))
				}
				dmgBonus += ch.growthPerEnemy * float64(len(c.mobsWithinLocked(cx, cy, ch.growthRadius)))
			}
			ctx := opCtx{slot: ch.slot, level: ch.level, target: target,
				px: ch.px, py: ch.py, hasPos: ch.hasPos,
				dmgBonus:    dmgBonus,
				durBonus:    ch.stunGrowth * float64(ch.pulseCount),
				radiusBonus: ch.radiusGrowth * float64(ch.pulseCount)}
			s.applyOpsLocked(c, ch.ops, ctx, now)
			// A per-wave escalating visual (Titanid's «Землетрясение» waves 2/3), owned to a
			// fresh stationary anchor at the channel's own ground point -- NOT the caster,
			// who this ground-anchored channel lets walk away mid-quake -- so a moving
			// Titanid doesn't drag the SELF-baked crack fx off the actual blast point.
			if fx := channelWavePulseFx(hs.av.Prefab, ch.slot, ch.pulseCount); fx != "" {
				wx, wy := ch.px, ch.py
				if !ch.hasPos {
					wx, wy = c.posAtLocked(float32(now))
				}
				anchor := s.spawnTrapAnchorLocked(c, wx, wy, now)
				uid := s.fxStartLocked(c, fx, anchor, 0, false, 0, 0)
				hs.scheduleFxEnd(uid, now+2.0)
				hs.anchorEnds = append(hs.anchorEnds, anchorEnd{id: anchor, at: now + 2.3})
			}
			ch.pulseCount++
			// Op.Count channel: stop the moment the authored number of pulses has been
			// delivered, without waiting for the duration to lapse.
			if ch.pulsesLeft > 0 {
				ch.pulsesLeft--
				if ch.pulsesLeft == 0 {
					continue
				}
			}
		}
		keep = append(keep, ch)
	}
	hs.channels = keep
}

// ---- bounces (chaining skull, Paralyzing-Cask-style) ----

// skullMoverSpeed replicates the client's SmoothMove mover on the server so hit
// timing matches the visual exactly. The Elgorm skill-1 skull prop
// (VFX_Avtr_Dsb_Elgorm_skill1_prop01) carries a SmoothMove with mBySpeed=true and
// mSpeed=15 (units/sec, read from the prefab in Avtr_Dsb_Elgorm.unity3d): its flight
// time is Distance/mSpeed, independent of the parabolic arc (mMaxHeight) and of the
// target moving (mTargetTime is fixed at launch). skullMinFlight floors near-zero
// hops so a coincident target still shows a brief throw rather than an instant hit.
const (
	skullMoverSpeed = 15.0
	skullMinFlight  = 0.05
)

// skullFlightTime is the server's copy of SmoothMove: seconds for the skull to cross
// the straight-line gap between two points at skullMoverSpeed.
func skullFlightTime(ax, ay, bx, by float32) float64 {
	t := math.Hypot(float64(bx-ax), float64(by-ay)) / skullMoverSpeed
	if t < skullMinFlight {
		t = skullMinFlight
	}
	return t
}

// bounceState is one in-flight chaining skull: it is currently flying toward
// flyingTo and lands its hit at arriveAt, then hops to a RANDOM enemy within radius
// that ISN'T the one it just struck. Like Paralyzing Cask the hop is unpredictable
// and may bounce back to enemies it already hit (so two mobs ping-pong the full step
// budget) -- it only avoids landing on the very mob it is standing on. Fire-and-
// forget once thrown.
type bounceState struct {
	slot, level    int
	remaining      int           // impacts still to land, incl. the skull currently in flight
	arriveAt       float64       // battle-time the in-flight skull reaches flyingTo
	radius         float64       // search radius around the last impact for the next target
	flyingTo       int32         // the mob the skull is flying toward now (damage lands on arrival)
	flyToX, flyToY float32       // that target's position at launch (fallback origin if it dies mid-flight)
	ops            []gamedata.Op // per-hit ops (single-target damage + stun)
	fx             string        // the skull fx spawned for each new hop
}

// startBounceLocked resolves an OpBounce: the skull is thrown from the caster and
// FLIES to the first enemy (the cast target, or nearest to the aim point), then
// keeps hopping to the nearest fresh enemy. Damage lands on IMPACT -- when the skull
// actually reaches each enemy, never on cast -- so every number ticks down as the
// skull touches its target. The first skull fx was already thrown by the payload
// system (PayloadFxAt "target", caster->target); this schedules that first impact,
// and each hop launches its own fx and schedules the next impact on arrival.
//
// The impact times come from skullFlightTime (Distance/skullMoverSpeed), a byte-for-
// byte copy of the client's SmoothMove mover, so server damage and the client visual
// land together at any hop distance.
func (s *Server) startBounceLocked(c *conn, op gamedata.Op, ctx opCtx, now float64) {
	hs := c.huntState
	total := int(op.Count.At(ctx.level))
	if total < 1 {
		total = 1
	}
	first := ctx.target
	if first == nil || first.dead {
		first = s.nearestMobLocked(c, ctx.px, ctx.py, op.Radius, 0)
	}
	if first == nil {
		return // nothing to hit -- the skull fizzles
	}
	cx, cy := c.posAtLocked(float32(now))
	hs.bounces = append(hs.bounces, bounceState{
		slot: ctx.slot, level: ctx.level,
		remaining: total, arriveAt: now + skullFlightTime(cx, cy, first.x, first.y),
		radius: op.Radius, flyingTo: first.id, flyToX: first.x, flyToY: first.y,
		ops: op.Ops, fx: hs.skillDef(ctx.slot).PayloadFx,
	})
}

// nearestMobLocked returns the closest living mob within r of (x,y), skipping
// `exclude` (0 = none), or nil. Used only to resolve the skull's FIRST target when
// the cast had no explicit target (nearest to the aim point).
func (s *Server) nearestMobLocked(c *conn, x, y float32, r float64, exclude int32) *mobState {
	var best *mobState
	bestD := r
	for _, m := range c.huntState.mobs {
		if m.dead || m.id == exclude || !m.enemyOf(c.playerTeam()) {
			continue
		}
		if d := math.Hypot(float64(m.x-x), float64(m.y-y)); d <= bestD {
			bestD, best = d, m
		}
	}
	return best
}

// randomMobInRangeLocked picks a uniformly RANDOM living mob within r of (x,y),
// skipping `exclude` (0 = none), or nil if none qualify. This is the skull's hop
// target rule -- Paralyzing-Cask-style, the chain jumps to a random nearby enemy
// (which may be one it already hit), not the nearest, so its path is unpredictable.
// The choice is made once here on the server and the resulting fx is broadcast, so
// every client sees the same hop sequence.
func (s *Server) randomMobInRangeLocked(c *conn, x, y float32, r float64, exclude int32) *mobState {
	var cands []*mobState
	for _, m := range c.huntState.mobs {
		if m.dead || m.id == exclude || !m.enemyOf(c.playerTeam()) {
			continue
		}
		if math.Hypot(float64(m.x-x), float64(m.y-y)) <= r {
			cands = append(cands, m)
		}
	}
	if len(cands) == 0 {
		return nil
	}
	return cands[s.randIntn(len(cands))]
}

// tickBouncesLocked advances every in-flight chaining skull: when it reaches its
// target it applies its ops, then hops to a random other enemy in range and re-arms
// -- or ends when its hits are spent or no other enemy is in range.
func (s *Server) tickBouncesLocked(c *conn, now float64) {
	hs := c.huntState
	if len(hs.bounces) == 0 {
		return
	}
	var keep []bounceState
	for _, b := range hs.bounces {
		if b.remaining <= 0 {
			continue
		}
		if now < b.arriveAt {
			keep = append(keep, b)
			continue
		}
		// The in-flight skull just reached its target: apply the hit there (skip if it
		// died mid-flight), then note where the skull now sits for the next search.
		originX, originY := b.flyToX, b.flyToY
		if victim := hs.mobs[b.flyingTo]; victim != nil && !victim.dead {
			s.applyOpsLocked(c, b.ops, opCtx{slot: b.slot, level: b.level,
				target: victim, px: victim.x, py: victim.y, hasPos: true}, now)
			originX, originY = victim.x, victim.y
		}
		b.remaining--
		if b.remaining <= 0 {
			continue // every hit landed
		}
		// Launch the skull at the next enemy -- a RANDOM one in range that isn't the mob
		// it just struck (Paralyzing-Cask hop: unpredictable, may revisit earlier
		// enemies, only fizzles when no other enemy is nearby). It flies from the struck
		// enemy and its hit is scheduled for the moment it arrives (Distance/
		// skullMoverSpeed, matching the client SmoothMove), so damage coincides with the
		// impact.
		next := s.randomMobInRangeLocked(c, originX, originY, b.radius, b.flyingTo)
		if next == nil {
			continue // chain fizzles: no other enemy in range
		}
		if uid := s.fxStartLocked(c, b.fx, b.flyingTo, next.id, false, 0, 0); uid != 0 {
			hs.scheduleFxEnd(uid, now+2)
		}
		b.arriveAt = now + skullFlightTime(originX, originY, next.x, next.y)
		b.flyingTo, b.flyToX, b.flyToY = next.id, next.x, next.y
		keep = append(keep, b)
	}
	hs.bounces = keep
}

// ---- summons ----

// allocSummonID returns a globally-unique summon object id. In a shared world the
// id comes from the instance-wide counter so two members' summons can never
// collide (the mob sim resolves hits/kill-credit by summon id across all members).
// A bare/solo conn with no instance falls back to its own per-conn counter.
func (c *conn) allocSummonID() int32 {
	if c.inst != nil {
		c.inst.nextSummonID++
		return c.inst.nextSummonID
	}
	c.huntState.nextSummonID++
	return c.huntState.nextSummonID
}

// summonAttackSpeedMulLocked returns a summon's current attack-speed multiplier: 1 (the
// normal 1-swing/sec cadence) unless an OpBuffStat On:"own_summons" buff (Anhel's «Гнев
// океана») is live on it.
func summonAttackSpeedMulLocked(sm *summonState, now float64) float64 {
	if sm.atkSpeedMul > 0 && now < sm.atkSpeedMulUntil {
		return sm.atkSpeedMul
	}
	return 1
}

// summonMoveSpeedMulLocked mirrors summonAttackSpeedMulLocked for MOVE speed (Grimlok's
// «Дикость» buffing his live dinosaur's run speed alongside its attack speed).
func summonMoveSpeedMulLocked(sm *summonState, now float64) float64 {
	if sm.moveSpeedMul > 0 && now < sm.moveSpeedMulUntil {
		return sm.moveSpeedMul
	}
	return 1
}

// applySummonOnHitOpsLocked runs a summon's own on-hit ops (Frost's ice elemental: slow +
// chill-interaction stun on every landing attack) against the mob it just struck. A no-op
// for any summon without onHitOps (every OTHER summon kind).
func (s *Server) applySummonOnHitOpsLocked(c *conn, sm *summonState, target *mobState, now float64) {
	if len(sm.onHitOps) == 0 {
		return
	}
	level := 1
	if hs := c.huntState; sm.slot >= 1 && sm.slot <= len(hs.skillLevel) {
		if lv := int(hs.skillLevel[sm.slot-1]); lv >= 1 {
			level = lv
		}
	}
	ctx := opCtx{slot: sm.slot, level: level, target: target, px: target.x, py: target.y, hasPos: true}
	s.applyOpsLocked(c, sm.onHitOps, ctx, now)
}

func (s *Server) summonLocked(c *conn, op gamedata.Op, ctx opCtx, now float64) {
	hs := c.huntState
	protoID, ok := hs.summonProtos[op.Unit]
	if !ok {
		return
	}
	n := int(op.Count.At(ctx.level))
	if n < 1 {
		n = 1
	}
	// A pet is UNIQUE: re-casting the skill replaces the one it made last time rather
	// than adding a second. Without this the player just spams the button and fields a
	// pack -- Grimlok's dinosaur lives 180s, so they would all still be alive.
	if op.Pet {
		for id, old := range hs.summons {
			if !old.pet || old.slot != ctx.slot || old.dead {
				continue
			}
			old.dead = true
			old.swingDoneAt = 0
			delete(hs.summons, id)
			s.removeSummonFromClientsLocked(c, old, now)
		}
	}
	// Spawn at the cast point for a ground-targeted summon (Elgorm's «Гвардия
	// мертвых» is aimed where the player clicks); fall back to the caster for a
	// self/untargeted summon.
	cx, cy := c.posAtLocked(float32(now))
	if ctx.hasPos {
		cx, cy = ctx.px, ctx.py
	}
	// DmgPctOfAttack (Lirvein's «Вендетта» daggers) ties each unit's damage to the
	// OWNER's live base attack at cast time instead of a static per-rank Dmg table.
	dmg := op.Dmg.At(ctx.level)
	if op.DmgPctOfAttack > 0 {
		dmg = op.DmgPctOfAttack * hs.baseAttackLocked(now)
	}
	if op.DmgPerSP > 0 {
		dmg += op.DmgPerSP * hs.spellPowerLocked(now)
	}
	// HpPctOfOwner (Anhel's clones) ties each unit's HP to the OWNER's live max HP at
	// cast time instead of a static per-rank HP table.
	hp := op.HP.At(ctx.level)
	if op.HpPctOfOwner > 0 {
		hp = op.HpPctOfOwner * hs.maxHPLocked(now)
	}
	if op.HpPerSP > 0 {
		hp += op.HpPerSP * hs.spellPowerLocked(now)
	}
	for i := 0; i < n; i++ {
		id := c.allocSummonID()
		// Give each summon a fixed slot angle on the ring so the group spawns
		// spread out (tightly, around the cast point) and keeps holding distinct
		// positions around the owner/target instead of stacking on one point.
		ang := float64(i) * 2 * math.Pi / float64(n)
		sx := cx + float32(summonSpawnRadius*math.Cos(ang))
		sy := cy + float32(summonSpawnRadius*math.Sin(ang))
		unitDmg := dmg
		// LastDouble (Lirvein's «Вендетта»): "последний кинжал нанесёт удвоенный урон".
		if op.LastDouble && i == n-1 {
			unitDmg *= 2
		}
		sm := &summonState{
			id: id, proto: protoID, x: sx, y: sy, fang: ang,
			team: c.playerTeam(), // fights for its owner's absolute side (1 Human / 2 Elf)
			hp:   hp, maxHP: hp,
			dmg:   unitDmg,
			until: now + op.Lifetime.At(ctx.level),
			pet:   op.Pet, slot: ctx.slot,
			stationary: op.Stationary,
			// Shoots instead of swinging, if its client prefab carries a projectile.
			ranged: summonRangedPrefabs[op.Unit],
			// Frost's «Исчадие мерзлоты»: the elemental's own on-hit slow/chill-stun combo.
			onHitOps: op.Ops,
		}
		// A pet summoned onto a target inherits that as its first order -- casting the
		// summon at an enemy means "go get it", not "stand there".
		if op.Pet && ctx.target != nil && !ctx.target.dead && ctx.target.enemyOf(c.playerTeam()) {
			sm.ordTarget = ctx.target.id
		}
		hs.summons[id] = sm
		// Render on every ready TEAMMATE, each in its own tracker. Outside an instance
		// c.members() is [c], so this is the owner-only path. In «Штурм» an ENEMY does
		// not get it here -- dotaVisionPassLocked (vision.go) reveals it once genuinely
		// in vision, the same fog delay every other unit gets. Arena has no vision pass
		// yet, so it still reveals to everyone immediately, unchanged.
		dotaFog := c.inst != nil && c.inst.dota != nil
		for _, mem := range c.members() {
			if mem.huntState == nil || (dotaFog && mem.playerTeam() != sm.team) {
				continue
			}
			s.revealSummonToMemberLocked(mem, sm, now)
		}
		// A persistent aura attached to the unit (Frost's ice elemental) -- started once,
		// world-scoped, parented to the summon so it dies with the body. Fired after the
		// reveal loop so the object exists on every client before the fx attaches.
		if op.SummonFx != "" {
			s.fxStartLocked(c, op.SummonFx, sm.id, 0, false, 0, 0)
		}
	}
}

// revealSummonToMemberLocked builds a summon on ONE member's client (tracker add +
// CREATE_OBJECT + stats SYNC + attack effector). No-op if already tracked. Mirrors
// revealMobToMemberLocked -- the summon prototype is party-wide-registered in every
// member's world-state chain, so no PrototypeInfo is needed here.
func (s *Server) revealSummonToMemberLocked(mem *conn, sm *summonState, now float64) {
	hs := mem.huntState
	if hs == nil || hs.tr.index(sm.id) >= 0 {
		return
	}
	idx := hs.tr.add(sm.id)
	frac := float32(1.0)
	if sm.maxHP > 0 {
		frac = float32(math.Max(sm.hp, 0) / sm.maxHP)
	}
	s.push(mem, battleproto.CmdCreateObject, amf.NewArray().
		Set("id", sm.id).Set("proto", sm.proto))
	s.push(mem, battleproto.CmdSync, amf.NewArray().Set("data",
		newSyncBlob(float32(now)).addObject(sm.id).
			position(idx, sm.x, sm.y, sm.vx, sm.vy, float32(now)).
			setFloats(syncHealth, idx, frac).
			setFloats(syncMaxHealth, idx, float32(sm.maxHP)).
			setFloats(syncSpeed, idx, summonSpeed).
			// Attack speed matches the summon's 1/s swing cadence (nextSwing += 1.0).
			// Without it mAttackSpeed=0 on the client freezes the swing clip at playback
			// speed 0 -- the ACTION_DONE re-shows the pose each swing but it never
			// actually animates (same latent bug the ally avatar had).
			setFloats(syncAttackSpeed, idx, 1.0).
			setFloats(syncRadius, idx, float32(summonRadius)).
			// Vision -- a summon is FRIEND to its own side, so on a fog map it spawns a
			// reveal zone for that side. See creepViewRadius.
			setFloats(syncViewRadius, idx, effectiveViewRadius(summonViewRadius)).
			setInt(syncTeam, idx, sm.team).
			build(hs.tr.count())))
	// ATTACK effector so the model swings when we push its ACTIONs.
	s.addAttackEffectorLocked(mem, sm.id, summonAttackProtoID, now)
}

// hideSummonFromMemberLocked removes a summon from ONE member's client (tracker
// swap-remove + SYNC-remove + DELETE_OBJECT). No-op if the member wasn't tracking it.
func (s *Server) hideSummonFromMemberLocked(mem *conn, sm *summonState, now float64) {
	s.untrackObjForMemberLocked(mem, sm.id, float32(now))
}

// removeSummonFromClientsLocked drops a summon from every member's client (expiry,
// death, or owner leave).
func (s *Server) removeSummonFromClientsLocked(c *conn, sm *summonState, now float64) {
	for _, mem := range c.members() {
		s.hideSummonFromMemberLocked(mem, sm, now)
	}
}

// petTargetLocked resolves a pet's current victim from its standing attack order, or
// nil if it has none. A live order survives; a dead/invalid one is RE-AIMED at the
// nearest enemy in seek range rather than dropped, mirroring how the player's own
// auto-attack chains onto the next mob after a kill (hitMobLocked/autoResumed) -- a pet
// that downs its target and then just stands there reads as broken.
//
// Re-aiming is the ONE thing a pet decides for itself, and only ever while it is already
// under an attack order: it never opens a fight the player did not start.
func (s *Server) petTargetLocked(c *conn, sm *summonState, now float64) *mobState {
	hs := c.huntState
	if sm.ordTarget == 0 {
		return nil
	}
	if m := hs.mobs[sm.ordTarget]; m != nil && !m.dead && m.enemyOf(c.playerTeam()) {
		return m
	}
	sm.ordTarget = 0
	var best *mobState
	bestD := summonSeek
	for _, m := range hs.mobs {
		if m.dead || !m.enemyOf(c.playerTeam()) {
			continue
		}
		if d := math.Hypot(float64(m.x-sm.x), float64(m.y-sm.y)); d < bestD {
			bestD, best = d, m
		}
	}
	if best != nil {
		sm.ordTarget = best.id
		sm.ordMove = false
	}
	return best
}

// orderPetsAttackLocked / orderPetsMoveLocked push the owner's order to their pets. The
// two hooks sit on the player's own order paths (startAttackLocked, handleMove), so a
// pet reacts on the same click the avatar does.
func (s *Server) orderPetsAttackLocked(c *conn, targetID int32) {
	hs := c.huntState
	if hs == nil {
		return
	}
	for _, sm := range hs.summons {
		if sm.pet && !sm.dead {
			sm.ordTarget = targetID
			sm.ordMove = false
		}
	}
}

func (s *Server) orderPetsMoveLocked(c *conn, x, y float32) {
	hs := c.huntState
	if hs == nil {
		return
	}
	for _, sm := range hs.summons {
		if sm.pet && !sm.dead {
			// A move order BREAKS the fight, exactly as it does for the avatar itself
			// (handleMove calls stopAttackLocked): the click means "disengage and go".
			sm.ordTarget = 0
			sm.ordX, sm.ordY, sm.ordMove = x, y, true
		}
	}
}

func (s *Server) tickSummonsLocked(c *conn, now float64) {
	hs := c.huntState
	for id, sm := range hs.summons {
		if sm.dead {
			continue
		}
		if now > sm.until || sm.hp <= 0 {
			sm.dead = true
			sm.swingDoneAt = 0
			delete(hs.summons, id)
			// Remove from every viewer's client (owner + teammates).
			s.removeSummonFromClientsLocked(c, sm, now)
			continue
		}
		// Resolve COMMITTED actions BEFORE any gate below can skip this summon: a wind-up
		// that has finished connects, and a bolt already drawn on every client lands
		// (mirrors the mob arm, which resolves its promises ahead of its own gates).
		s.resolveSummonSwingLocked(c, sm, now)
		s.resolveSummonBoltLocked(c, sm, now)
		// Close out a finished swing so the client drops the attack-action overlay
		// and the next DO_ACTION can re-trigger the WrapMode.Once attack clip.
		// Without this ACTION_DONE the summon swings exactly once then never
		// animates again (InstanceData.DoAction rejects the duplicate action id).
		// Broadcast so teammates' copies re-trigger too.
		if sm.swingDoneAt > 0 && now >= sm.swingDoneAt {
			sm.swingDoneAt = 0
			s.broadcastObjLocked(c, sm.id, battleproto.CmdActionDone, amf.NewArray().
				Set("id", sm.id).
				Set("action", summonAttackProtoID).
				Set("item", false).
				Set("cooldown", now))
		}
		// A STATIONARY turret (Morlokai's «Грозовой тотем») never moves: it holds its spawn
		// point and zaps a RANDOM enemy within range each second ("...случайным врагам
		// вокруг" -- not the nearest one, so a single unit parked next to it can't tank
		// every bolt while the rest of a pack takes none). Handled before the seek/escort
		// movement logic and always `continue`s so no move order is ever issued for it.
		if sm.stationary {
			var candidates []*mobState
			for _, m := range hs.mobs {
				if m.dead || !m.enemyOf(c.playerTeam()) {
					continue
				}
				if math.Hypot(float64(m.x-sm.x), float64(m.y-sm.y)) < stationaryTurretRange {
					candidates = append(candidates, m)
				}
			}
			var tgt *mobState
			if len(candidates) > 0 {
				tgt = candidates[s.randIntn(len(candidates))]
			}
			if tgt != nil && now >= sm.nextSwing {
				sm.nextSwing = now + 1.0/summonAttackSpeedMulLocked(sm, now)
				s.broadcastObjLocked(c, sm.id, battleproto.CmdAction,
					newActionArgs(sm.id, summonAttackProtoID, tgt.id, now,
						amf.NewArray().Set("x", 0.0).Set("y", 0.0)))
				sm.swingDoneAt = now + 0.9
				s.strikeSummonLocked(c, sm, tgt, now)
			}
			continue
		}
		// Find something to fight. Only ENEMIES: a summon fights for its owner, so it
		// is team 1 (see revealSummonToMemberLocked) and must skip the player's own
		// side. In Hunt every mob is team 0 -> teamVal() -1, so all of them stay
		// targets; in «Штурм» the mob set ALSO holds the player's allied creeps and
		// their base structures (team 1), which a pet would otherwise happily maul --
		// nearest-first, so it picked the friendly creeps marching right past it.
		var target *mobState
		best := summonSeek

		if sm.pet {
			// A PET fights on ORDERS. It does not pick its own fights and it does not
			// escort -- where it stands is the player's call.
			target = s.petTargetLocked(c, sm, now)
			if target == nil {
				if sm.ordMove {
					if math.Hypot(float64(sm.ordX-sm.x), float64(sm.ordY-sm.y)) > summonSlotTol {
						s.moveSummonLocked(c, sm, sm.ordX, sm.ordY, true, now)
					} else {
						sm.ordMove = false // arrived: hold here until told otherwise
						s.moveSummonLocked(c, sm, 0, 0, false, now)
					}
				} else {
					s.moveSummonLocked(c, sm, 0, 0, false, now) // no order: hold
				}
				continue
			}
			best = math.Hypot(float64(target.x-sm.x), float64(target.y-sm.y))
		} else {
			for _, m := range hs.mobs {
				// Judge the target against the OWNER's side (c is the summon's owner here),
				// the same authority the pet path uses -- a summon fights for whoever cast it.
				if m.dead || !m.enemyOf(c.playerTeam()) {
					continue
				}
				d := math.Hypot(float64(m.x-sm.x), float64(m.y-sm.y))
				if d < best {
					best, target = d, m
				}
			}
		}

		if target == nil {
			// No enemy nearby: escort the owner like a pet instead of freezing in
			// place. Each summon holds its own slot on a ring AROUND the owner
			// (surrounding them from all sides) rather than every pet piling onto
			// the owner's exact point.
			ax, ay := c.posAtLocked(float32(now))
			gx, gy := sm.ringPoint(ax, ay, summonRing)
			// Near a map edge some ring slots fall off the walkable area; clip each
			// slot back onto the map (from the on-map owner toward the slot) so a pet
			// never targets -- and slides onto -- ground outside the map.
			if c.nav != nil && !c.nav.Walkable(float64(gx), float64(gy)) {
				cgx, cgy := c.nav.Clip(float64(ax), float64(ay), float64(gx), float64(gy))
				gx, gy = float32(cgx), float32(cgy)
			}
			if math.Hypot(float64(gx-sm.x), float64(gy-sm.y)) > summonSlotTol {
				s.moveSummonLocked(c, sm, gx, gy, true, now)
			} else {
				s.moveSummonLocked(c, sm, 0, 0, false, now)
			}
			continue
		}
		reach := meleeReach + summonRadius + target.mob.Radius()
		if best > reach {
			// Approach from this summon's slot angle so several pets surround the
			// mob at melee distance instead of stacking on its centre. Aim just
			// inside the stop radius so best drops below `reach` and it commits to
			// the swing.
			gx, gy := sm.ringPoint(target.x, target.y, math.Max(0.4, reach-0.4))
			s.moveSummonLocked(c, sm, gx, gy, true, now)
			continue
		}
		// Strike FIRST, then settle. Committing the swing here sets swingDoneAt, so the
		// hold/sidestep below sees "mid-strike" on the very tick the swing starts and keeps
		// the pet planted -- otherwise it would shuffle off a crowding body on the same tick
		// it begins its strike, the exact «двигается во время замаха» regression the creep
		// path was already fixed for (and which the Hunt mob arm still has as a wart).
		if now >= sm.nextSwing {
			sm.nextSwing = now + 1.0/summonAttackSpeedMulLocked(sm, now)
			s.broadcastObjLocked(c, sm.id, battleproto.CmdAction,
				newActionArgs(sm.id, summonAttackProtoID, target.id, now,
					amf.NewArray().Set("x", 0.0).Set("y", 0.0)))
			// Close the swing a hair before the next one so a continuous attacker
			// re-triggers cleanly (mirrors the mob path).
			sm.swingDoneAt = now + 0.9
			s.strikeSummonLocked(c, sm, target, now)
		}
		s.moveSummonLocked(c, sm, 0, 0, false, now)
	}
}

// strikeSummonLocked resolves one summon swing. A MELEE summon connects at once, as it
// always has. A RANGED one (its client prefab carries a projectile) instead looses a bolt:
// SET_PROJECTILE tells every client to fly the prefab -- which is what draws Morlokai's
// lightning and plays its sound -- and the damage is committed to land on arrival, so the
// health drop and the visible impact happen at the same moment instead of the health
// dropping while nothing is drawn at all.
func (s *Server) strikeSummonLocked(c *conn, sm *summonState, target *mobState, now float64) {
	sm.swingHitAt = now + summonStrikeFrac/summonAttackSpeedMulLocked(sm, now)
	sm.swingTarget = target.id
}

// resolveSummonSwingLocked fires a wind-up that has run its course: a melee summon
// connects, a shooter looses its bolt (which then lands on arrival). The blow is
// COMMITTED at this point -- it connects even if the target has since moved -- but a
// target that died in the meantime is simply dropped.
func (s *Server) resolveSummonSwingLocked(c *conn, sm *summonState, now float64) {
	if sm.swingHitAt == 0 || now < sm.swingHitAt {
		return
	}
	sm.swingHitAt = 0
	target := c.huntState.mobs[sm.swingTarget]
	sm.swingTarget = 0
	if target == nil || target.dead {
		return
	}
	if !sm.ranged {
		s.hitMobLocked(c, target, sm.dmg, sm.id)
		s.applySummonOnHitOpsLocked(c, sm, target, now)
		return
	}
	flight := math.Hypot(float64(target.x-sm.x), float64(target.y-sm.y))/summonBoltSpeed + summonBoltMinFlight
	sm.projHitAt = now + flight
	sm.projTarget = target.id
	s.broadcastObjLocked(c, sm.id, battleproto.CmdSetProjectile, amf.NewArray().
		Set("source", sm.id).
		Set("target", target.id).
		Set("hit_at", sm.projHitAt))
}

// resolveSummonBoltLocked lands a bolt that has arrived. Like every other committed
// projectile in the engine the hit is a promise already made: it connects even if the
// target has since moved, and it is dropped only if the target is gone.
func (s *Server) resolveSummonBoltLocked(c *conn, sm *summonState, now float64) {
	if sm.projHitAt == 0 || now < sm.projHitAt {
		return
	}
	sm.projHitAt = 0
	tgt := c.huntState.mobs[sm.projTarget]
	sm.projTarget = 0
	if tgt == nil || tgt.dead {
		return
	}
	s.hitMobLocked(c, tgt, sm.dmg, sm.id)
	s.applySummonOnHitOpsLocked(c, sm, tgt, now)
}

// ringPoint returns this summon's formation slot: the point at its fixed angle on
// a ring of the given radius around (cx,cy). Anchoring the goal at a per-summon
// angle keeps a group spread out around the owner (escort) or the target (melee
// circle) instead of every unit converging on one point.
func (sm *summonState) ringPoint(cx, cy float32, radius float64) (float32, float32) {
	return cx + float32(radius*math.Cos(sm.fang)), cy + float32(radius*math.Sin(sm.fang))
}

// moveSummonLocked dead-reckons a summon and syncs velocity changes. Movement is
// clamped to walkable ground and routes around walls (same nav treatment as
// mobs) so a summon can't chase its target through geometry.
func (s *Server) moveSummonLocked(c *conn, sm *summonState, tx, ty float32, chase bool, now float64) {
	// Grimlok's «Дикость» also speeds up his live dinosaur's run, not just its swing.
	speed := float32(float64(summonSpeed) * summonMoveSpeedMulLocked(sm, now))
	dt := tickInterval.Seconds()
	px0, py0 := sm.x, sm.y
	sm.x += sm.vx * float32(dt)
	sm.y += sm.vy * float32(dt)
	if c.nav != nil && (sm.vx != 0 || sm.vy != 0) && !c.nav.Walkable(float64(sm.x), float64(sm.y)) {
		cx, cy := c.nav.Clip(float64(px0), float64(py0), float64(sm.x), float64(sm.y))
		sm.x, sm.y = float32(cx), float32(cy)
		// Same as the mob clamp: a clipped summon's cached route may point at an
		// occluded waypoint — exhaust it so aimAlong re-paths from the clipped spot.
		if len(sm.pf.pts) > 0 {
			sm.pf.idx = len(sm.pf.pts)
		}
	}
	// Separation from every other body. Without it a pet walked through mobs, creeps and
	// its packmates and came to rest inside whatever it was next to: ringPoint spreads
	// several pets of the SAME owner around a shared anchor, but a fixed formation angle
	// knows nothing about other bodies and was never collision.
	//
	// Never mid-swing, in either arm below: the client blends the attack clip over the run
	// clip, so an attacker that slides reads as broken. That is the same rule that keeps a
	// «Штурм» creep planted for its strike, and it was a reported bug once already.
	var sepx, sepy float32
	if sm.swingDoneAt == 0 {
		sepx, sepy = c.huntState.bodySeparation(c.instMembers(), now, sm.id, sm.x, sm.y, summonRadius)
	}
	sepN := float32(math.Hypot(float64(sepx), float64(sepy)))

	var nvx, nvy float32
	switch {
	case chase:
		blocked := c.nav != nil && !c.mobHasLoSLocked(sm.x, sm.y, tx, ty)
		gx, gy := c.aimAlong(&sm.pf, sm.x, sm.y, tx, ty, blocked, now)
		dx, dy := gx-sm.x, gy-sm.y
		d := float32(math.Hypot(float64(dx), float64(dy)))
		if d > 0.3 {
			ux, uy := dx/d, dy/d
			stx, sty := ux+sepx*mobSepWeight, uy+sepy*mobSepWeight
			sn := float32(math.Hypot(float64(stx), float64(sty)))
			if sn < 1e-3 {
				stx, sty, sn = ux, uy, 1
			}
			nvx, nvy = stx/sn*speed, sty/sn*speed
		}
	default:
		sm.pf.pts = nil
		// Holding is not the same as being welded in place. A pet ordered onto a spot
		// another body already occupies still has to get OUT of it -- otherwise it walks
		// the order straight into a creep and stops there, which is the whole complaint.
		// This is not drift: the push is zero the moment it is clear, so the pet settles
		// just outside the body it was standing in and holds there. Reduced speed so
		// getting unstuck reads as a shuffle, not a charge (mirrors the Hunt mob arm).
		if sepN > mobSepStep {
			nvx, nvy = sepx/sepN*speed*mobSidestepFrac, sepy/sepN*speed*mobSidestepFrac
		}
	}
	// A push is lateral, so it can aim at ground the straight heading never touched. Keep
	// the pet on the map: drop the push rather than step into geometry (the clamp above
	// would only drag it back out afterwards, which reads as sticking on the wall).
	if c.nav != nil && sepN > 0 && (nvx != 0 || nvy != 0) {
		step := float32(tickInterval.Seconds())
		if !c.nav.Walkable(float64(sm.x+nvx*step), float64(sm.y+nvy*step)) {
			nvx, nvy = 0, 0
			if chase {
				if dx, dy := tx-sm.x, ty-sm.y; math.Hypot(float64(dx), float64(dy)) > 0.3 {
					d := float32(math.Hypot(float64(dx), float64(dy)))
					nvx, nvy = dx/d*speed, dy/d*speed
				}
			}
		}
	}
	if nvx == sm.vx && nvy == sm.vy {
		return
	}
	// Sync a stop or a real course change at once; throttle the rest. Not syncing is
	// self-consistent, not a lie: the summon keeps integrating on the velocity the
	// client was last told, so the two stay aligned and the push simply applies once it
	// is worth a packet. Mirrors the mob chase guard.
	stopping := nvx == 0 && nvy == 0
	if !stopping && now-sm.posSyncAt <= 0.7 &&
		math.Hypot(float64(nvx-sm.vx), float64(nvy-sm.vy)) <= float64(speed)*0.3 {
		return
	}
	sm.vx, sm.vy = nvx, nvy
	sm.posSyncAt = now
	// Re-sync the new velocity to every viewer, each with its own tracking index.
	s.broadcastPosLocked(c, sm.id, sm.x, sm.y, sm.vx, sm.vy, float32(now))
}

// ---- mob statuses + AI ----

// ensureMobStatusFxLocked starts a looped status VFX on a mob exactly once, storing
// its world-fx id in *slot (idempotent: a nonzero slot means it is already running).
// An empty fx name is a no-op -- worldFxStartLocked returns 0, leaving *slot at 0.
func (s *Server) ensureMobStatusFxLocked(c *conn, m *mobState, slot *int32, fx string) {
	if *slot == 0 {
		*slot = s.worldFxStartLocked(c, fx, m.id, 0, false, 0, 0)
	}
}

// stunMobLocked applies a stun with its generic fx and stops the mob.
func (s *Server) stunMobLocked(c *conn, m *mobState, now, dur float64) {
	m.st.stunUntil = math.Max(m.st.stunUntil, now+dur)
	s.ensureMobStatusFxLocked(c, m, &m.st.stunFx, "StunEffect")
	s.stopMobLocked(c, m, now)
	s.cancelSwingLocked(c, m, now)
}

// cancelSwingLocked aborts m's committed-but-UNRELEASED swing or cast. The ACTION_DONE
// goes out FIRST -- resetInFlight would otherwise zero swingDoneAt without ever telling
// the client the swing ended, leaving the attack clip looping. The attack CADENCE
// (nextSwing) is deliberately preserved across the reset: losing the swing is the
// interrupt; handing back the cooldown too would make being stunned a reward.
//
// An ALREADY-LOOSED shell (projFlying) survives: its hit is a promise made to the client
// that cannot be withdrawn, so cancelling it would render a visible impact for zero
// damage. See mobState.projFlying. A melee swing has no such promise and is dropped.
//
// A stun is the only caller today, and wiring it here settles a rule the two drivers
// disagreed on. Hunt gated its AI on stunned() AFTER committing, so the release and the
// impact just waited and fired late -- an arrow leaving a bow that had been frozen for
// two seconds. «Штурм» had no stun gate upstream of dotaResolveSwingLocked at all, so a
// stunned cannon fired dead on schedule. Interrupting the wind-up is what the client
// renders either way: the stun clip replaces the attack clip.
func (s *Server) cancelSwingLocked(c *conn, m *mobState, now float64) {
	flying, hitAt, hitDmg, hitTarget := m.projFlying, m.hitAt, m.hitDmg, m.hitTarget
	s.closeMobSwingLocked(c, m, now)
	next := m.nextSwing
	m.resetInFlight()
	m.nextSwing = next
	if flying {
		m.projFlying, m.hitAt, m.hitDmg, m.hitTarget = true, hitAt, hitDmg, hitTarget
	}
}

// stopMobLocked zeroes a mob's velocity (with sync) if it was moving. The stop
// is broadcast to every member that renders the mob (each with its own index).
func (s *Server) stopMobLocked(c *conn, m *mobState, now float64) {
	if m.vx == 0 && m.vy == 0 {
		return
	}
	m.vx, m.vy = 0, 0
	s.broadcastPosLocked(c, m.id, m.x, m.y, 0, 0, float32(now))
}

// mobSpeed is a mob's effective move speed: its authored base times the status
// multiplier. THE ONE place that multiplier is computed, so a mob moves at the speed
// its SYNC advertises and every driver agrees.
//
// moveFactor already folds both the slow and every move_speed_pct mod (status.go), so
// it is applied exactly once. The three mob call sites used to re-multiply by
// modMul("move_speed_pct") on top of it, squaring the mod -- self-consistently, which
// is why it hid: movement matched the SYNC, both were just wrong. A 1.4x haste ran at
// 1.96x. The player path (effects.go, lobbyMoveSpeed * moveFactor) never carried the
// extra factor, which is what shows this was a slip and not a rule.
func mobSpeed(m *mobState, now float64) float64 {
	return m.mob.Speed * m.st.moveFactor(now)
}

// syncMobSpeedLocked pushes the mob's current SPEED stat (slow visuals) to every
// viewer.
func (s *Server) syncMobSpeedLocked(c *conn, m *mobState, now float64) {
	s.broadcastStatLocked(c, m.id, syncSpeed, float32(mobSpeed(m, now)), float32(now))
}

// mobTarget is the enemy a mob has picked this tick: the nearest member's avatar
// or one of the members' summons, with the owning connection for hit resolution.
type mobTarget struct {
	x, y   float32
	dist   float64
	radius float64
	obj    int32 // objID to swing at (0 = no valid target this tick)
	owner  *conn // member the target belongs to (for hit resolution)
	summon *summonState
}

// mobTargetLocked picks a mob's target across the whole party: the closest of
// every alive member's avatar and every member's summons. dist is +Inf (obj 0)
// when nothing is targetable, which drops the mob out of combat via the leash.
func mobTargetLocked(m *mobState, members []*conn, now float64) mobTarget {
	best := mobTarget{dist: math.Inf(1)}
	for _, mem := range members {
		hs := mem.huntState
		if hs == nil {
			continue
		}
		if hs.deadUntil == 0 && now >= hs.invisibleUntil {
			px, py := mem.posAtLocked(float32(now))
			if d := math.Hypot(float64(px-m.x), float64(py-m.y)); d < best.dist {
				best = mobTarget{x: px, y: py, dist: d, radius: hs.av.Radius(), obj: mem.objID, owner: mem}
			}
		}
		for _, sm := range hs.summons {
			if sm.dead {
				continue
			}
			if d := math.Hypot(float64(sm.x-m.x), float64(sm.y-m.y)); d < best.dist {
				best = mobTarget{x: sm.x, y: sm.y, dist: d, radius: summonRadius, obj: sm.id, owner: mem, summon: sm}
			}
		}
	}
	return best
}

// resolveMobHitLocked applies a mob's committed basic-attack hit to whatever it
// was aimed at (a member avatar or a summon), looked up by the stored objID.
func (s *Server) resolveMobHitLocked(m *mobState, members []*conn, now float64) {
	for _, mem := range members {
		hs := mem.huntState
		if hs == nil {
			continue
		}
		if mem.objID == m.hitTarget {
			hs.lastDamageWasSkill = false
			s.hitPlayerLocked(mem, m, m.hitDmg, now)
			return
		}
		if sm, ok := hs.summons[m.hitTarget]; ok && !sm.dead {
			s.hitSummonLocked(mem, m, sm, m.hitDmg, now)
			return
		}
	}
}

// mobTargetPosLocked returns the live position of the objID a ranged mob is aimed
// at (a member avatar or a summon) and whether it was found -- used to size the
// arrow's flight from the mob's current spot at the moment of release.
func mobTargetPosLocked(objID int32, members []*conn, now float64) (float32, float32, bool) {
	for _, mem := range members {
		hs := mem.huntState
		if hs == nil {
			continue
		}
		if mem.objID == objID {
			px, py := mem.posAtLocked(float32(now))
			return px, py, true
		}
		if sm, ok := hs.summons[objID]; ok && !sm.dead {
			return sm.x, sm.y, true
		}
	}
	return 0, 0, false
}

// clampArrowFlight sizes an arrow's flight from the caster→target gap: fast, but
// bounded so a point-blank shot still streaks visibly and a long shot never dawdles.
func clampArrowFlight(dist float64) float64 {
	f := dist / mobArrowSpeed
	if f < minArrowFlight {
		f = minArrowFlight
	}
	if f > maxArrowFlight {
		f = maxArrowFlight
	}
	return f
}

// mobArrowFlightLocked returns how long a just-released arrow should fly to its
// target, from the live gap at release, additionally capped so the hit lands before
// the mob's next swing (which would otherwise overwrite the pending hit and drop the
// damage). The release was scheduled to reserve this room, so the cap rarely bites.
func mobArrowFlightLocked(m *mobState, members []*conn, now float64) float64 {
	dist := m.mob.AttackRange // sensible fallback if the target vanished this tick
	if tx, ty, ok := mobTargetPosLocked(m.projTarget, members, now); ok {
		dist = math.Hypot(float64(tx-m.x), float64(ty-m.y))
	}
	f := clampArrowFlight(dist)
	if hi := m.nextSwing - now - 0.02; hi > 0.04 && f > hi {
		f = hi
	}
	return f
}

// closeMobSwingLocked ends a mob's in-flight basic-attack / cast animation on every
// viewer -- the ACTION_DONE that drops the attack-animation overlay (AnimationExt
// gates the swing clip on DoingAction, so without it the client loops the attack pose
// forever). Idempotent: a no-op when the mob isn't mid-swing. Used by the normal
// per-tick close-out AND by any path that abandons a live swing (leash/give-up on a
// player kill) -- there resetInFlight would otherwise zero swingDoneAt WITHOUT ever
// telling the client the swing ended, leaving the attack clip looping as the mob
// walks home.
func (s *Server) closeMobSwingLocked(c *conn, m *mobState, now float64) {
	if m.swingDoneAt == 0 {
		return
	}
	m.swingDoneAt = 0
	s.broadcastObjLocked(c, m.id, battleproto.CmdActionDone, amf.NewArray().
		Set("id", m.id).
		Set("action", m.attackProtoID()).
		Set("item", false).
		Set("cooldown", now))
}

// mobUpkeepLocked runs the timed upkeep every live mob needs whichever world
// simulation drives it: expiring status fx/mods, bleeding DoT streams and closing out a
// finished swing. It reports whether the mob is still alive -- a DoT slice can finish it
// off, and the caller must then skip the rest of its pass for it.
//
// Shared by the Hunt mob pass (tickMobsLocked) and the «Штурм» tick, which replaces that
// pass wholesale: without this, statuses on a lane creep never expired (a stun's VFX
// hung on it forever), DoTs never bled it, and its swing was never closed out.
const (
	// Ranged mobs and bosses carry a mana pool (mobState.maxMana); melee trash has none.
	// mobRangedManaCost is spent per ranged basic-attack shot, bossSkillManaCost per boss
	// skill cast. Regen (gamedata.MobManaRegenFrac() of the pool per second, an admin knob)
	// by default comfortably outpaces a mob's own steady spend, so a mob never self-starves
	// -- only a player's active mana-drain / mana-burn skill can. Lowering the knob (or 0)
	// makes a drained mob stay drained. defaultRangedMana/defaultBossMana back-fill pools.
	mobRangedManaCost = 6.0
	bossSkillManaCost = 25.0
	defaultRangedMana = 120.0
	defaultBossMana   = 320.0
)

func (s *Server) mobUpkeepLocked(c *conn, m *mobState, now float64) bool {
	st := &m.st

	// Mana regen for the ranged/boss pool, then push the mana bar to viewers if it moved
	// enough since the last sync -- so a bar drained by a shot/boss-cast/mana-burn visibly
	// drops and climbs back live (throttled; a no-op for melee). Runs every mob tick, but
	// only broadcasts on a meaningful change (syncMobManaLocked's threshold).
	if m.maxMana > 0 && m.mana < m.maxMana {
		m.mana = math.Min(m.maxMana, m.mana+m.maxMana*gamedata.MobManaRegenFrac()*tickInterval.Seconds())
	}
	s.syncMobManaLocked(c, m, now)

	// Status fx expiry (world-scoped: ended on every viewer).
	if st.stunFx != 0 && now >= st.stunUntil {
		s.worldFxEndLocked(c, st.stunFx)
		st.stunFx = 0
	}
	if st.rootFx != 0 && now >= st.rootUntil {
		s.worldFxEndLocked(c, st.rootFx)
		st.rootFx = 0
	}
	if st.slowFx != 0 && now >= st.slowUntil {
		s.worldFxEndLocked(c, st.slowFx)
		st.slowFx = 0
		s.syncMobSpeedLocked(c, m, now)
	}
	if st.atkSlowFx != 0 && now >= st.atkSlowUntil {
		s.worldFxEndLocked(c, st.atkSlowFx)
		st.atkSlowFx = 0
	}
	if st.silenceFx != 0 && now >= st.silenceUntil {
		s.worldFxEndLocked(c, st.silenceFx)
		st.silenceFx = 0
	}
	if st.chillFx != 0 && now >= st.chillUntil {
		s.worldFxEndLocked(c, st.chillFx)
		st.chillFx = 0
	}
	var expMods []statMod
	for _, mod := range st.mods {
		if mod.until == 0 || mod.until > now {
			expMods = append(expMods, mod)
		}
	}
	if len(expMods) != len(st.mods) {
		st.mods = expMods
		s.syncMobSpeedLocked(c, m, now)
	}

	// DoT streams. Snapshot the slice first: a tick that kills the mob resets
	// m.st (clearing st.dots) mid-loop, so indexing the live slice would panic.
	// Damage is applied in small dotTickInterval slices and the health bar is
	// synced ONCE per combat tick (5x/sec) so poison bleeds down smoothly instead
	// of lurching once a second. No RECEIVE_HIT per slice -- that would spray an
	// impact particle every tick; the acid VFX already signals the poison. Only
	// the KILLING slice goes through hitMobLocked (for the death sequence).
	dots := st.dots
	var dotKeep []overTime
	drained := false
	for i := range dots {
		if m.dead {
			break
		}
		d := dots[i]
		for d.nextTick <= now && d.nextTick <= d.until && !m.dead {
			slice := d.currentPerSec(d.nextTick) * dotTickInterval
			if m.hp-slice <= 0 {
				s.hitMobLocked(c, m, slice, d.srcObj) // finishing tick: full death path
			} else {
				m.hp -= slice
				drained = true
			}
			d.nextTick += dotTickInterval
		}
		if !m.dead && d.until > now {
			dotKeep = append(dotKeep, d)
		}
	}
	if m.dead {
		return false
	}
	if drained {
		s.syncMobHealthLocked(c, m)
	}
	st.dots = dotKeep
	// Poison fully cleared -> drop the shared acid visual.
	if len(dotKeep) == 0 && st.dotFx != 0 {
		s.worldFxEndLocked(c, st.dotFx)
		st.dotFx = 0
	}

	// Close out a finished swing: tell every viewer the attack action is done so
	// it drops the attack-animation overlay (and attacker highlight). Runs in
	// every branch, so a mob that leaves attack range mid-swing still stops.
	if m.swingDoneAt > 0 && now >= m.swingDoneAt {
		s.closeMobSwingLocked(c, m, now)
	}
	return true
}

// tickMobsLocked is the shared mob simulation, run once per world by the instance
// ticker (c is any live member -- all members share the mob set, nav and roster).
// Mobs target the nearest party member and every visual is broadcast to all
// viewers. With one member it is the old single-player pass, unchanged.
// mobSpawnLeashRadius is how far a homed mob may stray from its spawn before it turns
// back: a boss is pinned to its arena (bossSpawnLeash), everything else roams wider
// (mobSpawnLeash).
func mobSpawnLeashRadius(m *mobState) float64 {
	if m.boss {
		return bossSpawnLeash
	}
	return mobSpawnLeash
}

func (s *Server) tickMobsLocked(c *conn, now float64) {
	hs := c.huntState
	dt := tickInterval.Seconds()
	members := c.members()

	for _, m := range hs.mobs {
		if m.dead {
			// Revive at the authored spawn point once the respawn timer elapses; the
			// fog-of-war pass re-creates it on the client when the player is near.
			if m.respawnAt > 0 && now >= m.respawnAt {
				s.respawnMobLocked(c, m, now)
			}
			continue
		}
		// Interest management (create/remove on the party's clients, skip the AI of a
		// mob nobody is near). Provably a no-op rather than a rule -- see
		// mobInterestLocked.
		if !s.mobInterestLocked(c, m, now) {
			continue
		}
		st := &m.st

		if !s.mobUpkeepLocked(c, m, now) {
			continue // a DoT tick finished it off
		}

		// Knockback glide owns the mob until it lands: the shove already moved its
		// authoritative position and broadcast a client-side glide there, so hold every
		// AI/stun gate off it until the window elapses (then the stop is broadcast).
		if m.kbUntil > 0 && s.advanceKnockbackLocked(c, m, now) {
			continue
		}

		// Dead-reckon our copy of the mob position, clamped to walkable ground so
		// a chasing mob can never end up standing inside a wall/void.
		px0, py0 := m.x, m.y
		m.x += m.vx * float32(dt)
		m.y += m.vy * float32(dt)
		if c.nav != nil && (m.vx != 0 || m.vy != 0) && !c.nav.Walkable(float64(m.x), float64(m.y)) {
			cx, cy := c.nav.Clip(float64(px0), float64(py0), float64(m.x), float64(m.y))
			m.x, m.y = float32(cx), float32(cy)
			m.vx, m.vy = 0, 0
			// Separation steering can push a route-follower off its verified line
			// into a wall, leaving the cached waypoint occluded from the clipped
			// spot. Exhaust the route so aimAlong re-paths from here this same
			// tick instead of grinding at the corner for up to the 1s staleness
			// window. The len guard keeps the nil-route 1/s retry throttle intact.
			if len(m.pf.pts) > 0 {
				m.pf.idx = len(m.pf.pts)
			}
		}

		// Resolve COMMITTED actions before any gate below can skip this mob. They are
		// promises already made -- a loosed shell is flying on the client, and a swing
		// connects because range was met when it STARTED. A stun cancels the wind-up at
		// the moment it lands (cancelSwingLocked), so nothing reachable here is stale;
		// what must not happen is a committed hit STALLING behind the stun gate and
		// firing late. «Штурм» resolves in the same position, ahead of its own gates
		// (dotaResolveSwingLocked), so the two drivers now agree.
		//
		// Release a ranged mob's committed arrow once the draw animation ends (its
		// scheduled bow-release point). The client flies the projectile prefab over
		// hit_at-now, so compute the flight NOW from the live gap and set the hit to
		// land on arrival. Capped to resolve before the next swing so the cycle stays
		// clean.
		if m.projLaunchAt > 0 && now >= m.projLaunchAt {
			m.projLaunchAt = 0
			m.hitAt = now + mobArrowFlightLocked(m, members, now)
			m.projFlying = true
			s.broadcastObjLocked(c, m.id, battleproto.CmdSetProjectile, amf.NewArray().
				Set("source", m.id).
				Set("target", m.projTarget).
				Set("hit_at", m.hitAt))
		}
		// Land committed hits even if the target has since moved: a swing/cast connects
		// because range (or aim) was met when it STARTED, matching the client, which
		// locks onto the target at the start of the action. Basic attack hit -- resolved
		// to whichever member/summon it was aimed at:
		if m.hitAt > 0 && now >= m.hitAt {
			m.hitAt = 0
			m.projFlying = false
			s.resolveMobHitLocked(m, members, now)
		}
		// Boss skill impact:
		if m.skillHitAt > 0 && now >= m.skillHitAt {
			m.skillHitAt = 0
			s.landBossSkillLocked(c, m, members, now)
		}

		if st.stunned(now) {
			continue
		}

		// Leashed and walking home: regenerate and steer back to spawn, IGNORING the
		// world entirely (WoW-style evade). A returning mob may not be re-aggroed --
		// not by a player stepping back into range, nor by being hit (hitMobFlagsLocked
		// skips the aggro flag while returning) -- until it actually reaches spawn.
		// returnHomeStepLocked clears m.returning at mobHomeEpsilon, and the normal
		// aggro gate below picks it up FRESH on the next tick only if a target is still
		// in range. This stops a leashed mob (or kited boss) from being dragged right
		// back out the instant it turns for home.
		if m.returning {
			s.returnHomeStepLocked(c, m, now)
			continue
		}

		// Target: the nearest of every party member's avatar and their summons.
		tgt := mobTargetLocked(m, members, now)
		tx, ty := tgt.x, tgt.y
		dist := tgt.dist
		targetRadius := tgt.radius

		if !m.aggro {
			if dist <= mobAggroRange {
				m.aggro = true
			} else {
				continue
			}
		}
		// Strayed too far from its OWN spawn (boss pinned to 2m, trash to mobSpawnLeash):
		// give up and walk home even if a player is still right next to it, so a boss can't
		// be dragged out of its arena. Only homed mobs have a spawn to leash to.
		strayed := m.homed && math.Hypot(float64(m.x-m.spawnX), float64(m.y-m.spawnY)) > mobSpawnLeashRadius(m)
		if dist > mobLeashRange || tgt.obj == 0 || strayed {
			// Gave up: nobody (avatar or summon) is in reach. Walk back to spawn and
			// regenerate to full instead of freezing wherever the chase ended -- the
			// early returning branch above drives it home on subsequent ticks.
			// End any in-flight swing FIRST: a player killed mid-swing drops the mob
			// straight here while swingDoneAt is still in the future, and resetInFlight
			// would zero it without the ACTION_DONE -- leaving the attack clip looping
			// all the way home. closeMobSwingLocked sends it before we clear the state.
			m.aggro = false
			m.returning = true
			s.closeMobSwingLocked(c, m, now)
			m.resetInFlight()
			s.stopMobLocked(c, m, now)
			continue
		}

		// Stand still while an attack/cast animation is playing: a creature never
		// chases or shuffles mid-swing. swingDoneAt is cleared when the animation's
		// ACTION_DONE fires; skillHitAt while a cast wind-up is still pending.
		if m.swingDoneAt > 0 || m.skillHitAt > 0 {
			s.stopMobLocked(c, m, now)
			continue
		}
		// Off cooldown and in range? Cast a skill instead of a basic attack.
		if s.tryBossSkillLocked(c, m, members, now) {
			continue
		}

		// A mob may only attack a target it can actually reach: if a wall sits
		// between them (the mob chased to a chokepoint the player is behind),
		// treat it as out of range so it never hits "through" geometry -- the
		// cause of phantom damage from an unseen attacker. blocked also forces the
		// chase branch, where movement is nav-clipped so the mob stops at the wall.
		blocked := !c.mobHasLoSLocked(m.x, m.y, tx, ty)

		// Reach = base gap + both body radii (client's AvatarAI math): a melee mob
		// uses meleeReach, a ranged mob its own AttackRange. Big bosses reach from
		// farther because their radius is large.
		base := meleeReach
		if m.mob.AttackRange > 0 {
			base = m.mob.AttackRange
		}
		atkRange := base + m.mob.Radius() + targetRadius

		if dist > atkRange || blocked {
			// Stationary mobs (rooted plants/turrets) never chase or reposition:
			// they hold their spawn and wait for the target to re-enter range.
			if m.mob.Stationary || st.rooted(now) {
				s.stopMobLocked(c, m, now)
				continue
			}
			// Chase: with a clear line, head straight at the target; when a wall is
			// in the way (blocked), follow an A* route around it. Either way the
			// per-tick walkable clamp above keeps the mob out of geometry.
			gx, gy := c.aimAlong(&m.pf, m.x, m.y, tx, ty, blocked, now)
			dx, dy := gx-m.x, gy-m.y
			d := float32(math.Hypot(float64(dx), float64(dy)))
			if d < 0.2 {
				s.stopMobLocked(c, m, now) // wall-blocked: hold position
				continue
			}
			sp := mobSpeed(m, now)
			// Steer = chase direction + separation from packmates, so a group
			// converging on the target fans out instead of overlapping.
			ux, uy := dx/d, dy/d
			sepx, sepy := hs.bodySeparation(c.instMembers(), now, m.id, m.x, m.y, m.mob.Radius())
			stx, sty := ux+sepx*mobSepWeight, uy+sepy*mobSepWeight
			sn := float32(math.Hypot(float64(stx), float64(sty)))
			if sn < 1e-3 {
				stx, sty, sn = ux, uy, 1
			}
			nvx, nvy := stx/sn*float32(sp), sty/sn*float32(sp)
			if now-m.lastSync > 0.7 ||
				math.Hypot(float64(nvx-m.vx), float64(nvy-m.vy)) > float64(sp)*0.3 {
				m.vx, m.vy = nvx, nvy
				m.lastSync = now
				s.broadcastMobPosLocked(c, m, now)
			}
			continue
		}

		// In range with a clear path. If crowding a packmate, sidestep so the pack
		// spreads around the target rather than stacking on one point; otherwise
		// hold position. Either way it still swings (attacking while shuffling).
		sepx, sepy := hs.bodySeparation(c.instMembers(), now, m.id, m.x, m.y, m.mob.Radius())
		sn := float32(math.Hypot(float64(sepx), float64(sepy)))
		if sn > mobSepStep && !st.rooted(now) && !m.mob.Stationary {
			sp := mobSpeed(m, now)
			m.vx, m.vy = sepx/sn*float32(sp)*mobSidestepFrac, sepy/sn*float32(sp)*mobSidestepFrac
			m.lastSync = now
			s.broadcastMobPosLocked(c, m, now)
		} else {
			s.stopMobLocked(c, m, now)
		}
		if now < st.silenceUntil {
			continue
		}
		atkSpeed := m.mob.AttackSpeed * st.attackFactor(now)
		if now >= m.nextSwing && atkSpeed > 0 {
			// A ranged mob pays mana per shot; a starved caster/archer can't loose one and
			// waits for regen (melee mobs have maxMana 0 and skip this). This is the tactical
			// bite of the player's mana-drain / mana-burn skills (Neirofim, BlackDragon).
			if m.mob.AttackRange > 0 && m.maxMana > 0 && m.mana < mobRangedManaCost {
				m.nextSwing = now + 0.5
				continue
			}
			if m.mob.AttackRange > 0 && m.maxMana > 0 {
				m.mana -= mobRangedManaCost
			}
			m.nextSwing = now + 1/atkSpeed
			target := tgt.obj
			s.broadcastObjLocked(c, m.id, battleproto.CmdAction,
				newActionArgs(m.id, mobAttackProtoID(m.mobIdx), target, now,
					amf.NewArray().Set("x", 0.0).Set("y", 0.0)))
			// Schedule the matching ACTION_DONE so the client stops "doing the attack
			// action" once the swing plays. Without it the client keeps the attack
			// animation overlaid forever (AnimationExt gates it on DoingAction), so a
			// mob that chases out of range still visibly swings. Fire it a hair before
			// the next swing so a continuous attacker re-triggers cleanly.
			animEnd := now + math.Min(0.9/atkSpeed, 1.2)
			m.swingDoneAt = animEnd
			dmg := m.rollDamage(s)
			m.hitDmg = dmg
			m.hitTarget = target
			if m.mob.AttackRange > 0 {
				// Ranged mobs (skeleton archers, the shooter plant, caster bosses -- any
				// with an explicit AttackRange) fire a visible arrow/bolt so the player sees
				// it cross the gap instead of taking damage from afar. The arrow must leave
				// the bow at the END of the draw animation (like Elgorm's bolt), NOT partway
				// through it, so the release -- and the SET_PROJECTILE that spawns the
				// prefab -- is scheduled for animEnd (the ACTION_DONE moment). Only if that
				// end leaves too little room for a visible flight before the next swing is
				// the release pulled a touch earlier (never before the mid-swing point). The
				// hit lands on ARRIVAL, computed at release from the live gap: unlike a melee
				// swing (fixed mid-swing connect) a ranged hit can't precede its own
				// projectile, so hitAt is left 0 here and set when the arrow flies.
				release := animEnd
				if latest := m.nextSwing - clampArrowFlight(dist) - 0.02; release > latest {
					release = latest
				}
				if earliest := now + 0.5/atkSpeed; release < earliest {
					release = earliest
				}
				m.projLaunchAt = release
				m.projTarget = target
				m.hitAt = 0
			} else {
				// Melee: the weapon connects mid-swing, in place, no projectile.
				m.hitAt = now + 0.5/atkSpeed
			}
		}
	}
}

// broadcastMobPosLocked pushes a mob's current position+velocity to every viewer,
// each addressed by its own tracking index.
func (s *Server) broadcastMobPosLocked(c *conn, m *mobState, now float64) {
	s.broadcastPosLocked(c, m.id, m.x, m.y, m.vx, m.vy, float32(now))
}

// bodySeparation returns a steering push that keeps the body (selfID, x, y, radius r)
// clear of every other living body in the battle, so units converging on one target --
// or filing down one lane -- spread out instead of piling onto a single point. The
// magnitude grows as the two overlap more (0 at the pair's range, ~1 when touching).
// Bodies sharing the exact same point are parted deterministically by id so they never
// stay welded together.
//
// Range is per PAIR: max(mobSepRange, r+or+sepMargin). The floor keeps Hunt's packs
// spaced exactly as they were tuned; the radius term is what «Штурм» needs, where the
// bodies run from a 0.55m archer to a 3.0m altar and one flat number cannot serve both
// -- at a flat 1.8m a creep stands well inside the altar it is hitting.
//
// Widening the range never costs a unit its target: dotaReach adds BOTH radii on top of
// the 2.2m melee reach, so a creep held 3.95m off the altar's centre still swings at it
// from 5.8m. The two rules are measured the same way and cannot cross.
//
// members carries every player's summons (nil is fine). It is a separate argument
// because summons live in their OWNER's map, never in inst.mobs -- which is exactly why
// Grimlok's dinosaur had no collision at all: nothing pushed it and it pushed nothing.
func (hs *huntState) bodySeparation(members map[int32]*conn, now float64, selfID int32, x, y float32, r float64) (float32, float32) {
	var sx, sy float32
	push := func(oid int32, ox, oy float32, or float64) {
		if oid == selfID {
			return
		}
		rng := float32(math.Max(mobSepRange, r+or+sepMargin))
		dx, dy := x-ox, y-oy
		// The circular range is contained in this axis-aligned square. Most
		// creeps are on different parts of a lane, so reject them before the
		// comparatively expensive square root without changing the exact force
		// for any body that can contribute.
		if math.Abs(float64(dx)) >= float64(rng) || math.Abs(float64(dy)) >= float64(rng) {
			return
		}
		d := float32(math.Hypot(float64(dx), float64(dy)))
		if d >= rng {
			return
		}
		if d < 1e-3 {
			a := float64(selfID) * 2.3999632 // golden angle: distinct dir per id
			dx, dy, d = float32(math.Cos(a)), float32(math.Sin(a)), 1
		}
		w := (rng - d) / rng
		sx += dx / d * w
		sy += dy / d * w
	}
	var orderedMobs []*mobState
	if hs.inst != nil && hs.inst.dota != nil && hs.inst.dota.tickMobsOK && hs.inst.dota.tickMobsAt == now {
		// The spatial snapshot is built from the same id-sorted list.  Restore
		// that order for the small local candidate set before accumulating forces:
		// floating-point addition then remains bit-for-bit identical to the old
		// full scan, while distant bodies are never even considered.
		index := &hs.inst.dota.spatial
		queryRadius := float32(math.Max(mobSepRange, r+float64(index.maxRadius)+sepMargin))
		candidates := hs.separationCandidates[:0]
		index.forEachNearby(x, y, queryRadius, func(o *mobState) {
			candidates = append(candidates, o)
		})
		slices.SortFunc(candidates, func(a, b *mobState) int { return cmp.Compare(a.id, b.id) })
		hs.separationCandidates = candidates
		orderedMobs = candidates
	} else {
		mobIDs := hs.separationMobIDs[:0]
		for id := range hs.mobs {
			mobIDs = append(mobIDs, id)
		}
		slices.Sort(mobIDs)
		hs.separationMobIDs = mobIDs
		orderedMobs = hs.separationMobs[:0]
		for _, id := range mobIDs {
			orderedMobs = append(orderedMobs, hs.mobs[id])
		}
		hs.separationMobs = orderedMobs
	}
	for _, o := range orderedMobs {
		// Only an ACTIVE mob can be near enough to matter with a current position: an
		// inactive one is >34m away (past mobHideRadius) and its coordinates are stale
		// anyway, so skipping it is free and keeps this O(active^2) rather than
		// O(active x total) -- a Hunt map carries 400-600 spawns.
		//
		// Gate on active, not on shown: they agree in Hunt, but a «Штурм» creep is
		// always active, and testing the render flag here would silently return (0,0)
		// for a whole mode.
		if o.dead || !o.active {
			continue
		}
		push(o.id, o.x, o.y, o.mob.Radius())
	}
	memberIDs := hs.separationMemberIDs
	if hs.separationMembersAt != now {
		memberIDs = memberIDs[:0]
		for id := range members {
			memberIDs = append(memberIDs, id)
		}
		slices.Sort(memberIDs)
		hs.separationMemberIDs = memberIDs
		hs.separationMembersAt = now
	}
	for _, id := range memberIDs {
		mem := members[id]
		mhs := mem.huntState
		if mhs == nil {
			continue
		}
		// Liveness matches tickSummonsLocked's own test, not just the lazily-set flag:
		// an expired summon still sits in the map until its tick removes it.
		summonIDs := hs.separationSummonIDs[:0]
		for summonID := range mhs.summons {
			summonIDs = append(summonIDs, summonID)
		}
		slices.Sort(summonIDs)
		hs.separationSummonIDs = summonIDs
		for _, summonID := range summonIDs {
			sm := mhs.summons[summonID]
			if !sm.alive(now) {
				continue
			}
			push(sm.id, sm.x, sm.y, summonRadius)
		}
	}
	// Clamp to unit length. Every neighbour contributes up to 1, so a unit boxed in by
	// five others yielded a push ~5 long, which mobSepWeight then scaled to ~4.5 against a
	// chase direction of exactly 1 -- separation outvoting the goal by more than 4:1. The
	// units did not settle beside each other, they oscillated: flung apart, dragged back
	// by the march, flung apart again, at full speed, re-syncing every tick of it. Bounded,
	// the push is what it is meant to be: a correction to a course, never the course.
	if n := float32(math.Hypot(float64(sx), float64(sy))); n > 1 {
		sx, sy = sx/n, sy/n
	}
	return sx, sy
}

// instMembers is every conn sharing this one's battle (nil when unattached, as in a
// directly-constructed test state). Handed around as the live map rather than a copied
// slice: bodySeparation runs per unit per tick and must not allocate.
func (c *conn) instMembers() map[int32]*conn {
	if c.inst == nil {
		return nil
	}
	return c.inst.members
}

// nearestMemberDistLocked is the distance from a mob to the closest party member
// (avatars only; +Inf if none). Drives the unified fog: a mob is shown to the
// whole party while ANY member is near, hidden once EVERY member is far.
// mobViewDistLocked is the interest-management distance for m: the nearest party member,
// but ALSO counting any live vision ward (Urg's «Росток») as an eye. A mob inside a
// ward's radius counts as fully in view (distance 0), so it is revealed AND unshaded even
// with no player nearby -- «Росток» «открывает участок местности», i.e. shows the mobs
// standing on it (on a Hunt map the only "fog" is that distant mobs aren't created on the
// client at all, so this is what makes them appear). A mob just past the ward edge counts
// the ward as a distant viewer, so it fades in through the shade ring rather than popping.
func (s *Server) mobViewDistLocked(c *conn, m *mobState, now float64) float64 {
	// Op.RevealTarget (Velial's «Трибунал»): the marked target stays fully revealed to the
	// whole team regardless of distance, bypassing the fog entirely.
	if now < m.st.revealUntil {
		return 0
	}
	members := c.members()
	d := nearestMemberDistLocked(members, m, now)
	if d <= mobUnshadeRadius {
		return d // a player is already close enough; the ward scan can't lower it further
	}
	for _, mem := range members {
		hs := mem.huntState
		if hs == nil {
			continue
		}
		for _, w := range hs.wards {
			if now > w.until {
				continue
			}
			wd := math.Hypot(float64(w.x-m.x), float64(w.y-m.y))
			if wd <= w.radius {
				return 0 // inside the acorn's radius -> fully revealed
			}
			if wd < d {
				d = wd
			}
		}
	}
	return d
}

func nearestMemberDistLocked(members []*conn, m *mobState, now float64) float64 {
	best := math.Inf(1)
	for _, mem := range members {
		hs := mem.huntState
		if hs == nil {
			continue
		}
		px, py := mem.posAtLocked(float32(now))
		d := math.Hypot(float64(px-m.x), float64(py-m.y))
		// Op.Stat=="view_radius_pct" (Veritas «Метаморфоза»): a bigger view radius reveals
		// mobs from farther away, i.e. shrinks the EFFECTIVE distance used for the
		// reveal/hide comparison below -- Hunt has no fog plane, so this is the only real
		// vision-radius hook (mobRevealRadius/mobHideRadius are otherwise flat constants).
		if mul := hs.st.modMul(now, "view_radius_pct"); mul > 1 {
			d /= mul
		}
		if d < best {
			best = d
		}
	}
	return best
}

// mobInterestLocked runs Hunt's INTEREST MANAGEMENT for m -- create it on the party's
// clients when any member draws within mobRevealRadius, remove it once every member is
// past mobHideRadius -- and reports whether its AI should run this tick.
//
// This is not fog of war, despite what this pass used to be called. The Hunt scenes
// bake no WarFogPlane_prop01 at all (verified: map_4_0/4_1/4_2 contain zero war-fog
// objects; only map_0_0 and map_1_0 have one), so there is no fog on a Hunt map to
// gate anything. What this actually is: an object budget, plus mobShadeFx as a
// hand-rolled stand-in for the fog the map cannot render.
//
// Skipping the AI here is an OPTIMISATION, and provably not a rule: mobHideRadius (34)
// > mobLeashRange (22) > mobAggroRange (9), so a mob nobody is within 34 of has no
// legal action to take this tick. Nothing that IS a rule may be attached to it -- see
// the shown/active comment on mobState. «Штурм» keeps every unit active and rendered
// instead, which is why creeps go on breaking towers in fog.
func (s *Server) mobInterestLocked(c *conn, m *mobState, now float64) bool {
	// Admin fog-off for «Охота»: reveal and simulate every mob from the start, with
	// no reveal-on-approach gate and no shade overlay -- the whole map is visible.
	// (This pass runs only for Hunt/Arena worlds; «Штурм» has its own dotaTick.) A
	// late joiner still gets every shown mob via the join-time sync (see avatars.go).
	if !gamedata.HuntFog() {
		if !m.shown {
			s.revealMobLocked(c, m, now)
		}
		if m.shaded {
			s.worldFxEndLocked(c, m.shadeFxUID)
			m.shadeFxUID = 0
			m.shaded = false
		}
		m.active = m.shown
		return m.active
	}
	d := s.mobViewDistLocked(c, m, now)
	if m.shown {
		if d >= mobHideRadius {
			s.hideMobLocked(c, m, now)
			// Abandoned: hand it back pristine. Only a HOMED mob has anywhere to go --
			// this is the leash rule finishing early, not a consequence of being
			// unrendered, and it is why it is gated on the policy and not on the flag.
			// The party is already past the leash range, so the mob is walking home and
			// regenerating; resetToSpawn just lands it in the state that walk reaches.
			if m.homed {
				m.resetToSpawn()
			}
			m.active = false
			return false
		}
		s.updateMobShadeLocked(c, m, d)
	} else if d <= mobRevealRadius {
		s.revealMobLocked(c, m, now)
		s.updateMobShadeLocked(c, m, d)
	}
	m.active = m.shown
	return m.active
}

// updateMobShadeLocked toggles the translucent fog-ring overlay on a shown mob
// (world-scoped, so every member sees it identically): mobShadeFx (50%-alpha
// DeathShader) once the NEAREST member drifts to the outer reveal ring, ended as
// the party closes in. Hysteresis prevents a per-tick flicker at the boundary.
func (s *Server) updateMobShadeLocked(c *conn, m *mobState, d float64) {
	if m.shaded {
		if d <= mobUnshadeRadius {
			s.worldFxEndLocked(c, m.shadeFxUID)
			m.shadeFxUID = 0
			m.shaded = false
		}
	} else if d >= mobShadeRadius {
		m.shadeFxUID = s.worldFxStartLocked(c, mobShadeFx, m.id, 0, false, 0, 0)
		m.shaded = true
	}
}

// revealMobToMemberLocked creates the mob on ONE member's client (tracker add +
// CREATE_OBJECT + stats SYNC + attack effector). No-op if already tracked -- used
// both by the fog pass (fan out to all) and when a newcomer joins an ongoing world.
func (s *Server) revealMobToMemberLocked(mem *conn, m *mobState, now float64) {
	hs := mem.huntState
	if hs == nil || hs.tr.index(m.id) >= 0 {
		return
	}
	bt := float32(now)
	idx := hs.tr.add(m.id)
	maxHP := m.maxHealth()
	hpFrac := float32(1.0)
	if maxHP > 0 {
		hpFrac = float32(m.hp / maxHP)
	}
	// Ranged mobs and bosses carry a mana pool (m.maxMana); melee trash has 0. Send it so the
	// target-card mana bar reads the real value -- without a syncMana/syncMaxMana the client
	// defaults it to 0 (why «Скелет лучник» showed 0 mana). Same wire shape as the avatar:
	// syncMana is a FRACTION (mana/maxMana), syncMaxMana the absolute pool. Melee (maxMana 0)
	// sends 0/0, which is exactly its "no mana" state -- no regression.
	manaFrac := float32(0)
	if m.maxMana > 0 {
		manaFrac = float32(m.mana / m.maxMana)
	}
	dmgLo, dmgHi := m.dmgRange()
	s.push(mem, battleproto.CmdCreateObject, amf.NewArray().
		Set("id", m.id).Set("proto", mobProtoID(m.mobIdx)))
	// The enemy target-card (client ObjectInfo) shows the mob's level, read from the
	// object's InstanceData.Level -- which only SET_AVATAR / LEVEL_UP populate, and it
	// renders as Level+1. Mobs aren't player-bound, so CREATE_OBJECT/SYNC can't carry a
	// level; push a LEVEL_UP (OnLevelUp sets Data.Level for ANY object id) so a levelled
	// mob shows its real level instead of the default "1". mobState.level is 1-based; send
	// level-1 (0-based wire, matching the avatar path) so the card's +1 lands on the
	// intended number. Skip level<=1 (unlevelled trash + bosses) -- the card already reads
	// the intended "1" from the default Data.Level of 0.
	if m.level > 1 {
		s.push(mem, battleproto.CmdLevelUp, amf.NewArray().
			Set("id", m.id).Set("level", int32(m.level-1)))
	}
	s.push(mem, battleproto.CmdSync, amf.NewArray().Set("data",
		newSyncBlob(bt).addObject(m.id).
			position(idx, m.x, m.y, m.vx, m.vy, bt).
			setFloats(syncHealth, idx, hpFrac).
			setFloats(syncMaxHealth, idx, float32(maxHP)).
			setFloats(syncMana, idx, manaFrac).
			setFloats(syncMaxMana, idx, float32(m.maxMana)).
			setFloats(syncDmgMin, idx, float32(dmgLo)).
			setFloats(syncDmgMax, idx, float32(dmgHi)).
			setFloats(syncAttackSpeed, idx, float32(m.mob.AttackSpeed)).
			setFloats(syncSpeed, idx, float32(m.mob.Speed)).
			setFloats(syncRadius, idx, float32(m.mob.Radius())).
			// Vision. One value is correct for both modes because the CLIENT gates the
			// reveal zone on friendliness itself: WarFogObject.Update spawns one only
			// for a FRIEND/NEUTRAL object, so this is inert on a Hunt mob and on an
			// enemy creep, and lights the lane for the player's own creeps. Omitting it
			// is what left a «Штурм» lane black -- mViewRadius > 0f is the other half of
			// that gate, and the escorting creeps failed it.
			setFloats(syncViewRadius, idx, effectiveViewRadius(creepViewRadius)).
			setInt(syncTeam, idx, m.teamVal()).
			build(hs.tr.count())))
	s.addAttackEffectorLocked(mem, m.id, mobAttackProtoID(m.mobIdx), now)
}

// revealMobLocked marks the mob shown and creates it on every party member's
// client.
func (s *Server) revealMobLocked(c *conn, m *mobState, now float64) {
	m.shown = true
	for _, mem := range c.members() {
		s.revealMobToMemberLocked(mem, m, now)
	}
	m.lastSync = now
}

// respawnMobLocked revives a dead mob at its authored spawn point with full HP,
// all state cleared. It stays hidden (m.shown=false) so the fog-of-war pass
// re-creates it on the client the next tick if the player is in range.
// resetInFlight zeros a mob's in-flight basic-attack and skill state (the fields a
// committed-but-unresolved swing/cast reads). Shared by the respawn and
// return-home paths so the two can't drift as new in-flight fields are added.
func (m *mobState) resetInFlight() {
	m.hitAt, m.hitDmg, m.hitTarget = 0, 0, 0
	m.projLaunchAt, m.projTarget, m.projFlying = 0, 0, false
	m.swingDoneAt, m.nextSwing = 0, 0
	m.skillHitAt, m.skillDmg, m.skillRadius = 0, 0, 0
}

// resetToSpawn restores a LIVING mob to a pristine state at its spawn point: full
// HP, no aggro/return, no statuses, no in-flight action, cooldowns cleared. Called
// when the party abandons it past the fog-hide radius so a re-revealed mob is
// always fresh at home. The mob is hidden when this runs, so no client sync is
// needed -- the fog pass re-creates it at these values next time a member nears.
func (m *mobState) resetToSpawn() {
	m.x, m.y = m.spawnX, m.spawnY
	m.vx, m.vy = 0, 0
	m.hp = m.maxHealth()
	m.mana = m.maxMana // ranged/boss mana tops off with HP at home
	m.manaSyncFrac = 1 // full again; keeps the next live sync from re-announcing the refill
	m.aggro = false
	m.returning = false
	m.st = unitStatus{}
	m.pf = pathState{}
	m.resetInFlight()
	m.lastSync = 0
	for i := range m.skillReady {
		m.skillReady[i] = 0
	}
}

func (s *Server) respawnMobLocked(c *conn, m *mobState, now float64) {
	m.dead = false
	m.respawnAt = 0
	m.hp = m.maxHealth()
	m.mana = m.maxMana // full mana on respawn
	m.manaSyncFrac = 1
	m.x, m.y = m.spawnX, m.spawnY
	m.vx, m.vy = 0, 0
	m.aggro = false
	m.returning = false
	m.shown = false
	// Not simulated until the interest pass says so. active must never outlive shown
	// here: this runs from the top of the mob loop, which then skips the rest of the
	// tick for this mob, so a stale active=true would let mobSeparation steer the OTHER
	// mobs away from an invisible body that just teleported to its spawn.
	m.active = false
	m.shaded = false
	m.shadeFxUID = 0
	m.st = unitStatus{}
	m.pf = pathState{}
	m.resetInFlight()
	m.lastSync = 0
	for i := range m.skillReady {
		m.skillReady[i] = 0
	}
}

// returnMobHomeLocked sends a LIVING mob back to its spawn and drops any in-flight
// action, hiding it so the fog pass re-reveals it at home next tick. Used to evict
// mobs that were dragged onto a respawn checkpoint mid-fight (their HP/dead state is
// preserved -- this is a reposition, not a respawn).
func (s *Server) returnMobHomeLocked(c *conn, m *mobState, now float64) {
	m.x, m.y = m.spawnX, m.spawnY
	m.vx, m.vy = 0, 0
	m.aggro = false
	m.returning = false
	m.pf = pathState{}
	m.resetInFlight()
	if m.shown {
		s.hideMobLocked(c, m, now)
	}
}

// returnHomeStepLocked advances a leashed (returning) mob one tick toward its
// spawn: it regenerates HP (broadcast so the bar visibly refills for anyone
// watching it retreat) and steers home, routing around walls like the chase path.
// On arrival it tops off to full HP and clears the returning state. A returning
// mob ignores all targets -- the caller only enters here while no member is inside
// aggro range.
func (s *Server) returnHomeStepLocked(c *conn, m *mobState, now float64) {
	maxHP := m.maxHealth()
	// Regenerate toward full while walking back.
	if m.hp < maxHP {
		m.hp = math.Min(maxHP, m.hp+maxHP*gamedata.MobHPRegenFrac()*tickInterval.Seconds())
		s.syncMobHealthLocked(c, m)
	}
	sp := m.mob.Speed // reset ignores slows -- it heads home at full pace
	// Arrive when within a single step of home; a fixed per-tick step would
	// otherwise overshoot the sub-step epsilon and oscillate across the spawn point.
	arrive := math.Max(mobHomeEpsilon, sp*tickInterval.Seconds())
	dx, dy := m.spawnX-m.x, m.spawnY-m.y
	if math.Hypot(float64(dx), float64(dy)) <= arrive {
		// Home: finish the reset (guarantee full HP even if regen hadn't caught up).
		m.x, m.y = m.spawnX, m.spawnY
		m.returning = false
		m.pf = pathState{}
		if m.hp < maxHP {
			m.hp = maxHP
			s.syncMobHealthLocked(c, m)
		}
		s.stopMobLocked(c, m, now)
		return
	}
	// Steer home: straight line when clear, A* route around walls when blocked.
	blocked := c.nav != nil && !c.mobHasLoSLocked(m.x, m.y, m.spawnX, m.spawnY)
	gx, gy := c.aimAlong(&m.pf, m.x, m.y, m.spawnX, m.spawnY, blocked, now)
	sdx, sdy := gx-m.x, gy-m.y
	d := float32(math.Hypot(float64(sdx), float64(sdy)))
	if d < 0.2 {
		s.stopMobLocked(c, m, now) // wall-blocked corner: hold, re-path next tick
		return
	}
	nvx, nvy := sdx/d*float32(sp), sdy/d*float32(sp)
	if now-m.lastSync > 0.7 ||
		math.Hypot(float64(nvx-m.vx), float64(nvy-m.vy)) > float64(sp)*0.3 {
		m.vx, m.vy = nvx, nvy
		m.lastSync = now
		s.broadcastMobPosLocked(c, m, now)
	}
}

// hideMobFromMemberLocked removes the mob from ONE member's client (tracker
// swap-remove + SYNC-remove + DELETE_OBJECT). No-op if the member wasn't tracking it.
func (s *Server) hideMobFromMemberLocked(mem *conn, m *mobState, now float64) {
	s.untrackObjForMemberLocked(mem, m.id, float32(now))
}

// removeMobFromClientsLocked drops the mob from every member's client (fog-hide or
// corpse cleanup). The DELETE_OBJECT tears down the fog-ring fx child with the
// model, so the shade state is just cleared (no EFFECT_END needed).
func (s *Server) removeMobFromClientsLocked(c *conn, m *mobState, now float64) {
	m.shown = false
	m.active = false // off the clients and out of the simulation -- see mobState.active
	m.shaded = false
	m.shadeFxUID = 0
	for _, mem := range c.members() {
		s.hideMobFromMemberLocked(mem, m, now)
	}
}

// hideMobLocked removes the mob from every client and quiesces it (a hidden mob is
// far past leash range, so it drops aggro and any in-flight action).
func (s *Server) hideMobLocked(c *conn, m *mobState, now float64) {
	m.aggro = false
	m.vx, m.vy = 0, 0
	m.hitAt = 0
	m.projLaunchAt = 0
	m.skillHitAt = 0
	m.swingDoneAt = 0
	s.removeMobFromClientsLocked(c, m, now)
}

// nearestMemberLocked returns the closest alive party member's avatar to the mob
// (nil if the whole party is down), with its distance and position. Bosses aim
// their abilities at this member.
func nearestMemberLocked(members []*conn, m *mobState, now float64) (best *conn, dist float64, bx, by float32) {
	dist = math.Inf(1)
	for _, mem := range members {
		hs := mem.huntState
		if hs == nil || hs.deadUntil > 0 {
			continue
		}
		px, py := mem.posAtLocked(float32(now))
		if d := math.Hypot(float64(px-m.x), float64(py-m.y)); d < dist {
			best, dist, bx, by = mem, d, px, py
		}
	}
	return
}

// tryBossSkillLocked casts the first ready boss ability whose range covers the
// nearest party member, playing its attack animation, spawning its VFX, and
// scheduling the hit after the wind-up. Returns true if a skill was started this
// tick (the caller then skips the basic-attack/chase branch).
func (s *Server) tryBossSkillLocked(c *conn, m *mobState, members []*conn, now float64) bool {
	if len(m.mob.Skills) == 0 || m.skillHitAt > 0 {
		return false
	}
	if now < m.st.silenceUntil {
		return false // a silenced boss (Neirofim's «Молчание») can't cast
	}
	tgt, dist, px, py := nearestMemberLocked(members, m, now)
	if tgt == nil {
		return false // no target: the whole party is down
	}
	if !c.mobHasLoSLocked(m.x, m.y, px, py) {
		return false // don't cast through walls
	}
	for i := range m.mob.Skills {
		sk := m.mob.Skills[i]
		if now < m.skillReady[i] || dist > sk.Range {
			continue
		}
		// A boss pays mana to cast; a mana-starved boss (drained by Neirofim/Inshari) can't
		// and falls back to basic attacks. Bosses regen fast enough to self-sustain, so this
		// only bites under active player mana pressure.
		if m.maxMana > 0 && m.mana < bossSkillManaCost {
			continue
		}
		if m.maxMana > 0 {
			m.mana -= bossSkillManaCost
		}
		m.skillReady[i] = now + sk.Cooldown
		// Op.CastMark (Einzenhaim's «Изгнание колдовства»): this boss was marked by an
		// earlier spray -- casting now triggers the extra punish damage.
		if now < m.st.castMarkUntil {
			if dmg := m.st.castMarkDmg; dmg > 0 {
				s.hitMobLocked(c, m, dmg, m.st.castMarkOwner)
			}
			m.st.castMarkUntil = 0
		}
		// Play an attack animation (reuse the boss's attack action) turned to the
		// target, broadcast to every viewer, and close it after the wind-up.
		s.broadcastObjLocked(c, m.id, battleproto.CmdAction,
			newActionArgs(m.id, mobAttackProtoID(m.mobIdx), tgt.objID, now,
				amf.NewArray().Set("x", float64(px)).Set("y", float64(py))))
		m.swingDoneAt = now + math.Min(sk.CastTime+0.4, 1.5)
		// Spawn the ability VFX (prefab ships in the boss bundle), world-scoped so the
		// whole party sees the telegraph: at the target's feet for lobbed/ranged
		// skills, on the boss for point-blank ones.
		if sk.OnTarget {
			s.worldFxStartLocked(c, sk.Fx, m.id, tgt.objID, true, px, py)
		} else {
			s.worldFxStartLocked(c, sk.Fx, m.id, 0, false, 0, 0)
		}
		// Schedule the impact; root the boss during the cast.
		m.skillHitAt = now + sk.CastTime
		m.skillDmg = sk.Dmg
		m.skillRadius = sk.Radius
		m.skillCX, m.skillCY = px, py
		m.skillTargetObj = tgt.objID
		s.stopMobLocked(c, m, now)
		if debugCombat {
			log.Printf("battle: %s boss %d casts %q (dmg %.0f radius %.1f)",
				c.RemoteAddr(), m.id, sk.Name, sk.Dmg, sk.Radius)
		}
		return true
	}
	return false
}

// landBossSkillLocked applies a boss skill's damage when its wind-up completes. An
// AoE (Radius>0) hits every member still within Radius of the cast point (so
// moving out dodges it); a single-target skill hits the member it was aimed at.
func (s *Server) landBossSkillLocked(c *conn, m *mobState, members []*conn, now float64) {
	if m.skillRadius > 0 {
		for _, mem := range members {
			hs := mem.huntState
			if hs == nil || hs.deadUntil > 0 {
				continue
			}
			px, py := mem.posAtLocked(float32(now))
			if math.Hypot(float64(px-m.skillCX), float64(py-m.skillCY)) <= m.skillRadius {
				hs.lastDamageWasSkill = true
				s.hitPlayerLocked(mem, m, m.skillDmg, now)
			}
		}
		return
	}
	for _, mem := range members {
		if mem.objID == m.skillTargetObj {
			if hs := mem.huntState; hs != nil && hs.deadUntil == 0 {
				hs.lastDamageWasSkill = true
				s.hitPlayerLocked(mem, m, m.skillDmg, now)
			}
			return
		}
	}
}

// mobHasLoSLocked reports whether a straight walkable segment connects the mob
// to (tx,ty): nav.Clip returns the farthest reachable point, so if it stops well
// short of the target a wall is in the way.
func (c *conn) mobHasLoSLocked(mx, my, tx, ty float32) bool {
	if c.nav == nil {
		return true
	}
	cx, cy := c.nav.Clip(float64(mx), float64(my), float64(tx), float64(ty))
	return math.Hypot(cx-float64(tx), cy-float64(ty)) < 1.0
}

// ---- damage to the player / summons, death, respawn ----

func (s *Server) hitPlayerLocked(c *conn, m *mobState, dmg float64, now float64) {
	if debugCombat && c.huntState != nil && c.huntState.deadUntil == 0 {
		alive := 0
		for _, mm := range c.huntState.mobs {
			if !mm.dead {
				alive++
			}
		}
		px, py := c.posAtLocked(float32(now))
		log.Printf("battle: %s HIT-BY mob=%d mobpos=(%.1f,%.1f) selfpos=(%.1f,%.1f) dmg=%.0f aliveMobs=%d",
			c.RemoteAddr(), m.id, m.x, m.y, px, py, dmg, alive)
	}
	s.hitPlayerFromLocked(c, m.id, dmg, now, m, nil)
}

// blockIncomingHitLocked consumes a learned OpBlockHit passive against an incoming BASIC
// attack, negating it completely and starting the passive's internal cooldown. Reports
// whether the blow was swallowed.
//
// It reads hs.lastDamageWasSkill WITHOUT clearing it: the flag belongs to
// runDefenseProcsLocked, which consumes it on the same hit -- and which never runs when
// the hit is blocked, since the caller returns.
func (s *Server) blockIncomingHitLocked(c *conn, damagerID int32, now float64, attacker *mobState) bool {
	hs := c.huntState
	slot := hs.blockHitSlot
	if slot < 1 || now < hs.blockHitReadyAt {
		return false
	}
	level := int(hs.skillLevel[slot-1])
	if level < 1 {
		return false // passive not learned yet
	}
	if hs.lastDamageWasSkill {
		return false // «от базовой атаки» -- a mob/boss SKILL is not blocked
	}
	var op *gamedata.Op
	for i := range hs.kit.Skills[slot-1].Ops {
		if hs.kit.Skills[slot-1].Ops[i].Kind == gamedata.OpBlockHit {
			op = &hs.kit.Skills[slot-1].Ops[i]
			break
		}
	}
	if op == nil {
		return false
	}
	// Tell the client the hit was absorbed: flags 1 / damage 0 is the same wire shape the
	// dodge path uses, so the "no damage" popup and the animation match.
	s.broadcastAvatarObjLocked(c, battleproto.CmdReceiveHit, amf.NewArray().
		Set("object", c.objID).
		Set("damager", damagerID).
		Set("flags", int32(1)).
		Set("damage", 0.0))
	// Punish the attacker («атакующий теряет скорость атаки на 50% в течение 1 секунды»).
	if len(op.Ops) > 0 && attacker != nil {
		px, py := c.posAtLocked(float32(now))
		s.applyOpsLocked(c, op.Ops, opCtx{slot: slot, level: level, target: attacker,
			px: px, py: py, hasPos: true}, now)
	}
	if cd := op.Dur.At(level); cd > 0 {
		hs.blockHitReadyAt = now + cd
		s.pushPassiveProcCooldownLocked(c, slot, hs.blockHitReadyAt)
	}
	return true
}

// hitPlayerFromLocked applies incoming damage to player c and, on a lethal blow, kills
// it. The attacker is identified two ways: damagerID is the wire id shown on the
// RECEIVE_HIT / ON_KILL (a mob id or an enemy avatar's objID), while thornsMob XOR
// pvpAttacker names the live source so thorns can bounce back to it and, in «Арена», the
// killer can be credited a frag. Exactly one of the two is non-nil (both nil = an
// environmental hit that neither retaliates nor scores).
func (s *Server) hitPlayerFromLocked(c *conn, damagerID int32, dmg float64, now float64, thornsMob *mobState, pvpAttacker *conn) {
	hs := c.huntState
	if hs.deadUntil > 0 {
		return
	}
	// Block the whole blow? (Edilia's «Пыльца забвения».) This sits ABOVE armor and
	// absorb because the skill negates the hit rather than reducing it, and above the
	// dodge roll because it is not a roll at all. Basic attacks only, so a boss skill
	// still connects.
	if s.blockIncomingHitLocked(c, damagerID, now, thornsMob) {
		return
	}
	// Dodge?
	if s.randFloat64() < hs.st.modSum(now, "dodge_pct") {
		s.broadcastAvatarObjLocked(c, battleproto.CmdReceiveHit, amf.NewArray().
			Set("object", c.objID).
			Set("damager", damagerID).
			Set("flags", int32(1)).
			Set("damage", 0.0))
		return
	}
	armorMul := hs.st.modMul(now, "armor_pct")
	// Op.ZoneArmor (Inshari's «Угнетение»): her armor bonus only helps against an attacker
	// standing OUTSIDE the aura -- one fighting her INSIDE it (already taking the periodic
	// damage) hits at her normal, unbuffed armor.
	if hs.zoneArmorSlot > 0 {
		if level := int(hs.skillLevel[hs.zoneArmorSlot-1]); level >= 1 {
			for _, op := range hs.kit.Skills[hs.zoneArmorSlot-1].Ops {
				if op.Kind != gamedata.OpZoneArmor {
					continue
				}
				px, py := c.posAtLocked(float32(now))
				outside := true
				switch {
				case thornsMob != nil:
					outside = math.Hypot(float64(thornsMob.x-px), float64(thornsMob.y-py)) > op.Radius
				case pvpAttacker != nil:
					ax, ay := pvpAttacker.posAtLocked(float32(now))
					outside = math.Hypot(float64(ax-px), float64(ay-py)) > op.Radius
				}
				if outside {
					armorMul *= op.Value.At(level)
				}
				break
			}
		}
	}
	armor := (hs.av.PhysArmor + hs.st.modSum(now, "phys_armor")) * armorMul
	dmg *= armorMitigation(armor)
	// Rognar's «Костяной щит»: a flat, guaranteed cut to incoming damage («снижающий
	// получаемый физический урон на 50%») -- distinct from armor_pct, which multiplies the
	// ARMOR STAT and gets diminishing-returns'd by the shared 50/(a+50) curve (a ×1.5 armor
	// buff never actually nets a 50% reduction). This engine's incoming-damage path doesn't
	// tag hits as phys/magic (only the player's OWN outgoing attacks carry a Scale), so the
	// reduction applies to all incoming damage while the shield is up, not physical-only --
	// the closest honest equivalent given no dmg-type plumbing exists on this side.
	if reduce := hs.st.modSum(now, "dmg_reduction_pct"); reduce > 0 {
		dmg *= 1 - math.Min(reduce, 1)
	} else if reduce < 0 {
		// A NEGATIVE dmg_reduction_pct AMPLIFIES incoming damage (BlackDragon's «Неистовство»:
		// "дракон получает на 20% больше урона от вражеских атак" -- a flat, rank-constant
		// self-penalty, not the diminishing-returns armor curve).
		dmg *= 1 - reduce
	}
	dmg = hs.st.absorb(now, dmg)
	if thorns := hs.st.modSum(now, "thorns_pct"); thorns > 0 && dmg > 0 {
		// Reflect a share of the blow back at whatever dealt it: a mob directly, an enemy
		// player through the same PvP path (with the roles reversed so its own thorns
		// don't re-reflect endlessly -- the bounce passes no attacker). Sigilion's «Удар
		// наотмашь» is melee-only («При получении урона от ближней атаки») -- gated the same
		// way BlackDragon's «Кровь дракона» OnDamaged proc gates MeleeOnly.
		switch {
		case thornsMob != nil && thornsMob.mob.AttackRange <= 0:
			s.hitMobLocked(c, thornsMob, dmg*thorns, c.objID)
		case pvpAttacker != nil && pvpAttacker.huntState != nil && !pvpAttacker.huntState.hasProjectile:
			s.hitPlayerFromLocked(pvpAttacker, c.objID, dmg*thorns, now, nil, nil)
		}
	}
	if dmg <= 0 {
		return
	}
	// A skill/DoT path can identify a hero by damagerID without passing the
	// pvpAttacker pointer. Resolve that strictly from the instance; a creep id must
	// never become a hero merely because creditConnLocked has a PvE fallback.
	heroAttacker := pvpAttacker
	if heroAttacker == nil {
		heroAttacker = s.enemyHeroByObjIDLocked(c, damagerID)
	}
	s.noteHeroDamageLocked(c, heroAttacker, now)
	// Rognar's «Канал смерти»: forward a share of the blow to the linked unit -- as magic
	// damage to a linked enemy, or as healing to a linked friend.
	if hs.deathLinkObj != 0 && now < hs.deathLinkUntil && hs.deathLinkFrac > 0 {
		share := dmg * hs.deathLinkFrac
		if hs.deathLinkAlly {
			if a := c.friendlyMember(hs.deathLinkObj); a != nil {
				s.healPlayerLocked(a, share)
			}
		} else if lm := hs.mobs[hs.deathLinkObj]; lm != nil && !lm.dead {
			s.hitMobLocked(c, lm, share, c.objID)
		}
	}
	// Op.DamageShare (Kiona's «Лесной покров»): a share of THIS damage heals every living
	// ally near the marked player instead of just being dealt.
	if hs.st.cloakHealCoeff > 0 && now < hs.st.cloakUntil {
		if owner := s.creditConnLocked(c, hs.st.cloakOwner); owner != nil {
			share := dmg * hs.st.cloakHealCoeff
			px, py := c.posAtLocked(float32(now))
			for _, mem := range owner.members() {
				mh := mem.huntState
				if mh == nil || mh.deadUntil > 0 {
					continue
				}
				mx, my := mem.posAtLocked(float32(now))
				if math.Hypot(float64(mx-px), float64(my-py)) <= hs.st.cloakRadius {
					s.healPlayerLocked(mem, share)
				}
			}
		}
	}
	hs.hp -= dmg
	creditedKiller := heroAttacker
	if hs.hp <= 0 && creditedKiller == nil && c.inst != nil && c.inst.dota != nil {
		creditedKiller = s.recentHeroKillCreditLocked(c, now)
	}
	s.broadcastAvatarObjLocked(c, battleproto.CmdReceiveHit, amf.NewArray().
		Set("object", c.objID).
		Set("damager", damagerID).
		Set("flags", int32(0)).
		Set("damage", dmg))
	if c.inst != nil && c.inst.dota != nil && c.inst.dota.telemetry != nil {
		s.telemetryRecordDamageWithCreditLocked(c.inst.dota.telemetry, c, damagerID, heroAttacker, creditedKiller, dmg, hs.hp <= 0, c.inst.dota.telemetryMatchTimeLocked(now), now)
	}
	if hs.hp <= 0 {
		hs.hp = 0
		s.syncSelfLocked(c, syncHealth)
		killerID := damagerID
		if creditedKiller != nil {
			killerID = creditedKiller.objID
		}
		s.playerDieLocked(c, killerID, now)
		if creditedKiller != nil {
			if c.inst != nil && c.inst.dota != nil {
				s.dotaCreditHeroKillLocked(creditedKiller, c, now)
			} else {
				s.arenaCreditKillLocked(creditedKiller, c, now)
			}
		}
		return
	}
	s.syncSelfLocked(c, syncHealth)
	// «Каменная кожа»-style defensive procs harden the avatar when it is struck (and
	// survives) -- rolled here, on the incoming-damage path, not on the attack path.
	// thornsMob is nil in PvP: a self-buff proc still fires, a retaliate proc no-ops.
	s.runDefenseProcsLocked(c, thornsMob, dmg, now)
	// Rognar's «Костяной щит» counts down its hits and detonates on the third.
	if hs.shieldExplodeSlot != 0 {
		hs.shieldHitsLeft--
		if hs.shieldHitsLeft <= 0 {
			s.explodeBoneShieldLocked(c, now)
		}
	}
}

func (s *Server) hitSummonLocked(c *conn, m *mobState, sm *summonState, dmg float64, now float64) {
	// Already down: drop the hit. The summon twin of hitMobFlagsLocked's dead guard, and
	// needed for the same reason -- the reap runs in tickSummonsLocked on the OWNER's
	// tick, so a pet killed by one attacker stays in hs.summons with dead==false for the
	// rest of the pass. A second committed swing landing in that window (ordinary lane
	// focus-fire) would drive its HP further negative and broadcast a SECOND ON_KILL for
	// the same body. Guarding here rather than at the call sites covers both drivers.
	if sm.dead || sm.hp <= 0 {
		return
	}
	sm.hp -= dmg
	// Fan the hit flash + HP-bar drop out to every viewer (owner + teammates), each
	// with its own tracking index for the health SYNC.
	s.broadcastObjLocked(c, sm.id, battleproto.CmdReceiveHit, amf.NewArray().
		Set("object", sm.id).
		Set("damager", m.id).
		Set("flags", int32(0)).
		Set("damage", dmg))
	frac := float32(math.Max(sm.hp, 0) / sm.maxHP)
	s.broadcastStatLocked(c, sm.id, syncHealth, frac, float32(now))
	if sm.hp <= 0 {
		s.broadcastObjLocked(c, sm.id, battleproto.CmdOnKill, amf.NewArray().
			Set("killer", m.id).Set("id", sm.id))
		// removal happens on the next summon tick
	}
}

// tryReviveLocked resurrects the player in place instead of dying, when a learned
// OpRevive passive (Zamaran's «Возрождение») is off its internal cooldown. It
// restores the op's HP (scaled by powerMul, capped at max HP), starts the internal
// cooldown, plays the skill's resurrection fx, and returns true so the death
// sequence is skipped. Called at the top of playerDieLocked so it covers every
// death path (not just the direct melee hit).
func (s *Server) tryReviveLocked(c *conn, now float64) bool {
	hs := c.huntState
	slot := hs.reviveSlot
	if slot == 0 {
		return false
	}
	level := int(hs.skillLevel[slot-1])
	if level < 1 { // unlearned ult slot
		return false
	}
	if now < hs.reviveReadyAt {
		return false
	}
	def := hs.skillDef(slot)
	var rev *gamedata.Op
	for i := range def.Ops {
		if def.Ops[i].Kind == gamedata.OpRevive {
			rev = &def.Ops[i]
			break
		}
	}
	if rev == nil {
		return false
	}
	heal := rev.Value.At(level) * hs.powerMul()
	if maxHP := hs.maxHPLocked(now); heal > maxHP {
		heal = maxHP
	}
	if heal <= 0 {
		return false
	}
	hs.hp = heal
	hs.reviveReadyAt = now + rev.Dur.At(level)
	// Resurrection cleanses lingering DoTs so the revived hero doesn't instantly bleed
	// back out (mirrors the death cleanse below, without the death/respawn).
	hs.st.dots = nil
	s.syncSelfLocked(c, syncHealth)
	if uid := s.fxStartLocked(c, def.CastFx, c.objID, 0, false, 0, 0); uid != 0 {
		hs.scheduleFxEnd(uid, now+1.5)
	}
	return true
}

// ccImmuneBlockLocked reports whether an incoming crowd-control effect on the player
// is blocked by a learned OpImmune passive (Wilfang's «Защитный покров» — immunity
// to root/stun/silence), consuming it and starting its recovery cooldown. Returns
// false when there is no immunity or it is on cooldown, so the caller then applies
// the CC. No mob applies player-facing CC in the PvE hunt today, so this gate is
// latent until such a source is added; it is exercised directly by unit tests.
func (s *Server) ccImmuneBlockLocked(c *conn, now float64) bool {
	hs := c.huntState
	if now < hs.tempCCImmuneUntil {
		// Arianna's «Щит хранителя»/«Касание спасителя»: a cast-granted immunity window, not
		// the passive's consume-then-cooldown -- blocks every attempt until the shield ends.
		return true
	}
	slot := hs.ccImmuneSlot
	if slot == 0 {
		return false
	}
	level := int(hs.skillLevel[slot-1])
	if level < 1 {
		return false
	}
	if now < hs.ccImmuneReadyAt {
		return false
	}
	def := hs.skillDef(slot)
	for _, op := range def.Ops {
		if op.Kind == gamedata.OpImmune {
			hs.ccImmuneReadyAt = now + op.Dur.At(level)
			return true
		}
	}
	return false
}

// cleansePlayerDebuffsLocked reports whether the player has a learned OpCleanseOnHit
// passive (Abominator's «Окоченение») and, if so, sheds every hostile effect
// currently on them (stun/root/silence/slow/DoTs and any negative stat mod) in
// response to one just landing. Like ccImmuneBlockLocked, no mob or avatar applies
// player-facing CC/debuffs today, so this gate is latent until such a source exists;
// it is exercised directly by unit tests. Gated by cleanseReadyAt (PVP balance pass,
// mirroring ccImmuneBlockLocked's own readyAt gate) so that once a live source does
// exist, this can't shed the whole debuff stack on every single hit -- an unlimited,
// cooldown-free full cleanse would hard-counter every CC-based kit in the roster.
func (s *Server) cleansePlayerDebuffsLocked(c *conn, now float64) bool {
	hs := c.huntState
	slot := hs.cleanseSlot
	if slot == 0 || int(hs.skillLevel[slot-1]) < 1 {
		return false
	}
	if now < hs.cleanseReadyAt {
		return false
	}
	level := int(hs.skillLevel[slot-1])
	def := hs.skillDef(slot)
	for _, op := range def.Ops {
		if op.Kind == gamedata.OpCleanseOnHit {
			hs.cleanseReadyAt = now + op.Dur.At(level)
			break
		}
	}
	hs.st.stunUntil = 0
	hs.st.rootUntil = 0
	hs.st.silenceUntil = 0
	hs.st.slowUntil = 0
	hs.st.dots = nil
	var keep []statMod
	for _, m := range hs.st.mods {
		if m.value < 0 && m.until != 0 {
			continue // sheds only timed HOSTILE mods, not permanent passive self-buffs
		}
		keep = append(keep, m)
	}
	hs.st.mods = keep
	s.pushPlayerStatsLocked(c, now)
	return true
}

// abominatorFeastRange is how close a transformed Трупоглот must be to a dying enemy
// avatar to claim the permanent health from its death.
const abominatorFeastRange = 15.0

// playerDieLocked runs the death sequence: everything cancels, the corpse
// shows for corpseHide seconds, RESPAWN announces the timer, and the tick
// revives at deadUntil.
func (s *Server) playerDieLocked(c *conn, killer int32, now float64) {
	hs := c.huntState
	// Auto-revive passive (Zamaran): resurrect in place instead of dying.
	if s.tryReviveLocked(c, now) {
		return
	}
	// huntState.level is zero-based while Dota's table is indexed by the
	// displayed level (1..20 in this game).
	hs.deadUntil = now + dotaHeroRespawnDelay(hs)
	hs.diedAt = now
	hs.corpseHidden = false
	hs.deaths++
	hs.lastKiller = 0
	// A new life must not inherit a pre-death hero tag and later hand a creep kill
	// to an avatar from the previous life.
	hs.lastHeroDamager = 0
	hs.lastHeroDamageAt = 0
	if c.inst != nil {
		if killerConn := c.inst.members[killer]; killerConn != nil && killerConn != c {
			hs.lastKiller = killerConn.selfPlayerID
		}
	}
	s.broadcastPlayerStatsLocked(c)
	if c.inst != nil && c.inst.dota != nil {
		s.publishLiveFightLogLocked(c.inst)
	}

	// Gellar's «Порабощение»: «При смерти теряет половину из накопленных душ». Death also
	// cleanses Hekata's kill-window (its +30% buff is dropped with the other timed mods
	// below, so the on-kill bonus should end with it).
	if hs.soulSlot != 0 {
		hs.soulStacks /= 2
	}
	hs.killWindowUntil = 0
	hs.killWindowStacks = 0

	s.cancelOrderLocked(c)
	hs.channels = nil
	for slot := 1; slot <= 4; slot++ {
		if hs.toggleOn[slot-1] {
			s.toggleOffLocked(c, slot, now, false)
		}
	}
	s.stopAttackLocked(c, true)
	s.stopPvpAttackLocked(c, true)
	c.stopArrivalLocked()
	c.hasDest = false
	cx, cy := c.posAtLocked(float32(now))
	c.x, c.y, c.vx, c.vy, c.snapT = cx, cy, 0, 0, float32(now)

	// Drop timed buffs (icons/fx) — death cleanses.
	var keep []statMod
	for _, mod := range hs.st.mods {
		if mod.until == 0 {
			keep = append(keep, mod)
			continue
		}
		if mod.buffEffID != 0 {
			s.push(c, battleproto.CmdRemEffector, amf.NewArray().Set("id", mod.buffEffID))
		}
		s.fxEndLocked(c, mod.fxUID)
	}
	hs.st.mods = keep
	hs.st.shield = 0
	hs.st.hots = nil // death cleanses; stop healing the corpse
	hs.st.dots = nil

	// Freeze the avatar in place (velocity 0) BEFORE ON_KILL so the client stops
	// dead-reckoning it and the death clip plays at the death spot instead of
	// sliding along the last run leg.
	c.sendPosLocked(s, cx, cy, 0, 0, float32(now))

	// Broadcast the death so teammates see this avatar drop (the client plays the
	// death clip on the object they render for this player).
	s.broadcastAvatarObjLocked(c, battleproto.CmdOnKill, amf.NewArray().
		Set("killer", killer).Set("id", c.objID))

	// Abominator's «Трупоглот»: while transformed (TOGGLE slot 4 on), a nearby ENEMY
	// avatar's death grants a PERMANENT +20 max HP ("При смерти вражеской аватары
	// рядом, Трупоглот навсегда получит прибавку 20 к своему здоровью"). Only reachable
	// in PvP (Штурм) -- Hunt has no enemy avatars.
	for _, mem := range c.members() {
		mhs := mem.huntState
		if mhs == nil || mem == c || mhs.av.Prefab != "Avtr_HK_Abominator" || !mhs.toggleOn[3] {
			continue
		}
		if mhs.deadUntil > 0 || !arenaEnemies(c, mem) {
			continue
		}
		mx, my := mem.posAtLocked(float32(now))
		if math.Hypot(float64(mx-cx), float64(my-cy)) > abominatorFeastRange {
			continue
		}
		mhs.st.mods = append(mhs.st.mods, statMod{stat: "max_hp", value: 20, until: 0, src: "abominatorFeast"})
		s.pushPlayerStatsLocked(mem, now)
	}
	// Respawn timer (absolute battle time) is this player's own UI only.
	s.push(c, battleproto.CmdRespawn, amf.NewArray().
		Set("id", c.selfPlayerID).
		Set("time", hs.deadUntil))
}

// respawnPlayerLocked revives at the spawn point: the SYNC re-add re-enables
// the disabled GameObject client-side (GameObjManager.EnableGameObject:
// DeathBehaviour.Reborn + Animation.Reset + NetSyncTransform on), the no-time
// RESPAWN plays the reborn fx, and — because the client drops every effector
// of an object that left relevance — the whole effector set is re-sent.
// rebornActivateRange is how close the avatar must get to a Reborn_point for it
// to become the active respawn checkpoint.
const rebornActivateRange = 6.0

// tickRebornLocked activates the nearest Reborn_point the avatar is standing on,
// making it the respawn checkpoint (and deactivating the previous). The player
// starts on the battle-start Reborn and picks up later ones as they advance.
func (s *Server) tickRebornLocked(c *conn, now float64) {
	hs := c.huntState
	if hs == nil || hs.deadUntil > 0 || len(hs.m.Reborn) == 0 {
		return
	}
	px, py := c.posAtLocked(float32(now))
	for i, r := range hs.m.Reborn {
		if i == hs.rebornIdx {
			continue
		}
		if math.Hypot(float64(px)-r.X, float64(py)-r.Y) <= rebornActivateRange {
			hs.rebornIdx = i
			hs.respawnX, hs.respawnY = float32(r.X), float32(r.Y)
			if debugCombat {
				log.Printf("battle: %s reborn checkpoint %d activated at (%.1f,%.1f)",
					c.RemoteAddr(), i, r.X, r.Y)
			}
			break
		}
	}
}

func (s *Server) respawnPlayerLocked(c *conn, now float64) {
	hs := c.huntState
	hs.deadUntil = 0
	hs.diedAt = 0
	hs.hp = hs.maxHPLocked(now)
	hs.mana = hs.maxManaLocked(now)

	// «Арена»: pick a fresh spawn far from the enemy each death rather than a fixed
	// checkpoint, so a killed player doesn't rematerialise where they just fell. Set it
	// into the checkpoint fields the read below already uses.
	if c.inst != nil && c.inst.arena != nil {
		hs.respawnX, hs.respawnY = s.arenaSpawnLocked(c.inst, c, now)
	}

	// Respawn at the last-activated checkpoint (falls back to Spawn() before any
	// Reborn_point has been reached / when the map has no checkpoints).
	sx, sy := hs.respawnX, hs.respawnY
	if sx == 0 && sy == 0 {
		sx64, sy64 := hs.m.Spawn()
		sx, sy = float32(sx64), float32(sy64)
	}
	c.x, c.y, c.vx, c.vy, c.snapT = sx, sy, 0, 0, float32(now)

	if hs.corpseHidden {
		idx := hs.tr.add(c.objID)
		s.push(c, battleproto.CmdSync, amf.NewArray().Set("data",
			newSyncBlob(float32(now)).addObject(c.objID).
				position(idx, sx, sy, 0, 0, float32(now)).
				build(hs.tr.count())))
		hs.corpseHidden = false
	} else if idx := hs.tr.index(c.objID); idx >= 0 {
		s.push(c, battleproto.CmdSync, amf.NewArray().Set("data",
			newSyncBlob(float32(now)).position(idx, sx, sy, 0, 0, float32(now)).
				build(hs.tr.count())))
	}
	s.push(c, battleproto.CmdRespawn, amf.NewArray().Set("id", c.selfPlayerID))
	s.pushPlayerStatsLocked(c, now)
	s.syncSelfLocked(c, syncHealth, syncMana)
	s.broadcastPlayerStatsLocked(c)
	s.sendEffectorsLocked(c, now)
	// Re-show this avatar on teammates' clients (its corpse was removed from
	// them on death). An enemy respawning hero is still subject to fog: the
	// respawn packet is a local event, not a global reveal. The next vision pass
	// will introduce it to an enemy viewer only if the spawn point is visible.
	dotaFog := c.inst != nil && c.inst.dota != nil
	for _, other := range c.members() {
		if other == c || other.huntState == nil {
			continue
		}
		if dotaFog && other.playerTeam() != c.playerTeam() {
			visionSources := dotaTeamVisionSourcesLocked(c.inst, other.playerTeam(), now)
			if !botVisibleEnemyMemberLocked(c.inst, other.playerTeam(), c, now, visionSources) {
				continue
			}
		}
		s.renderAvatarWithStatsLocked(other, c, now)
	}
	// Mobs lose interest in the fresh corpse walker until re-aggroed; and any pack
	// dragged onto this checkpoint mid-fight is sent home, so the player never
	// resurrects inside an aggro bubble (spawn-camp death loop). The mob-free spawn
	// ring only controls where mobs SPAWN -- leash is player-relative, so without
	// this a chased pack freezes on the checkpoint and re-aggros the fresh player.
	//
	// Only HOMED mobs may be evicted. A «Штурм» creep has no home: it was produced by
	// a wave generator and its spawnX/spawnY are the zero value, so this used to
	// teleport the player's own lane to the middle of the map -- silently, since a
	// creep is never `shown` (so no DELETE_OBJECT) and returnMobHomeLocked zeroes the
	// velocity (so the position sync was suppressed too). The trigger is the ordinary
	// losing scenario: dying near your own base while your creeps push past it. Their
	// aggro is still cleared -- a creep re-picks its target every tick regardless.
	for _, m := range hs.mobs {
		m.aggro = false
		if !m.homed || m.dead {
			continue
		}
		if math.Hypot(float64(m.x-sx), float64(m.y-sy)) < respawnEvictRange {
			s.returnMobHomeLocked(c, m, now)
		}
	}
}
