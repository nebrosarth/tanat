package battleserver

import (
	"math"
	"sort"
	"strconv"
	"strings"

	"tanatserver/internal/gamedata"
)

// The team orchestrator owns only the strategic layer. It chooses one live-state
// objective per side and gives every bot an explicit assignment. The bot's local
// combat, retreat, recovery, and tower-safety code remains the authority that can
// decline or interrupt that assignment.
const (
	botMacroBase    = "base"
	botMacroAltar   = "altar"
	botMacroPush    = "push"
	botMacroRally   = "rally"
	botMacroCover   = "cover"
	botMacroLane    = "lane"
	botMacroRecover = "recover"

	botMacroReasonFullMobilization        = "full_mobilization"
	botMacroReasonMobilizationPreparation = "full_mobilization_prepare"
	botMacroReasonPartialMobilization     = "partial_mobilization"
	botMacroReasonObjectiveStaging        = "objective_staging"
)

type botMacroAssignment struct {
	Mode         string
	Lane         int
	FarmLane     int
	FarmLaneSet  bool
	BaselineLane int
	ObjectiveID  int32
	Role         string
	Reason       string
	Aggressive   bool
	Coverage     bool
}

type botTeamPlan struct {
	Team         int32
	AIVersion    int
	AIVersionSet bool
	Mode         string
	Lane         int
	ObjectiveID  int32
	Objective    string
	Reason       string
	FocusTarget  int32
	Assignments  map[int32]botMacroAssignment
}

const (
	botMacroBasePressureRadius = 42.0
	botMacroLanePointRadius    = 28.0
	botMacroCoverageRadius     = 34.0
	// Base pressure at or above the existing three-responder threshold is too
	// dangerous to spend a healthy bot on a counter-push.
	botMacroBaseSeverePressure = 6
	// A live plan is sticky until a replacement has a material live-state
	// advantage. This is deliberately a score margin, not a match-clock gate.
	botMacroSwitchMargin = 20.0
	// Barracks pressure is evaluated independently per lane. Do not swing the
	// whole defense to a neighboring barracks for a one- or two-point forecast
	// difference; that was the source of the observed lane-0/lane-1 thrash.
	botMacroBarracksSwitchDelta = 2
	// A barracks wave normally needs one committed defender. A second/third
	// responder is reserved for materially denser pressure; ordinary creep
	// contact must not pull the whole roster into the same base anchor.
	botMacroBarracksReinforceSeverity = 8
	botMacroBarracksCriticalSeverity  = 12
	// Once a front structure has lost at least 15% of its health, the lane has
	// strategic investment worth completing. This is separate from the 70%
	// execution finish window used by the last-stand movement exception.
	botMacroCommitDamageHPFrac = 0.85
)

// botPlanTeamsLocked is called exactly once before the ticker's member loop. It
// snapshots live member/mob state into two deterministic plans, then writes each
// bot's assignment before any bot is allowed to think.
func (s *Server) botPlanTeamsLocked(inst *huntInstance, now float64) {
	if inst == nil || inst.dota == nil || len(inst.bots) == 0 {
		return
	}
	d := inst.dota
	if d.teamPlans == nil {
		d.teamPlans = map[int32]botTeamPlan{}
	}
	if d.teamPlanTelemetryKey == nil {
		d.teamPlanTelemetryKey = map[int32]string{}
	}
	for _, team := range []int32{dotaTeamHuman, dotaTeamElf} {
		if !botAIProfileForVersion(botTeamAIVersionLocked(inst, team)).UsesTeamOrchestrator() {
			// AI-0 is intentionally outside the team-planning system. Remove any
			// stale state defensively, then leave the bot brains untouched so their
			// local phase/laning logic remains authoritative.
			delete(d.teamPlans, team)
			delete(d.teamPlanTelemetryKey, team)
			continue
		}
		plan := s.botPlanTeamLocked(inst, team, now)
		d.teamPlans[team] = plan
		key := botTeamPlanKey(plan)
		if d.teamPlanTelemetryKey[team] != key {
			d.teamPlanTelemetryKey[team] = key
			s.telemetryRecordBotTeamPlanLocked(inst, plan, now)
		}
		for id, brain := range inst.bots {
			if brain == nil || brain.c == nil || brain.c.playerTeam() != team {
				continue
			}
			assignment := plan.Assignments[id]
			previousAssignment := brain.macroAssignment
			baselineCoverage := assignment.Role == "lane_cover" && assignment.Reason == "baseline_lane_coverage"
			baseDefense := assignment.Mode == botMacroBase && assignment.Lane >= 0
			coverageHandoff := baselineCoverage && (previousAssignment.Mode != botMacroLane &&
				previousAssignment.Mode != botMacroCover || previousAssignment.Role != "lane_cover")
			baseFarmCoverageHandoff := assignment.Mode == botMacroBase && assignment.Role == "cover" &&
				assignment.FarmLaneSet && previousAssignment.Mode != botMacroBase
			farmLaneChanged := assignment.FarmLaneSet &&
				(!previousAssignment.FarmLaneSet || previousAssignment.FarmLane != assignment.FarmLane)
			farmAssignmentLane := assignment.Lane
			if (baselineCoverage || baseDefense) && assignment.FarmLaneSet && assignment.FarmLane >= 0 {
				farmAssignmentLane = assignment.FarmLane
			}
			if (baselineCoverage || baseDefense) && farmAssignmentLane >= 0 && brain.farmLane != farmAssignmentLane {
				// A live coverage hand-off must invalidate the previous movement
				// leg immediately. Leaving the old FarmLane/waypoint alive for one
				// think interval made a bot appear assigned to lane 0 while it kept
				// walking toward lane 1, which is enough to miss the next creep XP
				// event under the 20u proximity rule.
				brain.farmLane = farmAssignmentLane
				brain.farmTarget = 0
				brain.farmDecision = "lane_reassigned"
				brain.c.stopArrivalLocked()
			}
			// Any live assignment that changes the farm lane needs the same
			// spatial hand-off, including a base defender. A defender still has
			// an XP obligation on its authored line; without this flag it keeps
			// walking from the old lane while the next wave reaches the new one.
			if (farmLaneChanged || coverageHandoff || baseFarmCoverageHandoff) &&
				(baselineCoverage || baseDefense || assignment.Mode == botMacroLane || assignment.Mode == botMacroCover || assignment.Role == "defender") {
				brain.farmLaneArrivalPending = true
			}
			brain.macroAssignment = assignment
		}
	}
}

func (s *Server) botPlanTeamLocked(inst *huntInstance, team int32, now float64) botTeamPlan {
	plan := botTeamPlan{Team: team, AIVersion: botTeamAIVersionLocked(inst, team), AIVersionSet: true, Lane: -1, Assignments: map[int32]botMacroAssignment{}}
	if focus := botSelectTeamFocusTargetLocked(inst, team, now); focus != nil {
		plan.FocusTarget = focus.objID
	}
	previous, hasPrevious := inst.dota.teamPlans[team]
	hasPrevious = hasPrevious && len(previous.Assignments) > 0
	// Plan hysteresis and farm-lane retention were introduced after AI-10. The
	// baseline profile intentionally recomputes these decisions from live state,
	// which makes an AI-10 vs AI-20 match a real behavioural comparison.
	if !botAIProfileForVersion(plan.AIVersion).UsesPlanHysteresis() {
		hasPrevious = false
	}
	var bots []*botBrain
	active := 0
	for _, mem := range inst.members {
		if mem == nil || mem.huntState == nil || mem.playerTeam() != team {
			continue
		}
		if brain, ok := inst.bots[mem.objID]; ok {
			if brain != nil {
				bots = append(bots, brain)
				if mem.huntState.deadUntil == 0 && !brain.retreating {
					active++
				}
			}
			continue
		}
		if botHumanMacroActiveLocked(inst, mem, now) {
			active++
		}
	}
	sort.Slice(bots, func(i, j int) bool { return bots[i].c.objID < bots[j].c.objID })
	for _, brain := range bots {
		// Baseline lane ownership is immutable. Coordinated work is an overlay;
		// when a bot is not selected as a responder it explicitly covers this lane.
		plan.Assignments[brain.c.objID] = botMacroAssignment{
			Mode: botMacroLane, Lane: brain.lane, BaselineLane: brain.lane,
			Role: "lane_cover", Reason: "baseline_lane_coverage", Coverage: true,
		}
	}

	defenseStructure, defenseSeverity := s.botOwnDefenseStructureThreatLocked(inst, team, now)
	// A gun is a checkpoint, not a reason for a ready local push to abandon its
	// win condition. Keep the gun's local danger visible to the combat layer but
	// let a committed wave/group convert pressure on the enemy's next structure.
	// Barracks and altar threats still override the push immediately.
	_, offensiveObjective, _, _ := s.botBestMacroLaneLocked(inst, team, now)
	conversionReady := s.botObjectiveConversionReadyLocked(inst, team, offensiveObjective, now)
	ignoreGunDefense := defenseStructure != nil && defenseStructure.dotaRole == gamedata.DotaGun &&
		defenseSeverity > 0 && conversionReady
	if ignoreGunDefense && hasPrevious && previous.Mode == botMacroBase {
		// Keep a stable base plan while the own gun is under direct fire. The
		// base responder overlay can still reserve a counter-pusher for a low
		// enemy objective; alternating base/push every world tick strands both
		// groups between the two destinations.
		ignoreGunDefense = false
	}
	if !ignoreGunDefense && s.botPreserveCommittedPushAgainstGunLocked(inst, team, previous, hasPrevious, bots, defenseStructure, defenseSeverity, now) {
		// A gun contact is a temporary checkpoint trade, not a reason to reset
		// an already assembled offensive group. This hysteresis prevents the
		// observed push/base/push oscillation when a single projectile changes the
		// instantaneous conversion predicate on alternating world ticks.
		ignoreGunDefense = true
	}
	// A five-bot side was still losing a decisive race after a successful
	// mobilization: the opponent's gun was at a few percent, but a newly detected
	// full-health barracks wave made the whole plan flip to base defense. Keep the
	// barracks/altar priority when they are materially damaged; when the defended
	// object is only under contact and the already-committed enemy front object is
	// in its finish window, preserve the conversion and let the responder overlay
	// keep a defender. This is a live-state trade decision, not a roster-size or
	// match-clock shortcut.
	preserveCriticalFinish := s.botPreserveCriticalFinishLocked(inst, team, previous, hasPrevious, bots, defenseStructure, now)
	if defenseStructure != nil && defenseSeverity > 0 && !ignoreGunDefense && !preserveCriticalFinish {
		structure := defenseStructure
		plan.Mode = botMacroBase
		plan.ObjectiveID = structure.id
		plan.Lane = botNearestLaneToPointLocked(inst.dota, structure.x, structure.y)
		switch structure.dotaRole {
		case gamedata.DotaCreepTower:
			plan.Reason, plan.Objective = "own_barracks_under_live_threat", "barracks_defense"
		case gamedata.DotaGun:
			plan.Reason, plan.Objective = "own_gun_under_live_threat", "gun_defense"
		case gamedata.DotaAltar:
			plan.Reason, plan.Objective = "own_altar_under_live_pressure", "altar_defense"
		}
	} else {
		if enemyAltar := botTeamAltarLocked(inst, otherDotaTeam(team)); enemyAltar != nil &&
			!enemyAltar.dead && inst.dota.altarVulnerableLocked(enemyAltar) {
			plan.Mode, plan.Reason = botMacroAltar, "enemy_altar_open"
			plan.ObjectiveID, plan.Objective = enemyAltar.id, "enemy_altar"
			plan.Lane = botNearestLaneToPointLocked(inst.dota, enemyAltar.x, enemyAltar.y)
		} else {
			lane, objective, progress, coverage := s.botBestMacroLaneLocked(inst, team, now)
			continuedPush := false
			if preferredLane, preferredObjective, preferredProgress, preferredCoverage, ok :=
				s.botContinuePushLaneLocked(inst, team, previous, hasPrevious, now); ok {
				// A destroyed gun invalidates only the old object, not the lane
				// investment. Advance to the next live front object on that same
				// route so a push reaches barracks instead of scattering over three
				// partially damaged lanes.
				lane, objective, progress, coverage = preferredLane, preferredObjective, preferredProgress, preferredCoverage
				continuedPush = true
			}
			criticalLane, criticalObjective := s.botCriticalEnemyObjectiveLocked(inst, team)
			criticalFinish := criticalObjective != nil
			if criticalFinish {
				// A reachable damaged front objective is a conversion debt of its
				// own. Do not let a farm-debt rescue abandon a structure that is
				// already close to destruction.
				lane, objective, progress = criticalLane, criticalObjective, true
			}
			plan.Lane, plan.ObjectiveID = lane, 0
			if objective != nil {
				plan.ObjectiveID = objective.id
				plan.Objective = botMacroObjectiveName(objective)
			}
			farmRescue := botAIProfileForVersion(plan.AIVersion).UsesFarmRescue() &&
				s.botTeamFarmRescueRequiredLocked(inst, team)
			farmCoverageRequired := botAIProfileForVersion(plan.AIVersion).UsesFarmSafeWave() &&
				s.botTeamFarmCoverageRequiredLocked(inst, team, now)
			if farmRescue && hasPrevious && (previous.Mode == botMacroPush || previous.Mode == botMacroRally) &&
				previous.ObjectiveID != 0 && objective != nil && previous.ObjectiveID == objective.id &&
				botMacroObjectiveDamaged(objective) {
				// Farm debt belongs to the weakest available body, not to an
				// already-invested objective group. A first structure chip used to
				// be enough for the next planner pass to reset the whole push into
				// lane mode; the objective then healed its tempo advantage while
				// every responder walked back toward farm. Keep the route alive when
				// the named objective is still the live front and let the later farm
				// reserve/coverage pass peel only a spare bot when conversion remains
				// viable. This is keyed to the previous objective's live damage, not
				// to match time or the number of bots.
				progress, _ := s.botMacroLaneProgressLocked(inst, team, previous.Lane, objective, now)
				if progress {
					farmRescue = false
				}
			}
			if continuedPush && hasPrevious && (previous.Mode == botMacroPush || previous.Mode == botMacroAltar || botAnyMobilizationReason(previous.Reason)) && !farmCoverageRequired {
				// The previous lane commitment has just advanced to its next live
				// front object (normally because the old gun/barracks died). Keep the
				// conversion moving through the same lane; otherwise one destroyed
				// checkpoint immediately turns into a farm-debt rescue and hands the
				// initiative back to the opponent.
				if !farmCoverageRequired {
					farmRescue = false
				}
			}
			conversionReady := s.botObjectiveConversionReadyLocked(inst, team, objective, now)
			objectiveStaging := !conversionReady && s.botObjectiveStagingRequiredLocked(inst, team, objective, previous, hasPrevious, now)
			// When one live body is the only local coverage of a lane and its allied
			// wave has already reached the objective approach, keep that body in a
			// rally assignment until the local power gate is met. Sending the lone
			// responder straight into a full-health gun creates the same retreat loop
			// as a premature push; this is based on wave/coverage state, never on the
			// total number of avatars in the match.
			objectiveRally := !conversionReady && !objectiveStaging &&
				progress && coverage == 1 && s.botObjectiveHasAlliedWaveLocked(inst, team, objective)
			if objectiveStaging && !farmCoverageRequired {
				// Farm debt belongs to the weakest lane owner, not to the whole
				// offensive group. When the next front object is already locally
				// reachable, keep the group staged there and let the wave/power gate
				// decide when the structure can be hit.
				farmRescue = false
			}
			// A materially damaged front object is already a strategic
			// commitment. Do not split the team into a half-push merely because
			// enemy pressure makes the instantaneous conversion predicate false;
			// that is exactly when the full group must first heal and assemble.
			// Full mobilization is meaningful only after every roster member can
			// actually cast an ultimate. Before that breakpoint, keep the ordinary
			// conversion/farm loop alive so a low-level bot can reach its ult gate
			// instead of parking the whole team in an impossible preparation state.
			// A learned but cooling ultimate is the same tactical breakpoint: do
			// not freeze the roster in preparation for an attack that cannot launch.
			// When the local wave is actually in the conversion band and at least
			// two active responders cover it, that same launch-ready roster may
			// prepare a grouped attack. A full-health structure without a wave is
			// deliberately not enough: gathering under its fire only creates a
			// retreat loop. This is a tactical use of the ultimate gate, not a
			// match-clock phase, and it is suppressed while farm rescue is required.
			allUltimatesLearned := botMobilizationUltimatesLearnedLocked(bots)
			allUltimatesReady := allUltimatesLearned && botMobilizationUltimatesReadyLocked(bots, now)
			finishWindow := botMacroObjectiveFinishWindowLocked(objective)
			mobilizationContinuation := continuedPush && hasPrevious && botMobilizationReason(previous.Reason)
			// A structure already inside the execution window is a direct finish
			// debt. Do not hold its damage at 1-2% while waiting for a full-roster
			// rally; healthy assigned responders can close it immediately, and the
			// ordinary retreat gate still protects a critically injured bot.
			mobilizationOpportunity := progress && coverage >= 2 && conversionReady && !farmRescue
			mobilizationRequested := mobilizationContinuation || (allUltimatesReady &&
				((criticalFinish && !finishWindow) || (finishWindow && conversionReady) || mobilizationOpportunity))
			// Do not downgrade an already active objective conversion into a
			// non-aggressive muster on the same tick that the front object becomes
			// damaged enough to qualify for full mobilization. The group is already
			// doing useful work; resetting it to the rally point is exactly how a
			// winning five-bot side lost the final 30-second window. A fresh
			// mobilization may still start from lane/farm mode, and a previously
			// mobilized group remains governed by its launch gate.
			activeObjectiveCommit := hasPrevious && previous.Mode == botMacroPush &&
				previous.ObjectiveID != 0 && objective != nil && previous.ObjectiveID == objective.id &&
				previous.Reason != botMacroReasonFullMobilization &&
				previous.Reason != botMacroReasonMobilizationPreparation
			if !activeObjectiveCommit && hasPrevious && objective != nil {
				// A short defensive overlay can replace the team mode while one
				// responder is protecting a checkpoint. Preserve the offensive
				// conversion if that overlay still carries an explicit counter-push
				// assignment for this exact objective.
				for _, assignment := range previous.Assignments {
					if assignment.Role == botMacroCounterPushRole && assignment.ObjectiveID == objective.id {
						activeObjectiveCommit = true
						break
					}
				}
			}
			if activeObjectiveCommit {
				mobilizationRequested = false
			}
			mobilizationAlreadyCommitted := mobilizationContinuation && previous.Reason == botMacroReasonFullMobilization
			if !mobilizationAlreadyCommitted {
				mobilizationAlreadyCommitted = hasPrevious && previous.Reason == botMacroReasonFullMobilization &&
					previous.ObjectiveID != 0 && objective != nil && previous.ObjectiveID == objective.id
			}
			mobilizationReady := mobilizationRequested && (mobilizationAlreadyCommitted ||
				s.botMobilizationReadyLocked(inst, team, objective, bots, now))
			// Partial mobilization is the fallback when the objective is genuinely
			// convertible but a full-roster launch is blocked by a dead, injured,
			// cooling, or distant teammate. Send a compact healthy strike group now
			// and leave the remaining bodies on their farm/defense assignments. This
			// prevents a full-mobilization preparation state from freezing a winning
			// conversion until every roster member is available again.
			healthyResponders := s.botHealthyMacroResponderCountLocked(bots, now)
			partialWindow := criticalFinish || finishWindow || botMacroObjectiveCommitWindowLocked(objective) ||
				(conversionReady && !farmCoverageRequired)
			partialMobilizationRequested := conversionReady && !farmRescue && healthyResponders >= 2 && partialWindow &&
				(!mobilizationReady || !allUltimatesLearned)
			if hasPrevious && previous.Mode == botMacroPush && previous.Reason == botMacroReasonPartialMobilization &&
				previous.ObjectiveID != 0 && objective != nil && previous.ObjectiveID == objective.id && healthyResponders >= 2 &&
				!farmRescue && partialWindow {
				partialMobilizationRequested = true
			}
			// Coverage is a hard tactical prerequisite. If a visible wave has no
			// living teammate inside the authoritative XP radius, suspend every
			// optional objective/mobilization route until the orchestrator has
			// restored that presence. This is state-based: the gate opens again as
			// soon as the live wave is covered, with no opening-minute constant.
			if farmCoverageRequired {
				farmRescue = true
				mobilizationRequested = false
				partialMobilizationRequested = false
				objectiveRally = false
				objectiveStaging = false
			}
			// A visible enemy wave is an immediate XP obligation. Do not let a ready
			// objective, a mobilization continuation, or a conversion window pull the
			// last farm body out of an uncovered lane. The objective can wait for the
			// next covered wave; lost proximity XP cannot be recovered later.
			forceObjectivePush := !farmCoverageRequired && (criticalFinish ||
				mobilizationRequested || partialMobilizationRequested || conversionReady)
			if mobilizationRequested && !farmCoverageRequired {
				// A live finish window is a conversion commitment. Do not let a
				// marginal farm-debt rescue dissolve the group after it has already
				// invested damage into this exact front object.
				farmRescue = false
			}
			if farmRescue {
				plan.Mode = botMacroLane
				plan.Reason = "farm_debt_rescue"
			} else if forceObjectivePush {
				// A live wave and a favourable local combat state are enough to
				// convert the objective. No roster-size comparison is involved.
				plan.Mode = botMacroPush
				if partialMobilizationRequested {
					plan.Reason = botMacroReasonPartialMobilization
				} else if mobilizationRequested {
					if mobilizationReady {
						plan.Reason = botMacroReasonFullMobilization
					} else {
						plan.Reason = botMacroReasonMobilizationPreparation
					}
				} else {
					plan.Reason = "objective_conversion_ready"
				}
			} else if objectiveRally {
				plan.Mode = botMacroRally
				plan.Reason = "rally_for_conversion"
			} else if objectiveStaging {
				plan.Mode = botMacroPush
				plan.Reason = botMacroReasonObjectiveStaging
			} else if !progress || objective == nil {
				plan.Mode = botMacroLane
				plan.Reason = "no_live_lane_or_objective_progress"
			} else {
				// A full-health objective without a wave is not a staging target:
				// parking responders in its attack radius only feeds structure shots
				// while the next farm wave is still travelling. Keep baseline lane
				// ownership until the live conversion predicate turns true.
				plan.Mode = botMacroLane
				plan.Reason = "farm_lane_coverage"
			}
		}
	}
	if hasPrevious && !preserveCriticalFinish && s.botRetainTeamPlanLocked(inst, team, previous, plan, now) {
		desiredReason := plan.Reason
		previousReason := previous.Reason
		plan.Mode = previous.Mode
		plan.Lane = previous.Lane
		plan.ObjectiveID = previous.ObjectiveID
		plan.Objective = previous.Objective
		plan.Reason = previousReason
		// Plan hysteresis keeps the lane/objective stable, but preparation is a
		// live readiness phase. Do not freeze a preparation plan after the group
		// has actually become healthy and assembled, or keep an attack plan after
		// the group has lost readiness and must regroup.
		if botAnyMobilizationReason(desiredReason) {
			plan.Reason = desiredReason
		}
	}
	// An already active conversion is an execution state, not a staging
	// suggestion. If the live calculation briefly notices that a full
	// mobilization could also be requested, keep the named objective in its
	// aggressive conversion state; otherwise the next tick sends the same bots
	// back to muster and produces a prepare/convert oscillation.
	if hasPrevious && previous.Mode == botMacroPush && previous.Reason == "objective_conversion_ready" &&
		previous.ObjectiveID != 0 && previous.ObjectiveID == plan.ObjectiveID && botAnyMobilizationReason(plan.Reason) {
		plan.Mode = previous.Mode
		plan.Lane = previous.Lane
		plan.ObjectiveID = previous.ObjectiveID
		plan.Objective = previous.Objective
		plan.Reason = previous.Reason
	}
	// Once preparation has started, keep the same objective authoritative until
	// it dies (or an urgent own-base plan takes over). Re-evaluate only the
	// readiness phase so a group cannot oscillate between farm, gather, and push
	// while the building is still the live conversion target.
	if plan.Mode != botMacroBase && plan.Mode != botMacroAltar && hasPrevious &&
		botAnyMobilizationReason(previous.Reason) && previous.ObjectiveID != 0 {
		if objective := inst.mobs[previous.ObjectiveID]; objective != nil && !objective.dead &&
			botMacroObjectiveDamaged(objective) {
			plan.Mode = botMacroPush
			plan.Lane = previous.Lane
			plan.ObjectiveID = previous.ObjectiveID
			plan.Objective = previous.Objective
			if previous.Reason == botMacroReasonPartialMobilization {
				// Partial mobilization is already the fallback for an unavailable
				// roster member. Keep the healthy strike group on the damaged
				// objective instead of promoting it to a blocked full muster.
				plan.Reason = botMacroReasonPartialMobilization
			} else if previous.Reason == botMacroReasonFullMobilization {
				// Readiness is a launch gate, not a per-tick leash. Once the
				// group has begun attacking this exact objective, local combat and
				// retreat logic may peel an injured bot without resetting the whole
				// conversion back to the staging point.
				plan.Reason = botMacroReasonFullMobilization
			} else if s.botMobilizationReadyLocked(inst, team, objective, bots, now) {
				plan.Reason = botMacroReasonFullMobilization
			} else if previous.Reason == botMacroReasonMobilizationPreparation &&
				botMacroObjectiveFinishWindowLocked(objective) {
				// The named object entered its execution window while the roster
				// was preparing. Launch the full group when it has actually assembled.
				// If the roster is spatially split while the target is already in its
				// finish window, do not keep every nearby attacker parked in a muster
				// state: the available healthy strike group takes the objective while
				// distant members recover or converge. This is a live readiness/objective
				// decision, not a match-clock timeout.
				if botMobilizationUltimatesLearnedLocked(bots) && botMobilizationUltimatesReadyLocked(bots, now) {
					if s.botMobilizationReadyLocked(inst, team, objective, bots, now) {
						plan.Reason = botMacroReasonFullMobilization
					} else if s.botHealthyMacroResponderCountLocked(bots, now) >= 2 {
						plan.Reason = botMacroReasonPartialMobilization
					} else {
						plan.Reason = botMacroReasonMobilizationPreparation
					}
				} else {
					plan.Reason = "objective_conversion_ready"
				}
			} else if previous.Reason == botMacroReasonMobilizationPreparation &&
				!botMobilizationUltimatesReadyLocked(bots, now) {
				// Preparation is a launch gate, not a permanent lock. If the roster
				// cannot launch a full mobilization anymore (for example because a
				// member died or an ultimate is cooling), keep the damaged front
				// objective under ordinary conversion pressure instead of parking the
				// healthy survivors in a non-aggressive muster.
				plan.Reason = "objective_conversion_ready"
			} else {
				plan.Reason = botMacroReasonMobilizationPreparation
			}
		}
	}
	// A full mobilization is a commitment to finish the named structure, not
	// just a larger responder cap for one planning tick. Release viable
	// recovery/disengage latches before selecting responders so the same live
	// group can rejoin the push until the objective is destroyed. Critical HP
	// and dead bots remain excluded by the normal eligibility checks.
	s.botReleaseMobilizationRetreatsLocked(&plan, bots, inst, now)
	s.botReleaseObjectiveStagingRetreatsLocked(&plan, bots, inst, now)
	s.botReleaseObjectiveFinishRetreatsLocked(&plan, bots, inst, now)
	s.botAssignPlanRespondersLocked(&plan, previous, hasPrevious, bots, inst, team, now)
	if plan.Reason == botMacroReasonMobilizationPreparation {
		s.botAssignMobilizationPreparationLocked(&plan, bots, inst, now)
	}
	planObjective := botPlanObjectiveLocked(inst, team, plan)
	for _, brain := range bots {
		if brain.c.huntState.deadUntil > 0 {
			plan.Assignments[brain.c.objID] = botMacroAssignment{
				Mode: botMacroRecover, Lane: brain.lane, BaselineLane: brain.lane,
				Role: "dead", Reason: "dead_until_respawn",
			}
		} else if brain.retreating {
			assignment := plan.Assignments[brain.c.objID]
			altarAssault := plan.Mode == botMacroAltar && plan.Reason == "enemy_altar_open" &&
				assignment.Mode == botMacroAltar && s.botAltarAssaultResponderEligibleLocked(brain, planObjective, now)
			if altarAssault {
				// Keep the explicit altar-assault assignment. The local retreat
				// gate will clear this recoverable latch when the grouped ally
				// premise is true; overwriting it here would make that handoff
				// impossible until the bot reached the fountain.
				continue
			}
			plan.Assignments[brain.c.objID] = botMacroAssignment{
				Mode: botMacroRecover, Lane: brain.lane, BaselineLane: brain.lane,
				Role: "retreat", Reason: "local_retreat_authority",
			}
		}
	}
	// A push is a team objective, not a reason to suspend XP acquisition for
	// every member. Reserve one materially farm-indebted living bot on a live
	// wave when the remaining local power can still convert the objective. The
	// reservation is deliberately applied after responder selection: the
	// orchestrator knows exactly which body it can spare, while full
	// mobilization remains an explicit all-roster commitment.
	s.botAssignFarmLanesWithPreviousAtLocked(&plan, previous, hasPrevious, bots, inst, now)
	s.botAssignFarmReserveLocked(&plan, bots, inst, now)
	// Every live authored lane needs one local XP presence. A push assignment is
	// allowed to consume the spare bodies, but it must not consume the only
	// owner of a lane whose own wave is still alive. This is a live-wave
	// invariant, not an opening timer, so it remains useful after the first
	// wave and naturally disappears when a barracks stops producing creeps.
	s.botEnforceBaselineFarmCoverageLocked(&plan, bots, inst, now)
	return plan
}

// botEnforceBaselineFarmCoverageLocked preserves one farm-capable assignment
// per active authored lane. The old responder overlay could move every bot to
// one objective as soon as that objective became convertible; the other lanes
// then lost XP even though their own waves were still marching. Keeping one
// cover body per live lane retains map-wide XP while leaving every remaining
// healthy body available for the strategic push.
func (s *Server) botFarmCoverageDistanceLocked(brain *botBrain, lane int, team int32, visionSources []dotaVisionSource, now float64) float64 {
	if brain == nil || brain.c == nil || brain.c.inst == nil || brain.c.huntState == nil {
		return math.Inf(1)
	}
	c := brain.c
	cx, cy := c.posAtLocked(float32(now))
	bestEnemy := math.Inf(1)
	bestOwn := math.Inf(1)
	hasVisibleEnemy := false
	for _, mob := range botSortedMobs(c.inst) {
		if mob == nil || mob.dead || mob.structure || !botTeleportLaneCreep(mob) ||
			botLaneForCreep(c, mob) != lane {
			continue
		}
		if mob.team != team && (!mob.enemyOf(team) || !botVisibleEnemyMobLocked(team, mob, visionSources)) {
			continue
		}
		distance := math.Hypot(float64(mob.x-cx), float64(mob.y-cy))
		if mob.enemyOf(team) {
			hasVisibleEnemy = true
			if distance < bestEnemy {
				bestEnemy = distance
			}
		} else if distance < bestOwn {
			bestOwn = distance
		}
	}
	if hasVisibleEnemy {
		return bestEnemy
	}
	return bestOwn
}

// botFarmLaneHeroPressureLocked marks a lane owner as temporarily unavailable
// when a visible enemy avatar is already contesting that lane. The owner pass
// can then hand the lane to a spare before the retreating body leaves the XP
// radius. Active attack state is accepted as evidence even when the ordinary
// hero vision circle has just flickered at the boundary.
func (s *Server) botFarmLaneHeroPressureLocked(brain *botBrain, lane int, now float64) bool {
	if brain == nil || brain.c == nil || brain.c.inst == nil || brain.c.huntState == nil {
		return false
	}
	c := brain.c
	visionSources := dotaTeamVisionSourcesLocked(c.inst, c.playerTeam(), now)
	cx, cy := c.posAtLocked(float32(now))
	for _, enemy := range c.inst.members {
		if enemy == nil || enemy == c || enemy.huntState == nil || enemy.huntState.deadUntil > 0 || enemy.playerTeam() == c.playerTeam() {
			continue
		}
		ex, ey := enemy.posAtLocked(float32(now))
		active := enemy.huntState.pvpTarget == c.objID || enemy.huntState.attackTarget == c.objID ||
			c.huntState.pvpTarget == enemy.objID
		if !active && !botVisibleEnemyMemberLocked(c.inst, c.playerTeam(), enemy, now, visionSources) {
			continue
		}
		if botNearestLaneToPointLocked(c.inst.dota, ex, ey) != lane {
			continue
		}
		if active || math.Hypot(float64(cx-ex), float64(cy-ey)) <= botRetreatPressureRadius {
			return true
		}
	}
	return false
}

// botFarmOwnerUnavailableLocked starts a handoff before a lane owner reaches
// the hard retreat floor. A low-health body can still be alive and technically
// eligible, but keeping it as the formal owner delays the replacement until
// the next creep is already dying. This is live survivability/pressure state,
// not a match-clock rule.
func (s *Server) botFarmOwnerUnavailableLocked(brain *botBrain, lane int, now float64) bool {
	if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.huntState.deadUntil > 0 || brain.retreating {
		return true
	}
	if botHPFrac(brain.c.huntState, now) <= botSafeHPFrac {
		return true
	}
	return s.botFarmLaneHeroPressureLocked(brain, lane, now)
}

// botLaneHasLivingEnemyBarracksLocked is a map-state guarantee, not a vision
// shortcut: a living enemy barracks means this lane will keep producing enemy
// creeps even while the next wave is outside the team's sight. The caller may
// use it to retain a lane owner, but still must use normal fog-of-war checks
// before selecting or attacking a creep.
func botLaneHasLivingEnemyBarracksLocked(inst *huntInstance, team int32, lane int) bool {
	if inst == nil || inst.dota == nil || lane < 0 || lane >= len(inst.dota.m.Lanes) {
		return false
	}
	for _, mob := range botSortedMobs(inst) {
		if mob == nil || mob.dead || !mob.structure || mob.team == team || mob.dotaRole != gamedata.DotaCreepTower {
			continue
		}
		if botNearestLaneToPointLocked(inst.dota, mob.x, mob.y) == lane {
			return true
		}
	}
	return false
}

func (s *Server) botEnforceBaselineFarmCoverageLocked(plan *botTeamPlan, bots []*botBrain, inst *huntInstance, now float64) {
	if plan == nil || inst == nil || inst.dota == nil || len(inst.dota.m.Lanes) == 0 {
		return
	}
	// A full mobilization is an explicit all-roster commitment. Its launch gate
	// has already established the exceptional premise that permits the farm
	// coverage overlay to be suspended for the named objective. Partial
	// mobilization is not such an exception: it must still leave one live owner
	// on every active lane, otherwise its strike group simply converts one
	// structure while the other lanes lose XP.
	if botMobilizationReason(plan.Reason) || plan.Mode == botMacroAltar {
		return
	}
	// Ordinary conversion is still subordinate to live XP coverage. A wave that
	// dies outside the share radius is unrecoverable, while a full-health or
	// damaged checkpoint can wait for the next covered wave. Only the explicit
	// mobilization modes below are allowed to suspend this invariant; partial
	// mobilization itself reserves a farm/defense body through its responder cap.
	activeLanes := make(map[int]bool, len(inst.dota.m.Lanes))
	visibleEnemyLanes := make(map[int]bool, len(inst.dota.m.Lanes))
	visionSources := dotaTeamVisionSourcesLocked(inst, plan.Team, now)
	for _, mob := range botSortedMobs(inst) {
		if mob == nil || mob.dead || mob.structure || !botTeleportLaneCreep(mob) {
			continue
		}
		lane := botLaneForCreep(&conn{inst: inst}, mob)
		if lane < 0 || lane >= len(inst.dota.m.Lanes) {
			continue
		}
		// XP is awarded when the enemy lane creep dies. Own creeps are always
		// valid lane anchors; enemy creeps are added only when this team can
		// actually see them, so the reassignment does not bypass fog of war.
		if mob.team == plan.Team {
			activeLanes[lane] = true
		} else if mob.enemyOf(plan.Team) && botVisibleEnemyMobLocked(plan.Team, mob, visionSources) {
			visibleEnemyLanes[lane] = true
		}
	}
	// An enemy wave becomes a coverage trigger when the team has actually lost
	// a lane body to death/retreat. In the healthy steady state own creeps are
	// the lane anchor, preserving the strategic planner's ordinary push tests;
	// during a recovery hand-off, a visible enemy wave is the more urgent XP
	// obligation and must be covered even if our own front creep just died.
	for lane := range visibleEnemyLanes {
		activeLanes[lane] = true
	}
	// A standing enemy barracks is a persistent source of future XP events.
	// Keep one farm-capable body assigned to every such lane even when the
	// current enemy wave is hidden or between spawn/arrival ticks.
	for lane := range inst.dota.m.Lanes {
		if botLaneHasLivingEnemyBarracksLocked(inst, plan.Team, lane) {
			// A standing enemy barracks is a persistent future-wave source even
			// while the current wave is hidden by fog or the team is staging a
			// push elsewhere. Keep a coverage owner available; selecting/attacking
			// the actual creep still remains vision-bounded in the local brain.
			activeLanes[lane] = true
		}
	}
	// Authored lane ownership is the long-lived XP contract. A defensive plan
	// may temporarily move a cover body toward a threatened structure, but it
	// must not make that body change its farm line on every visibility update:
	// the next wave can already be inside XP range before the bot completes the
	// detour. Keep each currently living, non-retreating baseline lane active so
	// the owner pass below retains that lane; spare bodies remain free for the
	// strategic overlay and the emergency handoff logic.
	for _, brain := range bots {
		if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.huntState.deadUntil > 0 || brain.retreating {
			continue
		}
		assignment := plan.Assignments[brain.c.objID]
		lane := assignment.BaselineLane
		// A duplicate-baseline spare may be carrying a stable handoff to a
		// different line whose owner is recovering. Preserve that FarmLane here;
		// otherwise this coverage pass would immediately erase the handoff that
		// botAssignFarmLanesWithPreviousAtLocked just selected.
		if assignment.Role == "lane_cover" && assignment.FarmLaneSet &&
			assignment.FarmLane >= 0 && assignment.FarmLane != assignment.BaselineLane {
			lane = assignment.FarmLane
		}
		if lane < 0 || lane >= len(activeLanes) {
			lane = brain.lane
		}
		if lane >= 0 && lane < len(activeLanes) {
			activeLanes[lane] = true
		}
	}
	if len(activeLanes) == 0 {
		return
	}
	// A base plan can temporarily create two bodies at the same checkpoint:
	// one explicit defender and one cover responder carrying the same objective.
	// If the defender's authored lane is the uncovered lane, swap those roles:
	// keep the objective on the cover responder and release the defender to its
	// lane. This is still a one-defender invariant, and it prevents a retreating
	// owner from leaving a visible wave with no reachable XP body.
	if plan.Mode == botMacroBase {
		var defender, baseCover *botBrain
		for _, brain := range bots {
			if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.huntState.deadUntil > 0 || brain.retreating {
				continue
			}
			assignment := plan.Assignments[brain.c.objID]
			if assignment.Mode != botMacroBase || assignment.ObjectiveID == 0 {
				continue
			}
			if assignment.Role == "defender" {
				defender = brain
			} else if baseCover == nil && assignment.Coverage {
				baseCover = brain
			}
		}
		if defender != nil && baseCover != nil {
			defenderAssignment := plan.Assignments[defender.c.objID]
			lane := defenderAssignment.BaselineLane
			if lane < 0 {
				lane = defender.lane
			}
			objective := inst.mobs[defenderAssignment.ObjectiveID]
			liveStructureThreat := objective != nil && s.botDefenseStructureThreatSeverityLocked(inst, plan.Team, objective, now) > 0
			if activeLanes[lane] && !liveStructureThreat && s.botFarmCoverageDistanceLocked(defender, lane, plan.Team, visionSources, now) > 64.0 {
				coverAssignment := defenderAssignment
				coverAssignment.Mode = botMacroCover
				coverAssignment.Lane = lane
				coverAssignment.FarmLane = lane
				coverAssignment.FarmLaneSet = true
				coverAssignment.ObjectiveID = 0
				coverAssignment.Role = "lane_cover"
				coverAssignment.Reason = "baseline_lane_coverage"
				coverAssignment.Aggressive = false
				coverAssignment.Coverage = true
				plan.Assignments[defender.c.objID] = coverAssignment

				baseAssignment := plan.Assignments[baseCover.c.objID]
				baseAssignment.Role = "defender"
				baseAssignment.Coverage = false
				plan.Assignments[baseCover.c.objID] = baseAssignment
			}
		}
	}

	chosen := make(map[int32]bool, len(activeLanes))
	owner := make(map[int]int32, len(activeLanes))
	// Keep an already explicit lane/cover owner first while it is still close
	// enough to the live lane. A respawned/recovering owner can be hundreds of
	// units away from the next wave; retaining it in that case made the plan
	// look covered while every nearby teammate watched the wave die. The live
	// distance test is the hand-off trigger, not a match-time rule.
	for _, brain := range bots {
		if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.huntState.deadUntil > 0 || brain.retreating {
			continue
		}
		assignment := plan.Assignments[brain.c.objID]
		farmBorrowablePush := plan.Mode == botMacroPush && assignment.Mode == botMacroPush &&
			(assignment.Reason == botMacroReasonPartialMobilization || assignment.Reason == "objective_conversion_ready")
		baseFarmOwner := assignment.Mode == botMacroBase &&
			(assignment.Role == "defender" || assignment.Role == "cover") && assignment.FarmLaneSet
		if assignment.Mode != botMacroLane && assignment.Mode != botMacroCover && assignment.Role != botMacroCounterPushRole && !farmBorrowablePush && !baseFarmOwner {
			continue
		}
		// Baseline ownership is the stable movement contract. FarmLane may be
		// a temporary catch-up overlay from the previous planner pass; using it
		// as the owner key made the same bot oscillate between its authored line
		// and the objective lane every 200ms. Only use FarmLane when the authored
		// baseline is unavailable (for example after a lane's barracks is gone).
		lane := assignment.BaselineLane
		if baseFarmOwner && assignment.Role == "cover" && assignment.FarmLane >= 0 {
			lane = assignment.FarmLane
		}
		// BaselineLane is the stable authored route. Keep it even during a
		// visibility gap when its barracks is still alive; otherwise the farm
		// allocator alternates 0->1->0 every planner tick as waves enter/leave
		// vision, and the bot never reaches either wave. Only fall back to the
		// temporary FarmLane when this route has no future wave source and no
		// current live lane anchor.
		baselinePersistent := lane >= 0 && lane < len(activeLanes) &&
			(activeLanes[lane] || botLaneHasLivingEnemyBarracksLocked(inst, plan.Team, lane))
		if !baselinePersistent {
			lane = brain.lane
		}
		if lane < 0 || (lane < len(activeLanes) && !activeLanes[lane] &&
			!botLaneHasLivingEnemyBarracksLocked(inst, plan.Team, lane)) {
			lane = assignment.FarmLane
		}
		if lane < 0 || !activeLanes[lane] || owner[lane] != 0 {
			continue
		}
		if baseFarmOwner {
			// A base defender is also a farm owner on its authored line. Keep the
			// strategic objective assignment intact; botBaseDefenseTickLocked still
			// intercepts the threatened structure, but the owner ledger must not
			// manufacture a spare-lane handoff every planner tick.
			if assignment.Role == "defender" {
				// The authored lane is the source of truth for a defender. A stale
				// FarmLane from an earlier rescue pass must not make the defender
				// appear to cover one lane while its behavior walks to another.
				assignment.FarmLane = lane
				assignment.FarmLaneSet = true
			}
			plan.Assignments[brain.c.objID] = assignment
			owner[lane] = brain.c.objID
			chosen[brain.c.objID] = true
			continue
		}
		coverageDistance := s.botFarmCoverageDistanceLocked(brain, lane, plan.Team, visionSources, now)
		if s.botFarmOwnerUnavailableLocked(brain, lane, now) {
			// The lane owner is about to disengage or is already being chased. Do
			// not retain it as a formal owner while a spare can begin the handoff.
			continue
		}
		// A standing enemy barracks is enough to retain a distant owner while
		// the next wave is hidden, but not after a visible wave has materialized.
		// In that case the distance is concrete XP debt: release the stale owner
		// so the nearest healthy spare can take the hand-off immediately.
		handoffDistance := 64.0
		if visibleEnemyLanes[lane] {
			// A visible wave is an immediate XP obligation. The old broad
			// hand-off ring treated a body 20-30u away as covered and waited
			// until the creep had already died. Future/hidden waves retain the
			// wider stability ring so ownership does not oscillate in fog.
			handoffDistance = dotaXPShareRadius * 0.95
		}
		if coverageDistance > handoffDistance &&
			(!botLaneHasLivingEnemyBarracksLocked(inst, plan.Team, lane) || visibleEnemyLanes[lane]) {
			// Do not move the only remaining farm-capable body just because a
			// different visible lane is currently closer. With no spare, that
			// creates a map-iteration-dependent 0->2->0 oscillation and abandons
			// both waves. Keep the authored owner on its route; a real spare will
			// take the hand-off on a later pass.
			hasSpare := false
			for _, candidate := range bots {
				if candidate == nil || candidate == brain || candidate.c == nil || candidate.c.huntState == nil ||
					candidate.c.huntState.deadUntil > 0 || candidate.retreating ||
					botHPFrac(candidate.c.huntState, now) <= botSafeHPFrac || chosen[candidate.c.objID] {
					continue
				}
				candidateAssignment := plan.Assignments[candidate.c.objID]
				farmBorrowablePush := plan.Mode == botMacroPush && candidateAssignment.Mode == botMacroPush &&
					(candidateAssignment.Reason == botMacroReasonPartialMobilization || candidateAssignment.Reason == "objective_conversion_ready")
				if candidateAssignment.Mode == botMacroBase || candidateAssignment.Mode == botMacroRecover || candidateAssignment.Role == "defender" ||
					(botAnyMobilizationReason(candidateAssignment.Reason) && !farmBorrowablePush) {
					continue
				}
				hasSpare = true
				break
			}
			if hasSpare {
				continue
			}
		}
		// A cover responder selected for the objective is still a farm owner
		// only if its execution assignment is removed. Leaving ObjectiveID and
		// objective_staging attached makes botCoverageTickLocked keep walking to
		// the structure, so the farm lane would be present in telemetry but not
		// in the actual movement order.
		wasLaneAssignment := assignment.Mode == botMacroLane
		if !wasLaneAssignment {
			assignment.Mode = botMacroCover
		}
		assignment.Lane = lane
		assignment.FarmLane = lane
		assignment.FarmLaneSet = true
		assignment.ObjectiveID = 0
		if !wasLaneAssignment {
			assignment.Role = "lane_cover"
			assignment.Reason = "baseline_lane_coverage"
		}
		assignment.Aggressive = false
		assignment.Coverage = true
		plan.Assignments[brain.c.objID] = assignment
		owner[lane] = brain.c.objID
		chosen[brain.c.objID] = true
	}

	// Fill uncovered lanes from the nearest safe spare body. Distance to the
	// currently visible/owned lane wave is more useful than authored-lane rank
	// during a recovery hand-off: the objective is to keep XP coverage alive
	// while the original owner redeploys.
	lanes := make([]int, 0, len(activeLanes))
	for lane := range activeLanes {
		lanes = append(lanes, lane)
	}
	// Resolve the closest eligible body for the most reachable lane first.
	// Iterating numeric lane order let a distant top-lane assignment consume
	// the only nearby centre-lane cover during a recovery hand-off.
	sort.Slice(lanes, func(i, j int) bool {
		bestDistance := func(lane int) float64 {
			best := math.Inf(1)
			for _, brain := range bots {
				if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.huntState.deadUntil > 0 ||
					brain.retreating || botHPFrac(brain.c.huntState, now) <= botSafeHPFrac || chosen[brain.c.objID] {
					continue
				}
				assignment := plan.Assignments[brain.c.objID]
				farmBorrowablePush := plan.Mode == botMacroPush && assignment.Mode == botMacroPush &&
					(assignment.Reason == botMacroReasonPartialMobilization || assignment.Reason == "objective_conversion_ready")
				if assignment.Mode == botMacroBase || assignment.Mode == botMacroRecover || assignment.Role == "defender" ||
					(botAnyMobilizationReason(assignment.Reason) && !farmBorrowablePush) {
					continue
				}
				if distance := s.botFarmCoverageDistanceLocked(brain, lane, plan.Team, visionSources, now); distance < best {
					best = distance
				}
			}
			return best
		}
		left, right := bestDistance(lanes[i]), bestDistance(lanes[j])
		if left != right {
			return left < right
		}
		return lanes[i] < lanes[j]
	})
	for _, lane := range lanes {
		if owner[lane] != 0 {
			continue
		}
		var best *botBrain
		bestDistance := math.Inf(1)
		for _, brain := range bots {
			if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.huntState.deadUntil > 0 ||
				brain.retreating || botHPFrac(brain.c.huntState, now) <= botSafeHPFrac || chosen[brain.c.objID] {
				continue
			}
			assignment := plan.Assignments[brain.c.objID]
			farmBorrowablePush := plan.Mode == botMacroPush && assignment.Mode == botMacroPush &&
				(assignment.Reason == botMacroReasonPartialMobilization || assignment.Reason == "objective_conversion_ready")
			if assignment.Mode == botMacroBase || assignment.Mode == botMacroRecover || assignment.Role == "defender" ||
				(botAnyMobilizationReason(assignment.Reason) && !farmBorrowablePush) {
				continue
			}
			distance := s.botFarmCoverageDistanceLocked(brain, lane, plan.Team, visionSources, now)
			if distance < bestDistance || (distance == bestDistance && (best == nil || brain.c.objID < best.c.objID)) {
				best, bestDistance = brain, distance
			}
		}
		if best == nil {
			continue
		}
		assignment := plan.Assignments[best.c.objID]
		assignment.Mode = botMacroCover
		assignment.Lane = lane
		assignment.FarmLane = lane
		assignment.FarmLaneSet = true
		assignment.ObjectiveID = 0
		assignment.Role = "lane_cover"
		assignment.Reason = "baseline_lane_coverage"
		assignment.Aggressive = false
		assignment.Coverage = true
		plan.Assignments[best.c.objID] = assignment
		owner[lane] = best.c.objID
		chosen[best.c.objID] = true
	}
	// If the authored owner is recovering, a base cover is the next safest
	// hand-off body. Keep defenders on the structure whenever a cover exists,
	// but do not leave a visible lane completely unrepresented just because the
	// remaining bot still carries a base assignment.
	for lane := range activeLanes {
		if owner[lane] != 0 {
			continue
		}
		var best *botBrain
		bestDistance := math.Inf(1)
		for _, brain := range bots {
			if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.huntState.deadUntil > 0 ||
				brain.retreating || botHPFrac(brain.c.huntState, now) <= botSafeHPFrac || chosen[brain.c.objID] {
				continue
			}
			assignment := plan.Assignments[brain.c.objID]
			if assignment.Mode != botMacroBase || assignment.Role != "cover" || assignment.ObjectiveID == 0 {
				continue
			}
			distance := s.botFarmCoverageDistanceLocked(brain, lane, plan.Team, visionSources, now)
			if distance < bestDistance || (distance == bestDistance && (best == nil || brain.c.objID < best.c.objID)) {
				best, bestDistance = brain, distance
			}
		}
		if best == nil || bestDistance > botLaneApproachRadius {
			continue
		}
		assignment := plan.Assignments[best.c.objID]
		assignment.Mode = botMacroCover
		assignment.Lane = lane
		assignment.FarmLane = lane
		assignment.FarmLaneSet = true
		assignment.ObjectiveID = 0
		assignment.Role = "lane_cover"
		assignment.Reason = "baseline_lane_coverage"
		assignment.Aggressive = false
		assignment.Coverage = true
		plan.Assignments[best.c.objID] = assignment
		owner[lane] = best.c.objID
		chosen[best.c.objID] = true
	}
	// Last resort: if every cover body is dead or recovering, temporarily
	// release the nearest base defender to the uncovered wave. A visible lane
	// with no proximity-XP body is a concrete loss that cannot be recovered;
	// the structure defender can return after the hand-off, while the wave
	// reward cannot. A live barracks threat is the exception: removing the only
	// defender would turn a temporary XP gap into an immediate structure loss.
	for lane := range activeLanes {
		if owner[lane] != 0 {
			continue
		}
		if plan.Mode == botMacroBase {
			objective := botPlanObjectiveLocked(inst, plan.Team, *plan)
			if objective != nil && objective.dotaRole != gamedata.DotaGun &&
				s.botDefenseStructureThreatSeverityLocked(inst, plan.Team, objective, now) > 0 {
				continue
			}
		}
		var best *botBrain
		bestDistance := math.Inf(1)
		for _, brain := range bots {
			if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.huntState.deadUntil > 0 ||
				brain.retreating || botHPFrac(brain.c.huntState, now) <= botSafeHPFrac || chosen[brain.c.objID] {
				continue
			}
			assignment := plan.Assignments[brain.c.objID]
			if assignment.Mode != botMacroBase || assignment.Role != "defender" || assignment.ObjectiveID == 0 {
				continue
			}
			distance := s.botFarmCoverageDistanceLocked(brain, lane, plan.Team, visionSources, now)
			if distance < bestDistance || (distance == bestDistance && (best == nil || brain.c.objID < best.c.objID)) {
				best, bestDistance = brain, distance
			}
		}
		if best == nil || bestDistance > botLaneApproachRadius*1.5 {
			continue
		}
		assignment := plan.Assignments[best.c.objID]
		assignment.Mode = botMacroCover
		assignment.Lane = lane
		assignment.FarmLane = lane
		assignment.FarmLaneSet = true
		assignment.ObjectiveID = 0
		assignment.Role = "lane_cover"
		assignment.Reason = "baseline_lane_coverage"
		assignment.Aggressive = false
		assignment.Coverage = true
		plan.Assignments[best.c.objID] = assignment
		owner[lane] = best.c.objID
		chosen[best.c.objID] = true
	}
	// Repair the last source of authored-line displacement. The passes above
	// may legitimately borrow a spare cover body for an uncovered lane, but a
	// previous FarmLane can make that spare look like the owner of another
	// authored line. Prefer the final lane_cover whose BaselineLane matches the
	// line, and move only that one owner back; extra same-line bodies remain
	// available as dynamic hand-offs. This keeps the stable XP contract without
	// preventing a defender or a genuinely missing baseline owner from being
	// replaced by the nearest safe spare.
	baselineCandidates := make(map[int][]*botBrain, len(inst.dota.m.Lanes))
	for _, brain := range bots {
		if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.huntState.deadUntil > 0 || brain.retreating {
			continue
		}
		assignment := plan.Assignments[brain.c.objID]
		if assignment.Role != "lane_cover" ||
			(assignment.Mode != botMacroLane && assignment.Mode != botMacroCover) ||
			assignment.BaselineLane < 0 || assignment.BaselineLane >= len(inst.dota.m.Lanes) {
			continue
		}
		baselineCandidates[assignment.BaselineLane] = append(baselineCandidates[assignment.BaselineLane], brain)
	}
	for lane, candidates := range baselineCandidates {
		if len(candidates) == 0 {
			continue
		}
		sort.Slice(candidates, func(i, j int) bool {
			left := plan.Assignments[candidates[i].c.objID]
			right := plan.Assignments[candidates[j].c.objID]
			leftAtBaseline := left.Lane == lane
			rightAtBaseline := right.Lane == lane
			if leftAtBaseline != rightAtBaseline {
				return leftAtBaseline
			}
			return candidates[i].c.objID < candidates[j].c.objID
		})
		ownerBrain := candidates[0]
		assignment := plan.Assignments[ownerBrain.c.objID]
		if assignment.FarmLaneSet && assignment.FarmLane != lane {
			// This baseline body is currently a deliberate handoff spare. Its
			// FarmLane is the uncovered route, so repairing the authored baseline
			// here would reintroduce the same oscillation on the next tick.
			continue
		}
		assignment.Lane = lane
		assignment.FarmLane = lane
		assignment.FarmLaneSet = true
		assignment.ObjectiveID = 0
		assignment.Reason = "baseline_lane_coverage"
		assignment.Aggressive = false
		assignment.Coverage = true
		plan.Assignments[ownerBrain.c.objID] = assignment
	}
}

// botAssignFarmReserveLocked keeps XP acquisition alive during an ordinary
// objective push. A push assignment normally wins over local farm selection,
// which made a low-XP bot follow the group into structure range, retreat, and
// miss several complete waves. Reserve only the most indebted living bot, and
// only after the live objective remains convertible without that body. Full
// mobilization is intentionally excluded: it is the explicit all-roster
// operation and must not be silently reduced to a partial muster.
func (s *Server) botAssignFarmReserveLocked(plan *botTeamPlan, bots []*botBrain, inst *huntInstance, now float64) bool {
	if plan == nil || inst == nil || inst.dota == nil ||
		(plan.Mode != botMacroPush && plan.Mode != botMacroRally) ||
		botAnyMobilizationReason(plan.Reason) ||
		!botAIProfileForPlanLocked(inst, plan).UsesFarmDebt() {
		return false
	}
	candidate := botMostIndebtedLivingBotLocked(inst, plan.Team)
	if candidate == nil || botFarmDebtLocked(inst, candidate) < 2 {
		return false
	}

	// Choose a lane with an actual visible wave. Prefer the bot's baseline lane
	// when it can produce XP now; otherwise use the strongest live wave so the
	// reservation is a real catch-up action rather than a cosmetic role change.
	bestLane, bestScore := -1, 0
	baselineScore := botFarmLaneWaveScoreAtLocked(inst, plan.Team, candidate.lane, now)
	if baselineScore > 0 {
		// A reserve is still the owner of its authored line. Moving it to a
		// richer neighboring wave would recreate the exact lane starvation this
		// overlay is meant to prevent.
		bestLane, bestScore = candidate.lane, baselineScore
	} else {
		for lane := range inst.dota.m.Lanes {
			score := botFarmLaneWaveScoreAtLocked(inst, plan.Team, lane, now)
			if score <= 0 {
				continue
			}
			if score > bestScore || (score == bestScore && (bestLane < 0 || lane < bestLane)) {
				bestLane, bestScore = lane, score
			}
		}
	}
	if bestLane < 0 {
		return false
	}

	objective := botPlanObjectiveLocked(inst, plan.Team, *plan)
	if objective != nil && (plan.Reason == "objective_conversion_ready" ||
		botMacroObjectiveCommitWindowLocked(objective)) &&
		!s.botObjectiveConversionReadyExcludingLocked(inst, plan.Team, objective, candidate.c.objID, now) {
		return false
	}
	// Do not remove the only strategic body from a push. For a staged route one
	// healthy responder is enough; for an active conversion keep a pair after
	// the reserve. This is based on the current assignment and live readiness,
	// not on the number of avatars in the match.
	strategic := 0
	for _, brain := range bots {
		if brain == nil || brain == candidate || brain.c == nil || brain.c.huntState == nil ||
			brain.c.huntState.deadUntil > 0 || brain.retreating || botHPFrac(brain.c.huntState, now) <= botRetreatHPFrac {
			continue
		}
		assignment := plan.Assignments[brain.c.objID]
		if assignment.Mode == botMacroPush || assignment.Mode == botMacroCover {
			strategic++
		}
	}
	minimumStrategic := 1
	if plan.Reason == "objective_conversion_ready" || botMacroObjectiveCommitWindowLocked(objective) {
		minimumStrategic = 2
	}
	if strategic < minimumStrategic {
		return false
	}

	assignment := plan.Assignments[candidate.c.objID]
	assignment.Mode = botMacroLane
	assignment.Lane = bestLane
	assignment.FarmLane = bestLane
	assignment.FarmLaneSet = true
	assignment.ObjectiveID = 0
	assignment.Role = "farm_reserve"
	assignment.Reason = "farm_debt_rescue"
	assignment.Aggressive = false
	assignment.Coverage = false
	plan.Assignments[candidate.c.objID] = assignment
	return true
}

// botTeamFarmRescueRequiredLocked prevents a profitable-looking push from
// consuming an available farm opportunity. It is driven by actual creep XP grants
// and live waves, never by elapsed match time. Gold is deliberately not part of
// this decision: the bot economy is provisioned, while XP is the scarce resource.
func (s *Server) botTeamFarmRescueRequiredLocked(inst *huntInstance, team int32) bool {
	if inst == nil || inst.dota == nil {
		return false
	}
	now := float64(s.battleTime())
	if _, critical := s.botCriticalEnemyObjectiveLocked(inst, team); critical != nil {
		// A damaged objective normally wins over farm debt. The exception is a
		// genuinely spare, debt-heavy bot: if the same live-state conversion
		// remains valid after removing that bot's power, let it catch the next
		// wave while the execution group keeps its objective commitment.
		var weakest *botBrain
		for _, brain := range inst.bots {
			if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.playerTeam() != team ||
				brain.c.huntState.deadUntil > 0 {
				continue
			}
			if botFarmPriorityLessLocked(brain, weakest) {
				weakest = brain
			}
		}
		if weakest != nil && botFarmDebtLocked(inst, weakest) >= 2 &&
			s.botObjectiveConversionReadyExcludingLocked(inst, team, critical, weakest.c.objID, now) {
			return true
		}
		return false
	}
	// A live lane wave anywhere on the map is enough to keep the first-XP
	// obligation active.  Visibility still governs which target a bot can
	// select, but it must not make the macro director conclude that the opening
	// farm is complete merely because the wave is outside this bot's vision.
	firstWaveLive := false
	for _, mob := range inst.mobs {
		if mob != nil && !mob.dead && !mob.structure && mob.enemyOf(team) && botTeleportLaneCreep(mob) {
			firstWaveLive = true
			break
		}
	}
	if firstWaveLive {
		for _, brain := range inst.bots {
			if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.playerTeam() != team ||
				brain.c.huntState.deadUntil > 0 || brain.retreating {
				continue
			}
			if brain.farmXPEvents == 0 {
				return true
			}
		}
	}
	// Before a bot has received any creep XP, the first live wave is a hard
	// coverage obligation.  Do not let a merely attractive objective staging
	// plan pull that bot off its authored farm line.  This is deliberately a
	// world-state predicate (live wave + missing XP), not a match-clock phase.
	for lane := range inst.dota.m.Lanes {
		if botFarmLaneWaveScoreAtLocked(inst, team, lane, now) <= 0 {
			continue
		}
		for _, brain := range inst.bots {
			if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.playerTeam() != team ||
				brain.c.huntState.deadUntil > 0 || brain.retreating {
				continue
			}
			if brain.farmXPEvents == 0 {
				return true
			}
		}
	}
	// Once a live wave is already in a locally favourable conversion state, do
	// not let a marginal farm-debt difference pull the committed group away from
	// the structure. This uses the objective's local power state, not the roster
	// size of either team.
	_, objective, _, _ := s.botBestMacroLaneLocked(inst, team, float64(s.battleTime()))
	if s.botObjectiveConversionReadyLocked(inst, team, objective, now) {
		return false
	}
	activeWave := false
	for lane := range inst.dota.m.Lanes {
		if botFarmLaneWaveScoreAtLocked(inst, team, lane, now) > 0 {
			activeWave = true
			break
		}
	}
	if !activeWave {
		return false
	}
	for _, brain := range inst.bots {
		if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.playerTeam() != team || brain.c.huntState.deadUntil > 0 || brain.retreating {
			continue
		}
		// One received XP event is not a clean bill of health. If another live
		// teammate has converted more of the same wave, this bot is already behind
		// and the team must keep its farm coverage instead of declaring the rescue
		// complete after a single accidental proximity grant.
		if botFarmDebtLocked(inst, brain) > 0 {
			return true
		}
	}
	return false
}

// botTeamFarmCoverageRequiredLocked is the hard XP-presence gate used by the
// macro planner. A lane is actionable when it has an own creep anchor or a
// visible enemy lane creep; it is covered only when a living teammate is within
// the actual proximity-XP radius of that lane's current units. This prevents a
// ready structure from overriding an uncovered wave while keeping the decision
// honest under fog of war.
func (s *Server) botTeamFarmCoverageRequiredLocked(inst *huntInstance, team int32, now float64) bool {
	if inst == nil || inst.dota == nil {
		return false
	}
	sources := dotaTeamVisionSourcesLocked(inst, team, now)
	laneMobs := make(map[int][]*mobState, len(inst.dota.m.Lanes))
	for _, mob := range botSortedMobs(inst) {
		if mob == nil || mob.dead || mob.structure || !botTeleportLaneCreep(mob) {
			continue
		}
		lane := botLaneForCreep(&conn{inst: inst}, mob)
		if lane < 0 || lane >= len(inst.dota.m.Lanes) {
			continue
		}
		// The authoritative XP reward is granted from visible enemy creeps. Own
		// creeps are useful for selecting a lane, but they must not block an
		// objective conversion in the synthetic gap before the opposing wave is
		// visible: that would turn a harmless allied anchor into a false farm debt.
		if mob.enemyOf(team) && botVisibleEnemyMobLocked(team, mob, sources) {
			laneMobs[lane] = append(laneMobs[lane], mob)
		}
	}
	for _, mobs := range laneMobs {
		covered := false
		for _, brain := range inst.bots {
			if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.playerTeam() != team ||
				brain.c.huntState.deadUntil > 0 || brain.retreating {
				continue
			}
			x, y := brain.c.posAtLocked(float32(now))
			for _, mob := range mobs {
				if math.Hypot(float64(x-mob.x), float64(y-mob.y)) <= dotaXPShareRadius {
					covered = true
					break
				}
			}
			if covered {
				break
			}
		}
		if !covered {
			return true
		}
	}
	return false
}

// botObjectiveStagingRequiredLocked keeps a live offensive route coherent when
// its next structure is already reachable but the exact wave/power conversion
// predicate is temporarily false. The group should stage near that objective,
// not reset every member to a farm lane and surrender the route's tempo.
func (s *Server) botObjectiveStagingRequiredLocked(inst *huntInstance, team int32, objective *mobState, previous botTeamPlan, hasPrevious bool, now float64) bool {
	if inst == nil || objective == nil || objective.dead || !objective.structure ||
		!objective.enemyOf(team) || (objective.altar && !inst.dota.altarVulnerableLocked(objective)) {
		return false
	}
	if hasPrevious && previous.Mode == botMacroAltar {
		return false
	}
	if hasPrevious && previous.Mode == botMacroPush &&
		previous.Lane == botNearestLaneToPointLocked(inst.dota, objective.x, objective.y) {
		// A push that just destroyed its front checkpoint may stage at the next
		// checkpoint even when that object is still full health. The staging point
		// is behind the objective and does not issue attacks; this preserves the
		// lane investment without parking bodies inside gun range or resetting
		// them to unrelated farm lanes while the next wave travels.
		return true
	}
	// A fresh full-health objective can still be worth staging for when the
	// allied wave is already on the same route and at least two live teammates
	// cover that lane. The group then waits outside gun range for the wave to
	// arrive, while the normal farm overlay keeps one lane owner active. This
	// removes the dead zone where the team keeps farming beside a ready wave but
	// never assembles early enough to convert it.
	if maxHP := objective.maxHealth(); maxHP <= 0 || objective.hp/maxHP > botMacroCommitDamageHPFrac {
		if !s.botObjectiveHasAlliedWaveLocked(inst, team, objective) {
			return false
		}
		lane := botNearestLaneToPointLocked(inst.dota, objective.x, objective.y)
		_, coverage := s.botMacroLaneProgressLocked(inst, team, lane, objective, now)
		return coverage >= 2
	}
	nearby := 0
	for _, mem := range inst.members {
		if mem == nil || mem.huntState == nil || mem.huntState.deadUntil > 0 || mem.playerTeam() != team {
			continue
		}
		x, y := mem.posAtLocked(float32(now))
		if math.Hypot(float64(x-objective.x), float64(y-objective.y)) <= botObjectiveApproachRadius {
			nearby++
		}
	}
	return nearby >= 2
}

// botReleaseObjectiveStagingRetreatsLocked lets a regrouped, recoverable pair
// rejoin a non-aggressive objective staging plan. The normal fountain latch is
// still authoritative for lone bots, critical HP, and active incoming pressure.
func (s *Server) botReleaseObjectiveStagingRetreatsLocked(plan *botTeamPlan, bots []*botBrain, inst *huntInstance, now float64) {
	if plan == nil || plan.Mode != botMacroPush || plan.Reason != botMacroReasonObjectiveStaging {
		return
	}
	for _, brain := range bots {
		if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.huntState.deadUntil > 0 || !brain.retreating {
			continue
		}
		assignment := botMacroAssignment{
			Mode: plan.Mode, Lane: plan.Lane, ObjectiveID: plan.ObjectiveID,
			Reason: plan.Reason, Aggressive: false,
		}
		if s.botCanRejoinObjectivePlanLocked(brain, assignment, now, botHPFrac(brain.c.huntState, now), false) {
			botClearRetreatLocked(brain)
		}
	}
}

// botReleaseObjectiveFinishRetreatsLocked reopens a recoverable responder when
// an ordinary conversion has already reduced its named structure to the
// execution-debt window. A bot that is still above the hard retreat floor
// should close that debt; waiting for the normal recovery latch at this point
// lets a nearly finished gun survive while the whole team walks home.
func (s *Server) botReleaseObjectiveFinishRetreatsLocked(plan *botTeamPlan, bots []*botBrain, inst *huntInstance, now float64) {
	if plan == nil || plan.Mode != botMacroPush || plan.Reason != "objective_conversion_ready" || inst == nil {
		return
	}
	objective := inst.mobs[plan.ObjectiveID]
	if !botMacroObjectiveCommitWindowLocked(objective) ||
		!botObjectiveFinishGroupReadyLocked(inst, plan.Team, objective, now) {
		return
	}
	for _, brain := range bots {
		if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.huntState.deadUntil > 0 || !brain.retreating {
			continue
		}
		if botHPFrac(brain.c.huntState, now) > botRetreatHPFrac && !s.botIncomingPressureLocked(brain, now) {
			botClearRetreatLocked(brain)
		}
	}
}

func botMobilizationReason(reason string) bool {
	return reason == botMacroReasonFullMobilization || reason == botMacroReasonMobilizationPreparation
}

func botPartialMobilizationReason(reason string) bool {
	return reason == botMacroReasonPartialMobilization
}

func botAnyMobilizationReason(reason string) bool {
	return botMobilizationReason(reason) || botPartialMobilizationReason(reason)
}

// botPartialMobilizationLimit keeps a strike group large enough to convert a
// live structure while reserving at least one body for farm/defense whenever
// the roster permits it. The size follows the current live roster; it is not a
// special-case for the 4v5 test.
func botPartialMobilizationLimit(roster int) int {
	if roster <= 1 {
		return roster
	}
	// Keep one live body available for farm/defense, but let the partial group
	// use every other healthy responder. A half-sized group was too fragile
	// after one death: it left a nearly destroyed checkpoint with only two
	// attackers and no way to close the final damage window.
	limit := roster - 1
	if limit < 2 && roster >= 2 {
		limit = 2
		if limit > roster {
			limit = roster
		}
	}
	return limit
}

const botMobilizationGatherRadius = 42.0

// A staging point must be outside the same no-dive envelope used by combat.
// The extra margin keeps a bot from oscillating on the boundary while the
// objective or its hitbox moves slightly during a tick.
const botObjectiveStagingClearance = botNoDiveRadius + botStructureAvoidMargin + 8.0

// botMobilizationRallyPointLocked returns a safe lane staging point immediately
// behind the named objective. It is deliberately derived from lane geometry,
// not from a match-clock waypoint: both sides can use the same rule and the
// group can advance to the next front structure after every destruction.
func botMobilizationRallyPointLocked(inst *huntInstance, team int32, objective *mobState) (float32, float32, bool) {
	if inst == nil || inst.dota == nil || objective == nil {
		return 0, 0, false
	}
	lane := botNearestLaneToPointLocked(inst.dota, objective.x, objective.y)
	if lane < 0 || lane >= len(inst.dota.m.Lanes) || len(inst.dota.m.Lanes[lane]) == 0 {
		return objective.x, objective.y, true
	}
	points := inst.dota.m.Lanes[lane]
	index, _, ok := botLaneStructureFrontIndex(points, objective.x, objective.y)
	if !ok {
		return objective.x, objective.y, true
	}
	direction := 1
	if team == dotaTeamHuman {
		direction = -1
	}
	// Walk away from the objective until the first authored point outside its
	// danger envelope. A fixed waypoint offset is not stable across lanes: on
	// one lane it placed the group in gun range, while on another it left them
	// needlessly far away. The first safe point also keeps the gather radius
	// small enough for the mobilization readiness gate to remain useful.
	selected := index
	for step := 1; step < len(points); step++ {
		candidate := index + direction*step
		if candidate < 0 || candidate >= len(points) {
			break
		}
		selected = candidate
		distance := math.Hypot(float64(float32(points[candidate].X)-objective.x),
			float64(float32(points[candidate].Y)-objective.y))
		if distance >= botObjectiveStagingClearance {
			break
		}
	}
	return float32(points[selected].X), float32(points[selected].Y), true
}

// botMobilizationReadyLocked is the only gate that turns a preparation group
// into an attack group. Every living teammate must be above the normal safe HP
// hysteresis, have a learned ultimate whose cooldown has elapsed, and be inside
// the shared staging radius. A dead member blocks launch until it respawns and
// rejoins; this is a full-roster mobilization, not a partial attack.
func (s *Server) botMobilizationReadyLocked(inst *huntInstance, team int32, objective *mobState, bots []*botBrain, now float64) bool {
	rx, ry, ok := botMobilizationRallyPointLocked(inst, team, objective)
	if !ok {
		return false
	}
	if !botMobilizationUltimatesReadyLocked(bots, now) {
		return false
	}
	live := 0
	for _, brain := range bots {
		if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.playerTeam() != team {
			continue
		}
		if brain.c.huntState.deadUntil > 0 {
			return false
		}
		live++
		if botHPFrac(brain.c.huntState, now) < botSafeHPFrac {
			return false
		}
		x, y := brain.c.posAtLocked(float32(now))
		if math.Hypot(float64(x-rx), float64(y-ry)) > botMobilizationGatherRadius {
			return false
		}
	}
	return live > 0
}

// botMobilizationUltimatesReadyLocked keeps the launch decision tied to the
// actual skill state. A rank-0 ultimate is not usable even though its cooldown
// slot is numerically zero; a learned ultimate whose cooldown ends exactly at
// the planning tick is ready. A dead member blocks the full-roster launch until
// it has respawned.
func botMobilizationUltimatesReadyLocked(bots []*botBrain, now float64) bool {
	live := 0
	for _, brain := range bots {
		if brain == nil || brain.c == nil || brain.c.huntState == nil {
			continue
		}
		if brain.c.huntState.deadUntil > 0 {
			return false
		}
		live++
		hs := brain.c.huntState
		if hs.skillLevel[3] < 1 || now < hs.cooldownUntil[3] {
			return false
		}
	}
	return live > 0
}

// botMobilizationUltimatesLearnedLocked distinguishes an attainable launch
// gate from an impossible one. A rank-0 ultimate cannot become ready by
// waiting for cooldown; the team must keep farming until every bot has reached
// the authored ultimate level gate first.
func botMobilizationUltimatesLearnedLocked(bots []*botBrain) bool {
	roster := 0
	for _, brain := range bots {
		if brain == nil || brain.c == nil || brain.c.huntState == nil {
			continue
		}
		roster++
		if brain.c.huntState.skillLevel[3] < 1 {
			return false
		}
	}
	return roster > 0
}

// botReleaseMobilizationRetreatsLocked clears only stale retreat latches that
// are compatible with the current preparation phase. Critical HP still goes to
// recovery; a fully healed bot is allowed to leave the fountain and assemble.
func (s *Server) botReleaseMobilizationRetreatsLocked(plan *botTeamPlan, bots []*botBrain, inst *huntInstance, now float64) {
	if plan == nil || plan.Mode != botMacroPush || !botAnyMobilizationReason(plan.Reason) {
		return
	}
	for _, brain := range bots {
		if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.huntState.deadUntil > 0 || !brain.retreating {
			continue
		}
		frac := botHPFrac(brain.c.huntState, now)
		if plan.Reason == botMacroReasonMobilizationPreparation {
			if frac >= botSafeHPFrac {
				botClearRetreatLocked(brain)
			}
			continue
		}
		if frac > botRetreatHPFrac {
			botClearRetreatLocked(brain)
		}
	}
}

// botAssignMobilizationPreparationLocked makes preparation explicit. Healthy
// bots receive the exact same staging objective and no aggressive flag; bots
// below the safe HP band are sent to recovery instead of being dragged into the
// first wave one by one.
func (s *Server) botAssignMobilizationPreparationLocked(plan *botTeamPlan, bots []*botBrain, inst *huntInstance, now float64) {
	if plan == nil || plan.Mode != botMacroPush || plan.Reason != botMacroReasonMobilizationPreparation {
		return
	}
	for _, brain := range bots {
		if brain == nil || brain.c == nil || brain.c.huntState == nil {
			continue
		}
		id := brain.c.objID
		if brain.c.huntState.deadUntil > 0 {
			continue
		}
		if botHPFrac(brain.c.huntState, now) < botSafeHPFrac {
			botSetRetreatModeLocked(brain, botRetreatModeRecovery, now)
			plan.Assignments[id] = botMacroAssignment{
				Mode: botMacroRecover, Lane: brain.lane, BaselineLane: brain.lane,
				ObjectiveID: plan.ObjectiveID, Role: "mobilization_recovery",
				Reason: botMacroReasonMobilizationPreparation,
			}
			continue
		}
		plan.Assignments[id] = botMacroAssignment{
			Mode: botMacroPush, Lane: plan.Lane, BaselineLane: brain.lane,
			ObjectiveID: plan.ObjectiveID, Role: "muster", Reason: botMacroReasonMobilizationPreparation,
			Aggressive: false, Coverage: true,
		}
	}
}

// botCriticalEnemyObjectiveLocked finds a damaged front objective that can be
// finished without walking through another live structure. Historical damage
// is useful here: it is a direct signal that this lane has already invested
// time and should be completed before a farm-debt rescue pulls the group away.
func (s *Server) botCriticalEnemyObjectiveLocked(inst *huntInstance, team int32) (int, *mobState) {
	if inst == nil || inst.dota == nil {
		return -1, nil
	}
	bestLane := -1
	var best *mobState
	bestFrac := math.Inf(1)
	for lane := range inst.dota.m.Lanes {
		front, _ := s.botMacroLaneObjectiveLocked(inst, team, lane)
		if front == nil || front.altar || !botMacroObjectiveDamaged(front) {
			continue
		}
		maxHP := front.maxHealth()
		if maxHP <= 0 {
			continue
		}
		frac := front.hp / maxHP
		// A small chip is normal lane wear. The commit band is reserved for
		// an objective the team has materially invested in finishing; waiting
		// until the old 70% execution band made the bots abandon a 20%-damaged
		// gun for farm and lose the conversion window.
		if frac > botMacroCommitDamageHPFrac {
			continue
		}
		if best == nil || frac < bestFrac || (frac == bestFrac && (lane < bestLane || (lane == bestLane && front.id < best.id))) {
			bestLane, best, bestFrac = lane, front, frac
		}
	}
	return bestLane, best
}

// botAssignFarmLanesLocked is the single owner of farm-lane allocation. The
// local bot code never invents a lane: it only executes FarmLane from this
// assignment. Farm-eligible bots are distributed over the least occupied live
// wave lane with a deterministic baseline-lane tie-break. Base defenders are
// included as well: their objective remains authoritative, but when a safe wave
// is close they now have an explicit farm rotation instead of all sharing the
// objective lane.
func (s *Server) botAssignFarmLanesLocked(plan *botTeamPlan, bots []*botBrain, inst *huntInstance) {
	s.botAssignFarmLanesWithPreviousAtLocked(plan, botTeamPlan{}, false, bots, inst, float64(s.battleTime()))
}

func (s *Server) botAssignFarmLanesWithPreviousLocked(plan *botTeamPlan, previous botTeamPlan, hasPrevious bool, bots []*botBrain, inst *huntInstance) {
	s.botAssignFarmLanesWithPreviousAtLocked(plan, previous, hasPrevious, bots, inst, float64(s.battleTime()))
}

func (s *Server) botAssignFarmLanesWithPreviousAtLocked(plan *botTeamPlan, previous botTeamPlan, hasPrevious bool, bots []*botBrain, inst *huntInstance, now float64) {
	if plan == nil || inst == nil || inst.dota == nil || len(inst.dota.m.Lanes) == 0 {
		return
	}
	eligible := make([]*botBrain, 0, len(bots))
	for _, brain := range bots {
		if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.huntState.deadUntil > 0 || brain.retreating {
			continue
		}
		assignment := plan.Assignments[brain.c.objID]
		if assignment.Mode != botMacroLane && assignment.Mode != botMacroCover && assignment.Mode != botMacroBase {
			continue
		}
		eligible = append(eligible, brain)
	}
	if len(eligible) == 0 {
		return
	}
	version := botPlanAIVersionLocked(inst, plan)
	if !botAIProfileForVersion(version).UsesFarmLanePlan() {
		// AI-10 keeps the authored 2/1/2 lane identity and has no live-wave
		// rotation. Strategic assignments remain authoritative; this only
		// supplies the baseline farm lane consumed by local movement.
		for _, brain := range eligible {
			assignment := plan.Assignments[brain.c.objID]
			assignment.FarmLane = brain.lane
			assignment.FarmLaneSet = brain.lane >= 0 && brain.lane < len(inst.dota.m.Lanes)
			plan.Assignments[brain.c.objID] = assignment
		}
		return
	}
	if !botAIProfileForVersion(version).UsesFarmStability() {
		// AI-18 still allocates farm lanes, but AI-19 introduced stable live-wave
		// handoffs across planner recomputations.
		hasPrevious = false
	}
	weakest := botWeakestEligibleFarmBotLocked(inst, plan.Team, eligible)
	occupancy := make([]int, len(inst.dota.m.Lanes))
	activeWaveLanes := make([]bool, len(occupancy))
	ownWaveLanes := make([]bool, len(occupancy))
	anyActiveWave := false
	for lane := range occupancy {
		activeWaveLanes[lane] = botFarmLaneWaveScoreAtLocked(inst, plan.Team, lane, now) > 0
		anyActiveWave = anyActiveWave || activeWaveLanes[lane]
	}
	// Enemy-wave score is intentionally vision-bounded. An invisible enemy
	// wave must not become a target, but an authored own wave is still proof
	// that this line is live and needs a baseline owner. Use that safe anchor
	// when deciding whether the weakest bot may leave its authored lane.
	for _, mob := range botSortedMobs(inst) {
		if mob == nil || mob.dead || mob.structure || mob.team != plan.Team ||
			!botTeleportLaneCreep(mob) {
			continue
		}
		lane := botLaneForCreep(&conn{inst: inst}, mob)
		if lane >= 0 && lane < len(ownWaveLanes) {
			ownWaveLanes[lane] = true
		}
	}
	// Keep a valid farm lane across ordinary macro recomputation. The old
	// allocator re-solved from scratch every planner tick, so a wave-count tie
	// made the same bot bounce 0->1->2 even while the strategic objective was
	// unchanged. Reassign only when the old lane has no live wave and another
	// lane can actually convert into XP; if the map is temporarily empty, keep
	// the last lane and wait for the next wave.
	retained := make(map[int32]bool, len(eligible))
	retainedLane := make(map[int]bool, len(occupancy))
	// Baseline lane ownership is a safety invariant. The live-wave score is
	// visibility-bounded and can be zero before a wave reaches the current
	// vision envelope; using that score as the only allocator input would stack
	// every cover bot on the one visible lane and leave the other lines blind.
	// Reserve one eligible owner for each authored baseline lane first.
	distinctBaselines := make(map[int]bool, len(eligible))
	for _, brain := range eligible {
		if brain.lane >= 0 && brain.lane < len(occupancy) {
			distinctBaselines[brain.lane] = true
		}
	}
	if len(distinctBaselines) > 1 {
		// Real matches have authored lane ownership (the normal 2/1/2 split).
		// Keep that ownership stable and use a spare only to replace a baseline
		// bot that is currently unavailable. Re-solving all eligible bodies from
		// visible wave scores made a healthy spare bounce between lines every
		// planner tick, so it arrived at neither line in time for XP.
		baselineOwners := make(map[int]int, len(distinctBaselines))
		needed := make(map[int]bool, len(occupancy))
		for lane := range occupancy {
			needed[lane] = activeWaveLanes[lane] || ownWaveLanes[lane] ||
				botLaneHasLivingEnemyBarracksLocked(inst, plan.Team, lane)
		}
		for _, brain := range eligible {
			baseline := brain.lane
			if baseline >= 0 && baseline < len(occupancy) && needed[baseline] &&
				!s.botFarmOwnerUnavailableLocked(brain, baseline, now) {
				baselineOwners[baseline]++
			}
		}
		missing := make(map[int]bool, len(occupancy))
		for lane := range occupancy {
			if needed[lane] && baselineOwners[lane] == 0 {
				missing[lane] = true
			}
		}
		assigned := make(map[int32]bool, len(eligible))
		// A materially farm-indebted bot may catch up on a live wave when its
		// authored line has no current own or visible enemy creep. This is the
		// only normal exception to stable ownership; it is driven by live wave
		// presence and debt, not by elapsed match time.
		if weakest != nil {
			baseline := weakest.lane
			baselineLive := baseline >= 0 && baseline < len(occupancy) &&
				(activeWaveLanes[baseline] || ownWaveLanes[baseline])
			if !baselineLive {
				bestLane, bestScore := -1, 0
				for lane := range occupancy {
					score := botFarmLaneWaveScoreAtLocked(inst, plan.Team, lane, now)
					if score > bestScore || (score == bestScore && score > 0 && (bestLane < 0 || lane == baseline)) {
						bestLane, bestScore = lane, score
					}
				}
				if bestLane >= 0 {
					assignment := plan.Assignments[weakest.c.objID]
					assignment.FarmLane = bestLane
					assignment.FarmLaneSet = true
					plan.Assignments[weakest.c.objID] = assignment
					assigned[weakest.c.objID] = true
				}
			}
		}
		// Preserve an existing handoff while the original baseline owner is
		// unavailable. This is the spatially stable replacement for a retreating
		// lane owner; it is released automatically when the baseline lane has a
		// living eligible owner again.
		if hasPrevious {
			for _, brain := range eligible {
				old, ok := previous.Assignments[brain.c.objID]
				if !ok || !old.FarmLaneSet || !missing[old.FarmLane] {
					continue
				}
				assignment := plan.Assignments[brain.c.objID]
				assignment.FarmLane = old.FarmLane
				assignment.FarmLaneSet = true
				plan.Assignments[brain.c.objID] = assignment
				assigned[brain.c.objID] = true
				missing[old.FarmLane] = false
			}
		}
		// A duplicate-baseline spare that has already been handed to a live
		// line keeps that handoff until the line loses its future wave source.
		// Reverting it as soon as the original owner briefly becomes eligible
		// creates a route flapping loop: the spare turns around before reaching
		// the lane, the owner retreats again, and the next creep dies uncovered.
		// The barracks/active-wave check is the state-based release condition;
		// no elapsed-match exception is involved.
		if hasPrevious {
			for _, brain := range eligible {
				if assigned[brain.c.objID] {
					continue
				}
				old, ok := previous.Assignments[brain.c.objID]
				if !ok || !old.FarmLaneSet || old.FarmLane < 0 || old.FarmLane >= len(occupancy) || old.FarmLane == brain.lane {
					continue
				}
				if !activeWaveLanes[old.FarmLane] && !ownWaveLanes[old.FarmLane] &&
					!botLaneHasLivingEnemyBarracksLocked(inst, plan.Team, old.FarmLane) {
					continue
				}
				assignment := plan.Assignments[brain.c.objID]
				assignment.FarmLane = old.FarmLane
				assignment.FarmLaneSet = true
				plan.Assignments[brain.c.objID] = assignment
				assigned[brain.c.objID] = true
			}
		}
		// Every remaining eligible bot stays on its authored lane. A duplicate
		// baseline is intentionally retained as the spare pool for the handoff.
		for _, brain := range eligible {
			if assigned[brain.c.objID] {
				continue
			}
			assignment := plan.Assignments[brain.c.objID]
			assignment.FarmLane = brain.lane
			assignment.FarmLaneSet = brain.lane >= 0 && brain.lane < len(occupancy)
			plan.Assignments[brain.c.objID] = assignment
			assigned[brain.c.objID] = true
		}
		// Fill still-missing lines from duplicate-baseline spares. If several
		// lines are simultaneously uncovered, prefer the one with a visible wave
		// and then the closest available body; all choices remain state-based.
		for lane := range occupancy {
			if !missing[lane] {
				continue
			}
			var best *botBrain
			bestDistance := math.Inf(1)
			for _, brain := range eligible {
				if brain == nil || !assigned[brain.c.objID] {
					continue
				}
				assignment := plan.Assignments[brain.c.objID]
				if assignment.FarmLane != brain.lane || baselineOwners[brain.lane] < 2 {
					continue
				}
				distance := s.botFarmCoverageDistanceLocked(brain, lane, plan.Team, dotaTeamVisionSourcesLocked(inst, plan.Team, now), now)
				if distance < bestDistance || (distance == bestDistance && (best == nil || brain.c.objID < best.c.objID)) {
					best, bestDistance = brain, distance
				}
			}
			if best == nil {
				continue
			}
			assignment := plan.Assignments[best.c.objID]
			assignment.FarmLane = lane
			assignment.FarmLaneSet = true
			plan.Assignments[best.c.objID] = assignment
			missing[lane] = false
		}
		return
	}
	if len(distinctBaselines) > 1 {
		for _, brain := range eligible {
			baseline := brain.lane
			// A weakest bot may still be redirected to a different live wave when
			// its authored lane has no current wave (the catch-up behavior covered
			// by the farm-debt tests). Once that authored lane itself has a live
			// wave, however, moving its only baseline owner creates an uncovered
			// line after a death/redeploy hand-off.
			if anyActiveWave && brain == weakest && baseline >= 0 && baseline < len(activeWaveLanes) &&
				!activeWaveLanes[baseline] && !ownWaveLanes[baseline] {
				continue
			}
			if baseline < 0 || baseline >= len(occupancy) || retainedLane[baseline] {
				continue
			}
			assignment := plan.Assignments[brain.c.objID]
			assignment.FarmLane = baseline
			assignment.FarmLaneSet = true
			plan.Assignments[brain.c.objID] = assignment
			occupancy[baseline]++
			retained[brain.c.objID] = true
			retainedLane[baseline] = true
		}
	}
	if hasPrevious {
		for _, brain := range eligible {
			if retained[brain.c.objID] {
				continue
			}
			old, ok := previous.Assignments[brain.c.objID]
			if !ok || !old.FarmLaneSet || old.FarmLane < 0 || old.FarmLane >= len(occupancy) {
				continue
			}
			// Retain a previous owner while its lane has a live wave source. The
			// barracks is the state-based persistence signal: a brief visibility
			// gap must not make a reserve bot bounce 0->2->1->0 before it reaches
			// the next lane. A second owner is allowed on that persistent lane;
			// it is cheaper than abandoning the bot's current route and can be
			// borrowed only after a real lane hand-off is required.
			lanePersistent := botLaneHasLivingEnemyBarracksLocked(inst, plan.Team, old.FarmLane)
			canRetain := !anyActiveWave || activeWaveLanes[old.FarmLane] || lanePersistent
			if canRetain && retainedLane[old.FarmLane] && !lanePersistent {
				canRetain = false
			}
			if canRetain {
				assignment := plan.Assignments[brain.c.objID]
				assignment.FarmLane = old.FarmLane
				assignment.FarmLaneSet = true
				plan.Assignments[brain.c.objID] = assignment
				occupancy[old.FarmLane]++
				retained[brain.c.objID] = true
				retainedLane[old.FarmLane] = true
			}
		}
	}
	for _, brain := range eligible {
		if retained[brain.c.objID] {
			continue
		}
		assignment := plan.Assignments[brain.c.objID]
		baseline := assignment.BaselineLane
		if baseline < 0 || baseline >= len(occupancy) {
			baseline = brain.lane
		}
		// The weakest eligible bot gets first access to a lane that currently
		// contains live enemy creeps, but an unowned active lane always wins over
		// a larger wave score. Catch-up must not strand an entire lane: one bot
		// being behind is not permission to make another lane owner disappear.
		if brain == weakest {
			chosen := -1
			bestWaveScore := 0
			for lane := range occupancy {
				waveScore := botFarmLaneWaveScoreAtLocked(inst, plan.Team, lane, now)
				if waveScore == 0 {
					continue
				}
				if chosen < 0 ||
					(occupancy[lane] == 0 && occupancy[chosen] != 0) ||
					(occupancy[lane] == occupancy[chosen] &&
						(waveScore > bestWaveScore ||
							(waveScore == bestWaveScore && lane == baseline && chosen != baseline) ||
							(waveScore == bestWaveScore && lane != baseline && chosen != baseline && lane < chosen))) {
					chosen, bestWaveScore = lane, waveScore
				}
			}
			if chosen >= 0 {
				occupancy[chosen]++
				assignment.FarmLane = chosen
				assignment.FarmLaneSet = true
				plan.Assignments[brain.c.objID] = assignment
				continue
			}
		}

		// When a live wave exists, prefer the least occupied lane that can
		// convert now. This keeps the farm roster on active lanes without
		// stacking every defender on the same objective wave.
		candidateLanes := make([]int, 0, len(occupancy))
		for lane := range occupancy {
			if botFarmLaneWaveScoreAtLocked(inst, plan.Team, lane, now) > 0 {
				candidateLanes = append(candidateLanes, lane)
			}
		}
		if len(candidateLanes) == 0 {
			for lane := range occupancy {
				candidateLanes = append(candidateLanes, lane)
			}
		}
		minOccupancy := occupancy[candidateLanes[0]]
		for _, lane := range candidateLanes[1:] {
			if occupancy[lane] < minOccupancy {
				minOccupancy = occupancy[lane]
			}
		}
		chosen := -1
		for _, lane := range candidateLanes {
			if occupancy[lane] != minOccupancy {
				continue
			}
			if chosen < 0 || (lane == baseline && chosen != baseline) ||
				(lane != baseline && chosen != baseline && lane < chosen) {
				chosen = lane
			}
		}
		occupancy[chosen]++
		assignment.FarmLane = chosen
		assignment.FarmLaneSet = true
		plan.Assignments[brain.c.objID] = assignment
	}
}

// botWeakestEligibleFarmBotLocked keeps objective assignments authoritative
// while still ensuring the weakest bot among the actual farm roster gets the
// first live-wave opportunity. A globally weakest bot may be an assault/altar
// responder and must not steal that objective's slot.
func botWeakestEligibleFarmBotLocked(inst *huntInstance, team int32, eligible []*botBrain) *botBrain {
	if inst == nil {
		return nil
	}
	var weakest *botBrain
	for _, b := range eligible {
		if b == nil || b.c == nil || b.c.huntState == nil || b.c.playerTeam() != team || b.c.huntState.deadUntil > 0 || b.retreating {
			continue
		}
		if botFarmPriorityLessLocked(b, weakest) {
			weakest = b
		}
	}
	return weakest
}

func botPlanStrategicResponder(mode string, a botMacroAssignment) bool {
	if a.Role == "cover" {
		return mode == botMacroBase || mode == botMacroAltar || mode == botMacroPush || mode == botMacroRally
	}
	return a.Mode == mode
}

func botPlanResponderRank(mode string, a botMacroAssignment) int {
	if botPlanStrategicResponder(mode, a) && a.Role == "cover" {
		return 1
	}
	if botPlanStrategicResponder(mode, a) {
		return 0
	}
	return 2
}

func botPlanObjectiveLocked(inst *huntInstance, team int32, plan botTeamPlan) *mobState {
	if plan.ObjectiveID != 0 {
		if m := inst.mobs[plan.ObjectiveID]; m != nil && !m.dead {
			return m
		}
	}
	if plan.Mode == botMacroBase {
		return botTeamAltarLocked(inst, team)
	}
	if plan.Mode == botMacroAltar {
		return botTeamAltarLocked(inst, otherDotaTeam(team))
	}
	return nil
}

func (s *Server) botPlanPremiseValidLocked(inst *huntInstance, team int32, plan botTeamPlan, now float64) bool {
	// A destroyed/removed objective invalidates the old premise immediately.
	// Without this guard a still-live lane front could retain a dead barracks id
	// for one or more planning passes after the structure state changed.
	if plan.ObjectiveID != 0 {
		objective := inst.mobs[plan.ObjectiveID]
		if objective == nil || objective.dead {
			return false
		}
	}
	switch plan.Mode {
	case botMacroBase:
		objective := botPlanObjectiveLocked(inst, team, plan)
		return objective != nil && !objective.dead &&
			s.botDefenseStructureThreatSeverityLocked(inst, team, objective, now) > 0
	case botMacroAltar:
		objective := botPlanObjectiveLocked(inst, team, plan)
		return objective != nil && !objective.dead && inst.dota.altarVulnerableLocked(objective)
	case botMacroPush, botMacroRally, botMacroLane:
		if plan.Lane < 0 || plan.Lane >= len(inst.dota.m.Lanes) {
			return false
		}
		objective := botPlanObjectiveLocked(inst, team, plan)
		progress, coverage := s.botMacroLaneProgressLocked(inst, team, plan.Lane, objective, now)
		if plan.Mode == botMacroPush {
			// One covered responder is enough to keep an established push
			// sticky; losing the last live coverage is a material change.
			return progress && coverage >= 1
		}
		return progress
	default:
		return false
	}
}

func (s *Server) botPlanScoreLocked(inst *huntInstance, team int32, plan botTeamPlan, now float64) float64 {
	objectiveScore := 0.0
	if objective := botPlanObjectiveLocked(inst, team, plan); objective != nil {
		switch objective.dotaRole {
		case gamedata.DotaCreepTower:
			objectiveScore = 100
			if max := objective.maxHealth(); max > 0 {
				objectiveScore += (1 - objective.hp/max) * 35
			}
		case gamedata.DotaGun:
			objectiveScore = 35
		case gamedata.DotaGenerator:
			objectiveScore = 15
		}
	}
	switch plan.Mode {
	case botMacroBase:
		severity := s.botBaseDefensePressureSeverityLocked(inst, team, now)
		if objective := botPlanObjectiveLocked(inst, team, plan); objective != nil {
			severity = s.botDefenseStructureThreatSeverityLocked(inst, team, objective, now)
		}
		return 1000 + float64(severity)*100 + objectiveScore
	case botMacroAltar:
		return 900 + objectiveScore
	case botMacroPush:
		return 300 + objectiveScore
	case botMacroRally:
		return 220 + objectiveScore
	case botMacroLane:
		return 100 + objectiveScore
	default:
		return 0
	}
}

func (s *Server) botBaseDefensePressureSeverityLocked(inst *huntInstance, team int32, now float64) int {
	severity := s.botTeamBasePressureSeverityLocked(inst, team, now)
	for _, structure := range botSortedMobs(inst) {
		if structure.team != team || !structure.structure {
			continue
		}
		if structureSeverity := s.botDefenseStructureThreatSeverityLocked(inst, team, structure, now); structureSeverity > severity {
			severity = structureSeverity
		}
	}
	return severity
}

// botRetainTeamPlanLocked applies hysteresis to a live plan. Urgent base/altar
// premises still win immediately; ordinary lane/mode changes need a material
// score advantage or the current objective remains authoritative.
func (s *Server) botRetainTeamPlanLocked(inst *huntInstance, team int32, previous, desired botTeamPlan, now float64) bool {
	if !s.botPlanPremiseValidLocked(inst, team, previous, now) {
		return false
	}
	// Mobilization is an offensive launch phase. It must never survive a mode
	// transition into base/altar defense through the generic score hysteresis:
	// that used to leave a defending team with "muster" assignments while its
	// own altar was under attack. A live push can remain sticky; an urgent
	// defensive mode must rebuild its responders from the defensive premise.
	if botAnyMobilizationReason(previous.Reason) && desired.Mode != botMacroPush {
		return false
	}
	if previous.Mode == botMacroBase {
		previousObjective := botPlanObjectiveLocked(inst, team, previous)
		if previousObjective != nil && previousObjective.dotaRole == gamedata.DotaGun && desired.Mode != botMacroBase &&
			s.botObjectiveConversionReadyLocked(inst, team, botPlanObjectiveLocked(inst, team, desired), now) {
			// Do not let base-plan hysteresis re-install a gun defense after
			// a live local wave/group has earned an offensive conversion.
			return false
		}
	}
	if (previous.Mode == botMacroPush || previous.Mode == botMacroRally) && s.botTeamFarmRescueRequiredLocked(inst, team) {
		// Once a push has damaged its named structure, farm debt must not
		// cancel the conversion. That exact hand-off previously made the
		// five-bot side abandon a barracks at 29% HP and miss the 30-minute
		// finish by returning to lane mode for a marginal XP discrepancy.
		objective := botPlanObjectiveLocked(inst, team, previous)
		if objective != nil && botMacroObjectiveDamaged(objective) {
			return true
		}
		return false
	}
	if desired.Mode == botMacroBase && previous.Mode != botMacroBase {
		return false
	}
	if desired.Mode == botMacroAltar && previous.Mode != botMacroAltar {
		return false
	}
	if previous.Mode == desired.Mode && previous.Lane == desired.Lane && previous.ObjectiveID == desired.ObjectiveID {
		return true
	}
	if previous.Mode == botMacroBase && desired.Mode == botMacroBase && previous.ObjectiveID != desired.ObjectiveID {
		previousObjective := botPlanObjectiveLocked(inst, team, previous)
		desiredObjective := botPlanObjectiveLocked(inst, team, desired)
		if previousObjective != nil && desiredObjective != nil &&
			previousObjective.dotaRole == gamedata.DotaCreepTower && desiredObjective.dotaRole == gamedata.DotaCreepTower {
			previousSeverity := botBarracksThreatSeverityLocked(inst, team, previousObjective, now)
			desiredSeverity := botBarracksThreatSeverityLocked(inst, team, desiredObjective, now)
			if desiredSeverity <= previousSeverity+botMacroBarracksSwitchDelta {
				return true
			}
		}
	}
	return s.botPlanScoreLocked(inst, team, desired, now) <
		s.botPlanScoreLocked(inst, team, previous, now)+botMacroSwitchMargin
}

func (s *Server) botAssignPlanRespondersLocked(plan *botTeamPlan, previous botTeamPlan, hasPrevious bool, bots []*botBrain, inst *huntInstance, team int32, now float64) bool {
	if plan.Mode != botMacroBase && plan.Mode != botMacroAltar && plan.Mode != botMacroPush && plan.Mode != botMacroRally {
		return false
	}
	x, y := botMacroObjectivePointLocked(inst, team, plan.Lane, botPlanObjectiveLocked(inst, team, *plan))
	limit, minimum := 1, 1
	counterLane, counterObjective := -1, (*mobState)(nil)
	baseCounterPush := false
	switch plan.Mode {
	case botMacroBase:
		pressure := s.botBaseDefensePressureSeverityLocked(inst, team, now)
		objective := botPlanObjectiveLocked(inst, team, *plan)
		if objective != nil && objective.dotaRole == gamedata.DotaCreepTower {
			// Barracks defense is intentionally sparse: one bot stays committed,
			// while the rest keep farming their assigned lanes. Only a dense live
			// wave or hero-level pressure earns reinforcements.
			limit = 1
			if pressure >= botMacroBarracksReinforceSeverity {
				limit = 2
			}
			if pressure >= botMacroBarracksCriticalSeverity {
				limit = 3
			}
		} else {
			limit = 1
			if pressure >= 3 {
				limit = 2
			}
			if pressure >= 6 {
				limit = 3
			}
		}
		var counterProgress bool
		counterLane, counterObjective, counterProgress, _ = s.botBestMacroLaneLocked(inst, team, now)
		healthyResponders := s.botHealthyMacroResponderCountLocked(bots, now)
		if criticalLane, criticalObjective := s.botCriticalEnemyObjectiveLocked(inst, team); criticalObjective != nil {
			criticalProgress, criticalCoverage := s.botMacroLaneProgressLocked(inst, team, criticalLane, criticalObjective, now)
			criticalCommit := botMacroObjectiveCommitWindowLocked(criticalObjective)
			if criticalProgress || (criticalCommit && criticalCoverage > 0 && healthyResponders >= 2) {
				// A nearly dead enemy front object is a concrete conversion debt.
				// During gun defense, the ordinary lane score can still prefer a
				// healthy neighboring gun; that wastes the exact counter window that
				// lets a five-bot side trade one checkpoint for another.
				counterLane, counterObjective, counterProgress = criticalLane, criticalObjective, true
			} else {
				// Without a local responder pair, historical damage is not enough
				// to manufacture an offensive assignment. A live commit window plus
				// two healthy responders is the smallest safe trade: one stays on
				// the threatened gun and one closes the damaged enemy objective.
			}
		}
		ownObjective := botPlanObjectiveLocked(inst, team, *plan)
		counterFinish := botMacroObjectiveCommitWindowLocked(counterObjective)
		gunCanTrade := ownObjective != nil && ownObjective.dotaRole == gamedata.DotaGun && counterFinish
		// A damaged enemy gun is still worth trading against when our own
		// barracks is the threatened checkpoint. Keep one defender on the local
		// structure and let the remaining healthy body close the enemy gun; this
		// is the defensive form of partial mobilization and avoids losing a nearly
		// won objective while the whole roster sits under a barracks alert.
		barracksCanTrade := ownObjective != nil && ownObjective.dotaRole == gamedata.DotaCreepTower && counterFinish
		// A counter-push is an exchange, not a second lane assignment.  A wave
		// merely being present at a full-health enemy gun is not enough to spend
		// bodies while our own structure is under pressure: the old predicate sent
		// two healthy bots into the enemy gun, then lost one before either side had
		// created a real objective trade. Require objective debt, a live conversion
		// margin, or a safe wave staging window before spending bodies.
		counterObjectiveCommitted := counterObjective != nil &&
			botMacroObjectiveCommitWindowLocked(counterObjective)
		counterObjectiveReady := counterObjective != nil && counterProgress &&
			s.botObjectiveConversionReadyLocked(inst, team, counterObjective, now)
		counterStagingSafe := counterProgress && counterObjective != nil &&
			!counterObjective.altar &&
			!s.botVisibleEnemyHeroNearObjectiveLocked(inst, team, counterObjective, now)
		baseCounterPush = pressure < botMacroBaseSeverePressure && counterProgress &&
			healthyResponders >= 3 &&
			(counterObjectiveCommitted || counterObjectiveReady || counterStagingSafe)
		if !baseCounterPush && (gunCanTrade || barracksCanTrade) && counterProgress {
			// A gun is a checkpoint, not the win condition. When the opponent's
			// front objective is already materially damaged, keep one defender
			// and spend the remaining local power on the finish. Two healthy
			// responders are enough for the smallest legal trade; three or more
			// retain the existing two-body defensive shell.
			baseCounterPush = healthyResponders >= 2
		}
		if baseCounterPush {
			// Select the two closest safe defenders first. The counter-pusher is
			// selected from the remaining healthy roster below, so a defender is
			// never pulled away from the altar merely because it was selected as
			// the third responder.
			limit, minimum = 2, 2
			if (gunCanTrade || barracksCanTrade) && healthyResponders == 2 {
				// Leave one body on the threatened gun and use the other as the
				// counter-pusher. Selecting two defenders first would consume the
				// entire healthy roster before botAssignBaseCounterPushLocked runs.
				limit, minimum = 1, 1
			}
		}
	case botMacroAltar:
		limit = len(bots)
		minimum = 2
		objective := botPlanObjectiveLocked(inst, team, *plan)
		if !s.botObjectiveConversionReadyLocked(inst, team, objective, now) && limit > 3 {
			limit = 3
		}
	case botMacroPush:
		limit, minimum = 2, 2
		objective := botPlanObjectiveLocked(inst, team, *plan)
		conversionReady := s.botObjectiveConversionReadyLocked(inst, team, objective, now)
		finishWindow := botMacroObjectiveDamaged(objective)
		fullMobilization := plan.Reason == botMacroReasonFullMobilization
		partialMobilization := plan.Reason == botMacroReasonPartialMobilization
		objectiveStaging := plan.Reason == botMacroReasonObjectiveStaging
		coverageReserve := false
		for _, brain := range bots {
			if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.huntState.deadUntil > 0 ||
				brain.retreating || brain.lane == plan.Lane || botHPFrac(brain.c.huntState, now) < botSafeHPFrac {
				continue
			}
			coverageReserve = true
			break
		}
		if fullMobilization {
			limit = len(bots)
		} else if partialMobilization {
			limit = botPartialMobilizationLimit(len(bots))
			minimum = 1
			if limit >= 2 {
				minimum = 2
			}
			// Partial mobilization normally requires a pair so it can survive
			// the approach. Once the named structure is already in the commit
			// band, one recoverable finisher is enough to preserve the damage
			// investment while the other bodies heal or defend. This keeps the
			// launch gate conservative without turning it into a deadlock at the
			// end of an otherwise winning push.
			if botMacroObjectiveCommitWindowLocked(objective) {
				minimum = 1
			}
		} else if objectiveStaging {
			limit = len(bots)
			minimum = 1
			// Staging is a synchronized approach, not a reason to abandon
			// the lane that supplies the conversion wave.  If every eligible
			// bot is parked behind the next full-health structure, the wave
			// eventually dies or never reaches the objective and the group can
			// wait there indefinitely.  Keep one live body on the baseline
			// farm/escort route; the remaining responders still gather at the
			// shared staging point.  A one-bot roster remains fully eligible.
			if limit > 1 {
				limit--
			}
		} else if conversionReady && coverageReserve && !partialMobilization {
			// A confirmed local wave/power state earns a full healthy push even
			// against a full-health gun. The distinct baseline lane reserve keeps
			// the ordinary farm map alive; every other eligible body converts the
			// same objective instead of idling on a covered lane.
			limit = len(bots)
		}
		if conversionReady && finishWindow && !partialMobilization {
			// Once the local state is ready, keep every healthy eligible
			// responder on the same objective instead of peeling into lanes.
			limit = len(bots)
			minimum = 2
		}
		if botMacroObjectiveDamaged(objective) && !partialMobilization {
			// A first chip only proves that the conversion has started; it must
			// not collapse an already-ready push back to a three-body probe. Keep
			// the bounded response while staging, but once the live wave/power
			// predicate has granted objective_conversion_ready, every healthy
			// eligible responder should finish the same structure. This is keyed
			// to the objective state and assignment reason, never to team size.
			if conversionReady || plan.Reason == "objective_conversion_ready" {
				limit = len(bots)
			} else {
				limit = 3
			}
		}
		if !partialMobilization && objective != nil && objective.maxHealth() > 0 && objective.hp/objective.maxHealth() <= 0.70 {
			// A low-health barracks/gun is a finish signal. Spend every
			// healthy responder on it instead of preserving a farm split.
			limit = len(bots)
		}
	case botMacroRally:
		limit = 1
	}
	assigned := s.botAssignMacroRespondersLocked(plan, previous, hasPrevious, bots, x, y, limit, minimum, now)
	if assigned && baseCounterPush {
		s.botAssignBaseCounterPushLocked(plan, previous, hasPrevious, bots, counterLane, counterObjective, now)
	}
	return assigned
}

const botMacroCounterPushRole = "counter_push"

func (s *Server) botHealthyMacroResponderCountLocked(bots []*botBrain, now float64) int {
	healthy := 0
	for _, brain := range bots {
		if !s.botMacroResponderEligibleLocked(brain, now) {
			continue
		}
		if botHPFrac(brain.c.huntState, now) >= botSafeHPFrac {
			healthy++
		}
	}
	return healthy
}

// botRecoverableMacroResponderCountLocked counts bodies that can still take
// part in an already-issued objective finish. The normal healthy responder
// metric intentionally uses the safe HP band because it is used to launch a
// fresh grouped attack. A live conversion has a different tactical question:
// is there at least one avatar above the hard retreat floor who can close the
// damage already invested? Keeping those two metrics separate prevents a
// nearly destroyed gun from being abandoned merely because the rest of the
// group is recovering.
func (s *Server) botRecoverableMacroResponderCountLocked(bots []*botBrain, now float64) int {
	recoverable := 0
	for _, brain := range bots {
		if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.huntState.deadUntil > 0 {
			continue
		}
		if botHPFrac(brain.c.huntState, now) > botRetreatHPFrac {
			recoverable++
		}
	}
	return recoverable
}

func (s *Server) botAssignBaseCounterPushLocked(plan *botTeamPlan, previous botTeamPlan, hasPrevious bool, bots []*botBrain, lane int, objective *mobState, now float64) bool {
	if plan == nil || plan.Mode != botMacroBase || lane < 0 || objective == nil {
		return false
	}
	type candidate struct {
		brain    *botBrain
		distance float64
	}
	candidates := make([]candidate, 0, len(bots))
	for _, brain := range bots {
		if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.huntState.deadUntil > 0 || brain.retreating ||
			botHPFrac(brain.c.huntState, now) < botSafeHPFrac {
			continue
		}
		assignment := plan.Assignments[brain.c.objID]
		if assignment.Role == "defender" {
			continue
		}
		if assignment.Mode == botMacroBase && !botMacroObjectiveCommitWindowLocked(objective) {
			continue
		}
		if s.botCommittedStructureFocusLocked(brain.c, now) != nil &&
			!botMacroObjectiveCommitWindowLocked(objective) {
			continue
		}
		x, y := brain.c.posAtLocked(float32(now))
		candidates = append(candidates, candidate{
			brain: brain, distance: math.Hypot(float64(x-objective.x), float64(y-objective.y)),
		})
	}
	if len(candidates) == 0 {
		return false
	}
	maxCounterPush := 1
	if botMacroObjectiveCommitWindowLocked(objective) {
		// Preserve two bodies on the threatened structure. Any additional
		// healthy responders may trade into the nearly finished enemy objective;
		// the cap prevents a 3-bot defensive shell from becoming all-in, while a
		// larger live group can field the second finisher without using roster
		// size as a match rule.
		maxCounterPush = s.botHealthyMacroResponderCountLocked(bots, now) - 2
		if maxCounterPush < 1 {
			maxCounterPush = 1
		} else if maxCounterPush > 2 {
			maxCounterPush = 2
		}
	}
	selectedIDs := make(map[int32]bool, maxCounterPush)
	assigned := 0
	if hasPrevious && previous.Mode == botMacroBase && previous.ObjectiveID == plan.ObjectiveID {
		ids := make([]int32, 0, len(previous.Assignments))
		for id, assignment := range previous.Assignments {
			if assignment.Role == botMacroCounterPushRole && assignment.Mode == botMacroPush &&
				assignment.Lane == lane && assignment.ObjectiveID == objective.id {
				ids = append(ids, id)
			}
		}
		sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
		for _, id := range ids {
			if assigned >= maxCounterPush {
				break
			}
			for _, candidate := range candidates {
				if candidate.brain.c.objID != id || selectedIDs[id] {
					continue
				}
				assignment := plan.Assignments[id]
				assignment.Mode = botMacroPush
				assignment.Lane = lane
				assignment.ObjectiveID = objective.id
				assignment.BaselineLane = candidate.brain.lane
				assignment.Role = botMacroCounterPushRole
				assignment.Reason = "base_pressure_counter_push"
				assignment.Aggressive = true
				assignment.Coverage = false
				plan.Assignments[id] = assignment
				selectedIDs[id] = true
				assigned++
				break
			}
		}
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].distance != candidates[j].distance {
			return candidates[i].distance > candidates[j].distance
		}
		return candidates[i].brain.c.objID < candidates[j].brain.c.objID
	})
	for _, candidate := range candidates {
		if assigned >= maxCounterPush || selectedIDs[candidate.brain.c.objID] {
			break
		}
		brain := candidate.brain
		assignment := plan.Assignments[brain.c.objID]
		assignment.Mode = botMacroPush
		assignment.Lane = lane
		assignment.ObjectiveID = objective.id
		assignment.BaselineLane = brain.lane
		assignment.Role = botMacroCounterPushRole
		assignment.Reason = "base_pressure_counter_push"
		assignment.Aggressive = true
		assignment.Coverage = false
		plan.Assignments[brain.c.objID] = assignment
		selectedIDs[brain.c.objID] = true
		assigned++
	}
	return assigned > 0
}

func (s *Server) botMacroResponderEligibleLocked(brain *botBrain, now float64) bool {
	if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.huntState.deadUntil > 0 || brain.retreating {
		return false
	}
	return s.botCommittedStructureFocusLocked(brain.c, now) == nil
}

// botAltarAssaultResponderEligibleLocked is the terminal-objective exception
// to the ordinary "retreating bots are not responders" rule. Once the enemy
// altar is unshielded, a bot that has escaped immediate pressure but is still
// below the normal 55% rejoin band should be able to reform with the finish
// group. The hard retreat floor remains absolute; no match-clock or roster-size
// condition is involved.
func (s *Server) botAltarAssaultResponderEligibleLocked(brain *botBrain, objective *mobState, now float64) bool {
	if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.huntState.deadUntil > 0 ||
		objective == nil || objective.dead || !objective.altar || !objective.enemyOf(brain.c.playerTeam()) ||
		brain.c.altarShieldedLocked(objective) {
		return false
	}
	frac := botHPFrac(brain.c.huntState, now)
	if frac < botAltarAssaultRejoinHPFrac ||
		(frac < botSafeHPFrac && s.botIncomingPressureLocked(brain, now)) ||
		s.botCommittedStructureFocusLocked(brain.c, now) != nil {
		return false
	}
	return true
}

// botAssignMacroRespondersLocked overlays a bounded response on the baseline
// assignments. Existing healthy responders are retained first; only ineligible
// responders are replaced, with remaining candidates ordered deterministically.
func (s *Server) botAssignMacroRespondersLocked(plan *botTeamPlan, previous botTeamPlan, hasPrevious bool, bots []*botBrain, x, y float32, limit, minimum int, now float64) bool {
	type candidate struct {
		brain    *botBrain
		healthy  bool
		distance float64
		power    float64
		farmDebt int
	}
	var candidates []candidate
	byID := make(map[int32]*botBrain, len(bots))
	responderHPFrac := botSafeHPFrac
	altarAssault := plan.Mode == botMacroAltar && plan.Reason == "enemy_altar_open"
	var objective *mobState
	if len(bots) > 0 && bots[0] != nil && bots[0].c != nil {
		objective = botPlanObjectiveLocked(bots[0].c.inst, plan.Team, *plan)
	}
	if altarAssault {
		responderHPFrac = botAltarAssaultRejoinHPFrac
	}
	if plan.Reason == botMacroReasonFullMobilization || plan.Reason == botMacroReasonPartialMobilization || plan.Reason == "objective_conversion_ready" {
		// Full mobilization includes every living, non-critical body. Keeping
		// the normal 55% threshold here silently turns "all bots" into "all
		// healthy bots" exactly when a conversion needs reinforcement. The same
		// threshold applies to an ordinary conversion-ready push: a bot above the
		// hard retreat floor can still contribute to the grouped attack, while the
		// local retreat authority remains able to peel it out below that floor.
		responderHPFrac = botRetreatHPFrac
	} else if plan.Reason == botMacroReasonObjectiveStaging {
		// A staged group is already locally safe enough to regroup, but not
		// necessarily fountain-full. Use the same rejoin band as the staging
		// release so recoverable bodies can actually be selected into the group.
		responderHPFrac = botObjectiveRejoinHPFrac
	}
	for _, brain := range bots {
		eligible := s.botMacroResponderEligibleLocked(brain, now)
		if altarAssault {
			eligible = s.botAltarAssaultResponderEligibleLocked(brain, objective, now)
		}
		if !eligible {
			continue
		}
		frac := botHPFrac(brain.c.huntState, now)
		cx, cy := brain.c.posAtLocked(float32(now))
		candidates = append(candidates, candidate{
			brain: brain, healthy: frac > responderHPFrac,
			distance: math.Hypot(float64(cx-x), float64(cy-y)),
			power:    botMacroHeroPowerLocked(brain, now),
			farmDebt: botFarmDebtLocked(brain.c.inst, brain),
		})
		byID[brain.c.objID] = brain
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].healthy != candidates[j].healthy {
			return candidates[i].healthy
		}
		// In a non-critical defense assignment, keep the most indebted bot on
		// its farm lane when two responders are tactically close. A direct
		// structure threat still wins; this only prevents a needless farm loss
		// caused by choosing a 2u-closer but already well-fed responder.
		if plan.Mode == botMacroBase && math.Abs(candidates[i].distance-candidates[j].distance) <= 24 &&
			candidates[i].farmDebt != candidates[j].farmDebt {
			return candidates[i].farmDebt < candidates[j].farmDebt
		}
		if (plan.Mode == botMacroPush || plan.Mode == botMacroAltar) &&
			math.Abs(candidates[i].distance-candidates[j].distance) <= 24 &&
			math.Abs(candidates[i].power-candidates[j].power) > 0.05 {
			return candidates[i].power > candidates[j].power
		}
		if candidates[i].distance != candidates[j].distance {
			return candidates[i].distance < candidates[j].distance
		}
		return candidates[i].brain.c.objID < candidates[j].brain.c.objID
	})
	healthy := 0
	for _, candidate := range candidates {
		if candidate.healthy {
			healthy++
		}
	}
	if healthy < minimum {
		return false
	}
	preservePrevious := hasPrevious && previous.Mode == plan.Mode && previous.Lane == plan.Lane && previous.ObjectiveID == plan.ObjectiveID
	if preservePrevious {
		previousEligible := 0
		for id, assignment := range previous.Assignments {
			brain := byID[id]
			if brain != nil && botPlanStrategicResponder(previous.Mode, assignment) && botHPFrac(brain.c.huntState, now) > responderHPFrac {
				previousEligible++
			}
		}
		// A live plan may keep its eligible responders even when the
		// instantaneous pressure score would otherwise lower the cap.
		if previousEligible > limit {
			limit = previousEligible
		}
	}
	if limit > healthy {
		limit = healthy
	}
	selected := make([]*botBrain, 0, limit)
	selectedIDs := make(map[int32]bool, limit)
	if preservePrevious {
		ids := make([]int32, 0, len(previous.Assignments))
		for id, assignment := range previous.Assignments {
			if botPlanStrategicResponder(previous.Mode, assignment) {
				ids = append(ids, id)
			}
		}
		sort.Slice(ids, func(i, j int) bool {
			left, right := previous.Assignments[ids[i]], previous.Assignments[ids[j]]
			leftRank, rightRank := botPlanResponderRank(previous.Mode, left), botPlanResponderRank(previous.Mode, right)
			if leftRank != rightRank {
				return leftRank < rightRank
			}
			return ids[i] < ids[j]
		})
		for _, id := range ids {
			brain := byID[id]
			if brain == nil || len(selected) >= limit || botHPFrac(brain.c.huntState, now) <= responderHPFrac {
				continue
			}
			selected = append(selected, brain)
			selectedIDs[id] = true
		}
	}
	for _, candidate := range candidates {
		if len(selected) >= limit || selectedIDs[candidate.brain.c.objID] || !candidate.healthy {
			continue
		}
		selected = append(selected, candidate.brain)
		selectedIDs[candidate.brain.c.objID] = true
	}
	for i, brain := range selected {
		assignment := plan.Assignments[brain.c.objID]
		assignment.Mode = plan.Mode
		assignment.Lane = plan.Lane
		assignment.ObjectiveID = plan.ObjectiveID
		assignment.BaselineLane = brain.lane
		assignment.Reason = plan.Reason
		assignment.Aggressive = plan.Mode == botMacroPush || plan.Mode == botMacroAltar
		assignment.Coverage = i > 0
		switch plan.Mode {
		case botMacroBase:
			assignment.Role = "defender"
		case botMacroAltar:
			assignment.Role = "assault"
		case botMacroPush:
			assignment.Role = "push"
		case botMacroRally:
			assignment.Role = "rally"
		}
		if i > 0 {
			assignment.Role = "cover"
			if plan.Mode == botMacroBase {
				// Base cover is an emergency reserve, not a second defender.
				// Keep its route on the baseline farm lane so the reserve does
				// not abandon the lane while the named defender handles the
				// structure. Promotion can still overwrite this lane later if
				// the responder is actually needed.
				assignment.Lane = brain.lane
			}
			if plan.Mode != botMacroBase {
				assignment.Mode = botMacroCover
				assignment.Aggressive = false
			}
			objective := botPlanObjectiveLocked(brain.c.inst, plan.Team, *plan)
			if (plan.Mode == botMacroPush || plan.Mode == botMacroAltar) &&
				(plan.Reason == botMacroReasonFullMobilization || plan.Reason == botMacroReasonPartialMobilization || plan.Reason == "enemy_altar_open" ||
					s.botObjectiveConversionReadyLocked(brain.c.inst, plan.Team, objective, now) ||
					botMacroObjectiveCommitWindowLocked(objective)) {
				// Keep every locally eligible responder on the objective when
				// the wave/group state is already favourable. This is a tactical
				// commitment, not a check of total avatar count.
				assignment.Mode = plan.Mode
				assignment.Role = "push_support"
				assignment.Aggressive = true
			}
		}
		plan.Assignments[brain.c.objID] = assignment
	}
	return true
}

// botMacroHeroPowerLocked is a small responder-selection heuristic. It is not
// a team-size or roster comparison: it ranks only the live candidate's ability
// to survive and deal damage at the named objective right now.
func botMacroHeroPowerLocked(brain *botBrain, now float64) float64 {
	if brain == nil || brain.c == nil || brain.c.huntState == nil || brain.c.huntState.deadUntil > 0 {
		return 0
	}
	hp := botHPFrac(brain.c.huntState, now)
	if hp <= 0 {
		return 0
	}
	levelMul := 1 + math.Min(float64(brain.c.huntState.level), 19)*0.06
	killMul := 1 + math.Min(float64(brain.c.huntState.frags), 20)*0.015
	return hp * levelMul * killMul
}

func botHumanMacroActiveLocked(inst *huntInstance, c *conn, now float64) bool {
	if c == nil || c.huntState == nil || c.huntState.deadUntil > 0 {
		return false
	}
	if inst.dota.laneActiveHumans[c.objID] {
		return true
	}
	hx, hy := botHomeLocked(c)
	x, y := c.posAtLocked(float32(now))
	return dist2(x, y, hx, hy) > float32(botHumanLeftBaseRadius*botHumanLeftBaseRadius)
}

func (s *Server) botTeamBasePressureLocked(inst *huntInstance, team int32, now float64) bool {
	return s.botTeamBasePressureSeverityLocked(inst, team, now) > 0
}

// Severity is based only on live attackers/focus around the altar. Historical
// altar damage is deliberately excluded: once the pressure leaves, defense ends.
func (s *Server) botTeamBasePressureSeverityLocked(inst *huntInstance, team int32, now float64) int {
	return s.botTeamBasePressureSeverityRadiusLocked(inst, team, now, botMacroBasePressureRadius)
}

func (s *Server) botTeamBasePressureSeverityRadiusLocked(inst *huntInstance, team int32, now, radius float64) int {
	altar := botTeamAltarLocked(inst, team)
	if altar == nil || altar.dead {
		return 0
	}
	r2 := float32(radius * radius)
	severity := 0
	visionSources := dotaTeamVisionSourcesLocked(inst, team, now)
	for _, m := range inst.mobs {
		if m.dead || m.structure || !m.enemyOf(team) || !botVisibleEnemyMobLocked(team, m, visionSources) {
			continue
		}
		if dist2(m.x, m.y, altar.x, altar.y) <= r2 {
			severity++
			if dist2(m.x, m.y, altar.x, altar.y) <= 26*26 {
				severity++
			}
		}
	}
	for _, mem := range inst.members {
		if !botVisibleEnemyMemberLocked(inst, team, mem, now, visionSources) {
			continue
		}
		x, y := mem.posAtLocked(float32(now))
		if dist2(x, y, altar.x, altar.y) <= r2 {
			severity += 2
			for _, ally := range inst.members {
				if ally == nil || ally.huntState == nil || ally.playerTeam() != team || ally.huntState.deadUntil > 0 {
					continue
				}
				if mem.huntState.pvpTarget == ally.objID || mem.huntState.attackTarget == ally.objID {
					severity += 2
					break
				}
			}
		}
	}
	return severity
}

func botTeamAltarLocked(inst *huntInstance, team int32) *mobState {
	for _, m := range inst.mobs {
		if m.altar && m.team == team {
			return m
		}
	}
	return nil
}

func otherDotaTeam(team int32) int32 {
	if team == dotaTeamHuman {
		return dotaTeamElf
	}
	return dotaTeamHuman
}

func (s *Server) botBestMacroLaneLocked(inst *huntInstance, team int32, now float64) (int, *mobState, bool, int) {
	bestLane, bestScore := 0, math.Inf(-1)
	var bestObjective *mobState
	bestProgress, bestCoverage := false, 0
	for lane := range inst.dota.m.Lanes {
		objective, score := s.botMacroLaneObjectiveLocked(inst, team, lane)
		progress, coverage := s.botMacroLaneProgressLocked(inst, team, lane, objective, now)
		if progress {
			score += 30
		}
		if _, _, ok := botLaneFrontLockedForTeam(inst, team, lane); ok {
			score += 20
		}
		score += float64(coverage * 4)
		if score > bestScore || (score == bestScore && lane < bestLane) {
			bestLane, bestScore, bestObjective = lane, score, objective
			bestProgress, bestCoverage = progress, coverage
		}
	}
	return bestLane, bestObjective, bestProgress, bestCoverage
}

// botContinuePushLaneLocked keeps a live offensive investment on its lane after
// the named structure is destroyed. It is intentionally based on the previous
// plan's mode and the lane's current live front, never on total team size or a
// match-clock phase. Base/altar emergencies are handled before this helper is
// consulted by botPlanTeamLocked.
func (s *Server) botContinuePushLaneLocked(inst *huntInstance, team int32, previous botTeamPlan, hasPrevious bool, now float64) (int, *mobState, bool, int, bool) {
	if !hasPrevious || (previous.Mode != botMacroPush && previous.Mode != botMacroRally) ||
		previous.Lane < 0 || previous.Lane >= len(inst.dota.m.Lanes) {
		return 0, nil, false, 0, false
	}
	objective, _ := s.botMacroLaneObjectiveLocked(inst, team, previous.Lane)
	if objective == nil {
		return 0, nil, false, 0, false
	}
	progress, coverage := s.botMacroLaneProgressLocked(inst, team, previous.Lane, objective, now)
	if botAnyMobilizationReason(previous.Reason) {
		// Destroying the named checkpoint advances the route; it does not cancel
		// the already-launched group operation. In particular, do not require a
		// fresh wave/coverage sample in the same tick: the group may have just
		// destroyed the old objective and the wave can be a few frames behind.
		// The next live front object on this lane becomes the new objective while
		// the mobilization phase remains authoritative.
		if coverage == 0 {
			coverage = 1
		}
		return previous.Lane, objective, true, coverage, true
	}
	if !progress {
		return 0, nil, false, 0, false
	}
	return previous.Lane, objective, progress, coverage, true
}

func (s *Server) botMacroLaneProgressLocked(inst *huntInstance, team int32, lane int, objective *mobState, now float64) (bool, int) {
	_, _, hasFront := botLaneFrontLockedForTeam(inst, team, lane)
	coverage := 0
	for _, mem := range inst.members {
		if mem == nil || mem.huntState == nil || mem.huntState.deadUntil > 0 || mem.playerTeam() != team {
			continue
		}
		if brain := inst.bots[mem.objID]; brain != nil && brain.retreating {
			continue
		}
		if !isBotConn(mem) && !botHumanMacroActiveLocked(inst, mem, now) {
			continue
		}
		x, y := mem.posAtLocked(float32(now))
		if botPointNearLane(inst.dota.m.Lanes[lane], x, y, botMacroCoverageRadius) {
			coverage++
		}
	}
	if hasFront {
		return true, coverage
	}
	if objective != nil && botMacroObjectiveDamaged(objective) && coverage > 0 {
		return true, coverage
	}
	return botLaneFrontNearObjectiveLocked(inst, team, lane, objective), coverage
}

func botMacroObjectiveDamaged(objective *mobState) bool {
	return objective != nil && objective.maxHealth() > 0 && objective.hp < objective.maxHealth()*0.98
}

// botMacroObjectiveFinishWindowLocked is the execution-side finish signal. A
// materially damaged front structure is worth converting immediately even if
// a nearby enemy hero makes the broader conversion predicate temporarily
// pessimistic; otherwise support bots peel off exactly when their extra attack
// is needed to close the objective.
func botMacroObjectiveFinishWindowLocked(objective *mobState) bool {
	if objective == nil || objective.dead || !objective.structure {
		return false
	}
	maxHP := objective.maxHealth()
	return maxHP > 0 && objective.hp/maxHP <= 0.70
}

// botMacroObjectiveCommitWindowLocked is the broader execution-debt band. Once
// a front structure has lost this much HP, abandoning the lane for a marginal
// farm or hero duel wastes the damage already invested. The narrower finish
// window above remains reserved for the final all-in; this band only applies
// when a real assigned group is already close enough to convert the damage.
func botMacroObjectiveCommitWindowLocked(objective *mobState) bool {
	if objective == nil || objective.dead || !objective.structure {
		return false
	}
	maxHP := objective.maxHealth()
	return maxHP > 0 && objective.hp/maxHP <= botMacroCommitDamageHPFrac
}

// botPreserveCriticalFinishLocked keeps a committed conversion alive when a
// non-damaged checkpoint receives a fresh contact signal. A live barracks or
// altar that is already losing health still has priority; a full-health
// barracks merely seeing a wave must not cancel a five-body finish on an enemy
// gun that is already inside the execution window. The previous assignment is
// part of the premise so an incidental low-health building cannot manufacture a
// cross-map attack without an existing team commitment.
func (s *Server) botPreserveCriticalFinishLocked(inst *huntInstance, team int32, previous botTeamPlan, hasPrevious bool, bots []*botBrain, defenseStructure *mobState, now float64) bool {
	if inst == nil || inst.dota == nil || !hasPrevious || defenseStructure == nil || defenseStructure.dead ||
		defenseStructure.team != team || defenseStructure.altar {
		return false
	}
	if defenseStructure.dotaRole == gamedata.DotaCreepTower {
		maxHP := defenseStructure.maxHealth()
		if maxHP <= 0 || defenseStructure.hp/maxHP <= botMacroCommitDamageHPFrac {
			return false
		}
	}
	if previous.Mode == botMacroBase {
		// A base plan is already the stable defensive shell. Keep it and let
		// botAssignBaseCounterPushLocked express the finish trade explicitly;
		// converting the whole plan to push would reintroduce base/push churn.
		return false
	}
	_, critical := s.botCriticalEnemyObjectiveLocked(inst, team)
	if !botMacroObjectiveFinishWindowLocked(critical) {
		return false
	}
	committed := false
	if botAnyMobilizationReason(previous.Reason) && previous.ObjectiveID == critical.id {
		committed = true
	}
	if !committed {
		for _, assignment := range previous.Assignments {
			if assignment.Role == botMacroCounterPushRole && assignment.ObjectiveID == critical.id {
				committed = true
				break
			}
		}
	}
	if !committed {
		return false
	}
	// Keep a real execution body available after reserving the local defender.
	// A gun is a checkpoint and may be traded when one recoverable finisher can
	// close an already-damaged enemy front object. Barracks and altar pressure
	// retain the larger defensive premise: those structures are more valuable
	// than a single checkpoint finish.
	minimum := 3
	if defenseStructure.dotaRole == gamedata.DotaGun {
		minimum = 1
	}
	if minimum == 1 {
		return s.botRecoverableMacroResponderCountLocked(bots, now) >= minimum
	}
	return s.botHealthyMacroResponderCountLocked(bots, now) >= minimum
}

// botPreserveCommittedPushAgainstGunLocked keeps a live offensive commitment
// through ordinary gun pressure. A fresh push still needs a group, while a
// partial push in the objective commit band may retain one recoverable finisher
// to close its already-invested damage. Barracks and altar threats never enter
// this helper and remain hard overrides.
func (s *Server) botPreserveCommittedPushAgainstGunLocked(inst *huntInstance, team int32, previous botTeamPlan, hasPrevious bool, bots []*botBrain, defenseStructure *mobState, severity int, now float64) bool {
	if inst == nil || defenseStructure == nil || defenseStructure.dead ||
		defenseStructure.dotaRole != gamedata.DotaGun || !hasPrevious ||
		previous.ObjectiveID == 0 ||
		severity >= botMacroBaseSeverePressure {
		return false
	}
	objective := inst.mobs[previous.ObjectiveID]
	if objective == nil {
		return false
	}
	if objective.dead {
		// The exact tick that records a successful checkpoint kill can still
		// report a gun threat at home. Preserve the offensive investment long
		// enough to select the next live front on the same lane; otherwise the
		// team abandons a winning conversion and spends the next planning window
		// walking back to its own gun. A severe gun pressure signal remains a
		// hard defensive override, and the normal responder eligibility keeps
		// this trade from being made by a lone survivor.
		_, nextObjective, _, _, ok := s.botContinuePushLaneLocked(inst, team, previous, hasPrevious, now)
		if !ok || nextObjective == nil || nextObjective.dead || !nextObjective.enemyOf(team) {
			// Some authored routes have one checkpoint per lane. After that
			// checkpoint falls, the next live front can legitimately be on another
			// lane; retain the offensive mode while the normal planner selects it.
			_, nextObjective, _, _ = s.botBestMacroLaneLocked(inst, team, now)
		}
		if nextObjective == nil || nextObjective.dead || !nextObjective.enemyOf(team) {
			return false
		}
		return s.botHealthyMacroResponderCountLocked(bots, now) >= 3
	}
	// Preparation is non-aggressive while its named objective is live. If that
	// objective dies before launch, however, the mobilization operation must
	// advance rather than be cancelled by a simultaneous ordinary gun signal;
	// the next live front is the continuation target selected above.
	if previous.Reason == botMacroReasonMobilizationPreparation {
		return false
	}
	if !objective.structure || !objective.enemyOf(team) ||
		(objective.altar && !inst.dota.altarVulnerableLocked(objective)) {
		return false
	}
	if previous.Mode == botMacroAltar {
		if previous.Reason != "enemy_altar_open" || !objective.altar || !botMacroObjectiveFinishWindowLocked(objective) {
			return false
		}
	} else {
		if previous.Mode != botMacroPush ||
			(previous.Reason != "objective_conversion_ready" && previous.Reason != botMacroReasonObjectiveStaging &&
				previous.Reason != botMacroReasonFullMobilization &&
				(previous.Reason != botMacroReasonPartialMobilization || !botMacroObjectiveCommitWindowLocked(objective))) {
			return false
		}
	}
	minimum := 2
	if previous.Reason == botMacroReasonPartialMobilization && botMacroObjectiveCommitWindowLocked(objective) {
		minimum = 1
	}
	if minimum == 1 {
		return s.botRecoverableMacroResponderCountLocked(bots, now) >= minimum
	}
	return s.botHealthyMacroResponderCountLocked(bots, now) >= minimum
}

func botLaneFrontNearObjectiveLocked(inst *huntInstance, team int32, lane int, objective *mobState) bool {
	if objective == nil {
		return false
	}
	for _, m := range inst.mobs {
		if m.dead || m.structure || m.team != team || m.laneIdx < 0 || len(m.lane) == 0 ||
			len(m.lane) != len(inst.dota.m.Lanes[lane]) || m.lane[0] != inst.dota.m.Lanes[lane][0] {
			continue
		}
		if dist2(m.x, m.y, objective.x, objective.y) <= 45*45 {
			return true
		}
	}
	return false
}

func botMacroObjectivePointLocked(inst *huntInstance, team int32, lane int, objective *mobState) (float32, float32) {
	if objective != nil {
		return objective.x, objective.y
	}
	if x, y, ok := botLaneFrontLockedForTeam(inst, team, lane); ok {
		return x, y
	}
	if lane >= 0 && lane < len(inst.dota.m.Lanes) && len(inst.dota.m.Lanes[lane]) > 0 {
		p := inst.dota.m.Lanes[lane][len(inst.dota.m.Lanes[lane])/2]
		return float32(p.X), float32(p.Y)
	}
	return 0, 0
}

func (s *Server) botMacroLaneObjectiveLocked(inst *huntInstance, team int32, lane int) (*mobState, float64) {
	var best *mobState
	bestIndex := -1
	bestRole := -1
	bestDistance := math.Inf(1)
	for _, m := range inst.mobs {
		if m.dead || !m.structure || !m.enemyOf(team) || m.altar ||
			!botPointNearLane(inst.dota.m.Lanes[lane], m.x, m.y, botMacroLanePointRadius) {
			continue
		}
		index, distance, ok := botLaneStructureFrontIndex(inst.dota.m.Lanes[lane], m.x, m.y)
		if !ok {
			continue
		}
		role := botLaneStructureRolePriority(m.dotaRole)
		better := best == nil
		if !better {
			// Always select the first live obstacle in the direction of
			// travel. A role-value-only choice can send a group past a live
			// cannon toward a barracks that the safety route can never reach.
			if team == dotaTeamHuman {
				better = index < bestIndex
			} else {
				better = index > bestIndex
			}
			if !better && index == bestIndex {
				better = role > bestRole || (role == bestRole &&
					(distance < bestDistance || (distance == bestDistance && m.id < best.id)))
			}
		}
		if better {
			best, bestIndex, bestRole, bestDistance = m, index, role, distance
		}
	}
	if best == nil {
		return nil, -10 // a lane without a live enemy objective remains a valid rally fallback
	}
	if team == dotaTeamHuman {
		return best, float64(100 - bestIndex)
	}
	return best, float64(bestIndex)
}

func botMacroObjectiveValue(role gamedata.DotaRole) float64 {
	switch role {
	case gamedata.DotaCreepTower:
		return 100
	case gamedata.DotaGun:
		return 35
	case gamedata.DotaGenerator:
		return 15
	default:
		return 10
	}
}

// botLaneStructureFrontIndex projects a structure onto the authored lane. The
// lane waypoints are ordered from the human altar to the elf altar, so the
// lowest index is the first obstacle for the human push and the highest index
// is the first obstacle for the elf push. Choosing by role value alone used to
// select a barracks behind a live cannon and made the safety detour correctly
// prevent every bot from ever reaching that barracks.
func botLaneStructureFrontIndex(lane []gamedata.Vec2, x, y float32) (index int, distance float64, ok bool) {
	if len(lane) == 0 {
		return 0, 0, false
	}
	index, distance = 0, math.Inf(1)
	for i, point := range lane {
		candidate := math.Hypot(float64(float32(point.X)-x), float64(float32(point.Y)-y))
		if candidate < distance {
			index, distance = i, candidate
		}
	}
	return index, distance, distance <= botMacroLanePointRadius
}

func botLaneStructureRolePriority(role gamedata.DotaRole) int {
	switch role {
	case gamedata.DotaGun:
		return 3
	case gamedata.DotaCreepTower:
		return 2
	case gamedata.DotaGenerator:
		return 1
	default:
		return 0
	}
}

const (
	// Objective readiness is deliberately spatial and weighted. It describes
	// the combat power that can reach this exact building, not how many avatars
	// happen to exist in the match.
	botObjectivePowerRadius    = 72.0
	botObjectiveApproachRadius = 180.0
	botObjectiveMinPower       = 1.80
)

// botObjectiveHasAlliedWaveLocked reports whether this team's live lane wave is
// close enough to become a useful conversion wave for the named structure. It is
// intentionally independent of the conversion power calculation: callers use it
// to distinguish "wait for the wave/group" from "return to ordinary farming".
func (s *Server) botObjectiveHasAlliedWaveLocked(inst *huntInstance, team int32, objective *mobState) bool {
	if inst == nil || inst.dota == nil || objective == nil || objective.dead || !objective.structure ||
		!objective.enemyOf(team) {
		return false
	}
	lane := botNearestLaneToPointLocked(inst.dota, objective.x, objective.y)
	for _, mob := range botSortedMobs(inst) {
		if mob == nil || mob.dead || mob.structure || mob.team != team || !botTeleportLaneCreep(mob) {
			continue
		}
		if botNearestLaneToPointLocked(inst.dota, mob.x, mob.y) != lane {
			continue
		}
		if math.Hypot(float64(mob.x-objective.x), float64(mob.y-objective.y)) > botObjectiveApproachRadius {
			continue
		}
		maxHP := mob.maxHP
		if maxHP <= 0 {
			maxHP = mob.mob.Health
		}
		if maxHP > 0 && mob.hp > 0 {
			return true
		}
	}
	return false
}

// botObjectiveConversionReadyLocked reports whether a live objective can be
// converted by the current local state. A healthy nearby hero contributes more
// than a distant or low-health one; a live allied wave contributes soak power;
// nearby enemy heroes subtract pressure. This lets a side defend or push
// correctly in a full match, a partial lobby, or a match with human players,
// without branching on the total avatar count.
func (s *Server) botObjectiveConversionReadyLocked(inst *huntInstance, team int32, objective *mobState, now float64) bool {
	return s.botObjectiveConversionReadyExcludingLocked(inst, team, objective, 0, now)
}

// botObjectiveConversionReadyExcludingLocked is the same live conversion test
// with one allied hero removed from the power calculation. It is used only for
// farm-reserve decisions: the orchestrator must prove that reserving the most
// indebted bot does not turn a real objective finish into a suicide dive.
func (s *Server) botObjectiveConversionReadyExcludingLocked(inst *huntInstance, team int32, objective *mobState, excludeID int32, now float64) bool {
	if inst == nil || inst.dota == nil || objective == nil || objective.dead || !objective.structure ||
		!objective.enemyOf(team) || (objective.altar && !inst.dota.altarVulnerableLocked(objective)) {
		return false
	}

	unitPower := func(mem *conn, distance float64, retreating bool) float64 {
		if mem == nil || mem.huntState == nil || mem.huntState.deadUntil > 0 || retreating {
			return 0
		}
		hp := botHPFrac(mem.huntState, now)
		if hp <= botRetreatHPFrac {
			return 0
		}
		if hp > 1 {
			hp = 1
		}
		weight := 0.0
		switch {
		case distance <= botObjectivePowerRadius:
			weight = 1
		case distance <= botObjectiveApproachRadius:
			weight = 0.35 + 0.65*(botObjectiveApproachRadius-distance)/(botObjectiveApproachRadius-botObjectivePowerRadius)
		}
		if weight <= 0 {
			return 0
		}
		// Level is a small modifier, not a replacement for live HP and position.
		levelMul := 1 + math.Min(float64(mem.huntState.level), 19)*0.025
		return hp * weight * levelMul
	}

	alliedPower, enemyPower := 0.0, 0.0
	// Macro decisions must use the same information boundary as the client and
	// local combat layer. The server still knows every member's coordinates, but
	// an enemy outside this team's current vision must not contribute to the
	// visible pressure estimate.
	visionSources := s.dotaVisionSourcesLocked(inst, team, now)
	for _, mem := range inst.members {
		if mem == nil || mem.huntState == nil || mem.huntState.closed || mem.huntState.deadUntil > 0 {
			continue
		}
		if excludeID != 0 && mem.objID == excludeID {
			continue
		}
		x, y := mem.posAtLocked(float32(now))
		distance := math.Hypot(float64(x-objective.x), float64(y-objective.y))
		if distance > botObjectiveApproachRadius {
			continue
		}
		retreating := false
		if brain := inst.bots[mem.objID]; brain != nil {
			retreating = brain.retreating
		}
		power := unitPower(mem, distance, retreating)
		if mem.playerTeam() == team {
			alliedPower += power
		} else {
			if !botVisibleEnemyMemberLocked(inst, team, mem, now, visionSources) {
				continue
			}
			enemyPower += power
		}
	}

	// A creep wave is the structure's aggro/soak layer. Count its health as a
	// small continuous contribution rather than using a fixed creep count.
	wavePower := 0.0
	for _, mob := range botSortedMobs(inst) {
		if mob == nil || mob.dead || mob.structure || mob.team != team || !botTeleportLaneCreep(mob) {
			continue
		}
		if botNearestLaneToPointLocked(inst.dota, mob.x, mob.y) != botNearestLaneToPointLocked(inst.dota, objective.x, objective.y) {
			continue
		}
		distance := math.Hypot(float64(mob.x-objective.x), float64(mob.y-objective.y))
		if distance > botObjectivePowerRadius {
			continue
		}
		maxHP := mob.maxHP
		if maxHP <= 0 {
			maxHP = mob.mob.Health
		}
		if maxHP <= 0 {
			continue
		}
		hp := mob.hp / maxHP
		if hp < 0 {
			hp = 0
		} else if hp > 1 {
			hp = 1
		}
		wavePower += 0.12 * hp
	}

	objectiveFrac := 1.0
	if maxHP := objective.maxHealth(); maxHP > 0 {
		objectiveFrac = objective.hp / maxHP
	}
	required := botObjectiveMinPower
	// A full structure is a wave-first objective. A healthy grouped hero push can
	// look powerful enough on paper, but without allied creeps the structure
	// itself supplies all the aggro and repeatedly forces the group to retreat;
	// that loses both HP and the next farm cycle. Damaged structures are already
	// execution debt and remain eligible for a no-wave finish below.
	if objectiveFrac > 0.98 && wavePower <= 0 {
		return false
	}
	if objectiveFrac <= botMacroCommitDamageHPFrac {
		required -= 0.25
	}
	if objectiveFrac <= 0.70 {
		required -= 0.45
	}
	if wavePower > 0 {
		required -= math.Min(0.25, wavePower)
	}
	if alliedPower+wavePower < required {
		return false
	}
	// Keep a modest margin against nearby enemy heroes. Equal displayed power is
	// not actually equal in a live conversion: the defender has home-ground
	// movement, structure damage, and the ability to reinforce. The margin is
	// still small enough to allow a real advantage to convert, but prevents a
	// nominally even group from repeatedly feeding under tower pressure.
	if enemyPower > 0 && alliedPower+wavePower < enemyPower*1.08 {
		return false
	}
	return true
}

// botVisibleEnemyHeroNearObjectiveLocked is deliberately vision-bounded. A
// counter-push may stage behind a live allied wave when the target checkpoint
// is not visibly defended, but an unseen enemy outside the team's information
// boundary must not be invented as either a threat or a guarantee of safety.
func (s *Server) botVisibleEnemyHeroNearObjectiveLocked(inst *huntInstance, team int32, objective *mobState, now float64) bool {
	if inst == nil || objective == nil {
		return false
	}
	visionSources := dotaTeamVisionSourcesLocked(inst, team, now)
	for _, mem := range inst.members {
		if mem == nil || mem.huntState == nil || mem.huntState.deadUntil > 0 || mem.playerTeam() == team ||
			!botVisibleEnemyMemberLocked(inst, team, mem, now, visionSources) {
			continue
		}
		x, y := mem.posAtLocked(float32(now))
		if math.Hypot(float64(x-objective.x), float64(y-objective.y)) <= botObjectiveApproachRadius {
			return true
		}
	}
	return false
}

func botLaneFrontLockedForTeam(inst *huntInstance, team int32, lane int) (float32, float32, bool) {
	if lane < 0 || lane >= len(inst.dota.m.Lanes) {
		return 0, 0, false
	}
	var best *mobState
	bestIdx := -1
	if team == dotaTeamElf {
		bestIdx = len(inst.dota.m.Lanes[lane]) + 1
	}
	for _, m := range botSortedMobs(inst) {
		if m.dead || m.structure || m.team != team || m.laneIdx < 0 || len(m.lane) == 0 ||
			len(m.lane) != len(inst.dota.m.Lanes[lane]) || m.lane[0] != inst.dota.m.Lanes[lane][0] {
			continue
		}
		if (team == dotaTeamHuman && (m.laneIdx > bestIdx || (m.laneIdx == bestIdx && (best == nil || m.id < best.id)))) ||
			(team == dotaTeamElf && (m.laneIdx < bestIdx || (m.laneIdx == bestIdx && (best == nil || m.id < best.id)))) {
			best, bestIdx = m, m.laneIdx
		}
	}
	if best == nil {
		return 0, 0, false
	}
	return best.x, best.y, true
}

func botPointNearLane(lane []gamedata.Vec2, x, y float32, radius float64) bool {
	for _, p := range lane {
		if math.Hypot(float64(float32(p.X)-x), float64(float32(p.Y)-y)) <= radius {
			return true
		}
	}
	return false
}

func botMacroObjectiveName(m *mobState) string {
	if m == nil {
		return "lane"
	}
	switch m.dotaRole {
	case gamedata.DotaCreepTower:
		return "barracks"
	case gamedata.DotaGun:
		return "gun"
	case gamedata.DotaGenerator:
		return "generator"
	default:
		return "structure"
	}
}

func botTeamPlanKey(plan botTeamPlan) string {
	var b strings.Builder
	b.WriteString(strconv.Itoa(plan.AIVersion))
	b.WriteByte(':')
	b.WriteString(plan.Mode)
	b.WriteByte(':')
	b.WriteString(strconv.Itoa(plan.Lane))
	b.WriteByte(':')
	b.WriteString(strconv.FormatInt(int64(plan.ObjectiveID), 10))
	b.WriteByte(':')
	b.WriteString(strconv.FormatInt(int64(plan.FocusTarget), 10))
	b.WriteByte(':')
	ids := make([]int32, 0, len(plan.Assignments))
	for id := range plan.Assignments {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	for _, id := range ids {
		a := plan.Assignments[id]
		b.WriteString(strconv.FormatInt(int64(id), 10))
		b.WriteByte('=')
		b.WriteString(a.Mode)
		b.WriteByte('/')
		b.WriteString(strconv.Itoa(a.Lane))
		b.WriteByte('/')
		b.WriteString(strconv.Itoa(a.FarmLane))
		b.WriteByte('/')
		if a.FarmLaneSet {
			b.WriteByte('1')
		} else {
			b.WriteByte('0')
		}
		b.WriteByte('/')
		b.WriteString(strconv.Itoa(a.BaselineLane))
		b.WriteByte('/')
		b.WriteString(a.Role)
		b.WriteByte('/')
		b.WriteString(a.Reason)
		b.WriteByte('/')
		b.WriteString(strconv.FormatInt(int64(a.ObjectiveID), 10))
		b.WriteByte('/')
		if a.Coverage {
			b.WriteByte('1')
		} else {
			b.WriteByte('0')
		}
		b.WriteByte(';')
	}
	return b.String()
}

func (s *Server) botBaseDefenseTickLocked(b *botBrain, now float64) {
	c, hs := b.c, b.c.huntState
	if hs.attackTarget != 0 {
		if target := hs.mobs[hs.attackTarget]; target != nil && !target.structure &&
			target.enemyOf(c.playerTeam()) && !s.botFarmMayAttackCreepLocked(b, now) {
			s.stopAttackLocked(c, false)
		} else {
			return
		}
	}
	_ = s.botFarmApproachTargetLocked(b, now)
	// A wave in the predictive ring is not yet an ordinary farm target, but a
	// defender must still intercept it before generic XP-centre movement runs.
	if target := s.botBarracksInterceptTargetLocked(b, now); target != nil {
		cx, cy := c.posAtLocked(float32(now))
		if math.Hypot(float64(target.x-cx), float64(target.y-cy)) > botBarracksImmediateRadius*2 {
			// Predictive defense used to walk directly to the front creep. That
			// put the defender on the enemy side of the wave: the rear creep could
			// die outside XP range while the defender was taking avoidable aggro.
			// Intercept the wave from the home-side safe point, exactly like every
			// other farm route. The objective remains the macro anchor; only the
			// local approach geometry changes.
			if px, py, safe := s.botFarmTargetSafePointLocked(b, target, now); safe &&
				s.botBaseFarmPointSafeLocked(b, px, py, now) {
				b.farmTarget = target.id
				b.farmDecision = "defense_intercept"
				b.farmLane = botLaneForCreep(c, target)
				b.farmCatchUp = botFarmCatchUpLocked(b)
				s.botMoveTowardLocked(b, px, py, now)
				return
			}
			// If the map cannot produce a safe point, preserve the old direct
			// intercept as a last-resort structure-defense action.
			b.farmTarget = target.id
			b.farmDecision = "defense_intercept"
			b.farmLane = botLaneForCreep(c, target)
			b.farmCatchUp = botFarmCatchUpLocked(b)
			s.botMoveTowardLocked(b, target.x, target.y, now)
			return
		}
	}
	target := s.botFarmTargetLocked(b, now, botLaneApproachRadius, false)
	if px, py, ok := s.botFarmTargetSafePointLocked(b, target, now); ok {
		cx, cy := c.posAtLocked(float32(now))
		if math.Hypot(float64(px-cx), float64(py-cy)) > dotaXPShareRadius*0.5 &&
			s.botBaseFarmPointSafeLocked(b, px, py, now) && !s.botIncomingPressureLocked(b, now) {
			if target != nil && b.macroAssignment.Role == "defender" &&
				s.botBaseFarmTargetSafeLocked(b, target, now) {
				distance := math.Hypot(float64(target.x-cx), float64(target.y-cy))
				attackReach := hs.effAttackRangeLocked(now) + hs.av.Radius() + target.mob.Radius()
				if distance <= attackReach && distance <= dotaXPShareRadius &&
					s.botFarmMayAttackCreepLocked(b, now) {
					s.startAttackLocked(c, target)
				} else if distance <= dotaXPShareRadius*1.25 {
					// A defender may clear a wave under its own structure, but
					// the last step into the XP envelope still uses the same rear
					// anchor as an ordinary laner. The wider direct intercept is
					// reserved for genuinely distant predictive defense.
					s.botMoveToFarmTargetLocked(b, target, now)
				} else {
					s.botMoveTowardLocked(b, target.x, target.y, now)
				}
				return
			}
			if target != nil && s.botMoveToFarmTargetLocked(b, target, now) {
				return
			}
			if s.botMoveToFarmCoverageLocked(b, now) {
				return
			}
		}
	}
	if target != nil &&
		s.botBaseFarmTargetSafeLocked(b, target, now) {
		if b.macroAssignment.Role == "defender" {
			// The primary defender is already operating inside its own
			// structure's protected ring. Preserve the explicit intercept order
			// here; the XP-safe anchor governs lane/cover bodies, while this role
			// must still clear a wave that has reached the objective.
			cx, cy := c.posAtLocked(float32(now))
			distance := math.Hypot(float64(target.x-cx), float64(target.y-cy))
			attackReach := hs.effAttackRangeLocked(now) + hs.av.Radius() + target.mob.Radius()
			if distance <= attackReach && distance <= dotaXPShareRadius {
				if s.botFarmMayAttackCreepLocked(b, now) {
					s.startAttackLocked(c, target)
				} else {
					s.botMoveToFarmTargetLocked(b, target, now)
				}
			} else if distance <= dotaXPShareRadius*1.25 {
				s.botMoveToFarmTargetLocked(b, target, now)
			} else {
				s.botMoveTowardLocked(b, target.x, target.y, now)
			}
			return
		}
		if s.botMoveToFarmTargetLocked(b, target, now) {
			return
		}
	}
	// A defender must keep the objective as its strategic anchor, but a live wave
	// already inside the local defense radius is free XP. Multi-creep clear comes
	// before a single low-HP target because gold ownership is not the objective.
	if s.botConsiderWaveClearAbilityLocked(b, now) {
		return
	}
	if px, py, ok := s.botFarmTargetSafePointLocked(b, target, now); ok {
		cx, cy := c.posAtLocked(float32(now))
		if math.Hypot(float64(px-cx), float64(py-cy)) > dotaXPShareRadius*0.5 &&
			s.botBaseFarmPointSafeLocked(b, px, py, now) {
			if target != nil && s.botMoveToFarmTargetLocked(b, target, now) {
				return
			}
			if s.botMoveToFarmCoverageLocked(b, now) {
				return
			}
		}
	}
	if target := s.botFindLaneTargetLocked(b, now, botGroupEngageRadius, false); target != nil {
		if s.botMoveToFarmTargetLocked(b, target, now) {
			return
		}
	}
	// Defense is an anchor, not an idle waypoint. If the threatened lane has a
	// nearby safe wave, close on it and convert the defense assignment into XP;
	// only fall back to the barracks when no local farm target is available.
	if target := s.botFarmApproachTargetLocked(b, now); target != nil {
		if s.botMoveToFarmTargetLocked(b, target, now) {
			return
		}
	}
	// The macro premise intentionally survives in a wider release ring than the
	// local farm radius.  In that gap, returning to the already-safe barracks
	// creates a stationary defender: the wave is close enough to keep the base
	// plan alive, but too far away for botFarmApproachTargetLocked to claim it.
	// Intercept the live approaching wave instead.  This keeps the defender's
	// strategic anchor while turning the retained defense into useful movement
	// and XP pressure.
	if objective := c.inst.mobs[b.macroAssignment.ObjectiveID]; objective != nil && !objective.dead {
		s.botMoveTowardLocked(b, objective.x, objective.y, now)
		return
	}
	if altar := botTeamAltarLocked(c.inst, c.playerTeam()); altar != nil && !altar.dead {
		s.botMoveTowardLocked(b, altar.x, altar.y, now)
	}
}

// botBaseFarmTargetSafeLocked allows a defender to step into the XP radius of
// a wave under its own threatened structure. The generic no-dive predicate is
// intentionally conservative for attackers, but it must not strand the only
// local XP body just outside a friendly gun/barracks while enemy heroes are
// absent. A visible enemy hero near the structure still vetoes the step.
func (s *Server) botBaseFarmTargetSafeLocked(b *botBrain, target *mobState, now float64) bool {
	if b == nil || b.c == nil || target == nil {
		return false
	}
	if !s.botEnemyStructureDangerLocked(b.c, target.x, target.y) {
		return true
	}
	objective := b.c.inst.mobs[b.macroAssignment.ObjectiveID]
	if objective == nil || objective.dead || objective.team != b.c.playerTeam() ||
		math.Hypot(float64(target.x-objective.x), float64(target.y-objective.y)) > botMacroBasePressureRadius {
		return false
	}
	return !s.botVisibleEnemyHeroNearObjectiveLocked(b.c.inst, b.c.playerTeam(), objective, now)
}

func (s *Server) botBaseFarmPointSafeLocked(b *botBrain, x, y float32, now float64) bool {
	if b == nil || b.c == nil {
		return false
	}
	if !s.botEnemyStructureDangerLocked(b.c, x, y) {
		return true
	}
	objective := b.c.inst.mobs[b.macroAssignment.ObjectiveID]
	if objective == nil || objective.dead || objective.team != b.c.playerTeam() ||
		math.Hypot(float64(x-objective.x), float64(y-objective.y)) > botMacroBasePressureRadius {
		return false
	}
	return !s.botVisibleEnemyHeroNearObjectiveLocked(b.c.inst, b.c.playerTeam(), objective, now)
}

// botCoverageTickLocked is the narrow behavior for a cover assignment. It keeps
// the bot on the planned lane without turning the role into another group bot;
// local combat still gets first refusal through botTickLocked.
func (s *Server) botCoverageTickLocked(b *botBrain, now float64) {
	c, hs := b.c, b.c.huntState
	if hs.attackTarget != 0 {
		if target := hs.mobs[hs.attackTarget]; target != nil && !target.structure &&
			target.enemyOf(c.playerTeam()) && !s.botFarmMayAttackCreepLocked(b, now) {
			s.stopAttackLocked(c, false)
		} else {
			return
		}
	}
	if botFirstFarmWavePendingLocked(b, now) {
		// Keep the cover body staged on its assigned lane before the first
		// wave exists. Returning to base here loses the race to the first
		// creep after a safe walk or a cancelled teleport.
		lane := botFarmLaneLocked(b)
		if lane >= 0 {
			lx, ly, ok := s.botFarmPrestagePointLocked(b, lane, now)
			if !ok {
				lx, ly = s.botLanePoint(b, now)
			}
			s.botMoveTowardLocked(b, lx, ly, now)
		} else {
			lx, ly := s.botLanePoint(b, now)
			s.botMoveTowardLocked(b, lx, ly, now)
		}
		return
	}
	// A cover assignment is normally allowed to farm its baseline lane. That
	// exception ends when the orchestrator has named a damaged front structure:
	// the support is now needed to convert accumulated pressure, not to clear a
	// nearby creep. The same local readiness gate used by the orchestrator keeps
	// an undamaged, unsupported tower from pulling every cover bot off farm.
	if assignment := b.macroAssignment; assignment.Mode == botMacroCover && assignment.ObjectiveID != 0 {
		objective := c.inst.mobs[assignment.ObjectiveID]
		if objective != nil && !objective.dead && objective.structure && objective.enemyOf(c.playerTeam()) &&
			(assignment.Reason == botMacroReasonFullMobilization || botMacroObjectiveFinishWindowLocked(objective) ||
				s.botObjectiveConversionReadyLocked(c.inst, c.playerTeam(), objective, now)) {
			if s.botMacroObjectiveTickLocked(b, now) {
				return
			}
		}
	}
	if s.botConsiderWaveClearAbilityLocked(b, now) {
		return
	}
	if s.botHoldFarmXPLocked(b, now) {
		return
	}
	previousFarmTarget := b.farmTarget
	target := s.botFarmTargetLocked(b, now, botLaneApproachRadius, false)
	if previousFarmTarget != 0 {
		if urgent := s.botUrgentFarmCoverageTargetLocked(b, now); urgent != nil {
			if s.botMoveToFarmTargetLocked(b, urgent, now) {
				return
			}
		}
	}
	if px, py, ok := s.botFarmTargetSafePointLocked(b, target, now); ok {
		cx, cy := c.posAtLocked(float32(now))
		if math.Hypot(float64(px-cx), float64(py-cy)) > dotaXPShareRadius*0.5 &&
			!s.botEnemyStructureDangerLocked(c, px, py) && !s.botIncomingPressureLocked(b, now) {
			s.botMoveTowardLocked(b, px, py, now)
			return
		}
	}
	if target != nil &&
		!s.botEnemyStructureDangerLocked(c, target.x, target.y) {
		if s.botMoveToFarmTargetLocked(b, target, now) {
			return
		}
	}
	if px, py, ok := s.botFarmTargetSafePointLocked(b, target, now); ok {
		cx, cy := c.posAtLocked(float32(now))
		if math.Hypot(float64(px-cx), float64(py-cy)) > dotaXPShareRadius*0.5 &&
			!s.botEnemyStructureDangerLocked(c, px, py) {
			s.botMoveTowardLocked(b, px, py, now)
			return
		}
	}
	if target := s.botFindLaneTargetLocked(b, now, botLaneEngageRadius, true); target != nil {
		if s.botMoveToFarmTargetLocked(b, target, now) {
			return
		}
	}
	// Cover is a lane-presence role, not a stationary waypoint role. Acquire the
	// current safe wave before falling back to the push point; this closes the gap
	// that left cover bots in lane_move for most of the match.
	if target := s.botFarmApproachTargetLocked(b, now); target != nil {
		if s.botMoveToFarmTargetLocked(b, target, now) {
			return
		}
	}
	if s.botMacroObjectiveTickLocked(b, now) {
		return
	}
	lane := b.macroAssignment.Lane
	if b.farmLane >= 0 && b.farmLane != lane {
		lane = b.farmLane
	}
	if prestageX, prestageY, prestageOK := s.botFarmPrestagePointLocked(b, lane, now); prestageOK &&
		!s.botEnemyStructureDangerLocked(c, prestageX, prestageY) {
		s.botMoveTowardLocked(b, prestageX, prestageY, now)
		return
	}
	px, py := s.botPushPointLocked(b, lane, now)
	s.botMoveTowardLocked(b, px, py, now)
}

// botRecoveryFarmTickLocked keeps a recovering bot as a local XP body when
// retreating to the fountain would abandon an otherwise safe wave. Recovery is
// still authoritative under hero/structure pressure; this only applies when
// the bot can heal or reposition without crossing a live danger boundary.
func (s *Server) botRecoveryFarmTickLocked(b *botBrain, now float64) bool {
	if b == nil || !b.retreating || b.retreatMode != botRetreatModeRecovery ||
		b.c == nil || b.c.huntState == nil || botHPFrac(b.c.huntState, now) <= botRetreatHPFrac ||
		botNearbyEnemyHeroPressureLocked(b, now) > 0 || s.botIncomingPressureLocked(b, now) {
		return false
	}
	urgentTarget := s.botUrgentFarmCoverageTargetLocked(b, now)
	if urgentTarget != nil {
		cx, cy := b.c.posAtLocked(float32(now))
		distance := math.Hypot(float64(urgentTarget.x-cx), float64(urgentTarget.y-cy))
		// A recovering body may finish the last few metres to an uncovered
		// wave when it is still above the hard retreat floor. This is limited
		// to the local hand-off band and never overrides hero/structure danger.
		if distance <= dotaXPShareRadius*1.75 && !s.botEnemyStructureDangerLocked(b.c, urgentTarget.x, urgentTarget.y) {
			if s.botMoveToFarmTargetLocked(b, urgentTarget, now) {
				return true
			}
		}
	}
	if s.botIncomingPressureLocked(b, now) {
		return false
	}
	px, py, ok := s.botFarmSafeAnchorLocked(b, now)
	if !ok || s.botEnemyStructureDangerLocked(b.c, px, py) {
		return false
	}
	cx, cy := b.c.posAtLocked(float32(now))
	distance := math.Hypot(float64(px-cx), float64(py-cy))
	// A recovery body farther away than this is no longer a useful immediate
	// XP guard; it should complete the retreat and let the orchestrator hand the
	// lane to a healthier teammate.
	if distance > botLaneApproachRadius {
		return false
	}
	if target := s.botUrgentFarmCoverageTargetLocked(b, now); target != nil {
		if s.botMoveToFarmTargetLocked(b, target, now) {
			return true
		}
	}
	if target := s.botFindLaneTargetLocked(b, now, botLaneEngageRadius, false); target != nil {
		if s.botMoveToFarmTargetLocked(b, target, now) {
			return true
		}
	}
	if distance > dotaXPShareRadius*0.5 {
		s.botMoveTowardLocked(b, px, py, now)
	}
	return true
}
