package session

// CLANS. A clan is a top-level entity (its own clans table); membership + rank live
// on the hero row (Hero.ClanID / Hero.ClanRole), so the roster is derived by scanning
// heroes -- a hero is in at most one clan. Everything here is authoritative: the client
// pre-checks money/name/tag/permissions but the server re-enforces them and answers
// with the client's Clan.ErrorCode ints. Pending invites + per-invitee cooldowns are
// transient in-memory state (like party/friend requests), not persisted.
//
// Wire protocol (obj="clan"): create, info, remove_user, remove, change_role, invite,
// invite_answer, plus the MPD pushes info_mpd/remove_user_mpd/invite_mpd/
// invite_answer_mpd -- handled in ctrlserver/clan.go.

import (
	"log"
	"regexp"
	"sort"
	"strings"
)

// Clan roles, low->high (Clan.Role in the client). A member's power to invite/kick/
// edit-roles/disband is gated on these.
const (
	ClanRoleWarrior   int32 = 1
	ClanRoleRecruiter int32 = 2
	ClanRoleCommander int32 = 3
	ClanRoleDeputy    int32 = 4
	ClanRoleHead      int32 = 5
)

// Clan.RemoveReason, sent on clan|remove_user_mpd so the recipient knows whether they
// were kicked/left (a single user removed) or the whole clan was disbanded.
const (
	ClanRemoveReasonUser int32 = 1
	ClanRemoveReasonClan int32 = 2
)

// Clan.ErrorCode values the client switches on. ClanOK (0) means success.
const (
	ClanOK                        int32 = 0
	ClanErrSystem                 int32 = 7000
	ClanErrNotEnoughMoney         int32 = 7010
	ClanErrBadTag                 int32 = 7011
	ClanErrBadName                int32 = 7012
	ClanErrTagExist               int32 = 7013
	ClanErrNameExist              int32 = 7014
	ClanErrWrongParams            int32 = 7015
	ClanErrYouNotMember           int32 = 7017
	ClanErrUserNotMember          int32 = 7018
	ClanErrForbidden              int32 = 7019
	ClanErrWrongRole              int32 = 7020
	ClanErrUserAlreadyMember      int32 = 7021
	ClanErrAlreadyInvited         int32 = 7022
	ClanErrNotEnoughSpace         int32 = 7023
	ClanErrNotInvited             int32 = 7024
	ClanErrUserOffline            int32 = 7025
	ClanErrNoHead                 int32 = 7026
	ClanErrTwoHead                int32 = 7027
	ClanErrHeadNeeded             int32 = 7028
	ClanErrUserInBattle           int32 = 7029
	ClanErrCantDeleteClanInCastle int32 = 7030
	ClanErrCantDeleteUserInCastle int32 = 7031
	ClanErrCantInviteTimeout      int32 = 7032
	ClanErrPlayerIgnore           int32 = 7033
)

// Clan tuning. The client shows a hardcoded 200000-gold create price; the server holds
// the canonical value and enforces it. Member cap is flat for v1 (a level curve is
// deferred). Invite cooldown rate-limits re-inviting the same player.
const (
	ClanCreatePrice       int32 = 200000
	ClanMaxMembers        int   = 50
	clanInviteCooldownSec int64 = 60
)

// Clan is the persistent clan header (roster is derived from heroes).
type Clan struct {
	ID        int32
	Name      string
	Tag       string
	Level     int32
	Rating    int32
	HeadID    int32
	CreatedAt int64
}

// clanInvite is one pending invitation held for an invitee until they answer.
type clanInvite struct {
	ClanID    int32
	InviterID int32
	CreatedAt int64
}

// ClanMemberView is one roster entry (location is filled by the caller from live
// presence, not stored here).
type ClanMemberView struct {
	UserID int32
	Nick   string
	Role   int32
}

// ClanView is a read-only snapshot of a clan header + roster for the clan window.
type ClanView struct {
	ID      int32
	Name    string
	Tag     string
	Level   int32
	Rating  int32
	Members []ClanMemberView
}

// Name/tag rules (Clan.CheckName): trimmed length, single alphabet (Latin XOR
// Cyrillic) plus digits/space/underscore. The 2011 client's regex also accidentally
// allowed commas; we drop that authoring artifact.
var (
	clanLatinRe    = regexp.MustCompile(`^[A-Za-z0-9 _]+$`)
	clanCyrillicRe = regexp.MustCompile(`^[А-Яа-яЁё0-9 _]+$`)
)

func clanNameValid(s string) bool {
	n := len([]rune(strings.TrimSpace(s)))
	return n >= 4 && n <= 30 && clanCharsetOK(s)
}

func clanTagValid(s string) bool {
	n := len([]rune(strings.TrimSpace(s)))
	return n >= 2 && n <= 4 && clanCharsetOK(s)
}

func clanCharsetOK(s string) bool {
	s = strings.TrimSpace(s)
	return clanLatinRe.MatchString(s) || clanCyrillicRe.MatchString(s)
}

// CreateClan validates + charges the founder, creates the clan, and makes the founder
// its HEAD. Returns the new clan id and ClanOK, or 0 and a Clan.ErrorCode.
func (s *Store) CreateClan(founderID int32, name, tag string) (int32, int32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	u := s.usersByID[founderID]
	if u == nil || u.Hero == nil {
		return 0, ClanErrSystem
	}
	if u.Hero.ClanID != 0 {
		return 0, ClanErrUserAlreadyMember
	}
	name = strings.TrimSpace(name)
	tag = strings.TrimSpace(tag)
	if !clanNameValid(name) {
		return 0, ClanErrBadName
	}
	if !clanTagValid(tag) {
		return 0, ClanErrBadTag
	}
	for _, c := range s.clansByID {
		if strings.EqualFold(c.Name, name) {
			return 0, ClanErrNameExist
		}
		if strings.EqualFold(c.Tag, tag) {
			return 0, ClanErrTagExist
		}
	}
	if u.Hero.Money < ClanCreatePrice {
		return 0, ClanErrNotEnoughMoney
	}
	u.Hero.Money -= ClanCreatePrice
	id := s.nextClanID
	s.nextClanID++
	c := &Clan{ID: id, Name: name, Tag: tag, Level: 1, Rating: 0, HeadID: founderID, CreatedAt: nowUnix()}
	s.clansByID[id] = c
	u.Hero.ClanID = id
	u.Hero.ClanRole = ClanRoleHead
	s.persistClanLocked(c)
	s.persistNextClanIDLocked()
	s.saveUserLocked(u)
	return id, ClanOK
}

// HeroClanInfo returns a hero's clan id + tag for the game_info / hero-data payloads,
// or (0, "") when the hero has no clan. Cheap: a single map lookup.
func (s *Store) HeroClanInfo(userID int32) (clanID int32, tag string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	u := s.usersByID[userID]
	if u == nil || u.Hero == nil || u.Hero.ClanID == 0 {
		return 0, ""
	}
	if c := s.clansByID[u.Hero.ClanID]; c != nil {
		return c.ID, c.Tag
	}
	return 0, ""
}

// ClanByID / ClanByTag / ClanOfUser return a roster snapshot for the clan window.
func (s *Store) ClanByID(id int32) (*ClanView, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if c := s.clansByID[id]; c != nil {
		return s.clanViewLocked(c), true
	}
	return nil, false
}

func (s *Store) ClanByTag(tag string) (*ClanView, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, c := range s.clansByID {
		if strings.EqualFold(c.Tag, tag) {
			return s.clanViewLocked(c), true
		}
	}
	return nil, false
}

func (s *Store) ClanOfUser(userID int32) (*ClanView, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	u := s.usersByID[userID]
	if u == nil || u.Hero == nil || u.Hero.ClanID == 0 {
		return nil, false
	}
	if c := s.clansByID[u.Hero.ClanID]; c != nil {
		return s.clanViewLocked(c), true
	}
	return nil, false
}

// clanViewLocked builds a ClanView (header + sorted roster). Caller holds s.mu.
func (s *Store) clanViewLocked(c *Clan) *ClanView {
	return &ClanView{ID: c.ID, Name: c.Name, Tag: c.Tag, Level: c.Level, Rating: c.Rating, Members: s.clanRosterLocked(c.ID)}
}

// clanRosterLocked returns every hero in a clan, HEAD first then by role desc, then nick.
func (s *Store) clanRosterLocked(clanID int32) []ClanMemberView {
	var out []ClanMemberView
	for _, u := range s.usersByID {
		if u.Hero != nil && u.Hero.ClanID == clanID {
			out = append(out, ClanMemberView{UserID: u.ID, Nick: u.Username, Role: u.Hero.ClanRole})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Role != out[j].Role {
			return out[i].Role > out[j].Role
		}
		return out[i].Nick < out[j].Nick
	})
	return out
}

func (s *Store) clanMemberCountLocked(clanID int32) int {
	n := 0
	for _, u := range s.usersByID {
		if u.Hero != nil && u.Hero.ClanID == clanID {
			n++
		}
	}
	return n
}

// RemoveClanMember kicks targetID (or, when actorID==targetID, leaves). Enforces
// rank-superiority for a kick; HEAD may not leave (must transfer or disband first).
// Returns ClanOK on success or a Clan.ErrorCode.
func (s *Store) RemoveClanMember(actorID, targetID int32) int32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	actor := s.usersByID[actorID]
	if actor == nil || actor.Hero == nil || actor.Hero.ClanID == 0 {
		return ClanErrYouNotMember
	}
	target := s.usersByID[targetID]
	if target == nil || target.Hero == nil || target.Hero.ClanID != actor.Hero.ClanID {
		return ClanErrUserNotMember
	}
	if actorID == targetID {
		if actor.Hero.ClanRole == ClanRoleHead {
			return ClanErrHeadNeeded
		}
	} else {
		// Kick: at least RECRUITER, and must strictly outrank the target.
		if actor.Hero.ClanRole < ClanRoleRecruiter || actor.Hero.ClanRole <= target.Hero.ClanRole {
			return ClanErrForbidden
		}
	}
	target.Hero.ClanID = 0
	target.Hero.ClanRole = 0
	s.saveUserLocked(target)
	return ClanOK
}

// DisbandClan deletes the whole clan (HEAD only). Returns ClanOK + every member's id
// (so the caller can push the disband notice), or a Clan.ErrorCode + nil.
func (s *Store) DisbandClan(actorID int32) (int32, []int32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	actor := s.usersByID[actorID]
	if actor == nil || actor.Hero == nil || actor.Hero.ClanID == 0 {
		return ClanErrYouNotMember, nil
	}
	if actor.Hero.ClanRole != ClanRoleHead {
		return ClanErrForbidden, nil
	}
	clanID := actor.Hero.ClanID
	var members []int32
	var affected []*User
	for _, u := range s.usersByID {
		if u.Hero != nil && u.Hero.ClanID == clanID {
			members = append(members, u.ID)
			u.Hero.ClanID = 0
			u.Hero.ClanRole = 0
			affected = append(affected, u)
		}
	}
	delete(s.clansByID, clanID)
	s.deleteClanLocked(clanID)
	for _, u := range affected {
		s.saveUserLocked(u)
	}
	return ClanOK, members
}

// ChangeClanRoles applies a batch of userID->newRole edits atomically (all-or-nothing).
// DEPUTY+ required; every edit must be to a lower-ranked member and below the actor's
// own rank -- except promoting someone to HEAD, which requires the actor to BE the HEAD
// and transfers headship (the old HEAD auto-demotes to DEPUTY, keeping exactly one HEAD).
func (s *Store) ChangeClanRoles(actorID int32, roles map[int32]int32) int32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	actor := s.usersByID[actorID]
	if actor == nil || actor.Hero == nil || actor.Hero.ClanID == 0 {
		return ClanErrYouNotMember
	}
	if actor.Hero.ClanRole < ClanRoleDeputy {
		return ClanErrForbidden
	}
	clanID := actor.Hero.ClanID

	headAssignments := 0
	var newHead int32
	for uid, newRole := range roles {
		if newRole < ClanRoleWarrior || newRole > ClanRoleHead {
			return ClanErrWrongRole
		}
		t := s.usersByID[uid]
		if t == nil || t.Hero == nil || t.Hero.ClanID != clanID {
			return ClanErrUserNotMember
		}
		if newRole == ClanRoleHead {
			headAssignments++
			newHead = uid
			continue
		}
		// Non-head edit: outrank the target's current role, and assign below own rank.
		if actor.Hero.ClanRole <= t.Hero.ClanRole || newRole >= actor.Hero.ClanRole {
			return ClanErrForbidden
		}
	}
	if headAssignments > 0 {
		if actor.Hero.ClanRole != ClanRoleHead {
			return ClanErrForbidden
		}
		if headAssignments > 1 {
			return ClanErrTwoHead
		}
		if newHead == actorID {
			return ClanErrWrongRole
		}
	}

	// Apply (validated above).
	affected := map[int32]*User{}
	for uid, newRole := range roles {
		t := s.usersByID[uid]
		t.Hero.ClanRole = newRole
		affected[uid] = t
	}
	if headAssignments == 1 {
		actor.Hero.ClanRole = ClanRoleDeputy
		affected[actorID] = actor
		if c := s.clansByID[clanID]; c != nil {
			c.HeadID = newHead
			s.persistClanLocked(c)
		}
	}
	for _, u := range affected {
		s.saveUserLocked(u)
	}
	return ClanOK
}

// RecordClanInvite validates that inviterID (RECRUITER+ in a clan) may invite targetID,
// registers the pending invite, and applies the per-invitee cooldown. Returns the
// inviting clan's name (for the push), the seconds-remaining (only meaningful with
// ClanErrCantInviteTimeout), and a Clan.ErrorCode.
func (s *Store) RecordClanInvite(inviterID, targetID int32) (clanName string, cooldownRemain int32, errCode int32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	inviter := s.usersByID[inviterID]
	if inviter == nil || inviter.Hero == nil || inviter.Hero.ClanID == 0 {
		return "", 0, ClanErrYouNotMember
	}
	if inviter.Hero.ClanRole < ClanRoleRecruiter {
		return "", 0, ClanErrForbidden
	}
	clanID := inviter.Hero.ClanID
	c := s.clansByID[clanID]
	if c == nil {
		return "", 0, ClanErrSystem
	}
	target := s.usersByID[targetID]
	if target == nil || target.Hero == nil {
		return "", 0, ClanErrWrongParams
	}
	if target.Hero.ClanID != 0 {
		return "", 0, ClanErrUserAlreadyMember
	}
	if containsID(target.Ignores, inviterID) {
		return "", 0, ClanErrPlayerIgnore
	}
	if _, exists := s.clanInvites[targetID]; exists {
		return "", 0, ClanErrAlreadyInvited
	}
	if s.clanMemberCountLocked(clanID) >= ClanMaxMembers {
		return "", 0, ClanErrNotEnoughSpace
	}
	now := nowUnix()
	if cd, ok := s.clanInviteCD[targetID]; ok && now < cd {
		return "", int32(cd - now), ClanErrCantInviteTimeout
	}
	s.clanInvites[targetID] = clanInvite{ClanID: clanID, InviterID: inviterID, CreatedAt: now}
	s.clanInviteCD[targetID] = now + clanInviteCooldownSec
	return c.Name, 0, ClanOK
}

// AnswerClanInvite resolves the invitee's pending invite. On accept the invitee joins
// as a WARRIOR. Returns the clan id, the inviter id (to notify), whether the invitee
// actually joined, and a Clan.ErrorCode (0 on success incl. a clean decline).
func (s *Store) AnswerClanInvite(targetID int32, accept bool) (clanID, inviterID int32, joined bool, errCode int32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	inv, ok := s.clanInvites[targetID]
	if !ok {
		return 0, 0, false, ClanErrNotInvited
	}
	delete(s.clanInvites, targetID)
	if !accept {
		return inv.ClanID, inv.InviterID, false, ClanOK
	}
	target := s.usersByID[targetID]
	if target == nil || target.Hero == nil {
		return inv.ClanID, inv.InviterID, false, ClanErrSystem
	}
	if target.Hero.ClanID != 0 {
		return inv.ClanID, inv.InviterID, false, ClanErrUserAlreadyMember
	}
	if s.clansByID[inv.ClanID] == nil {
		return inv.ClanID, inv.InviterID, false, ClanErrSystem
	}
	if s.clanMemberCountLocked(inv.ClanID) >= ClanMaxMembers {
		return inv.ClanID, inv.InviterID, false, ClanErrNotEnoughSpace
	}
	target.Hero.ClanID = inv.ClanID
	target.Hero.ClanRole = ClanRoleWarrior
	s.saveUserLocked(target)
	return inv.ClanID, inv.InviterID, true, ClanOK
}

// ---- persistence ----

// metaNextClanID holds the clan-id high-water mark so a disbanded clan's id is never reused.
const metaNextClanID = "next_clan_id"

// loadClansLocked reconstructs the clan headers from the database (roster is derived
// from the already-loaded heroes). Called from loadAllLocked at open.
func (s *Store) loadClansLocked() error {
	if s.db == nil {
		return nil
	}
	rows, err := s.db.Query(`SELECT id, name, tag, level, rating, head_user_id, created_at FROM clans`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		c := &Clan{}
		if err := rows.Scan(&c.ID, &c.Name, &c.Tag, &c.Level, &c.Rating, &c.HeadID, &c.CreatedAt); err != nil {
			return err
		}
		s.clansByID[c.ID] = c
		if c.ID >= s.nextClanID {
			s.nextClanID = c.ID + 1
		}
	}
	if err := rows.Err(); err != nil {
		return err
	}
	var stored int32
	if err := s.db.QueryRow(`SELECT value FROM meta WHERE key=?`, metaNextClanID).Scan(&stored); err == nil {
		if stored > s.nextClanID {
			s.nextClanID = stored
		}
	}
	return nil
}

func (s *Store) persistClanLocked(c *Clan) {
	if s.db == nil {
		return
	}
	if _, err := s.db.Exec(
		`INSERT INTO clans(id, name, tag, level, rating, head_user_id, created_at)
		 VALUES(?, ?, ?, ?, ?, ?, ?)
		 ON CONFLICT(id) DO UPDATE SET
		   name=excluded.name, tag=excluded.tag, level=excluded.level,
		   rating=excluded.rating, head_user_id=excluded.head_user_id`,
		c.ID, c.Name, c.Tag, c.Level, c.Rating, c.HeadID, c.CreatedAt); err != nil {
		log.Printf("session: persist clan %d failed: %v", c.ID, err)
	}
}

func (s *Store) deleteClanLocked(id int32) {
	if s.db == nil {
		return
	}
	if _, err := s.db.Exec(`DELETE FROM clans WHERE id=?`, id); err != nil {
		log.Printf("session: delete clan %d failed: %v", id, err)
	}
}

func (s *Store) persistNextClanIDLocked() {
	if s.db == nil {
		return
	}
	if _, err := s.db.Exec(
		`INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value`,
		metaNextClanID, s.nextClanID); err != nil {
		log.Printf("session: persist next_clan_id failed: %v", err)
	}
}
