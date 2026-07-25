package session

// Per-hero GLOBAL BUFFS -- the store side of the timed hero-wide effects that runes and
// elixirs start (a rune's +stat for 2h, an elixir's xp/money/drop multiplier for days). A
// buff is keyed by the ARTICLE that granted it (so re-using the same rune/elixir REFRESHES
// rather than stacks) and carries an absolute expiry. The effect itself (what a given article
// does) lives in gamedata; this file only owns the mutable per-hero {article -> expiry} state
// and persists it, so a multi-day elixir survives relogs. It mirrors the client's
// GlobalBuffsUpdateMpdArg ({buff_id: expiry}).

// GlobalBuff is one active timed buff: the granting article and its unix expiry.
type GlobalBuff struct {
	ArticleID   int32
	ExpiresUnix int64
}

// AddGlobalBuff starts (or refreshes) the buff granted by articleID for durationSec seconds
// from now, returning the new absolute expiry. Re-using the same article REPLACES its timer
// (no stacking). ok=false if the account has no hero or the duration is non-positive.
func (s *Store) AddGlobalBuff(userID, articleID int32, durationSec int64) (expiresUnix int64, ok bool) {
	if durationSec <= 0 {
		return 0, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	u, found := s.usersByID[userID]
	if !found || u.Hero == nil {
		return 0, false
	}
	h := u.Hero
	expiresUnix = nowUnix() + durationSec
	for i := range h.Buffs {
		if h.Buffs[i].ArticleID == articleID {
			h.Buffs[i].ExpiresUnix = expiresUnix
			s.saveUserLocked(u)
			return expiresUnix, true
		}
	}
	h.Buffs = append(h.Buffs, GlobalBuff{ArticleID: articleID, ExpiresUnix: expiresUnix})
	s.saveUserLocked(u)
	return expiresUnix, true
}

// HeroActiveBuffs returns a copy of the hero's still-active buffs (expired ones dropped
// lazily, like HeroQuests drops elapsed cooldowns). nil if no account/hero. The Battle server
// reads this at match entry to fold rune stats / elixir multipliers; the Ctrl channel reads it
// for user|get_global_buffs.
func (s *Store) HeroActiveBuffs(userID int32) []GlobalBuff {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, found := s.usersByID[userID]
	if !found || u.Hero == nil {
		return nil
	}
	now := nowUnix()
	out := make([]GlobalBuff, 0, len(u.Hero.Buffs))
	for _, b := range u.Hero.Buffs {
		if b.ExpiresUnix > now {
			out = append(out, b)
		}
	}
	return out
}
