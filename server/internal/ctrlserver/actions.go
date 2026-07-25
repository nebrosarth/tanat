package ctrlserver

import (
	"log"
	"math/rand"
	"strconv"
	"time"

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/gamedata"
)

// common|action is the city "use / open" verb (HeroSender.DoAction sends
// {action_id, artifact_id[, check]}, where artifact_id is the bag row id the client
// currently shows). It dispatches on the addressed item's family: FORTUNE CHESTS open (roll a
// server-authoritative reward + credit it), and RUNES / hero-wide ELIXIRS are consumed to
// start a timed GLOBAL buff. Rewards/effects arrive via the usual pushes (user|money,
// user|update_bag_mpd, user|update_global_buffs_mpd) -- never in this reply; the ack is bare.
// Mastery elixirs / totems carry no action, so this acks harmlessly for them.
func (s *Server) handleCommonAction(req ctrlproto.Request, resp *ctrlproto.Response) {
	resp.Ack("common", "action") // always ack; results arrive via push
	u := s.userFromSession(req)
	if u == nil || u.Hero == nil {
		return
	}
	artifactID, ok := req.Params.GetInt("artifact_id")
	if !ok {
		return
	}
	article, ok := s.Store.BagArticleAtIndex(u.ID, artifactID)
	if !ok {
		return
	}
	if _, isChest := gamedata.ChestByArticle(article); isChest {
		s.openChest(u.ID, artifactID, article, resp)
		return
	}
	if dur, ok := buffDuration(article); ok {
		s.useBuffItem(u.ID, artifactID, article, dur, resp)
		return
	}
	// Not an openable/usable item -- nothing to do (already acked).
}

// buffDuration returns the buff duration (seconds) an article grants if it is a usable rune or
// a hero-wide elixir, else ok=false.
func buffDuration(article int32) (int64, bool) {
	if r, ok := gamedata.RuneByArticle(article); ok {
		return r.Duration, true
	}
	if e, ok := gamedata.ElixirByArticle(article); ok && e.Kind == gamedata.KindElixir {
		return e.Duration, true
	}
	return 0, false
}

// openChest consumes one chest and credits a server-rolled reward (coins + optional bonus
// item), pushing the coin balance and the chest-stack delta.
func (s *Server) openChest(userID, rowID, article int32, resp *ctrlproto.Response) {
	chest, _ := gamedata.ChestByArticle(article)
	newCount, ok := s.Store.ConsumeOneBagArticle(userID, article)
	if !ok {
		return // already gone (raced double click)
	}
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	rew := gamedata.RollChest(chest, rng)

	var money, diamonds int32
	if rew.Coins > 0 {
		money, diamonds, _ = s.Store.AddHeroMoney(userID, rew.Coins)
	}
	for _, it := range rew.Items {
		s.Store.AddBagItem(userID, it.Article, it.Count)
	}
	log.Printf("chest: hero %d opened chest %d -> %d coins + %d bonus item(s)", userID, article, rew.Coins, len(rew.Items))

	if rew.Coins > 0 {
		resp.Add("user", "money", amf.NewArray().Set("money", money).Set("money_d", diamonds))
	}
	rows := amf.NewArray()
	rows.Add(bagRow(rowID, article, newCount))
	s.pushBagUpdate(userID, rows)
}

// useBuffItem consumes one rune/elixir and starts (or refreshes) its timed global buff,
// pushing the bag delta and the updated global-buffs set.
func (s *Server) useBuffItem(userID, rowID, article int32, durationSec int64, resp *ctrlproto.Response) {
	// A hero may hold only ONE rune per category (attack_power / health / mana) at a time, and
	// while it is live it CANNOT be re-used -- not by a different rune of the same category (no
	// stacking) NOR by the same rune again (no refresh / no re-consume). Once activated, a rune
	// simply keeps acting until it expires (in a match its stats are folded permanently at entry,
	// so it lasts to the end of the match regardless of the city timer). A refused use consumes
	// nothing. Elixirs have their own categories and don't resolve via RuneByArticle -> unaffected.
	if newRune, isRune := gamedata.RuneByArticle(article); isRune {
		for _, b := range s.Store.HeroActiveBuffs(userID) {
			if other, ok := gamedata.RuneByArticle(b.ArticleID); ok && other.Stat == newRune.Stat {
				log.Printf("buff: hero %d refused rune %d -- category %q already active (article %d)",
					userID, article, newRune.Stat, b.ArticleID)
				return // a rune of this category (incl. the same one) is active -> forbid re-use
			}
		}
	}
	newCount, ok := s.Store.ConsumeOneBagArticle(userID, article)
	if !ok {
		return
	}
	if _, ok := s.Store.AddGlobalBuff(userID, article, durationSec); !ok {
		// Couldn't apply (shouldn't happen for a live hero) -- refund the consumed item.
		s.Store.AddBagItem(userID, article, 1)
		return
	}
	log.Printf("buff: hero %d used item %d -> global buff for %ds", userID, article, durationSec)
	rows := amf.NewArray()
	rows.Add(bagRow(rowID, article, newCount))
	s.pushBagUpdate(userID, rows)
	s.pushGlobalBuffs(userID)
}

// handleGetGlobalBuffs answers user|get_global_buffs with the hero's active buffs as
// {buffs:{"<article>": seconds-remaining}} (GlobalBuffsUpdateMpdArgParser reads buffs.*).
func (s *Server) handleGetGlobalBuffs(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	buffs := amf.NewArray()
	if u != nil && u.Hero != nil {
		now := time.Now().Unix()
		for _, b := range s.Store.HeroActiveBuffs(u.ID) {
			if rem := b.ExpiresUnix - now; rem > 0 {
				buffs.Set(strconv.Itoa(int(b.ArticleID)), int32(rem))
			}
		}
	}
	resp.Add("user", "get_global_buffs", amf.NewArray().Set("buffs", buffs))
}

// pushGlobalBuffs delivers the hero's current active buffs over MPD as
// user|update_global_buffs_mpd {buffs:{"<article>": seconds-remaining}}. No-op without a live
// MPD socket; the next user|get_global_buffs re-syncs.
func (s *Server) pushGlobalBuffs(userID int32) {
	if s.MPD == nil {
		return
	}
	now := time.Now().Unix()
	buffs := amf.NewArray()
	for _, b := range s.Store.HeroActiveBuffs(userID) {
		if rem := b.ExpiresUnix - now; rem > 0 {
			buffs.Set(strconv.Itoa(int(b.ArticleID)), int32(rem))
		}
	}
	s.MPD.Push(userID, "user|update_global_buffs", amf.NewArray().Set("buffs", buffs))
}
