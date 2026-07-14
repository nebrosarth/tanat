package ctrlserver

import (
	"strconv"

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/session"
)

// This file implements the Ctrl-channel lobby data the central-square UI reads
// after entry: money, hero appearance/level/stats, the permanently-equipped
// "hero set" items and the bag. Wire shapes are taken from the client's
// HeroMoneyArgParser / HeroGameInfoArgParser / HeroDataListArgParser /
// FullHeroInfoArgParser / UserBagArgParser.

// handleUserMoney answers user|money -> {money, money_d} (HeroMoneyArgParser).
func (s *Server) handleUserMoney(req ctrlproto.Request, resp *ctrlproto.Response) {
	money, diamonds := int32(0), int32(0)
	if h := s.heroFromSession(req); h != nil {
		money, diamonds = h.Money, h.DiamondMoney
	}
	resp.Add("user", "money", amf.NewArray().
		Set("money", money).
		Set("money_d", diamonds))
}

// handleUserBag answers user|bag -> {user_money, bag:[...]} (UserBagArgParser).
// The bag (consumable inventory) is empty for now.
func (s *Server) handleUserBag(req ctrlproto.Request, resp *ctrlproto.Response) {
	money := int32(0)
	if h := s.heroFromSession(req); h != nil {
		money = h.Money
	}
	resp.Add("user", "bag", amf.NewArray().
		Set("user_money", money).
		Set("bag", amf.NewArray()))
}

// handleUserGameInfo answers user|game_info; ParseGameInfo reads the fields
// straight off the packet arguments.
func (s *Server) handleUserGameInfo(req ctrlproto.Request, resp *ctrlproto.Response) {
	resp.Add("user", "game_info", s.gameInfoFields(s.heroFromSession(req)))
}

// handleHeroGetDataList answers hero|get_data_list -> {data:{"<id>":{...}}}.
func (s *Server) handleHeroGetDataList(req ctrlproto.Request, resp *ctrlproto.Response) {
	resp.Add("hero", "get_data_list", amf.NewArray().
		Set("data", heroDataMapOf(s.heroFromSession(req))))
}

// handleFullHeroInfo answers user|full_hero_info -> {visual_data, hero_data,
// nick, last_visit} (FullHeroInfoArgParser). visual_data uses the same map shape
// as hero|get_data_list; hero_data uses the game-info shape.
func (s *Server) handleFullHeroInfo(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	var h *session.Hero
	nick := ""
	if u != nil {
		h = u.Hero
		nick = u.Username
	}
	resp.Add("user", "full_hero_info", amf.NewArray().
		Set("visual_data", heroDataMapOf(h)).
		Set("hero_data", s.gameInfoFields(h)).
		Set("nick", nick).
		Set("last_visit", int32(0)))
}

// gameInfoFields builds the HeroGameInfo body ({user_id, level, exp, next_exp,
// stats, clan_info, buffs}). Stats default to 0; no clan; no buffs.
func (s *Server) gameInfoFields(h *session.Hero) *amf.MixedArray {
	id, level, exp, next := int32(-1), int32(1), int32(0), int32(100)
	if h != nil {
		id, level, exp, next = h.ID, h.Level, h.Exp, h.NextExp
	}
	stats := amf.NewArray().
		Set("HERO_RATING", int32(0)).Set("ASSISTS", int32(0)).
		Set("AVATARKILLS", int32(0)).Set("CREEPKILLS", int32(0)).
		Set("DEATHS", int32(0)).Set("FIGHTS", int32(0)).
		Set("WINS", int32(0)).Set("LOSE", int32(0)).
		Set("FIGHTS_LEAVING", int32(0)).Set("HONOR", int32(0))
	return amf.NewArray().
		Set("user_id", id).
		Set("level", level).
		Set("exp", exp).
		Set("next_exp", next).
		Set("stats", stats).
		Set("clan_info", amf.NewArray().Set("id", int32(-1)).Set("tag", "")).
		Set("buffs", amf.NewArray())
}

// heroDataMapOf builds the {"<heroId>": {load, dressed_items, clan_info,
// user_info}} map the client parses in HeroDataListArgParser.ParseData.
func heroDataMapOf(h *session.Hero) *amf.MixedArray {
	m := amf.NewArray()
	if h == nil {
		return m
	}
	entry := amf.NewArray().
		Set("load", heroViewArray(h)).
		Set("dressed_items", dressedItemsArray(h)).
		Set("clan_info", amf.NewArray().Set("id", int32(-1)).Set("tag", "")).
		Set("user_info", amf.NewArray().
			Set("level", h.Level).
			Set("exp", h.Exp).
			Set("next_exp", h.NextExp).
			Set("rating", int32(0)))
	m.Set(strconv.Itoa(int(h.ID)), entry)
	return m
}

// heroViewArray is the hero appearance sub-array ("load").
func heroViewArray(h *session.Hero) *amf.MixedArray {
	return amf.NewArray().
		Set("race", h.Race).
		Set("gender", h.Gender).
		Set("face", h.Face).
		Set("hair", h.Hair).
		Set("dist_mark", h.DistMark).
		Set("skin_color", h.SkinColor).
		Set("hair_color", h.HairColor)
}

// dressedItemsArray is the dense list of permanently-equipped hero-set items.
func dressedItemsArray(h *session.Hero) *amf.MixedArray {
	arr := amf.NewArray()
	for _, it := range h.Dressed {
		arr.Add(amf.NewArray().
			Set("id", it.ID).
			Set("artikul_id", it.ArticleID).
			Set("cnt", it.Count).
			Set("slot", it.Slot))
	}
	return arr
}

// ---- Phase 3: lobby services (empty-but-valid so the menus don't error) ----

// handleStoreList answers store|list -> {store:{}} (empty shop for now).
func (s *Server) handleStoreList(req ctrlproto.Request, resp *ctrlproto.Response) {
	resp.Add("store", "list", amf.NewArray().Set("store", amf.NewArray()))
}

// handleCastleList answers castle|list. CastleListArgParser requires "castles".
func (s *Server) handleCastleList(req ctrlproto.Request, resp *ctrlproto.Response) {
	resp.Add("castle", "list", amf.NewArray().Set("castles", amf.NewArray()))
}

// handleGroupList answers user|group_list -> {users:[], leader:-1} (solo).
func (s *Server) handleGroupList(req ctrlproto.Request, resp *ctrlproto.Response) {
	resp.Add("user", "group_list", amf.NewArray().
		Set("users", amf.NewArray()).
		Set("leader", int32(-1)))
}

// handleCanReconnect answers common|can_reconnect with {answer:false} so the
// client (which sends this at startup when it has a saved session file) drops
// the stale session and does a clean fresh login. We don't restore battle state
// across restarts, so reconnect is intentionally declined.
func (s *Server) handleCanReconnect(req ctrlproto.Request, resp *ctrlproto.Response) {
	resp.Add("common", "can_reconnect", amf.NewArray().
		Set("answer", false).
		Set("timer", float64(0)))
}

// heroFromSession returns the session user's hero, or nil.
func (s *Server) heroFromSession(req ctrlproto.Request) *session.Hero {
	if u := s.userFromSession(req); u != nil {
		return u.Hero
	}
	return nil
}
