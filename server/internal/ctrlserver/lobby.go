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
// Each bag entry is {id, artikul_id, cnt, used} -- the same shape as a
// permanently-dressed hero-set item (DressedItem), minus the equip slot. The
// per-entry "id" is synthesized here (1, 2, 3...) since only the article id +
// count are persisted (session.BagItem); it only needs to be unique within one
// response, not stable across requests.
func (s *Server) handleUserBag(req ctrlproto.Request, resp *ctrlproto.Response) {
	money := int32(0)
	bag := amf.NewArray()
	if h := s.heroFromSession(req); h != nil {
		money = h.Money
		for i, bi := range h.Bag {
			bag.Add(amf.NewArray().
				Set("id", int32(i+1)).
				Set("artikul_id", bi.ArticleID).
				Set("cnt", bi.Count).
				Set("used", int32(0)))
		}
		// Unequipped wearable gear shares the one owned inventory with potions, but as
		// discrete instances with stable ids (>= heroItemInstanceBase, so they never
		// collide with the 1-based potion rows above). These are what user|dress and
		// store|sell address by id.
		for _, ow := range h.Owned {
			bag.Add(bagRow(ow.ID, ow.ArticleID, 1))
		}
	}
	resp.Add("user", "bag", amf.NewArray().
		Set("user_money", money).
		Set("bag", bag))
}

// handleUserGameInfo answers user|game_info; ParseGameInfo reads the fields
// straight off the packet arguments. The client fires this both for its own hero
// (no user_id / user_id == self) and, in the shared central square, for every OTHER
// occupant's hero id (BindPlayerAvatar auto-requests it when a "Hero"-prefab avatar
// is bound). Honor the requested user_id so another player's level/rating resolve;
// fall back to the session hero (self path, unchanged) when it is absent/unknown.
func (s *Server) handleUserGameInfo(req ctrlproto.Request, resp *ctrlproto.Response) {
	h := s.heroFromSession(req)
	if id, ok := req.Params.GetInt("user_id"); ok {
		if u, ok := s.Store.ByID(id); ok && u.Hero != nil {
			h = u.Hero
		}
	}
	resp.Add("user", "game_info", s.gameInfoFields(h))
}

// handleHeroGetDataList answers hero|get_data_list -> {data:{"<id>":{...}}}. The
// client sends an "id" array; in the central square that array holds OTHER
// occupants' hero ids (auto-requested on avatar bind, since the "Hero" prefab drives
// SetHeroView from this response). Answer for every requested id looked up in the
// store, so other players render as real customized heroes rather than bodiless
// movers; fall back to the session hero (self path, unchanged) when no id resolves.
func (s *Server) handleHeroGetDataList(req ctrlproto.Request, resp *ctrlproto.Response) {
	data := amf.NewArray()
	for _, h := range s.requestedHeroes(req) {
		addHeroData(data, h)
	}
	if len(data.Assoc) == 0 {
		addHeroData(data, s.heroFromSession(req))
	}
	resp.Add("hero", "get_data_list", amf.NewArray().Set("data", data))
}

// requestedHeroes resolves the heroes named by the request's "id" array (the ids the
// client asked hero|get_data_list for) against the store. Unknown ids are skipped.
func (s *Server) requestedHeroes(req ctrlproto.Request) []*session.Hero {
	if req.Params == nil {
		return nil
	}
	ids, ok := req.Params.GetArray("id")
	if !ok {
		return nil
	}
	var out []*session.Hero
	for _, v := range ids.Dense {
		var id int32
		switch n := v.(type) {
		case int32:
			id = n
		case float64:
			id = int32(n)
		default:
			continue
		}
		if u, ok := s.Store.ByID(id); ok && u.Hero != nil {
			out = append(out, u.Hero)
		}
	}
	return out
}

// handleHeroGetData answers hero|get_data -> {load:{id, race, gender, ...}} for a
// SINGLE hero. This is the request the client auto-fires (HeroSender.DataRequest)
// when it binds a "Hero"-prefab avatar that appears AFTER the initial world load --
// exactly another player walking into the square (Battle.BindPlayerAvatar ->
// UpdateHeroData). Without it the other player shows a nickname but no body, since
// SetHeroView only runs once this response's mView arrives. HeroDataArgParser wants
// the view under a "load" key (its presence sets mPersExists=true) with the hero id
// inside it; an unknown id gets {create:{id}} (mPersExists=false -> client skips).
func (s *Server) handleHeroGetData(req ctrlproto.Request, resp *ctrlproto.Response) {
	h := s.heroFromSession(req)
	if id, ok := req.Params.GetInt("id"); ok {
		if u, ok := s.Store.ByID(id); ok && u.Hero != nil {
			h = u.Hero
		}
	}
	if h == nil {
		resp.Add("hero", "get_data", amf.NewArray().
			Set("create", amf.NewArray().Set("id", int32(-1))))
		return
	}
	load := heroViewArray(h).Set("id", h.ID)
	resp.Add("hero", "get_data", amf.NewArray().Set("load", load))
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
	addHeroData(m, h)
	return m
}

// addHeroData adds h's {load, dressed_items, clan_info, user_info} entry to m, keyed
// by the hero id (a no-op for a nil hero). Shared by hero|get_data_list (which may
// carry several occupants) and user|full_hero_info (a single hero).
func addHeroData(m *amf.MixedArray, h *session.Hero) {
	if h == nil {
		return
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

// handleStoreList lives in hero_items.go (the city gear shop).

// handleCastleList answers castle|list. CastleListArgParser requires "castles".
func (s *Server) handleCastleList(req ctrlproto.Request, resp *ctrlproto.Response) {
	resp.Add("castle", "list", amf.NewArray().Set("castles", amf.NewArray()))
}

// handleGroupList answers user|group_list -> {users:[...], leader}. It returns the
// party of the requester (or of the user_id queried), each member row carrying the
// fields GroupWindow reads {id, nick, online, level, race, gender, clan_info}.
//
// leader is 0 when the requester leads (or is solo), the leader's id when someone else
// leads, and -1 only when inspecting ANOTHER user who is solo. Reporting a solo
// requester as leader=0 is deliberate: the client sets mLeaderId=SelfHeroId on
// leader==0 (Group.OnGroupList), which makes Group.IsLeader true -- and the central-
// square popup only offers "invite to group" (GROUP_INVITE) to a leader. Returning -1
// there (mLeaderId=-1, IsLeader=false) leaves a lone player with only "ask to join",
// so the invite flow is never reachable and no group can ever be started by inviting.
func (s *Server) handleGroupList(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	users := amf.NewArray()
	leader := int32(-1)
	if u != nil {
		target := u.ID
		if id, ok := req.Params.GetInt("user_id"); ok && id > 0 {
			target = id
		}
		if g := s.Store.GroupOf(target); g != nil {
			leader = g.Leader
			if g.Leader == u.ID {
				leader = 0
			}
			for _, mid := range g.Members {
				users.Add(s.groupMemberRow(mid))
			}
		} else if target == u.ID {
			// Solo requester: their own (trivial) leader, so GROUP_INVITE shows. members
			// stays empty (IsEmpty true), so "ask to join" stays available too.
			leader = 0
		}
	}
	resp.Add("user", "group_list", amf.NewArray().Set("users", users).Set("leader", leader))
}

// groupMemberRow builds one party-list row for a member id.
func (s *Server) groupMemberRow(uid int32) *amf.MixedArray {
	nick := ""
	var level, race int32
	gender := false
	online := s.MPD != nil && s.MPD.Online(uid)
	if mu, ok := s.Store.ByID(uid); ok {
		nick = mu.Username
		if mu.Hero != nil {
			level, race, gender = mu.Hero.Level, mu.Hero.Race, mu.Hero.Gender
		}
	}
	return amf.NewArray().
		Set("id", uid).
		Set("nick", nick).
		Set("online", online).
		Set("level", level).
		Set("race", race).
		Set("gender", gender).
		Set("clan_info", amf.NewArray().Set("id", int32(-1)).Set("tag", ""))
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
