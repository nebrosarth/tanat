// Package ctrlserver implements the HTTP "Ctrl" channel
// (POST /entry_point.php) that the real client talks to, as reverse
// engineered from TanatKernel.CtrlEntryPoint / CtrlServerConnection.
//
// Coverage so far: user|login, hero|create. Everything else is acknowledged
// generically (status:100, no extra fields) and logged, so we can point the
// real client at this server and observe exactly what it sends next -
// that's a far more reliable way to fill in the remaining ~150 commands
// than guessing from static analysis alone.
package ctrlserver

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"net/http"
	"sync"

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/gamedata"
	"tanatserver/internal/mpd"
	"tanatserver/internal/session"
)

type Server struct {
	Store *session.Store

	// BattleHost/BattlePorts are handed to the client in the common|area_conf
	// response so it knows where to open the Battle TCP connection. Set by
	// cmd/ctrlserver after starting the battle listener. Used for the central
	// square (and as the fallback launch target when MatchLauncher is nil).
	BattleHost  string
	BattlePorts []int32

	// MatchLauncher, when set, provides a dedicated per-match Battle server (with
	// its own battle clock) for each Штурм/Охота launch, so the in-battle timer
	// counts from match start rather than the shared server's uptime. nil falls
	// back to the shared BattleHost/BattlePorts. Set by cmd/ctrlserver.
	MatchLauncher MatchLauncher

	// MPD is the push-channel hub (chat lines, party invites, online status). Set by
	// cmd/ctrlserver after starting the MPD listener; nil in tests that don't push.
	// MPDHost/MPDPorts are advertised to the client in the chat|conf packet so it
	// opens the MPD socket.
	MPD      *mpd.Hub
	MPDHost  string
	MPDPorts []int32

	// DotaMatchSize is how many players a «Штурм» (DOTA) match waits for before it
	// starts (configurable 1-10 via --dota-players; clamped by SetDotaMatchSize). 1 =
	// the solo instant-match. Read under fightMu.
	DotaMatchSize int32

	// fightMu guards the «Штурм» (DOTA) matchmaking state: fightSel is the in-flight
	// selection per user (chosen map + avatar + room), held between fight|select_avatar
	// and the arg-less fight|ready; dotaQueue is the per-map waiting list the matcher
	// fills until DotaMatchSize players are ready to form a match; nextDotaRoom hands
	// each formed match a unique shared-world room id so concurrent matches don't merge.
	// See dota.go.
	fightMu      sync.Mutex
	fightSel     map[int32]fightSelection
	dotaQueue    map[int32][]int32
	nextDotaRoom int32
}

func New() *Server {
	return &Server{
		Store:         session.NewStore(),
		BattleHost:    "127.0.0.1",
		BattlePorts:   []int32{9339},
		DotaMatchSize: 1, // solo instant-match until configured otherwise
	}
}

// MatchLauncher creates (or reuses) a per-match Battle server for a room and
// returns the host and port the client should reconnect to for that battle.
// Implemented by battleserver.MatchHost; kept as an interface here so ctrlserver
// does not import battleserver (preserving the layer boundary).
type MatchLauncher interface {
	Launch(mapID, room int32) (host string, port int32, err error)
}

// launchTarget returns the {ip, port[]} the client should reconnect to for a
// battle in room. With a MatchLauncher set it is that room's dedicated per-match
// server (its own clock makes the in-battle timer match-relative); otherwise the
// shared BattleHost/BattlePorts. A launcher error is logged and falls back to the
// shared server so a launch never fails outright on this account.
func (s *Server) launchTarget(mapID, room int32) (ip string, ports *amf.MixedArray) {
	if s.MatchLauncher != nil {
		if host, port, err := s.MatchLauncher.Launch(mapID, room); err == nil {
			out := amf.NewArray()
			out.Add(port)
			return host, out
		} else {
			log.Printf("ctrl: per-match battle server for map=%d room=%d failed (%v); using shared server", mapID, room, err)
		}
	}
	ports = amf.NewArray()
	for _, p := range s.BattlePorts {
		ports.Add(p)
	}
	return s.BattleHost, ports
}

// DotaMatchMin/Max bound the configurable «Штурм» match size.
const (
	DotaMatchMin int32 = 1
	DotaMatchMax int32 = 10
)

// SetDotaMatchSize sets how many players a «Штурм» match waits for, clamped to
// [DotaMatchMin, DotaMatchMax]. Returns the value actually applied.
func (s *Server) SetDotaMatchSize(n int32) int32 {
	if n < DotaMatchMin {
		n = DotaMatchMin
	}
	if n > DotaMatchMax {
		n = DotaMatchMax
	}
	s.fightMu.Lock()
	s.DotaMatchSize = n
	s.fightMu.Unlock()
	return n
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/entry_point.php", s.handleEntryPoint)
	mux.HandleFunc("/version", s.handleVersion)
	// Prototype files the client downloads at connect (item_proto/avatars/
	// quests/tasks). Each is a raw AMF MixedArray whose Dense list holds the
	// prototypes (see PropertyHolder/CtrlAvatarStore/QuestStore.Retrieve). We
	// serve an empty one so those loads succeed instead of 404-ing and retrying
	// forever. Real content can be filled in later.
	mux.HandleFunc("/xml/items.amf", s.handleItemsAmf)
	mux.HandleFunc("/xml/avatars.amf", s.handleAvatarsAmf)
	mux.HandleFunc("/xml/quests.amf", s.handleQuestsAmf)
	// tasks.amf carries the PvP battle-tasks (map_1_0 «Штурм», QUEST_TASK). The client merges it
	// into the SAME QuestStore as quests.amf, so it uses the identical encoder (see tasks.go).
	mux.HandleFunc("/xml/tasks.amf", s.handleTasksAmf)
	return mux
}

// handleEmptyProto serves an AMF-encoded empty MixedArray (bytes 09 01 01) for
// the prototype download endpoints.
func (s *Server) handleEmptyProto(w http.ResponseWriter, r *http.Request) {
	writeAMF(w, amf.NewArray(), "proto ")
}

// handleAvatarsAmf serves /xml/avatars.amf: the battle-avatar prototype list
// the client's CtrlAvatarStore parses at connect (see hunt.go).
func (s *Server) handleAvatarsAmf(w http.ResponseWriter, r *http.Request) {
	writeAMF(w, s.handleAvatarsProto(), "avatars.amf ")
}

// handleItemsAmf serves /xml/items.amf: the Ctrl-side item catalog
// PropertyHolder.RetrieveProperties(Stream) parses at connect
// (CtrlServerConnection.DownloadItemPrototypes), keyed by article id and
// consumed by CachedCtrlPrototypeProvider whenever anything (the lobby bag,
// the in-battle bag) resolves a CtrlPrototype.Article. This is a completely
// separate, AMF-native mechanism from the Battle channel's XML-in-AMF
// PROTOTYPE_INFO (internal/battleserver/consumables.go's itemProtoDesc) --
// serving it empty (the old handleEmptyProto stub) left every synthetic
// potion article id's CtrlPrototype.Article permanently null: the lobby bag
// silently drops such items (SelfHero.CreateThing's ctrlPrototype.Article ==
// null guard) and the in-battle bag NullReferenceExceptions instead
// (BattleScreen.UpdateInventory/InventoryMenu.SetItems dereference
// CtrlProto.Article.X unconditionally), aborting that UI's entire refresh.
func (s *Server) handleItemsAmf(w http.ResponseWriter, r *http.Request) {
	writeAMF(w, s.handleItemsProto(), "items.amf ")
}

// handleQuestsAmf serves /xml/quests.amf: the baked PvE quest catalog the client's
// QuestStore.Retrieve parses at connect (CtrlServerConnection.DownloadQuests). See quests.go.
func (s *Server) handleQuestsAmf(w http.ResponseWriter, r *http.Request) {
	writeAMF(w, s.handleQuestsProto(), "quests.amf ")
}

// handleTasksAmf serves /xml/tasks.amf: the baked PvP battle-task catalog. The client downloads
// it into the SAME QuestStore as quests.amf (CtrlServerConnection.DownloadQuests(TasksUrl)), so
// its shape is identical -- a dense array of quest objects. See tasks.go.
func (s *Server) handleTasksAmf(w http.ResponseWriter, r *http.Request) {
	writeAMF(w, s.handleTasksProto(), "tasks.amf ")
}

// ctrlItemKindPotion is ShopGUI.ItemType.POTION (=19 in the decompiled enum).
// CtrlPrototype.PArticle.mKindId defaults to 0 (ItemType.QUEST_ITEM) when the
// "kind_id" field is absent, and FormatedTipMgr's tooltip renders that as the
// generic "QUEST_ITEM_TEXT" locale line ("Предмет, требующийся для задания")
// -- reported live as every potion showing a bogus "quest item" description.
const ctrlItemKindPotion int32 = 19

// handleItemsProto builds one PropertyHolder entry per consumable: "id" plus
// PCtrlDesc's five keys (id/title/short/long/icon -- ALL required, since
// PropertyHolder.RetrieveProperty<T> loads PCtrlDesc/PArticle/PPrefab for an
// entry inside one shared try/catch and a KeyNotFoundException from a missing
// required field drops the whole entry, not just that property) and a handful
// of PArticle fields (all read via the non-throwing TryGet, so optional).
// tree_id/tree_slot are deliberately omitted: CtrlPrototype.IsConsumable()
// actually means "is a skill-tree upgrade item" (gates BattleScreen's
// tree-panel vs. normal-bag routing), NOT "is a drinkable potion" -- setting
// it would misroute every potion out of the normal bag.
func (s *Server) handleItemsProto() *amf.MixedArray {
	root := amf.NewArray()
	for _, it := range gamedata.Items() {
		root.Add(amf.NewArray().
			Set("id", it.ArticleID).
			Set("title", it.NameKey).
			Set("short", "").
			Set("long", it.DescKey).
			Set("icon", it.Icon).
			Set("price", int32(0)).
			Set("sell_price", int32(0)).
			Set("type_id", int32(1)).
			Set("kind_id", ctrlItemKindPotion).
			Set("min_hero_level", int32(1)).
			Set("min_ava_level", int32(0)).
			Set("cnt", int32(1)).
			Set("price_type", int32(1)).
			Set("cooldown", it.Cooldown).
			Set("sort", int32(0)).
			Set("flags", int32(0)))
	}
	// Avatar battle-tree items ("предметы аватаров"): the in-battle DotA-style
	// item build. Same items.amf blob, but each carries tree_id/tree_slot/
	// tree_parents (which is what routes it into a BattleItemMenu tab and gates
	// its purchase) and a params list the tooltip resolves against the item's
	// baked LongDesc placeholders. price is real here (checked against the
	// player's in-battle VirtualMoney before the buy is allowed).
	for _, it := range gamedata.AvatarItems() {
		root.Add(treeArticleEntry(it))
	}
	// Hero gear ("предметы героев"): the persistent WEARABLE Set pieces the city shop
	// sells and the paperdoll equips. Each is a plain PArticle (price/kind_id/params +
	// the baked title/icon/long) the shop's article-id lists resolve against.
	for _, w := range gamedata.Wearables() {
		root.Add(wearableArticleEntry(w))
	}
	return root
}

// treeArticleEntry builds one avatar-tree article for items.amf. Beyond the five
// structurally-required PCtrlDesc keys (id/title/short/long/icon), it carries the
// PArticle fields the item tree actually reads: tree_id (the tab), tree_slot (the
// grid cell), tree_parents (the purchase edges -- empty for a root), price/
// price_type (affordability), min_ava_level, and params (the stat tooltip).
// params is a DENSE array of {skill_id, impact, value}; tree_parents a dense
// array of ints -- exactly the shapes CtrlPrototype.PArticle.Load parses.
func treeArticleEntry(it gamedata.AvatarItem) *amf.MixedArray {
	parents := amf.NewArray()
	for _, p := range it.Parents {
		parents.Add(p)
	}
	params := amf.NewArray()
	for _, st := range it.Stats {
		params.Add(amf.NewArray().
			Set("skill_id", st.Name).
			Set("impact", st.Impact()).
			Set("value", st.Value))
	}
	return amf.NewArray().
		Set("id", it.ArticleID).
		Set("title", it.NameKey).
		Set("short", "").
		Set("long", it.DescKey).
		Set("icon", it.Icon).
		Set("price", it.Price).
		Set("sell_price", int32(0)).
		Set("type_id", int32(2)). // CtrlThing.PlaceType.AVATAR
		Set("kind_id", gamedata.AvatarItemKindID()).
		Set("min_hero_level", int32(1)).
		Set("min_ava_level", it.MinAvaLvl).
		Set("cnt", int32(1)).
		Set("price_type", int32(1)). // virtual money (in-battle gold)
		Set("sort", it.TreeSlot).
		Set("flags", int32(0)).
		Set("tree_id", it.TreeID).
		Set("tree_slot", it.TreeSlot).
		Set("tree_parents", parents).
		Set("params", params)
}

// clientVersion must match TanatApp.mVersion (the Launcher component's
// serialized version string, "1.11" for this client build) so Updater's
// version check reports IDENTICAL and the login screen isn't left disabled
// with an "update in progress" popup (see Launcher.StartVersionUpdate: only
// IDENTICAL and FORBIDDEN skip the download flow, and FORBIDDEN sets
// LoginScreen.IsUpdating = true, which disables the login button).
const clientVersion = "1.11"

// handleVersion answers the client's startup auto-update check
// (Updater.CheckVersion does GET {autoupdate_addr}/version, comparing the
// first line of the response against its own version string).
func (s *Server) handleVersion(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain")
	_, _ = w.Write([]byte(clientVersion))
}

func (s *Server) handleEntryPoint(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "read error", http.StatusBadRequest)
		return
	}

	resp := ctrlproto.NewResponse()

	if len(body) == 0 {
		s.writeResponse(w, resp)
		return
	}

	dec := amf.NewDecoder(bytes.NewReader(body))
	v, err := dec.DecodeMessage()
	if err != nil {
		log.Printf("ctrl: decode error: %v (%d bytes)", err, len(body))
		s.writeResponse(w, resp)
		return
	}
	root, ok := v.(*amf.MixedArray)
	if !ok {
		log.Printf("ctrl: root is not an array: %T", v)
		s.writeResponse(w, resp)
		return
	}

	req := ctrlproto.ParseRequest(root)
	if req.IsPing {
		s.writeResponse(w, resp)
		return
	}

	log.Printf("ctrl: recv %s|%s sess_uid=%s counter=%d params=%s",
		req.Object, req.Action, req.SessUID, req.Counter, dumpArray(req.Params))

	s.dispatch(req, resp)
	s.writeResponse(w, resp)
}

func (s *Server) dispatch(req ctrlproto.Request, resp *ctrlproto.Response) {
	switch ctrlproto.CmdKey(req.Object, req.Action) {
	case ctrlproto.CmdKey("user", "login"):
		s.handleLogin(req, resp)
	case ctrlproto.CmdKey("hero", "create"):
		s.handleHeroCreate(req, resp)
	case ctrlproto.CmdKey("common", "area_conf"):
		s.handleAreaConf(req, resp)
	case ctrlproto.CmdKey("user", "money"):
		s.handleUserMoney(req, resp)
	case ctrlproto.CmdKey("user", "bag"):
		s.handleUserBag(req, resp)
	case ctrlproto.CmdKey("user", "game_info"):
		s.handleUserGameInfo(req, resp)
	case ctrlproto.CmdKey("user", "full_hero_info"):
		s.handleFullHeroInfo(req, resp)
	case ctrlproto.CmdKey("hero", "get_data_list"):
		s.handleHeroGetDataList(req, resp)
	case ctrlproto.CmdKey("hero", "get_data"):
		s.handleHeroGetData(req, resp)
	case ctrlproto.CmdKey("chat", "add"):
		s.handleChatAdd(req, resp)
	case ctrlproto.CmdKey("avatar", "list"):
		s.handleAvatarListReal(req, resp)
	case ctrlproto.CmdKey("store", "list"):
		s.handleStoreList(req, resp)
	case ctrlproto.CmdKey("store", "buy"):
		s.handleStoreBuy(req, resp)
	case ctrlproto.CmdKey("store", "sell"):
		s.handleStoreSell(req, resp)
	case ctrlproto.CmdKey("user", "dress"):
		s.handleUserDress(req, resp)
	case ctrlproto.CmdKey("user", "undress"):
		s.handleUserUndress(req, resp)
	case ctrlproto.CmdKey("castle", "list"):
		s.handleCastleList(req, resp)
	case ctrlproto.CmdKey("user", "group_list"):
		s.handleGroupList(req, resp)
	case ctrlproto.CmdKey("user", "join_from_group_request"):
		s.handleGroupInvite(req, resp)
	case ctrlproto.CmdKey("user", "join_from_group_answer"):
		s.handleGroupInviteAnswer(req, resp)
	case ctrlproto.CmdKey("user", "join_to_group_request"):
		s.handleGroupJoinRequest(req, resp)
	case ctrlproto.CmdKey("user", "join_to_group_answer"):
		s.handleGroupJoinAnswer(req, resp)
	case ctrlproto.CmdKey("user", "leave_group"):
		s.handleGroupLeave(req, resp)
	case ctrlproto.CmdKey("user", "remove_from_group"):
		s.handleGroupKick(req, resp)
	case ctrlproto.CmdKey("user", "change_leader"):
		s.handleGroupChangeLeader(req, resp)
	case ctrlproto.CmdKey("user", "get_bw_list"):
		s.handleGetBwList(req, resp)
	case ctrlproto.CmdKey("user", "add_to_list"):
		s.handleAddToList(req, resp)
	case ctrlproto.CmdKey("user", "remove_from_list"):
		s.handleRemoveFromList(req, resp)
	case ctrlproto.CmdKey("user", "friend_answer"):
		s.handleFriendAnswer(req, resp)
	case ctrlproto.CmdKey("user", "find"):
		s.handleUserFind(req, resp)
	case ctrlproto.CmdKey("common", "can_reconnect"):
		s.handleCanReconnect(req, resp)
	case ctrlproto.CmdKey("arena", "get_maps_info"):
		s.handleArenaMapsInfoReal(req, resp)
	case ctrlproto.CmdKey("arena", "get_map_type_descs"):
		s.handleMapTypeDescs(req, resp)
	case ctrlproto.CmdKey("arena", "get_maps"):
		s.handleArenaGetMaps(req, resp)
	case ctrlproto.CmdKey("hunt", "join"):
		s.handleHuntJoin(req, resp)
	case ctrlproto.CmdKey("hunt", "ready"):
		s.handleHuntReady(req, resp)
	case ctrlproto.CmdKey("hunt", "accept"):
		// Validation-only on the client (group-invite confirmation).
		resp.Ack("hunt", "accept")
	case ctrlproto.CmdKey("fight", "join"):
		s.handleFightJoin(req, resp)
	case ctrlproto.CmdKey("fight", "in_request"):
		s.handleFightInRequest(req, resp)
	case ctrlproto.CmdKey("fight", "select_avatar"):
		s.handleFightSelectAvatar(req, resp)
	case ctrlproto.CmdKey("fight", "ready"):
		s.handleFightReady(req, resp)
	case ctrlproto.CmdKey("fight", "desert"):
		s.handleFightDesert(req, resp)
	case ctrlproto.CmdKey("user", "leave_info"):
		// BattleScreen asks for the desertion/karma penalty shown on the exit
		// button. UserLeaveInfoArgParser requires current_karma/new_karma/labels/
		// labels_limit/time; zeros = no penalty for leaving a hunt.
		resp.Add("user", "leave_info", amf.NewArray().
			Set("current_karma", int32(0)).
			Set("new_karma", int32(0)).
			Set("labels", int32(0)).
			Set("labels_limit", int32(0)).
			Set("time", int32(0)))
	case ctrlproto.CmdKey("npc", "list"):
		// CentralSquareScreen.Show calls Npcs.UpdateContent(): the race-appropriate
		// quest NPCs (quests.go).
		s.handleNpcList(req, resp)
	case ctrlproto.CmdKey("quest", "update"):
		// SelfQuests.UpdateContent(): the hero's accepted quest state.
		s.handleQuestUpdate(req, resp)
	case ctrlproto.CmdKey("quest", "accept"):
		s.handleQuestAccept(req, resp)
	case ctrlproto.CmdKey("quest", "cancel"):
		s.handleQuestCancel(req, resp)
	case ctrlproto.CmdKey("quest", "done"):
		s.handleQuestDone(req, resp)
	default:
		log.Printf("ctrl: UNHANDLED %s|%s -> generic ack", req.Object, req.Action)
		resp.Ack(req.Object, req.Action)
	}
}

// PLACEHOLDER AUTH: see internal/session doc comment. Any email/password
// logs in and auto-registers on first use.
func (s *Server) handleLogin(req ctrlproto.Request, resp *ctrlproto.Response) {
	email := req.Params.StringOr("email", "")
	passwd := req.Params.StringOr("passwd", "")
	version := req.Params.StringOr("version", "")
	log.Printf("ctrl: login email=%s version=%s", email, version)

	u, sessKey := s.Store.LoginOrRegister(email, passwd)

	resp.Add("user", "login", amf.NewArray().
		Set("id", u.ID).
		Set("username", u.Username).
		Set("sess_key", sessKey).
		Set("flags", int32(0)))

	// Bundled proactively so the client's LoginPerformer (which waits for
	// BOTH user|login and common|hero_conf) completes the login flow in one
	// round trip. See TanatKernel.LoginPerformer.UpdateInProgressCore.
	//
	// The "id" here MUST be the user's own id, not a placeholder: HeroStore
	// separately subscribes to common|hero_conf too (OnSelfHeroData) and
	// registers a Hero object under this id, and SelfHero.Hero looks it up
	// by mUserData.UserId - i.e. Hero.Id == User.Id in this data model. Get
	// this wrong and every UI path that touches SelfHero.Hero (race select,
	// hero creation, appearance customization) null-refs on the client.
	heroConf := amf.NewArray()
	if u.HasHero {
		heroConf.Set("load", amf.NewArray().Set("id", u.ID))
	} else {
		heroConf.Set("create", amf.NewArray().Set("id", u.ID))
	}
	resp.Add("common", "hero_conf", heroConf)

	// A returning player (hero already exists) enters the world straight from
	// login: the client never requests area_conf itself on initial entry -- it
	// waits for the server to send it (OnBattleServerData -> auto-connect). We
	// push it here the same way we bundle hero_conf. (New accounts get it in
	// the hero|create response instead, once the hero exists.)
	if u.HasHero {
		resp.Add("common", "area_conf", s.areaConfFields(u, defaultAreaID(u)))
		log.Printf("ctrl: bundling area_conf into login (returning hero) for user %d", u.ID)
		// Volunteer the MPD credentials alongside area_conf so a returning player opens
		// the push socket on entry. A new account gets it in hero|create instead (once
		// it actually enters the world), so the client only ever connects once.
		s.addChatConf(resp, u.ID, sessKey)
	}
}

// addChatConf bundles the chat|conf packet (MPD host/port + this user's id and
// session key as the MPD credentials) so the client opens the push socket. The
// client never requests chat|conf -- it subscribes and connects on receipt, so we
// volunteer it alongside area_conf. No-op when the MPD server isn't configured.
func (s *Server) addChatConf(resp *ctrlproto.Response, uid int32, sid string) {
	if s.MPD == nil || s.MPDHost == "" {
		return
	}
	ports := amf.NewArray()
	for _, p := range s.MPDPorts {
		ports.Add(p)
	}
	resp.Add("chat", "conf", amf.NewArray().
		Set("chat_server_host", s.MPDHost).
		Set("chat_server_port", ports).
		Set("chat_server_uid", uid).
		Set("chat_server_sid", sid))
}

func (s *Server) handleHeroCreate(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	if u == nil {
		resp.Fail("hero", "create", 6013) // WRONG_SESSION, see LoginPerformer.LoginFailReason
		return
	}
	race := req.Params.IntOr("race", 0)
	gender := req.Params.IntOr("gender", 0) != 0
	face := req.Params.IntOr("face", 0)
	hair := req.Params.IntOr("hair", 0)
	distMark := req.Params.IntOr("dist_mark", 0)
	skinColor := req.Params.IntOr("skin_color", 0)
	hairColor := req.Params.IntOr("hair_color", 0)

	h := s.Store.CreateHero(u, race, gender, face, hair, distMark, skinColor, hairColor)
	log.Printf("ctrl: hero %d created for user %d (race=%d)", h.ID, u.ID, race)
	resp.Ack("hero", "create")

	// After creating the hero, send the client into the central square. The
	// client's CustomizeHeroScreen shows loading_screen and then just waits;
	// nothing on the client requests area_conf, so the server must volunteer it
	// (bundled in this same response). area_conf -> OnBattleServerData ->
	// BattleServerConnection.Connect kicks off the Battle TCP handshake.
	resp.Add("common", "area_conf", s.areaConfFields(u, defaultAreaID(u)))
	log.Printf("ctrl: bundling area_conf into hero|create for user %d (race=%d)", u.ID, race)
	s.addChatConf(resp, u.ID, req.SessKey)
}

// heroRaceElf mirrors TanatKernel.HeroRace.ELF (HUMAN=1, ELF=2).
const heroRaceElf int32 = 2

// Central-square area ids are the client's TanatKernel.Location enum values, sent
// verbatim as area_id by PortalSelector (SendReconnectRequest((int)mLocation)) and
// echoed back so the client's mCurrentLocation matches. NOT 1/2.
const (
	areaCSHuman int32 = 367 // Location.CS_HUMAN
	areaCSElf   int32 = 368 // Location.CS_ELF
)

// handleAreaConf answers common|area_conf (requested by
// CtrlServerConnection.SendReconnectRequest when the client wants to enter a
// location) with the Battle server coordinates and the scene to load. The
// central-square scene loads locally from data/scenes/<scene>.unity3d, so we
// just pick cs_human / cs_elf by the hero's race; SceneManager then shows
// CentralSquareScreen (SceneConfig.mScreen == "cs").
//
// Shape (see ServerDataArgParser): {area_conf:{ip, port:[ints], scene, passwd,
// area_id}, log}. "log" and status live at the top level; the rest nest under
// "area_conf".
func (s *Server) handleAreaConf(req ctrlproto.Request, resp *ctrlproto.Response) {
	u := s.userFromSession(req)
	areaID := defaultAreaID(u)
	// Honor an explicitly requested location so mCurrentLocation matches what
	// the client asked for (SendReconnectRequest(area_id) on teleport).
	if id, ok := req.Params.GetInt("area_id"); ok {
		areaID = id
	}
	// Remember the target square so the Battle server renders the right scene's
	// walkability/spawn even when a player visits the OTHER race's city (the scene
	// is chosen by area, not race).
	if u != nil {
		s.Store.SetLobbyArea(u.ID, areaID)
	}
	resp.Add("common", "area_conf", s.areaConfFields(u, areaID))
}

// isElf reports whether the session's hero is an elf. Single source of truth for
// the elf branch, so the coupled area_id (2) and scene (cs_elf) choices can't drift.
func isElf(u *session.User) bool {
	return u != nil && u.Hero != nil && u.Hero.Race == heroRaceElf
}

// defaultAreaID picks the location id for a hero's home square by race (the client's
// Location enum value, so mCurrentLocation is valid on first entry).
func defaultAreaID(u *session.User) int32 {
	if isElf(u) {
		return areaCSElf
	}
	return areaCSHuman
}

// sceneForArea maps a central-square area id to its scene bundle: CS_ELF (368) = elf
// city (cs_elf), everything else = the human cathedral square (cs_human). Chosen by
// AREA, not race, so a hero can travel to the other race's city via the portal.
func sceneForArea(areaID int32) string {
	if areaID == areaCSElf {
		return "cs_elf"
	}
	return "cs_human"
}

// areaConfFields builds the common|area_conf response body the client parses in
// ServerDataArgParser: {area_conf:{ip, port:[ints], scene, passwd, area_id},
// log}. scene must equal a SceneConfig.mSceneName; the central-square hub is
// cs_human / cs_elf by hero race, loaded locally from data/scenes/<scene>.unity3d
// (SceneConfig.mScreen == "cs" then shows CentralSquareScreen).
func (s *Server) areaConfFields(u *session.User, areaID int32) *amf.MixedArray {
	scene := sceneForArea(areaID)
	ports := amf.NewArray()
	for _, p := range s.BattlePorts {
		ports.Add(p)
	}
	area := amf.NewArray().
		Set("ip", s.BattleHost).
		Set("port", ports).
		Set("scene", scene).
		Set("passwd", "").
		Set("area_id", areaID)

	log.Printf("ctrl: area_conf -> ip=%s ports=%v scene=%s area_id=%d",
		s.BattleHost, s.BattlePorts, scene, areaID)

	return amf.NewArray().Set("area_conf", area).Set("log", int32(0))
}

func (s *Server) userFromSession(req ctrlproto.Request) *session.User {
	u, ok := s.Store.BySessKey(req.SessKey)
	if !ok {
		return nil
	}
	return u
}

func (s *Server) writeResponse(w http.ResponseWriter, resp *ctrlproto.Response) {
	writeAMF(w, resp.Root(), "")
}

// writeAMF sets the octet-stream content type and encodes v as an AMF message,
// logging "ctrl: <label>encode error" on failure. label is a caller prefix
// (e.g. "proto ", "avatars.amf ", or "" for the generic response path).
func writeAMF(w http.ResponseWriter, v interface{}, label string) {
	w.Header().Set("Content-Type", "application/octet-stream")
	if err := amf.NewEncoder().EncodeMessage(w, v); err != nil {
		log.Printf("ctrl: %sencode error: %v", label, err)
	}
}

func dumpArray(m *amf.MixedArray) string {
	if m == nil {
		return "<nil>"
	}
	return amfDump(m)
}

func amfDump(m *amf.MixedArray) string {
	var b bytes.Buffer
	b.WriteByte('{')
	first := true
	for k, v := range m.Assoc {
		if !first {
			b.WriteString(", ")
		}
		first = false
		b.WriteString(k)
		b.WriteString("=")
		switch val := v.(type) {
		case *amf.MixedArray:
			b.WriteString(amfDump(val))
		default:
			b.WriteString(toStr(val))
		}
	}
	b.WriteByte('}')
	return b.String()
}

func toStr(v interface{}) string {
	if s, ok := v.(string); ok {
		return s
	}
	if v == nil {
		return "nil"
	}
	return fmt.Sprint(v)
}
