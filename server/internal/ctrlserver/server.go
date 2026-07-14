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

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/session"
)

type Server struct {
	Store *session.Store

	// BattleHost/BattlePorts are handed to the client in the common|area_conf
	// response so it knows where to open the Battle TCP connection. Set by
	// cmd/ctrlserver after starting the battle listener.
	BattleHost  string
	BattlePorts []int32
}

func New() *Server {
	return &Server{
		Store:       session.NewStore(),
		BattleHost:  "127.0.0.1",
		BattlePorts: []int32{9339},
	}
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
	mux.HandleFunc("/xml/items.amf", s.handleEmptyProto)
	mux.HandleFunc("/xml/avatars.amf", s.handleAvatarsAmf)
	mux.HandleFunc("/xml/quests.amf", s.handleEmptyProto)
	mux.HandleFunc("/xml/tasks.amf", s.handleEmptyProto)
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
	case ctrlproto.CmdKey("avatar", "list"):
		s.handleAvatarListReal(req, resp)
	case ctrlproto.CmdKey("store", "list"):
		s.handleStoreList(req, resp)
	case ctrlproto.CmdKey("castle", "list"):
		s.handleCastleList(req, resp)
	case ctrlproto.CmdKey("user", "group_list"):
		s.handleGroupList(req, resp)
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
		// CentralSquareScreen.Show calls Npcs.UpdateContent(); NpcListArgParser
		// requires an "npcs" associative map (empty = no NPCs). A bare ack
		// throws "key not found: npcs at list" on the client.
		resp.Add("npc", "list", amf.NewArray().Set("npcs", amf.NewArray()))
	case ctrlproto.CmdKey("quest", "update"):
		// Likewise SelfQuests.UpdateContent(); QuestListArgParser requires a
		// "quests" associative map (empty = no quests).
		resp.Add("quest", "update", amf.NewArray().Set("quests", amf.NewArray()))
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
	}
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
