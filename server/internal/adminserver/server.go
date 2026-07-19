// Package adminserver is the private, password-protected web console for
// operating a running Tanat server: it lists accounts and edits their money,
// level and ban state; tunes the live gameplay knobs (mob/hero scaling, XP/coin
// event multipliers, the new-hero wallet); toggles fog of war; and overrides
// individual avatar/mob combat stats -- all WITHOUT a restart, because the game
// reads these through gamedata's runtime settings layer.
//
// It runs as a fourth listener in the same process (alongside Ctrl/Battle/MPD)
// and shares the one session.Store, so an edit is immediately visible to the
// game handlers and is persisted through the same SQLite backing.
//
// AUTH: a single operator password (constant-time compared). A successful login
// mints a random session, delivered as an HttpOnly, SameSite=Strict cookie, plus
// a CSRF token the browser must echo in the X-CSRF-Token header on every mutating
// request -- so a valid cookie alone (a cross-site forgery) cannot change state.
// It speaks plain HTTP (no TLS), so run it on a trusted network or behind a
// reverse proxy / SSH tunnel if exposed to the internet.
package adminserver

import (
	"crypto/rand"
	"crypto/subtle"
	_ "embed" // for the //go:embed of the admin UI
	"encoding/hex"
	"encoding/json"
	"log"
	"net/http"
	"strconv"
	"sync"
	"time"

	"tanatserver/internal/gamedata"
	"tanatserver/internal/session"
)

//go:embed ui.html
var uiHTML []byte

const (
	sessionCookie   = "tanat_admin"
	csrfHeader      = "X-CSRF-Token"
	sessionTTL      = 12 * time.Hour
	loginFailDelay  = 300 * time.Millisecond // throttle brute-force password guessing
	SettingsMetaKey = "admin_settings"       // meta row holding the persisted gamedata.Settings JSON
)

// Server is the admin console. Construct with New, then Handler()/ListenAndServe.
type Server struct {
	store    *session.Store
	password string
	started  time.Time

	mu       sync.Mutex
	sessions map[string]adminSession
}

type adminSession struct {
	csrf    string
	expires time.Time
}

// New builds an admin server bound to store, authenticated by password. An empty
// password locks the console (every login is refused) -- callers must supply one.
func New(store *session.Store, password string) *Server {
	return &Server{
		store:    store,
		password: password,
		started:  time.Now(),
		sessions: map[string]adminSession{},
	}
}

// LoadSettings applies the persisted admin settings blob (if any) from the store
// to gamedata, so tuning survives a restart. Call once at boot, before serving.
// A malformed blob is logged and ignored (the authored defaults stand).
func LoadSettings(store *session.Store) {
	blob, ok := store.GetMeta(SettingsMetaKey)
	if !ok {
		return
	}
	if err := gamedata.ApplyJSON([]byte(blob)); err != nil {
		log.Printf("adminserver: ignoring malformed persisted settings: %v", err)
		return
	}
	log.Printf("adminserver: applied persisted gameplay settings")
}

// ListenAndServe serves the console on addr (blocks). Bind to a trusted interface.
func (s *Server) ListenAndServe(addr string) error {
	return http.ListenAndServe(addr, s.Handler())
}

// Handler returns the console's HTTP handler (useful for tests / embedding).
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleRoot)
	mux.HandleFunc("/api/login", s.handleLogin)
	mux.HandleFunc("/api/logout", s.guard(true, s.handleLogout))
	mux.HandleFunc("/api/state", s.guard(false, s.handleState))
	mux.HandleFunc("/api/players", s.guard(false, s.handlePlayers))
	mux.HandleFunc("/api/player/money", s.guard(true, s.handlePlayerMoney))
	mux.HandleFunc("/api/player/progress", s.guard(true, s.handlePlayerProgress))
	mux.HandleFunc("/api/player/ban", s.guard(true, s.handlePlayerBan))
	mux.HandleFunc("/api/player/delete", s.guard(true, s.handlePlayerDelete))
	mux.HandleFunc("/api/catalog", s.guard(false, s.handleCatalog))
	mux.HandleFunc("/api/player/inventory", s.guard(false, s.handlePlayerInventory))
	mux.HandleFunc("/api/player/quest", s.guard(true, s.handlePlayerQuest))
	mux.HandleFunc("/api/player/grant", s.guard(true, s.handlePlayerGrant))
	mux.HandleFunc("/api/settings", s.guard(true, s.handleSettings))
	mux.HandleFunc("/api/avatar", s.guard(true, s.handleAvatarOverride))
	mux.HandleFunc("/api/mob", s.guard(true, s.handleMobOverride))
	return mux
}

// ---- auth ----

func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(uiHTML)
}

type loginReq struct {
	Password string `json:"password"`
}

func (s *Server) handleLogin(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "POST only")
		return
	}
	var req loginReq
	if !readJSON(w, r, &req) {
		return
	}
	if s.password == "" || subtle.ConstantTimeCompare([]byte(req.Password), []byte(s.password)) != 1 {
		time.Sleep(loginFailDelay)
		writeErr(w, http.StatusUnauthorized, "неверный пароль")
		return
	}
	tok, csrf := randToken(), randToken()
	s.mu.Lock()
	s.sessions[tok] = adminSession{csrf: csrf, expires: time.Now().Add(sessionTTL)}
	s.mu.Unlock()
	http.SetCookie(w, &http.Cookie{
		Name:     sessionCookie,
		Value:    tok,
		Path:     "/",
		HttpOnly: true,
		SameSite: http.SameSiteStrictMode,
		MaxAge:   int(sessionTTL / time.Second),
	})
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "csrf": csrf})
}

func (s *Server) handleLogout(w http.ResponseWriter, r *http.Request) {
	if c, err := r.Cookie(sessionCookie); err == nil {
		s.mu.Lock()
		delete(s.sessions, c.Value)
		s.mu.Unlock()
	}
	http.SetCookie(w, &http.Cookie{Name: sessionCookie, Value: "", Path: "/", MaxAge: -1})
	writeJSON(w, http.StatusOK, map[string]any{"ok": true})
}

// authed returns the caller's session (validated, unexpired) if any.
func (s *Server) authed(r *http.Request) (adminSession, bool) {
	c, err := r.Cookie(sessionCookie)
	if err != nil {
		return adminSession{}, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	sess, ok := s.sessions[c.Value]
	if !ok {
		return adminSession{}, false
	}
	if time.Now().After(sess.expires) {
		delete(s.sessions, c.Value)
		return adminSession{}, false
	}
	return sess, true
}

// guard wraps an API handler with authentication and, for mutating endpoints, a
// POST-method + CSRF-header check.
func (s *Server) guard(mutating bool, h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		sess, ok := s.authed(r)
		if !ok {
			writeErr(w, http.StatusUnauthorized, "требуется вход")
			return
		}
		if mutating {
			if r.Method != http.MethodPost {
				writeErr(w, http.StatusMethodNotAllowed, "POST only")
				return
			}
			if subtle.ConstantTimeCompare([]byte(r.Header.Get(csrfHeader)), []byte(sess.csrf)) != 1 {
				writeErr(w, http.StatusForbidden, "bad CSRF token")
				return
			}
		}
		h(w, r)
	}
}

// ---- state / catalog ----

type statusView struct {
	UptimeSeconds int64 `json:"uptime_seconds"`
	StartedUnix   int64 `json:"started_unix"`
	Accounts      int   `json:"accounts"`
	Online        int   `json:"online"`
}

type avatarView struct {
	ID        int32              `json:"id"`
	Prefab    string             `json:"prefab"`
	ShortName string             `json:"short_name"`
	Type      int32              `json:"type"`
	Stats     map[string]float64 `json:"stats"`
	Override  map[string]float64 `json:"override,omitempty"`
}

type mobView struct {
	Index    int                `json:"index"`
	Prefab   string             `json:"prefab"`
	NameKey  string             `json:"name_key"`
	IsBoss   bool               `json:"is_boss"`
	Stats    map[string]float64 `json:"stats"`
	Override map[string]float64 `json:"override,omitempty"`
}

func (s *Server) handleState(w http.ResponseWriter, r *http.Request) {
	// guard() already validated the session; re-read it so the page can recover its
	// CSRF token after a reload (the cookie survives, the in-memory JS token does not).
	sess, _ := s.authed(r)
	players := s.store.ListPlayers()
	online := 0
	for _, p := range players {
		if p.Online {
			online = online + 1
		}
	}
	set := gamedata.Snapshot()

	avatars := make([]avatarView, 0, len(gamedata.Avatars()))
	for _, a := range gamedata.Avatars() {
		avatars = append(avatars, avatarView{
			ID:        a.ID,
			Prefab:    a.Prefab,
			ShortName: a.ShortName,
			Type:      a.Type,
			Stats:     statMap(gamedata.AvatarStatFields(), func(f string) float64 { return gamedata.AvatarStat(a, f) }),
			Override:  set.AvatarOverrides[a.ID],
		})
	}
	mobs := make([]mobView, 0, len(gamedata.Mobs()))
	for i, m := range gamedata.Mobs() {
		mobs = append(mobs, mobView{
			Index:    i,
			Prefab:   m.Prefab,
			NameKey:  m.NameKey,
			IsBoss:   gamedata.IsBoss(i),
			Stats:    statMap(gamedata.MobStatFields(), func(f string) float64 { return gamedata.MobStat(m, f) }),
			Override: set.MobOverrides[int32(i)],
		})
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"csrf": sess.csrf,
		"status": statusView{
			UptimeSeconds: int64(time.Since(s.started).Seconds()),
			StartedUnix:   s.started.Unix(),
			Accounts:      len(players),
			Online:        online,
		},
		"settings":           set,
		"avatars":            avatars,
		"mobs":               mobs,
		"avatar_stat_fields": gamedata.AvatarStatFields(),
		"mob_stat_fields":    gamedata.MobStatFields(),
	})
}

func (s *Server) handlePlayers(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"players": s.store.ListPlayers()})
}

// ---- player mutations ----

func (s *Server) handlePlayerMoney(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID       int32 `json:"id"`
		Money    int32 `json:"money"`
		Diamonds int32 `json:"diamonds"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	if !s.store.SetHeroMoney(req.ID, req.Money, req.Diamonds) {
		writeErr(w, http.StatusBadRequest, "у аккаунта нет героя")
		return
	}
	log.Printf("admin: set money id=%d money=%d diamonds=%d", req.ID, req.Money, req.Diamonds)
	writeJSON(w, http.StatusOK, map[string]any{"ok": true})
}

func (s *Server) handlePlayerProgress(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID      int32 `json:"id"`
		Level   int32 `json:"level"`
		Exp     int32 `json:"exp"`
		NextExp int32 `json:"next_exp"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	if !s.store.SetHeroProgress(req.ID, req.Level, req.Exp, req.NextExp) {
		writeErr(w, http.StatusBadRequest, "у аккаунта нет героя")
		return
	}
	log.Printf("admin: set progress id=%d level=%d exp=%d", req.ID, req.Level, req.Exp)
	writeJSON(w, http.StatusOK, map[string]any{"ok": true})
}

func (s *Server) handlePlayerBan(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID     int32 `json:"id"`
		Banned bool  `json:"banned"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	if !s.store.SetBanned(req.ID, req.Banned) {
		writeErr(w, http.StatusBadRequest, "нет такого аккаунта")
		return
	}
	log.Printf("admin: set banned id=%d banned=%v", req.ID, req.Banned)
	writeJSON(w, http.StatusOK, map[string]any{"ok": true})
}

func (s *Server) handlePlayerDelete(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID int32 `json:"id"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	if !s.store.DeleteAccount(req.ID) {
		writeErr(w, http.StatusBadRequest, "нет такого аккаунта")
		return
	}
	log.Printf("admin: DELETED account id=%d", req.ID)
	writeJSON(w, http.StatusOK, map[string]any{"ok": true})
}

// ---- quests & item grants (per-player) ----

// questCatView is one catalog quest as the UI's quest picker shows it: the id to address
// it by, its objective count (the Progress ceiling), reward, and the locale name key.
type questCatView struct {
	ID      int32  `json:"id"`
	Key     string `json:"key"`
	MapID   int32  `json:"map_id"`
	Kind    int32  `json:"kind"`
	PveType int32  `json:"pve_type"`
	Count   int32  `json:"count"`
	Money   int32  `json:"money"`
	Exp     int32  `json:"exp"`
	NameKey string `json:"name_key"`
}

type itemCatView struct {
	Article int32  `json:"article"`
	Kind    int    `json:"kind"`
	Tier    int    `json:"tier"`
	NameKey string `json:"name_key"`
	Icon    string `json:"icon"`
}

type wearableCatView struct {
	Article  int32  `json:"article"`
	Race     string `json:"race"`
	Tier     int32  `json:"tier"`
	Color    string `json:"color"`
	Slot     string `json:"slot"`
	MinLevel int32  `json:"min_level"`
	Price    int32  `json:"price"`
	NameKey  string `json:"name_key"`
}

// handleCatalog serves the static gamedata catalogs the UI needs to build its quest and
// item pickers: every quest, every consumable (potion), every wearable. Read-only; the UI
// fetches it once after login.
func (s *Server) handleCatalog(w http.ResponseWriter, r *http.Request) {
	quests := make([]questCatView, 0, len(gamedata.Quests()))
	for _, q := range gamedata.Quests() {
		quests = append(quests, questCatView{
			ID: q.ID, Key: q.Key, MapID: q.MapID, Kind: q.Kind, PveType: q.PveType,
			Count: q.Count, Money: q.Money, Exp: q.Exp, NameKey: q.NameKey,
		})
	}
	potions := make([]itemCatView, 0, len(gamedata.Items()))
	for _, it := range gamedata.Items() {
		potions = append(potions, itemCatView{
			Article: it.ArticleID, Kind: int(it.Kind), Tier: it.Tier, NameKey: it.NameKey, Icon: it.Icon,
		})
	}
	wearables := make([]wearableCatView, 0, len(gamedata.Wearables()))
	for _, wr := range gamedata.Wearables() {
		wearables = append(wearables, wearableCatView{
			Article: wr.ArticleID, Race: wr.Race, Tier: wr.Tier, Color: wr.Color, Slot: wr.Slot,
			MinLevel: wr.MinHeroLevel, Price: wr.Price, NameKey: wr.NameKey,
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"quests":    quests,
		"potions":   potions,
		"wearables": wearables,
	})
}

// handlePlayerInventory returns one hero's live quest states + bag/owned/dressed items so
// the operator can see what to edit. Read-only GET with an ?id= query param.
func (s *Server) handlePlayerInventory(w http.ResponseWriter, r *http.Request) {
	id64, err := strconv.ParseInt(r.URL.Query().Get("id"), 10, 32)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "bad id")
		return
	}
	id := int32(id64)
	quests, ok := s.store.AdminHeroQuests(id)
	if !ok {
		writeErr(w, http.StatusBadRequest, "у аккаунта нет героя")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"quests":  quests,
		"bag":     s.store.HeroBag(id),
		"owned":   s.store.HeroOwned(id),
		"dressed": s.store.HeroDressed(id),
	})
}

// handlePlayerQuest sets (upsert) or removes one quest's progress for a hero. It only
// touches quest STATE, never the quest definition. Status/Progress are clamped to the
// client's valid ranges; a REPLAY quest sent as WAIT_COOLDOWN gets its remaining cooldown
// from the request (0 = re-offerable now).
func (s *Server) handlePlayerQuest(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID            int32 `json:"id"`
		QuestID       int32 `json:"quest_id"`
		Status        int32 `json:"status"`
		Progress      int32 `json:"progress"`
		CooldownUntil int64 `json:"cooldown_until"`
		Remove        bool  `json:"remove"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	q, known := gamedata.QuestByID(req.QuestID)
	if !known {
		writeErr(w, http.StatusBadRequest, "нет такого квеста")
		return
	}
	if req.Remove {
		if !s.store.AdminRemoveQuest(req.ID, req.QuestID) {
			writeErr(w, http.StatusBadRequest, "у героя нет этого квеста")
			return
		}
		log.Printf("admin: removed quest %d from hero %d", req.QuestID, req.ID)
		writeJSON(w, http.StatusOK, map[string]any{"ok": true})
		return
	}
	// Clamp to valid ranges so a hand-typed value can't wedge the client's journal.
	if req.Status < gamedata.QuestStatusWaitCooldown || req.Status > gamedata.QuestStatusClosed {
		writeErr(w, http.StatusBadRequest, "недопустимый статус")
		return
	}
	if req.Progress < 0 {
		req.Progress = 0
	}
	if req.Progress > q.Count {
		req.Progress = q.Count
	}
	if !s.store.AdminSetQuestState(req.ID, req.QuestID, req.Status, req.Progress, req.CooldownUntil) {
		writeErr(w, http.StatusBadRequest, "у аккаунта нет героя")
		return
	}
	log.Printf("admin: set quest %d for hero %d -> status=%d progress=%d", req.QuestID, req.ID, req.Status, req.Progress)
	writeJSON(w, http.StatusOK, map[string]any{"ok": true})
}

// grantCountCap bounds a single grant so a mistyped count can't balloon the store.
const grantCountCap int32 = 999

// handlePlayerGrant gives a hero one or more items free of charge. It auto-detects the
// item family from the article id: a consumable (potion) merges into the persistent bag;
// a wearable mints that many discrete unequipped instances into the owned bag.
func (s *Server) handlePlayerGrant(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID        int32 `json:"id"`
		ArticleID int32 `json:"article_id"`
		Count     int32 `json:"count"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	if req.Count < 1 {
		req.Count = 1
	}
	if req.Count > grantCountCap {
		req.Count = grantCountCap
	}
	switch {
	case isItem(req.ArticleID):
		if !s.store.AddBagItem(req.ID, req.ArticleID, req.Count) {
			writeErr(w, http.StatusBadRequest, "у аккаунта нет героя")
			return
		}
		log.Printf("admin: granted potion %d x%d to hero %d", req.ArticleID, req.Count, req.ID)
	case isWearable(req.ArticleID):
		added, ok := s.store.AdminGrantWearable(req.ID, req.ArticleID, req.Count)
		if !ok {
			writeErr(w, http.StatusBadRequest, "у аккаунта нет героя")
			return
		}
		log.Printf("admin: granted wearable %d x%d to hero %d (%d instances)", req.ArticleID, req.Count, req.ID, len(added))
	default:
		writeErr(w, http.StatusBadRequest, "нет такого предмета")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true})
}

func isItem(article int32) bool     { _, ok := gamedata.ItemByArticle(article); return ok }
func isWearable(article int32) bool { _, ok := gamedata.WearableByArticle(article); return ok }

// ---- gameplay settings ----

type settingsReq struct {
	FogOfWar           bool    `json:"fog_of_war"`
	HuntFog            bool    `json:"hunt_fog"`
	MobHPPerLevel      float64 `json:"mob_hp_per_level"`
	MobDmgPerLevel     float64 `json:"mob_dmg_per_level"`
	MobXPPerLevel      float64 `json:"mob_xp_per_level"`
	MobCoinPerLevel    float64 `json:"mob_coin_per_level"`
	HeroPowerPerLevel  float64 `json:"hero_power_per_level"`
	HeroHealthPerLevel float64 `json:"hero_health_per_level"`
	XPMultiplier       float64 `json:"xp_multiplier"`
	CoinMultiplier     float64 `json:"coin_multiplier"`
	NewHeroMoney       int32   `json:"new_hero_money"`
	NewHeroDiamond     int32   `json:"new_hero_diamond"`
}

// handleSettings replaces the SCALAR gameplay knobs (fog + all multipliers/slopes
// + new-hero wallet), leaving the per-entity stat overrides untouched, then
// persists the whole settings blob.
func (s *Server) handleSettings(w http.ResponseWriter, r *http.Request) {
	var req settingsReq
	if !readJSON(w, r, &req) {
		return
	}
	gamedata.Update(func(st *gamedata.Settings) {
		st.FogOfWarEnabled = req.FogOfWar
		st.HuntFogEnabled = req.HuntFog
		st.MobHPPerLevel = req.MobHPPerLevel
		st.MobDmgPerLevel = req.MobDmgPerLevel
		st.MobXPPerLevel = req.MobXPPerLevel
		st.MobCoinPerLevel = req.MobCoinPerLevel
		st.HeroPowerPerLevel = req.HeroPowerPerLevel
		st.HeroHealthPerLevel = req.HeroHealthPerLevel
		st.XPMultiplier = req.XPMultiplier
		st.CoinMultiplier = req.CoinMultiplier
		st.NewHeroMoney = req.NewHeroMoney
		st.NewHeroDiamond = req.NewHeroDiamond
	})
	s.persistSettings()
	log.Printf("admin: updated gameplay settings (fog=%v huntfog=%v xpx%.2f coinx%.2f)", req.FogOfWar, req.HuntFog, req.XPMultiplier, req.CoinMultiplier)
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "settings": gamedata.Snapshot()})
}

func (s *Server) handleAvatarOverride(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID       int32              `json:"id"`
		Override map[string]float64 `json:"override"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	if _, ok := gamedata.AvatarByID(req.ID); !ok {
		writeErr(w, http.StatusBadRequest, "нет такого аватара")
		return
	}
	clean := filterStats(req.Override, gamedata.AvatarStatFields())
	gamedata.Update(func(st *gamedata.Settings) {
		if len(clean) == 0 {
			delete(st.AvatarOverrides, req.ID)
		} else {
			st.AvatarOverrides[req.ID] = clean
		}
	})
	s.persistSettings()
	log.Printf("admin: avatar %d override -> %v", req.ID, clean)
	writeJSON(w, http.StatusOK, map[string]any{"ok": true})
}

func (s *Server) handleMobOverride(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Index    int                `json:"index"`
		Override map[string]float64 `json:"override"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	if req.Index < 0 || req.Index >= len(gamedata.Mobs()) {
		writeErr(w, http.StatusBadRequest, "нет такого моба")
		return
	}
	clean := filterStats(req.Override, gamedata.MobStatFields())
	gamedata.Update(func(st *gamedata.Settings) {
		if len(clean) == 0 {
			delete(st.MobOverrides, int32(req.Index))
		} else {
			st.MobOverrides[int32(req.Index)] = clean
		}
	})
	s.persistSettings()
	log.Printf("admin: mob %d override -> %v", req.Index, clean)
	writeJSON(w, http.StatusOK, map[string]any{"ok": true})
}

// persistSettings writes the live settings to the store's meta table so tuning
// survives a restart. Best-effort: a failure is logged, not surfaced (the live
// in-memory settings are already applied).
func (s *Server) persistSettings() {
	blob, err := gamedata.MarshalSettings()
	if err != nil {
		log.Printf("adminserver: marshal settings failed: %v", err)
		return
	}
	if err := s.store.SetMeta(SettingsMetaKey, string(blob)); err != nil {
		log.Printf("adminserver: persist settings failed: %v", err)
	}
}

// ---- helpers ----

func statMap(fields []string, get func(string) float64) map[string]float64 {
	m := make(map[string]float64, len(fields))
	for _, f := range fields {
		m[f] = get(f)
	}
	return m
}

// filterStats keeps only the entries whose key is a known stat field, dropping
// anything else so junk never reaches the persisted override map.
func filterStats(in map[string]float64, allowed []string) map[string]float64 {
	ok := make(map[string]bool, len(allowed))
	for _, f := range allowed {
		ok[f] = true
	}
	out := make(map[string]float64)
	for k, v := range in {
		if ok[k] {
			out[k] = v
		}
	}
	return out
}

func randToken() string {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		// crypto/rand.Read never fails on the supported platforms; if it somehow
		// did, a fixed token is still better than panicking the whole server.
		return "0000000000000000000000000000000000000000000000000000000000000000"
	}
	return hex.EncodeToString(b)
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]any{"error": msg})
}

// readJSON decodes the request body into v, writing a 400 and returning false on
// failure. Bodies are capped to keep a malformed client from allocating without
// bound.
func readJSON(w http.ResponseWriter, r *http.Request, v any) bool {
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20))
	if err := dec.Decode(v); err != nil {
		writeErr(w, http.StatusBadRequest, "bad JSON body")
		return false
	}
	return true
}
