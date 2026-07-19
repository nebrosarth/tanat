package session

// Persistent hero-gear inventory: the store-side of "предметы героев". A wearable is
// bought in the city shop (store|buy -> Owned), dressed onto a paperdoll slot
// (user|dress -> moved into Dressed) for permanent cross-match stat bonuses, and can be
// undressed or sold back. Every owned/worn instance carries a STABLE per-hero id (minted
// here) because the client addresses a specific instance by id in dress/undress/sell.
// The gamedata catalog (article -> stats/price/slot) lives outside this package; these
// methods only mutate + persist state and are trusted with pre-validated article ids.

// heroItemInstanceBase is the first wearable instance id. It sits well above the small,
// 1-based ids the potion bag synthesizes per response (handleUserBag), so a wearable
// instance id can never be confused with a potion row within one user|bag reply.
const heroItemInstanceBase int32 = 1000

// mintItemIDLocked returns a fresh, never-reused instance id for a hero. Caller holds s.mu.
func mintItemIDLocked(h *Hero) int32 {
	if h.NextItemID < heroItemInstanceBase {
		h.NextItemID = heroItemInstanceBase
	}
	id := h.NextItemID
	h.NextItemID++
	return id
}

// HeroOwned returns a copy of the hero's unequipped wearables (nil if no account/hero).
func (s *Store) HeroOwned(userID int32) []OwnedItem {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, found := s.usersByID[userID]
	if !found || u.Hero == nil {
		return nil
	}
	out := make([]OwnedItem, len(u.Hero.Owned))
	copy(out, u.Hero.Owned)
	return out
}

// HeroDressed returns a copy of the hero's equipped gear (nil if no account/hero). Used by
// the Battle server to fold each dressed article's stats into the avatar at battle entry.
func (s *Store) HeroDressed(userID int32) []DressedItem {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, found := s.usersByID[userID]
	if !found || u.Hero == nil {
		return nil
	}
	out := make([]DressedItem, len(u.Hero.Dressed))
	copy(out, u.Hero.Dressed)
	return out
}

// HeroLevelRace returns a hero's persistent level and race UNDER the lock, so a Ctrl
// buy/dress gate reads them without racing the Battle server's AddHeroExp (which writes
// Level under s.mu on another goroutine). ok=false if the user/hero is missing.
func (s *Store) HeroLevelRace(userID int32) (level, race int32, ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, found := s.usersByID[userID]
	if !found || u.Hero == nil {
		return 0, 0, false
	}
	return u.Hero.Level, u.Hero.Race, true
}

// BuyWearables atomically debits totalPrice from the hero's gold and mints one owned
// instance per article, appending them to Owned. It is the affordability gate: ok=false
// with no change when the hero is missing, totalPrice is negative, or gold is insufficient.
// Returns the new money/diamond totals and the freshly-minted instances (for the
// user|update_bag_mpd delivery push). Callers pre-validate each article id.
func (s *Store) BuyWearables(userID, totalPrice int32, articles []int32) (money, diamonds int32, added []OwnedItem, ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, found := s.usersByID[userID]
	if !found || u.Hero == nil {
		return 0, 0, nil, false
	}
	h := u.Hero
	if totalPrice < 0 || h.Money < totalPrice {
		return h.Money, h.DiamondMoney, nil, false
	}
	h.Money -= totalPrice
	added = make([]OwnedItem, 0, len(articles))
	for _, art := range articles {
		it := OwnedItem{ID: mintItemIDLocked(h), ArticleID: art}
		h.Owned = append(h.Owned, it)
		added = append(added, it)
	}
	s.saveUserLocked(u)
	return h.Money, h.DiamondMoney, added, true
}

// SellWearable removes one owned (unequipped) instance and credits refund to the hero's
// gold. ok=false with no change if the instance is not in Owned (already dressed, sold, or
// never owned). Returns the sold instance's article id and the new balances.
func (s *Store) SellWearable(userID, instanceID, refund int32) (money, diamonds, articleID int32, ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, found := s.usersByID[userID]
	if !found || u.Hero == nil {
		return 0, 0, 0, false
	}
	h := u.Hero
	for i := range h.Owned {
		if h.Owned[i].ID == instanceID {
			articleID = h.Owned[i].ArticleID
			h.Owned = append(h.Owned[:i], h.Owned[i+1:]...)
			if refund > 0 {
				h.Money += refund
			}
			s.saveUserLocked(u)
			return h.Money, h.DiamondMoney, articleID, true
		}
	}
	return h.Money, h.DiamondMoney, 0, false
}

// DressResult reports the outcome of a DressWearable move so the caller can build the
// wire replies: the article now worn, and (if the target slot was occupied) the instance
// that was displaced back into the bag.
type DressResult struct {
	DressedArticle int32
	Displaced      bool
	DisplacedID    int32
	DisplacedArtcl int32
}

// DressWearable moves an owned instance into the given paperdoll slot. If the slot already
// holds an item, that item is moved back into Owned (swap) and reported in the result so
// the caller can refresh the bag. ok=false with no change if instanceID is not owned.
// Slot/kind/level/race validation happens in the caller (which has the gamedata catalog);
// this method re-checks only ownership under the lock.
func (s *Store) DressWearable(userID, instanceID, slot int32) (res DressResult, ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, found := s.usersByID[userID]
	if !found || u.Hero == nil {
		return res, false
	}
	h := u.Hero
	idx := -1
	for i := range h.Owned {
		if h.Owned[i].ID == instanceID {
			idx = i
			break
		}
	}
	if idx < 0 {
		return res, false
	}
	article := h.Owned[idx].ArticleID
	h.Owned = append(h.Owned[:idx], h.Owned[idx+1:]...)
	// Evict any current occupant of the slot back into the bag.
	for i := range h.Dressed {
		if h.Dressed[i].Slot == slot {
			res.Displaced = true
			res.DisplacedID = h.Dressed[i].ID
			res.DisplacedArtcl = h.Dressed[i].ArticleID
			h.Owned = append(h.Owned, OwnedItem{ID: h.Dressed[i].ID, ArticleID: h.Dressed[i].ArticleID})
			h.Dressed = append(h.Dressed[:i], h.Dressed[i+1:]...)
			break
		}
	}
	h.Dressed = append(h.Dressed, DressedItem{ID: instanceID, ArticleID: article, Count: 1, Slot: slot})
	res.DressedArticle = article
	s.saveUserLocked(u)
	return res, true
}

// UndressWearable moves the item worn in slot back into Owned. ok=false with no change if
// no item occupies that slot. Returns the freed instance's id and article.
func (s *Store) UndressWearable(userID, slot int32) (freedID, freedArticle int32, ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	u, found := s.usersByID[userID]
	if !found || u.Hero == nil {
		return 0, 0, false
	}
	h := u.Hero
	for i := range h.Dressed {
		if h.Dressed[i].Slot == slot {
			freedID = h.Dressed[i].ID
			freedArticle = h.Dressed[i].ArticleID
			h.Owned = append(h.Owned, OwnedItem{ID: freedID, ArticleID: freedArticle})
			h.Dressed = append(h.Dressed[:i], h.Dressed[i+1:]...)
			s.saveUserLocked(u)
			return freedID, freedArticle, true
		}
	}
	return 0, 0, false
}
