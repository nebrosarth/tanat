package session

import (
	"path/filepath"
	"testing"
)

// clanHero registers an account, gives it a hero with the given money, and returns it.
func clanHero(t *testing.T, s *Store, email string, money int32) *User {
	t.Helper()
	u, _, ok := s.LoginOrRegister(email, "pw")
	if !ok {
		t.Fatalf("login %s failed", email)
	}
	s.CreateHero(u, 1, false, 0, 0, 0, 0, 0)
	u.Hero.Money = money
	return u
}

// clanJoin makes member join head's clan (via the real invite/answer path).
func clanJoin(t *testing.T, s *Store, head, member *User) {
	t.Helper()
	if _, _, c := s.RecordClanInvite(head.ID, member.ID); c != ClanOK {
		t.Fatalf("invite of %d failed: %d", member.ID, c)
	}
	if _, _, joined, c := s.AnswerClanInvite(member.ID, true); c != ClanOK || !joined {
		t.Fatalf("answer for %d failed: code=%d joined=%v", member.ID, c, joined)
	}
}

func TestCreateClanChargesAndMakesHead(t *testing.T) {
	s := NewStore()
	u := clanHero(t, s, "founder@x", 250000)
	id, code := s.CreateClan(u.ID, "TestClan", "TC")
	if code != ClanOK || id == 0 {
		t.Fatalf("create: id=%d code=%d", id, code)
	}
	if u.Hero.Money != 250000-ClanCreatePrice {
		t.Errorf("money after create = %d, want %d", u.Hero.Money, 250000-ClanCreatePrice)
	}
	if u.Hero.ClanID != id || u.Hero.ClanRole != ClanRoleHead {
		t.Errorf("founder clan=%d role=%d, want clan=%d role=HEAD", u.Hero.ClanID, u.Hero.ClanRole, id)
	}

	poor := clanHero(t, s, "poor@x", 100)
	if _, code := s.CreateClan(poor.ID, "PoorClan", "PC"); code != ClanErrNotEnoughMoney {
		t.Errorf("underfunded create code = %d, want 7010", code)
	}
	if poor.Hero.ClanID != 0 {
		t.Error("underfunded founder should not have joined a clan")
	}
}

func TestCreateClanUniqueAndValidated(t *testing.T) {
	s := NewStore()
	a := clanHero(t, s, "a@x", 500000)
	if _, c := s.CreateClan(a.ID, "Alpha", "AL"); c != ClanOK {
		t.Fatalf("first create failed: %d", c)
	}
	b := clanHero(t, s, "b@x", 500000)
	if _, c := s.CreateClan(b.ID, "Alpha", "BB"); c != ClanErrNameExist {
		t.Errorf("dup name code = %d, want 7014", c)
	}
	if _, c := s.CreateClan(b.ID, "Beta", "al"); c != ClanErrTagExist { // NOCASE
		t.Errorf("dup tag (case-insensitive) code = %d, want 7013", c)
	}
	if _, c := s.CreateClan(b.ID, "no", "BB"); c != ClanErrBadName {
		t.Errorf("short name code = %d, want 7012", c)
	}
	if _, c := s.CreateClan(b.ID, "GoodName", "x"); c != ClanErrBadTag {
		t.Errorf("short tag code = %d, want 7011", c)
	}
	if _, c := s.CreateClan(b.ID, "MixedКир", "BB"); c != ClanErrBadName {
		t.Errorf("mixed-alphabet name code = %d, want 7012", c)
	}
}

func TestClanHeadCannotLeave(t *testing.T) {
	s := NewStore()
	u := clanHero(t, s, "head@x", 300000)
	s.CreateClan(u.ID, "HeadClan", "HC")
	if c := s.RemoveClanMember(u.ID, u.ID); c != ClanErrHeadNeeded {
		t.Errorf("head self-leave code = %d, want 7028", c)
	}
}

func TestClanKickSuperiority(t *testing.T) {
	s := NewStore()
	head := clanHero(t, s, "h@x", 300000)
	s.CreateClan(head.ID, "KickClan", "KC")
	m1 := clanHero(t, s, "m1@x", 1000)
	m2 := clanHero(t, s, "m2@x", 1000)
	clanJoin(t, s, head, m1)
	clanJoin(t, s, head, m2)

	// A warrior cannot kick a peer warrior.
	if c := s.RemoveClanMember(m1.ID, m2.ID); c != ClanErrForbidden {
		t.Errorf("warrior kicking peer code = %d, want 7019", c)
	}
	// Promote m1 to RECRUITER; now it can kick the warrior m2.
	if c := s.ChangeClanRoles(head.ID, map[int32]int32{m1.ID: ClanRoleRecruiter}); c != ClanOK {
		t.Fatalf("promote failed: %d", c)
	}
	if c := s.RemoveClanMember(m1.ID, m2.ID); c != ClanOK {
		t.Fatalf("recruiter kicking warrior code = %d, want ok", c)
	}
	if m2.Hero.ClanID != 0 {
		t.Error("kicked member still shows a clan")
	}
}

func TestClanHeadTransferKeepsSingleHead(t *testing.T) {
	s := NewStore()
	head := clanHero(t, s, "h@x", 300000)
	id, _ := s.CreateClan(head.ID, "XferClan", "XC")
	m1 := clanHero(t, s, "m1@x", 1000)
	clanJoin(t, s, head, m1)

	if c := s.ChangeClanRoles(head.ID, map[int32]int32{m1.ID: ClanRoleHead}); c != ClanOK {
		t.Fatalf("headship transfer code = %d, want ok", c)
	}
	if m1.Hero.ClanRole != ClanRoleHead {
		t.Errorf("promoted member role = %d, want HEAD", m1.Hero.ClanRole)
	}
	if head.Hero.ClanRole != ClanRoleDeputy {
		t.Errorf("old head role = %d, want DEPUTY (auto-demote)", head.Hero.ClanRole)
	}
	view, _ := s.ClanByID(id)
	heads := 0
	for _, m := range view.Members {
		if m.Role == ClanRoleHead {
			heads++
		}
	}
	if heads != 1 {
		t.Errorf("clan has %d HEADs after transfer, want exactly 1", heads)
	}
}

func TestClanInviteAnswerFlow(t *testing.T) {
	s := NewStore()
	head := clanHero(t, s, "h@x", 300000)
	id, _ := s.CreateClan(head.ID, "InvClan", "IC")
	m := clanHero(t, s, "m@x", 1000)

	if _, _, c := s.RecordClanInvite(head.ID, m.ID); c != ClanOK {
		t.Fatalf("invite code = %d", c)
	}
	// A second invite while one is pending is rejected.
	if _, _, c := s.RecordClanInvite(head.ID, m.ID); c != ClanErrAlreadyInvited {
		t.Errorf("double invite code = %d, want 7022", c)
	}
	cid, inviter, joined, c := s.AnswerClanInvite(m.ID, true)
	if c != ClanOK || !joined || cid != id || inviter != head.ID {
		t.Fatalf("answer: clan=%d inviter=%d joined=%v code=%d", cid, inviter, joined, c)
	}
	if m.Hero.ClanID != id || m.Hero.ClanRole != ClanRoleWarrior {
		t.Errorf("joiner clan=%d role=%d, want clan=%d role=WARRIOR", m.Hero.ClanID, m.Hero.ClanRole, id)
	}
	// Answering again (no pending invite) is NOT_INVITED.
	if _, _, _, c := s.AnswerClanInvite(m.ID, true); c != ClanErrNotInvited {
		t.Errorf("answer without invite code = %d, want 7024", c)
	}
}

func TestClanPersistAcrossReopen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "clan.db")
	s1 := NewPersistentStore(path)

	u, _, _ := s1.LoginOrRegister("founder@x", "pw")
	s1.CreateHero(u, 1, false, 0, 0, 0, 0, 0)
	s1.AddHeroMoney(u.ID, ClanCreatePrice) // starter wallet + enough to afford it
	id, code := s1.CreateClan(u.ID, "PersistClan", "PC")
	if code != ClanOK {
		t.Fatalf("create failed: %d (money=%v)", code, u.Hero.Money)
	}
	m, _, _ := s1.LoginOrRegister("member@x", "pw")
	s1.CreateHero(m, 2, false, 0, 0, 0, 0, 0)
	clanJoin(t, s1, u, m)
	s1.Close()

	s2 := NewPersistentStore(path)
	defer s2.Close()
	view, ok := s2.ClanByID(id)
	if !ok {
		t.Fatal("clan missing after reopen")
	}
	if view.Name != "PersistClan" || view.Tag != "PC" {
		t.Errorf("reopened clan = %q/%q, want PersistClan/PC", view.Name, view.Tag)
	}
	if len(view.Members) != 2 {
		t.Fatalf("reopened roster has %d members, want 2", len(view.Members))
	}
	mu, _ := s2.ByID(u.ID)
	if mu.Hero.ClanID != id || mu.Hero.ClanRole != ClanRoleHead {
		t.Errorf("founder membership not persisted: clan=%d role=%d", mu.Hero.ClanID, mu.Hero.ClanRole)
	}
	if mm, _ := s2.ByID(m.ID); mm.Hero.ClanID != id || mm.Hero.ClanRole != ClanRoleWarrior {
		t.Errorf("member membership not persisted: clan=%d role=%d", mm.Hero.ClanID, mm.Hero.ClanRole)
	}
}
