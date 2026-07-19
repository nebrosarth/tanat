package session

// SQLITE PERSISTENCE. The store keeps the full account/hero object graph in
// memory as the live working set (so every caller that holds a *User/*Hero
// pointer keeps working), and write-through-persists each mutated account to a
// single SQLite database via saveUserLocked. Only PERSISTENT state lives in the
// DB: accounts, heroes, bags, owned/dressed gear, quests, and the social lists.
// Transient state (sessions, pending battles, lobby area, parties, friend
// REQUESTS) stays in memory and is intentionally not persisted -- clients
// re-login and rejoin.
//
// The driver is modernc.org/sqlite (pure Go, no cgo) so the server stays a
// plain `go build` with no C toolchain, and the DB links statically into the
// exe. All store access is already serialized under s.mu, and the connection
// pool is pinned to a single connection, so the app never contends with itself
// for the database.

import (
	"database/sql"
	"log"
	"os"
	"path/filepath"

	_ "modernc.org/sqlite" // registers the "sqlite" database/sql driver
)

// schemaStmts is the DDL, one statement per element (executed in order so a
// child table's foreign key can reference an already-created parent). Every
// child table cascades on account deletion, so an admin "delete account" is a
// single DELETE FROM users.
var schemaStmts = []string{
	`CREATE TABLE IF NOT EXISTS meta (
		key   TEXT PRIMARY KEY,
		value TEXT NOT NULL
	)`,
	`CREATE TABLE IF NOT EXISTS users (
		id            INTEGER PRIMARY KEY,
		email         TEXT NOT NULL UNIQUE,
		password_hash TEXT NOT NULL DEFAULT '',
		username      TEXT NOT NULL DEFAULT '',
		created_at    INTEGER NOT NULL DEFAULT 0,
		banned        INTEGER NOT NULL DEFAULT 0
	)`,
	`CREATE TABLE IF NOT EXISTS heroes (
		user_id       INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
		race          INTEGER NOT NULL DEFAULT 0,
		gender        INTEGER NOT NULL DEFAULT 0,
		face          INTEGER NOT NULL DEFAULT 0,
		hair          INTEGER NOT NULL DEFAULT 0,
		dist_mark     INTEGER NOT NULL DEFAULT 0,
		skin_color    INTEGER NOT NULL DEFAULT 0,
		hair_color    INTEGER NOT NULL DEFAULT 0,
		money         INTEGER NOT NULL DEFAULT 0,
		diamond_money INTEGER NOT NULL DEFAULT 0,
		level         INTEGER NOT NULL DEFAULT 1,
		exp           INTEGER NOT NULL DEFAULT 0,
		next_exp      INTEGER NOT NULL DEFAULT 0,
		next_item_id  INTEGER NOT NULL DEFAULT 0
	)`,
	`CREATE TABLE IF NOT EXISTS bag_items (
		user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
		article_id INTEGER NOT NULL,
		count      INTEGER NOT NULL,
		PRIMARY KEY (user_id, article_id)
	)`,
	`CREATE TABLE IF NOT EXISTS owned_items (
		user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
		instance_id INTEGER NOT NULL,
		article_id  INTEGER NOT NULL,
		PRIMARY KEY (user_id, instance_id)
	)`,
	`CREATE TABLE IF NOT EXISTS dressed_items (
		user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
		slot        INTEGER NOT NULL,
		instance_id INTEGER NOT NULL,
		article_id  INTEGER NOT NULL,
		count       INTEGER NOT NULL,
		PRIMARY KEY (user_id, slot)
	)`,
	`CREATE TABLE IF NOT EXISTS quests (
		user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
		quest_id       INTEGER NOT NULL,
		status         INTEGER NOT NULL,
		progress       INTEGER NOT NULL,
		cooldown_until INTEGER NOT NULL,
		PRIMARY KEY (user_id, quest_id)
	)`,
	`CREATE TABLE IF NOT EXISTS friends (
		user_id   INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
		friend_id INTEGER NOT NULL,
		PRIMARY KEY (user_id, friend_id)
	)`,
	`CREATE TABLE IF NOT EXISTS ignores (
		user_id   INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
		ignore_id INTEGER NOT NULL,
		PRIMARY KEY (user_id, ignore_id)
	)`,
	`INSERT INTO meta(key, value) VALUES('schema_version', '1')
		ON CONFLICT(key) DO NOTHING`,
}

// NewPersistentStore opens (creating if absent) the SQLite database at path,
// loads every account into memory, and -- on a brand-new DB with a legacy
// accounts.json sibling -- imports that JSON once. A DB that can't be opened
// degrades to an in-memory store (logged) rather than taking the server down.
func NewPersistentStore(path string) *Store {
	s := NewStore()
	s.path = path

	db, err := openDB(path)
	if err != nil {
		log.Printf("session: could not open sqlite %s: %v (in-memory only)", path, err)
		return s
	}
	s.db = db

	if err := s.loadAllLocked(); err != nil {
		log.Printf("session: load from %s failed: %v", path, err)
	}

	// One-time migration: a fresh DB next to an old accounts.json inherits it.
	if len(s.usersByID) == 0 {
		if jsonPath := filepath.Join(filepath.Dir(path), "accounts.json"); jsonPath != path {
			if _, statErr := os.Stat(jsonPath); statErr == nil {
				if n, impErr := s.importLegacyJSONLocked(jsonPath); impErr != nil {
					log.Printf("session: legacy import from %s failed: %v", jsonPath, impErr)
				} else {
					log.Printf("session: imported %d account(s) from %s into sqlite", n, jsonPath)
					if renErr := os.Rename(jsonPath, jsonPath+".imported"); renErr != nil {
						log.Printf("session: could not archive %s: %v", jsonPath, renErr)
					}
				}
			}
		}
	}

	log.Printf("session: sqlite store %s ready (%d account(s))", path, len(s.usersByID))
	return s
}

// openDB opens the database, pins a single connection (all store access is
// serialized under s.mu, so one connection never contends), applies the
// durability/concurrency pragmas, and runs the schema DDL.
func openDB(path string) (*sql.DB, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	// A single connection sidesteps per-connection PRAGMA drift and any
	// "database is locked" contention entirely (the app already serializes).
	db.SetMaxOpenConns(1)
	for _, pragma := range []string{
		"PRAGMA busy_timeout=5000",
		"PRAGMA journal_mode=WAL",
		"PRAGMA synchronous=NORMAL",
		"PRAGMA foreign_keys=ON",
	} {
		if _, err := db.Exec(pragma); err != nil {
			db.Close()
			return nil, err
		}
	}
	for _, stmt := range schemaStmts {
		if _, err := db.Exec(stmt); err != nil {
			db.Close()
			return nil, err
		}
	}
	if err := migrateSchema(db); err != nil {
		db.Close()
		return nil, err
	}
	return db, nil
}

// migrateSchema brings an EXISTING database (created by an older build, whose
// CREATE TABLE IF NOT EXISTS therefore left it untouched) up to the current
// column set. Each entry is applied only when the column is actually missing, so
// this is idempotent and safe to run on every boot. Adding a column to a large
// SQLite table is a cheap metadata-only operation.
func migrateSchema(db *sql.DB) error {
	adds := []struct{ table, column, ddl string }{
		{"users", "banned", "ALTER TABLE users ADD COLUMN banned INTEGER NOT NULL DEFAULT 0"},
	}
	for _, a := range adds {
		has, err := columnExists(db, a.table, a.column)
		if err != nil {
			return err
		}
		if has {
			continue
		}
		if _, err := db.Exec(a.ddl); err != nil {
			return err
		}
		log.Printf("session: migrated schema: added %s.%s", a.table, a.column)
	}
	return nil
}

// columnExists reports whether table has a column of the given name, via
// PRAGMA table_info (which lists one row per column).
func columnExists(db *sql.DB, table, column string) (bool, error) {
	rows, err := db.Query("PRAGMA table_info(" + table + ")")
	if err != nil {
		return false, err
	}
	defer rows.Close()
	for rows.Next() {
		// PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk.
		var cid int
		var name, ctype string
		var notnull, pk int
		var dflt sql.NullString
		if err := rows.Scan(&cid, &name, &ctype, &notnull, &dflt, &pk); err != nil {
			return false, err
		}
		if name == column {
			return true, nil
		}
	}
	return false, rows.Err()
}

// Close releases the database handle (no-op for an in-memory store). Callers
// should Close on shutdown; tests must Close before their temp dir is removed
// (Windows can't delete a file with an open handle).
func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.db == nil {
		return nil
	}
	err := s.db.Close()
	s.db = nil
	return err
}

// loadAllLocked reconstructs the in-memory account graph from the database.
// Called once at open (before any concurrency), so it does not take s.mu.
func (s *Store) loadAllLocked() error {
	if s.db == nil {
		return nil
	}

	rows, err := s.db.Query(`SELECT id, email, password_hash, username, created_at, banned FROM users`)
	if err != nil {
		return err
	}
	for rows.Next() {
		u := &User{}
		var banned int32
		if err := rows.Scan(&u.ID, &u.Email, &u.PassHash, &u.Username, &u.CreatedAt, &banned); err != nil {
			rows.Close()
			return err
		}
		u.Banned = banned != 0
		s.usersByEmail[u.Email] = u
		s.usersByID[u.ID] = u
		if u.ID >= s.nextUserID {
			s.nextUserID = u.ID + 1
		}
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return err
	}

	// Honor the durably-stored id high-water mark if it is ahead of max(id)+1
	// (it will be after an account was deleted, leaving a gap we must not reuse).
	var stored int32
	if err := s.db.QueryRow(`SELECT value FROM meta WHERE key=?`, metaNextUserID).Scan(&stored); err == nil {
		if stored > s.nextUserID {
			s.nextUserID = stored
		}
	}

	hrows, err := s.db.Query(`SELECT user_id, race, gender, face, hair, dist_mark, skin_color,
		hair_color, money, diamond_money, level, exp, next_exp, next_item_id FROM heroes`)
	if err != nil {
		return err
	}
	for hrows.Next() {
		var uid, gender int32
		h := &Hero{}
		if err := hrows.Scan(&uid, &h.Race, &gender, &h.Face, &h.Hair, &h.DistMark,
			&h.SkinColor, &h.HairColor, &h.Money, &h.DiamondMoney, &h.Level,
			&h.Exp, &h.NextExp, &h.NextItemID); err != nil {
			hrows.Close()
			return err
		}
		h.ID = uid // Hero.ID == User.ID by design (SelfHero.Hero lookup by user id).
		h.Gender = gender != 0
		if u := s.usersByID[uid]; u != nil {
			h.sanitize()
			u.Hero = h
			u.HasHero = true
		}
	}
	hrows.Close()
	if err := hrows.Err(); err != nil {
		return err
	}

	if err := s.loadChild(`SELECT user_id, article_id, count FROM bag_items ORDER BY user_id, article_id`,
		func(rows *sql.Rows) error {
			var uid, art, cnt int32
			if err := rows.Scan(&uid, &art, &cnt); err != nil {
				return err
			}
			if h := s.heroOf(uid); h != nil {
				h.Bag = append(h.Bag, BagItem{ArticleID: art, Count: cnt})
			}
			return nil
		}); err != nil {
		return err
	}

	if err := s.loadChild(`SELECT user_id, instance_id, article_id FROM owned_items ORDER BY user_id, instance_id`,
		func(rows *sql.Rows) error {
			var uid, iid, art int32
			if err := rows.Scan(&uid, &iid, &art); err != nil {
				return err
			}
			if h := s.heroOf(uid); h != nil {
				h.Owned = append(h.Owned, OwnedItem{ID: iid, ArticleID: art})
			}
			return nil
		}); err != nil {
		return err
	}

	if err := s.loadChild(`SELECT user_id, slot, instance_id, article_id, count FROM dressed_items ORDER BY user_id, slot`,
		func(rows *sql.Rows) error {
			var uid, slot, iid, art, cnt int32
			if err := rows.Scan(&uid, &slot, &iid, &art, &cnt); err != nil {
				return err
			}
			if h := s.heroOf(uid); h != nil {
				h.Dressed = append(h.Dressed, DressedItem{ID: iid, ArticleID: art, Count: cnt, Slot: slot})
			}
			return nil
		}); err != nil {
		return err
	}

	if err := s.loadChild(`SELECT user_id, quest_id, status, progress, cooldown_until FROM quests ORDER BY user_id, quest_id`,
		func(rows *sql.Rows) error {
			var uid int32
			var qs QuestState
			if err := rows.Scan(&uid, &qs.QuestID, &qs.Status, &qs.Progress, &qs.CooldownUntil); err != nil {
				return err
			}
			if h := s.heroOf(uid); h != nil {
				h.Quests = append(h.Quests, qs)
			}
			return nil
		}); err != nil {
		return err
	}

	if err := s.loadChild(`SELECT user_id, friend_id FROM friends ORDER BY user_id, friend_id`,
		func(rows *sql.Rows) error {
			var uid, fid int32
			if err := rows.Scan(&uid, &fid); err != nil {
				return err
			}
			if u := s.usersByID[uid]; u != nil {
				u.Friends = append(u.Friends, fid)
			}
			return nil
		}); err != nil {
		return err
	}

	if err := s.loadChild(`SELECT user_id, ignore_id FROM ignores ORDER BY user_id, ignore_id`,
		func(rows *sql.Rows) error {
			var uid, iid int32
			if err := rows.Scan(&uid, &iid); err != nil {
				return err
			}
			if u := s.usersByID[uid]; u != nil {
				u.Ignores = append(u.Ignores, iid)
			}
			return nil
		}); err != nil {
		return err
	}

	return nil
}

// heroOf returns the in-memory hero for a user id, or nil.
func (s *Store) heroOf(uid int32) *Hero {
	if u := s.usersByID[uid]; u != nil {
		return u.Hero
	}
	return nil
}

// loadChild runs query and hands each row to apply, which scans it directly
// (scanning the user_id itself, so it dispatches to the right owner).
func (s *Store) loadChild(query string, apply func(rows *sql.Rows) error) error {
	rows, err := s.db.Query(query)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		if err := apply(rows); err != nil {
			return err
		}
	}
	return rows.Err()
}

// persistUsersLocked write-throughs every given account's full aggregate (row,
// hero, bag, owned/dressed gear, quests, social lists) in ONE transaction, so a
// multi-account change is all-or-nothing. Nil users are skipped; a no-op (nil
// error) for an in-memory store. Caller holds s.mu.
func (s *Store) persistUsersLocked(us ...*User) error {
	if s.db == nil {
		return nil
	}
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	for _, u := range us {
		if u == nil {
			continue
		}
		if err := writeUserTx(tx, u); err != nil {
			_ = tx.Rollback()
			return err
		}
	}
	return tx.Commit()
}

// saveUserLocked persists one account, logging (but not surfacing) a failure --
// the in-memory state stays authoritative and the next mutation rewrites the
// whole aggregate, so a transient write error self-heals. Callers that must know
// whether the write survived (the one-shot legacy import) use persistUsersLocked.
func (s *Store) saveUserLocked(u *User) {
	if err := s.persistUsersLocked(u); err != nil {
		id := int32(0)
		if u != nil {
			id = u.ID
		}
		log.Printf("session: persist user %d failed: %v", id, err)
	}
}

// saveAllLocked flushes every account in one transaction (used by the public
// Save and by tests poking a hero directly).
func (s *Store) saveAllLocked() {
	if s.db == nil {
		return
	}
	all := make([]*User, 0, len(s.usersByID))
	for _, u := range s.usersByID {
		all = append(all, u)
	}
	if err := s.persistUsersLocked(all...); err != nil {
		log.Printf("session: flush-all failed: %v", err)
	}
}

// metaNextUserID is the meta key holding the id counter's high-water mark, so a
// gap left by (future) account deletion is never re-handed to a new account.
const metaNextUserID = "next_user_id"

// persistNextUserIDLocked durably records the current id counter. Best-effort:
// on failure loadAllLocked still recovers max(id)+1, which is correct unless the
// highest account was deleted -- a case this counter exists to survive.
func (s *Store) persistNextUserIDLocked() {
	if s.db == nil {
		return
	}
	if _, err := s.db.Exec(
		`INSERT INTO meta(key, value) VALUES(?, ?)
		 ON CONFLICT(key) DO UPDATE SET value=excluded.value`,
		metaNextUserID, s.nextUserID); err != nil {
		log.Printf("session: persist next_user_id failed: %v", err)
	}
}

// writeUserTx upserts the account row, then rewrites the hero and every child
// collection (delete-then-insert, the aggregate-root pattern -- simple and
// obviously correct for the handful of rows a hero owns). The users row is
// written first so child foreign keys always find their parent.
func writeUserTx(tx *sql.Tx, u *User) error {
	if _, err := tx.Exec(
		`INSERT INTO users(id, email, password_hash, username, created_at, banned)
		 VALUES(?, ?, ?, ?, ?, ?)
		 ON CONFLICT(id) DO UPDATE SET
		   email=excluded.email,
		   password_hash=excluded.password_hash,
		   username=excluded.username,
		   banned=excluded.banned`,
		u.ID, u.Email, u.PassHash, u.Username, u.CreatedAt, boolToInt(u.Banned)); err != nil {
		return err
	}

	if u.Hero == nil {
		// No hero: drop the hero row and anything that hangs off it.
		for _, tbl := range []string{"heroes", "bag_items", "owned_items", "dressed_items", "quests"} {
			if _, err := tx.Exec("DELETE FROM "+tbl+" WHERE user_id=?", u.ID); err != nil {
				return err
			}
		}
	} else {
		h := u.Hero
		if _, err := tx.Exec(
			`INSERT INTO heroes(user_id, race, gender, face, hair, dist_mark, skin_color,
			   hair_color, money, diamond_money, level, exp, next_exp, next_item_id)
			 VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			 ON CONFLICT(user_id) DO UPDATE SET
			   race=excluded.race, gender=excluded.gender, face=excluded.face, hair=excluded.hair,
			   dist_mark=excluded.dist_mark, skin_color=excluded.skin_color, hair_color=excluded.hair_color,
			   money=excluded.money, diamond_money=excluded.diamond_money, level=excluded.level,
			   exp=excluded.exp, next_exp=excluded.next_exp, next_item_id=excluded.next_item_id`,
			u.ID, h.Race, boolToInt(h.Gender), h.Face, h.Hair, h.DistMark, h.SkinColor,
			h.HairColor, h.Money, h.DiamondMoney, h.Level, h.Exp, h.NextExp, h.NextItemID); err != nil {
			return err
		}
		if err := rewriteChild(tx, "bag_items", u.ID,
			`INSERT INTO bag_items(user_id, article_id, count) VALUES(?, ?, ?)`,
			len(h.Bag), func(i int) []any { return []any{u.ID, h.Bag[i].ArticleID, h.Bag[i].Count} }); err != nil {
			return err
		}
		if err := rewriteChild(tx, "owned_items", u.ID,
			`INSERT INTO owned_items(user_id, instance_id, article_id) VALUES(?, ?, ?)`,
			len(h.Owned), func(i int) []any { return []any{u.ID, h.Owned[i].ID, h.Owned[i].ArticleID} }); err != nil {
			return err
		}
		if err := rewriteChild(tx, "dressed_items", u.ID,
			`INSERT INTO dressed_items(user_id, slot, instance_id, article_id, count) VALUES(?, ?, ?, ?, ?)`,
			len(h.Dressed), func(i int) []any {
				return []any{u.ID, h.Dressed[i].Slot, h.Dressed[i].ID, h.Dressed[i].ArticleID, h.Dressed[i].Count}
			}); err != nil {
			return err
		}
		if err := rewriteChild(tx, "quests", u.ID,
			`INSERT INTO quests(user_id, quest_id, status, progress, cooldown_until) VALUES(?, ?, ?, ?, ?)`,
			len(h.Quests), func(i int) []any {
				return []any{u.ID, h.Quests[i].QuestID, h.Quests[i].Status, h.Quests[i].Progress, h.Quests[i].CooldownUntil}
			}); err != nil {
			return err
		}
	}

	// Social lists persist with the account, independent of the hero.
	if err := rewriteChild(tx, "friends", u.ID,
		`INSERT INTO friends(user_id, friend_id) VALUES(?, ?)`,
		len(u.Friends), func(i int) []any { return []any{u.ID, u.Friends[i]} }); err != nil {
		return err
	}
	if err := rewriteChild(tx, "ignores", u.ID,
		`INSERT INTO ignores(user_id, ignore_id) VALUES(?, ?)`,
		len(u.Ignores), func(i int) []any { return []any{u.ID, u.Ignores[i]} }); err != nil {
		return err
	}
	return nil
}

// rewriteChild clears a user's rows in one child table and re-inserts n of them,
// sourcing each insert's args from argsAt(i).
func rewriteChild(tx *sql.Tx, table string, uid int32, insertSQL string, n int, argsAt func(i int) []any) error {
	if _, err := tx.Exec("DELETE FROM "+table+" WHERE user_id=?", uid); err != nil {
		return err
	}
	for i := 0; i < n; i++ {
		if _, err := tx.Exec(insertSQL, argsAt(i)...); err != nil {
			return err
		}
	}
	return nil
}

func boolToInt(b bool) int32 {
	if b {
		return 1
	}
	return 0
}
