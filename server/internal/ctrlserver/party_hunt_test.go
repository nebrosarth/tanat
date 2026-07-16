package ctrlserver

import (
	"testing"
	"time"

	"tanatserver/internal/amf"
	"tanatserver/internal/gamedata"
)

// postHunt sends a hunt|<action> Ctrl request on behalf of the session.
func postHunt(t *testing.T, url, sess, action string, params *amf.MixedArray, counter int32) *amf.MixedArray {
	t.Helper()
	return postEnvelope(t, url, amf.NewArray().Set("object", "hunt").Set("action", action).
		Set("params", params).Set("sess_uid", int32(0)).Set("sess_key", sess).Set("counter", counter))
}

// TestPartyHuntLeaderInvitesMembers: a group leader joining a hunt fans a
// fight|invite_mpd out to the party members (carrying the map id + leader nick) but
// NOT back to the leader, and a non-leader member joining does not re-invite anyone.
func TestPartyHuntLeaderInvitesMembers(t *testing.T) {
	h := newFriendsHarness(t)
	leaderID, leaderSid := h.login("leader@example.com", 1)
	memberID, memberSid := h.login("member@example.com", 2)

	leaderConn, brLeader := dialMPD(t, h.mpd, leaderID, leaderSid)
	_, brMember := dialMPD(t, h.mpd, memberID, memberSid)
	for i := 0; i < 50 && !(h.hub.Online(leaderID) && h.hub.Online(memberID)); i++ {
		time.Sleep(10 * time.Millisecond)
	}

	// Form a party with leader as leader, member as a member.
	if _, ok := h.srv.Store.JoinGroup(leaderID, memberID); !ok {
		t.Fatal("could not form party")
	}
	leaderUser, _ := h.srv.Store.ByID(leaderID)
	leaderNick := leaderUser.Username

	mapID := gamedata.HuntMaps()[0].ID

	// Leader joins the hunt -> member gets the invite push.
	postHunt(t, h.url, leaderSid, "join", amf.NewArray().Set("map_id", mapID), 10)
	inv := readPushArgs(t, brMember, "fight|invite")
	if got, _ := inv.GetInt("map_id"); got != mapID {
		t.Errorf("invite map_id = %d, want %d", got, mapID)
	}
	if got, _ := inv.GetString("nick"); got != leaderNick {
		t.Errorf("invite nick = %q, want %q", got, leaderNick)
	}

	// The leader must not invite themselves.
	assertNoFrame(t, leaderConn, brLeader, "leader should not receive a hunt invite")

	// The member accepting re-runs hunt|join; as a non-leader they must not re-invite
	// the leader (no invite loop).
	postHunt(t, h.url, memberSid, "join", amf.NewArray().Set("map_id", mapID), 11)
	assertNoFrame(t, leaderConn, brLeader, "member join should not invite the leader")
}
