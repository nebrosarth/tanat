package ctrlserver

import (
	"time"

	"tanatserver/internal/amf"
	"tanatserver/internal/ctrlproto"
	"tanatserver/internal/session"
)

// Chat: the client SENDS a line over the HTTP Ctrl channel (chat|add with {type,
// recipient_list, message}); the server fans it out to the recipients over the MPD
// PUSH channel as chat|message (ChatMsgArgParser reads arguments.{msg,type,ctime,
// from,recipient_list}). The visible result is the push, so the HTTP reply is a bare
// ack. `type` is a ChatChannel enum NAME ("area", "private_msg", "group", "clan",
// ...); recipient_list holds the nicks the sender typed in "[nick]" (private msg). The
// client never echoes locally, so the sender is always in the recipient set to see
// their own line.
func (s *Server) handleChatAdd(req ctrlproto.Request, resp *ctrlproto.Response) {
	resp.Ack("chat", "add")
	u := s.userFromSession(req)
	if u == nil || s.MPD == nil {
		return
	}
	channel := req.Params.StringOr("type", "area")
	msg := req.Params.StringOr("message", "")
	if msg == "" {
		return
	}
	reqNicks := stringArray(req.Params, "recipient_list") // targets the client named

	// A message that names recipients is a directed whisper, whatever tab it was typed
	// in: the "private message" popup (CentralSquareScreen) only prepends "[nick]" to
	// the input via AddRecipientToMsg -- it does NOT switch the send mode off area. So
	// without this, a right-click -> private message goes out as type="area" and the
	// server would broadcast it to the whole square. Promote it to private_msg so it is
	// delivered only to the named nicks and both ends render it in the private tab (the
	// receiver routes purely by this type).
	if len(reqNicks) > 0 {
		channel = "private_msg"
	}

	recipients := s.chatRecipients(u, channel, reqNicks)
	if len(recipients) == 0 {
		recipients = []int32{u.ID} // at least echo to the sender
	}

	// recipient_list is relayed verbatim (the nicks the sender addressed) so the
	// client can render "to [nick]" on a private line.
	nicks := amf.NewArray()
	for _, n := range reqNicks {
		nicks.Add(n)
	}
	// ctime is a unix timestamp (~1.75e9) -- past AMF3's 29-bit integer range, so it
	// must ride as a double (the client reads it via SafeGet<int> -> Convert.ChangeType,
	// which accepts a double). Sent as an int it would be silently masked to 29 bits.
	ctime := float64(time.Now().Unix())
	for _, rid := range recipients {
		s.MPD.Push(rid, "chat|message", amf.NewArray().
			Set("type", channel).
			Set("msg", msg).
			Set("ctime", ctime).
			Set("from", u.Username).
			Set("recipient_list", nicks))
	}
}

// chatRecipients resolves the user ids a message reaches, per channel. Every set
// includes the sender (the client shows its own line only via the echo).
func (s *Server) chatRecipients(u *session.User, channel string, nicks []string) []int32 {
	switch channel {
	case "area", "", "system", "trade":
		// area = everyone in the sender's central square. trade/system are server-wide
		// in the real game; scoping them to the square is a safe stub until a global
		// online roster exists. AreaMembers already includes the sender.
		return s.MPD.AreaMembers(s.MPD.AreaOf(u.ID))
	case "private_msg":
		// Deliver to each addressed nick, plus the sender (to echo their own line).
		out := []int32{u.ID}
		for _, n := range nicks {
			if tu, ok := s.Store.ByUsername(n); ok {
				out = append(out, tu.ID)
			}
		}
		return dedupIDs(out)
	case "group", "team":
		// The sender's party (see session.Group). Solo -> just the sender.
		if g := s.Store.GroupOf(u.ID); g != nil {
			return g.Members
		}
		return []int32{u.ID}
	case "clan":
		// Clan chat needs clan membership, which the server does not model yet (every
		// hero carries clan_info {id:-1}). Until a clan system exists there is no one to
		// fan out to, so the line only echoes to the sender.
		return []int32{u.ID}
	default:
		return []int32{u.ID}
	}
}

// stringArray reads a dense AMF array of strings from params[key] (nil if absent).
func stringArray(params *amf.MixedArray, key string) []string {
	if params == nil {
		return nil
	}
	arr, ok := params.GetArray(key)
	if !ok {
		return nil
	}
	var out []string
	for _, v := range arr.Dense {
		if str, ok := v.(string); ok && str != "" {
			out = append(out, str)
		}
	}
	return out
}

// dedupIDs removes duplicate user ids, preserving order.
func dedupIDs(ids []int32) []int32 {
	seen := map[int32]bool{}
	var out []int32
	for _, id := range ids {
		if !seen[id] {
			seen[id] = true
			out = append(out, id)
		}
	}
	return out
}
