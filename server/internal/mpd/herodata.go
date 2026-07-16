package mpd

import (
	"strconv"

	"tanatserver/internal/amf"
	"tanatserver/internal/session"
)

// Hero appearance over MPD: when a player walks into a central square AFTER another
// client has already loaded the scene, that client binds the newcomer's avatar via
// Battle.BindPlayerAvatars -> HeroStore.UpdateHeroesInfo with _send=false (mLoaded is
// already true). _send=false means the client does NOT request the hero's data over
// the Ctrl channel -- it expects the data to be waiting in HeroStore.mHeroesWait,
// populated ONLY by a server-initiated hero.get_data_list_mpd broadcast (see
// HeroStore.OnHeroDataListBroadCast). Without this push the newcomer renders as a
// bodiless nickname. The payload mirrors HeroDataListArgParser.ParseData:
//
//	{ data: { "<heroId>": { load:{race,gender,face,hair,dist_mark,skin_color,hair_color},
//	                        dressed_items:[{id,artikul_id,cnt,slot}],
//	                        clan_info:{id,tag}, user_info:{level,exp,next_exp,rating} } } }

// heroEntry builds the per-hero {load, dressed_items, clan_info, user_info} value the
// client keys by hero id. Mirrors ctrlserver.addHeroData (the Ctrl get_data_list
// response) so a hero looks identical whether fetched or broadcast.
func heroEntry(h *session.Hero) *amf.MixedArray {
	load := amf.NewArray().
		Set("race", h.Race).
		Set("gender", h.Gender).
		Set("face", h.Face).
		Set("hair", h.Hair).
		Set("dist_mark", h.DistMark).
		Set("skin_color", h.SkinColor).
		Set("hair_color", h.HairColor)
	dressed := amf.NewArray()
	for _, it := range h.Dressed {
		dressed.Add(amf.NewArray().
			Set("id", it.ID).
			Set("artikul_id", it.ArticleID).
			Set("cnt", it.Count).
			Set("slot", it.Slot))
	}
	return amf.NewArray().
		Set("load", load).
		Set("dressed_items", dressed).
		Set("clan_info", amf.NewArray().Set("id", int32(-1)).Set("tag", "")).
		Set("user_info", amf.NewArray().
			Set("level", h.Level).
			Set("exp", h.Exp).
			Set("next_exp", h.NextExp).
			Set("rating", int32(0)))
}

// PushHeroData broadcasts the appearance of heroUserIDs to target's MPD socket as
// hero.get_data_list_mpd, seeding the client's mHeroesWait so a post-load avatar bind
// renders the real customized body. No-op if target is offline or nothing resolves.
func (h *Hub) PushHeroData(target int32, heroUserIDs []int32) {
	data := amf.NewArray()
	for _, uid := range heroUserIDs {
		u, ok := h.Store.ByID(uid)
		if !ok || u.Hero == nil {
			continue
		}
		data.Set(strconv.Itoa(int(uid)), heroEntry(u.Hero))
	}
	if len(data.Assoc) == 0 {
		return
	}
	h.Push(target, "hero|get_data_list", amf.NewArray().Set("data", data))
}
