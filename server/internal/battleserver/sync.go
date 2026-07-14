package battleserver

import (
	"bytes"
	"encoding/binary"
	"sort"
)

// This file mirrors the client's object-tracking and SYNC (518) blob format
// (TrackingIdManager + SyncPacket.Parse) for battles with multiple objects.
//
// The client keeps a per-connection list of "tracked ids"; every SYNC blob
// addresses objects by their INDEX in that list, so the server must maintain
// an exact mirror: adds append (index = count at that moment), removals
// swap-with-last. Getting one removal wrong desyncs every later index.

// tracker mirrors TanatKernel.TrackingIdManager.
type tracker struct {
	ids []int32 // index = tracking index
}

// add registers a new tracked object and returns its tracking index.
func (t *tracker) add(id int32) int {
	t.ids = append(t.ids, id)
	return len(t.ids) - 1
}

// index returns the current tracking index of id, or -1.
func (t *tracker) index(id int32) int {
	for i, v := range t.ids {
		if v == id {
			return i
		}
	}
	return -1
}

// remove drops id via the client's swap-with-last rule and returns the index
// the removal entry must carry (-1 if the id was not tracked).
func (t *tracker) remove(id int32) int {
	i := t.index(id)
	if i < 0 {
		return -1
	}
	last := len(t.ids) - 1
	t.ids[i] = t.ids[last]
	t.ids = t.ids[:last]
	return i
}

func (t *tracker) count() int { return len(t.ids) }

// newIds entry flag bits (TrackingIdManager).
const (
	syncAddMask uint32 = 0x80000000 // payload = object id: register + visible
	syncRemMask uint32 = 0x40000000 // payload = tracking INDEX: swap-remove
)

// syncBlob accumulates one SYNC "data" payload: visibility entries plus typed
// values addressed by tracking index. Layout produced by build() (little
// endian, consumed exactly by SyncPacket.Parse):
//
//	float32  time
//	int16    newIds count
//	int32    newIds[] (flagged: add=objId|0x80000000, remove=index|0x40000000)
//	uint64   type mask
//	per set type bit, ascending:
//	  byte[W]  object bitmask, W = ceil(trackedCount/8), LSB-first bit = index
//	  values   per set index ascending: POSITION 5xfloat32, POS_ANGLE 3x,
//	           TEAM/immunity types 1xint32, everything else 1xfloat32
type syncBlob struct {
	time float32
	news []uint32
	f    map[uint64]map[int][]float32
	i    map[uint64]map[int]int32
}

func newSyncBlob(t float32) *syncBlob {
	return &syncBlob{
		time: t,
		f:    map[uint64]map[int][]float32{},
		i:    map[uint64]map[int]int32{},
	}
}

// addObject emits an add entry for objID (register + first visibility).
func (b *syncBlob) addObject(objID int32) *syncBlob {
	b.news = append(b.news, uint32(objID)|syncAddMask)
	return b
}

// removeIndex emits a removal entry for a tracking index (obtained from
// tracker.remove, which must be called in the same order).
func (b *syncBlob) removeIndex(idx int) *syncBlob {
	b.news = append(b.news, uint32(idx)|syncRemMask)
	return b
}

// setFloats stages float values of one sync type for the object at idx.
func (b *syncBlob) setFloats(typ uint64, idx int, vals ...float32) *syncBlob {
	m, ok := b.f[typ]
	if !ok {
		m = map[int][]float32{}
		b.f[typ] = m
	}
	m[idx] = vals
	return b
}

// setInt stages an int32-encoded sync type (TEAM/MAG_IMM/PHYS_IMM/SILENCE).
func (b *syncBlob) setInt(typ uint64, idx int, v int32) *syncBlob {
	m, ok := b.i[typ]
	if !ok {
		m = map[int]int32{}
		b.i[typ] = m
	}
	m[idx] = v
	return b
}

// position stages a POSITION sample (x, y, velX, velY, snapshot time).
func (b *syncBlob) position(idx int, x, y, vx, vy, t float32) *syncBlob {
	return b.setFloats(syncPosition, idx, x, y, vx, vy, t)
}

// build serializes the blob. trackedCount is the tracker's count AFTER the
// adds/removals staged in this blob (the client sizes the per-type object
// bitmask from its post-update id list).
func (b *syncBlob) build(trackedCount int) []byte {
	buf := new(bytes.Buffer)
	le := binary.LittleEndian
	_ = binary.Write(buf, le, b.time)
	_ = binary.Write(buf, le, int16(len(b.news)))
	for _, n := range b.news {
		_ = binary.Write(buf, le, n)
	}

	var mask uint64
	for typ := range b.f {
		mask |= typ
	}
	for typ := range b.i {
		mask |= typ
	}
	_ = binary.Write(buf, le, mask)

	width := trackedCount / 8
	if trackedCount%8 > 0 {
		width++
	}
	if width == 0 {
		width = 1
	}

	var types []uint64
	for typ := range b.f {
		types = append(types, typ)
	}
	for typ := range b.i {
		types = append(types, typ)
	}
	sort.Slice(types, func(a, c int) bool { return types[a] < types[c] })

	for _, typ := range types {
		bm := make([]byte, width)
		var idxs []int
		if m, ok := b.f[typ]; ok {
			for idx := range m {
				idxs = append(idxs, idx)
			}
		} else {
			for idx := range b.i[typ] {
				idxs = append(idxs, idx)
			}
		}
		sort.Ints(idxs)
		for _, idx := range idxs {
			bm[idx/8] |= 1 << (idx % 8)
		}
		buf.Write(bm)
		for _, idx := range idxs {
			if m, ok := b.f[typ]; ok {
				for _, v := range m[idx] {
					_ = binary.Write(buf, le, v)
				}
			} else {
				_ = binary.Write(buf, le, b.i[typ][idx])
			}
		}
	}
	return buf.Bytes()
}
