// Package battleproto implements the Battle-channel wire format, reverse
// engineered from TanatKernel's BattlePacketManager, BattlePacket and
// BattleCmdId. Unlike the Ctrl channel (HTTP request/response), the Battle
// channel is a raw TCP stream of length-prefixed chunks:
//
//	[chunkSize:4 BE][pktSize:4 BE][pkt AMF][pktSize:4 BE][pkt AMF]...
//
// where chunkSize = sum over packets of (4 + pktSize). Each packet body is a
// single AMF value: a MixedArray {cmdId:int, arguments:MixedArray,
// requestId:int, status?:bool, error?:string} (see BattlePacket.Serialize and
// its Variable ctor).
//
// The channel's AMF Formatter keeps ONE string-reference table alive for the
// whole connection (BattlePacketManager only clears it on Clear()), so packets
// are decoded with a single connection-scoped Decoder (Reset preserves the ref
// table) and encoded with a no-ref Encoder (see amf.NewRawEncoder).
package battleproto

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"

	"tanatserver/internal/amf"
)

// CmdID mirrors TanatKernel.BattleCmdId.
type CmdID int32

const (
	CmdEcho          CmdID = 0
	CmdConnect       CmdID = 1
	CmdReady         CmdID = 2
	CmdGetTime       CmdID = 3
	CmdCameraMove    CmdID = 4
	CmdCameraZoom    CmdID = 5
	CmdCameraAttach  CmdID = 6
	CmdSetBeacon     CmdID = 7
	CmdMovePlayer    CmdID = 8
	CmdDoAction      CmdID = 9
	CmdEnter         CmdID = 10
	CmdStopPlayer    CmdID = 11
	CmdUpgradeSkill  CmdID = 12
	CmdBuy           CmdID = 13
	CmdSell          CmdID = 14
	CmdEquipItem     CmdID = 15
	CmdSetState      CmdID = 16
	CmdForceRespawn  CmdID = 17
	CmdUseObject     CmdID = 18
	CmdGetDropInfo   CmdID = 19
	CmdPickUp        CmdID = 20
	CmdDropItem      CmdID = 21
	CmdPlayerReg     CmdID = 512
	CmdPlayerUnreg   CmdID = 513
	CmdGameData      CmdID = 514
	CmdSetAvatar     CmdID = 515
	CmdCreateObject  CmdID = 516
	CmdDeleteObject  CmdID = 517
	CmdSync          CmdID = 518
	CmdAction        CmdID = 519
	CmdActionDone    CmdID = 520
	CmdLevelUp       CmdID = 521
	CmdEffectStart   CmdID = 522
	CmdEffectEnd     CmdID = 523
	CmdOnKill        CmdID = 524
	CmdBattleEnd     CmdID = 525
	CmdSetMoney      CmdID = 526
	CmdAddToInv      CmdID = 527
	CmdRemFromInv    CmdID = 528
	CmdReceiveHit    CmdID = 531
	CmdSetProjectile CmdID = 533
	CmdRespawn       CmdID = 534
	CmdItemEquip     CmdID = 535
	CmdPrototypeInfo CmdID = 536
	CmdRefreshDrop   CmdID = 537
	CmdAddEffector   CmdID = 538
	CmdRemEffector   CmdID = 539
	CmdNotifyBeacon  CmdID = 540
	CmdPlayerStats   CmdID = 541
	CmdPlayerOnline  CmdID = 542
	CmdQuestTask     CmdID = 543
	CmdStartBattle   CmdID = 544
	CmdAddBuff       CmdID = 546
	CmdOrderDone     CmdID = 547
	CmdDebug         CmdID = 2048
)

var cmdNames = map[CmdID]string{
	CmdEcho: "ECHO", CmdConnect: "CONNECT", CmdReady: "READY", CmdGetTime: "GET_TIME",
	CmdCameraMove: "CAMERA_MOVE", CmdCameraZoom: "CAMERA_ZOOM", CmdCameraAttach: "CAMERA_ATTACH",
	CmdSetBeacon: "SET_BEACON", CmdMovePlayer: "MOVE_PLAYER", CmdDoAction: "DO_ACTION",
	CmdEnter: "ENTER", CmdStopPlayer: "STOP_PLAYER", CmdUpgradeSkill: "UPGRADE_SKILL",
	CmdBuy: "BUY", CmdSell: "SELL", CmdEquipItem: "EQUIP_ITEM", CmdSetState: "SET_STATE",
	CmdForceRespawn: "FORCE_RESPAWN", CmdUseObject: "USE_OBJECT", CmdGetDropInfo: "GET_DROP_INFO",
	CmdPickUp: "PICK_UP", CmdDropItem: "DROP_ITEM", CmdPlayerReg: "PLAYER_REG",
	CmdPlayerUnreg: "PLAYER_UNREG", CmdGameData: "GAME_DATA", CmdSetAvatar: "SET_AVATAR",
	CmdCreateObject: "CREATE_OBJECT", CmdDeleteObject: "DELETE_OBJECT", CmdSync: "SYNC",
	CmdAction: "ACTION", CmdActionDone: "ACTION_DONE", CmdLevelUp: "LEVEL_UP",
	CmdEffectStart: "EFFECT_START", CmdEffectEnd: "EFFECT_END", CmdOnKill: "ON_KILL",
	CmdBattleEnd: "BATTLE_END", CmdSetMoney: "SET_MONEY", CmdAddToInv: "ADD_TO_INVENTORY",
	CmdRemFromInv: "REM_FROM_INVENTORY", CmdReceiveHit: "RECEIVE_HIT", CmdSetProjectile: "SET_PROJECTILE",
	CmdRespawn: "RESPAWN", CmdItemEquip: "ITEM_EQUIP", CmdPrototypeInfo: "PROTOTYPE_INFO",
	CmdRefreshDrop: "REFRESH_DROP_CONTENT", CmdAddEffector: "ADD_EFFECTOR", CmdRemEffector: "REMOVE_EFFECTOR",
	CmdNotifyBeacon: "NOTIFY_BEACON", CmdPlayerStats: "PLAYER_STATS", CmdPlayerOnline: "PLAYER_ONLINE",
	CmdQuestTask: "QUEST_TASK", CmdStartBattle: "START_BATTLE", CmdAddBuff: "ADD_BUFF",
	CmdOrderDone: "ORDER_DONE", CmdDebug: "DEBUG",
}

// Name returns the BattleCmdId enum name for logging, or the numeric id.
func (c CmdID) Name() string {
	if n, ok := cmdNames[c]; ok {
		return n
	}
	return fmt.Sprintf("CMD_%d", int32(c))
}

// Packet is a decoded Battle packet (mirrors BattlePacket's fields).
type Packet struct {
	Cmd       CmdID
	Args      *amf.MixedArray
	RequestID int32 // -1 when the wire packet carried no requestId
	Status    bool
	Error     string
}

// Reader decodes a stream of Battle chunks into individual packets, preserving
// the connection-wide AMF string reference table across packets.
type Reader struct {
	br      *bufio.Reader
	dec     *amf.Decoder
	pending []Packet
}

func NewReader(r io.Reader) *Reader {
	return &Reader{
		br:  bufio.NewReader(r),
		dec: amf.NewDecoder(bytes.NewReader(nil)),
	}
}

// Read returns the next packet, reading and buffering whole chunks as needed.
// It returns io.EOF when the stream ends cleanly at a chunk boundary.
func (r *Reader) Read() (Packet, error) {
	for len(r.pending) == 0 {
		if err := r.readChunk(); err != nil {
			return Packet{}, err
		}
	}
	p := r.pending[0]
	r.pending = r.pending[1:]
	return p, nil
}

func (r *Reader) readChunk() error {
	outer, err := readSize(r.br)
	if err != nil {
		return err
	}
	if outer <= 0 {
		return fmt.Errorf("battleproto: invalid chunk size %d", outer)
	}
	buf := make([]byte, outer)
	if _, err := io.ReadFull(r.br, buf); err != nil {
		return err
	}
	off := 0
	for off < len(buf) {
		if len(buf)-off < 4 {
			return fmt.Errorf("battleproto: truncated packet-size prefix")
		}
		inner := int(int32(binary.BigEndian.Uint32(buf[off : off+4])))
		off += 4
		if inner <= 0 || inner > len(buf)-off {
			return fmt.Errorf("battleproto: invalid packet size %d (remaining %d)", inner, len(buf)-off)
		}
		body := buf[off : off+inner]
		off += inner
		r.dec.Reset(bytes.NewReader(body))
		v, err := r.dec.DecodeValue()
		if err != nil {
			return fmt.Errorf("battleproto: decode packet body: %w", err)
		}
		ma, ok := v.(*amf.MixedArray)
		if !ok {
			return fmt.Errorf("battleproto: packet body not an array: %T", v)
		}
		p, err := packetFromArray(ma)
		if err != nil {
			return err
		}
		r.pending = append(r.pending, p)
	}
	return nil
}

// packetFromArray mirrors the BattlePacket(Variable, IPacketNumManager) ctor.
func packetFromArray(m *amf.MixedArray) (Packet, error) {
	cmd, ok := m.GetInt("cmdId")
	if !ok {
		return Packet{}, fmt.Errorf("battleproto: missing cmdId")
	}
	p := Packet{Cmd: CmdID(cmd), Status: true, RequestID: -1}
	if st, ok := m.GetBool("status"); ok {
		p.Status = st
	}
	if !p.Status {
		p.Error = m.StringOr("error", "")
	}
	if rid, ok := m.GetInt("requestId"); ok {
		p.RequestID = rid
	}
	if args, ok := m.GetArray("arguments"); ok {
		p.Args = args
	} else {
		p.Args = amf.NewArray()
	}
	return p, nil
}

// Write serializes one packet as its own single-packet chunk. Each call uses a
// fresh no-ref encoder so the client (whose decoder shares one connection-wide
// ref table) never needs to resolve a back-reference we didn't emit.
func Write(w io.Writer, p Packet) error {
	body, err := encodeBody(p)
	if err != nil {
		return err
	}
	inner := len(body)
	outer := inner + 4
	var hdr [8]byte
	binary.BigEndian.PutUint32(hdr[0:4], uint32(int32(outer)))
	binary.BigEndian.PutUint32(hdr[4:8], uint32(int32(inner)))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	_, err = w.Write(body)
	return err
}

func encodeBody(p Packet) ([]byte, error) {
	args := p.Args
	if args == nil {
		args = amf.NewArray()
	}
	m := amf.NewArray().
		Set("arguments", args).
		Set("cmdId", int32(p.Cmd)).
		Set("requestId", p.RequestID).
		Set("status", p.Status)
	if !p.Status && p.Error != "" {
		m.Set("error", p.Error)
	}
	var buf bytes.Buffer
	if err := amf.NewRawEncoder().EncodeMessage(&buf, m); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func readSize(br *bufio.Reader) (int, error) {
	var b [4]byte
	if _, err := io.ReadFull(br, b[:]); err != nil {
		return 0, err
	}
	return int(int32(binary.BigEndian.Uint32(b[:]))), nil
}
