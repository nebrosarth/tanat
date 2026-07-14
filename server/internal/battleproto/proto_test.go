package battleproto

import (
	"bytes"
	"encoding/binary"
	"io"
	"testing"

	"tanatserver/internal/amf"
)

// TestWriteReadRoundTrip checks that packets written as single-packet chunks
// decode back with identical fields, in order.
func TestWriteReadRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	want := []Packet{
		{Cmd: CmdConnect, RequestID: 7, Status: true, Args: amf.NewArray().Set("clientId", int32(42)).Set("battleId", int32(1))},
		{Cmd: CmdGetTime, RequestID: 8, Status: true, Args: amf.NewArray().Set("time", 12.5)},
		{Cmd: CmdEnter, RequestID: 9, Status: true, Args: amf.NewArray()},
	}
	for _, p := range want {
		if err := Write(&buf, p); err != nil {
			t.Fatalf("Write(%s): %v", p.Cmd.Name(), err)
		}
	}

	r := NewReader(&buf)
	for i, w := range want {
		got, err := r.Read()
		if err != nil {
			t.Fatalf("Read #%d: %v", i, err)
		}
		if got.Cmd != w.Cmd || got.RequestID != w.RequestID || got.Status != w.Status {
			t.Errorf("#%d header = {%s req=%d status=%v}, want {%s req=%d status=%v}",
				i, got.Cmd.Name(), got.RequestID, got.Status, w.Cmd.Name(), w.RequestID, w.Status)
		}
	}
	if _, err := r.Read(); err != io.EOF {
		t.Errorf("after last packet: err = %v, want EOF", err)
	}
}

// TestConnectFieldsRoundTrip verifies the CONNECT reply args survive decode
// exactly, since they are what the client parses into ConnectArg.
func TestConnectFieldsRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	if err := Write(&buf, Packet{
		Cmd: CmdConnect, RequestID: 3, Status: true,
		Args: amf.NewArray().Set("clientId", int32(1001)).Set("battleId", int32(55)),
	}); err != nil {
		t.Fatal(err)
	}
	got, err := NewReader(&buf).Read()
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := got.Args.GetInt("clientId"); v != 1001 {
		t.Errorf("clientId = %d, want 1001", v)
	}
	if v, _ := got.Args.GetInt("battleId"); v != 55 {
		t.Errorf("battleId = %d, want 55", v)
	}
}

// TestCrossPacketStringRefs reproduces the client's Battle encoder, which keeps
// one string-ref table alive across every packet in the connection: packet 2's
// repeated keys ("arguments","cmdId","requestId") are emitted as back-references
// to packet 1. The Reader must decode both with a persistent ref table.
func TestCrossPacketStringRefs(t *testing.T) {
	enc := amf.NewEncoder() // ref-enabled, NOT reset between EncodeValue calls

	body1 := clientEncodePacket(t, enc, CmdConnect, 1, amf.NewArray().Set("clientId", int32(42)))
	body2 := clientEncodePacket(t, enc, CmdGetTime, 2, amf.NewArray())

	// Sanity: packet 2 must actually be shorter than a fresh encoding, i.e. it
	// used cross-packet references (otherwise this test proves nothing).
	fresh := clientEncodePacket(t, amf.NewEncoder(), CmdGetTime, 2, amf.NewArray())
	if len(body2) >= len(fresh) {
		t.Fatalf("packet 2 did not use back-references (len %d >= fresh %d)", len(body2), len(fresh))
	}

	chunk := assembleChunk(body1, body2)

	r := NewReader(bytes.NewReader(chunk))
	p1, err := r.Read()
	if err != nil {
		t.Fatalf("read p1: %v", err)
	}
	if p1.Cmd != CmdConnect || p1.RequestID != 1 {
		t.Errorf("p1 = {%s req=%d}, want {CONNECT req=1}", p1.Cmd.Name(), p1.RequestID)
	}
	if v, _ := p1.Args.GetInt("clientId"); v != 42 {
		t.Errorf("p1 clientId = %d, want 42", v)
	}
	p2, err := r.Read()
	if err != nil {
		t.Fatalf("read p2 (cross-packet refs): %v", err)
	}
	if p2.Cmd != CmdGetTime || p2.RequestID != 2 {
		t.Errorf("p2 = {%s req=%d}, want {GET_TIME req=2}", p2.Cmd.Name(), p2.RequestID)
	}
}

// clientEncodePacket mirrors BattlePacket.Serialize under a shared (persistent)
// encoder, producing one packet body's bytes as the real client would.
func clientEncodePacket(t *testing.T, enc *amf.Encoder, cmd CmdID, reqID int32, args *amf.MixedArray) []byte {
	t.Helper()
	m := amf.NewArray().
		Set("arguments", args).
		Set("cmdId", int32(cmd)).
		Set("requestId", reqID).
		Set("status", true)
	var b bytes.Buffer
	if err := enc.EncodeValue(&b, m); err != nil {
		t.Fatalf("encode packet: %v", err)
	}
	return b.Bytes()
}

func assembleChunk(bodies ...[]byte) []byte {
	var inner bytes.Buffer
	for _, body := range bodies {
		var sz [4]byte
		binary.BigEndian.PutUint32(sz[:], uint32(len(body)))
		inner.Write(sz[:])
		inner.Write(body)
	}
	var out bytes.Buffer
	var sz [4]byte
	binary.BigEndian.PutUint32(sz[:], uint32(inner.Len()))
	out.Write(sz[:])
	out.Write(inner.Bytes())
	return out.Bytes()
}
