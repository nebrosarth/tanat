package mpd

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"io"

	"tanatserver/internal/amf"
)

// MPD framing (TanatKernel.MpdConnection): each message is a 4-byte BIG-ENDIAN
// length prefix counting only the AMF body, followed by that AMF body. This is a
// SINGLE length prefix -- distinct from the Battle channel's double-length framing.
// The client reads the size (ReadSize reverses to network order), then that many
// body bytes, then AMF-deserializes; a body may hold one value (login/auth) or a
// MixedArray of pushed commands.

// readFrame reads one length-prefixed AMF body from r.
func readFrame(r *bufio.Reader) ([]byte, error) {
	var lenBuf [4]byte
	if _, err := io.ReadFull(r, lenBuf[:]); err != nil {
		return nil, err
	}
	n := binary.BigEndian.Uint32(lenBuf[:])
	body := make([]byte, n)
	if _, err := io.ReadFull(r, body); err != nil {
		return nil, err
	}
	return body, nil
}

// frame wraps an AMF body with its 4-byte big-endian length prefix.
func frame(body []byte) []byte {
	out := make([]byte, 4+len(body))
	binary.BigEndian.PutUint32(out[:4], uint32(len(body)))
	copy(out[4:], body)
	return out
}

// decodeLogin reads the client's first frame: an AMF array {id:int, sid:string}
// (MpdConnection.SendLogin). id = chat_server_uid, sid = chat_server_sid, both
// values the server handed the client in the chat|conf packet.
func decodeLogin(body []byte) (id int32, sid string, ok bool) {
	v, err := amf.NewDecoder(bytes.NewReader(body)).DecodeValue()
	if err != nil {
		return 0, "", false
	}
	m, isArr := v.(*amf.MixedArray)
	if !isArr {
		return 0, "", false
	}
	id, _ = m.GetInt("id")
	sid, _ = m.GetString("sid")
	return id, sid, true
}

// encodePush frames a push root MixedArray. Uses the raw (no string-ref) encoder,
// matching the client's per-frame ClearRefTables behaviour (like the Battle channel).
func encodePush(root *amf.MixedArray) []byte {
	var buf bytes.Buffer
	_ = amf.NewRawEncoder().EncodeMessage(&buf, root)
	return frame(buf.Bytes())
}

// authAck is the framed AMF integer 100 the client waits for after login
// (MpdConnection.Parse: any int != 100 -> AUTHORIZATION_FAILED). Bytes: 00 00 00 02
// 04 64.
var authAck = func() []byte {
	var buf bytes.Buffer
	_ = amf.NewRawEncoder().EncodeMessage(&buf, int32(100))
	return frame(buf.Bytes())
}()
