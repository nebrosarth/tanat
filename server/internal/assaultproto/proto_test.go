package assaultproto

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"io"
	"testing"

	"tanatserver/internal/battleserver"
)

func requestFrame(command uint16, payload []byte) []byte {
	body := new(bytes.Buffer)
	body.Write(magic[:])
	_ = binary.Write(body, binary.LittleEndian, Version)
	_ = binary.Write(body, binary.LittleEndian, command)
	body.Write(payload)
	frame := new(bytes.Buffer)
	_ = binary.Write(frame, binary.LittleEndian, uint32(body.Len()))
	frame.Write(body.Bytes())
	return frame.Bytes()
}

func BenchmarkWriteResult(b *testing.B) {
	var result battleserver.StepResultV1
	encoder := NewResultEncoder()
	out := bufio.NewWriterSize(io.Discard, resultBodySize+4)
	b.ReportAllocs()
	b.SetBytes(resultBodySize)
	for i := 0; i < b.N; i++ {
		if err := encoder.Write(out, &result); err != nil {
			b.Fatal(err)
		}
	}
}

func TestReadStepRequest(t *testing.T) {
	payload := make([]byte, battleserver.AssaultHeroCount*5)
	payload[0] = byte(battleserver.AssaultActionAttack)
	binary.LittleEndian.PutUint16(payload[1:3], 42)
	payload[3], payload[4] = 9, 2
	req, err := ReadRequest(bytes.NewReader(requestFrame(CommandStep, payload)))
	if err != nil {
		t.Fatal(err)
	}
	if got := req.Actions[0]; got.Kind != battleserver.AssaultActionAttack || got.Target != 42 || got.Direction != 9 || got.Distance != 2 {
		t.Fatalf("action=%+v", got)
	}
}

func TestReadAI40MirrorReset(t *testing.T) {
	const payloadSize = 8 + 4 + battleserver.AssaultHeroCount*4 + battleserver.AssaultHeroCount
	payload := make([]byte, payloadSize)
	binary.LittleEndian.PutUint64(payload[:8], 77)
	binary.LittleEndian.PutUint32(payload[8:12], 500)
	controllerOffset := 12 + battleserver.AssaultHeroCount*4
	for i := 0; i < battleserver.AssaultHeroCount; i++ {
		payload[controllerOffset+i] = byte(battleserver.AssaultControllerAI40)
	}
	req, err := ReadRequest(bytes.NewReader(requestFrame(CommandReset, payload)))
	if err != nil {
		t.Fatal(err)
	}
	for i, controller := range req.Reset.Controllers {
		if controller != battleserver.AssaultControllerAI40 {
			t.Fatalf("controller[%d]=%d, want AI-40", i, controller)
		}
	}
}

func TestWriteResultIsFramedAndVersioned(t *testing.T) {
	var result battleserver.StepResultV1
	result.SchemaHash = battleserver.AssaultSchemaHashV1
	result.RewardHash = battleserver.AssaultRewardHashV2
	result.Step = 3
	var out bytes.Buffer
	if err := WriteResult(&out, result); err != nil {
		t.Fatal(err)
	}
	body, err := readFrame(&out)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(body[:4], magic[:]) || binary.LittleEndian.Uint16(body[4:6]) != Version ||
		binary.LittleEndian.Uint16(body[6:8]) != ResponseResult ||
		!bytes.Equal(body[40:72], battleserver.AssaultRewardHashV2[:]) || binary.LittleEndian.Uint32(body[72:76]) != 3 {
		t.Fatalf("bad result header: %x", body[:80])
	}
}
