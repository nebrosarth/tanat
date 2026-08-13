// Package assaultproto implements the versioned binary stdin/stdout boundary
// between Go rollout environments and Python AI-40 training workers.
package assaultproto

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"

	"tanatserver/internal/battleserver"
)

const (
	// Version 3 adds the explicit AI-40 controller value used by mirror
	// self-play resets.  The wire layout is unchanged, but older runners must
	// fail closed instead of interpreting the new controller as invalid.
	Version        uint16 = 3
	CommandReset   uint16 = 1
	CommandStep    uint16 = 2
	CommandClose   uint16 = 3
	ResponseResult uint16 = 100
	ResponseError  uint16 = 255
	maxFrameSize          = 4 << 20
)

var magic = [4]byte{'T', 'A', 'N', 'T'}

type Request struct {
	Command uint16
	Reset   battleserver.AssaultResetV1
	Actions [battleserver.AssaultHeroCount]battleserver.HeroActionV1
}

func ReadRequest(r io.Reader) (Request, error) {
	body, err := readFrame(r)
	if err != nil {
		return Request{}, err
	}
	if len(body) < 8 || !bytes.Equal(body[:4], magic[:]) {
		return Request{}, errors.New("assault protocol: bad magic")
	}
	version := binary.LittleEndian.Uint16(body[4:6])
	if version != Version {
		return Request{}, fmt.Errorf("assault protocol: version %d, want %d", version, Version)
	}
	command := binary.LittleEndian.Uint16(body[6:8])
	p := body[8:]
	request := Request{Command: command}
	switch command {
	case CommandReset:
		const size = 8 + 4 + battleserver.AssaultHeroCount*4 + battleserver.AssaultHeroCount
		if len(p) != size {
			return Request{}, fmt.Errorf("assault protocol: RESET payload=%d, want %d", len(p), size)
		}
		request.Reset.Seed = int64(binary.LittleEndian.Uint64(p[:8]))
		request.Reset.MaxSteps = binary.LittleEndian.Uint32(p[8:12])
		off := 12
		for i := range request.Reset.Roster {
			request.Reset.Roster[i] = int32(binary.LittleEndian.Uint32(p[off : off+4]))
			off += 4
		}
		for i := range request.Reset.Controllers {
			request.Reset.Controllers[i] = battleserver.AssaultControllerV1(p[off])
			off++
		}
	case CommandStep:
		const actionSize = 5
		if len(p) != battleserver.AssaultHeroCount*actionSize {
			return Request{}, fmt.Errorf("assault protocol: STEP payload=%d, want %d", len(p), battleserver.AssaultHeroCount*actionSize)
		}
		for i := range request.Actions {
			off := i * actionSize
			request.Actions[i] = battleserver.HeroActionV1{
				Kind:      battleserver.AssaultActionKindV1(p[off]),
				Target:    binary.LittleEndian.Uint16(p[off+1 : off+3]),
				Direction: p[off+3],
				Distance:  p[off+4],
			}
		}
	case CommandClose:
		if len(p) != 0 {
			return Request{}, errors.New("assault protocol: CLOSE must have empty payload")
		}
	default:
		return Request{}, fmt.Errorf("assault protocol: unknown command %d", command)
	}
	return request, nil
}

func WriteResult(w io.Writer, result battleserver.StepResultV1) error {
	return NewResultEncoder().Write(w, &result)
}

const resultBodySize = 138 + battleserver.AssaultHeroCount*(battleserver.AssaultHeroFeatureSize*4+
	battleserver.AssaultMaxEntities*battleserver.AssaultEntityFeatures*4+
	battleserver.AssaultGlobalFeatures*4+
	battleserver.AssaultMaxEntities+
	battleserver.AssaultActionKinds+
	battleserver.AssaultMaxEntities+
	4*battleserver.AssaultMaxEntities)

// ResultEncoder owns one reusable frame buffer. A rollout worker emits one
// fixed-size observation after every step, so rebuilding a bytes.Buffer and
// invoking reflective binary.Write once per float only creates GC pressure.
type ResultEncoder struct {
	body []byte
}

func NewResultEncoder() *ResultEncoder {
	return &ResultEncoder{body: make([]byte, resultBodySize)}
}

func (e *ResultEncoder) Write(w io.Writer, result *battleserver.StepResultV1) error {
	if result == nil {
		return errors.New("assault protocol: nil result")
	}
	if cap(e.body) < resultBodySize {
		e.body = make([]byte, resultBodySize)
	}
	body := e.body[:resultBodySize]
	off := 0
	putBytes := func(values []byte) {
		copy(body[off:], values)
		off += len(values)
	}
	putU16 := func(value uint16) {
		binary.LittleEndian.PutUint16(body[off:off+2], value)
		off += 2
	}
	putU32 := func(value uint32) {
		binary.LittleEndian.PutUint32(body[off:off+4], value)
		off += 4
	}
	putF32 := func(value float32) { putU32(math.Float32bits(value)) }
	putF32s := func(values []float32) {
		for _, value := range values {
			putF32(value)
		}
	}
	putBytes(magic[:])
	putU16(Version)
	putU16(ResponseResult)
	putBytes(result.SchemaHash[:])
	putBytes(result.RewardHash[:])
	putU32(result.Step)
	putF32(result.Elapsed)
	if result.Done {
		body[off] = 1
	} else {
		body[off] = 0
	}
	off += 4 // done plus three padding bytes; reused padding stays zero.
	putU32(uint32(result.Winner))
	putBytes(result.Invalid[:])
	for _, reward := range result.Reward {
		putF32(reward)
	}
	for i := range result.Observations {
		obs := &result.Observations[i]
		putF32s(obs.Hero[:])
		for j := range obs.Entities {
			putF32s(obs.Entities[j][:])
		}
		putF32s(obs.Global[:])
		putBytes(obs.EntityMask[:])
		putBytes(obs.ActionMask.Kinds[:])
		putBytes(obs.ActionMask.Targets[:])
		for j := range obs.ActionMask.SkillTarget {
			putBytes(obs.ActionMask.SkillTarget[j][:])
		}
	}
	if off != resultBodySize {
		return fmt.Errorf("assault protocol: encoded result size %d, want %d", off, resultBodySize)
	}
	return writeFrame(w, body)
}

func WriteError(w io.Writer, message string) error {
	b := new(bytes.Buffer)
	b.Write(magic[:])
	writeU16(b, Version)
	writeU16(b, ResponseError)
	b.WriteString(message)
	return writeFrame(w, b.Bytes())
}

func readFrame(r io.Reader) ([]byte, error) {
	var length uint32
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return nil, err
	}
	if length < 8 || length > maxFrameSize {
		return nil, fmt.Errorf("assault protocol: invalid frame length %d", length)
	}
	body := make([]byte, length)
	if _, err := io.ReadFull(r, body); err != nil {
		return nil, err
	}
	return body, nil
}

func writeFrame(w io.Writer, body []byte) error {
	bw, ok := w.(*bufio.Writer)
	if !ok {
		bw = bufio.NewWriter(w)
	}
	var prefix [4]byte
	binary.LittleEndian.PutUint32(prefix[:], uint32(len(body)))
	if _, err := bw.Write(prefix[:]); err != nil {
		return err
	}
	if _, err := bw.Write(body); err != nil {
		return err
	}
	return bw.Flush()
}

func writeU16(w io.Writer, v uint16)  { _ = binary.Write(w, binary.LittleEndian, v) }
func writeU32(w io.Writer, v uint32)  { _ = binary.Write(w, binary.LittleEndian, v) }
func writeI32(w io.Writer, v int32)   { _ = binary.Write(w, binary.LittleEndian, v) }
func writeF32(w io.Writer, v float32) { writeU32(w, math.Float32bits(v)) }
func writeF32s(w io.Writer, values []float32) {
	for _, v := range values {
		writeF32(w, v)
	}
}
