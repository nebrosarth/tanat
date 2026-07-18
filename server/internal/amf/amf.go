// Package amf implements the exact wire subset of AMF3 used by Tanat Online's
// client (see decompiled AMF.dll: Formatter, U29Converter, StrConverter,
// DoubleConverter, ByteArrayConverter, ArrayConverter). It intentionally does
// NOT implement full AMF3 (no Object/traits, no Date, no array-by-reference):
// the original client never sends or expects those markers.
package amf

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

type Marker byte

const (
	MarkerUndefined Marker = 0
	MarkerNull      Marker = 1
	MarkerFalse     Marker = 2
	MarkerTrue      Marker = 3
	MarkerInteger   Marker = 4
	MarkerDouble    Marker = 5
	MarkerString    Marker = 6
	MarkerXmlDoc    Marker = 7
	MarkerDate      Marker = 8
	MarkerArray     Marker = 9
	MarkerObject    Marker = 10
	MarkerXml       Marker = 11
	MarkerByteArray Marker = 12
)

// MixedArray mirrors AMF.MixedArray: an AMF3 array has an associative
// (string-keyed) part and a dense (index-keyed) part, encoded together.
type MixedArray struct {
	Assoc map[string]interface{}
	Dense []interface{}
}

func NewArray() *MixedArray {
	return &MixedArray{Assoc: map[string]interface{}{}}
}

func (m *MixedArray) Set(key string, val interface{}) *MixedArray {
	m.Assoc[key] = val
	return m
}

func (m *MixedArray) Add(val interface{}) *MixedArray {
	m.Dense = append(m.Dense, val)
	return m
}

func (m *MixedArray) GetString(key string) (string, bool) {
	v, ok := m.Assoc[key]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok
}

func (m *MixedArray) GetInt(key string) (int32, bool) {
	v, ok := m.Assoc[key]
	if !ok {
		return 0, false
	}
	switch n := v.(type) {
	case int32:
		return n, true
	case float64:
		return int32(n), true
	}
	return 0, false
}

func (m *MixedArray) GetFloat(key string) (float64, bool) {
	v, ok := m.Assoc[key]
	if !ok {
		return 0, false
	}
	switch n := v.(type) {
	case float64:
		return n, true
	case int32:
		return float64(n), true
	}
	return 0, false
}

func (m *MixedArray) GetBool(key string) (bool, bool) {
	v, ok := m.Assoc[key]
	if !ok {
		return false, false
	}
	b, ok := v.(bool)
	return b, ok
}

func (m *MixedArray) GetArray(key string) (*MixedArray, bool) {
	v, ok := m.Assoc[key]
	if !ok {
		return nil, false
	}
	a, ok := v.(*MixedArray)
	return a, ok
}

func (m *MixedArray) StringOr(key, def string) string {
	if v, ok := m.GetString(key); ok {
		return v
	}
	return def
}

func (m *MixedArray) IntOr(key string, def int32) int32 {
	if v, ok := m.GetInt(key); ok {
		return v
	}
	return def
}

func (m *MixedArray) BoolOr(key string, def bool) bool {
	if v, ok := m.GetBool(key); ok {
		return v
	}
	return def
}

// ---- Encoder ----

type Encoder struct {
	refs  map[string]int
	next  int
	noRef bool
}

func NewEncoder() *Encoder {
	return &Encoder{refs: map[string]int{}}
}

// NewRawEncoder returns an encoder that always writes strings inline, never as
// back-references. Use it for the Battle channel: unlike the Ctrl channel
// (which clears its Formatter's string-ref table before/after every message),
// the Battle Formatter keeps ONE ref table alive for the whole connection
// (BattlePacketManager only clears it on Clear()/disconnect). The client's
// decoder still adds each inline string to its table, but since it only ever
// resolves a reference we actually emit, never emitting one is correct
// regardless of the client's running indices. See internal/battleproto.
func NewRawEncoder() *Encoder {
	return &Encoder{refs: map[string]int{}, noRef: true}
}

// EncodeMessage writes a single top-level AMF value (always a *MixedArray in
// this protocol). The string reference table is local to one message, matching
// Formatter.ClearRefTables() being called before/after every Serialize in the
// original client.
func (e *Encoder) EncodeMessage(w io.Writer, v interface{}) error {
	e.refs = map[string]int{}
	e.next = 0
	bw := bufio.NewWriter(w)
	if err := e.encodeValue(bw, v); err != nil {
		return err
	}
	return bw.Flush()
}

// EncodeValue writes a single AMF value WITHOUT resetting the string reference
// table (unlike EncodeMessage). It mirrors Decoder.DecodeValue: used to emit a
// framed stream of values that share one connection-wide ref table, e.g. to
// reproduce the client's Battle-channel encoder in tests.
func (e *Encoder) EncodeValue(w io.Writer, v interface{}) error {
	bw := bufio.NewWriter(w)
	if err := e.encodeValue(bw, v); err != nil {
		return err
	}
	return bw.Flush()
}

func (e *Encoder) encodeValue(w *bufio.Writer, v interface{}) error {
	switch val := v.(type) {
	case nil:
		return w.WriteByte(byte(MarkerNull))
	case bool:
		if val {
			return w.WriteByte(byte(MarkerTrue))
		}
		return w.WriteByte(byte(MarkerFalse))
	case int:
		return e.encodeInt(w, int32(val))
	case int32:
		return e.encodeInt(w, val)
	case uint32:
		return e.encodeInt(w, int32(val))
	case float64:
		return e.encodeDouble(w, val)
	case float32:
		return e.encodeDouble(w, float64(val))
	case string:
		if err := w.WriteByte(byte(MarkerString)); err != nil {
			return err
		}
		return e.encodeString(w, val)
	case []byte:
		if err := w.WriteByte(byte(MarkerByteArray)); err != nil {
			return err
		}
		if err := encodeU29(w, int32(len(val))<<1|1); err != nil {
			return err
		}
		_, err := w.Write(val)
		return err
	case *MixedArray:
		// A typed nil (*MixedArray)(nil) stored in an interface does NOT match `case nil`
		// above -- it lands here with val==nil. Encoding it as a null marker (instead of
		// dereferencing val.Dense in encodeArray, which panics) matches an untyped nil and
		// keeps a stray nil array from taking down the whole battle server.
		if val == nil {
			return w.WriteByte(byte(MarkerNull))
		}
		return e.encodeArray(w, val)
	default:
		return fmt.Errorf("amf: unsupported value type %T", v)
	}
}

func (e *Encoder) encodeInt(w *bufio.Writer, val int32) error {
	if err := w.WriteByte(byte(MarkerInteger)); err != nil {
		return err
	}
	return encodeU29(w, val)
}

func (e *Encoder) encodeDouble(w *bufio.Writer, val float64) error {
	if err := w.WriteByte(byte(MarkerDouble)); err != nil {
		return err
	}
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], math.Float64bits(val))
	_, err := w.Write(buf[:])
	return err
}

func (e *Encoder) encodeString(w *bufio.Writer, s string) error {
	if !e.noRef {
		if id, ok := e.refs[s]; ok {
			return encodeU29(w, int32(id)<<1)
		}
	}
	b := []byte(s)
	if err := encodeU29(w, int32(len(b))<<1|1); err != nil {
		return err
	}
	if _, err := w.Write(b); err != nil {
		return err
	}
	if !e.noRef && s != "" {
		e.refs[s] = e.next
		e.next++
	}
	return nil
}

func (e *Encoder) encodeArray(w *bufio.Writer, m *MixedArray) error {
	if err := w.WriteByte(byte(MarkerArray)); err != nil {
		return err
	}
	if err := encodeU29(w, int32(len(m.Dense))<<1|1); err != nil {
		return err
	}
	for k, v := range m.Assoc {
		if err := e.encodeString(w, k); err != nil {
			return err
		}
		if err := e.encodeValue(w, v); err != nil {
			return err
		}
	}
	if err := e.encodeString(w, ""); err != nil {
		return err
	}
	for _, v := range m.Dense {
		if err := e.encodeValue(w, v); err != nil {
			return err
		}
	}
	return nil
}

// encodeU29 replicates U29Converter.EncodeValue byte-for-byte, including its
// 29-bit truncation via `val &= 0x1FFFFFFF`.
func encodeU29(w *bufio.Writer, val int32) error {
	v := val & 0x1FFFFFFF
	switch {
	case v <= 127:
		return w.WriteByte(byte(v & 0x7F))
	case v <= 16383:
		b0 := byte(((v >> 7) & 0x7F) | 0x80)
		b1 := byte(v & 0x7F)
		return writeBytes(w, b0, b1)
	case v <= 2097151:
		b0 := byte(((v >> 14) & 0x7F) | 0x80)
		b1 := byte(((v >> 7) & 0x7F) | 0x80)
		b2 := byte(v & 0x7F)
		return writeBytes(w, b0, b1, b2)
	default:
		b0 := byte(((v >> 22) & 0x7F) | 0x80)
		b1 := byte(((v >> 15) & 0x7F) | 0x80)
		b2 := byte(((v >> 8) & 0x7F) | 0x80)
		b3 := byte(v & 0xFF)
		return writeBytes(w, b0, b1, b2, b3)
	}
}

func writeBytes(w *bufio.Writer, bs ...byte) error {
	_, err := w.Write(bs)
	return err
}

// ---- Decoder ----

type Decoder struct {
	r    *bufio.Reader
	refs map[int]string
	next int
}

func NewDecoder(r io.Reader) *Decoder {
	return &Decoder{r: bufio.NewReader(r), refs: map[int]string{}}
}

// DecodeMessage reads a single top-level AMF value with a fresh string
// reference table.
func (d *Decoder) DecodeMessage() (interface{}, error) {
	d.refs = map[int]string{}
	d.next = 0
	return d.decodeValue()
}

// Reset rebinds the decoder to a new reader WITHOUT clearing the string
// reference table, so cross-message string references keep resolving. The
// Battle channel frames each packet separately but shares one connection-wide
// ref table (see NewRawEncoder), so successive packets are decoded with
// Reset + DecodeValue rather than DecodeMessage.
func (d *Decoder) Reset(r io.Reader) {
	d.r = bufio.NewReader(r)
}

// DecodeValue decodes a single AMF value WITHOUT resetting the string
// reference table (unlike DecodeMessage). Pair with Reset for framed streams.
func (d *Decoder) DecodeValue() (interface{}, error) {
	return d.decodeValue()
}

// More reports whether there is another byte available, i.e. whether another
// top-level message follows in the stream (the client's CtrlParser loops
// while bytes remain).
func (d *Decoder) More() bool {
	_, err := d.r.Peek(1)
	return err == nil
}

func (d *Decoder) decodeValue() (interface{}, error) {
	b, err := d.r.ReadByte()
	if err != nil {
		return nil, err
	}
	switch Marker(b) {
	case MarkerUndefined, MarkerNull:
		return nil, nil
	case MarkerFalse:
		return false, nil
	case MarkerTrue:
		return true, nil
	case MarkerInteger:
		return decodeU29(d.r)
	case MarkerDouble:
		var buf [8]byte
		if _, err := io.ReadFull(d.r, buf[:]); err != nil {
			return nil, err
		}
		return math.Float64frombits(binary.BigEndian.Uint64(buf[:])), nil
	case MarkerString, MarkerXml, MarkerXmlDoc:
		return d.decodeString()
	case MarkerByteArray:
		n, err := decodeU29(d.r)
		if err != nil {
			return nil, err
		}
		if n&1 == 0 {
			return nil, fmt.Errorf("amf: byte array by reference not supported")
		}
		length := int(n >> 1)
		buf := make([]byte, length)
		if _, err := io.ReadFull(d.r, buf); err != nil {
			return nil, err
		}
		return buf, nil
	case MarkerArray:
		return d.decodeArray()
	default:
		return nil, fmt.Errorf("amf: unsupported marker %d", b)
	}
}

func (d *Decoder) decodeString() (string, error) {
	n, err := decodeU29(d.r)
	if err != nil {
		return "", err
	}
	if n&1 != 0 {
		length := int(n >> 1)
		if length == 0 {
			return "", nil
		}
		if length < 0 {
			return "", fmt.Errorf("amf: negative string length")
		}
		buf := make([]byte, length)
		if _, err := io.ReadFull(d.r, buf); err != nil {
			return "", err
		}
		s := string(buf)
		d.refs[d.next] = s
		d.next++
		return s, nil
	}
	id := int(n >> 1)
	return d.refs[id], nil
}

func (d *Decoder) decodeArray() (*MixedArray, error) {
	n, err := decodeU29(d.r)
	if err != nil {
		return nil, err
	}
	if n&1 == 0 {
		return nil, fmt.Errorf("amf: array by reference not supported")
	}
	denseCount := int(n >> 1)
	m := NewArray()
	for {
		key, err := d.decodeString()
		if err != nil {
			return nil, err
		}
		if key == "" {
			break
		}
		val, err := d.decodeValue()
		if err != nil {
			return nil, err
		}
		if val == nil {
			break
		}
		m.Assoc[key] = val
	}
	for i := 0; i < denseCount; i++ {
		val, err := d.decodeValue()
		if err != nil {
			return nil, err
		}
		m.Dense = append(m.Dense, val)
	}
	return m, nil
}

// decodeU29 replicates U29Converter.DecodeValue byte-for-byte, including the
// 4th-byte-is-8-bits quirk and bit-28 sign extension of the AMF3 U29 format.
func decodeU29(r *bufio.Reader) (int32, error) {
	var num int32
	count := 0
	for {
		b, err := r.ReadByte()
		if err != nil {
			return 0, err
		}
		if count == 3 {
			num = (num << 8) | int32(b)
		} else {
			num = (num << 7) | int32(b&0x7F)
		}
		if b&0x80 == 0 {
			break
		}
		count++
		if count >= 4 {
			break
		}
	}
	if num&0x10000000 != 0 {
		num |= -268435456 // sign-extend bit 28 across the top 3 bits (0xF0000000)
	}
	return num, nil
}
