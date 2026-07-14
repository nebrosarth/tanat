package amf

import (
	"bytes"
	"testing"
)

func TestIntRoundTrip(t *testing.T) {
	cases := []int32{0, 1, 127, 128, 16383, 16384, 2097151, 2097152, 268435455, -1, -268435456, 1234567}
	for _, c := range cases {
		var buf bytes.Buffer
		enc := NewEncoder()
		if err := enc.EncodeMessage(&buf, c); err != nil {
			t.Fatalf("encode %d: %v", c, err)
		}
		dec := NewDecoder(&buf)
		got, err := dec.DecodeMessage()
		if err != nil {
			t.Fatalf("decode %d: %v", c, err)
		}
		gi, ok := got.(int32)
		if !ok {
			t.Fatalf("decoded type %T for %d", got, c)
		}
		want := c & 0x1FFFFFFF
		if want&0x10000000 != 0 {
			want |= -268435456
		}
		if gi != want {
			t.Fatalf("int roundtrip: put %d got %d want %d", c, gi, want)
		}
	}
}

func TestKnownIntBytes(t *testing.T) {
	var buf bytes.Buffer
	enc := NewEncoder()
	_ = enc.EncodeMessage(&buf, int32(0))
	if got := buf.Bytes(); !bytes.Equal(got, []byte{byte(MarkerInteger), 0x00}) {
		t.Fatalf("int(0) = % x", got)
	}

	buf.Reset()
	_ = enc.EncodeMessage(&buf, int32(300))
	// 300 = 0b1_00101100 -> two-byte U29: [ (0x82) , 0x2C ]
	if got := buf.Bytes(); !bytes.Equal(got, []byte{byte(MarkerInteger), 0x82, 0x2C}) {
		t.Fatalf("int(300) = % x", got)
	}
}

func TestStringRoundTripAndRef(t *testing.T) {
	arr := NewArray()
	arr.Set("a", "hello")
	arr.Set("b", "hello") // should become a back-reference on encode
	arr.Set("n", int32(42))
	arr.Set("f", 3.5)
	arr.Set("flag", true)
	arr.Add("first")
	arr.Add("second")

	var buf bytes.Buffer
	enc := NewEncoder()
	if err := enc.EncodeMessage(&buf, arr); err != nil {
		t.Fatalf("encode: %v", err)
	}
	dec := NewDecoder(&buf)
	got, err := dec.DecodeMessage()
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	m, ok := got.(*MixedArray)
	if !ok {
		t.Fatalf("decoded type %T", got)
	}
	if s, _ := m.GetString("a"); s != "hello" {
		t.Fatalf("a = %q", s)
	}
	if s, _ := m.GetString("b"); s != "hello" {
		t.Fatalf("b (ref) = %q", s)
	}
	if n, _ := m.GetInt("n"); n != 42 {
		t.Fatalf("n = %d", n)
	}
	if f, _ := m.GetFloat("f"); f != 3.5 {
		t.Fatalf("f = %v", f)
	}
	if b, _ := m.GetBool("flag"); !b {
		t.Fatalf("flag = %v", b)
	}
	if len(m.Dense) != 2 || m.Dense[0] != "first" || m.Dense[1] != "second" {
		t.Fatalf("dense = %v", m.Dense)
	}
}

func TestNestedArray(t *testing.T) {
	inner := NewArray().Set("x", 1.5).Set("y", 2.5)
	outer := NewArray().Set("targetPos", inner).Set("rel", true)

	var buf bytes.Buffer
	enc := NewEncoder()
	if err := enc.EncodeMessage(&buf, outer); err != nil {
		t.Fatalf("encode: %v", err)
	}
	dec := NewDecoder(&buf)
	got, err := dec.DecodeMessage()
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	m := got.(*MixedArray)
	sub, ok := m.GetArray("targetPos")
	if !ok {
		t.Fatalf("no targetPos")
	}
	if x, _ := sub.GetFloat("x"); x != 1.5 {
		t.Fatalf("x = %v", x)
	}
}

func TestByteArrayRoundTrip(t *testing.T) {
	data := []byte{1, 2, 3, 4, 250}
	var buf bytes.Buffer
	enc := NewEncoder()
	if err := enc.EncodeMessage(&buf, data); err != nil {
		t.Fatalf("encode: %v", err)
	}
	dec := NewDecoder(&buf)
	got, err := dec.DecodeMessage()
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	gb, ok := got.([]byte)
	if !ok || !bytes.Equal(gb, data) {
		t.Fatalf("got %v want %v", got, data)
	}
}

func TestMultipleMessagesInStream(t *testing.T) {
	var buf bytes.Buffer
	enc := NewEncoder()
	_ = enc.EncodeMessage(&buf, int32(1))
	_ = enc.EncodeMessage(&buf, "two")
	_ = enc.EncodeMessage(&buf, NewArray().Set("k", int32(3)))

	dec := NewDecoder(&buf)
	v1, _ := dec.DecodeMessage()
	if v1.(int32) != 1 {
		t.Fatalf("v1 = %v", v1)
	}
	if !dec.More() {
		t.Fatalf("expected more")
	}
	v2, _ := dec.DecodeMessage()
	if v2.(string) != "two" {
		t.Fatalf("v2 = %v", v2)
	}
	v3, _ := dec.DecodeMessage()
	m := v3.(*MixedArray)
	if n, _ := m.GetInt("k"); n != 3 {
		t.Fatalf("v3.k = %v", n)
	}
	if dec.More() {
		t.Fatalf("expected no more")
	}
}
