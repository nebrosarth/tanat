package ai42preflight

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"reflect"
	"sort"
	"strconv"
	"strings"
)

func sha256Hex(payload []byte) string {
	sum := sha256.Sum256(payload)
	return fmt.Sprintf("%x", sum[:])
}

func validHash(value string) bool {
	if len(value) != sha256.Size*2 || strings.ToLower(value) != value {
		return false
	}
	_, err := strconv.ParseUint(value[:16], 16, 64)
	if err != nil {
		return false
	}
	for _, r := range value[16:] {
		if !((r >= '0' && r <= '9') || (r >= 'a' && r <= 'f')) {
			return false
		}
	}
	return true
}

// canonicalJSON implements the compact sorted-key UTF-8 JSON convention used
// by trajectory_ai42.py. In particular, integral floats retain Python's .0
// spelling and HTML characters are not escaped.
func canonicalJSON(value any) ([]byte, error) {
	var b bytes.Buffer
	if err := writeCanonical(&b, reflect.ValueOf(value)); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

func writeCanonical(b *bytes.Buffer, value reflect.Value) error {
	if !value.IsValid() {
		b.WriteString("null")
		return nil
	}
	if value.Kind() == reflect.Interface {
		if value.IsNil() {
			b.WriteString("null")
			return nil
		}
		return writeCanonical(b, value.Elem())
	}
	if value.Type() == reflect.TypeOf(json.Number("")) {
		text := value.String()
		if _, err := strconv.ParseFloat(text, 64); err != nil {
			return fmt.Errorf("invalid JSON number %q", text)
		}
		b.WriteString(text)
		return nil
	}
	switch value.Kind() {
	case reflect.Pointer:
		if value.IsNil() {
			b.WriteString("null")
			return nil
		}
		return writeCanonical(b, value.Elem())
	case reflect.Bool:
		if value.Bool() {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case reflect.String:
		// strconv.Quote uses UTF-8 and does not apply encoding/json's HTML
		// escaping, which matches ensure_ascii=False in Python.
		b.WriteString(strconv.Quote(value.String()))
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		b.WriteString(strconv.FormatInt(value.Int(), 10))
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		b.WriteString(strconv.FormatUint(value.Uint(), 10))
	case reflect.Float32, reflect.Float64:
		number := value.Float()
		if math.IsNaN(number) || math.IsInf(number, 0) {
			return fmt.Errorf("non-finite JSON number")
		}
		text := strconv.FormatFloat(number, 'g', -1, value.Type().Bits())
		if !strings.ContainsAny(text, ".eE") {
			text += ".0"
		}
		b.WriteString(text)
	case reflect.Slice, reflect.Array:
		b.WriteByte('[')
		for index := 0; index < value.Len(); index++ {
			if index != 0 {
				b.WriteByte(',')
			}
			if err := writeCanonical(b, value.Index(index)); err != nil {
				return err
			}
		}
		b.WriteByte(']')
	case reflect.Map:
		if value.IsNil() {
			b.WriteString("null")
			return nil
		}
		if value.Type().Key().Kind() != reflect.String {
			return fmt.Errorf("JSON object keys must be strings")
		}
		keys := value.MapKeys()
		sort.Slice(keys, func(i, j int) bool { return keys[i].String() < keys[j].String() })
		b.WriteByte('{')
		for index, key := range keys {
			if index != 0 {
				b.WriteByte(',')
			}
			b.WriteString(strconv.Quote(key.String()))
			b.WriteByte(':')
			if err := writeCanonical(b, value.MapIndex(key)); err != nil {
				return err
			}
		}
		b.WriteByte('}')
	default:
		return fmt.Errorf("unsupported JSON value %s", value.Type())
	}
	return nil
}

func decodeCanonical(raw []byte, name string, limit int) (any, error) {
	if len(raw) > limit {
		return nil, fmt.Errorf("%s exceeds %d-byte limit", name, limit)
	}
	var value any
	if err := decodeWithDuplicateCheck(raw, &value); err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	canonical, err := canonicalJSON(value)
	if err != nil {
		return nil, fmt.Errorf("%s cannot be canonicalized: %w", name, err)
	}
	if !bytes.Equal(raw, canonical) {
		return nil, fmt.Errorf("%s is not canonical JSON", name)
	}
	return value, nil
}

func decodeWithDuplicateCheck(raw []byte, target *any) error {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var walk func() error
	walk = func() error {
		token, err := decoder.Token()
		if err != nil {
			return err
		}
		if delim, ok := token.(json.Delim); ok {
			switch delim {
			case '{':
				seen := map[string]struct{}{}
				for decoder.More() {
					keyToken, err := decoder.Token()
					if err != nil {
						return err
					}
					key, ok := keyToken.(string)
					if !ok {
						return fmt.Errorf("object key is not a string")
					}
					if _, exists := seen[key]; exists {
						return fmt.Errorf("duplicate JSON key %q", key)
					}
					seen[key] = struct{}{}
					if err := walk(); err != nil {
						return err
					}
				}
			case '[':
				for decoder.More() {
					if err := walk(); err != nil {
						return err
					}
				}
			}
			_, err = decoder.Token()
			return err
		}
		return nil
	}
	if err := walk(); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return fmt.Errorf("trailing JSON data")
	}
	decoder = json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	return decoder.Decode(target)
}

func object(value any, name string) (map[string]any, error) {
	result, ok := value.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%s must be an object", name)
	}
	return result, nil
}

func exactFields(value map[string]any, required, optional map[string]struct{}, name string) error {
	for key := range value {
		if _, ok := required[key]; !ok {
			if _, ok = optional[key]; !ok {
				return fmt.Errorf("%s has unknown field %q", name, key)
			}
		}
	}
	for key := range required {
		if _, ok := value[key]; !ok {
			return fmt.Errorf("%s is missing field %q", name, key)
		}
	}
	return nil
}

func asNumber(value any, name string) (float64, error) {
	number, ok := value.(json.Number)
	if !ok {
		return 0, fmt.Errorf("%s must be a number", name)
	}
	parsed, err := strconv.ParseFloat(string(number), 64)
	if err != nil || math.IsNaN(parsed) || math.IsInf(parsed, 0) {
		return 0, fmt.Errorf("%s must be finite", name)
	}
	return parsed, nil
}

func asInt(value any, name string, minimum, maximum int64) (int64, error) {
	number, ok := value.(json.Number)
	if !ok {
		return 0, fmt.Errorf("%s must be an integer", name)
	}
	parsed, err := strconv.ParseInt(string(number), 10, 64)
	if err != nil || parsed < minimum || parsed > maximum {
		return 0, fmt.Errorf("%s is outside its bounded integer range", name)
	}
	return parsed, nil
}

func asBool(value any, name string) (bool, error) {
	result, ok := value.(bool)
	if !ok {
		return false, fmt.Errorf("%s must be a boolean", name)
	}
	return result, nil
}

func asString(value any, name string) (string, error) {
	result, ok := value.(string)
	if !ok || result == "" || strings.IndexByte(result, 0) >= 0 {
		return "", fmt.Errorf("%s must be a non-empty string without NUL", name)
	}
	return result, nil
}

func cloneJSON(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		result := make(map[string]any, len(typed))
		for key, item := range typed {
			result[key] = cloneJSON(item)
		}
		return result
	case []any:
		result := make([]any, len(typed))
		for index, item := range typed {
			result[index] = cloneJSON(item)
		}
		return result
	default:
		return typed
	}
}
