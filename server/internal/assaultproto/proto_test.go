package assaultproto

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"io"
	"math"
	"strings"
	"testing"

	"tanatserver/internal/battleserver"
)

func requestFrame(command uint16, payload []byte) []byte {
	return requestFrameVersion(Version, command, payload)
}

func TestAI42SchemaHashMatchesCrossLanguageGolden(t *testing.T) {
	const want = "915e2e4547ccf727567304839f4780c60d48521f3dd1f0dbef7c4a5cc9131274"
	if got := hex.EncodeToString(AI42SchemaHash[:]); got != want {
		t.Fatalf("AI-42 schema hash drift: got %s, want %s", got, want)
	}
}

func TestAI42EvaluationControlledActionIsByteExact(t *testing.T) {
	const schemaWant = "a54f64514781db87ed2624720916c454d21a41ee2aabca6f094b0924e58e8bef"
	if got := hex.EncodeToString(AI42EvaluationSchemaHash[:]); got != schemaWant {
		t.Fatalf("AI-42 evaluation schema hash drift: got %s, want %s", got, schemaWant)
	}
	payload := make([]byte, controlledActionPayloadSize)
	payload[0] = byte(battleserver.AssaultControlIdle)
	payload[1] = byte(battleserver.AssaultActionMove)
	binary.LittleEndian.PutUint16(payload[2:4], 0x1234)
	payload[4], payload[5] = 80, 14
	request, err := ReadRequest(bytes.NewReader(requestFrameVersion(
		VersionAI42Evaluation, CommandStep, payload,
	)))
	if err != nil {
		t.Fatal(err)
	}
	if request.Controls[0] != battleserver.AssaultControlIdle ||
		request.Actions[0] != (battleserver.HeroActionV1{
			Kind: battleserver.AssaultActionMove, Target: 0x1234, Direction: 80, Distance: 14,
		}) {
		t.Fatalf("decoded controlled action=%+v/%d", request.Actions[0], request.Controls[0])
	}
	if newResultFrameLayout(VersionAI42Evaluation).bodySize != resultBodySizeV14 {
		t.Fatal("AI-42 evaluation response has an unexpected layout")
	}
}

func TestAI42DAggerInterventionWireIsByteExact(t *testing.T) {
	const schemaWant = "06369e1df3d48649c080938403ff0f5d7310a74b65b33903d1454872a45d1a28"
	if got := hex.EncodeToString(AI42DAggerSchemaHash[:]); got != schemaWant {
		t.Fatalf("AI-42 DAgger schema hash drift: got %s, want %s", got, schemaWant)
	}
	payload := make([]byte, daggerActionPayloadSize)
	payload[0] = byte(battleserver.AssaultControlIssue)
	payload[1] = byte(battleserver.AssaultActionAttack)
	binary.LittleEndian.PutUint16(payload[2:4], 0x1234)
	payload[6] = 1
	request, err := ReadRequest(bytes.NewReader(requestFrameVersion(
		VersionAI42DAgger, CommandStep, payload,
	)))
	if err != nil {
		t.Fatal(err)
	}
	if request.Interventions[0] != 1 || request.Actions[0].Target != 0x1234 {
		t.Fatalf("decoded DAgger request=%+v intervention=%d", request.Actions[0], request.Interventions[0])
	}
	payload[6] = 2
	if _, err := ReadRequest(bytes.NewReader(requestFrameVersion(
		VersionAI42DAgger, CommandStep, payload,
	))); err == nil || !strings.Contains(err.Error(), "intervention[0]") {
		t.Fatalf("invalid intervention was accepted: %v", err)
	}

	var result battleserver.StepResultV1
	result.RewardHash = battleserver.AssaultRewardHashV5
	result.TeacherStatus[0] = battleserver.AssaultTeacherStatusAction
	result.ActiveOrder[0] = 1
	var out bytes.Buffer
	if err := NewResultEncoder().WriteVersion(&out, &result, VersionAI42DAgger); err != nil {
		t.Fatal(err)
	}
	frame := append([]byte(nil), out.Bytes()...)
	body, err := readFrame(&out)
	if err != nil {
		t.Fatal(err)
	}
	layout := newResultFrameLayout(VersionAI42DAgger)
	if body[layout.teacherStatusOffset] != battleserver.AssaultTeacherStatusAction ||
		body[layout.activeOrderOffset] != 1 {
		t.Fatal("v15 teacher/active-order fields did not round-trip")
	}
	decoded, err := ReadResultVersion(bytes.NewReader(frame), VersionAI42DAgger)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.TeacherStatus[0] != battleserver.AssaultTeacherStatusAction || decoded.ActiveOrder[0] != 1 {
		t.Fatalf("decoded v15 fields drifted: teacher=%d active=%d", decoded.TeacherStatus[0], decoded.ActiveOrder[0])
	}
}

func TestReadResultVersionRoundTripsCompleteAI42DAggerRecord(t *testing.T) {
	var source battleserver.StepResultV1
	source.RewardHash = battleserver.AssaultRewardHashV5
	source.Step, source.Elapsed, source.Done, source.Winner = 17, 3.4, true, -1
	source.Invalid[2], source.Reward[2] = 1, -2.5
	source.Observations[2].Hero[3] = 4.5
	source.Observations[2].Abilities[1][7] = 5.5
	source.Observations[2].Entities[9][11] = 6.5
	source.Observations[2].Global[4] = 7.5
	source.Observations[2].EntityMask[9] = 1
	source.Observations[2].ActionMask.Kinds[3] = 1
	source.Observations[2].ActionMask.Targets[9] = 1
	source.Observations[2].ActionMask.SkillTarget[1][9] = 1
	source.TeacherIntent[2] = battleserver.HeroActionV1{Kind: 3, Target: 9, Direction: 8, Distance: 7}
	source.TeacherStatus[2] = battleserver.AssaultTeacherStatusAction
	source.ExecutedActions[2] = battleserver.HeroActionV1{Kind: 4, Target: 8, Direction: 7, Distance: 6}
	source.ExecutedValid[2], source.RejectionReason[2], source.ActiveOrder[2] = 1, 0, 1
	var encoded bytes.Buffer
	if err := NewResultEncoder().WriteVersion(&encoded, &source, VersionAI42DAgger); err != nil {
		t.Fatal(err)
	}
	decoded, err := ReadResultVersion(&encoded, VersionAI42DAgger)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Step != source.Step || decoded.Elapsed != source.Elapsed || decoded.Done != source.Done || decoded.Winner != source.Winner ||
		decoded.Invalid != source.Invalid || decoded.Reward != source.Reward || decoded.Observations != source.Observations ||
		decoded.TeacherIntent != source.TeacherIntent || decoded.TeacherStatus != source.TeacherStatus ||
		decoded.ExecutedActions != source.ExecutedActions || decoded.ExecutedValid != source.ExecutedValid ||
		decoded.RejectionReason != source.RejectionReason || decoded.ActiveOrder != source.ActiveOrder {
		t.Fatal("decoded v15 scalar result differs from encoded source")
	}
}

func TestAI42EvaluationActiveOrderResponseIsByteExact(t *testing.T) {
	var result battleserver.StepResultV1
	result.RewardHash = battleserver.AssaultRewardHashV5
	result.ActiveOrder[3] = 1
	var out bytes.Buffer
	if err := NewResultEncoder().WriteVersion(&out, &result, VersionAI42Evaluation); err != nil {
		t.Fatal(err)
	}
	body, err := readFrame(&out)
	if err != nil {
		t.Fatal(err)
	}
	layout := newResultFrameLayout(VersionAI42Evaluation)
	if len(body) != resultBodySizeV14 || body[layout.activeOrderOffset+3] != 1 {
		t.Fatalf("v14 active-order bytes drifted: size=%d value=%d", len(body), body[layout.activeOrderOffset+3])
	}
}

func requestFrameVersion(version, command uint16, payload []byte) []byte {
	body := new(bytes.Buffer)
	body.Write(magic[:])
	_ = binary.Write(body, binary.LittleEndian, version)
	_ = binary.Write(body, binary.LittleEndian, command)
	body.Write(payload)
	frame := new(bytes.Buffer)
	_ = binary.Write(frame, binary.LittleEndian, uint32(body.Len()))
	frame.Write(body.Bytes())
	return frame.Bytes()
}

func TestAI41TeacherFrameIsByteExactAndAI42AppendLayoutIsActive(t *testing.T) {
	var result battleserver.StepResultV1
	result.Step = 0x11223344
	result.Elapsed = 12.5
	result.Done = true
	result.Winner = -2
	result.Invalid[0] = 7
	result.Reward[0] = 1.25
	result.RewardHash = battleserver.AssaultRewardHashV5
	result.Observations[0].Hero[0] = 1.5
	result.Observations[0].Abilities[0][0] = 2.5
	result.Observations[0].Entities[0][0] = 3.5
	result.Observations[0].Global[0] = 4.5
	result.Observations[0].EntityMask[0] = 5
	result.Observations[0].ActionMask.Kinds[0] = 6
	result.Observations[0].ActionMask.Targets[0] = 7
	result.Observations[0].ActionMask.SkillTarget[0][0] = 8
	result.TeacherActions[0] = battleserver.HeroActionV1{
		Kind: battleserver.AssaultActionAttack, Target: 0x1234, Direction: 9, Distance: 8,
	}
	result.TeacherValid[0] = 1

	var out bytes.Buffer
	if err := NewResultEncoder().WriteVersion(&out, &result, VersionAI41Teacher); err != nil {
		t.Fatal(err)
	}
	body, err := readFrame(&out)
	if err != nil {
		t.Fatal(err)
	}
	want := make([]byte, resultBodySizeV3)
	copy(want[0:4], magic[:])
	binary.LittleEndian.PutUint16(want[4:6], VersionAI41Teacher)
	binary.LittleEndian.PutUint16(want[6:8], ResponseResult)
	copy(want[8:40], battleserver.AssaultSchemaHashV4[:])
	copy(want[40:72], battleserver.AssaultRewardHashV5[:])
	binary.LittleEndian.PutUint32(want[72:76], result.Step)
	binary.LittleEndian.PutUint32(want[76:80], math.Float32bits(result.Elapsed))
	want[80] = 1
	binary.LittleEndian.PutUint32(want[84:88], uint32(result.Winner))
	want[88] = result.Invalid[0]
	binary.LittleEndian.PutUint32(want[98:102], math.Float32bits(result.Reward[0]))
	binary.LittleEndian.PutUint32(want[138:142], math.Float32bits(1.5))
	binary.LittleEndian.PutUint32(want[266:270], math.Float32bits(2.5))
	binary.LittleEndian.PutUint32(want[906:910], math.Float32bits(3.5))
	binary.LittleEndian.PutUint32(want[7050:7054], math.Float32bits(4.5))
	want[7178], want[7274], want[7282], want[7378] = 5, 6, 7, 8
	want[76378] = byte(battleserver.AssaultActionAttack)
	binary.LittleEndian.PutUint16(want[76379:76381], 0x1234)
	want[76381], want[76382], want[76428] = 9, 8, 1
	if !bytes.Equal(body, want) {
		t.Fatalf("strategic teacher frame drifted; first mismatch body=%x want=%x", body[72:142], want[72:142])
	}

	teacher := newResultFrameLayout(VersionAI41Teacher)
	ai42 := newResultFrameLayout(VersionAI42)
	if teacher.bodySize != resultBodySizeV3 || ai42.bodySize != resultBodySizeV13 ||
		!supportedVersion(VersionAI42) || resultBodySizeV13 != 76508 {
		t.Fatalf("unexpected active/v13 layouts: teacher=%+v ai42=%+v", teacher, ai42)
	}
	for _, field := range []struct {
		name string
		off  int
		size int
	}{
		{"result.observation[0].hero", 138, 128},
		{"result.observation[0].abilities", 266, 640},
		{"result.teacher_actions", 76378, 50},
		{"result.teacher_valid", 76428, 10},
	} {
		var got resultLayoutField
		found := false
		for _, candidate := range teacher.fields {
			if candidate.name == field.name {
				got, found = candidate, true
				break
			}
		}
		if !found || got.offset != field.off || got.size != field.size {
			t.Fatalf("field %q = %+v, want offset=%d size=%d", field.name, got, field.off, field.size)
		}
	}
	if got := newVectorResultLayoutVersion(2, VersionAI41Teacher); got.size != 152802 {
		t.Fatalf("teacher vector size=%d, want 152802", got.size)
	}
	if got := newVectorResultLayoutVersion(2, VersionAI42); got.size != 152942 {
		t.Fatalf("AI-42 vector size=%d, want 152942", got.size)
	}
	for _, field := range []struct {
		name string
		off  int
		size int
	}{
		{"result.teacher_intent", 76378, 50},
		{"result.teacher_status", 76428, 10},
		{"result.executed_actions", 76438, 50},
		{"result.executed_valid", 76488, 10},
		{"result.rejection_reason", 76498, 10},
	} {
		var got resultLayoutField
		for _, candidate := range ai42.fields {
			if candidate.name == field.name {
				got = candidate
				break
			}
		}
		if got.offset != field.off || got.size != field.size {
			t.Fatalf("field %q = %+v, want offset=%d size=%d", field.name, got, field.off, field.size)
		}
	}
	if got := hex.EncodeToString(AI42RewardHash[:]); got != hex.EncodeToString(battleserver.AssaultRewardHashV5[:]) {
		t.Fatalf("AI-42 reward hash=%s, want strategic V5", got)
	}
}

func TestFloatSerializationIsAlwaysLittleEndian(t *testing.T) {
	body := make([]byte, 8)
	copyFloat32(body, 0, []float32{
		math.Float32frombits(0x01020304), math.Float32frombits(0xa1b2c3d4),
	})
	want := []byte{0x04, 0x03, 0x02, 0x01, 0xd4, 0xc3, 0xb2, 0xa1}
	if !bytes.Equal(body, want) {
		t.Fatalf("float bytes=%x, want little-endian %x", body, want)
	}

	var result battleserver.StepResultV1
	result.RewardHash = battleserver.AssaultRewardHashV5
	result.Observations[0].Hero[0] = math.Float32frombits(0x01020304)
	var out bytes.Buffer
	if err := NewResultEncoder().WriteVersion(&out, &result, VersionAI41Teacher); err != nil {
		t.Fatal(err)
	}
	frame, err := readFrame(&out)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(frame[138:142], want[:4]) {
		t.Fatalf("encoded hero float bytes=%x, want %x", frame[138:142], want[:4])
	}
}

func TestAI42AppendFieldsAreByteExact(t *testing.T) {
	var result battleserver.StepResultV1
	result.RewardHash = battleserver.AssaultRewardHashV5
	result.TeacherIntent[0] = battleserver.HeroActionV1{
		Kind: battleserver.AssaultActionMove, Direction: 80, Distance: 14,
	}
	result.TeacherStatus[0] = battleserver.AssaultTeacherStatusAction
	result.ExecutedActions[0] = battleserver.HeroActionV1{
		Kind: battleserver.AssaultActionAttack, Target: 0x1234,
	}
	result.ExecutedValid[0] = 1
	result.RejectionReason[1] = battleserver.AssaultRejectionReasonMasked
	var out bytes.Buffer
	if err := NewResultEncoder().WriteVersion(&out, &result, VersionAI42); err != nil {
		t.Fatal(err)
	}
	body, err := readFrame(&out)
	if err != nil {
		t.Fatal(err)
	}
	if len(body) != 76508 || !bytes.Equal(body[40:72], battleserver.AssaultRewardHashV5[:]) {
		t.Fatalf("v13 size/reward hash = %d/%x", len(body), body[40:44])
	}
	if body[76378] != byte(battleserver.AssaultActionMove) || body[76379] != 0 ||
		body[76380] != 0 || body[76381] != 80 || body[76382] != 14 ||
		body[76428] != battleserver.AssaultTeacherStatusAction ||
		body[76438] != byte(battleserver.AssaultActionAttack) ||
		binary.LittleEndian.Uint16(body[76439:76441]) != 0x1234 ||
		body[76488] != 1 || body[76499] != battleserver.AssaultRejectionReasonMasked {
		t.Fatalf("v13 append bytes drifted")
	}
}

func TestProtocolDiagnosticsIdentifyTruncatedTrailingAndReservedFrames(t *testing.T) {
	tests := []struct {
		name string
		data []byte
		want []string
	}{
		{"truncated actions", requestFrame(CommandStep, make([]byte, actionPayloadSize-1)), []string{"field=STEP.actions", "offset=57"}},
		{"trailing actions", requestFrame(CommandStep, make([]byte, actionPayloadSize+1)), []string{"field=STEP.trailing", "offset=58"}},
		{"v13 close payload", requestFrameVersion(VersionAI42, CommandClose, nil), nil},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := ReadRequest(bytes.NewReader(test.data))
			if test.want == nil {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil {
				t.Fatal("ReadRequest unexpectedly succeeded")
			}
			for _, part := range test.want {
				if !strings.Contains(err.Error(), part) {
					t.Fatalf("error=%q, missing %q", err, part)
				}
			}
		})
	}

	truncated := make([]byte, 4+7)
	binary.LittleEndian.PutUint32(truncated[:4], 8)
	if _, err := ReadRequest(bytes.NewReader(truncated)); err == nil ||
		!strings.Contains(err.Error(), "field=frame.body") || !strings.Contains(err.Error(), "offset=11") {
		t.Fatalf("truncated frame error=%v", err)
	}
}

func TestReadAI41RequestVersion(t *testing.T) {
	req, err := ReadRequest(bytes.NewReader(requestFrameVersion(VersionAI41, CommandClose, nil)))
	if err != nil {
		t.Fatal(err)
	}
	if req.Version != VersionAI41 || req.Command != CommandClose {
		t.Fatalf("request version/command = %d/%d", req.Version, req.Command)
	}
}

func TestReadAI41WrongLaneRequestVersion(t *testing.T) {
	req, err := ReadRequest(bytes.NewReader(requestFrameVersion(VersionAI41WrongLane, CommandClose, nil)))
	if err != nil {
		t.Fatal(err)
	}
	if req.Version != VersionAI41WrongLane || req.Command != CommandClose {
		t.Fatalf("request version/command = %d/%d", req.Version, req.Command)
	}
}

func TestReadAI41EvaluationRequestVersion(t *testing.T) {
	req, err := ReadRequest(bytes.NewReader(requestFrameVersion(VersionAI41Evaluation, CommandClose, nil)))
	if err != nil {
		t.Fatal(err)
	}
	if req.Version != VersionAI41Evaluation || req.Command != CommandClose {
		t.Fatalf("request version/command = %d/%d", req.Version, req.Command)
	}
}

func TestReadAI41NavigationRequestVersions(t *testing.T) {
	for _, version := range []uint16{
		VersionAI41Navigation, VersionAI41NavigationEvaluation,
		VersionAI41Strategic, VersionAI41StrategicEvaluation, VersionAI41Teacher,
	} {
		req, err := ReadRequest(bytes.NewReader(requestFrameVersion(version, CommandClose, nil)))
		if err != nil {
			t.Fatal(err)
		}
		if req.Version != version || req.Command != CommandClose {
			t.Fatalf("request version/command = %d/%d", req.Version, req.Command)
		}
	}
}

func TestWriteAI41TeacherResultIncludesTeacherLabels(t *testing.T) {
	var result battleserver.StepResultV1
	result.RewardHash = battleserver.AssaultRewardHashV5
	result.TeacherActions[3] = battleserver.HeroActionV1{
		Kind: battleserver.AssaultActionAttack, Target: 17, Direction: 2, Distance: 1,
	}
	result.TeacherValid[3] = 1
	var out bytes.Buffer
	if err := NewResultEncoder().WriteVersion(&out, &result, VersionAI41Teacher); err != nil {
		t.Fatal(err)
	}
	body, err := readFrame(&out)
	if err != nil {
		t.Fatal(err)
	}
	labelOffset := resultBodySizeV2
	if len(body) != resultBodySizeV3 ||
		binary.LittleEndian.Uint16(body[4:6]) != VersionAI41Teacher ||
		!bytes.Equal(body[8:40], battleserver.AssaultSchemaHashV4[:]) ||
		body[labelOffset+3*5] != byte(battleserver.AssaultActionAttack) ||
		binary.LittleEndian.Uint16(body[labelOffset+3*5+1:labelOffset+3*5+3]) != 17 ||
		body[labelOffset+battleserver.AssaultHeroCount*5+3] != 1 {
		t.Fatal("bad AI-41 teacher result layout")
	}
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

func BenchmarkWriteAI42ResultNoDrift(b *testing.B) {
	var result battleserver.StepResultV1
	result.ExecutedActions[0] = battleserver.HeroActionV1{Kind: 2, Target: 7, Direction: 40, Distance: 3}
	result.ExecutedValid[0] = 1
	result.TeacherStatus[0] = battleserver.AssaultTeacherStatusAction

	encoder := NewResultEncoder()
	var out bytes.Buffer
	if err := encoder.WriteVersion(&out, &result, VersionAI42); err != nil {
		b.Fatal(err)
	}
	want := append([]byte(nil), out.Bytes()...)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out.Reset()
		if err := encoder.WriteVersion(&out, &result, VersionAI42); err != nil {
			b.Fatal(err)
		}
		if !bytes.Equal(out.Bytes(), want) {
			b.Fatalf("v13 frame drift at iteration %d", i)
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

func TestReadVectorStepRequest(t *testing.T) {
	payload := make([]byte, 4+2*actionPayloadSize)
	binary.LittleEndian.PutUint32(payload[:4], 2)
	payload[4] = byte(battleserver.AssaultActionMove)
	payload[4+actionPayloadSize] = byte(battleserver.AssaultActionAttack)
	req, err := ReadRequest(bytes.NewReader(requestFrame(CommandVectorStep, payload)))
	if err != nil {
		t.Fatal(err)
	}
	if len(req.VectorActions) != 2 ||
		req.VectorActions[0][0].Kind != battleserver.AssaultActionMove ||
		req.VectorActions[1][0].Kind != battleserver.AssaultActionAttack {
		t.Fatalf("bad vector actions: %+v", req.VectorActions)
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

func TestWriteAI41ResultIncludesAbilityTensor(t *testing.T) {
	var result battleserver.StepResultV1
	result.SchemaHash = battleserver.AssaultSchemaHashV1
	result.RewardHash = battleserver.AssaultRewardHashV2
	result.Observations[0].Abilities[0][0] = 12.5
	var out bytes.Buffer
	if err := NewResultEncoder().WriteVersion(&out, &result, VersionAI41); err != nil {
		t.Fatal(err)
	}
	body, err := readFrame(&out)
	if err != nil {
		t.Fatal(err)
	}
	const observationOffset = 138
	abilityOffset := observationOffset + battleserver.AssaultHeroFeatureSize*4
	if len(body) != resultBodySizeV2 ||
		binary.LittleEndian.Uint16(body[4:6]) != VersionAI41 ||
		!bytes.Equal(body[8:40], battleserver.AssaultSchemaHashV2[:]) ||
		math.Float32frombits(binary.LittleEndian.Uint32(body[abilityOffset:])) != 12.5 {
		t.Fatalf("bad AI-41 result layout")
	}
}

func TestWriteAI41WrongLaneResultUsesVersionedContracts(t *testing.T) {
	var result battleserver.StepResultV1
	result.SchemaHash = battleserver.AssaultSchemaHashV1
	result.RewardHash = battleserver.AssaultRewardHashV3
	result.Observations[0].Abilities[0][0] = 7.5
	result.Observations[0].Global[8] = 1
	var out bytes.Buffer
	if err := NewResultEncoder().WriteVersion(&out, &result, VersionAI41WrongLane); err != nil {
		t.Fatal(err)
	}
	body, err := readFrame(&out)
	if err != nil {
		t.Fatal(err)
	}
	if len(body) != resultBodySizeV2 ||
		binary.LittleEndian.Uint16(body[4:6]) != VersionAI41WrongLane ||
		!bytes.Equal(body[8:40], battleserver.AssaultSchemaHashV3[:]) ||
		!bytes.Equal(body[40:72], battleserver.AssaultRewardHashV3[:]) {
		t.Fatal("bad AI-41 wrong-lane result contract")
	}
}

func TestWriteAI41NavigationResultUsesV4Schema(t *testing.T) {
	var result battleserver.StepResultV1
	result.SchemaHash = battleserver.AssaultSchemaHashV1
	result.RewardHash = battleserver.AssaultRewardHashV3
	var out bytes.Buffer
	if err := NewResultEncoder().WriteVersion(&out, &result, VersionAI41Navigation); err != nil {
		t.Fatal(err)
	}
	body, err := readFrame(&out)
	if err != nil {
		t.Fatal(err)
	}
	if len(body) != resultBodySizeV2 ||
		binary.LittleEndian.Uint16(body[4:6]) != VersionAI41Navigation ||
		!bytes.Equal(body[8:40], battleserver.AssaultSchemaHashV4[:]) ||
		!bytes.Equal(body[40:72], battleserver.AssaultRewardHashV3[:]) {
		t.Fatal("bad AI-41 navigation result contract")
	}
}

func TestWriteVectorResultIsSingleFramedBatch(t *testing.T) {
	results := make([]battleserver.StepResultV1, 2)
	pointers := make([]*battleserver.StepResultV1, len(results))
	for i := range results {
		results[i].SchemaHash = battleserver.AssaultSchemaHashV1
		results[i].RewardHash = battleserver.AssaultRewardHashV2
		results[i].Step = uint32(i + 7)
		pointers[i] = &results[i]
	}
	var out bytes.Buffer
	if err := NewVectorResultEncoder().Write(&out, pointers); err != nil {
		t.Fatal(err)
	}
	body, err := readFrame(&out)
	if err != nil {
		t.Fatal(err)
	}
	layout := newVectorResultLayout(2)
	if len(body) != layout.size ||
		binary.LittleEndian.Uint16(body[6:8]) != ResponseVectorResult ||
		binary.LittleEndian.Uint32(body[72:76]) != 2 ||
		binary.LittleEndian.Uint32(body[layout.steps:layout.steps+4]) != 7 ||
		binary.LittleEndian.Uint32(body[layout.steps+4:layout.steps+8]) != 8 {
		t.Fatalf("bad vector result header/records")
	}
}
