package ai42dataset

import "testing"

func BenchmarkCaptureFinalizeOneTerminalTick(b *testing.B) {
	b.ReportAllocs()
	for iteration := 0; iteration < b.N; iteration++ {
		metadata := testMetadata()
		capture, err := NewCapture(metadata)
		if err != nil {
			b.Fatal(err)
		}
		result := testResult(0, true)
		actions, parents, boundaries, outcomes := testRows(metadata.MatchID, 0)
		for slot := 0; slot < HeroCount; slot++ {
			outcomes[slot] = Outcome{Reward: result.Reward[slot], Terminal: true}
		}
		if err := capture.Append(&result, actions, parents, boundaries, outcomes); err != nil {
			b.Fatal(err)
		}
		if _, err := capture.Finalize(); err != nil {
			b.Fatal(err)
		}
	}
}
