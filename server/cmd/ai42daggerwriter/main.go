// Command ai42daggerwriter consumes an alternating stream of v15 scalar STEP
// request/result frames and publishes one strict native AI-42 dataset shard.
// Policy inference stays outside this process; no observation is rebuilt as a
// Python record on the durable capture path.
package main

import (
	"flag"
	"fmt"
	"os"

	"tanatserver/internal/ai42dagger"
)

func main() {
	schedulePath := flag.String("schedule", "", "canonical runtime/schedule JSON")
	outputPath := flag.String("output", "", "one-match generation directory")
	matchIndex := flag.Int("match-index", 0, "schedule entry index")
	reserveTicks := flag.Int("reserve-ticks", 2048, "initial native column capacity")
	flag.Parse()
	if *schedulePath == "" || *outputPath == "" {
		fatal("-schedule and -output are required")
	}
	if *matchIndex < 0 || *reserveTicks < 0 {
		fatal("-match-index and -reserve-ticks must be non-negative")
	}
	result, err := ai42dagger.WriteStream(os.Stdin, ai42dagger.WriterOptions{
		SchedulePath: *schedulePath, OutputPath: *outputPath,
		MatchIndex: *matchIndex, ReserveTicks: *reserveTicks,
	})
	if err != nil {
		fatal("%v", err)
	}
	fmt.Printf("match_id=%s ticks=%d output=%s\n", result.MatchID, result.Ticks, *outputPath)
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "ai42daggerwriter: "+format+"\n", args...)
	os.Exit(2)
}
