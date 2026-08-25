package ai42preflight

import (
	"context"
	"errors"
	"io"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

func TestCommandRunnerTimeoutDoesNotBlockOnWorkerStdin(t *testing.T) {
	t.Setenv("AI42_PREFLIGHT_HELPER", "1")
	runner, err := NewCommandRunner([]string{os.Args[0], "-test.run=TestCommandRunnerHelper"})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	_, err = runner.Run(ctx, []byte(`{"request":"bounded"}`))
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("timeout error=%v", err)
	}
}

func TestCommandRunnerWaitDelayBoundsInheritedPipeDescendant(t *testing.T) {
	t.Setenv("AI42_PREFLIGHT_PIPE_PARENT", "1")
	runner, err := NewCommandRunner([]string{os.Args[0], "-test.run=TestCommandRunnerPipeParentHelper"})
	if err != nil {
		t.Fatal(err)
	}
	started := time.Now()
	_, err = runner.Run(context.Background(), []byte(`{"request":"bounded"}`))
	if err == nil {
		t.Fatal("expected inherited-pipe wait failure")
	}
	if elapsed := time.Since(started); elapsed > 3*time.Second {
		t.Fatalf("runner waited %s for inherited descendant pipe", elapsed)
	}
	// The orphan deliberately outlives command.Wait/WaitDelay. Let that test
	// fixture exit before this package returns so Windows can unlink the test
	// executable; the bounded-runner assertion above excludes this cleanup wait.
	time.Sleep(2200 * time.Millisecond)
}

func TestCommandRunnerPipeParentHelper(t *testing.T) {
	if os.Getenv("AI42_PREFLIGHT_PIPE_PARENT") != "1" {
		return
	}
	child := exec.Command(os.Args[0], "-test.run=TestCommandRunnerPipeDescendantHelper")
	child.Env = append(os.Environ(), "AI42_PREFLIGHT_PIPE_DESCENDANT=1")
	child.Stdin, child.Stdout, child.Stderr = os.Stdin, os.Stdout, os.Stderr
	if err := child.Start(); err != nil {
		os.Exit(2)
	}
}

func TestCommandRunnerPipeDescendantHelper(t *testing.T) {
	if os.Getenv("AI42_PREFLIGHT_PIPE_DESCENDANT") != "1" {
		return
	}
	time.Sleep(2 * time.Second)
}

func TestCommandRunnerHelper(t *testing.T) {
	if os.Getenv("AI42_PREFLIGHT_HELPER") != "1" {
		return
	}
	_, _ = io.ReadAll(os.Stdin)
	select {}
}

func TestReadBoundedRejectsOverflow(t *testing.T) {
	if _, err := readBounded(strings.NewReader("abcd"), 3); err == nil {
		t.Fatal("expected bounded stream overflow")
	}
}
