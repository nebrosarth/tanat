package main

import (
	"errors"
	"flag"
	"os"
	"strings"
	"testing"

	"tanatserver/internal/ai42preflight"
)

func TestParseOptionsRejectsExecuteAndUnknownFlags(t *testing.T) {
	for _, args := range [][]string{{"--execute"}, {"--unknown"}, {"--worker-command", "python"}, {"--worker-arg", "--unsafe"}} {
		if _, err := parseOptions(args); err == nil || strings.Contains(err.Error(), "Python fallback") {
			t.Fatalf("args=%q error=%v", args, err)
		}
	}
}

func TestParseOptionsRequiresExplicitWorkerAndTimeout(t *testing.T) {
	if _, err := parseOptions(nil); err == nil || !strings.Contains(err.Error(), "torch-python") {
		t.Fatalf("missing worker error=%v", err)
	}
	python := os.Args[0]
	if _, err := parseOptions([]string{"--torch-python", python}); err == nil || !strings.Contains(err.Error(), "worker-timeout") {
		t.Fatalf("missing timeout error=%v", err)
	}
	if _, err := parseOptions([]string{"--torch-python", python, "--worker-timeout", "1s", "extra"}); err == nil || !strings.Contains(err.Error(), "positional") {
		t.Fatalf("positional error=%v", err)
	}
	if _, err := parseOptions([]string{"--torch-python", python, "--worker-timeout", "10m1s"}); err == nil || !strings.Contains(err.Error(), "(0,10m0s]") {
		t.Fatalf("timeout bound error=%v", err)
	}
	if _, err := parseOptions([]string{"--help"}); !errors.Is(err, flag.ErrHelp) {
		t.Fatalf("help error=%v", err)
	}
}

func TestParseOptionsBuildsOnlyFixedWorkerModule(t *testing.T) {
	options, err := parseOptions([]string{"--torch-python", os.Args[0], "--worker-timeout", "1s"})
	if err != nil {
		t.Fatal(err)
	}
	if len(options.WorkerCommand) != 3 || options.WorkerCommand[0] != options.TorchPythonPath || options.WorkerCommand[1] != "-m" || options.WorkerCommand[2] != ai42preflight.TorchWorkerModule {
		t.Fatalf("worker command=%q", options.WorkerCommand)
	}
}
