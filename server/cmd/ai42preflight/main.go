package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"time"

	"tanatserver/internal/ai42preflight"
)

type controllerFlags []uint8

func (values *controllerFlags) String() string {
	return fmt.Sprint([]uint8(*values))
}

func (values *controllerFlags) Set(raw string) error {
	parsed, err := strconv.Atoi(raw)
	if err != nil || parsed < 0 || parsed > 3 {
		return fmt.Errorf("must be an integer in [0,3]")
	}
	*values = append(*values, uint8(parsed))
	return nil
}

func parseOptions(args []string) (ai42preflight.Options, error) {
	set := flag.NewFlagSet("ai42preflight", flag.ContinueOnError)
	set.SetOutput(io.Discard)
	set.Usage = func() {}
	var options ai42preflight.Options
	var supervisionControllers controllerFlags
	var torchPython string
	var timeout string
	set.StringVar(&options.ConfigPath, "config", "", "strict AI-42 preflight config")
	set.StringVar(&options.DatasetPath, "dataset", "", "validated durable-v2 dataset directory")
	set.StringVar(&options.DatasetHash, "dataset-hash", "", "expected dataset manifest SHA-256")
	set.StringVar(&options.ProfilePath, "profile", "", "frozen class profile JSON")
	set.StringVar(&options.ProfileHash, "profile-hash", "", "exact profile file SHA-256")
	set.StringVar(&options.WarmStartPath, "warm-start", "", "accepted warm-start checkpoint")
	set.StringVar(&options.OutputPath, "output", "", "preflight artifact directory")
	set.StringVar(&options.ReportPath, "report", "", "canonical report path")
	set.StringVar(&options.Device, "device", "auto", "Torch device")
	set.BoolVar(&options.AllowWarmStartDatasetChange, "allow-warm-start-dataset-change", false, "allow a model-only warm start from another verified dataset")
	set.Var(&supervisionControllers, "supervision-controller", "controller ID to supervise; repeat for multiple IDs")
	set.StringVar(&torchPython, "torch-python", "", "Python executable for the fixed Torch probe worker module")
	set.StringVar(&timeout, "worker-timeout", "", "worker timeout, for example 5m")
	if err := set.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			printUsage(os.Stdout)
			return ai42preflight.Options{}, flag.ErrHelp
		}
		return ai42preflight.Options{}, fmt.Errorf("invalid CLI flags: %w", err)
	}
	if set.NArg() != 0 {
		return ai42preflight.Options{}, fmt.Errorf("unexpected positional argument %q", set.Arg(0))
	}
	if torchPython == "" {
		return ai42preflight.Options{}, fmt.Errorf("--torch-python is required")
	}
	torchPython, err := filepath.Abs(torchPython)
	if err != nil {
		return ai42preflight.Options{}, fmt.Errorf("--torch-python: resolve path: %w", err)
	}
	if err := validateExecutablePath(torchPython); err != nil {
		return ai42preflight.Options{}, fmt.Errorf("--torch-python: %w", err)
	}
	options.TorchPythonPath = torchPython
	options.SupervisionControllers = append([]uint8(nil), supervisionControllers...)
	options.WorkerCommand = []string{torchPython, "-m", ai42preflight.TorchWorkerModule}
	if timeout == "" {
		return ai42preflight.Options{}, fmt.Errorf("--worker-timeout is required")
	}
	options.WorkerTimeout, err = time.ParseDuration(timeout)
	if err != nil || options.WorkerTimeout <= 0 || options.WorkerTimeout > ai42preflight.MaxWorkerTimeout {
		return ai42preflight.Options{}, fmt.Errorf("--worker-timeout must be in (0,%s]", ai42preflight.MaxWorkerTimeout)
	}
	return options, nil
}

func validateExecutablePath(path string) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("must not be a symlink")
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("must be a regular file")
	}
	if runtime.GOOS != "windows" && info.Mode().Perm()&0o111 == 0 {
		return fmt.Errorf("must be executable")
	}
	return nil
}

func printUsage(writer io.Writer) {
	_, _ = fmt.Fprintln(writer, "Usage: ai42preflight --config FILE --dataset DIR --warm-start FILE --output DIR --torch-python FILE --worker-timeout DURATION [flags]")
	_, _ = fmt.Fprintln(writer, "Flags: --config --dataset --dataset-hash --profile --profile-hash --warm-start --output --report --device --supervision-controller --allow-warm-start-dataset-change --torch-python --worker-timeout")
}

func main() {
	if len(os.Args) == 1 {
		printUsage(os.Stderr)
		os.Exit(2)
	}
	options, err := parseOptions(os.Args[1:])
	if errors.Is(err, flag.ErrHelp) {
		os.Exit(0)
	}
	if err != nil {
		_, _ = fmt.Fprintln(os.Stderr, err)
		printUsage(os.Stderr)
		os.Exit(2)
	}
	if _, err := ai42preflight.Run(context.Background(), options); err != nil {
		_, _ = fmt.Fprintln(os.Stderr, "AI-42 native preflight failed:", err)
		os.Exit(1)
	}
}
