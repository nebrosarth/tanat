package ai42preflight

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"time"
)

const commandWaitDelay = time.Second
const deterministicCUBLASWorkspace = ":4096:8"

type commandRunner struct{ command []string }

// NewCommandRunner returns the production direct-process runner. command must
// contain the executable followed by optional arguments; it is never passed
// through a shell.
func NewCommandRunner(command []string) (TorchRunner, error) {
	if len(command) == 0 || command[0] == "" {
		return nil, fmt.Errorf("worker command is required")
	}
	if err := validateExecutablePath(command[0]); err != nil {
		return nil, fmt.Errorf("worker executable: %w", err)
	}
	return &commandRunner{command: append([]string(nil), command...)}, nil
}

func (runner *commandRunner) Run(ctx context.Context, request []byte) (TorchOutput, error) {
	if len(request) > MaxWorkerRequest {
		return TorchOutput{}, fmt.Errorf("worker request exceeds %d-byte limit", MaxWorkerRequest)
	}
	if ctx == nil {
		ctx = context.Background()
	}
	commandContext, cancel := context.WithCancel(ctx)
	defer cancel()
	command := exec.CommandContext(commandContext, runner.command[0], runner.command[1:]...)
	// PyTorch deterministic algorithms require an explicit cuBLAS workspace
	// configuration on CUDA 10.2 and newer. Set it on the supervised worker so
	// every native preflight has the same deterministic contract, regardless of
	// the parent shell environment.
	command.Env = append(os.Environ(), "CUBLAS_WORKSPACE_CONFIG="+deterministicCUBLASWorkspace)
	// Let os/exec own the stdin copy goroutine. A synchronous StdinPipe.Write
	// could otherwise block forever if a faulty worker never reads its request,
	// bypassing the supervisor timeout.
	command.Stdin = bytes.NewReader(request)
	stdout, err := command.StdoutPipe()
	if err != nil {
		return TorchOutput{}, fmt.Errorf("open worker stdout: %w", err)
	}
	stderr, err := command.StderrPipe()
	if err != nil {
		return TorchOutput{}, fmt.Errorf("open worker stderr: %w", err)
	}
	// CommandContext cancels the direct child, but descendants can retain the
	// inherited stdout/stderr pipe handles. WaitDelay closes those pipes after
	// a bounded grace period so Wait cannot hang on an orphaned descendant.
	command.WaitDelay = commandWaitDelay
	if err := command.Start(); err != nil {
		return TorchOutput{}, fmt.Errorf("start worker: %w", err)
	}

	var waitGroup sync.WaitGroup
	var output, diagnostics []byte
	var outputErr, diagnosticsErr error
	waitGroup.Add(2)
	go func() {
		defer waitGroup.Done()
		output, outputErr = readBounded(stdout, MaxWorkerStdout)
		if outputErr != nil {
			cancel()
		}
	}()
	go func() {
		defer waitGroup.Done()
		diagnostics, diagnosticsErr = readBounded(stderr, MaxWorkerStderr)
		if diagnosticsErr != nil {
			cancel()
		}
	}()
	waitErr := command.Wait()
	waitGroup.Wait()
	if ctx.Err() != nil {
		return TorchOutput{Stdout: output, Stderr: diagnostics}, ctx.Err()
	}
	if outputErr != nil {
		return TorchOutput{Stdout: output, Stderr: diagnostics}, outputErr
	}
	if diagnosticsErr != nil {
		return TorchOutput{Stdout: output, Stderr: diagnostics}, diagnosticsErr
	}
	exitCode := 0
	if waitErr != nil {
		if exitError, ok := waitErr.(*exec.ExitError); ok {
			exitCode = exitError.ExitCode()
		} else {
			return TorchOutput{Stdout: output, Stderr: diagnostics}, fmt.Errorf("wait for worker: %w", waitErr)
		}
	}
	return TorchOutput{Stdout: output, Stderr: diagnostics, ExitCode: exitCode}, nil
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
	// Windows does not expose executable permission bits. The process
	// creation APIs still validate the image/script association at Start.
	if info.Mode().Perm()&0o111 == 0 && os.PathSeparator != '\\' {
		return fmt.Errorf("must be executable")
	}
	return nil
}

func readBounded(reader io.Reader, limit int) ([]byte, error) {
	buffer := make([]byte, 0, min(limit, 64*1024))
	scratch := make([]byte, 32*1024)
	for {
		count, err := reader.Read(scratch)
		if count > 0 {
			if len(buffer)+count <= limit {
				buffer = append(buffer, scratch[:count]...)
			} else {
				remaining := limit - len(buffer)
				if remaining > 0 {
					buffer = append(buffer, scratch[:remaining]...)
				}
				return buffer, fmt.Errorf("worker stream exceeded %d-byte limit", limit)
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return buffer, err
		}
	}
	return buffer, nil
}
func min(left, right int) int {
	if left < right {
		return left
	}
	return right
}
