// Command assaultenv exposes one synchronous AssaultEnv over stdin/stdout.
// Run multiple processes for parallel rollout workers.
package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"log"
	"os"

	"tanatserver/internal/assaultproto"
	"tanatserver/internal/battleserver"
)

func main() {
	// Battle-server logging is intentionally verbose enough to fill a subprocess
	// stderr pipe during a rollout. Keep workers silent unless explicitly debugging.
	if os.Getenv("TANAT_ASSAULTENV_DEBUG") == "" {
		log.SetOutput(io.Discard)
	}
	if err := run(os.Stdin, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(input io.Reader, output io.Writer) error {
	env := battleserver.NewAssaultEnv()
	defer env.Close()
	resultEncoder := assaultproto.NewResultEncoder()
	in := bufio.NewReaderSize(input, 64<<10)
	out := bufio.NewWriterSize(output, 128<<10)
	for {
		request, err := assaultproto.ReadRequest(in)
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			_ = assaultproto.WriteError(out, err.Error())
			return err
		}
		switch request.Command {
		case assaultproto.CommandReset:
			result, err := env.Reset(request.Reset)
			if err != nil {
				if writeErr := assaultproto.WriteError(out, err.Error()); writeErr != nil {
					return writeErr
				}
				continue
			}
			if err := resultEncoder.Write(out, &result); err != nil {
				return err
			}
		case assaultproto.CommandStep:
			result, err := env.Step(request.Actions)
			if err != nil {
				if writeErr := assaultproto.WriteError(out, err.Error()); writeErr != nil {
					return writeErr
				}
				continue
			}
			if err := resultEncoder.Write(out, &result); err != nil {
				return err
			}
		case assaultproto.CommandClose:
			return nil
		}
	}
}
