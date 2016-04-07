/* Copyright (c) 2015, Google Inc.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION
 * OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
 * CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. */

package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// TODO(davidben): Link tests with the malloc shim and port -malloc-test to this runner.

var (
	useValgrind     = flag.Bool("valgrind", false, "If true, run code under valgrind")
	useGDB          = flag.Bool("gdb", false, "If true, run BoringSSL code under gdb")
	buildDir        = flag.String("build-dir", "build", "The build directory to run the tests from.")
	jsonOutput      = flag.String("json-output", "", "The file to output JSON results to.")
	mallocTest      = flag.Int64("malloc-test", -1, "If non-negative, run each test with each malloc in turn failing from the given number onwards.")
	mallocTestDebug = flag.Bool("malloc-test-debug", false, "If true, ask each test to abort rather than fail a malloc. This can be used with a specific value for --malloc-test to identity the malloc failing that is causing problems.")
)

type test []string

// testOutput is a representation of Chromium's JSON test result format. See
// https://www.chromium.org/developers/the-json-test-results-format
type testOutput struct {
	Version           int                   `json:"version"`
	Interrupted       bool                  `json:"interrupted"`
	PathDelimiter     string                `json:"path_delimiter"`
	SecondsSinceEpoch float64               `json:"seconds_since_epoch"`
	NumFailuresByType map[string]int        `json:"num_failures_by_type"`
	Tests             map[string]testResult `json:"tests"`
}

type testResult struct {
	Actual       string `json:"actual"`
	Expected     string `json:"expected"`
	IsUnexpected bool   `json:"is_unexpected"`
}

func newTestOutput() *testOutput {
	return &testOutput{
		Version:           3,
		PathDelimiter:     ".",
		SecondsSinceEpoch: float64(time.Now().UnixNano()) / float64(time.Second/time.Nanosecond),
		NumFailuresByType: make(map[string]int),
		Tests:             make(map[string]testResult),
	}
}

func (t *testOutput) addResult(name, result string) {
	if _, found := t.Tests[name]; found {
		panic(name)
	}
	t.Tests[name] = testResult{
		Actual:       result,
		Expected:     "PASS",
		IsUnexpected: result != "PASS",
	}
	t.NumFailuresByType[result]++
}

func (t *testOutput) writeTo(name string) error {
	file, err := os.Create(name)
	if err != nil {
		return err
	}
	defer file.Close()
	out, err := json.MarshalIndent(t, "", "  ")
	if err != nil {
		return err
	}
	_, err = file.Write(out)
	return err
}

func valgrindOf(dbAttach bool, path string, args ...string) *exec.Cmd {
	valgrindArgs := []string{"--error-exitcode=99", "--track-origins=yes", "--leak-check=full"}
	if dbAttach {
		valgrindArgs = append(valgrindArgs, "--db-attach=yes", "--db-command=xterm -e gdb -nw %f %p")
	}
	valgrindArgs = append(valgrindArgs, path)
	valgrindArgs = append(valgrindArgs, args...)

	return exec.Command("valgrind", valgrindArgs...)
}

func gdbOf(path string, args ...string) *exec.Cmd {
	xtermArgs := []string{"-e", "gdb", "--args"}
	xtermArgs = append(xtermArgs, path)
	xtermArgs = append(xtermArgs, args...)

	return exec.Command("xterm", xtermArgs...)
}

type moreMallocsError struct{}

func (moreMallocsError) Error() string {
	return "child process did not exhaust all allocation calls"
}

var errMoreMallocs = moreMallocsError{}

func runTestOnce(test test, mallocNumToFail int64) (passed bool, err error) {
	prog := path.Join(*buildDir, test[0])
	args := test[1:]
	var cmd *exec.Cmd
	if *useValgrind {
		cmd = valgrindOf(false, prog, args...)
	} else if *useGDB {
		cmd = gdbOf(prog, args...)
	} else {
		cmd = exec.Command(prog, args...)
	}
	var stdoutBuf bytes.Buffer
	var stderrBuf bytes.Buffer
	cmd.Stdout = &stdoutBuf
	cmd.Stderr = &stderrBuf
	if mallocNumToFail >= 0 {
		cmd.Env = os.Environ()
		cmd.Env = append(cmd.Env, "MALLOC_NUMBER_TO_FAIL="+strconv.FormatInt(mallocNumToFail, 10))
		if *mallocTestDebug {
			cmd.Env = append(cmd.Env, "MALLOC_ABORT_ON_FAIL=1")
		}
		cmd.Env = append(cmd.Env, "_MALLOC_CHECK=1")
	}

	if err := cmd.Start(); err != nil {
		return false, err
	}
	if err := cmd.Wait(); err != nil {
		if exitError, ok := err.(*exec.ExitError); ok {
			if exitError.Sys().(syscall.WaitStatus).ExitStatus() == 88 {
				return false, errMoreMallocs
			}
		}
		fmt.Print(string(stderrBuf.Bytes()))
		return false, err
	}
	fmt.Print(string(stderrBuf.Bytes()))

	// Account for Windows line-endings.
	stdout := bytes.Replace(stdoutBuf.Bytes(), []byte("\r\n"), []byte("\n"), -1)

	if bytes.HasSuffix(stdout, []byte("PASS\n")) &&
		(len(stdout) == 5 || stdout[len(stdout)-6] == '\n') {
		return true, nil
	}
	return false, nil
}

func runTest(test test) (bool, error) {
	if *mallocTest < 0 {
		return runTestOnce(test, -1)
	}

	for mallocNumToFail := int64(*mallocTest); ; mallocNumToFail++ {
		if passed, err := runTestOnce(test, mallocNumToFail); err != errMoreMallocs {
			if err != nil {
				err = fmt.Errorf("at malloc %d: %s", mallocNumToFail, err)
			}
			return passed, err
		}
	}
}

// shortTestName returns the short name of a test. Except for evp_test, it
// assumes that any argument which ends in .txt is a path to a data file and not
// relevant to the test's uniqueness.
func shortTestName(test test) string {
	var args []string
	for _, arg := range test {
		if test[0] == "crypto/evp/evp_test" || !strings.HasSuffix(arg, ".txt") {
			args = append(args, arg)
		}
	}
	return strings.Join(args, " ")
}

// setWorkingDirectory walks up directories as needed until the current working
// directory is the top of a BoringSSL checkout.
func setWorkingDirectory() {
	for i := 0; i < 64; i++ {
		if _, err := os.Stat("BUILDING.md"); err == nil {
			return
		}
		os.Chdir("..")
	}

	panic("Couldn't find BUILDING.md in a parent directory!")
}

func parseTestConfig(filename string) ([]test, error) {
	in, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer in.Close()

	decoder := json.NewDecoder(in)
	var result []test
	if err := decoder.Decode(&result); err != nil {
		return nil, err
	}
	return result, nil
}

func main() {
	flag.Parse()
	setWorkingDirectory()

	tests, err := parseTestConfig("util/all_tests.json")
	if err != nil {
		fmt.Printf("Failed to parse input: %s\n", err)
		os.Exit(1)
	}

	testOutput := newTestOutput()
	var failed []test
	for _, test := range tests {
		fmt.Printf("%s\n", strings.Join([]string(test), " "))

		name := shortTestName(test)
		passed, err := runTest(test)
		if err != nil {
			fmt.Printf("%s failed to complete: %s\n", test[0], err)
			failed = append(failed, test)
			testOutput.addResult(name, "CRASHED")
		} else if !passed {
			fmt.Printf("%s failed to print PASS on the last line.\n", test[0])
			failed = append(failed, test)
			testOutput.addResult(name, "FAIL")
		} else {
			testOutput.addResult(name, "PASS")
		}
	}

	if *jsonOutput != "" {
		if err := testOutput.writeTo(*jsonOutput); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		}
	}

	if len(failed) > 0 {
		fmt.Printf("\n%d of %d tests failed:\n", len(failed), len(tests))
		for _, test := range failed {
			fmt.Printf("\t%s\n", strings.Join([]string(test), " "))
		}
		os.Exit(1)
	}

	fmt.Printf("\nAll tests passed!\n")
}
