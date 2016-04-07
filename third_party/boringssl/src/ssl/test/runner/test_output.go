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

package runner

import (
	"encoding/json"
	"os"
	"time"
)

// testOutput is a representation of Chromium's JSON test result format. See
// https://www.chromium.org/developers/the-json-test-results-format
type testOutput struct {
	Version           int                   `json:"version"`
	Interrupted       bool                  `json:"interrupted"`
	PathDelimiter     string                `json:"path_delimiter"`
	SecondsSinceEpoch float64               `json:"seconds_since_epoch"`
	NumFailuresByType map[string]int        `json:"num_failures_by_type"`
	Tests             map[string]testResult `json:"tests"`
	allPassed         bool
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
		allPassed:         true,
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
	if result != "PASS" {
		t.allPassed = false
	}
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
