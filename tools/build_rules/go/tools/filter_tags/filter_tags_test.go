package main

import (
	"go/build"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

var testFileCGO = `
// This file is not intended to actually build.

package cgo

/*
#include <stdio.h>
#include <stdlib.h>

void myprint(char* s) {
		printf("%s", s);
}
*/

import "C"

func main() {
	C.myprint("hello")
}
`

var testFileFILENAMETAG = `
// This file is not intended to actually compile.

package filenametag_darwin
`

var testFileIGNORE = `
// This file is not intended to actually build.

//+build ignore

package ignore
`

var testFileTAGS = `
// This file is not intended to actually build.

//+build arm,darwin linux,mips

package tags
`

func TestTags(t *testing.T) {
	tempdir, err := ioutil.TempDir("", "goruletest")
	if err != nil {
		t.Fatalf("Error creating temporary directory: %v", err)
	}
	defer os.RemoveAll(tempdir)

	for k, v := range map[string]string{
		"cgo.go":    testFileCGO,
		"darwin.go": testFileFILENAMETAG,
		"ignore.go": testFileIGNORE,
		"tags.go":   testFileTAGS,
	} {
		p := filepath.Join(tempdir, k)
		if err := ioutil.WriteFile(p, []byte(v), 0644); err != nil {
			t.Fatalf("WriteFile(%s): %v", p, err)
		}
	}

	testContext := build.Default
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}

	err = os.Chdir(tempdir)
	if err != nil {
		t.Fatalf("Chdir(%s): %v", tempdir, err)
	}
	defer os.Chdir(wd)

	// Test tags.go (tags in +build comments)
	testContext.BuildTags = []string{"arm", "darwin"}
	inputs := []string{"tags.go"}
	outputs, err := filterFilenames(testContext, inputs)
	if err != nil {
		t.Errorf("filterFilenames(%s): %v", inputs, err)
	}

	if !reflect.DeepEqual(inputs, outputs) {
		t.Error("Output missing an expected file: tags.go")
	}

	testContext.BuildTags = []string{"arm, linux"}
	outputs, err = filterFilenames(testContext, inputs)
	if err != nil {
		t.Errorf("filterFilenames(%s): %v", inputs, err)
	}

	if !reflect.DeepEqual([]string{}, outputs) {
		t.Error("Output contains an unexpected file: tags.go")
	}

	// Test ignore.go (should not build a file with +ignore comment)
	testContext.BuildTags = []string{}
	inputs = []string{"ignore.go"}
	outputs, err = filterFilenames(testContext, inputs)
	if err != nil {
		t.Errorf("filterFilenames(%s): %v", inputs, err)
	}

	if !reflect.DeepEqual([]string{}, outputs) {
		t.Error("Output contains an unexpected file: ignore.go")
	}
}
