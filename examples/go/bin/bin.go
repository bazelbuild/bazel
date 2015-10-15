package main

import (
	"fmt"

	"github_com/user/vendored"

	"github.com/bazelbuild/bazel/examples/go/lib"
)

func main() {
	fmt.Println("meaning: ", lib.Meaning())
	fmt.Println("vendored: ", vendored.Vendored())
}
