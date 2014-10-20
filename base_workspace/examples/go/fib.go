package main

import (
	"fmt"

	// TODO(bazel-team): remap compiler inputs so this is "go/lib1"
	"examples/go/lib1/lib1"
)

func main() {
	fmt.Println("Fib(5):", lib1.Fib(5))
}
