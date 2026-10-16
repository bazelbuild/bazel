// Example for using runfiles with Go.
package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/bazelbuild/rules_go/go/runfiles"
)

const (
	someFile = "examples/runfile.txt"
)

func main() {
	r, err := runfiles.New()
	if err != nil {
		log.Fatalf("Failed to create Runfiles object: %v", err)
	}

	root := runfiles.CallerRepository()
	if root == "" {
		root = "_main"
	}

	path, err := r.Rlocation(filepath.Join(root, someFile))
	if err != nil {
		log.Fatalf("Failed to find rlocation: %v", err)
	}

	fmt.Println("The content of my runfile is:")
	data, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("Failed to read file: %v", err)
	}
	fmt.Print(string(data))
}
