package main

import (
	"flag"
	"fmt"
	"go/build"
	"log"
	"path/filepath"
	"strings"
)

// Returns an array of strings containing only the filenames that should build
// according to the Context given.
func filterFilenames(bctx build.Context, inputs []string) ([]string, error) {
	outputs := []string{}

	for _, filename := range inputs {
		fullPath, err := filepath.Abs(filename)
		if err != nil {
			return nil, err
		}
		dir, base := filepath.Split(fullPath)

		matches, err := bctx.MatchFile(dir, base)
		if err != nil {
			return nil, err
		}

		if matches {
			outputs = append(outputs, filename)
		}
	}
	return outputs, nil
}

func main() {
	cgo := flag.Bool("cgo", false, "Sets whether cgo-using files are allowed to pass the filter.")
	tags := flag.String("tags", "", "Only pass through files that match these tags.")
	flag.Parse()

	bctx := build.Default
	bctx.BuildTags = strings.Split(*tags, ",")
	bctx.CgoEnabled = *cgo // Worth setting? build.MatchFile ignores this.

	outputs, err := filterFilenames(bctx, flag.Args())
	if err != nil {
		log.Fatalf("build_tags error: %v\n", err)
	}

	fmt.Println(strings.Join(outputs, " "))
}
