// Bare bones Go testing support for Bazel.

package main

import (
	"flag"
	"go/ast"
	"go/parser"
	"go/token"
	"log"
	"os"
	"strings"
	"text/template"
)

// Cases holds template data.
type Cases struct {
	Package string
	RunDir  string
	Names   []string
}

func main() {
	pkg := flag.String("package", "", "package from which to import test methods.")
	out := flag.String("output", "", "output file to write. Defaults to stdout.")
	flag.Parse()

	if *pkg == "" {
		log.Fatal("must set --package.")
	}

	outFile := os.Stdout
	if *out != "" {
		var err error
		outFile, err = os.Create(*out)
		if err != nil {
			log.Fatalf("os.Create(%q): %v", *out, err)
		}
		defer outFile.Close()
	}

	cases := Cases{
		Package: *pkg,
		RunDir:  os.Getenv("RUNDIR"),
	}
	testFileSet := token.NewFileSet()
	for _, f := range flag.Args() {
		parse, err := parser.ParseFile(testFileSet, f, nil, parser.ParseComments)
		if err != nil {
			log.Fatalf("ParseFile(%q): %v", f, err)
		}

		for _, d := range parse.Decls {
			fn, ok := d.(*ast.FuncDecl)
			if !ok {
				continue
			}
			if fn.Recv != nil {
				continue
			}
			if !strings.HasPrefix(fn.Name.Name, "Test") {
				continue
			}
			cases.Names = append(cases.Names, fn.Name.Name)
		}
	}

	tpl := template.Must(template.New("source").Parse(`
package main
import (
	"os"
	"testing"

        undertest "{{.Package}}"
)

func everything(pat, str string) (bool, error) {
	return true, nil
}

var tests = []testing.InternalTest{
{{range .Names}}
   {"{{.}}", undertest.{{.}} },
{{end}}
}

func main() {
  os.Chdir("{{.RunDir}}")
  testing.Main(everything, tests, nil, nil)
}
`))
	if err := tpl.Execute(outFile, &cases); err != nil {
		log.Fatalf("template.Execute(%v): %v", cases, err)
	}
}
