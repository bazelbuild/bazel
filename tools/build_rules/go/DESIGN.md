Go support in Bazel
===================

Objective
---------

Provide Go support in Bazel that feels familiar to existing Go users and
existing Go tools.


Use cases
---------

Users might find benefit from using Bazel for Go in the following scenarios:

* Multi-language/multi-platform builds, where not all platforms are supported by `go` natively. For example: test iOS app running against a Go backend from a single tool, or Bazel itself, where the BUILD file formatter (`buildifier`) is written Go, and Bazel (Java/C++) interact.
* Large Go projects, where correctness and scalability of `go` become a problem.
* Go projects where the executable is not the final product.
* Projects with complex native code builds, eg. a Go server that uses first party native code through CGO.
* Projects with complex code generation steps, such as protocol buffers.

Constraints
-----------

The Go rules should not impose restrictions on how non-Go projects should organize their source trees.

Proposal
--------

* Bazel will support `go_library`, `go_binary` and `go_test` rules. These take go files as srcs, and `go_library` rules as `deps`. Along with the rules, we will have a tool that generates the main.go file based on the sources of a go_test rule.
* The go rules use a global setting `GO_PREFIX`. `GO_PREFIX` is set through Bazel's `WORKSPACE` mechanism, and may be empty. It is recommended that Go projects use the canonical import name (eg. `github.com/name/project`) as `GO_PREFIX` in Bazel.
* The go rules will support dependencies on go_library targets with other names, eg. `//a/b:c`. These are to be imported as `GO_PREFIX/a/b/c`. This convention is typically used for depending on generated Go code.
* The go rules will support a magic target name `go_default_library`. A dependency on `//a/b:go_default_library` will be staged by Bazel so it can be imported in a go file as `GO_PREFIX/a/b`, rather than `GO_PREFIX/a/b/go_default_library`.
* For making Bazel work with Go, we will have a tool called `glaze`, which analyzes the Go source files in a directory, and emits a BUILD file to match. Glaze must be run by hand (or, as an editor hook) when modifying Go code.
* When Glaze encounters an import line `GO_PREFIX/a/b` for which `a/b/` is a directory, it will write a dependency on `//a/b:go_default_library`.
* If a target has a dependency that contains a `vendor` directory component, the compiler will be invoked with a corresponding `-importmap` option, eg. a dependency on `x/y/vendor/domain/p/q:target` will yield `-importmap=domain/p/q/target=GO_PREFIX/x/y/vendor/domain/p/q/target`.
   * Multiple dependencies that map to the same importmap key is an analysis-time error.
* When Glaze encounters an import line that can be satisfied from a `vendor/` directory, as specified in the [Go vendoring decision](https://docs.google.com/document/d/1Bz5-UB7g2uPBdOx-rw5t9MxJwkfpx90cqG9AFL0JAYo/), it will emit the full target name of the vendored library.

Caveats
-------

Is not fully compatible with the `go` tool:
* `go` still cannot handle generated source code (ie. protocol buffers)
* the workspace will need to be in a directory called `src` for `go` to work with it (possibly through a symlink)
* Does not address compatibility with `go generate`:
   * Bazel puts sources and artifacts in different directories, so `go` tooling does not work out of the box if only parts of a go library are generated.
   * We could export the generated sources as some sort of tree with `//go:generate` lines in the source, but the command line will not run if the generator was built by Bazel too.
   * We could import generated sources by extracting `//go:generate` lines. Since the lines do not declare the sources of the invoked tooling, this will be hard to automatically get right, though.
* The `GO_PREFIX` that `WORKSPACE` sets is similar to the workspace name (see build encyclopedia), but it should only affect Go. Since Go prefixes are URLs, they contain dots, so using the `WORKSPACE` name would break Python imports.
* Does not specify how native interoperability (eg. cgo) should work.

Implementation plan
-------------------

* Implement `go_prefix` support in `WORKSPACE`
* Reintroduce the Skylark Go rules, but supporting `go_default_library` and `go_prefix`.
* Ship them in Bazel; this yields bare bones rules.
* Open-source the `BUILD` file formatter, and use it as basis to create `glaze`.
* Open-source compiler glue program and use it in the Skylark rules. This will yields conformance with //+ build tags, and other build system directives embedded in source code.
* Consider cleaning and opening up internal rules.
