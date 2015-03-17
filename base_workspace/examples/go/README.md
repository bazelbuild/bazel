
Go rules
--------

The files here demonstrate how to use the rules for Go supplied under
`//tools/build_rules:go_rules.bzl`.  The rules should be considered
experimental. They support:

  * libraries
  * binaries
  * tests

They currently do not support:

  * coverage
  * race detector
  * multiple target configurations
  * //+build tags
  * C/C++ interoperation (cgo, swig etc.)


Testing
-------

Setup a symlink to the Go installation you wish to use,

    ln -s /usr/lib/golang/ tools/go/go_root

or

    ln -s $(go env GOROOT) tools/go/go_root

To build something, run

    bazel build examples/go/...

To run a test, run

	bazel test --test_arg=--test.v examples/go/lib1:lib1_test


Writing BUILD rules
-------------------

In the bazel model of compiling Go, each directory can hold multiple
packages, rather than just one in the standard "go" tool. Suppose you
have

    dir/f.go:

    package p
    func F() {}

then in the BUILD file could say

    go_library(
      name = "q",
      srcs = ["f.go"])

and you import it with its Bazel name,

    import "dir/p/q"

this add the declared package name as namespace, i.e., it is
equivalent to

    import p "dir/p/q"

so you use it as follows

    import "dir/p/q"
    main() {
      p.F()
    }


Writing tests
-------------

For tests, you should create a separate target,

    go_test(
      name = "q_test",
      srcs = [ "f_test.go" ],
      deps = [ ":q" ])

if the test code is in the same package as the library under test
(e.g., because you are testing private parts of the library), you should
use the `library` attribute,

    go_test(
      name = "q_test",
      srcs = [ "f_test.go" ],
      library = ":q" )

This causes the test and the library under test to be compiled
together.


FAQ
---

# Why does this not follow the external Go conventions?

These rules were inspired on Google's internal Go rules, which work
like this. For Bazel, these make more sense, because directories in
Bazel do not correspond to single targets.


# Do I have to specify dependencies twice?

Yes, once in the BUILD file, once in the source file. Bazel does not
examine file contents, so it cannot infer the dependencies.  It is
possible to generate the BUILD file from the Go sources through a
separate program, though.


Disclaimer
----------

These rules are not supported by Google's Go team.
