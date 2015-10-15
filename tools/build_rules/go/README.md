Go rules
--------

The rules should be considered experimental. They support:

* libraries
* binaries
* tests
* vendoring

They currently do not support (in order of importance):

* Darwin
* //+build tags
* auto generated BUILD files.
* C/C++ interoperation (cgo, swig etc.)
* race detector
* coverage
* test sharding

Setup
-----

* Decide on the name of your package, eg. "github.com/joe/project"

* Copy tools/build_rules/go/toolchain/WORKSPACE.go-toolchain to WORKSPACE

* Add a BUILD file to the top of your workspace, containing

    load("/tools/build_rules/go/def", "go_prefix")
    go_prefix("github.com/joe/project")

* For a library "github.com/joe/project/lib", create lib/BUILD, containing

    load("/tools/build_rules/go/def", "go_library")
    go_library(
      name = "go_default_library",
      srcs = ["file.go"])

* Inside your project, you can use this library by declaring a dependency

    go_binary( ...
      deps = ["//lib:go_default_library"])

* In this case, import the library as "github.com/joe/project/lib".

* For vendored libraries, you may depend on
  "//lib/vendor/github_com/user/project:go_default_library". Vendored
  libraries should have BUILD files like normal libraries.

* To declare a test,

    go_test(
      name = "mytest",
      srcs = ["file_test.go"],
      library = ":go_default_library")

FAQ
---

# Can I still use the =go= tool?

Yes, this setup was deliberately chosen to be compatible with the =go=
tool. Make sure your workspace appears under

    $GOROOT/src/github.com/joe/project/

eg.

    mkdir -p $GOROOT/src/github.com/joe/
    ln -s my/bazel/workspace $GOROOT/src/github.com/joe/project

and it should work.

Disclaimer
----------

These rules are not supported by Google's Go team.
