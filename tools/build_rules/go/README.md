# Go rules

<div class="toc">
  <h2>Rules</h2>
  <ul>
    <li><a href="#go_prefix">go_prefix</a></li>
    <li><a href="#go_library">go_library</a></li>
    <li><a href="#go_binary">go_binary</a></li>
    <li><a href="#go_test">go_test</a></li>
  </ul>
</div>

## Overview

The rules should be considered experimental. They support:

* libraries
* binaries
* tests
* vendoring

They currently do not support (in order of importance):

* `//+build` tags
* auto generated BUILD files.
* C/C++ interoperation (cgo, swig etc.)
* race detector
* coverage
* test sharding

## Setup

* Decide on the name of your package, eg. `github.com/joe/project`
* Add the following to your WORKSPACE file:

    ```python
    load("/tools/build_rules/go/def", "go_repositories")

    go_repositories()
    ```

* Add a `BUILD` file to the top of your workspace, declaring the name of your
  workspace using `go_prefix`. It is strongly recommended that the prefix is not
  empty.

    ```python
    load("/tools/build_rules/go/def", "go_prefix")

    go_prefix("github.com/joe/project")
    ```

* For a library `github.com/joe/project/lib`, create `lib/BUILD`, containing

    ```python
    load("/tools/build_rules/go/def", "go_library")

    go_library(
        name = "go_default_library",
        srcs = ["file.go"]
    )
    ```

* Inside your project, you can use this library by declaring a dependency

    ```python
    go_binary(
        ...
        deps = ["//lib:go_default_library"]
    )
    ```

* In this case, import the library as `github.com/joe/project/lib`.
* For vendored libraries, you may depend on
  `//lib/vendor/github.com/user/project:go_default_library`. Vendored
  libraries should have BUILD files like normal libraries.
* To declare a test,

    ```python
    go_test(
        name = "mytest",
        srcs = ["file_test.go"],
        library = ":go_default_library"
    )
    ```

## FAQ

### Can I still use the `go` tool?

Yes, this setup was deliberately chosen to be compatible with the `go`
tool. Make sure your workspace appears under

```sh
$GOROOT/src/github.com/joe/project/
```

eg.

```sh
mkdir -p $GOROOT/src/github.com/joe/
ln -s my/bazel/workspace $GOROOT/src/github.com/joe/project
```

and it should work.

## Disclaimer

These rules are not supported by Google's Go team.

<a name="go_prefix"></a>
## go\_prefix

```python
go_prefix(prefix)
```

<table class="table table-condensed table-bordered table-params">
  <colgroup>
    <col class="col-param" />
    <col class="param-description" />
  </colgroup>
  <thead>
    <tr>
      <th colspan="2">Attributes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>prefix</code></td>
      <td>
        <code>String, required</code>
        <p>Global prefix used to fully quality all Go targets.</p>
        <p>
          In Go, imports are always fully qualified with a URL, eg.
          <code>github.com/user/project</code>. Hence, a label <code>//foo:bar
          </code> from within a Bazel workspace must be referred to as
          <code>github.com/user/project/foo/bar</code>. To make this work, each
          rule must know the repository's URL. This is achieved, by having all
          go rules depend on a globally unique target that has a
          <code>go_prefix</code> transitive info provider.
        </p>
      </td>
    </tr>
  </tbody>
</table>

<a name="go_library"></a>
## go\_library

```python
go_library(name, srcs, deps, data)
```
<table class="table table-condensed table-bordered table-params">
  <colgroup>
    <col class="col-param" />
    <col class="param-description" />
  </colgroup>
  <thead>
    <tr>
      <th colspan="2">Attributes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>name</code></td>
      <td>
        <code>Name, required</code>
        <p>A unique name for this rule.</p>
      </td>
    </tr>
    <tr>
      <td><code>srcs</code></td>
      <td>
        <code>List of labels, required</code>
        <p>List of Go <code>.go</code> source files used to build the
        library</p>
      </td>
    </tr>
    <tr>
      <td><code>deps</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>List of other libraries to linked to this library target</p>
      </td>
    </tr>
    <tr>
      <td><code>data</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>List of files needed by this rule at runtime.</p>
      </td>
    </tr>
  </tbody>
</table>

<a name="go_binary"></a>
## go\_binary

```python
go_binary(name, srcs, deps, data)
```
<table class="table table-condensed table-bordered table-params">
  <colgroup>
    <col class="col-param" />
    <col class="param-description" />
  </colgroup>
  <thead>
    <tr>
      <th colspan="2">Attributes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>name</code></td>
      <td>
        <code>Name, required</code>
        <p>A unique name for this rule.</p>
      </td>
    </tr>
    <tr>
      <td><code>srcs</code></td>
      <td>
        <code>List of labels, required</code>
        <p>List of Go <code>.go</code> source files used to build the
        binary</p>
      </td>
    </tr>
    <tr>
      <td><code>deps</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>List of other libraries to linked to this binary target</p>
      </td>
    </tr>
    <tr>
      <td><code>data</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>List of files needed by this rule at runtime.</p>
      </td>
    </tr>
  </tbody>
</table>

<a name="go_test"></a>
## go\_test

```python
go_test(name, srcs, deps, data)
```
<table class="table table-condensed table-bordered table-params">
  <colgroup>
    <col class="col-param" />
    <col class="param-description" />
  </colgroup>
  <thead>
    <tr>
      <th colspan="2">Attributes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>name</code></td>
      <td>
        <code>Name, required</code>
        <p>A unique name for this rule.</p>
      </td>
    </tr>
    <tr>
      <td><code>srcs</code></td>
      <td>
        <code>List of labels, required</code>
        <p>List of Go <code>.go</code> source files used to build the
        test</p>
      </td>
    </tr>
    <tr>
      <td><code>deps</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>List of other libraries to linked to this test target</p>
      </td>
    </tr>
    <tr>
      <td><code>data</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>List of files needed by this rule at runtime.</p>
      </td>
    </tr>
  </tbody>
</table>
