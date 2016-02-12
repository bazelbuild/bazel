# D rules

## Rules

<div class="toc">
  <h2>Rules</h2>
  <ul>
    <li><a href="#d_library">d_library</a></li>
    <li><a href="#d_source_library">d_source_library</a></li>
    <li><a href="#d_binary">d_binary</a></li>
    <li><a href="#d_test">d_test</a></li>
    <li><a href="#d_docs">d_docs</a></li>
  </ul>
</div>

## Setup

To use the D rules, add the following to your `WORKSPACE` file to add the
external repositories for the D toolchain:

```python
load("@bazel_tools//tools/build_defs/d:d.bzl", "d_repositories")

d_repositories()
```

## Roadmap

* Generate documentation using [`ddox`](https://github.com/rejectedsoftware/ddox)
  for `d_docs` rule.
* Support for other options as defined in the [Dub package
  format](http://code.dlang.org/package-format?lang=json)
* Support for specifying different configurations of a library, closer to
  [Dub's model for configurations](http://code.dlang.org/package-format?lang=json#configurations)
* Workspace rule for retrieving dependencies from [Dub](http://code.dlang.org/)

<a name="d_library"></a>
## d_library

```python
d_library(name, srcs, deps, includes, linkopts, versions)
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
        <p>
          This name will be used as the name of the library built by this rule.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>srcs</code></td>
      <td>
        <code>List of labels, required</code>
        <p>List of D <code>.d</code> source files used to build the library.</p>
      </td>
    </tr>
    <tr>
      <td><code>deps</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>List of libraries to be linked to this library target.</p>
        <p>
          These can either be other <code>d_library</code> targets,
          source-only <code>d_source_library</code> targets, or
          <code>cc_library</code> targets if linking a native library.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>imports</code></td>
      <td>
        <code>List of strings, optional</code>
        <p>List of import dirs to add to the compile line.</p>
        <p>
          These will be passed to the D compiler via <code>-I</code> flags.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>linkopts</code></td>
      <td>
        <code>List of strings, optional</code>
        <p>List of flags that are added to the D linker command.</p>
        <p>
          These will be passed to the D compiler via <code>-L</code> flags.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>versions</code></td>
      <td>
        <code>List of strings, optional</code>
        <p>List of versions to be defined during compilation.</p>
        <p>
          Versions are used for conditional compilation and are enabled in the
          code using <code>version</code> condition blocks. These versions
          listed here will be passed to the D compiler using
          <code>-version</code> flags.
        </p>
      </td>
    </tr>
  </tbody>
</table>

### Example

Suppose you have the following directory structure for a D project:

```
[workspace]/
    WORKSPACE
    foo/
        BUILD
        foo.d
        bar.d
        baz.d
```

The library `foo` is built using a `d_library` target:

`foo/BUILD`:

```python
load("@bazel_tools//tools/build_defs/d/d", "d_library")

d_binary(
    name = "foo",
    srcs = [
        "foo.d",
        "bar.d",
        "baz.d",
    ],
)
```

<a name="d_source_library"></a>
## d_source_library

```python
d_source_library(name, srcs, deps, includes, linkopts, versions)
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
        <p>
          List of D <code>.d</code> source files that comprises this source
          library target.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>deps</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>List of library targets depended on by this target.</p>
        <p>
          These can either be other <code>d_source_library</code> targets or
          <code>cc_library</code> targets, such as when this source library
          target implements the D interface for a native library. Any native
          libraries will be linked by <code>d_library</code> targets that
          depend on this target.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>imports</code></td>
      <td>
        <code>List of strings, optional</code>
        <p>List of import dirs to add to the compile line.</p>
        <p>
          These will be passed to the D compiler via <code>-I</code> flags for
          any <code>d_library</code> targets that depend on this target.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>linkopts</code></td>
      <td>
        <code>List of strings, optional</code>
        <p>List of flags that are added to the D linker command.</p>
        <p>
          These will be passed to the D compiler via <code>-L</code> flags for
          any <code>d_library</code> targets that depend on this target.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>versions</code></td>
      <td>
        <code>List of strings, optional</code>
        <p>List of version flags to be defined during compilation.</p>
        <p>
          Versions are used for conditional compilation and are enabled in the
          code using <code>version</code> condition blocks. These versions
          listed here will be passed to the D compiler using
          <code>-version</code> flags for any <code>d_library</code> targets
          that depend on this target.
        </p>
      </td>
    </tr>
  </tbody>
</table>

### Example

Suppose you have the following directory structure for a project building a
C library and a [D interface](http://dlang.org/interfaceToC.html) for the C
library:

```
[workspace]/
    WORKSPACE
    greeter/
        BUILD
        native_greeter.c
        native_greeter.h
        native_greeter.d
    hello_world
        BUILD
        hello_world.d
```

Build the C library using the `cc_library` rule and then use the
`d_source_library` to define the target for the D interface for the C
`native_greeter` library:

`greeter/BUILD`:

```python
load("@bazel_tools//tools/build_defs/d/d", "d_source_library")

cc_library(
    name = "native_greeter_lib",
    srcs = ["native_greeter.c"],
    hdrs = ["native_greeter.h"],
)

d_source_library(
    name = "native_greeter",
    srcs = ["native_greeter.d"],
    deps = [":native_greeter_lib"],
)
```

Other targets can directly depend on the `d_source_library` target to link
the C library:

`hello_world/BUILD`:

```python
load("@bazel_tools//tools/build_defs/d/d", "d_source_library")

d_binary(
    name = "hello_world",
    srcs = ["hello_world.d"],
    deps = ["//greeter:native_greeter"],
)
```

<a name="d_binary"></a>
## d_binary

```python
d_binary(name, srcs, deps, includes, linkopts, versions)
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
        <p>
          This name will be used as the name of the binary built by this rule.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>srcs</code></td>
      <td>
        <code>List of labels, required</code>
        <p>List of D <code>.d</code> source files used to build the binary.</p>
      </td>
    </tr>
    <tr>
      <td><code>deps</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>List of libraries to be linked to this binary target.</p>
        <p>
          These can either be other <code>d_library</code> targets,
          source-only <code>d_source_library</code> targets, or
          <code>cc_library</code> targets if linking a native library.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>imports</code></td>
      <td>
        <code>List of strings, optional</code>
        <p>List of import dirs to add to the compile line.</p>
        <p>
          These will be passed to the D compiler via <code>-I</code> flags.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>linkopts</code></td>
      <td>
        <code>List of strings, optional</code>
        <p>List of flags that are added to the D linker command.</p>
        <p>
          These will be passed to the D compiler via <code>-L</code> flags.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>versions</code></td>
      <td>
        <code>List of strings, optional</code>
        <p>List of versions to be defined during compilation.</p>
        <p>
          Versions are used for conditional compilation and are enabled in the
          code using <code>version</code> condition blocks. These versions
          listed here will be passed to the D compiler using
          <code>-version</code> flags.
        </p>
      </td>
    </tr>
  </tbody>
</table>

Suppose you have the following directory structure for a D project:

```
[workspace]/
    WORKSPACE
    hello_lib/
        BUILD
        greeter.d
    hello_world
        BUILD
        hello_world.d
```

The source file `hello_lib/greeter.d` defines a module `greeter`:

```d
module greeter;
...
```

The `hello_lib` library is built using a `d_library` target:

`hello_lib/BUILD`:

```python
load("@bazel_tools//tools/build_defs/d/d", "d_library")

d_library(
    name = "hello_lib",
    srcs = ["greeter.d"],
)
```

By default, import paths are from the root of the workspace. Thus, the source
for the `hello_world` binary, `hello_world.d`, would import the `greeter`
module as follows:

```d
import hello_lib.greeter;
```

However, this can be changed via the `imports` attribute on the `d_library`
rule.

The `hello_world` binary is built using a `d_binary` target:

`hello_world/BUILD`:

```python
load("@bazel_tools//tools/build_defs/d/d", "d_library")

d_binary(
    name = "hello_world",
    srcs = ["hello_world.d"],
    deps = ["//hello_lib"],
)
```

<a name="d_test"></a>
## d_test

```python
d_test(name, srcs, deps, includes, linkopts, versions)
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
        <p>
          This name will be used as the name of the test built by this rule.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>srcs</code></td>
      <td>
        <code>List of labels, required</code>
        <p>List of D <code>.d</code> source files used to build the test.</p>
      </td>
    </tr>
    <tr>
      <td><code>deps</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>List of libraries to be linked to this test target.</p>
        <p>
          These can either be other <code>d_library</code> targets,
          source-only <code>d_source_library</code> targets, or
          <code>cc_library</code> targets if linking a native library.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>imports</code></td>
      <td>
        <code>List of strings, optional</code>
        <p>List of import dirs to add to the compile line.</p>
        <p>
          These will be passed to the D compiler via <code>-I</code> flags.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>linkopts</code></td>
      <td>
        <code>List of strings, optional</code>
        <p>List of flags that are added to the D linker command.</p>
        <p>
          These will be passed to the D compiler via <code>-L</code> flags.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>versions</code></td>
      <td>
        <code>List of strings, optional</code>
        <p>List of versions to be defined during compilation.</p>
        <p>
          Versions are used for conditional compilation and are enabled in the
          code using <code>version</code> condition blocks. These versions
          listed here will be passed to the D compiler using
          <code>-version</code> flags.
        </p>
      </td>
    </tr>
  </tbody>
</table>

### Example

Suppose you have the following directory structure for a D project:

```
[workspace]/
    WORKSPACE
    hello_lib/
        BUILD
        greeter.d
        greeter_test.d
```

`hello_lib/greeter.d`:

```d
module greeter;

import std.stdio;
import std.string;

class Greeter {
 private string greeting;

 public:
  this(in string greeting) {
    this.greeting = greeting.dup;
  }

  string makeGreeting(in immutable string thing) {
    return format("%s %s!", this.greeting, thing);
  }

  void greet(in immutable string thing) {
    writeln(makeGreeting(thing));
  }
}
```

`hello_lib/greeter_test.d`:

```d
import hello_lib.greeter;

unittest {
  auto greeter = new Greeter("Hello");
  assert(greeter.makeGreeting("world") == "Hello world!");
}

void main() {}
```

To build the library and unit test:

`hello_lib/BUILD`:

```python
load("@bazel_tools//tools/build_defs/d/d", "d_library", "d_test")

d_library(
    name = "greeter",
    srcs = ["greeter.d"],
)

d_test(
    name = "greeter_test",
    srcs = ["greeter_test.d"],
    deps = [":greeter"],
)
```

The unit test can then be run using:

```sh
bazel test //hello_lib:greeter_test
```

<a name="d_docs"></a>
## d_docs

```python
d_docs(name, dep)
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
      <td><code>dep</code></td>
      <td>
        <code>Label, required</code>
        <p>The label of the target to generate code documentation for.</p>
        <p>
          <code>d_docs</code> can generate HTML code documentation for the
          source files of <code>d_library</code>, <code>d_source_library</code>,
          <code>d_binary</code>, or <code>d_test</code> targets.
        </p>
      </td>
    </tr>
  </tbody>
</table>

### Example

Suppose you have the following directory structure for a D project:

```
[workspace]/
    WORKSPACE
    foo/
        BUILD
        foo.d
        bar.d
        baz.d
```

The `foo/` directory contains the sources for the `d_library` `foo`. To
generate HTML documentation for the `foo` library, define a `d_docs` target
that takes the `d_library` `foo` as its dependency:

`foo/BUILD`:

```python
load("@bazel_tools//tools/build_defs/d/d", "d_library", "d_docs")

d_library(
    name = "foo",
    srcs = [
        "foo.d",
        "bar.d",
        "baz.d",
    ],
)

d_docs(
    name = "foo_docs",
    dep = ":foo",
)
```

Running `bazel build //foo:foo_docs` will generate a zip file containing the
HTML documentation generated from the source files. See the official D language
documentation on the [Documentation Generator](http://dlang.org/ddoc.html) for
more information on the conventions for source documentation.
