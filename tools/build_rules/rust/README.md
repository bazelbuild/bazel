# Rust Rules

<div class="toc">
  <h2>Rules</h2>
  <ul>
    <li><a href="#rust_library">rust_library</a></li>
    <li><a href="#rust_binary">rust_binary</a></li>
    <li><a href="#rust_test">rust_test</a></li>
    <li><a href="#rust_bench_test">rust_bench_test</a></li>
    <li><a href="#rust_doc">rust_doc</a></li>
    <li><a href="#rust_doc_test">rust_doc_test</a></li>
  </ul>
</div>

## Overview

These build rules are used for building [Rust][rust] projects with Bazel.

[rust]: http://www.rust-lang.org/

<a name="setup"></a>
## Setup

To use the Rust rules, add the following to your `WORKSPACE` file to add the
external repositories for the Rust toolchain:

```python
load("@bazel_tools//tools/build_rules/rust:rust.bzl", "rust_repositories")

rust_repositories()
```

<a name="roadmap"></a>
## Roadmap

* Add `rust_toolchain` rule to make it easy to use a custom Rust toolchain.
* Add tool for taking `Cargo.toml` and generating a `WORKSPACE` file with
  workspace rules for pulling external dependencies.
* Improve expressiveness of features and support for [Cargo's feature
  groups](http://doc.crates.io/manifest.html#the-[features]-section).
* Add `cargo_crate` workspace rule for pulling crates from
  [Cargo](https://crates.io/).

<a name="rust_library"></a>
## rust_library

```python
rust_library(name, srcs, deps, data, crate_features, rustc_flags)
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
          This name will also be used as the name of the library crate built by
          this rule.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>srcs</code></td>
      <td>
        <code>List of labels, required</code>
        <p>List of Rust <code>.rs</code> source files used to build the
        library.</p>
        <p>
          If <code>srcs</code> contains more than one file, then there must be
          a file either named <code>lib.rs</code>. Otherwise,
          <code>crate_root</code> must be set to the source file that is the
          root of the crate to be passed to <code>rustc</code> to build this
          crate.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>crate_root</code></td>
      <td>
        <code>Label, optional</code>
        <p>
          The file that will be passed to <code>rustc</code> to be used for
          building this crate.
        </p>
        <p>
          If <code>crate_root</code> is not set, then this rule will look for
          a <code>lib.rs</code> file or the single file in <code>srcs</code>
          if <code>srcs</code> contains only one file.
        </p>
      </td>
    </td>
    <tr>
      <td><code>deps</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>List of other libraries to be linked to this library target.</p>
        <p>
          These can be either other <code>rust_library</code> targets or
          <code>cc_library</code> targets if linking a native library.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>data</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>List of files used by this rule at runtime.</p>
        <p>
          This attribute can be used to specify any data files that are embedded
          into the library, such as via the
          <a href="https://doc.rust-lang.org/std/macro.include_str!.html target="_blank"><code>include_str!</code></a>
          macro.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>crate_features</code></td>
      <td>
        <code>List of strings, optional</code>
        <p>List of features to enable for this crate.</p>
        <p>
          Features are defined in the code using the
          <code>#[cfg(feature = "foo")]</code> configuration option. The
          features listed here will be passed to <code>rustc</code> with
          <code>--cfg feature="${feature_name}"</code> flags.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>rustc_flags</code></td>
      <td>
        <code>List of strings, optional</code>
        <p>List of compiler flags passed to <code>rustc</code>.</p>
      </td>
    </tr>
  </tbody>
</table>

### Example

Suppose you have the following directory structure for a simple Rust library
crate:

```
[workspace]/
    WORKSPACE
    hello_lib/
        BUILD
        src/
            greeter.rs
            lib.rs
```

`hello_lib/src/greeter.rs`:

```rust
pub struct Greeter {
    greeting: String,
}

impl Greeter {
    pub fn new(greeting: &str) -> Greeter {
        Greeter { greeting: greeting.to_string(), }
    }

    pub fn greet(&self, thing: &str) {
        println!("{} {}", &self.greeting, thing);
    }
}
```

`hello_lib/src/lib.rs`:


```rust
pub mod greeter;
```

`hello_lib/BUILD`:

```python
package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_rules/rust:rust.bzl", "rust_library")

rust_library(
    name = "hello_lib",
    srcs = [
        "src/greeter.rs",
        "src/lib.rs",
    ],
)
```

Build the library:

```
$ bazel build //hello_lib
INFO: Found 1 target...
Target //examples/rust/hello_lib:hello_lib up-to-date:
  bazel-bin/examples/rust/hello_lib/libhello_lib.rlib
INFO: Elapsed time: 1.245s, Critical Path: 1.01s
```

<a name="rust_binary"></a>
## rust_binary

```
rust_binary(name, srcs, deps, data, crate_features, rustc_flags)
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
          This name will also be used as the name of the binary crate built by
          this rule.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>srcs</code></td>
      <td>
        <code>List of labels, required</code>
        <p>List of Rust <code>.rs</code> source files used to build the
        binary.</p>
        <p>
          If <code>srcs</code> contains more than one file, then there must be
          a file either named <code>main.rs</code>. Otherwise,
          <code>crate_root</code> must be set to the source file that is the
          root of the crate to be passed to <code>rustc</code> to build this
          crate.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>crate_root</code></td>
      <td>
        <code>Label, optional</code>
        <p>
          The file that will be passed to <code>rustc</code> to be used for
          building this crate.
        </p>
        <p>
          If <code>crate_root</code> is not set, then this rule will look for
          a <code>main.rs</code> file or the single file in <code>srcs</code>
          if <code>srcs</code> contains only one file.
        </p>
      </td>
    </td>
    <tr>
      <td><code>deps</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>List of other libraries to be linked to this library target.</p>
        <p>
          These must be <code>rust_library</code> targets.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>data</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>List of files used by this rule at runtime.</p>
        <p>
          This attribute can be used to specify any data files that are embedded
          into the library, such as via the
          <a href="https://doc.rust-lang.org/std/macro.include_str!.html target="_blank"><code>include_str!</code></a>
          macro.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>crate_features</code></td>
      <td>
        <code>List of strings, optional</code>
        <p>List of features to enable for this crate.</p>
        <p>
          Features are defined in the code using the
          <code>#[cfg(feature = "foo")]</code> configuration option. The
          features listed here will be passed to <code>rustc</code> with
          <code>--cfg feature="${feature_name}"</code> flags.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>rustc_flags</code></td>
      <td>
        <code>List of strings, optional</code>
        <p>List of compiler flags passed to <code>rustc</code>.</p>
      </td>
    </tr>
  </tbody>
</table>

### Example

Suppose you have the following directory structure for a Rust project with a
library crate, `hello_lib`, and a binary crate, `hello_world` that uses the
`hello_lib` library:

```
[workspace]/
    WORKSPACE
    hello_lib/
        BUILD
        src/
            lib.rs
    hello_world/
        BUILD
        src/
            main.rs
```

`hello_lib/src/lib.rs`:

```rust
pub struct Greeter {
    greeting: String,
}

impl Greeter {
    pub fn new(greeting: &str) -> Greeter {
        Greeter { greeting: greeting.to_string(), }
    }

    pub fn greet(&self, thing: &str) {
        println!("{} {}", &self.greeting, thing);
    }
}
```

`hello_lib/BUILD`:

```python
package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_rules/rust:rust.bzl", "rust_library")

rust_library(
    name = "hello_lib",
    srcs = ["src/lib.rs"],
)
```

`hello_world/src/main.rs`:

```rust
extern crate hello_lib;

fn main() {
    let hello = hello_lib::Greeter::new("Hello");
    hello.greet("world");
}
```

`hello_world/BUILD`:

```python
load("@bazel_tools//tools/build_rules/rust:rust.bzl", "rust_binary")

rust_binary(
    name = "hello_world",
    srcs = ["src/main.rs"],
    deps = ["//hello_lib"],
)
```

Build and run `hello_world`:

```
$ bazel run //hello_world
INFO: Found 1 target...
Target //examples/rust/hello_world:hello_world up-to-date:
  bazel-bin/examples/rust/hello_world/hello_world
INFO: Elapsed time: 1.308s, Critical Path: 1.22s

INFO: Running command line: bazel-bin/examples/rust/hello_world/hello_world
Hello world
```

<a name="rust_test"></a>
## rust_test

```python
rust_test(name, srcs, deps, data, crate_features, rustc_flags)
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
          This name will also be used as the name of the binary test crate
          built by this rule.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>srcs</code></td>
      <td>
        <code>List of labels, required</code>
        <p>List of Rust <code>.rs</code> source files used to build the
        library.</p>
        <p>
          If <code>srcs</code> contains more than one file, then there must be
          a file either named <code>lib.rs</code>. Otherwise,
          <code>crate_root</code> must be set to the source file that is the
          root of the crate to be passed to <code>rustc</code> to build this
          crate.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>crate_root</code></td>
      <td>
        <code>Label, optional</code>
        <p>
          The file that will be passed to <code>rustc</code> to be used for
          building this crate.
        </p>
        <p>
          If <code>crate_root</code> is not set, then this rule will look for
          a <code>lib.rs</code> file or the single file in <code>srcs</code>
          if <code>srcs</code> contains only one file.
        </p>
      </td>
    </td>
    <tr>
      <td><code>deps</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>List of other libraries to be linked to this test target.</p>
        <p>
          These must be <code>rust_library</code> targets.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>data</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>List of files used by this rule at runtime.</p>
        <p>
          This attribute can be used to specify any data files that are embedded
          into the library, such as via the
          <a href="https://doc.rust-lang.org/std/macro.include_str!.html target="_blank"><code>include_str!</code></a>
          macro.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>crate_features</code></td>
      <td>
        <code>List of strings, optional</code>
        <p>List of features to enable for this crate.</p>
        <p>
          Features are defined in the code using the
          <code>#[cfg(feature = "foo")]</code> configuration option. The
          features listed here will be passed to <code>rustc</code> with
          <code>--cfg feature="${feature_name}"</code> flags.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>rustc_flags</code></td>
      <td>
        <code>List of strings, optional</code>
        <p>List of compiler flags passed to <code>rustc</code>.</p>
      </td>
    </tr>
  </tbody>
</table>

### Example

Suppose you have the following directory structure for a Rust library crate
with unit test code in the library sources:

```
[workspace]/
    WORKSPACE
    hello_lib/
        BUILD
        src/
            lib.rs
```

`hello_lib/src/lib.rs`:

```rust
pub struct Greeter {
    greeting: String,
}

impl Greeter {
    pub fn new(greeting: &str) -> Greeter {
        Greeter { greeting: greeting.to_string(), }
    }

    pub fn greet(&self, thing: &str) {
        println!("{} {}", &self.greeting, thing);
    }
}

#[cfg(test)]
mod test {
    use super::Greeter;

    #[test]
    fn test_greeting() {
        let hello = Greeter::new("Hi");
        assert_eq!("Hi Rust", hello.greeting("Rust"));
    }
}
```

To build and run the tests, simply add a `rust_test` rule with no `srcs` and
only depends on the `hello_lib` `rust_library` target:

`hello_lib/BUILD`:

```python
package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_rules/rust:rust.bzl", "rust_library", "rust_test")

rust_library(
    name = "hello_lib",
    srcs = ["src/lib.rs"],
)

rust_test(
    name = "hello_lib_test",
    deps = [":hello_lib"],
)
```

Run the test with `bazel build //hello_lib:hello_lib_test`.

### Example: `test` directory

Integration tests that live in the [`tests` directory][int-tests], they are
essentially built as separate crates. Suppose you have the following directory
structure where `greeting.rs` is an integration test for the `hello_lib`
library crate:

[int-tests]: http://doc.rust-lang.org/book/testing.html#the-tests-directory

```
[workspace]/
    WORKSPACE
    hello_lib/
        BUILD
        src/
            lib.rs
        tests/
            greeting.rs
```

`hello_lib/tests/greeting.rs`:

```rust
extern crate hello_lib;

use hello_lib;

#[test]
fn test_greeting() {
    let hello = greeter::Greeter::new("Hello");
    assert_eq!("Hello world", hello.greeting("world"));
}
```

To build the `greeting.rs` integration test, simply add a `rust_test` target
with `greeting.rs` in `srcs` and a dependency on the `hello_lib` target:

`hello_lib/BUILD`:

```python
package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_rules/rust:rust.bzl", "rust_library", "rust_test")

rust_library(
    name = "hello_lib",
    srcs = ["src/lib.rs"],
)

rust_test(
    name = "greeting_test",
    srcs = ["tests/greeting.rs"],
    deps = [":hello_lib"],
)
```

Run the test with `bazel build //hello_lib:hello_lib_test`.

<a name="rust_bench_test"></a>
## rust\_bench\_test

```python
rust_bench_test(name, srcs, deps, data, crate_features, rustc_flags)
```

**Warning**: This rule is currently experimental. [Rust Benchmark
tests][rust-bench] require the `Bencher` interface in the unstable `libtest`
crate, which is behind the `test` unstable feature gate. As a result, using
this rule would require using a nightly binary release of Rust. A
`rust_toolchain` rule will be added in the [near future](#roadmap) to make it
easy to use a custom Rust toolchain, such as a nightly release.

[rust-bench]: https://doc.rust-lang.org/book/benchmark-tests.html

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
          This name will also be used as the name of the binary test crate
          built by this rule.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>srcs</code></td>
      <td>
        <code>List of labels, required</code>
        <p>List of Rust <code>.rs</code> source files used to build the
        library.</p>
        <p>
          If <code>srcs</code> contains more than one file, then there must be
          a file either named <code>lib.rs</code>. Otherwise,
          <code>crate_root</code> must be set to the source file that is the
          root of the crate to be passed to <code>rustc</code> to build this
          crate.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>crate_root</code></td>
      <td>
        <code>Label, optional</code>
        <p>
          The file that will be passed to <code>rustc</code> to be used for
          building this crate.
        </p>
        <p>
          If <code>crate_root</code> is not set, then this rule will look for
          a <code>lib.rs</code> file or the single file in <code>srcs</code>
          if <code>srcs</code> contains only one file.
        </p>
      </td>
    </td>
    <tr>
      <td><code>deps</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>List of other libraries to be linked to this test target.</p>
        <p>
          These must be <code>rust_library</code> targets.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>data</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>List of files used by this rule at runtime.</p>
        <p>
          This attribute can be used to specify any data files that are embedded
          into the library, such as via the
          <a href="https://doc.rust-lang.org/std/macro.include_str!.html target="_blank"><code>include_str!</code></a>
          macro.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>crate_features</code></td>
      <td>
        <code>List of strings, optional</code>
        <p>List of features to enable for this crate.</p>
        <p>
          Features are defined in the code using the
          <code>#[cfg(feature = "foo")]</code> configuration option. The
          features listed here will be passed to <code>rustc</code> with
          <code>--cfg feature="${feature_name}"</code> flags.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>rustc_flags</code></td>
      <td>
        <code>List of strings, optional</code>
        <p>List of compiler flags passed to <code>rustc</code>.</p>
      </td>
    </tr>
  </tbody>
</table>

### Example

Suppose you have the following directory structure for a Rust project with a
library crate, `fibonacci` with benchmarks under the `benches/` directory:

```
[workspace]/
    WORKSPACE
    fibonacci/
        BUILD
        src/
            lib.rs
        benches/
            fibonacci_bench.rs
```

`fibonacci/src/lib.rs`:

```rust
pub fn fibonacci(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    let mut n1: u64 = 0;
    let mut n2: u64 = 1;
    for _ in 1..n {
        let sum = n1 + n2;
        n1 = n2;
        n2 = sum;
    }
    n2
}
```

`fibonacci/benches/fibonacci_bench.rs`:

```rust
#![feature(test)]

extern crate test;
extern crate fibonacci;

use test::Bencher;

#[bench]
fn bench_fibonacci(b: &mut Bencher) {
    b.iter(|| fibonacci::fibonacci(40));
}
```

To build the benchmark test, simply add a `rust_bench_test` target:

`fibonacci/BUILD`:

```python
package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_rules/rust:rust.bzl", "rust_library", "rust_bench_test")

rust_library(
    name = "fibonacci",
    srcs = ["src/lib.rs"],
)

rust_bench_test(
    name = "fibonacci_bench",
    srcs = ["benches/fibonacci_bench.rs"],
    deps = [":fibonacci"],
)
```

Run the benchmark test using: `bazel build //fibonacci:fibonacci_bench`.

<a name="rust_doc"></a>
## rust_doc

```python
rust_doc(name, dep, markdown_css, html_in_header, html_before_content, html_after_content)
```

<table class="table table-condensed table-bordered table-params">
  <colgroup>
    <col class="col-param" />
    <col class="param-description" />
  </colgroup>
  <thead>
    <tr>
      <th>Attribute</th>
      <th>Description</th>
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
          <code>rust_doc</code> can generate HTML code documentation for the
          source files of <code>rust_library</code> or <code>rust_binary</code>
          targets.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>markdown_css</code></td>
      <td>
        <code>List of Labels, optional</code>
        <p>
          CSS files to include via <code>&lt;link&gt;</code> in a rendered
          Markdown file.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>html_in_header</code></td>
      <td>
        <code>Label, optional</code>
        <p>File to add to <code>&lt;head&gt;</code>.</p>
      </td>
    </tr>
    <tr>
      <td><code>html_before_content</code></td>
      <td>
        <code>Label, optional</code>
        <p>File to add in <code>&lt;body&gt;</code>, before content.</p>
      </td>
    </tr>
    <tr>
      <td><code>html_after_content</code></td>
      <td>
        <code>Label, optional</code>
        <p>File to add in <code>&lt;body&gt;</code>, after content.</p>
      </td>
    </tr>
  </tbody>
</table>

### Example

Suppose you have the following directory structure for a Rust library crate:

```
[workspace]/
    WORKSPACE
    hello_lib/
        BUILD
        src/
            lib.rs
```

To build [`rustdoc`][rustdoc] documentation for the `hello_lib` crate, define
a `rust_doc` rule that depends on the the `hello_lib` `rust_library` target:

[rustdoc]: https://doc.rust-lang.org/book/documentation.html

```python
package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_rules/rust:rust.bzl", "rust_library", "rust_doc")

rust_library(
    name = "hello_lib",
    srcs = ["src/lib.rs"],
)

rust_doc(
    name = "hello_lib_doc",
    dep = ":hello_lib",
)
```

Running `bazel build //hello_lib:hello_lib_doc` will build a zip file containing
the documentation for the `hello_lib` library crate generated by `rustdoc`.

<a name="rust_doc_test"></a>
## rust\_doc\_test

```python
rust_doc_test(name, dep)
```

<table class="table table-condensed table-bordered table-params">
  <colgroup>
    <col class="col-param" />
    <col class="param-description" />
  </colgroup>
  <thead>
    <tr>
      <th>Attribute</th>
      <th>Description</th>
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
        <p>The label of the target to run documentation tests for.</p>
        <p>
          <code>rust_doc_test</code> can run documentation tests for the
          source files of <code>rust_library</code> or <code>rust_binary</code>
          targets.
        </p>
      </td>
    </tr>
  </tbody>
</table>

### Example

Suppose you have the following directory structure for a Rust library crate:

```
[workspace]/
    WORKSPACE
    hello_lib/
        BUILD
        src/
            lib.rs
```

To run [documentation tests][doc-test] for the `hello_lib` crate, define a
`rust_doc_test` target that depends on the `hello_lib` `rust_library` target:

[doc-test]: https://doc.rust-lang.org/book/documentation.html#documentation-as-tests

```python
package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_rules/rust:rust.bzl", "rust_library", "rust_doc_test")

rust_library(
    name = "hello_lib",
    srcs = ["src/lib.rs"],
)

rust_doc_test(
    name = "hello_lib_doc_test",
    dep = ":hello_lib",
)
```

Running `bazel test //hello_lib:hello_lib_doc_test` will run all documentation
tests for the `hello_lib` library crate.
