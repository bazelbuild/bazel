# Rust Rules for Bazel

## Overview

These build rules are used for building [Rust][rust] projects with Bazel.

* [Setup](#setup)
* [Basic Example](#basic-example)
* [Build Rule Reference](#reference)
  * [`rust_library`](#reference-rust_library)
  * [`rust_binary`](#reference-rust_binary)
  * [`rust_test`](#reference-rust_test)
* [Roadmap](#roadmap)

[rust]: http://www.rust-lang.org/

<a name="setup"></a>
## Setup

To use the Rust rules, simply copy the contents of `rust.WORKSPACE` to your
`WORKSPACE` file.

<a name="basic-example"></a>
## Basic Example

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

load("/tools/build_rules/rust/rust", "rust_library")

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

Now, let's add a binary crate that uses the `hello_lib` library. The directory
structure now looks like the following:

```
[workspace]/
    WORKSPACE
    hello_lib/
        BUILD
        src/
            greeter.rs
            lib.rs
    hello_world/
        BUILD
        src/
            main.rs
```

`hello_world/src/main.rs`:

```rust
extern crate hello_lib;

use hello_lib::greeter;

fn main() {
    let hello = greeter::Greeter::new("Hello");
    hello.greet("world");
}
```

`hello_world/BUILD`:

```python
load("/tools/build_rules/rust/rust", "rust_binary")

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

<a name="reference"></a>
## Build Rule Reference

<a name="reference-rust_library"></a>
### `rust_library`

`rust_library(name, srcs, deps, data, crate_features, rustc_flags)`

<table>
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
          There must be a file either named <code>lib.rs</code> or with a name
          matching the name of this crate. For example, if the name of a given
          rule is <code>foo</code>, then there must be a file named
          <code>lib.rs</code> or <code>foo.rs</code> in <code>srcs</code>.
          This file will be passed to <code>rustc</code> as the crate root.
        </p>
      </td>
    </tr>
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

<a name="reference-rust_binary"></a>
### `rust_binary`

`rust_binary(name, srcs, deps, data, crate_features, rustc_flags)`

<table>
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
          There must be a file either named <code>main.rs</code> or with a name
          matching the name of this crate that contains the <code>main</code>
          function. For example, if the name of a given
          rule is <code>foo</code>, then there must be a file named
          <code>main.rs</code> or <code>foo.rs</code> in <code>srcs</code>.
          This file will be passed to <code>rustc</code> as the crate root.
        </p>
      </td>
    </tr>
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

<a name="reference-rust_test"></a>
### `rust_test`

`rust_test(name, srcs, deps, data, crate_features, rustc_flags)`

<table>
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
          There must be a file either with a name matching the name of this
          test. For example, if the name of a <code>rust_test</code> rule is
          <code>foo</code>, then there must be a file named <code>foo.rs</code>
          in <code>srcs</code>.  This file will be passed to <code>rustc</code>
          as the crate root.
        </p>
      </td>
    </tr>
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

<a name="#roadmap"></a>
## Roadmap

### Near-term roadmap

* Implement `rust_bench_test` rule for running benchmarks.
* Enable `rust_test` to depend solely on a `rust_library` since many projects
  intermix `#[test]` methods in implementation source.
* Improve documentation with more detailed examples.
* Implement `rust_doc` rule for generating [rustdoc][rustdoc] documentation.

[rustdoc]: https://doc.rust-lang.org/book/documentation.html#about-rustdoc

### Longer-term roadmap

* Add tool for taking `Cargo.toml` and generating a `WORKSPACE` file with
  workspace rules for pulling external dependencies.
* Improve expressiveness of features and support for [Cargo's feature
  groups](http://doc.crates.io/manifest.html#the-[features]-section).
* Add `cargo_crate` workspace rule for pulling crates from
  [Cargo](https://crates.io/).
