Project: /_project.yaml
Book: /_book.yaml

# Dependencies

{% include "_buttons.html" %}

A target `A` _depends upon_ a target `B` if `B` is needed by `A` at build or
execution time. The _depends upon_ relation induces a
[Directed Acyclic Graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph){: .external}
(DAG) over targets, and it is called a _dependency graph_.

A target's _direct_ dependencies are those other targets reachable by a path
of length 1 in the dependency graph. A target's _transitive_ dependencies are
those targets upon which it depends via a path of any length through the graph.

In fact, in the context of builds, there are two dependency graphs, the graph
of _actual dependencies_ and the graph of _declared dependencies_. Most of the
time, the two graphs are so similar that this distinction need not be made, but
it is useful for the discussion below.

## Actual and declared dependencies {:#actual-and-declared-dependencies}

A target `X` is _actually dependent_ on target `Y` if `Y` must be present,
built, and up-to-date in order for `X` to be built correctly. _Built_ could
mean generated, processed, compiled, linked, archived, compressed, executed, or
any of the other kinds of tasks that routinely occur during a build.

A target `X` has a _declared dependency_ on target `Y` if there is a dependency
edge from `X` to `Y` in the package of `X`.

For correct builds, the graph of actual dependencies _A_ must be a subgraph of
the graph of declared dependencies _D_. That is, every pair of
directly-connected nodes `x --> y` in _A_ must also be directly connected in
_D_. It can be said that _D_ is an _overapproximation_ of _A_.

Important: _D_ should not be too much of an overapproximation of _A_ because
redundant declared dependencies can make builds slower and binaries larger.

`BUILD` file writers must explicitly declare all of the actual direct
dependencies for every rule to the build system, and no more.

Failure to observe this principle causes undefined behavior: the build may fail,
but worse, the build may depend on some prior operations, or upon transitive
declared dependencies the target happens to have. Bazel checks for missing
dependencies and report errors, but it's not possible for this checking to be
complete in all cases.

You need not (and should not) attempt to list everything indirectly imported,
even if it is _needed_ by `A` at execution time.

During a build of target `X`, the build tool inspects the entire transitive
closure of dependencies of `X` to ensure that any changes in those targets are
reflected in the final result, rebuilding intermediates as needed.

The transitive nature of dependencies leads to a common mistake. Sometimes,
code in one file may use code provided by an _indirect_ dependency — a
transitive but not direct edge in the declared dependency graph. Indirect
dependencies don't appear in the `BUILD` file. Because the rule doesn't
directly depend on the provider, there is no way to track changes, as shown in
the following example timeline:

### 1. Declared dependencies match actual dependencies {:#this-is-fine}

At first, everything works. The code in package `a` uses code in package `b`.
The code in package `b` uses code in package `c`, and thus `a` transitively
depends on `c`.

<table class="cyan">
  <tr>
    <th><code>a/BUILD</code></th>
    <th><code><strong>b</strong>/BUILD</code></th>
  </tr>
  <tr>
    <td>
      <pre>rule(
    name = "a",
    srcs = "a.in",
    deps = "//b:b",
)
      </pre>
    </td>
    <td>
      <pre>
rule(
    name = "b",
    srcs = "b.in",
    deps = "//c:c",
)
      </pre>
    </td>
  </tr>
  <tr class="alt">
    <td><code>a / a.in</code></td>
    <td><code>b / b.in</code></td>
  </tr>
  <tr>
    <td><pre>
import b;
b.foo();
    </pre>
    </td>
    <td>
      <pre>
import c;
function foo() {
  c.bar();
}
      </pre>
    </td>
  </tr>
  <tr>
    <td>
      <figure>
        <img src="/docs/images/a_b_c.svg"
             alt="Declared dependency graph with arrows connecting a, b, and c">
        <figcaption><b>Declared</b> dependency graph</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/docs/images/a_b_c.svg"
             alt="Actual dependency graph that matches the declared dependency
                  graph with arrows connecting a, b, and c">
        <figcaption><b>Actual</b> dependency graph</figcaption>
      </figure>
    </td>
  </tr>
</table>

The declared dependencies overapproximate the actual dependencies. All is well.

### 2. Adding an undeclared dependency {:#undeclared-dependency}

A latent hazard is introduced when someone adds code to `a` that creates a
direct _actual_ dependency on `c`, but forgets to declare it in the build file
`a/BUILD`.

<table class="cyan">
  <tr>
    <th><code>a / a.in</code></th>
    <th>&nbsp;</th>
  </tr>
  <tr>
    <td>
      <pre>
        import b;
        import c;
        b.foo();
        c.garply();
      </pre>
    </td>
    <td>&nbsp;</td>
  </tr>
  <tr>
    <td>
      <figure>
        <img src="/docs/images/a_b_c.svg"
             alt="Declared dependency graph with arrows connecting a, b, and c">
        <figcaption><b>Declared</b> dependency graph</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/docs/images/a_b_c_ac.svg"
             alt="Actual dependency graph with arrows connecting a, b, and c. An
                  arrow now connects A to C as well. This does not match the
                  declared dependency graph">
        <figcaption><b>Actual</b> dependency graph</figcaption>
      </figure>
    </td>
  </tr>
</table>

The declared dependencies no longer overapproximate the actual dependencies.
This may build ok, because the transitive closures of the two graphs are equal,
but masks a problem: `a` has an actual but undeclared dependency on `c`.

### 3. Divergence between declared and actual dependency graphs {:#divergence}

The hazard is revealed when someone refactors `b` so that it no longer depends on
`c`, inadvertently breaking `a` through no
fault of their own.

<table class="cyan">
  <tr>
    <th>&nbsp;</th>
    <th><code><strong>b</strong>/BUILD</code></th>
  </tr>
  <tr>
    <td>&nbsp;</td>
    <td>
      <pre>rule(
    name = "b",
    srcs = "b.in",
    <strong>deps = "//d:d",</strong>
)
      </pre>
    </td>
  </tr>
  <tr class="alt">
    <td>&nbsp;</td>
    <td><code>b / b.in</code></td>
  </tr>
  <tr>
    <td>&nbsp;</td>
    <td>
      <pre>
      import d;
      function foo() {
        d.baz();
      }
      </pre>
    </td>
  </tr>
  <tr>
    <td>
      <figure>
        <img src="/docs/images/ab_c.svg"
             alt="Declared dependency graph with arrows connecting a and b.
                  b no longer connects to c, which breaks a's connection to c">
        <figcaption><b>Declared</b> dependency graph</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="/docs/images/a_b_a_c.svg"
             alt="Actual dependency graph that shows a connecting to b and c,
                  but b no longer connects to c">
        <figcaption><b>Actual</b> dependency graph</figcaption>
      </figure>
    </td>
  </tr>
</table>

The declared dependency graph is now an underapproximation of the actual
dependencies, even when transitively closed; the build is likely to fail.

The problem could have been averted by ensuring that the actual dependency from
`a` to `c` introduced in Step 2 was properly declared in the `BUILD` file.

## Types of dependencies {:#types-of-dependencies}

Most build rules have three attributes for specifying different kinds of
generic dependencies: `srcs`, `deps` and `data`. These are explained below. For
more details, see
[Attributes common to all rules](/reference/be/common-definitions).

Many rules also have additional attributes for rule-specific kinds of
dependencies, for example, `compiler` or `resources`. These are detailed in the
[Build Encyclopedia](/reference/be/).

### `srcs` dependencies {:#srcs-dependencies}

Files consumed directly by the rule or rules that output source files.

### `deps` dependencies {:#deps-dependencies}

Rule pointing to separately-compiled modules providing header files,
symbols, libraries, data, etc.

### `data` dependencies {:#data-dependencies}

A build target might need some data files to run correctly. These data files
aren't source code: they don't affect how the target is built. For example, a
unit test might compare a function's output to the contents of a file. When you
build the unit test you don't need the file, but you do need it when you run
the test. The same applies to tools that are launched during execution.

The build system runs tests in an isolated directory where only files listed as
`data` are available. Thus, if a binary/library/test needs some files to run,
specify them (or a build rule containing them) in `data`. For example:

```
# I need a config file from a directory named env:
java_binary(
    name = "setenv",
    ...
    data = [":env/default_env.txt"],
)

# I need test data from another directory
sh_test(
    name = "regtest",
    srcs = ["regtest.sh"],
    data = [
        "//data:file1.txt",
        "//data:file2.txt",
        ...
    ],
)
```

These files are available using the relative path `path/to/data/file`. In tests,
you can refer to these files by joining the paths of the test's source
directory and the workspace-relative path, for example,
`${TEST_SRCDIR}/workspace/path/to/data/file`.

## Using labels to reference directories {:#using-labels-reference-directories}

As you look over our `BUILD` files, you might notice that some `data` labels
refer to directories. These labels end with `/.` or `/` like these examples,
which you should not use:

<p><span class="compare-worse">Not recommended</span> —
  <code>data = ["//data/regression:unittest/."]</code>
</p>

<p><span class="compare-worse">Not recommended</span> —
  <code>data = ["testdata/."]</code>
</p>

<p><span class="compare-worse">Not recommended</span> —
  <code>data = ["testdata/"]</code>
</p>


This seems convenient, particularly for tests because it allows a test to
use all the data files in the directory.

But try not to do this. In order to ensure correct incremental rebuilds (and
re-execution of tests) after a change, the build system must be aware of the
complete set of files that are inputs to the build (or test). When you specify
a directory, the build system performs a rebuild only when the directory itself
changes (due to addition or deletion of files), but won't be able to detect
edits to individual files as those changes don't affect the enclosing directory.
Rather than specifying directories as inputs to the build system, you should
enumerate the set of files contained within them, either explicitly or using the
[`glob()`](/reference/be/functions#glob) function. (Use `**` to force the
`glob()` to be recursive.)


<p><span class="compare-better">Recommended</span> —
  <code>data = glob(["testdata/**"])</code>
</p>

Unfortunately, there are some scenarios where directory labels must be used.
For example, if the `testdata` directory contains files whose names don't
conform to the [label syntax](/concepts/labels#labels-lexical-specification),
then explicit enumeration of files, or use of the
[`glob()`](/reference/be/functions#glob) function produces an invalid labels
error. You must use directory labels in this case, but beware of the
associated risk of incorrect rebuilds described above.

If you must use directory labels, keep in mind that you can't refer to the
parent package with a relative `../` path; instead, use an absolute path like
`//data/regression:unittest/.`.

Note: Directory labels are only valid for data dependencies. If you try to use
a directory as a label in an argument other than `data`, it will fail and you
will get a (probably cryptic) error message.

Any external rule, such as a test, that needs to use multiple files must
explicitly declare its dependence on all of them. You can use `filegroup()` to
group files together in the `BUILD` file:

```
filegroup(
        name = 'my_data',
        srcs = glob(['my_unittest_data/*'])
)
```

You can then reference the label `my_data` as the data dependency in your test.

<table class="columns">
  <tr>
    <td><a class="button button-with-icon button-primary"
           href="/concepts/build-files">
        <span class="material-icons" aria-hidden="true">arrow_back</span>BUILD files</a>
    </td>
    <td><a class="button button-with-icon button-primary"
           href="/concepts/visibility">
        Visibility<span class="material-icons icon-after" aria-hidden="true">arrow_forward</span></a>
    </td>
  </tr>
</table>

