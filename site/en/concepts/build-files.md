Project: /_project.yaml
Book: /_book.yaml

# BUILD files

{% include "_buttons.html" %}

The previous sections described packages, targets and labels, and the
build dependency graph abstractly. This section describes the concrete syntax
used to define a package.

By definition, every package contains a `BUILD` file, which is a short
program.

Note: The `BUILD` file can be named either `BUILD` or `BUILD.bazel`. If both
files exist, `BUILD.bazel` takes precedence over `BUILD`.
For simplicity's sake, the documentation refers to these files simply as `BUILD`
files.

`BUILD` files are evaluated using an imperative language,
[Starlark](https://github.com/bazelbuild/starlark/){: .external}.

They are interpreted as a sequential list of statements.

In general, order does matter: variables must be defined before they are
used, for example. However, most `BUILD` files consist only of declarations of
build rules, and the relative order of these statements is immaterial; all
that matters is _which_ rules were declared, and with what values, by the
time package evaluation completes.

When a build rule function, such as `cc_library`, is executed, it creates a
new target in the graph. This target can later be referred using a label.

In simple `BUILD` files, rule declarations can be re-ordered freely without
changing the behavior.

To encourage a clean separation between code and data, `BUILD` files cannot
contain function definitions, `for` statements or `if` statements (but list
comprehensions and `if` expressions are allowed). Functions can be declared in
`.bzl` files instead. Additionally, `*args` and `**kwargs` arguments are not
allowed in `BUILD` files; instead list all the arguments explicitly.

Crucially, programs in Starlark can't perform arbitrary I/O. This invariant
makes the interpretation of `BUILD` files hermetic — dependent only on a known
set of inputs, which is essential for ensuring that builds are reproducible.
For more details, see [Hermeticity](/basics/hermeticity).

`BUILD` files should be written using only ASCII characters, although
technically they are interpreted using the Latin-1 character set.

Because `BUILD` files need to be updated whenever the dependencies of the
underlying code change, they are typically maintained by multiple people on a
team. `BUILD` file authors should comment liberally to document the role
of each build target, whether or not it is intended for public use, and to
document the role of the package itself.

## Loading an extension {:#load}

Bazel extensions are files ending in `.bzl`. Use the `load` statement to import
a symbol from an extension.

```
load("//foo/bar:file.bzl", "some_library")
```

This code loads the file `foo/bar/file.bzl` and adds the `some_library` symbol
to the environment. This can be used to load new rules, functions, or constants
(for example, a string or a list). Multiple symbols can be imported by using
additional arguments to the call to `load`. Arguments must be string literals
(no variable) and `load` statements must appear at top-level — they cannot be
in a function body.

The first argument of `load` is a [label](/concepts/labels) identifying a
`.bzl` file. If it's a relative label, it is resolved with respect to the
package (not directory) containing the current `bzl` file. Relative labels in
`load` statements should use a leading `:`.

`load` also supports aliases, therefore, you can assign different names to the
imported symbols.

```
load("//foo/bar:file.bzl", library_alias = "some_library")
```

You can define multiple aliases within one `load` statement. Moreover, the
argument list can contain both aliases and regular symbol names. The following
example is perfectly legal (please note when to use quotation marks).

```
load(":my_rules.bzl", "some_rule", nice_alias = "some_other_rule")
```

In a `.bzl` file, symbols starting with `_` are not exported and cannot be
loaded from another file.

You can use [load visibility](/concepts/visibility#load-visibility) to restrict
who may load a `.bzl` file.

## Types of build rules {:#types-of-build-rules}

The majority of build rules come in families, grouped together by
language. For example, `cc_binary`, `cc_library`
and `cc_test` are the build rules for C++ binaries,
libraries, and tests, respectively. Other languages use the same
naming scheme, with a different prefix, such as `java_*` for
Java. Some of these functions are documented in the
[Build Encyclopedia](/reference/be/overview), but it is possible
for anyone to create new rules.

* `*_binary` rules build executable programs in a given language. After a
  build, the executable will reside in the build tool's binary
  output tree at the corresponding name for the rule's label,
  so `//my:program` would appear at (for example) `$(BINDIR)/my/program`.

  In some languages, such rules also create a runfiles directory
  containing all the files mentioned in a `data`
  attribute belonging to the rule, or any rule in its transitive
  closure of dependencies; this set of files is gathered together in
  one place for ease of deployment to production.

* `*_test` rules are a specialization of a `*_binary` rule, used for automated
  testing. Tests are simply programs that return zero on success.

  Like binaries, tests also have runfiles trees, and the files
  beneath it are the only files that a test may legitimately open
  at runtime. For example, a program `cc_test(name='x',
  data=['//foo:bar'])` may open and read `$TEST_SRCDIR/workspace/foo/bar` during execution.
  (Each programming language has its own utility function for
  accessing the value of `$TEST_SRCDIR`, but they are all
  equivalent to using the environment variable directly.)
  Failure to observe the rule will cause the test to fail when it is
  executed on a remote testing host.

* `*_library` rules specify separately-compiled modules in the given
    programming language. Libraries can depend on other libraries,
    and binaries and tests can depend on libraries, with the expected
    separate-compilation behavior.

<table class="columns">
  <tr>
    <td><a class="button button-with-icon button-primary"
           href="/concepts/labels">
        <span class="material-icons" aria-hidden="true">arrow_back</span>Labels</a>
    </td>
    <td><a class="button button-with-icon button-primary"
           href="/concepts/dependencies">
        Dependencies<span class="material-icons icon-after" aria-hidden="true">arrow_forward</span></a>
    </td>
  </tr>
</table>
