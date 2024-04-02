Project: /_project.yaml
Book: /_book.yaml

# Visibility

{% include "_buttons.html" %}

This page covers Bazel's two visibility systems:
[target visibility](#target-visibility) and [load visibility](#load-visibility).

Both types of visibility help other developers distinguish between your
library's public API and its implementation details, and help enforce structure
as your workspace grows. You can also use visibility when deprecating a public
API to allow current users while denying new ones.

## Target visibility {:#target-visibility}

**Target visibility** controls who may depend on your target — that is, who may
use your target's label inside an attribute such as `deps`.

A target `A` is visible to a target `B` if they are in the same package, or if
`A` grants visibility to `B`'s package. Thus, packages are the unit of
granularity for deciding whether or not to allow access. If `B` depends on `A`
but `A` is not visible to `B`, then any attempt to build `B` fails during
[analysis](/reference/glossary#analysis-phase).

Note that granting visibility to a package does not by itself grant visibility
to its subpackages. For more details on package and subpackages, see
[Concepts and terminology](/concepts/build-ref).

For prototyping, you can disable target visibility enforcement by setting the
flag `--check_visibility=false`. This should not be done for production usage in
submitted code.

The primary way to control visibility is with the
[`visibility`](/reference/be/common-definitions#common.visibility) attribute on
rule targets. This section describes the format of this attribute, and how to
determine a target's visibility.

### Visibility specifications {:#visibility-specifications}

All rule targets have a `visibility` attribute that takes a list of labels. Each
label has one of the following forms. With the exception of the last form, these
are just syntactic placeholders that do not correspond to any actual target.

*   `"//visibility:public"`: Grants access to all packages. (May not be combined
    with any other specification.)

*   `"//visibility:private"`: Does not grant any additional access; only targets
    in this package can use this target. (May not be combined with any other
    specification.)

*   `"//foo/bar:__pkg__"`: Grants access to `//foo/bar` (but not its
    subpackages).

*   `"//foo/bar:__subpackages__"`: Grants access `//foo/bar` and all of its
    direct and indirect subpackages.

*   `"//some_pkg:my_package_group"`: Grants access to all of the packages that
    are part of the given [`package_group`](/reference/be/functions#package_group).

    *   Package groups use a
        [different syntax](/reference/be/functions#package_group.packages) for
        specifying packages. Within a package group, the forms
        `"//foo/bar:__pkg__"` and `"//foo/bar:__subpackages__"` are respectively
        replaced by `"//foo/bar"` and `"//foo/bar/..."`. Likewise,
        `"//visibility:public"` and `"//visibility:private"` are just `"public"`
        and `"private"`.

For example, if `//some/package:mytarget` has its `visibility` set to
`[":__subpackages__", "//tests:__pkg__"]`, then it could be used by any target
that is part of the `//some/package/...` source tree, as well as targets defined
in `//tests/BUILD`, but not by targets defined in `//tests/integration/BUILD`.

**Best practice:** To make several targets visible to the same set
of packages, use a `package_group` instead of repeating the list in each
target's `visibility` attribute. This increases readability and prevents the
lists from getting out of sync.

**Best practice:** When granting visibility to another team's project, prefer
`__subpackages__` over `__pkg__` to avoid needless visibility churn as that
project evolves and adds new subpackages.

Note: The `visibility` attribute may not specify non-`package_group` targets.
Doing so triggers a "Label does not refer to a package group" or "Cycle in
dependency graph" error.

### Rule target visibility {:#rule-target-visibility}

A rule target's visibility is:

1. The value of its `visibility` attribute, if set; or else

2. The value of the
[`default_visibility`](/reference/be/functions#package.default_visibility)
argument of the [`package`](/reference/be/functions#package) statement in the
target's `BUILD` file, if such a declaration exists; or else

3. `//visibility:private`.

**Best practice:** Avoid setting `default_visibility` to public. It may be
convenient for prototyping or in small codebases, but the risk of inadvertently
creating public targets increases as the codebase grows. It's better to be
explicit about which targets are part of a package's public interface.

#### Example {:#rule-target-visibility-example}

File `//frobber/bin/BUILD`:

```python
# This target is visible to everyone
cc_binary(
    name = "executable",
    visibility = ["//visibility:public"],
    deps = [":library"],
)

# This target is visible only to targets declared in the same package
cc_library(
    name = "library",
    # No visibility -- defaults to private since no
    # package(default_visibility = ...) was used.
)

# This target is visible to targets in package //object and //noun
cc_library(
    name = "subject",
    visibility = [
        "//noun:__pkg__",
        "//object:__pkg__",
    ],
)

# See package group "//frobber:friends" (below) for who can
# access this target.
cc_library(
    name = "thingy",
    visibility = ["//frobber:friends"],
)
```

File `//frobber/BUILD`:

```python
# This is the package group declaration to which target
# //frobber/bin:thingy refers.
#
# Our friends are packages //frobber, //fribber and any
# subpackage of //fribber.
package_group(
    name = "friends",
    packages = [
        "//fribber/...",
        "//frobber",
    ],
)
```

### Generated file target visibility {:#generated-file-target-visibility}

A generated file target has the same visibility as the rule target that
generates it.

### Source file target visibility {:#source-file-target-visibility}

You can explicitly set the visibility of a source file target by calling
[`exports_files`](/reference/be/functions#exports_files). When no `visibility`
argument is passed to `exports_files`, it makes the visibility public.
`exports_files` may not be used to override the visibility of a generated file.

For source file targets that do not appear in a call to `exports_files`, the
visibility depends on the value of the flag
[`--incompatible_no_implicit_file_export`](https://github.com/bazelbuild/bazel/issues/10225){: .external}:

*   If the flag is set, the visibility is private.

*   Else, the legacy behavior applies: The visibility is the same as the
    `BUILD` file's `default_visibility`, or private if a default visibility is
    not specified.

Avoid relying on the legacy behavior. Always write an `exports_files`
declaration whenever a source file target needs non-private visibility.

**Best practice:** When possible, prefer to expose a rule target rather than a
source file. For example, instead of calling `exports_files` on a `.java` file,
wrap the file in a non-private `java_library` target. Generally, rule targets
should only directly reference source files that live in the same package.

#### Example {:#source-file-visibility-example}

File `//frobber/data/BUILD`:

```python
exports_files(["readme.txt"])
```

File `//frobber/bin/BUILD`:

```python
cc_binary(
  name = "my-program",
  data = ["//frobber/data:readme.txt"],
)
```

### Config setting visibility {:#config-setting-visibility}

Historically, Bazel has not enforced visibility for
[`config_setting`](/reference/be/general#config_setting) targets that are
referenced in the keys of a [`select()`](/reference/be/functions#select). There
are two flags to remove this legacy behavior:

*   [`--incompatible_enforce_config_setting_visibility`](https://github.com/bazelbuild/bazel/issues/12932){: .external}
    enables visibility checking for these targets. To assist with migration, it
    also causes any `config_setting` that does not specify a `visibility` to be
    considered public (regardless of package-level `default_visibility`).

*   [`--incompatible_config_setting_private_default_visibility`](https://github.com/bazelbuild/bazel/issues/12933){: .external}
    causes `config_setting`s that do not specify a `visibility` to respect the
    package's `default_visibility` and to fallback on private visibility, just
    like any other rule target. It is a no-op if
    `--incompatible_enforce_config_setting_visibility` is not set.

Avoid relying on the legacy behavior. Any `config_setting` that is intended to
be used outside the current package should have an explicit `visibility`, if the
package does not already specify a suitable `default_visibility`.

### Package group target visibility {:#package-group-target-visibility}

`package_group` targets do not have a `visibility` attribute. They are always
publicly visible.

### Visibility of implicit dependencies {:#visibility-implicit-dependencies}

Some rules have [implicit dependencies](/extending/rules#private_attributes_and_implicit_dependencies) —
dependencies that are not spelled out in a `BUILD` file but are inherent to
every instance of that rule. For example, a `cc_library` rule might create an
implicit dependency from each of its rule targets to an executable target
representing a C++ compiler.

The visibility of such an implicit dependency is checked with respect to the
package containing the `.bzl` file in which the rule (or aspect) is defined. In
our example, the C++ compiler could be private so long as it lives in the same
package as the definition of the `cc_library` rule. As a fallback, if the
implicit dependency is not visible from the definition, it is checked with
respect to the `cc_library` target.

If you want to restrict the usage of a rule to certain packages, use
[load visibility](#load-visibility) instead.

## Load visibility {:#load-visibility}

**Load visibility** controls whether a `.bzl` file may be loaded from other
`BUILD` or `.bzl` files outside the current package.

In the same way that target visibility protects source code that is encapsulated
by targets, load visibility protects build logic that is encapsulated by `.bzl`
files. For instance, a `BUILD` file author might wish to factor some repetitive
target definitions into a macro in a `.bzl` file. Without the protection of load
visibility, they might find their macro reused by other collaborators in the
same workspace, so that modifying the macro breaks other teams' builds.

Note that a `.bzl` file may or may not have a corresponding source file target.
If it does, there is no guarantee that the load visibility and the target
visibility coincide. That is, the same `BUILD` file might be able to load the
`.bzl` file but not list it in the `srcs` of a [`filegroup`](/reference/be/general#filegroup),
or vice versa. This can sometimes cause problems for rules that wish to consume
`.bzl` files as source code, such as for documentation generation or testing.

For prototyping, you may disable load visibility enforcement by setting
`--check_bzl_visibility=false`. As with `--check_visibility=false`, this should
not be done for submitted code.

Load visibility is available as of Bazel 6.0.

### Declaring load visibility {:#declaring-load-visibility}

To set the load visibility of a `.bzl` file, call the
[`visibility()`](/rules/lib/globals/bzl#visibility) function from within the file.
The argument to `visibility()` is a list of package specifications, just like
the [`packages`](/reference/be/functions#package_group.packages) attribute of
`package_group`. However, `visibility()` does not accept negative package
specifications.

The call to `visibility()` must only occur once per file, at the top level (not
inside a function), and ideally immediately following the `load()` statements.

Unlike target visibility, the default load visibility is always public. Files
that do not call `visibility()` are always loadable from anywhere in the
workspace. It is a good idea to add `visibility("private")` to the top of any
new `.bzl` file that is not specifically intended for use outside the package.

### Example {:#load-visibility-example}

```python
# //mylib/internal_defs.bzl

# Available to subpackages and to mylib's tests.
visibility(["//mylib/...", "//tests/mylib/..."])

def helper(...):
    ...
```

```python
# //mylib/rules.bzl

load(":internal_defs.bzl", "helper")
# Set visibility explicitly, even though public is the default.
# Note the [] can be omitted when there's only one entry.
visibility("public")

myrule = rule(
    ...
)
```

```python
# //someclient/BUILD

load("//mylib:rules.bzl", "myrule")          # ok
load("//mylib:internal_defs.bzl", "helper")  # error

...
```

### Load visibility practices {:#load-visibility-practices}

This section describes tips for managing load visibility declarations.

#### Factoring visibilities {:#factoring-visibilities}

When multiple `.bzl` files should have the same visibility, it can be helpful to
factor their package specifications into a common list. For example:

```python
# //mylib/internal_defs.bzl

visibility("private")

clients = [
    "//foo",
    "//bar/baz/...",
    ...
]
```

```python
# //mylib/feature_A.bzl

load(":internal_defs.bzl", "clients")
visibility(clients)

...
```

```python
# //mylib/feature_B.bzl

load(":internal_defs.bzl", "clients")
visibility(clients)

...
```

This helps prevent accidental skew between the various `.bzl` files'
visibilities. It also is more readable when the `clients` list is large.

#### Composing visibilities {:#composing-visibilities}

Sometimes a `.bzl` file might need to be visible to an allowlist that is
composed of multiple smaller allowlists. This is analogous to how a
`package_group` can incorporate other `package_group`s via its
[`includes`](/reference/be/functions#package_group.includes) attribute.

Suppose you are deprecating a widely used macro. You want it to be visible only
to existing users and to the packages owned by your own team. You might write:

```python
# //mylib/macros.bzl

load(":internal_defs.bzl", "our_packages")
load("//some_big_client:defs.bzl", "their_remaining_uses")

# List concatenation. Duplicates are fine.
visibility(our_packages + their_remaining_uses)
```

#### Deduplicating with package groups {:#deduplicating-with-package-groups}

Unlike target visibility, you cannot define a load visibility in terms of a
`package_group`. If you want to reuse the same allowlist for both target
visibility and load visibility, it's best to move the list of package
specifications into a .bzl file, where both kinds of declarations may refer to
it. Building off the example in [Factoring visibilities](#factoring-visibilities)
above, you might write:

```python
# //mylib/BUILD

load(":internal_defs", "clients")

package_group(
    name = "my_pkg_grp",
    packages = clients,
)
```

This only works if the list does not contain any negative package
specifications.

#### Protecting individual symbols {:#protecting-individual-symbols}

Any Starlark symbol whose name begins with an underscore cannot be loaded from
another file. This makes it easy to create private symbols, but does not allow
you to share these symbols with a limited set of trusted files. On the other
hand, load visibility gives you control over what other packages may see your
`.bzl file`, but does not allow you to prevent any non-underscored symbol from
being loaded.

Luckily, you can combine these two features to get fine-grained control.

```python
# //mylib/internal_defs.bzl

# Can't be public, because internal_helper shouldn't be exposed to the world.
visibility("private")

# Can't be underscore-prefixed, because this is
# needed by other .bzl files in mylib.
def internal_helper(...):
    ...

def public_util(...):
    ...
```

```python
# //mylib/defs.bzl

load(":internal_defs", "internal_helper", _public_util="public_util")
visibility("public")

# internal_helper, as a loaded symbol, is available for use in this file but
# can't be imported by clients who load this file.
...

# Re-export public_util from this file by assigning it to a global variable.
# We needed to import it under a different name ("_public_util") in order for
# this assignment to be legal.
public_util = _public_util
```

#### bzl-visibility Buildifier lint {:#bzl-visibility-buildifier-lint}

There is a [Buildifier lint](https://github.com/bazelbuild/buildtools/blob/master/WARNINGS.md#bzl-visibility)
that provides a warning if users load a file from a directory named `internal`
or `private`, when the user's file is not itself underneath the parent of that
directory. This lint predates the load visibility feature and is unnecessary in
workspaces where `.bzl` files declare visibilities.
