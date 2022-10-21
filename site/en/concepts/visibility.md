Project: /_project.yaml
Book: /_book.yaml

# Visibility

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

    *   Package groups use a different syntax for specifying packages. Within a
        package group, the forms `"//foo/bar:__pkg__"` and
        `"//foo/bar:__subpackages__"` are respectively replaced by `"//foo/bar"`
        and `"//foo/bar/..."`. Likewise, `"//visibility:public"` and
        `"//visibility:private"` are just `"public"` and `"private"`.

For example, if `//some/package:mytarget` has its `visibility` set to
`[":__subpackages__", "//tests:__pkg__"]`, then it could be used by any target
that is part of the `//some/package/...` source tree, as well as targets defined
in `//tests/BUILD`, but not by targets defined in `//tests/integration/BUILD`.

**Best practice:** To make several targets visible to the same set
of packages, use a `package_group` instead of repeating the list in each
target's `visibility` attribute. This increases readability and prevents the
lists from getting out of sync.

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

Currently, for visibility purposes these implicit dependencies are treated like
any other dependency. This means that the target being depended on (such as our
C++ compiler) must be visible to every instance of the rule. In practice this
usually means the target must have public visibility.

You can change this behavior by setting
[`--incompatible_visibility_private_attributes_at_definition`](https://github.com/bazelbuild/proposals/blob/master/designs/2019-10-15-tool-visibility.md){: .external}. When enabled, the
target in question need only be visible to the rule declaring it an implicit
dependency. That is, it must be visible to the package containing the `.bzl`
file in which the rule is defined. In our example, the C++ compiler could be
private so long as it lives in the same package as the definition of the
`cc_library` rule.

## Load visibility {:#load-visibility}

**Load visibility** controls whether a `.bzl` file may be loaded from other
`BUILD` or `.bzl` files.

`BUILD` and `.bzl` files, as processed by Bazel during loading, are not
considered to be targets and therefore are not subject to visibility. It is
possible to load a `.bzl` file from anywhere in the workspace.

However, users may choose to run the Buildifier linter.
The [bzl-visibility](https://github.com/bazelbuild/buildtools/blob/master/WARNINGS.md#bzl-visibility)
check provides a warning if users `load` from beneath a subdirectory named
`internal` or `private`.
