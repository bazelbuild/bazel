Project: /_project.yaml
Book: /_book.yaml

# Visibility

This page covers visibility specifications, best practices, and examples.

Visibility controls whether a target can be used (depended on) by targets in
other packages. This helps other people distinguish between your library's
public API and its implementation details, and is an important tool to help
enforce structure as your workspace grows.

If you need to disable the visibility check (for example when experimenting),
use `--check_visibility=false`.

For more details on package and subpackages, see
[Concepts and terminology](/concepts/build-ref).

## Visibility specifications {:#visibility-specifications}

All rule targets have a `visibility` attribute that takes a list of labels. One
target is visible to another if they are in the same package, or if visibility
is granted to the depending target's package.

Each label has one of the following forms. With the exception of the last form,
these are just syntactic placeholders that do not correspond to any actual
target.

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

As a special case, `package_group` targets themselves do not have a `visibility`
attribute; they are always publicly visible.

Visibility cannot be set to specific non-package_group targets. That triggers a
"Label does not refer to a package group" or "Cycle in dependency graph" error.

## Visibility of a rule target {:#rule-target-visibility}

If a rule target does not set the `visibility` attribute, its visibility is
given by the
[`default_visibility`](/reference/be/functions#package.default_visibility) that was
specified in the [`package`](/reference/be/functions#package) statement of the
target's BUILD file. If there is no such `default_visibility` declaration, the
visibility is `//visibility:private`.

`config_setting` visibility has historically not been enforced.
`--incompatible_enforce_config_setting_visibility` and
`--incompatible_config_setting_private_default_visibility` provide migration
logic for converging with other rules.

If `--incompatible_enforce_config_setting_visibility=false`, every
`config_setting` is unconditionally visible to all targets.

Else if `--incompatible_config_setting_private_default_visibility=false`, any
`config_setting` that doesn't explicitly set visibility is `//visibility:public`
(ignoring package [`default_visibility`](/reference/be/functions#package.default_visibility)).

Else if `--incompatible_config_setting_private_default_visibility=true`,
`config_setting` uses the same visibility logic as all other rules.

Best practice is to treat all `config_setting` targets like other rules:
explicitly set `visibility` on any `config_setting` used anywhere outside its
package.

### Example {:#rule-target-visibility-example}

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

## Visibility of a generated file target {:#generated-file-visibility}

A generated file target has the same visibility as the rule target that
generates it.

## Visibility of a source file target {:#source-file-visibility}

By default, source file targets are visible only from the same package. To make
a source file accessible from another package, use
[`exports_files`](/reference/be/functions#exports_files).

If the call to `exports_files` specifies the visibility attribute, that
visibility applies. Otherwise, the file is public (the `default_visibility`
is ignored).

When possible, prefer exposing a library or another type of rule instead of a
source file. For example, declare a `java_library` instead of exporting a
`.java` file. It's good form for a rule target to only directly include sources
in its own package.

### Example {:#source-file-visibility-example}

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

### Legacy behavior {:#legacy-behavior}

If the flag [`--incompatible_no_implicit_file_export`](https://github.com/bazelbuild/bazel/issues/10225){: .external}
is not set, a legacy behavior applies instead.

With the legacy behavior, files used by at least one rule target in the package
are implicitly exported using the `default_visibility` specification. See the
[design proposal](https://github.com/bazelbuild/proposals/blob/master/designs/2019-10-24-file-visibility.md#example-and-description-of-the-problem){: .external}
for more details.

## Visibility of bzl files {:#visibility-bzl-files}

`BUILD` and `.bzl` files, as processed by Bazel during loading, are not
considered to be targets and therefore are not subject to visibility. It is
therefore possible to load a `.bzl` file from anywhere in the workspace.

However, users may choose to run the Buildifier linter.
The [bzl-visibility](https://github.com/bazelbuild/buildtools/blob/master/WARNINGS.md#bzl-visibility)
check provides a warning if users `load` from beneath a subdirectory named
`internal` or `private`.

## Visibility of implicit dependencies {:#visibility-implicit-dependencies}

Some rules have implicit dependencies. For example, a C++ rule might implicitly
depend on a C++ compiler.

Currently, implicit dependencies are treated like normal dependencies. They need
to be visible by all instances of the rule. This behavior can be changed using
[`--incompatible_visibility_private_attributes_at_definition`](https://github.com/bazelbuild/proposals/blob/master/designs/2019-10-15-tool-visibility.md){: .external}.

## Best practices {:#best-practices}

* Avoid setting the default visibility to public. It may be convenient for
prototyping or in small codebases, but it is discouraged in large codebases: try
to be explicit about which targets are part of the public interface.

* Use `package_group` to share the visibility specifications among multiple
  targets. This is especially useful when targets in many BUILD files should be
  exposed to the same set of packages.

* Use fine-grained visibility specifications when deprecating a target. Restrict
  the visibility to the current users to avoid new dependencies.
