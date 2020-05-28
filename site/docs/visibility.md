---
layout: documentation
title: Visibility
---
All rule targets have a `visibility` attribute that takes a list of labels. One
target is visible to another if they are in the same package, or if they are
granted visibility by one of the labels.

Each label has one of the following forms:

*   `"//visibility:public"`: Anyone can use this target. (May not be combined
    with any other specification.)

*   `"//visibility:private"`: Only targets in this package can use this
    target. (May not be combined with any other specification.)

*   `"//foo/bar:__pkg__"`: Grants access to targets defined in `//foo/bar` (but
    not its subpackages). Here, `__pkg__` is a special piece of syntax
    representing all of the targets in a package.

*   `"//foo/bar:__subpackages__"`: Grants access to targets defined in
    `//foo/bar`, or any of its direct or indirect subpackages. Again,
    `__subpackages__` is special syntax.

*   `"//foo/bar:my_package_group"`: Grants access to all of the packages named
    by the given [package group](be/functions.html#package_group). This can be
    used to allow access to an entire subtree, such as `//myproj/...`.

For example, if `//some/package:mytarget` has its `visibility` set to
`[":__subpackages__", "//tests:__pkg__"]`, then it could be used by any target
that is part of the `//some/package/...` source tree, as well as targets defined
in `//tests/BUILD`, but not by targets defined in `//tests/integration/BUILD`.

As a special case, `package_group` targets themselves do not have a `visibility`
attribute; they are always publicly visible.

## Visibility of a rule target

If a rule target does not set the `visibility` attribute, its visibility is
given by the
[`default_visibility`](be/functions.html#package.default_visibility) that was
specified in the [`package`](functions.html#package) statement of the target's
BUILD file. If there is no such `default_visibility` declaration, the visibility
is `//visibility:private`.

### Example

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

## Visibility of a file target

By default, file targets are visible only from the same package. To make a file
accessible from another package, use
[`exports_files`](be/functions.html#exports_files).

If the call to `exports_files` specifies the visibility attribute, that
visibility applies. Otherwise, the file is public (the `default_visibility`
is ignored).

When possible, prefer exposing a library or another type of rule instead of a
source file. For example, declare a `java_library` instead of exporting a
`.java` file. It's good form for a rule target to only directly include sources
in its own package.

### Example

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

### Legacy behavior

If the flag [`--incompatible_no_implicit_file_export`](https://github.com/bazelbuild/bazel/issues/10225)
is not set, a legacy behavior applies instead.

With the legacy behavior, files used by at least one rule target in the package
are implicitly exported using the `default_visibility` specification. See the
[design proposal](https://github.com/bazelbuild/proposals/blob/master/designs/2019-10-24-file-visibility.md#example-and-description-of-the-problem)
for more details.

## Visibility of bzl files

`load` statements are currently not subject to visibility. It is possible to
load a `bzl` file anywhere in the workspace.

## Visibility of implicit dependencies

Some rules have implicit dependencies. For example, a C++ rule might implicitly
depend on a C++ compiler.

Currently, implicit dependencies are treated like normal dependencies. They need
to be visible by all instances of the rule. This behavior can be changed using
[`--incompatible_visibility_private_attributes_at_definition`](https://github.com/bazelbuild/proposals/blob/master/designs/2019-10-15-tool-visibility.md).

## Best practices

* Avoid setting the default visibility to public. It may be convenient for
prototyping or in small codebases, but it is discouraged in large codebases: try
to be explicit about which targets are part of the public interface.

* Use `package_group` to share the visibility specifications among multiple
  targets. This is especially useful when targets in many BUILD files should be
  exposed to the same set of packages.

* Use fine-grained visibility specifications when deprecating a target. Restrict
  the visibility to the current users to avoid new dependencies.
