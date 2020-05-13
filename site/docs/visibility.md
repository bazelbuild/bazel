# Visibility

Visibility controls whether a target can be used by other packages. It is an
important tool to structure the code when a codebase grows: it lets developers
distinguish between implementation details and libraries other people can depend
on.

If you need to disable the visibility check (for example when experimenting),
use `--check_visibility=false`.

## Visibility labels

There are five forms a visibility label can take:

*   `["//visibility:public"]`: Anyone can use this rule.
*   `["//visibility:private"]`: Only rules in this package can use this rule.
    Rules in `javatests/foo/bar` can always use rules in `java/foo/bar`.
*   `["//some/package:__pkg__", "//other/package:__pkg__"]`: Only rules in
    `some/package/BUILD` and `other/package/BUILD` have access to this rule.
    Note that sub-packages do not have access to the rule; for example,
    `//some/package/foo:bar` or `//other/package/testing:bla` wouldn't have
    access. `__pkg__` is a special target and must be used verbatim. It
    represents all of the rules in the package.
*   `["//project:__subpackages__", "//other:__subpackages__"]`: Only rules in
    packages `project` or `other` or in one of their sub-packages have access to
    this rule. For example, `//project:rule`, `//project/library:lib` or
    `//other/testing/internal:munge` are allowed to depend on this rule (but not
    `//independent:evil`)
*   `["//some/package:my_package_group"]`: A
    [package group](be/functions.html#package_group) is a named set of package
    names. Package groups can also grant access rights to entire subtrees,
    e.g.`//myproj/...`.

The visibility specifications of `//visibility:public` and
`//visibility:private` can not be combined with any other visibility
specifications.

A visibility specification may contain a combination of package labels (i.e.
`//foo:__pkg__`) and `package_group`s.

## Visibility of a rule

If a rule does specify the visibility attribute, that specification overrides
any [`default_visibility`](be/functions.html#package.default_visibility)
attribute of the [`package`](functions.html#package) statement in the BUILD
file containing the rule.

Otherwise, if a rule does not specify the visibility attribute, the
default_visibility of the package is used.

Otherwise, if the default_visibility for the package is not specified,
`//visibility:private` is used.

### Example

File `//frobber/bin/BUILD`:

```python
# This rule is visible to everyone
cc_binary(
    name = "executable",
    visibility = ["//visibility:public"],
    deps = [":library"],
)

# This rule is visible only to rules declared in the same package
cc_library(
    name = "library",
    visibility = ["//visibility:private"],
)

# This rule is visible to rules in package //object and //noun
cc_library(
    name = "subject",
    visibility = [
        "//noun:__pkg__",
        "//object:__pkg__",
    ],
)

# See package group "//frobber:friends" (below) for who can
# access this rule.
cc_library(
    name = "thingy",
    visibility = ["//frobber:friends"],
)
```

File `//frobber/BUILD`:

```python
# This is the package group declaration to which rule
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

## Visibility of a file

By default, files are visible only from the same package. To make a file
accessible from another package, use
[`exports_files`](be/functions.html#exports_files).

If the call to `exports_files` rule does specify the visibility attribute, that
specification applies. Otherwise, the file is public (the `default_visibility`
is ignored).

### Legacy behavior

If the flag [`--incompatible_no_implicit_file_export`](https://github.com/bazelbuild/bazel/issues/10225)
is not set, a legacy behavior applies instead.

With the legacy behavior, files used by at least one rule in the file are
implicitly exported using the `default_visibility` specification. See the
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
