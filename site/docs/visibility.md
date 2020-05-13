# Visibility

Visibility controls whether a target can be used by other packages. It is an
important tool to structure the code when a codebase grows: it lets developers
distinguish between implementation details and libraries other people can depend
on.

There are five forms (and one temporary form) a visibility label can take:

*   `["//visibility:public"]`: Anyone can use this rule.
*   `["//visibility:private"]`: Only rules in this package can use this rule.
    Rules in `javatests/foo/bar` can always use rules in `java/foo/bar`.
*   `["//some/package:__pkg__", "//other/package:__pkg__"]`: Only rules in
    `some/package` and `other/package` (defined in `some/package/BUILD` and
    `other/package/BUILD`) have access to this rule. Note that sub-packages do
    not have access to the rule; for example, `//some/package/foo:bar` or
    `//other/package/testing:bla` wouldn't have access. `__pkg__` is a special
    target and must be used verbatim. It represents all of the rules in the
    package.
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

If a rule does specify the visibility attribute, that specification overrides
any [`default_visibility`](be/functions.html#package.default_visibility)
attribute of the [`package`](functions.html#package) statement in the BUILD
file containing the rule.

Otherwise, if a rule does not specify the visibility attribute, the
default_visibility of the package is used (except for
[`exports_files`](be/functions.html#exports_files)).

Otherwise, if the default_visibility for the package is not specified,
`//visibility:private` is used.

## Example

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
