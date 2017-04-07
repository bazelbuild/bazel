---
layout: contribute
title: Recursive WORKSPACE file parsing
---

# Design Document: Recursive WORKSPACE file parsing

**Design documents are not descriptions of the current functionality of Bazel.
Always go to the documentation for current information.**


**Status**: Unimplemented

**Author**: [kchodorow@](mailto:kchodorow@google.com)

**Design document published**: 19 September 2016

## Objective

Users are annoyed by having to specify all deps in their WORKSPACE file.  To
avoid this inconvenience, Bazel could load subprojects' WORKSPACE files
automatically.

## Non-Goals

* Solve the problem of people specifying two different repositories with the
  same name (e.g., @util).
* Solve the problem of people specifying two different names for the same
  repository (`@guava` and `@com_google_guava`).

## Resolution

When a repository is defined in multiple files, which definition "wins"?  What
causes a conflict/error?

### Defined in the main repository's WORKSPACE file

This definition wins, regardless of other definitions.

### In a line

Note: as an intermediate step, this can be disabled, but the end goal is to
allow this so that intermediate dependencies that the top level doesn't care
about don't need to be resolved.

Suppose we have a main repository that depends on repo x, and x depends on repo
y:

<img src="/assets/ws-line.png" class="img-responsive">

In this case, version 1 of "foo" wins.  This way, if a library has already
figured out which version works for them, its reverse dependencies do not have
to think about it.

This will also work if a parent overrides its children's versions, even if it
has multiple children.

### Different lines

If there is no obvious hierarchy and multiple versions are specified, error out.
Report what each chain of dependencies was that wanted the dep and at which
versions:

<img src="/assets/ws-multiline.png" class="img-responsive">

In this case, Bazel would error out with:

```
ERROR: Conflicting definitions of 'foo': bazel-external/y/WORKSPACE:2 repository(name = 'foo' version = '1')
  requested by bazel-external/x/WORKSPACE:2 repository(name = 'y')
  requested by WORKSPACE:3 repository(name = 'x')
vs. bazel-external/a/WORKSPACE:2 repository(name = 'foo' version = '2')
  requested by WORKSPACE:2 repository(name = 'a')
```

This is also the case with diamond dependencies:

<img src="/assets/ws-diamond.png" class="img-responsive">

This would print:

```
ERROR: Conflicting definitions of 'foo': bazel-external/x/WORKSPACE:2 repository(name = 'foo' version = '2')
  requested by WORKSPACE:2 repository(name = 'x')
vs. bazel-external/z/WORKSPACE:2 repository(name = 'foo' version = '1')
  requested by bazel-external/y/WORKSPACE:2 repository(name = 'z')
  requested by WORKSPACE:3 repository(name = 'y')
```

## Upgrade path

I think that this should be fairly straightforward, as any repository used by
the main repository or any subrepository had to be declared in the WORKSPACE
already, so it will take precedence.

To be extra safe, we can start with adding a `recursive = False` attribute to
the `workspace` rule, which we can then flip the default of.

## Implementation

There are two options for implementing this:

* We always download/link every external dependency before a build can happen.
  E.g., if @x isn't defined in the WORKSPACE file, we have to recursively
  traverse all of the repositories to know which repository does define it and
  if there are any conflicting definitions.  This is correct, but will be
  frustrating to users and may not even work in some cases (e.g., if an
  OS-X-only skylark repository rule is fetched on Linux).
* Every time a new WORKSPACE file is fetched, we check its repository rules
  against the ones already defined and look for version conflicts.  This would
  entirely miss certain version conflicts until certain dependencies are built,
  but will have better performance.

I think users will rebel unless we go with Option 2.  However, this can have
some weird effects: suppose we have the diamond dependency above, and the user's
BUILD file contains:

```
cc_library(
    name = "bar",
    deps = ["@x//:dep"],  # using @foo version 2
)

cc_library(
    name = "baz",
    deps = ["@y//:dep"],  # using @foo version 1
)
```

If they build :bar and their coworker builds :baz, the two builds will work and
get different versions of @foo.  However, as soon as one of them tries to build
both, they'll get the version mismatch error.

This is suboptimal, but I can't think of a way that all three of these can be
satisfied:

* The user doesn't have to declare everything at the top level.
* Bazel doesn't have to load everything.
* Bazel can immediately detect any conflicts.

This could be enforced by a CI on presubmit, which I think is good enough.

Whenever Bazel creates a new repository, it will attempt to parse the WORKSPACE
file and do Skyframe lookups against each repository name. If the repository
name is not defined, it will be initialized to the current WORKSPACE's
definition. If it already exists, the existing value will be compared.

For now, we'll be very picky about equality: `maven_jar` and `new_http_archive`
of the same Maven artifact will count as different repositories.  For both
native and skylark repository rules, they will have to be equal to not conflict.

One issue is that is a little tricky but I think will work out: the WORKSPACE
file is parsed incrementally. Suppose the main WORKSPACE loads x.bzl, which
declares @y and @z.  If @y depends on @foo version 1 and @z depends on @foo
version 2, this will throw a Skyframe error, even if @foo is later declared in
the WORKSPACE file.  However, this should be okay, because if these dependencies
actually need @foo, it would need to be declared before them in the WS file
already.

## Supplementary changes

Not strictly required, but as part of this I'm planning to implement:

* A `bazel-external` convenience symlink (to the `[output_base]/external`
  directory) so users can easily inspect their external repositories.
* Add an option to generate all WORKSPACE definitions (so generate a flat
  WORKSPACE file from the hierarchy).

## Concerns

Questions users might have.

*Where did @x come from?*

Bazel will create a `bazel-external/@x.version` should contain the WORKSPACE (or
.bzl file) where we got @x's def and other WORKSPACE files that contain it.

*Which version of @x is going to be chosen?*

See resolution section above. Perhaps people could query for //external:x?

*I want to use a different version of @x.*

Declare @x in your WORKSPACE file, it'll override "lower" rules.

*When I update @x, what else will change?*

Because @x might declare repo @y and @y's version might change as well, we'd
need a different way to query for this.  We could implement deps() for repo
rules or have some other mechanism for this.

## Thoughts on future development

Moving towards the user-as-conflict-resolver model (vs. the
user-as-transcriber-of-deps model) means that repositories that the user may not
even be aware of might be available in their workspace.  I think this kind of
paves the way towards a nice auto-fetch system where a user could just depend on
`@com_google_guava//whatever` in their BUILD file, and Bazel could figure out
how to make `@com_google_guava` available.

## References

[So you want to write a package manager](https://medium.com/@sdboyer/so-you-want-to-write-a-package-manager-4ae9c17d9527#.d90oxolzk)
does a good job outlining many of these challenges, but their suggested approach
(use semantic versioning for dependency resolution) cannot be used by Bazel for
the general case.
