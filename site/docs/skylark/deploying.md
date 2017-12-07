---
layout: documentation
title: Deploying new Skylark rules
---

# Deploying new Skylark rules

This documentation is for Skylark rule writers who are planning to make their
rules available to others.

## Hosting and naming rules

New rules should go into their own GitHub repository under your organization.
Contact the [bazel-dev mailing list](https://groups.google.com/forum/#!forum/bazel-dev)
if you feel like your rules belong in the [bazelbuild](https://github.com/bazelbuild)
organization.

Repository names for Bazel rules are standardized on the following format:
*<organization>/rules_<name>*.
See [examples on GitHub](https://github.com/search?q=rules+bazel&type=Repositories).
For consistency, you must follow this same format when publishing your Bazel rules.

Make sure to use a descriptive GitHub repository description and `README.md`
title, example:

* Repository: *bazelbuild/rules_go*
* Repository description: *"Go rules for Bazel"*
* Repository tags: `golang`, `bazel`
* `README.md` header: *"Go rules for [Bazel](https://bazel.build)"*
(note the link to https://bazel.build which will guide users who are unfamiliar
with Bazel to the right place)

Rules can be grouped either by language (e.g., Scala) or some notion of platform
(e.g., Android.

## Rule content

Every rule repository should have a certain layout so that users can quickly
understand new rules.

For example, suppose we are writing new Skylark rules for the (make-believe)
mockascript language. We would have the following structure:

```
.travis.yml
README.md
WORKSPACE
mockascript/
  BUILD
  mockascript.bzl
tests/
  BUILD
  some_test.sh
  another_test.py
examples/
  BUILD
  bin.mocs
  lib.mocs
  test.mocs
```

### README.md

At the top level, there should be a README.md that contains (at least) what
users will need to copy-paste into their WORKSPACE file to use your rule.
In general, this will be a `git_repository` pointing to your GitHub repo and
a macro call that downloads/configures any tools your rule needs. For example,
for the [Go
rules](https://github.com/bazelbuild/rules_go/blob/master/README.md#setup), this
looks like:

```
git_repository(
    name = "io_bazel_rules_go",
    remote = "https://github.com/bazelbuild/rules_go.git",
    tag = "0.0.2",
)
load("@io_bazel_rules_go//go:def.bzl", "go_repositories")

go_repositories()
```

If your rules depend on another repository's rules, specify both in the
`README.md` (see the [Skydoc rules](https://github.com/bazelbuild/skydoc#setup),
which depend on the Sass rules, for an example of this).

### Tests

There should be tests that verify that the rules are working as expected. This
can either be in the standard location for the language the rules are for or a
`tests/` directory at the top level.

### Optional: Examples

It is useful to users to have an `examples/` directory that shows users a couple
of basic ways that the rules can be used.

## Testing

Set up Travis as described in their [getting started
docs](https://docs.travis-ci.com/user/getting-started/). Then add a
`.travis.yml` file to your repository with the following content:

```
# On trusty images, the Bazel apt repository can be used.
addons:
  apt:
    sources:
    - sourceline: 'deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8'
      key_url: 'https://bazel.build/bazel-release.pub.gpg'
    packages:
    - bazel

script:
  - bazel build //...
  - bazel test //...
```

Right now Bazel has to be compiled from source, as Travis does not support a
version of GCC that works with the precompiled Bazel binaries. Thus, the
`before_install` steps download the Bazel source, compile it, and "install" the
Bazel binary in /usr/bin.

If your repository is under the [bazelbuild organization](https://github.com/bazelbuild),
contact the [bazel-dev](https://groups.google.com/forum/#!forum/bazel-dev) list
to have it added to [ci.bazel.build](http://ci.bazel.build).

## Documentation

See the [Skydoc documentation](https://github.com/bazelbuild/skydoc) for
instructions on how to comment your rules so that documentation can be generated
automatically.

## FAQs

### Why can't we add our rule to the main Bazel GitHub repository?

We want to decouple rules from Bazel releases as much as possible. It's clearer
who owns individual rules, reducing the load on Bazel developers. For our users,
decoupling makes it easier to modify, upgrade, downgrade, and replace rules.
Contributing to rules can be lighter weight than contributing to Bazel -
depending on the rules -, including full submit access to the corresponding
GitHub repository. Getting submit access to Bazel itself is a much more involved
process.

The downside is a more complicated one-time installation process for our users:
they have to copy-paste a rule into their WORKSPACE file, as shown in the
README section above.

We used to have all of the Skylark rules in the Bazel repository (under
`//tools/build_rules` or `//tools/build_defs`). We still have a couple rules
there, but we are working on moving the remaining rules out.
