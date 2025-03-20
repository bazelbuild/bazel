Project: /_project.yaml
Book: /_book.yaml

# Deploying Rules

{% include "_buttons.html" %}

This page is for rule writers who are planning to make their rules available
to others.

We recommend you start a new ruleset from the template repository:
https://github.com/bazel-contrib/rules-template
That template follows the recommendations below, and includes API documentation generation
and sets up a CI/CD pipeline to make it trivial to distribute your ruleset.

## Hosting and naming rules

New rules should go into their own GitHub repository under your organization.
Start a thread on [GitHub](https://github.com/bazelbuild/bazel/discussions)
if you feel like your rules belong in the [bazelbuild](https://github.com/bazelbuild)
organization.

Repository names for Bazel rules are standardized on the following format:
`$ORGANIZATION/rules_$NAME`.
See [examples on GitHub](https://github.com/search?q=rules+bazel&type=Repositories).
For consistency, you should follow this same format when publishing your Bazel rules.

Make sure to use a descriptive GitHub repository description and `README.md`
title, example:

* Repository name: `bazelbuild/rules_go`
* Repository description: *Go rules for Bazel*
* Repository tags: `golang`, `bazel`
* `README.md` header: *Go rules for [Bazel](https://bazel.build)*
(note the link to https://bazel.build which will guide users who are unfamiliar
with Bazel to the right place)

Rules can be grouped either by language (such as Scala), runtime platform
(such as Android), or framework (such as Spring).

## Repository content

Every rule repository should have a certain layout so that users can quickly
understand new rules.

For example, when writing new rules for the (make-believe)
`mockascript` language, the rule repository would have the following structure:

```
/
  LICENSE
  README
  MODULE.bazel
  mockascript/
    constraints/
      BUILD
    runfiles/
      BUILD
      runfiles.mocs
    BUILD
    defs.bzl
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

### MODULE.bazel

In the project's `MODULE.bazel`, you should define the name that users will use
to reference your rules. If your rules belong to the
[bazelbuild](https://github.com/bazelbuild) organization, you must use
`rules_<lang>` (such as `rules_mockascript`). Otherwise, you should name your
repository `<org>_rules_<lang>` (such as `build_stack_rules_proto`). Please
start a thread on [GitHub](https://github.com/bazelbuild/bazel/discussions)
if you feel like your rules should follow the convention for rules in the
[bazelbuild](https://github.com/bazelbuild) organization.

In the following sections, assume the repository belongs to the
[bazelbuild](https://github.com/bazelbuild) organization.

```
module(name = "rules_mockascript")
```

### README

At the top level, there should be a `README` that contains a brief description
of your ruleset, and the API users should expect.

### Rules

Often times there will be multiple rules provided by your repository. Create a
directory named by the language and provide an entry point - `defs.bzl` file
exporting all rules (also include a `BUILD` file so the directory is a package).
For `rules_mockascript` that means there will be a directory named
`mockascript`, and a `BUILD` file and a `defs.bzl` file inside:

```
/
  mockascript/
    BUILD
    defs.bzl
```

### Constraints

If your rule defines
[toolchain](/extending/toolchains) rules,
it's possible that you'll need to define custom `constraint_setting`s and/or
`constraint_value`s. Put these into a `//<LANG>/constraints` package. Your
directory structure will look like this:

```
/
  mockascript/
    constraints/
      BUILD
    BUILD
    defs.bzl
```

Please read
[github.com/bazelbuild/platforms](https://github.com/bazelbuild/platforms)
for best practices, and to see what constraints are already present, and
consider contributing your constraints there if they are language independent.
Be mindful of introducing custom constraints, all users of your rules will
use them to perform platform specific logic in their `BUILD` files (for example,
using [selects](/reference/be/functions#select)).
With custom constraints, you define a language that the whole Bazel ecosystem
will speak.

### Runfiles library

If your rule provides a standard library for accessing runfiles, it should be
in the form of a library target located at `//<LANG>/runfiles` (an abbreviation
of `//<LANG>/runfiles:runfiles`). User targets that need to access their data
dependencies will typically add this target to their `deps` attribute.

### Repository rules

#### Dependencies

Your rules might have external dependencies, which you'll need to specify in
your MODULE.bazel file.

#### Registering toolchains

Your rules might also register toolchains, which you can also specify in the
MODULE.bazel file.

Note that in order to resolve toolchains in the analysis phase Bazel needs to
analyze all `toolchain` targets that are registered. Bazel will not need to
analyze all targets referenced by `toolchain.toolchain` attribute. If in order
to register toolchains you need to perform complex computation in the
repository, consider splitting the repository with `toolchain` targets from the
repository with `<LANG>_toolchain` targets. Former will be always fetched, and
the latter will only be fetched when user actually needs to build `<LANG>` code.


#### Release snippet

In your release announcement provide a snippet that your users can copy-paste
into their `MODULE.bazel` file. This snippet in general will look as follows:

```
bazel_dep(name = "rules_<LANG>", version = "<VERSION>")
```


### Tests

There should be tests that verify that the rules are working as expected. This
can either be in the standard location for the language the rules are for or a
`tests/` directory at the top level.

### Examples (optional)

It is useful to users to have an `examples/` directory that shows users a couple
of basic ways that the rules can be used.

## CI/CD

Many rulesets use GitHub Actions. See the configuration used in the [rules-template](https://github.com/bazel-contrib/rules-template/tree/main/.github/workflows) repo, which are simplified using a "reusable workflow" hosted in the bazel-contrib
org. `ci.yaml` runs tests on each PR and `main` comit, and `release.yaml` runs anytime you push a tag to the repository.
See comments in the rules-template repo for more information.

If your repository is under the [bazelbuild organization](https://github.com/bazelbuild),
you can [ask to add](https://github.com/bazelbuild/continuous-integration/issues/new?template=adding-your-project-to-bazel-ci.md&title=Request+to+add+new+project+%5BPROJECT_NAME%5D&labels=new-project)
it to [ci.bazel.build](http://ci.bazel.build).

## Documentation

See the [Stardoc documentation](https://github.com/bazelbuild/stardoc) for
instructions on how to comment your rules so that documentation can be generated
automatically.

The [rules-template docs/ folder](https://github.com/bazel-contrib/rules-template/tree/main/docs)
shows a simple way to ensure the Markdown content in the `docs/` folder is always up-to-date
as Starlark files are updated.

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
they have to add a dependency on your ruleset in their `MODULE.bazel` file.

We used to have all of the rules in the Bazel repository (under
`//tools/build_rules` or `//tools/build_defs`). We still have a couple rules
there, but we are working on moving the remaining rules out.
