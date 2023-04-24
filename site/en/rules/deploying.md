Project: /_project.yaml
Book: /_book.yaml

# Deploying Rules

{% include "_buttons.html" %}

This page is for rule writers who are planning to make their rules available
to others.

## Hosting and naming rules

New rules should go into their own GitHub repository under your organization.
Start a thread on [GitHub](https://github.com/bazelbuild/bazel/discussions)
if you feel like your rules belong in the [bazelbuild](https://github.com/bazelbuild)
organization.

Repository names for Bazel rules are standardized on the following format:
`$ORGANIZATION/rules_$NAME`.
See [examples on GitHub](https://github.com/search?q=rules+bazel&type=Repositories).
For consistency, you must follow this same format when publishing your Bazel rules.

Make sure to use a descriptive GitHub repository description and `README.md`
title, example:

* Repository name: `bazelbuild/rules_go`
* Repository description: *Go rules for Bazel*
* Repository tags: `golang`, `bazel`
* `README.md` header: *Go rules for [Bazel](https://bazel.build)*
(note the link to https://bazel.build which will guide users who are unfamiliar
with Bazel to the right place)

Rules can be grouped either by language (such as Scala) or platform
(such as Android).

## Repository content

Every rule repository should have a certain layout so that users can quickly
understand new rules.

For example, when writing new rules for the (make-believe)
`mockascript` language, the rule repository would have the following structure:

```
/
  LICENSE
  README
  WORKSPACE
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

### WORKSPACE

In the project's `WORKSPACE`, you should define the name that users will use
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
workspace(name = "rules_mockascript")
```

### README

At the top level, there should be a `README` that contains (at least) what
users will need to copy-paste into their `WORKSPACE` file to use your rule.
In general, this will be a `http_archive` pointing to your GitHub release and
a macro call that downloads/configures any tools your rule needs. For example,
for the [Go
rules](https://github.com/bazelbuild/rules_go#setup), this
looks like:

```
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "rules_go",
    urls = ["https://github.com/bazelbuild/rules_go/releases/download/0.18.5/rules_go-0.18.5.tar.gz"],
    sha256 = "a82a352bffae6bee4e95f68a8d80a70e87f42c4741e6a448bec11998fcc82329",
)
load("@rules_go//go:deps.bzl", "go_rules_dependencies", "go_register_toolchains")
go_rules_dependencies()
go_register_toolchains()
```

If your rules depend on another repository's rules, specify that in the
rules documentation (for example, see the
[Skydoc rules](https://skydoc.bazel.build/docs/getting_started_stardoc.html),
which depend on the Sass rules), and provide a `WORKSPACE`
macro that will download all dependencies (see `rules_go` above).

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

Your rules might have external dependencies. To make depending on your rules
simpler, please provide a `WORKSPACE` macro that will declare dependencies on
those external dependencies. Do not declare dependencies of tests there, only
dependencies that rules require to work. Put development dependencies into the
`WORKSPACE` file.

Create a file named `<LANG>/repositories.bzl` and provide a single entry point
macro named `rules_<LANG>_dependencies`. Our directory will look as follows:

```
/
  mockascript/
    constraints/
      BUILD
    BUILD
    defs.bzl
    repositories.bzl
```


#### Registering toolchains

Your rules might also register toolchains. Please provide a separate `WORKSPACE`
macro that registers these toolchains. This way users can decide to omit the
previous macro and control dependencies manually, while still being allowed to
register toolchains.

Therefore add a `WORKSPACE` macro named `rules_<LANG>_toolchains` into
`<LANG>/repositories.bzl` file.

Note that in order to resolve toolchains in the analysis phase Bazel needs to
analyze all `toolchain` targets that are registered. Bazel will not need to
analyze all targets referenced by `toolchain.toolchain` attribute. If in order
to register toolchains you need to perform complex computation in the
repository, consider splitting the repository with `toolchain` targets from the
repository with `<LANG>_toolchain` targets. Former will be always fetched, and
the latter will only be fetched when user actually needs to build `<LANG>` code.


#### Release snippet

In your release announcement provide a snippet that your users can copy-paste
into their `WORKSPACE` file. This snippet in general will look as follows:

```
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "rules_<LANG>",
    urls = ["<url_to_the_release.zip"],
    sha256 = "4242424242",
)
load("@rules_<LANG>//<LANG>:repositories.bzl", "rules_<LANG>_dependencies", "rules_<LANG>_toolchains")
rules_<LANG>_dependencies()
rules_<LANG>_toolchains()
```


### Tests

There should be tests that verify that the rules are working as expected. This
can either be in the standard location for the language the rules are for or a
`tests/` directory at the top level.

### Examples (optional)

It is useful to users to have an `examples/` directory that shows users a couple
of basic ways that the rules can be used.

## Testing

Set up Travis as described in their [getting started
docs](https://docs.travis-ci.com/user/getting-started/). Then add a
`.travis.yml` file to your repository with the following content:

```
dist: xenial  # Ubuntu 16.04

# On trusty (or later) images, the Bazel apt repository can be used.
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

If your repository is under the [bazelbuild organization](https://github.com/bazelbuild),
you can [ask to add](https://github.com/bazelbuild/continuous-integration/issues/new?template=adding-your-project-to-bazel-ci.md&title=Request+to+add+new+project+%5BPROJECT_NAME%5D&labels=new-project)
it to [ci.bazel.build](http://ci.bazel.build).

## Documentation

See the [Stardoc documentation](https://github.com/bazelbuild/stardoc) for
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
they have to copy-paste a rule into their `WORKSPACE` file, as shown in the
`README.md` section above.

We used to have all of the rules in the Bazel repository (under
`//tools/build_rules` or `//tools/build_defs`). We still have a couple rules
there, but we are working on moving the remaining rules out.
