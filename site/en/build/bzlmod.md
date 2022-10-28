Project: /_project.yaml
Book: /_book.yaml

# Manage external dependencies with Bzlmod

*Bzlmod* is the codename of the new [external dependency](/docs/external) system
introduced in Bazel 5.0. It was introduced to address several pain points of the
old system that couldn't feasibly be fixed incrementally; see the
[Problem Statement section of the original design doc](https://docs.google.com/document/d/1moQfNcEIttsk6vYanNKIy3ZuK53hQUFq1b1r0rmsYVg/edit#heading=h.xxnnwabymk1v){: .external}
for more details.

In Bazel 5.0, Bzlmod is not turned on by default; the flag
`--experimental_enable_bzlmod` needs to be specified for the following to take
effect. As the flag name suggests, this feature is currently *experimental*;
APIs and behaviors may change until the feature officially launches.

To migrate your project to Bzlmod, follow the [Bzlmod Migration Guide](https://docs.google.com/document/d/1JtXIVnXyFZ4bmbiBCr5gsTH4-opZAFf5DMMb-54kES0/edit?usp=sharing).
You can also find example Bzlmod usages in the [examples](https://github.com/bazelbuild/examples/tree/main/bzlmod) repository.

## Bazel Modules {:#modules}

The old `WORKSPACE`-based external dependency system is centered around
*repositories* (or *repos*), created via *repository rules* (or *repo rules*).
While repos are still an important concept in the new system, *modules* are the
core units of dependency.

A *module* is essentially a Bazel project that can have multiple versions, each
of which publishes metadata about other modules that it depends on. This is
analogous to familiar concepts in other dependency management systems: a Maven
*artifact*, an npm *package*, a Cargo *crate*, a Go *module*, etc.

A module simply specifies its dependencies using `name` and `version` pairs,
instead of specific URLs in `WORKSPACE`. The dependencies are then looked up in
a [Bazel registry](#registries); by default, the
[Bazel Central Registry](#bazel-central-registry). In your workspace, each
module then gets turned into a repo.

### MODULE.bazel {:#module-bazel}

Every version of every module has a `MODULE.bazel` file declaring its
dependencies and other metadata. Here's a basic example:

```python
module(
    name = "my-module",
    version = "1.0",
)

bazel_dep(name = "rules_cc", version = "0.0.1")
bazel_dep(name = "protobuf", version = "3.19.0")
```

The `MODULE.bazel` file should be located at the root of the workspace directory
(next to the `WORKSPACE` file). Unlike with the `WORKSPACE` file, you don't need
to specify your *transitive* dependencies; instead, you should only specify
*direct* dependencies, and the `MODULE.bazel` files of your dependencies are
processed to discover transitive dependencies automatically.

The `MODULE.bazel` file is similar to `BUILD` files as it doesn't support any
form of control flow; it additionally forbids `load` statements. The directives
`MODULE.bazel` files support are:

*   [`module`](/rules/lib/globals#module), to specify metadata
    about the current module, including its name, version, and so on;
*   [`bazel_dep`](/rules/lib/globals#bazel_dep), to specify direct
    dependencies on other Bazel modules;
*   Overrides, which can only be used by the root module (that is, not by a
    module which is being used as a dependency) to customize the behavior of a
    certain direct or transitive dependency:
    *   [`single_version_override`](/rules/lib/globals#single_version_override)
    *   [`multiple_version_override`](/rules/lib/globals#multiple_version_override)
    *   [`archive_override`](/rules/lib/globals#archive_override)
    *   [`git_override`](/rules/lib/globals#git_override)
    *   [`local_path_override`](/rules/lib/globals#local_path_override)
*   Directives related to [module extensions](#module-extensions):
    *   [`use_extension`](/rules/lib/globals#use_extension)
    *   [`use_repo`](/rules/lib/globals#use_repo)

### Version format {:#version-format}

Bazel has a diverse ecosystem and projects use various versioning schemes. The
most popular by far is [SemVer](https://semver.org){: .external}, but there are
also prominent projects using different schemes such as
[Abseil](https://github.com/abseil/abseil-cpp/releases){: .external}, whose
versions are date-based, for example `20210324.2`).

For this reason, Bzlmod adopts a more relaxed version of the SemVer spec. The
differences include:

*   SemVer prescribes that the "release" part of the version must consist of 3
    segments: `MAJOR.MINOR.PATCH`. In Bazel, this requirement is loosened so
    that any number of segments is allowed.
*   In SemVer, each of the segments in the "release" part must be digits only.
    In Bazel, this is loosened to allow letters too, and the comparison
    semantics match the "identifiers" in the "prerelease" part.
*   Additionally, the semantics of major, minor, and patch version increases are
    not enforced. (However, see [compatibility level](#compatibility-level) for
    details on how we denote backwards compatibility.)

Any valid SemVer version is a valid Bazel module version. Additionally, two
SemVer versions `a` and `b` compare `a < b` iff the same holds when they're
compared as Bazel module versions.

### Version resolution {:#version-resolution}

The diamond dependency problem is a staple in the versioned dependency
management space. Suppose you have the following dependency graph:

```
       A 1.0
      /     \
   B 1.0    C 1.1
     |        |
   D 1.0    D 1.1
```

Which version of D should be used? To resolve this question, Bzlmod uses the
[Minimal Version Selection](https://research.swtch.com/vgo-mvs){: .external}
(MVS) algorithm introduced in the Go module system. MVS assumes that all new
versions of a module are backwards compatible, and thus simply picks the highest
version specified by any dependent (D 1.1 in our example). It's called "minimal"
because D 1.1 here is the *minimal* version that could satisfy our requirements;
even if D 1.2 or newer exists, we don't select them. This has the added benefit
that the version selection is *high-fidelity* and *reproducible*.

Version resolution is performed locally on your machine, not by the registry.

### Compatibility level {:#compatibility-level}

Note that MVS's assumption about backwards compatibility is feasible because it
simply treats backwards incompatible versions of a module as a separate module.
In terms of SemVer, that means A 1.x and A 2.x are considered distinct modules,
and can coexist in the resolved dependency graph. This is, in turn, made
possible by the fact that the major version is encoded in the package path in
Go, so there aren't any compile-time or linking-time conflicts.

In Bazel, we don't have such guarantees. Thus we need a way to denote the "major
version" number in order to detect backwards incompatible versions. This number
is called the *compatibility level*, and is specified by each module version in
its `module()` directive. With this information in hand, we can throw an error
when we detect that versions of the same module with different compatibility
levels exist in the resolved dependency graph.

### Repository names {:#repository-names}

In Bazel, every external dependency has a repository name. Sometimes, the same
dependency might be used via different repository names (for example, both
`@io_bazel_skylib` and `@bazel_skylib` mean
[Bazel skylib](https://github.com/bazelbuild/bazel-skylib){: .external}), or the same
repository name might be used for different dependencies in different projects.

In Bzlmod, repositories can be generated by Bazel modules and
[module extensions](#module-extensions). To resolve repository name conflicts,
we are embracing the [repository mapping](/docs/external#shadowing-dependencies)
mechanism in the new system. Here are two important concepts:

*   **Canonical repository name**: The globally unique repository name for each
    repository. This will be the directory name the repository lives in.
    <br>It's constructed as follows (**Warning**: the canonical name format is
    not an API you should depend on, it's subject to change at any time):

    *   For Bazel module repos: `{{ "<var>" }}module_name{{ "</var>" }}~{{ "<var>" }}version{{ "</var>" }}`
        <br> (<b>Example</b>. `@bazel_skylib~1.0.3`)
    *   For module extension repos: `{{ "<var>" }}module_name{{ "</var>" }}~{{ "<var>" }}version{{ "</var>" }}~{{ "<var>" }}extension_name{{ "</var>" }}~{{ "<var>" }}repo_name{{ "</var>" }}`
        <br>(<b>Example</b>. `@rules_cc~0.0.1~cc_configure~local_config_cc`)

*   **Apparent repository name**: The repository name to be used in the `BUILD` and
    `.bzl` files within a repo. The same dependency could have different apparent
    names in different repos.
    <br>It's determined as follows:

    *   For Bazel module repos: `{{ "<var>" }}module_name{{ "</var>" }}` by
        default, or the name specified by the `repo_name` attribute in
        [`bazel_dep`](/rules/lib/globals#bazel_dep).
    *   For module extension repos: repository name introduced via
        [`use_repo`](/rules/lib/globals#use_repo).

Every repository has a repository mapping dictionary of its direct dependencies,
which is a map from the apparent repository name to the canonical repository name.
We use the repository mapping to resolve the repository name when constructing a
label. Note that, there is no conflict of canonical repository names, and the
usages of apparent repository names can be discovered by parsing the `MODULE.bazel`
file, therefore conflicts can be easily caught and resolved without affecting
other dependencies.

### Strict deps {:#strict-deps}

The new dependency specification format allows us to perform stricter checks. In
particular, we now enforce that a module can only use repos created from its
direct dependencies. This helps prevent accidental and hard-to-debug breakages
when something in the transitive dependency graph changes.

Strict deps is implemented based on
[repository mapping](/docs/external#shadowing-dependencies). Basically, the
repository mapping for each repo contains all of its *direct dependencies*, any
other repository is not visible. Visible dependencies for each repository are
determined as follows:

*   A Bazel module repo can see all repos introduced in the `MODULE.bazel` file
    via [`bazel_dep`](/rules/lib/globals#bazel_dep) and
    [`use_repo`](/rules/lib/globals#use_repo).
*   A module extension repo can see all visible dependencies of the module that
    provides the extension, plus all other repos generated by the same module
    extension.

## Registries {:#registries}

Bzlmod discovers dependencies by requesting their information from Bazel
*registries*. A Bazel registry is simply a database of Bazel modules. The only
supported form of registries is an [*index registry*](#index-registry), which is
a local directory or a static HTTP server following a specific format. In the
future, we plan to add support for *single-module registries*, which are simply
git repos containing the source and history of a project.

### Index registry {:#index-registry}

An index registry is a local directory or a static HTTP server containing
information about a list of modules, including their homepage, maintainers, the
`MODULE.bazel` file of each version, and how to fetch the source of each
version. Notably, it does *not* need to serve the source archives itself.

An index registry must follow the format below:

*   `/bazel_registry.json`: A JSON file containing metadata for the registry like:
    * `mirrors`, specifying the list of mirrors to use for source archives.
    * `module_base_path`, specifying the base path for modules with
      `local_repository` type in the `source.json` file.
*   `/modules`: A directory containing a subdirectory for each module in this
    registry.
*   `/modules/$MODULE`: A directory containing a subdirectory for each version
    of this module, as well as the following file:
    *   `metadata.json`: A JSON file containing information about the module,
        with the following fields:
        *   `homepage`: The URL of the project's homepage.
        *   `maintainers`: A list of JSON objects, each of which corresponds to
            the information of a maintainer of the module *in the registry*.
            Note that this is not necessarily the same as the *authors* of the
            project.
        *   `versions`: A list of all the versions of this module to be found in
            this registry.
        *   `yanked_versions`: A list of *yanked* versions of this module. This
            is currently a no-op, but in the future, yanked versions will be
            skipped or yield an error.
*   `/modules/$MODULE/$VERSION`: A directory containing the following files:
    *   `MODULE.bazel`: The `MODULE.bazel` file of this module version.
    *   `source.json`: A JSON file containing information on how to fetch the
        source of this module version.
      * The default type is "archive" with the following fields:
          *   `url`: The URL of the source archive.
          *   `integrity`: The
              [Subresource Integrity](https://w3c.github.io/webappsec-subresource-integrity/#integrity-metadata-description){: .external}
              checksum of the archive.
          *   `strip_prefix`: A directory prefix to strip when extracting the
              source archive.
          *   `patches`: A list of strings, each of which names a patch file to
              apply to the extracted archive. The patch files are located under
              the `/modules/$MODULE/$VERSION/patches` directory.
          *   `patch_strip`: Same as the `--strip` argument of Unix patch.
      * The type can be changed to use a local path with these fields:
          *   `type`: `local_path`
          *   `path`: The local path to the repo, calculated as following:
              * If path is an absolute path, will be used as it is.
              * If path is a relative path and `module_base_path` is an absolute path,
                path is resolved to `<module_base_path>/<path>`
              * If path and `module_base_path` are both relative paths, path is
                resolved to `<registry_path>/<module_base_path>/<path>`.
                Registry must be hosted locally and used by `--registry=file://<registry_path>`.
                Otherwise, Bazel will throw an error.
    *   `patches/`: An optional directory containing patch files, only used when `source.json` has "archive" type.

### Bazel Central Registry {:#bazel-central-registry}

Bazel Central Registry (BCR) is an index registry located at
[bcr.bazel.build](https://bcr.bazel.build){: .external}. Its contents
are backed by the GitHub repo
[`bazelbuild/bazel-central-registry`](https://github.com/bazelbuild/bazel-central-registry){: .external}.

The BCR is maintained by the Bazel community; contributors are welcome to submit
pull requests. See
[Bazel Central Registry Policies and Procedures](https://docs.google.com/document/d/1ReuBBp4EHnsuvcpfXM6ITDmP2lrOu8DGlePMUKvDnXM/edit?usp=sharing){: .external}.

In addition to following the format of a normal index registry, the BCR requires
a `presubmit.yml` file for each module version
(`/modules/$MODULE/$VERSION/presubmit.yml`). This file specifies a few essential
build and test targets that can be used to sanity-check the validity of this
module version, and is used by the BCR's CI pipelines to ensure interoperability
between modules in the BCR.

### Selecting registries {:#selecting-registries}

The repeatable Bazel flag `--registry` can be used to specify the list of
registries to request modules from, so you could set up your project to fetch
dependencies from a third-party or internal registry. Earlier registries take
precedence. For convenience, you can put a list of `--registry` flags in the
`.bazelrc` file of your project.

Note: If your registry is hosted on GitHub (for example, as a fork of
`bazelbuild/bazel-central-registry`) then your `--registry` value needs a raw
GitHub address under `raw.githubusercontent.com`. For example, on the `main`
branch of the `my-org` fork, you would set
`--registry=https://raw.githubusercontent.com/my-org/bazel-central-registry/main/`.

## Module Extensions {:#module-extensions}

Module extensions allow you to extend the module system by reading input data
from modules across the dependency graph, performing necessary logic to resolve
dependencies, and finally creating repos by calling repo rules. They are similar
in function to today's `WORKSPACE` macros, but are more suited in the world of
modules and transitive dependencies.

Module extensions are defined in `.bzl` files, just like repo rules or
`WORKSPACE` macros. They're not invoked directly; rather, each module can
specify pieces of data called *tags* for extensions to read. Then, after module
version resolution is done, module extensions are run. Each extension is run
once after module resolution (still before any build actually happens), and
gets to read all the tags belonging to it across the entire dependency graph.

```
          [ A 1.1                ]
          [   * maven.dep(X 2.1) ]
          [   * maven.pom(...)   ]
              /              \
   bazel_dep /                \ bazel_dep
            /                  \
[ B 1.2                ]     [ C 1.0                ]
[   * maven.dep(X 1.2) ]     [   * maven.dep(X 2.1) ]
[   * maven.dep(Y 1.3) ]     [   * cargo.dep(P 1.1) ]
            \                  /
   bazel_dep \                / bazel_dep
              \              /
          [ D 1.4                ]
          [   * maven.dep(Z 1.4) ]
          [   * cargo.dep(Q 1.1) ]
```

In the example dependency graph above, `A 1.1` and `B 1.2` etc are Bazel modules;
you can think of each one as a `MODULE.bazel` file. Each module can specify some
tags for module extensions; here some are specified for the extension "maven",
and some are specified for "cargo". When this dependency graph is finalized (for
example, maybe `B 1.2` actually has a `bazel_dep` on `D 1.3` but got upgraded to
`D 1.4` due to `C`), the extensions "maven" is run, and it gets to read all the
`maven.*` tags, using information therein to decide which repos to create.
Similarly for the "cargo" extension.

### Extension usage {:#extension-usage}

Extensions are hosted in Bazel modules themselves, so to use an extension in
your module, you need to first add a `bazel_dep` on that module, and then call
the [`use_extension`](/rules/lib/globals#use_extension) built-in
function to bring it into scope. Consider the following example, a snippet from
a `MODULE.bazel` file to use a hypothetical "maven" extension defined in the
`rules_jvm_external` module:

```python
bazel_dep(name = "rules_jvm_external", version = "1.0")
maven = use_extension("@rules_jvm_external//:extensions.bzl", "maven")
```

After bringing the extension into scope, you can then use the dot-syntax to
specify tags for it. Note that the tags need to follow the schema defined by the
corresponding *tag classes* (see [extension definition](#extension-definition)
below). Here's an example specifying some `maven.dep` and `maven.pom` tags.

```python
maven.dep(coord="org.junit:junit:3.0")
maven.dep(coord="com.google.guava:guava:1.2")
maven.pom(pom_xml="//:pom.xml")
```

If the extension generates repos that you want to use in your module, use the
[`use_repo`](/rules/lib/globals#use_repo) directive to declare
them. This is to satisfy the strict deps condition and avoid local repo name
conflict.

```python
use_repo(
    maven,
    "org_junit_junit",
    guava="com_google_guava_guava",
)
```

The repos generated by an extension are part of its API, so from the tags you
specified, you should know that the "maven" extension is going to generate a
repo called "org_junit_junit", and one called "com_google_guava_guava". With
`use_repo`, you can optionally rename them in the scope of your module, like to
"guava" here.

### Extension definition {:#extension-definition}

Module extensions are defined similarly to repo rules, using the
[`module_extension`](/rules/lib/globals#module_extension) function.
Both have an implementation function; but while repo rules have a number of
attributes, module extensions have a number of
[`tag_class`es](/rules/lib/globals#tag_class), each of which has a
number of attributes. The tag classes define schemas for tags used by this
extension. Continuing our example of the hypothetical "maven" extension above:

```python
# @rules_jvm_external//:extensions.bzl
maven_dep = tag_class(attrs = {"coord": attr.string()})
maven_pom = tag_class(attrs = {"pom_xml": attr.label()})
maven = module_extension(
    implementation=_maven_impl,
    tag_classes={"dep": maven_dep, "pom": maven_pom},
)
```

These declarations make it clear that `maven.dep` and `maven.pom` tags can be
specified, using the attribute schema defined above.

The implementation function is similar to a `WORKSPACE` macro, except that it
gets a [`module_ctx`](/rules/lib/module_ctx) object, which grants
access to the dependency graph and all pertinent tags. The implementation
function should then call repo rules to generate repos:

```python
# @rules_jvm_external//:extensions.bzl
load("//:repo_rules.bzl", "maven_single_jar")
def _maven_impl(ctx):
  coords = []
  for mod in ctx.modules:
    coords += [dep.coord for dep in mod.tags.dep]
  output = ctx.execute(["coursier", "resolve", coords])  # hypothetical call
  repo_attrs = process_coursier(output)
  [maven_single_jar(**attrs) for attrs in repo_attrs]
```

In the example above, we go through all the modules in the dependency graph
(`ctx.modules`), each of which is a
[`bazel_module`](/rules/lib/bazel_module) object whose `tags` field
exposes all the `maven.*` tags on the module. Then we invoke the CLI utility
Coursier to contact Maven and perform resolution. Finally, we use the resolution
result to create a number of repos, using the hypothetical `maven_single_jar`
repo rule.

## External links

*   [Bazel External Dependencies Overhaul](https://docs.google.com/document/d/1moQfNcEIttsk6vYanNKIy3ZuK53hQUFq1b1r0rmsYVg/edit){: .external}
    (original Bzlmod design doc)
*   [Bazel Central Registry Policies and Procedures](https://docs.google.com/document/d/1ReuBBp4EHnsuvcpfXM6ITDmP2lrOu8DGlePMUKvDnXM/edit?usp=sharing){: .external}
*   [Bazel Central Registry GitHub repo](https://github.com/bazelbuild/bazel-central-registry){: .external}
*   [BazelCon 2021 talk on Bzlmod](https://www.youtube.com/watch?v=TxOCKtU39Fs){: .external}
