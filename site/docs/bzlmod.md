---
layout: documentation
title: Bzlmod user guide
category: getting-started
---

# Bzlmod User Guide

*Bzlmod* is the codename of the new [external dependency](external.md) system
introduced in Bazel 5.0. It was introduced to address several pain points of the
old system that couldn't feasibly be fixed incrementally; see the
[Problem Statement section of the original design doc](https://docs.google.com/document/d/1moQfNcEIttsk6vYanNKIy3ZuK53hQUFq1b1r0rmsYVg/edit#heading=h.xxnnwabymk1v)
for more details.

In Bazel 5.0, Bzlmod is not turned on by default; the flag
`--experimental_enable_bzlmod` needs to be specified for the following to take
effect. As the flag name suggests, this feature is currently *experimental*;
APIs and behaviors may change until the feature officially launches.

## Bazel Modules

The old WORKSPACE-based external dependency system is centered around
*repositories* (or *repos*), created via *repository rules* (or *repo rules*).
While repos are still an important concept in the new system, *modules* are the
core units of dependency.

A *module* is essentially a Bazel project that can have multiple versions, each
of which publishes metadata about other modules that it depends on. This is
analogous to familiar concepts in other dependency management systems: a Maven
*artifact*, an npm *package*, a Cargo *crate*, a Go *module*, etc.

A module simply specifies its dependencies using "name" and "version" pairs, as
opposed to specific URLs in WORKSPACE. The dependencies are then looked up in a
[Bazel registry](#registries); by default, the
[Bazel Central Registry](#bazel-central-registry). In your workspace, each
module then gets turned into a repo.

### MODULE.bazel

Every version of every module has a MODULE.bazel file declaring its dependencies
and other metadata. Here's a basic example:

```python
module(
    name = "my-module",
    version = "1.0",
)

bazel_dep(name = "rules_cc", version = "0.0.1")
bazel_dep(name = "protobuf", version = "3.19.0")
```

The MODULE.bazel file should be located at the root of the workspace directory
(next to the WORKSPACE file). Unlike with the WORKSPACE file, you don't need to
specify your *transitive* dependencies; instead, you should only specify
*direct* dependencies, and the MODULE.bazel files of your dependencies will be
processed to discover transitive dependencies automatically.

The MODULE.bazel file is similar to BUILD files in that it doesn't support any
form of control flow; it additionally forbids `load` statements. The directives
it supports are:

*   [`module`](skylark/lib/globals.html#module), to specify metadata about the
    current module, including its name, version, and so on;
*   [`bazel_dep`](skylark/lib/globals.html#bazel_dep), to specify direct
    dependencies on other Bazel modules;
*   Overrides, which can only be used by the root module (that is, not by a
    module which is being used as a dependency) to customize the behavior of a
    certain direct or transitive dependency:
    *   [`single_version_override`](skylark/lib/globals.html#single_version_override)
    *   [`multiple_version_override`](skylark/lib/globals.html#multiple_version_override)
    *   [`archive_override`](skylark/lib/globals.html#archive_override)
    *   [`git_override`](skylark/lib/globals.html#git_override)
    *   [`local_path_override`](skylark/lib/globals.html#local_path_override)
*   Directives related to [module extensions](#module-extensions):
    *   [`use_extension`](skylark/lib/globals.html#use_extension)
    *   [`use_repo`](skylark/lib/globals.html#use_repo)

### Version format

Bazel has a diverse ecosystem and projects are using all kinds of versioning
schemes. The most popular by far is [SemVer](https://semver.org), but there are
also prominent projects using different schemes such as
[Abseil](https://github.com/abseil/abseil-cpp/releases), whose versions are
date-based, for example `20210324.2`).

For this reason, Bzlmod adopts a more relaxed version of the SemVer spec, in
particular allowing any number of sequences of digits in the "release" part of
the version (instead of exactly 3 as SemVer prescribes: `MAJOR.MINOR.PATCH`).
Additionally, the semantics of major, minor, and patch version increases are not
enforced. (However, see [compatibility level](#compatibility-level) for details
on how we denote backwards compatibility.) Other parts of the SemVer spec, such
as a hyphen denoting a prerelease version, are not modified.

Note that this version format is subject to change before the official launch of
Bzlmod.

### Version resolution

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
[Minimal Version Selection](https://research.swtch.com/vgo-mvs) (MVS) algorithm
introduced in the Go module system. MVS assumes that all new versions of a
module are backwards compatible, and thus simply picks the highest version
specified by any dependent (so D 1.1 in our example). It's called "minimal"
because D 1.1 here is the *minimal* version that could satisfy our requirements;
even if D 1.2 or newer exists, we don't select them. This has the added benefit
that the version selection is *high-fidelity* and *reproducible*.

Version resolution is performed wholly locally on your machine, not by the
registry.

### Compatibility level

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

### Repository names

In Bazel, every external dependency has a repository name. Sometimes, the same
dependency might be used via different repository names (eg. Both
`@io_bazel_skylib` and `@bazel_skylib` mean
[Bazel skylib](https://github.com/bazelbuild/bazel-skylib)), or the same
repository name might be used for different dependencies in different projects.

In Bzlmod, repositories can be generated by Bazel modules and
[module extensions](#module-extensions). To resolve repository name conflicts,
we are embracing the [repository mapping](external.md#shadowing-dependencies)
mechanism in the new system. Here are two important concepts:

*   **Canonical repository name**: The globally unique repository name for each
    repository. This will be the directory name the repository lives in.
    <br>It's constructed as follows (**Warning**: the canonical name format is
    not an API you should depend on, it's subject to change at any time):

    *   For Bazel module repos: `<module name>.<version>`
        <br> (eg. `@bazel_skylib.1.0.3`)
    *   For module extension repos: `<module name>.<version>.<extension name>.<repo name>`
        <br>(eg. `@rules_cc.0.0.1.cc_configure.local_config_cc`)

*   **Local repository name**: The repository name to be used in the BUILD and
    bzl files within a repo. The same dependency could have different local
    names for different repos.
    <br>It's determined as follows:

    *   For Bazel module repos: `<module name>` by default, or the name
        specified by the `repo_name` attribute in
        [`bazel_dep`](skylark/lib/globals.html#bazel_dep).
    *   For module extension repos: repository name introduced via
        [`use_repo`](skylark/lib/globals.html#use_repo).

Every repository has a repository mapping dictionary of its direct dependencies,
which is a map from the local repository name to the canonical repository name.
We use the repository mapping to resolve the repository name when constructing a
label. Note that, there is no conflict of canonical repository names, and the
usages of local repository names can be discovered by parsing the MODULE.bazel
file, therefore conflicts can be easily caught and resolved without affecting
other dependencies.

### Strict deps

The new dependency specification format allows us to perform stricter checks. In
particular, we now enforce that a module can only use repos created from its
direct dependencies. This helps prevent accidental and hard-to-debug breakages
when something in the transitive dependency graph changes.

Strict deps is implemented based on
[repository mapping](external.md#shadowing-dependencies). Basically, the
repository mapping for each repo contains all of its *direct dependencies*, any
other repository is not visible. Visible dependencies for each repository are
determined as follows:

*   A Bazel module repo can see all repos introduced in the MODULE.bazel file
    via [`bazel_dep`](skylark/lib/globals.html#bazel_dep) and
    [`use_repo`](skylark/lib/globals.html#use_repo).
*   A module extension repo can see all visible dependencies of the module that
    provides the extension, plus all other repos generated by the same module
    extension.

## Registries

Bzlmod discovers dependencies by requesting their information from Bazel
*registries*. A Bazel registry is simply a database of Bazel modules. The only
supported form of registries is an [*index registry*](#index-registry), which is
a local directory or a static HTTP server following a specific format. In the
future, we plan to add support for *single-module registries*, which are simply
git repos containing the source and history of a project.

### Index registry

An index registry is a local directory or a static HTTP server containing
information about a list of modules, including their homepage, maintainers, the
MODULE.bazel file of each version, and how to fetch the source of each version.
Notably, it does *not* need to serve the source archives itself.

An index registry must follow the format below:

*   `/bazel_registry.json`: A JSON file containing metadata for the registry.
    Currently, it only has one key, `mirrors`, specifying the list of mirrors to
    use for source archives.
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
    *   `MODULE.bazel`: The MODULE.bazel file of this module version.
    *   `source.json`: A JSON file containing information on how to fetch the
        source of this module version, with the following fields:
        *   `url`: The URL of the source archive.
        *   `integrity`: The
            [Subresource Integrity](https://w3c.github.io/webappsec-subresource-integrity/#integrity-metadata-description)
            checksum of the archive.
        *   `strip_prefix`: A directory prefix to strip when extracting the
            source archive.
        *   `patches`: A list of strings, each of which names a patch file to
            apply to the extracted archive. The patch files are located under
            the `/modules/$MODULE/$VERSION/patches` directory.
        *   `patch_strip`: Same as the `--strip` argument of Unix patch.
    *   `patches/`: An optional directory containing patch files.

### Bazel Central Registry

By default, all dependencies are requested from the *Bazel Central Registry*
(BCR), which is an index registry located at
[registry.bazel.build](https://registry.bazel.build). Its contents are backed by
the GitHub repo
[bazelbuild/bazel-central-registry](https://github.com/bazelbuild/bazel-central-registry).

The BCR is maintained by the Bazel community; contributors are welcome to submit
PRs. See
[Bazel Central Registry Policies and Procedures](https://docs.google.com/document/d/1ReuBBp4EHnsuvcpfXM6ITDmP2lrOu8DGlePMUKvDnXM/edit?usp=sharing).

In addition to following the format of a normal index registry, the BCR requires
a `presubmit.yml` file for each module version
(`/modules/$MODULE/$VERSION/presubmit.yml`). This file specifies a few essential
build and test targets that can be used to sanity-check the validity of this
module version, and is used by the BCR's CI pipelines to ensure interoperability
between modules in the BCR.

### Selecting registries

The repeatable Bazel flag `--registry` can be used to specify the list of
registries to request modules from, so you could set up your project to fetch
dependencies from a third-party or internal registry. Earlier registries take
precedence. For convenience, you can put a list of `--registry` flags in the
.bazelrc file of your project.

## Module Extensions

Module extensions allow you to extend the module system. TBA...

## External links

*   [Bazel External Dependencies Overhaul](https://docs.google.com/document/d/1moQfNcEIttsk6vYanNKIy3ZuK53hQUFq1b1r0rmsYVg/edit)
    (original Bzlmod design doc)
*   [Bazel Central Registry Policies and Procedures](https://docs.google.com/document/d/1ReuBBp4EHnsuvcpfXM6ITDmP2lrOu8DGlePMUKvDnXM/edit?usp=sharing)
*   [Bazel Central Registry GitHub repo](https://github.com/bazelbuild/bazel-central-registry)
    <!-- TODO: Add BazelCon talk video -->
