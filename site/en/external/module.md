Project: /_project.yaml
Book: /_book.yaml

# Bazel modules

{% include "_buttons.html" %}

A Bazel **module** is a Bazel project that can have multiple versions, each of
which publishes metadata about other modules that it depends on. This is
analogous to familiar concepts in other dependency management systems, such as a
Maven *artifact*, an npm *package*, a Go *module*, or a Cargo *crate*.

A module must have a `MODULE.bazel` file at its repo root. This file is the
module's manifest, declaring its name, version, list of direct dependencies, and
other information. For a basic example:

```python
module(name = "my-module", version = "1.0")

bazel_dep(name = "rules_cc", version = "0.0.1")
bazel_dep(name = "protobuf", version = "3.19.0")
```

See the [full list](/rules/lib/globals/module) of directives available in
`MODULE.bazel` files.

To perform module resolution, Bazel starts by reading the root module's
`MODULE.bazel` file, and then repeatedly requests any dependency's
`MODULE.bazel` file from a [Bazel registry](/external/registry) until it
discovers the entire dependency graph.

By default, Bazel then [selects](#version-selection) one version of each module
to use. Bazel represents each module with a repo, and consults the registry
again to learn how to define each of the repos.

## Version format {:#version-format}

Bazel has a diverse ecosystem and projects use various versioning schemes. The
most popular by far is [SemVer](https://semver.org){: .external}, but there are
also prominent projects using different schemes such as
[Abseil](https://github.com/abseil/abseil-cpp/releases){: .external}, whose
versions are date-based, for example `20210324.2`).

For this reason, Bazel adopts a more relaxed version of the SemVer spec. The
differences include:

*   SemVer prescribes that the "release" part of the version must consist of 3
    segments: `MAJOR.MINOR.PATCH`. In Bazel, this requirement is loosened so
    that any number of segments is allowed.
*   In SemVer, each of the segments in the "release" part must be digits only.
    In Bazel, this is loosened to allow letters too, and the comparison
    semantics match the "identifiers" in the "prerelease" part.
*   Additionally, the semantics of major, minor, and patch version increases are
    not enforced. However, see [compatibility level](#compatibility_level) for
    details on how we denote backwards compatibility.

Any valid SemVer version is a valid Bazel module version. Additionally, two
SemVer versions `a` and `b` compare `a < b` if and only if the same holds when
they're compared as Bazel module versions.

Finally, to learn more about module versioning, [see the `MODULE.bazel`
FAQ](faq#module-versioning-best-practices).

## Version selection {:#version-selection}

Consider the diamond dependency problem, a staple in the versioned dependency
management space. Suppose you have the dependency graph:

```
       A 1.0
      /     \
   B 1.0    C 1.1
     |        |
   D 1.0    D 1.1
```

Which version of `D` should be used? To resolve this question, Bazel uses the
[Minimal Version Selection](https://research.swtch.com/vgo-mvs){: .external}
(MVS) algorithm introduced in the Go module system. MVS assumes that all new
versions of a module are backwards compatible, and so picks the highest version
specified by any dependent (`D 1.1` in our example). It's called "minimal"
because `D 1.1` is the earliest version that could satisfy our requirements —
even if `D 1.2` or newer exists, we don't select them. Using MVS creates a
version selection process that is *high-fidelity* and *reproducible*.

### Yanked versions

The registry can declare certain versions as *yanked* if they should be avoided
(such as for security vulnerabilities). Bazel throws an error when selecting a
yanked version of a module. To fix this error, either upgrade to a newer,
non-yanked version, or use the
[`--allow_yanked_versions`](/reference/command-line-reference#flag--allow_yanked_versions)
flag to explicitly allow the yanked version.

## Compatibility level

In Go, MVS's assumption about backwards compatibility works because it treats
backwards incompatible versions of a module as a separate module. In terms of
SemVer, that means `A 1.x` and `A 2.x` are considered distinct modules, and can
coexist in the resolved dependency graph. This is, in turn, made possible by
encoding the major version in the package path in Go, so there aren't any
compile-time or linking-time conflicts. Bazel, however, cannot provide such
guarantees because it follows [a relaxed version of SemVer](#version-format).

Thus, Bazel needs the equivalent of the SemVer major version number to detect
backwards incompatible ("breaking") versions. This number is called the
*compatibility level*, and is specified by each module version in its
[`module()`](/rule/lib/globals/module#module) directive. With this information,
Bazel can throw an error if it detects that versions of the _same module_ with
_different compatibility levels_ exist in the resolved dependency graph.

Finally, incrementing the compatibility level can be disruptive to the users.
To learn more about when and how to increment it, [check the `MODULE.bazel`
FAQ](faq#incrementing-compatibility-level).

## Overrides

Specify overrides in the `MODULE.bazel` file to alter the behavior of Bazel
module resolution. Only the root module's overrides take effect — if a module is
used as a dependency, its overrides are ignored.

Each override is specified for a certain module name, affecting all of its
versions in the dependency graph. Although only the root module's overrides take
effect, they can be for transitive dependencies that the root module does not
directly depend on.

### Single-version override

The [`single_version_override`](/rules/lib/globals/module#single_version_override)
serves multiple purposes:

*   With the `version` attribute, you can pin a dependency to a specific
    version, regardless of which versions of the dependency are requested in the
    dependency graph.
*   With the `registry` attribute, you can force this dependency to come from a
    specific registry, instead of following the normal [registry
    selection](/external/registry#selecting_registries) process.
*   With the `patch*` attributes, you can specify a set of patches to apply to
    the downloaded module.

These attributes are all optional and can be mixed and matched with each other.

### Multiple-version override

A [`multiple_version_override`](/rules/lib/globals/module#multiple_version_override)
can be specified to allow multiple versions of the same module to coexist in the
resolved dependency graph.

You can specify an explicit list of allowed versions for the module, which must
all be present in the dependency graph before resolution — there must exist
*some* transitive dependency depending on each allowed version. After
resolution, only the allowed versions of the module remain, while Bazel upgrades
other versions of the module to the nearest higher allowed version at the same
compatibility level. If no higher allowed version at the same compatibility
level exists, Bazel throws an error.

For example, if versions `1.1`, `1.3`, `1.5`, `1.7`, and `2.0` exist in the
dependency graph before resolution and the major version is the compatibility
level:

*   A multiple-version override allowing `1.3`, `1.7`, and `2.0` results in
    `1.1` being upgraded to `1.3`, `1.5` being upgraded to `1.7`, and other
    versions remaining the same.
*   A multiple-version override allowing `1.5` and `2.0` results in an error, as
    `1.7` has no higher version at the same compatibility level to upgrade to.
*   A multiple-version override allowing `1.9` and `2.0` results in an error, as
    `1.9` is not present in the dependency graph before resolution.

Additionally, users can also override the registry using the `registry`
attribute, similarly to single-version overrides.

### Non-registry overrides

Non-registry overrides completely remove a module from version resolution. Bazel
does not request these `MODULE.bazel` files from a registry, but instead from
the repo itself.

Bazel supports the following non-registry overrides:

*   [`archive_override`](/rules/lib/globals/module#archive_override)
*   [`git_override`](/rules/lib/globals/module#git_override)
*   [`local_path_override`](/rules/lib/globals/module#local_path_override)

Note that setting a version value in the source archive `MODULE.bazel` can have
downsides when the module is being overridden with a non-registry override. To
learn more about this [see the `MODULE.bazel`
FAQ](faq#module-versioning-best-practices).

## Define repos that don't represent Bazel modules {:#use_repo_rule}

With `bazel_dep`, you can define repos that represent other Bazel modules.
Sometimes there is a need to define a repo that does _not_ represent a Bazel
module; for example, one that contains a plain JSON file to be read as data.

In this case, you could use the [`use_repo_rule`
directive](/rules/lib/globals/module#use_repo_rule) to directly define a repo
by invoking a repo rule. This repo will only be visible to the module it's
defined in.

Under the hood, this is implemented using the same mechanism as [module
extensions](/external/extension), which lets you define repos with more
flexibility.

## Repository names and strict deps

The [apparent name](/external/overview#apparent-repo-name) of a repo backing a
module to its direct dependents defaults to its module name, unless the
`repo_name` attribute of the [`bazel_dep`](/rules/lib/globals/module#bazel_dep)
directive says otherwise. Note that this means a module can only find its direct
dependencies. This helps prevent accidental breakages due to changes in
transitive dependencies.

The [canonical name](/external/overview#canonical-repo-name) of a repo backing a
module is either `{{ "<var>" }}module_name{{ "</var>" }}+{{ "<var>" }}version{{
"</var>" }}` (for example, `bazel_skylib+1.0.3`) or `{{ "<var>" }}module_name{{
"</var>" }}+` (for example, `bazel_features+`), depending on whether there are
multiple versions of the module in the entire dependency graph (see
[`multiple_version_override`](/rules/lib/globals/module#multiple_version_override)).
Note that **the canonical name format** is not an API you should depend on and
**is subject to change at any time**. Instead of hard-coding the canonical name,
use a supported way to get it directly from Bazel:

*    In BUILD and `.bzl` files, use
     [`Label.repo_name`](/rules/lib/builtins/Label#repo_name) on a `Label` instance
     constructed from a label string given by the apparent name of the repo, e.g.,
     `Label("@bazel_skylib").repo_name`.
*    When looking up runfiles, use
     [`$(rlocationpath ...)`](https://bazel.build/reference/be/make-variables#predefined_label_variables)
     or one of the runfiles libraries in
     `@bazel_tools//tools/{bash,cpp,java}/runfiles` or, for a ruleset `rules_foo`,
     in `@rules_foo//foo/runfiles`.
*    When interacting with Bazel from an external tool such as an IDE or language
     server, use the `bazel mod dump_repo_mapping` command to get the mapping from
     apparent names to canonical names for a given set of repositories.

[Module extensions](/external/extension) can also introduce additional repos
into the visible scope of a module.
