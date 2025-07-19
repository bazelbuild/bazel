Project: /_project.yaml
Book: /_book.yaml

# Frequently asked questions

{% include "_buttons.html" %}
{# disableFinding("repo") #}
{# disableFinding(HEADING_STACKED) #}

This page answers some frequently asked questions about external dependencies in
Bazel.

## MODULE.bazel {:#module-bazel}

### How should I version a Bazel module? {:#module-versioning-best-practices}

Setting `version` with the [`module`] directive in the source archive
`MODULE.bazel` can have several downsides and unintended side effects if not
managed carefully:

*   Duplication: releasing a new version of a module typically involves both
    incrementing the version in `MODULE.bazel` and tagging the release, two
    separate steps that can fall out of sync. While automation can
    reduce this risk, it's simpler and safer to avoid it altogether.

*   Inconsistency: users overriding a module with a specific commit using a
    [non-registry override] will see an incorrect version. for example, if the
    `MODULE.bazel` in the source archive sets `version = "0.3.0"` but
    additional commits have been made since that release, a user overriding
    with one of those commits would still see `0.3.0`. In reality, the version
    should reflect that it's ahead of the release, for example `0.3.1-rc1`.

*   Non-registry override issues: using placeholder values can cause issues
    when users override a module with a non-registry override. For example,
    `0.0.0` doesn't sort as the highest version, which is usually the expected
    behavior users want when doing a non-registry override.

Thus, it's best to avoid setting the version in the source archive
`MODULE.bazel`. Instead, set it in the `MODULE.bazel` stored in the registry
(e.g., the [Bazel Central Registry]), which is the actual source of truth for
the module version during Bazel's external dependency resolution (see [Bazel
registries]).

This is usually automated, for example the [`rules-template`] example rule
repository uses a [bazel-contrib/publish-to-bcr publish.yaml GitHub Action] to
publish the release to the BCR. The action [generates a patch for the source
archive `MODULE.bazel`] with the release version. This patch is stored in the
registry and is applied when the module is fetched during Bazel's external
dependency resolution.

This way, the version in the releases in the registry will be correctly set to
the released version and thus, `bazel_dep`, `single_version_override` and
`multiple_version_override` will work as expected, while avoiding potential
issues when doing a non-registry override because the version in the source
archive will be the default value (`''`), which will always be handled
correctly (it's the default version value after all) and will behave as
expected when sorting (the empty string is treated as the highest version).

[Bazel Central Registry]: https://registry.bazel.build/
[Bazel registries]: https://bazel.build/external/registry
[bazel-contrib/publish-to-bcr publish.yaml GitHub Action]: https://github.com/bazel-contrib/publish-to-bcr/blob/v0.2.2/.github/workflows/publish.yaml
[generates a patch for the source archive `MODULE.bazel`]: https://github.com/bazel-contrib/publish-to-bcr/blob/v0.2.2/src/domain/create-entry.ts#L176-L216
[`module`]: /rules/lib/globals/module#module
[non-registry override]: module.md#non-registry_overrides
[`rules-template`]: https://github.com/bazel-contrib/rules-template

### When should I increment the compatibility level? {:#incrementing-compatibility-level}

The [`compatibility_level`](module.md#compatibility_level) of a Bazel module
should be incremented _in the same commit_ that introduces a backwards
incompatible ("breaking") change.

However, Bazel can throw an error if it detects that versions of the _same
module_ with _different compatibility levels_ exist in the resolved dependency
graph. This can happen when for example' two modules depend on versions of a
third module with different compatibility levels.

Thus, incrementing `compatibility_level` too frequently can be very disruptive
and is discouraged. To avoid this situation, the `compatibility_level` should be
incremented _only_ when the breaking change affects most use cases and isn't
easy to migrate and/or work-around.

### Why does MODULE.bazel not support `load`s? {:#why-does-module-bazel-not-support-loads}

During dependency resolution, the MODULE.bazel file of all referenced external
dependencies are fetched from registries. At this stage, the source archives of
the dependencies are not fetched yet; so if the MODULE.bazel file `load`s
another file, there is no way for Bazel to actually fetch that file without
fetching the entire source archive. Note the MODULE.bazel file itself is
special, as it's directly hosted on the registry.

There are a few use cases that people asking for `load`s in MODULE.bazel are
generally interested in, and they can be solved without `load`s:

*   Ensuring that the version listed in MODULE.bazel is consistent with build
    metadata stored elsewhere, for example in a .bzl file: This can be achieved
    by using the
    [`native.module_version`](/rules/lib/toplevel/native#module_version) method
    in a .bzl file loaded from a BUILD file.
*   Splitting up a very large MODULE.bazel file into manageable sections,
    particularly for monorepos: The root module can use the
    [`include`](/rules/lib/globals/module#include) directive to split its
    MODULE.bazel file into multiple segments. For the same reason we don't allow
    `load`s in MODULE.bazel files, `include` cannot be used in non-root modules.
*   Users of the old WORKSPACE system might remember declaring a repo, and then
    immediately `load`ing from that repo to perform complex logic. This
    capability has been replaced by [module extensions](extension).

### Can I specify a SemVer range for a `bazel_dep`? {:#can-i-specify-a-semver-range-for-a-bazel-dep}

No. Some other package managers like [npm][npm-semver] and [Cargo][cargo-semver]
support version ranges (implicitly or explicitly), and this often requires a
constraint solver (making the output harder to predict for users) and makes
version resolution nonreproducible without a lockfile.

Bazel instead uses [Minimal Version Selection](module#version-selection) like
Go, which in contrast makes the output easy to predict and guarantees
reproducibility. This is a tradeoff that matches Bazel's design goals.

Furthermore, Bazel module versions are [a superset of
SemVer](module#version-format), so what makes sense in a strict SemVer
environment doesn't always carry over to Bazel module versions.

### Can I automatically get the latest version for a `bazel_dep`? {:#can-i-automatically-get-the-latest-version-for-a-bazel-dep}

Some users occasionally ask for the ability to specify `bazel_dep(name = "foo",
version = "latest")` to automatically get the latest version of a dep. This is
similar to [the question about SemVer
ranges](#can-i-specify-a-semver-range-for-a-bazel-dep), and the answer is also
no.

The recommended solution here is to have automation take care of this. For
example, [Renovate](https://docs.renovatebot.com/modules/manager/) supports
Bazel modules.

Sometimes, users asking this question are really looking for a way to quickly
iterate during local development. This can be achieved by using a
[`local_path_override`](/rules/lib/globals/module#local_path_override).

### Why all these `use_repo`s? {:#why-all-these-use-repos}

Module extension usages in MODULE.bazel files sometimes come with a big
`use_repo` directive. For example, a typical usage of the
[`go_deps` extension][go_deps] from `gazelle` might look like:

```python
go_deps = use_extension("@gazelle//:extensions.bzl", "go_deps")
go_deps.from_file(go_mod = "//:go.mod")
use_repo(
    go_deps,
    "com_github_gogo_protobuf",
    "com_github_golang_mock",
    "com_github_golang_protobuf",
    "org_golang_x_net",
    ...  # potentially dozens of lines...
)
```

The long `use_repo` directive may seem redundant, since the information is
arguably already in the referenced `go.mod` file.

The reason Bazel needs this `use_repo` directive is that it runs module
extensions lazily. That is, a module extension is only run if its result is
observed. Since a module extension's "output" is repo definitions, this means
that we only run a module extension if a repo it defines is requested (for
instance, if the target `@org_golang_x_net//:foo` is built, in the example
above). However, we don't know which repos a module extension would define until
after we run it. This is where the `use_repo` directive comes in; the user can
tell Bazel which repos they expect the extension to generate, and Bazel would
then only run the extension when these specific repos are used.

To help the maintain this `use_repo` directive, a module extension can return
an [`extension_metadata`](/rules/lib/builtins/module_ctx#extension_metadata)
object from its implementation function. The user can run the `bazel mod tidy`
command to update the `use_repo` directives for these module extensions.

## Bzlmod migration {:#bzlmod-migration}

### Which is evaluated first, MODULE.bazel or WORKSPACE? {:#which-is-evaluated-first-module-bazel-or-workspace}

When both `--enable_bzlmod` and `--enable_workspace` are set, it's natural to
wonder which system is consulted first. The short answer is that MODULE.bazel
(Bzlmod) is evaluated first.

The long answer is that "which evaluates first" is not the right question to
ask; rather, the right question to ask is: in the context of the repo with
[canonical name](overview#canonical-repo-name) `@@foo`, what does the [apparent
repo name](overview#apparent-repo-name) `@bar` resolve to? Alternatively, what
is the repo mapping of `@@base`?

Labels with apparent repo names (a single leading `@`) can refer to different
things based on the context they're resolved from. When you see a label
`@bar//:baz` and wonder what it actually points to, you need to first find out
what the context repo is: for example, if the label is in a BUILD file located
in the repo `@@foo`, then the context repo is `@@foo`.

Then, depending on what the context repo is, the ["repository
visibility" table](migration#repository-visibility) in the migration guide can
be used to find out which repo an apparent name actually resolves to.

*   If the context repo is the main repo (`@@`):
    1.  If `bar` is an apparent repo name introduced by the root module's
        MODULE.bazel file (through any of
        [`bazel_dep`](/rules/lib/globals/module#bazel_dep.repo_name),
        [`use_repo`](/rules/lib/globals/module#use_repo),
        [`module`](/rules/lib/globals/module#module.repo_name),
        [`use_repo_rule`](/rules/lib/globals/module#use_repo_rule)), then `@bar`
        resolves to what that MODULE.bazel file claims.
    2.  Otherwise, if `bar` is a repo defined in WORKSPACE (which means that its
        canonical name is `@@bar`), then `@bar` resolves to `@@bar`.
    3.  Otherwise, `@bar` resolves to something like
        `@@[unknown repo 'bar' requested from @@]`, and this will ultimately
        result in an error.
*   If the context repo is a Bzlmod-world repo (that is, it corresponds to a
    non-root Bazel module, or is generated by a module extension), then it
    will only ever see other Bzlmod-world repos, and no WORKSPACE-world repos.
    *   Notably, this includes any repos introduced in a `non_module_deps`-like
        module extension in the root module, or `use_repo_rule` instantiations
        in the root module.
*   If the context repo is defined in WORKSPACE:
    1.  First, check if the context repo definition has the magical
        `repo_mapping` attribute. If so, go through the mapping first (so for a
        repo defined with `repo_mapping = {"@bar": "@baz"}`, we would be looking
        at `@baz` below).
    2.  If `bar` is an apparent repo name introduced by the root module's
        MODULE.bazel file, then `@bar` resolves to what that MODULE.bazel file
        claims. (This is the same as item 1 in the main repo case.)
    3.  Otherwise, `@bar` resolves to `@@bar`. This most likely will point to a
        repo `bar` defined in WORKSPACE; if such a repo is not defined, Bazel
        will throw an error.

For a more succinct version:

*   Bzlmod-world repos (excluding the main repo) will only see Bzlmod-world
    repos.
*   WORKSPACE-world repos (including the main repo) will first see what the root
    module in the Bzlmod world defines, then fall back to seeing WORKSPACE-world
    repos.

Of note, labels in the Bazel command line (including Starlark flags, label-typed
flag values, and build/test target patterns) are treated as having the main repo
as the context repo.

## Other {:#other}

### How do I prepare and run an offline build? {:#how-do-i-prepare-and-run-an-offline-build}

Use the `bazel fetch` command to prefetch repos. You can use the `--repo` flag
(like `bazel fetch --repo @foo`) to fetch only the repo `@foo` (resolved in the
context of the main repo, see [question
above](#which-is-evaluated-first-module-bazel-or-workspace)), or use a target
pattern (like `bazel fetch @foo//:bar`) to fetch all transitive dependencies of
`@foo//:bar` (this is equivalent to `bazel build --nobuild @foo//:bar`).

The make sure no fetches happen during a build, use `--nofetch`. More precisely,
this makes any attempt to run a non-local repository rule fail.

If you want to fetch repos _and_ modify them to test locally, consider using
the [`bazel vendor`](vendor) command.

### How do I use HTTP proxies? {:#how-do-i-use-http-proxies}

Bazel respects the `http_proxy` and `HTTPS_PROXY` environment variables commonly
accepted by other programs, such as
[curl](https://everything.curl.dev/usingcurl/proxies/env.html).

### How do I make Bazel prefer IPv6 in dual-stack IPv4/IPv6 setups? {:#ipv6}

On IPv6-only machines, Bazel can download dependencies with no changes. However,
on dual-stack IPv4/IPv6 machines Bazel follows the same convention as Java,
preferring IPv4 if enabled. In some situations, for example when the IPv4
network cannot resolve/reach external addresses, this can cause `Network
unreachable` exceptions and build failures. In these cases, you can override
Bazel's behavior to prefer IPv6 by using the
[`java.net.preferIPv6Addresses=true` system
property](https://docs.oracle.com/javase/8/docs/api/java/net/doc-files/net-properties.html).
Specifically:

*   Use `--host_jvm_args=-Djava.net.preferIPv6Addresses=true` [startup
    option](/docs/user-manual#startup-options), for example by adding the
    following line in your [`.bazelrc` file](/run/bazelrc):

    `startup --host_jvm_args=-Djava.net.preferIPv6Addresses=true`

*   When running Java build targets that need to connect to the internet (such
    as for integration tests), use the
    `--jvmopt=-Djava.net.preferIPv6Addresses=true` [tool
    flag](/docs/user-manual#jvmopt). For example, include in your [`.bazelrc`
    file](/run/bazelrc):

    `build --jvmopt=-Djava.net.preferIPv6Addresses`

*   If you are using
    [`rules_jvm_external`](https://github.com/bazelbuild/rules_jvm_external) for
    dependency version resolution, also add
    `-Djava.net.preferIPv6Addresses=true` to the `COURSIER_OPTS` environment
    variable to [provide JVM options for
    Coursier](https://github.com/bazelbuild/rules_jvm_external#provide-jvm-options-for-coursier-with-coursier_opts).

### Can repo rules be run remotely with remote execution? {:#can-repo-rules-be-run-remotely-with-remote-execution}

No; or at least, not yet. Users employing remote execution services to speed up
their builds may notice that repo rules are still run locally. For example, an
`http_archive` would be first downloaded onto the local machine (using any local
download cache if applicable), extracted, and then each source file would be
uploaded to the remote execution service as an input file. It's natural to ask
why the remote execution service doesn't just download and extract that archive,
saving a useless roundtrip.

Part of the reason is that repo rules (and module extensions) are akin to
"scripts" that are run by Bazel itself. A remote executor doesn't necessarily
even have a Bazel installed.

Another reason is that Bazel often needs the BUILD files in the downloaded and
extracted archives to perform loading and analysis, which _are_ performed
locally.

There are preliminary ideas to solve this problem by re-imagining repo rules as
build rules, which would naturally allow them to be run remotely, but conversely
raise new architectural concerns (for example, the `query` commands would
potentially need to run actions, complicating their design).

For more previous discussion on this topic, see [A way to support repositories
that need Bazel for being
fetched](https://github.com/bazelbuild/bazel/discussions/20464).

[npm-semver]: https://docs.npmjs.com/about-semantic-versioning
[cargo-semver]: https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#version-requirement-syntax
[go_deps]: https://github.com/bazel-contrib/rules_go/blob/master/docs/go/core/bzlmod.md#specifying-external-dependencies

