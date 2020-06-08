---
layout: documentation
title: User's guide
---

# A user's guide to Bazel

To run Bazel, go to your base [workspace](build-ref.html#workspace) directory
or any of its subdirectories and type `bazel`.

<pre>
  % bazel help
                             [Bazel release bazel-&lt;<i>version</i>&gt;]
  Usage: bazel &lt;command&gt; &lt;options&gt; ...

  Available commands:
    <a href='user-manual.html#analyze-profile'>analyze-profile</a>     Analyzes build profile data.
    <a href='user-manual.html#aquery'>aquery</a>              Executes a query on the <a href='#analysis-phase'>post-analysis</a> action graph.
    <a href='#build'>build</a>               Builds the specified targets.
    <a href='user-manual.html#canonicalize'>canonicalize-flags</a>  Canonicalize Bazel flags.
    <a href='user-manual.html#clean'>clean</a>               Removes output files and optionally stops the server.

    <a href='user-manual.html#query'>cquery</a>              Executes a <a href='#analysis-phase'>post-analysis</a> dependency graph query.

    <a href='user-manual.html#dump'>dump</a>                Dumps the internal state of the Bazel server process.

    <a href='user-manual.html#help'>help</a>                Prints help for commands, or the index.
    <a href='user-manual.html#info'>info</a>                Displays runtime info about the bazel server.

    <a href='#fetch'>fetch</a>               Fetches all external dependencies of a target.
    <a href='user-manual.html#mobile-install'>mobile-install</a>      Installs apps on mobile devices.

    <a href='user-manual.html#query'>query</a>               Executes a dependency graph query.

    <a href='user-manual.html#run'>run</a>                 Runs the specified target.
    <a href='user-manual.html#shutdown'>shutdown</a>            Stops the Bazel server.
    <a href='user-manual.html#test'>test</a>                Builds and runs the specified test targets.
    <a href='user-manual.html#version'>version</a>             Prints version information for Bazel.

  Getting more help:
    bazel help &lt;command&gt;
                     Prints help and options for &lt;command&gt;.
    bazel help <a href='user-manual.html#startup_options'>startup_options</a>
                     Options for the JVM hosting Bazel.
    bazel help <a href='#target-patterns'>target-syntax</a>
                     Explains the syntax for specifying targets.
    bazel help info-keys
                     Displays a list of keys used by the info command.

</pre>

The `bazel` tool performs many functions, called commands. The most commonly
used ones are `bazel build` and `bazel test`. You can browse the online help
messages using `bazel help`.

<a id="build"></a>

## Building programs with Bazel

### The `build` command

Type `bazel build` followed by the name of the [target](#target-patterns) you
wish to build. Here's a typical session:

```
% bazel build //foo
INFO: Analyzed target //foo:foo (14 packages loaded, 48 targets configured).
INFO: Found 1 target...
Target //foo:foo up-to-date:
  bazel-bin/foo/foo
INFO: Elapsed time: 9.905s, Critical Path: 3.25s
INFO: Build completed successfully, 6 total actions
```

Bazel prints the progress messages as it loads all the packages in the
transitive closure of dependencies of the requested target, then analyzes them
for correctness and to create the build actions, finally executing the compilers
and other tools of the build.

Bazel prints progress messages during the [execution phase](#execution-phase) of
the build, showing the current build step (compiler, linker, etc.) that is being
started, and the number completed over the total number of build actions. As the
build starts the number of total actions will often increase as Bazel discovers
the entire action graph, but the number will usually stabilize within a few
seconds.

At the end of the build Bazel prints which targets were requested, whether or
not they were successfully built, and if so, where the output files can be
found. Scripts that run builds can reliably parse this output; see
[`--show_result`](user-manual.html#flag--show_result) for more details.

Typing the same command again:

```
% bazel build //foo
INFO: Analyzed target //foo:foo (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
Target //foo:foo up-to-date:
  bazel-bin/foo/foo
INFO: Elapsed time: 0.144s, Critical Path: 0.00s
INFO: Build completed successfully, 1 total action
```

We see a "null" build: in this case, there are no packages to re-load, since
nothing has changed, and no build steps to execute. (If something had changed in
"foo" or some of its dependencies, resulting in the re-execution of some build
actions, we would call it an "incremental" build, not a "null" build.)

Before you can start a build, you will need a Bazel workspace. This is simply a
directory tree that contains all the source files needed to build your
application. Bazel allows you to perform a build from a completely read-only
volume.

<a id="target-patterns"></a>

### Specifying targets to build

Bazel allows a number of ways to specify the targets to be built. Collectively,
these are known as _target patterns_. This syntax is used in commands like
`build`, `test`, or `query`.

Whereas [labels](build-ref.html#labels) are used to specify individual targets,
e.g. for declaring dependencies in BUILD files, Bazel's target patterns are a
syntax for specifying multiple targets: they are a generalization of the label
syntax for _sets_ of targets, using wildcards. In the simplest case, any valid
label is also a valid target pattern, identifying a set of exactly one target.

All target patterns starting with `//` are resolved relative to the current
workspace.

<table>
<tr>
  <td><code>//foo/bar:wiz</code></td>
  <td>Just the single target <code>//foo/bar:wiz</code>.</td>
</tr>
<tr>
  <td><code>//foo/bar</code></td>
  <td>Equivalent to <code>//foo/bar:bar</code>.</td>
</tr>
<tr>
  <td><code>//foo/bar:all</code></td>
  <td>All rules in the package <code>foo/bar</code>.</td>
</tr>
<tr>
  <td><code>//foo/...</code></td>
  <td>All rules in all packages beneath the directory <code>foo</code>.</td>
</tr>
<tr>
  <td><code>//foo/...:all</code></td>
  <td>All rules in all packages beneath the directory <code>foo</code>.</td>
</tr>
<tr>
  <td><code>//foo/...:*</code></td>
  <td>All targets (rules and files) in all packages beneath the directory <code>foo</code>.</td>
</tr>
<tr>
  <td><code>//foo/...:all-targets</code></td>
  <td>All targets (rules and files) in all packages beneath the directory <code>foo</code>.</td>
</tr>
</table>

Target patterns which do not begin with `//` are resolved relative to the
current _working directory_. These examples assume a working directory of `foo`:

<table>
<tr>
  <td><code>:foo</code></td>
  <td>Equivalent to  <code>//foo:foo</code>.</td>
</tr>
<tr>
  <td><code>bar:wiz</code></td>
  <td>Equivalent to <code>//foo/bar:wiz</code>.</td>
</tr>
<tr>
  <td><code>bar/wiz</code></td>
  <td>Equivalent to:
                      <code>//foo/bar/wiz:wiz</code> if <code>foo/bar/wiz</code> is a package,
                      <code>//foo/bar:wiz</code> if <code>foo/bar</code> is a package,
                      <code>//foo:bar/wiz</code> otherwise.
  </td>
</tr>
<tr>
  <td><code>bar:all</code></td>
  <td>Equivalent to  <code>//foo/bar:all</code>.</td>
</tr>
<tr>
  <td><code>:all</code></td>
  <td>Equivalent to  <code>//foo:all</code>.</td>
</tr>
<tr>
  <td><code>...:all</code></td>
  <td>Equivalent to  <code>//foo/...:all</code>.</td>
</tr>
<tr>
  <td><code>...</code></td>
  <td>Equivalent to  <code>//foo/...:all</code>.</td>
</tr>
<tr>
  <td><code>bar/...:all</code></td>
  <td>Equivalent to  <code>//foo/bar/...:all</code>.</td>
</tr>
</table>

By default, directory symlinks are followed for recursive target patterns,
except those that point to under the output base, such as the convenience
symlinks that are created in the root directory of the workspace.

In addition, Bazel does not follow symlinks when evaluating recursive target
patterns in any directory that contains a file named as follows:
`DONT_FOLLOW_SYMLINKS_WHEN_TRAVERSING_THIS_DIRECTORY_VIA_A_RECURSIVE_TARGET_PATTERN`

`foo/...` is a wildcard over _packages_, indicating all packages recursively
beneath directory `foo` (for all roots of the package path). `:all` is a
wildcard over _targets_, matching all rules within a package. These two may be
combined, as in `foo/...:all`, and when both wildcards are used, this may be
abbreviated to `foo/...`.

In addition, `:*` (or `:all-targets`) is a wildcard that matches _every target_
in the matched packages, including files that aren't normally built by any rule,
such as `_deploy.jar` files associated with `java_binary` rules.

This implies that `:*` denotes a _superset_ of `:all`; while potentially
confusing, this syntax does allow the familiar `:all` wildcard to be used for
typical builds, where building targets like the `_deploy.jar` is not desired.

In addition, Bazel allows a slash to be used instead of the colon required by
the label syntax; this is often convenient when using Bash filename expansion.
For example, `foo/bar/wiz` is equivalent to `//foo/bar:wiz` (if there is a
package `foo/bar`) or to `//foo:bar/wiz` (if there is a package `foo`).

Many Bazel commands accept a list of target patterns as arguments, and they all
honor the prefix negation operator `-`. This can be used to subtract a set of
targets from the set specified by the preceding arguments. Note that this means
order matters. For example,

```
bazel build foo/... bar/...
```

means "build all targets beneath `foo` _and_ all targets beneath `bar`", whereas

```
bazel build -- foo/... -foo/bar/...
```

means "build all targets beneath `foo` _except_ those beneath `foo/bar`". (The
`--` argument is required to prevent the subsequent arguments starting with `-`
from being interpreted as additional options.)

It's important to point out though that subtracting targets this way will not
guarantee that they are not built, since they may be dependencies of targets
that weren't subtracted. For example, if there were a target `//foo:all-apis`
that among others depended on `//foo/bar:api`, then the latter would be built as
part of building the former.

Targets with `tags = ["manual"]` will not be included in wildcard target
patterns (`...`, `:*`, `:all`, etc.). You should specify such test targets with
explicit target patterns on the command line if you want Bazel to build/test
them.

<a id="fetch"></a>
### Fetching external dependencies

By default, Bazel will download and symlink external dependencies during the
build. However, this can be undesirable, either because you'd like to know
when new external dependendencies are added or because you'd like to
"prefetch" dependencies (say, before a flight where you'll be offline). If you
would like to prevent new dependencies from being added during builds, you
can specify the `--fetch=false` flag. Note that this flag only
applies to repository rules that do not point to a directory in the local
file system. Changes, for example, to `local_repository`,
`new_local_repository` and Android SDK and NDK repository rules
will always take effect regardless of the value `--fetch` .

If you disallow fetching during builds and Bazel finds new external
dependencies, your build will fail.

You can manually fetch dependencies by running `bazel fetch`. If
you disallow during-build fetching, you'll need to run `bazel fetch`:

- Before you build for the first time.
- After you add a new external dependency.

Once it has been run, you should not need to run it again until the WORKSPACE
file changes.

`fetch` takes a list of targets to fetch dependencies for. For
example, this would fetch dependencies needed to build `//foo:bar`
and `//bar:baz`:

```
$ bazel fetch //foo:bar //bar:baz
```

To fetch all external dependencies for a workspace, run:

```
$ bazel fetch //...
```

You do not need to run bazel fetch at all if you have all of the tools you are
using (from library jars to the JDK itself) under your workspace root.
However, if you're using anything outside of the workspace directory then Bazel
will automatically run `bazel fetch` before running
`bazel build`.

<a id="repository-cache"></a>
#### The repository cache

Bazel tries to avoid fetching the same file several times, even if the same
file is needed in different workspaces, or if the definition of an external
repository changed but it still needs the same file to download. To do so,
bazel caches all files downloaded in the repository cache which, by default,
is  located at `~/.cache/bazel/_bazel_$USER/cache/repos/v1/`. The
location can be changed by the `--repository_cache` option. The
cache is shared between all workspaces and installed versions of bazel.
An entry is taken from the cache if
Bazel knows for sure that it has a copy of the correct file, that is, if the
download request has a SHA256 sum of the file specified and a file with that
hash is in the cache. So specifying a hash for each external file is
not only a good idea from a security perspective; it also helps avoiding
unnecessary downloads.

Upon each cache hit, the modification time of the file in the cache is
updated. In this way, the last use of a file in the cache directory can easily
be determined, for example to manually clean up the cache. The cache is never
cleaned up automatically, as it might contain a copy of a file that is no
longer available upstream.

<a id="distdir"></a>
#### Distribution files directories

The distribution directory is another Bazel mechanism to avoid unnecessary
downloads. Bazel searches distribution directories before the repository cache.
The primary difference is that the distribution directory requires manual
preparation.

Using the
[`--distdir=/path/to-directory`](https://docs.bazel.build/versions/master/command-line-reference.html#flag--distdir)
option, you can specify additional read-only directories to look for files
instead of fetching them. A file is taken from such a directory if the file name
is equal to the base name of the URL and additionally the hash of the file is
equal to the one specified in the download request. This only works if the
file hash is specified in the WORKSPACE declaration.

While the condition on the file name is not necessary for correctness, it
reduces the number of candidate files to one per specified directory. In this
way, specifying distribution files directories remains efficient, even if the
number of files in such a directory grows large.

#### Running Bazel in an airgapped environment

To keep Bazel's binary size small, Bazel's implicit dependencies are fetched
over the network while running for the first time. These implicit dependencies
contain toolchains and rules that may not be necessary for everyone. For
example, Android tools are unbundled and fetched only when building Android
projects.

However, these implicit dependencies may cause problems when running
Bazel in an airgapped environment, even if you have vendored all of your
WORKSPACE dependencies. To solve that, you can prepare a distribution directory
containing these dependencies on a machine with network access, and then
transfer them to the airgapped environment with an offline approach.

To prepare the [distribution directory](distribution-files-directories), use the
[`--distdir`](https://docs.bazel.build/versions/master/command-line-reference.html#flag--distdir)
flag. You will need to do this once for every new Bazel binary version, since
the implicit dependencies can be different for every release.

To build these dependencies outside of your airgapped environment, first
checkout the Bazel source tree at the right version:

```
git clone https://github.com/bazelbuild/bazel "$BAZEL_DIR"
cd "$BAZEL_DIR"
git checkout "$BAZEL_VERSION"
```

Then, build the tarball containing the implicit runtime dependencies for that
specific Bazel version:

```
bazel build @additional_distfiles//:archives.tar
```

Export this tarball to a directory that can be copied into your airgapped
environment. Note the `--strip-components` flag, because `--distdir` can be
quite finicky with the directory nesting level:


```
tar xvf bazel-bin/external/additional_distfiles/archives.tar \
  -C "$NEW_DIRECTORY" --strip-components=3
```

Finally, when you use Bazel in your airgapped environment, pass the `--distdir`
flag pointing to the directory. For convenience, you can add it as an `.bazelrc`
entry:

```
build --distdir=path/to/directory
```


<a id="configurations"></a>

### Build configurations and cross-compilation

All the inputs that specify the behavior and result of a given build can be
divided into two distinct categories. The first kind is the intrinsic
information stored in the BUILD files of your project: the build rule, the
values of its attributes, and the complete set of its transitive dependencies.
The second kind is the external or environmental data, supplied by the user or
by the build tool: the choice of target architecture, compilation and linking
options, and other toolchain configuration options. We refer to a complete set
of environmental data as a **configuration**.

In any given build, there may be more than one configuration. Consider a
cross-compile, in which you build a `//foo:bin` executable for a 64-bit
architecture, but your workstation is a 32-bit machine. Clearly, the build will
require building `//foo:bin` using a toolchain capable of creating 64-bit
executables, but the build system must also build various tools used during the
build itself&mdash;for example tools that are built from source, then
subsequently used in, say, a genrule&mdash;and these must be built to run on
your workstation. Thus we can identify two configurations: the **host
configuration**, which is used for building tools that run during the build, and
the **target configuration** (or _request configuration_, but we say "target
configuration" more often even though that word already has many meanings),
which is used for building the binary you ultimately requested.

Typically, there are many libraries that are prerequisites of both the requested
build target (`//foo:bin`) and one or more of the host tools, for example some
base libraries. Such libraries must be built twice, once for the host
configuration, and once for the target configuration. Bazel takes care of
ensuring that both variants are built, and that the derived files are kept
separate to avoid interference; usually such targets can be built concurrently,
since they are independent of each other. If you see progress messages
indicating that a given target is being built twice, this is most likely the
explanation.

Bazel uses one of two ways to select the host configuration, based on the
`--distinct_host_configuration` option. This boolean option is somewhat subtle,
and the setting may improve (or worsen) the speed of your builds.

#### `--distinct_host_configuration=false`

We do not recommend this option.


-   If you frequently make changes to your request configuration, such as
    alternating between `-c opt` and `-c dbg` builds, or between simple- and
    cross-compilation, you will typically rebuild the majority of your codebase
    each time you switch.

When this option is false, the host and request configurations are identical:
all tools required during the build will be built in exactly the same way as
target programs. This setting means that no libraries need to be built twice
during a single build.

However, it does mean that any change to your request configuration also affects
your host configuration, causing all the tools to be rebuilt, and then anything
that depends on the tool output to be rebuilt too. Thus, for example, simply
changing a linker option between builds might cause all tools to be re-linked,
and then all actions using them re-executed, and so on, resulting in a very
large rebuild. Also, please note: if your host architecture is not capable of
running your target binaries, your build will not work.

#### `--distinct_host_configuration=true` _(default)_

If this option is true, then instead of using the same configuration for the
host and request, a completely distinct host configuration is used. The host
configuration is derived from the target configuration as follows:

-   Use the same version of Crosstool (`--crosstool_top`) as specified in the
    request configuration, unless `--host_crosstool_top` is specified.
-   Use the value of `--host_cpu` for `--cpu` (default: `k8`).
-   Use the same values of these options as specified in the request
    configuration: `--compiler`, `--use_ijars`, and if `--host_crosstool_top` is
    used, then the value of `--host_cpu` is used to look up a
    `default_toolchain` in the Crosstool (ignoring `--compiler`) for the host
    configuration.
-   Use the value of `--host_javabase` for `--javabase`
-   Use the value of `--host_java_toolchain` for `--java_toolchain`
-   Use optimized builds for C++ code (`-c opt`).
-   Generate no debugging information (`--copt=-g0`).
-   Strip debug information from executables and shared libraries
    (`--strip=always`).
-   Place all derived files in a special location, distinct from that used by
    any possible request configuration.
-   Suppress stamping of binaries with build data (see `--embed_*` options).
-   All other values remain at their defaults.

There are many reasons why it might be preferable to select a distinct host
configuration from the request configuration. Some are too esoteric to mention
here, but two of them are worth pointing out.

Firstly, by using stripped, optimized binaries, you reduce the time spent
linking and executing the tools, the disk space occupied by the tools, and the
network I/O time in distributed builds.

Secondly, by decoupling the host and request configurations in all builds, you
avoid very expensive rebuilds that would result from minor changes to the
request configuration (such as changing a linker options does), as described
earlier.

That said, for certain builds, this option may be a hindrance. In particular,
builds in which changes of configuration are infrequent (especially certain Java
builds), and builds where the amount of code that must be built in both host and
target configurations is large, may not benefit.

<a id="correctness"></a>

### Correct incremental rebuilds

One of the primary goals of the Bazel project is to ensure correct incremental
rebuilds. Previous build tools, especially those based on Make, make several
unsound assumptions in their implementation of incremental builds.

Firstly, that timestamps of files increase monotonically. While this is the
typical case, it is very easy to fall afoul of this assumption; syncing to an
earlier revision of a file causes that file's modification time to decrease;
Make-based systems will not rebuild.

More generally, while Make detects changes to files, it does not detect changes
to commands. If you alter the options passed to the compiler in a given build
step, Make will not re-run the compiler, and it is necessary to manually discard
the invalid outputs of the previous build using `make clean`.

Also, Make is not robust against the unsuccessful termination of one of its
subprocesses after that subprocess has started writing to its output file. While
the current execution of Make will fail, the subsequent invocation of Make will
blindly assume that the truncated output file is valid (because it is newer than
its inputs), and it will not be rebuilt. Similarly, if the Make process is
killed, a similar situation can occur.

Bazel avoids these assumptions, and others. Bazel maintains a database of all
work previously done, and will only omit a build step if it finds that the set
of input files (and their timestamps) to that build step, and the compilation
command for that build step, exactly match one in the database, and, that the
set of output files (and their timestamps) for the database entry exactly match
the timestamps of the files on disk. Any change to the input files or output
files, or to the command itself, will cause re-execution of the build step.

The benefit to users of correct incremental builds is: less time wasted due to
confusion. (Also, less time spent waiting for rebuilds caused by use of `make
clean`, whether necessary or pre-emptive.)

#### Build consistency and incremental builds

Formally, we define the state of a build as _consistent_ when all the expected
output files exist, and their contents are correct, as specified by the steps or
rules required to create them. When you edit a source file, the state of the
build is said to be _inconsistent_, and remains inconsistent until you next run
the build tool to successful completion. We describe this situation as _unstable
inconsistency_, because it is only temporary, and consistency is restored by
running the build tool.

There is another kind of inconsistency that is pernicious: _stable
inconsistency_. If the build reaches a stable inconsistent state, then repeated
successful invocation of the build tool does not restore consistency: the build
has gotten "stuck", and the outputs remain incorrect. Stable inconsistent states
are the main reason why users of Make (and other build tools) type `make clean`.
Discovering that the build tool has failed in this manner (and then recovering
from it) can be time consuming and very frustrating.

Conceptually, the simplest way to achieve a consistent build is to throw away
all the previous build outputs and start again: make every build a clean build.
This approach is obviously too time-consuming to be practical (except perhaps
for release engineers), and therefore to be useful, the build tool must be able
to perform incremental builds without compromising consistency.

Correct incremental dependency analysis is hard, and as described above, many
other build tools do a poor job of avoiding stable inconsistent states during
incremental builds. In contrast, Bazel offers the following guarantee: after a
successful invocation of the build tool during which you made no edits, the
build will be in a consistent state. (If you edit your source files during a
build, Bazel makes no guarantee about the consistency of the result of the
current build. But it does guarantee that the results of the _next_ build will
restore consistency.)

As with all guarantees, there comes some fine print: there are some known ways
of getting into a stable inconsistent state with Bazel. We won't guarantee to
investigate such problems arising from deliberate attempts to find bugs in the
incremental dependency analysis, but we will investigate and do our best to fix
all stable inconsistent states arising from normal or "reasonable" use of the
build tool.

If you ever detect a stable inconsistent state with Bazel, please report a bug.


<a id="sandboxing"></a>

#### Sandboxed execution

Bazel uses sandboxes to guarantee that actions run hermetically<sup>1</sup> and
correctly. Bazel runs _Spawns_ (loosely speaking: actions) in sandboxes that
only contain the minimal set of files the tool requires to do its job. Currently
sandboxing works on Linux 3.12 or newer with the `CONFIG_USER_NS` option
enabled, and also on macOS 10.11 or newer.

Bazel will print a warning if your system does not support sandboxing to alert
you to the fact that builds are not guaranteed to be hermetic and might affect
the host system in unknown ways. To disable this warning you can pass the
`--ignore_unsupported_sandboxing` flag to Bazel.

<sup>1</sup>: Hermeticity means that the action only uses its declared input
files and no other files in the filesystem, and it only produces its declared
output files.

On some platforms such as [Google Kubernetes
Engine](https://cloud.google.com/kubernetes-engine/) cluster nodes or Debian,
user namespaces are deactivated by default due to security
concerns. This can be checked by looking at the file
`/proc/sys/kernel/unprivileged_userns_clone`: if it exists and contains a 0,
then user namespaces can be activated with
`sudo sysctl kernel.unprivileged_userns_clone=1`.

In some cases, the Bazel sandbox fails to execute rules because of the system
setup. The symptom is generally a failure that output a message similar to
`namespace-sandbox.c:633: execvp(argv[0], argv): No such file or directory`.
In that case, try to deactivate the sandbox for genrules with
`--strategy=Genrule=standalone` and for other rules with
`--spawn_strategy=standalone`. Also please report a bug on our
issue tracker and mention which Linux distribution you're using so that we can
investigate and provide a fix in a subsequent release.


<a id="phases"></a>

### Phases of a build

In Bazel, a build occurs in three distinct phases; as a user, understanding the
difference between them provides insight into the options which control a build
(see below).

#### Loading phase

The first is **loading** during which all the necessary BUILD files for the
initial targets, and their transitive closure of dependencies, are loaded,
parsed, evaluated and cached.

For the first build after a Bazel server is started, the loading phase typically
takes many seconds as many BUILD files are loaded from the file system. In
subsequent builds, especially if no BUILD files have changed, loading occurs
very quickly.

Errors reported during this phase include: package not found, target not found,
lexical and grammatical errors in a BUILD file, and evaluation errors.

#### Analysis phase

The second phase, **analysis**, involves the semantic analysis and validation of
each build rule, the construction of a build dependency graph, and the
determination of exactly what work is to be done in each step of the build.

Like loading, analysis also takes several seconds when computed in its entirety.
However, Bazel caches the dependency graph from one build to the next and only
reanalyzes what it has to, which can make incremental builds extremely fast in
the case where the packages haven't changed since the previous build.

Errors reported at this stage include: inappropriate dependencies, invalid
inputs to a rule, and all rule-specific error messages.

The loading and analysis phases are fast because Bazel avoids unnecessary file
I/O at this stage, reading only BUILD files in order to determine the work to be
done. This is by design, and makes Bazel a good foundation for analysis tools,
such as Bazel's [query](#query) command, which is implemented atop the loading
phase.

#### Execution phase

The third and final phase of the build is **execution**. This phase ensures that
the outputs of each step in the build are consistent with its inputs, re-running
compilation/linking/etc. tools as necessary. This step is where the build spends
the majority of its time, ranging from a few seconds to over an hour for a large
build. Errors reported during this phase include: missing source files, errors
in a tool executed by some build action, or failure of a tool to produce the
expected set of outputs.

<a id="client/server"></a>

## Client/server implementation

The Bazel system is implemented as a long-lived server process. This allows it
to perform many optimizations not possible with a batch-oriented implementation,
such as caching of BUILD files, dependency graphs, and other metadata from one
build to the next. This improves the speed of incremental builds, and allows
different commands, such as `build` and `query` to share the same cache of
loaded packages, making queries very fast.

When you run `bazel`, you're running the client. The client finds the server
based on the output base, which by default is determined by the path of the base
workspace directory and your userid, so if you build in multiple workspaces,
you'll have multiple output bases and thus multiple Bazel server processes.
Multiple users on the same workstation can build concurrently in the same
workspace because their output bases will differ (different userids). If the
client cannot find a running server instance, it starts a new one. The server
process will stop after a period of inactivity (3 hours, by default, which can
be modified using the startup option `--max_idle_secs`).

For the most part, the fact that there is a server running is invisible to the
user, but sometimes it helps to bear this in mind. For example, if you're
running scripts that perform a lot of automated builds in different directories,
it's important to ensure that you don't accumulate a lot of idle servers; you
can do this by explicitly shutting them down when you're finished with them, or
by specifying a short timeout period.

The name of a Bazel server process appears in the output of `ps x` or `ps -e f`
as <code>bazel(<i>dirname</i>)</code>, where _dirname_ is the basename of the
directory enclosing the root of your workspace directory. For example:

```
% ps -e f
16143 ?        Sl     3:00 bazel(src-johndoe2) -server -Djava.library.path=...
```

This makes it easier to find out which server process belongs to a given
workspace. (Beware that with certain other options to `ps`, Bazel server
processes may be named just `java`.) Bazel servers can be stopped using the
[shutdown](#shutdown) command.

When running `bazel`, the client first checks that the server is the appropriate
version; if not, the server is stopped and a new one started. This ensures that
the use of a long-running server process doesn't interfere with proper
versioning.


<a id="bazelrc"></a>

## `.bazelrc`, the Bazel configuration file

Bazel accepts many options. Typically, some of these are varied frequently (for
example, `--subcommands`) while others stay the same across several builds (e.g.
`--package_path`). To avoid having to specify these unchanged options for every
build (and other commands) Bazel allows you to specify options in a
configuration file.

### Where are the `.bazelrc` files?
Bazel looks for optional configuration files in the following locations,
in the order shown below. The options are interpreted in this order, so
options in later files can override a value from an earlier file if a
conflict arises. All options that control which of these files are loaded are
startup options, which means they must occur after `bazel` and
before the command (`build`, `test`, etc).

1.  **The system RC file**, unless `--nosystem_rc` is present.

    Path:

    - On Linux/macOS/Unixes: `/etc/bazel.bazelrc`
    - On Windows: `%ProgramData%\bazel.bazelrc`

    It is not an error if this file does not exist.

    If another system-specified location is required, you must build a custom
    Bazel binary, overriding the `BAZEL_SYSTEM_BAZELRC_PATH` value in
    [`//src/main/cpp:option_processor`](https://github.com/bazelbuild/bazel/blob/0.28.0/src/main/cpp/BUILD#L141).
    The system-specified location may contain environment variable references,
    such as `${VAR_NAME}` on Unix or `%VAR_NAME%` on Windows.

2.  **The workspace RC file**, unless `--noworkspace_rc` is present.

    Path: `.bazelrc` in your workspace directory (next to the main
    `WORKSPACE` file).

    It is not an error if this file does not exist.

3.  **The home RC file**, unless `--nohome_rc` is present.

    Path:

    - On Linux/macOS/Unixes: `$HOME/.bazelrc`
    - On Windows: `%USERPROFILE%\.bazelrc` if exists, or `%HOME%/.bazelrc`

    It is not an error if this file does not exist.

4.  **The user-specified RC file**, if specified with
    <code>--bazelrc=<var>file</var></code>

    This flag is optional. However, if the flag is specified, then the file must
    exist.

### `.bazelrc` syntax and semantics

Like all UNIX "rc" files, the `.bazelrc` file is a text file with a line-based
grammar. Empty lines and lines starting with `#` (comments) are ignored. Each
line contains a sequence of words, which are tokenized according to the same
rules as the Bourne shell.

#### Imports

Lines that start with `import` or `try-import` are special: use these to load
other "rc" files. To specify a path that is relative to the workspace root,
write `import %workspace%/path/to/bazelrc`.

The difference between `import` and `try-import` is that Bazel fails if the
`import`'ed file is missing (or can't be read), but not so for a `try-import`'ed
file.

Import precedence:

-   Options in the imported file take precedence over options specified before
    the import statement.
-   Options specified after the import statement take precedence over the
    options in the imported file.
-   Options in files imported later take precedence over files imported earlier.

#### Option defaults

Most lines of a bazelrc define default option values. The first word on each
line specifies when these defaults are applied:

-   `startup`: startup options, which go before the command, and are described
    in `bazel help startup_options`.
-   `common`: options that apply to all Bazel commands.
-   _`command`_: Bazel command, such as `build` or `query` to which the options
    apply. These options also apply to all commands that inherit from the
    specified command. (For example, `test` inherits from `build`.)

Each of these lines may be used more than once and the arguments that follow the
first word are combined as if they had appeared on a single line. (Users of CVS,
another tool with a "Swiss army knife" command-line interface, will find the
syntax similar to that of `.cvsrc`.) For example, the lines:

```
build --test_tmpdir=/tmp/foo --verbose_failures
build --test_tmpdir=/tmp/bar
```

are combined as:

```
build --test_tmpdir=/tmp/foo --verbose_failures --test_tmpdir=/tmp/bar
```

so the effective flags are `--verbose_failures` and `--test_tmpdir=/tmp/bar`.

Option precedence:

-   Options on the command line always take precedence over those in rc files.
    For example, if a rc file says `build -c opt` but the command line flag is
    `-c dbg`, the command line flag takes precedence.
-   Within the rc file, precedence is governed by specificity: lines for a more
    specific command take precedence over lines for a less specific command.

    Specificity is defined by inheritance. Some commands inherit options from
    other commands, making the inheriting command more specific than the base
    command. For example `test` inherits from the `build` command, so all `bazel
    build` flags are valid for `bazel test`, and all `build` lines apply also to
    `bazel test` unless there's a `test` line for the same option. If the rc
    file says:

    ```
    test -c dbg --test_env=PATH
    build -c opt --verbose_failures
    ```

    then `bazel build //foo` will use `-c opt --verbose_failures`, and `bazel
    test //foo` will use `--verbose_failures -c dbg --test_env=PATH`.

    The inheritance (specificity) graph is:

    *   Every command inherits from `common`
    *   The following commands inherit from (and are more specific than)
        `build`: `test`, `run`, `clean`, `mobile-install`, `info`,
        `print_action`, `config`, `cquery`, and `aquery`
    *   `coverage` inherits from `test`

-   Two lines specifying options for the same command at equal specificity are
    parsed in the order in which they appear within the file.

-   Because this precedence rule does not match the file order, it helps
    readability if you follow the precedence order within rc files: start with
    `common` options at the top, and end with the most-specific commands at the
    bottom of the file. This way, the order in which the options are read is the
    same as the order in which they are applied, which is more intuitive.

The arguments specified on a line of an rc file may include arguments that are
not options, such as the names of build targets, and so on. These, like the
options specified in the same files, have lower precedence than their siblings
on the command line, and are always prepended to the explicit list of non-
option arguments.

#### `--config`

In addition to setting option defaults, the rc file can be used to group options
and provide a shorthand for common groupings. This is done by adding a `:name`
suffix to the command. These options are ignored by default, but will be
included when the option <code>--config=<var>name</var></code> is present,
either on the command line or in a `.bazelrc` file, recursively, even inside of
another config definition. The options specified by `command:name` will only be
expanded for applicable commands, in the precedence order described above.

Note that configs can be defined in any `.bazelrc` file, and that all lines of
the form `command:name` (for applicable commands) will be expanded, across the
different rc files. In order to avoid name conflicts, we suggest that configs
defined in personal rc files start with an underscore (`_`) to avoid
unintentional name sharing.

`--config=foo` expands to the options defined in the rc files "in-place" so that
the options specified for the config have the same precedence that the
`--config=foo` option had.

#### Example

Here's an example `~/.bazelrc` file:

```
# Bob's Bazel option defaults

startup --host_jvm_args=-XX:-UseParallelGC
import /home/bobs_project/bazelrc
build --show_timestamps --keep_going --jobs 600
build --color=yes
query --keep_going

# Definition of --config=memcheck
build:memcheck --strip=never --test_timeout=3600
```


<a id="startup files"></a>
### Other files governing Bazel's behavior

#### `.bazelignore`

You can specify directories within the workspace
that you want Bazel to ignore, such as related projects
that use other build systems. Place a file called
`.bazelignore` at the root of the workspace
and add the directories you want Bazel to ignore, one per
line. Entries are relative to the workspace root.


<a id="scripting"></a>

## Calling Bazel from scripts

Bazel can be called from scripts in order to perform a build, run tests or query
the dependency graph. Bazel has been designed to enable effective scripting, but
this section lists some details to bear in mind to make your scripts more
robust.


### Choosing the output base

The `--output_base` option controls where the Bazel process should write the
outputs of a build to, as well as various working files used internally by
Bazel, one of which is a lock that guards against concurrent mutation of the
output base by multiple Bazel processes.

Choosing the correct output base directory for your script depends on several
factors. If you need to put the build outputs in a specific location, this will
dictate the output base you need to use. If you are making a "read only" call to
Bazel (e.g. `bazel query`), the locking factors will be more important. In
particular, if you need to run multiple instances of your script concurrently,
you will need to give each one a different (or random) output base.

If you use the default output base value, you will be contending for the same
lock used by the user's interactive Bazel commands. If the user issues
long-running commands such as builds, your script will have to wait for those
commands to complete before it can continue.

### Notes about Server Mode

By default, Bazel uses a long-running [server process](#client/server) as an
optimization. When running Bazel in a script, don't forget to call `shutdown`
when you're finished with the server, or, specify `--max_idle_secs=5` so that
idle servers shut themselves down promptly.

### What exit code will I get?

Bazel attempts to differentiate failures due to the source code under
consideration from external errors that prevent Bazel from executing properly.
Bazel execution can result in following exit codes:

**Exit Codes common to all commands:**

-   `0` - Success
-   `2` - Command Line Problem, Bad or Illegal flags or command combination, or
    Bad Environment Variables. Your command line must be modified.
-   `8` - Build Interrupted but we terminated with an orderly shutdown.
-   `32` - External Environment Failure not on this machine.
-   `33` - OOM failure. You need to modify your command line.

-   `34` - Reserved for Google-internal use.
-   `35` - Reserved for Google-internal use.
-   `36` - Local Environmental Issue, suspected permanent.
-   `37` - Unhandled Exception / Internal Bazel Error.
-   `38` - Reserved for Google-internal use.
-   `40-44` - Reserved for errors in Bazel's command line launcher,
    `bazel.cc` that are not command line
    related. Typically these are related to bazel server
    being unable to launch itself.
-   `45` - Error publishing results to the Build Event Service.

**Return codes for commands `bazel build`, `bazel test`:**

-   `1` - Build failed.
-   `3` - Build OK, but some tests failed or timed out.
-   `4` - Build successful but no tests were found even though testing was
    requested.


**For `bazel run`:**

-   `1` - Build failed.
-   If the build succeeds but the executed subprocess returns a non-zero exit
    code it will be the exit code of the command as well.

**For `bazel query`:**

-   `3` - Partial success, but the query encountered 1 or more errors in the
    input BUILD file set and therefore the results of the operation are not 100%
    reliable. This is likely due to a `--keep_going` option on the command line.
-   `7` - Command failure.

Future Bazel versions may add additional exit codes, replacing generic failure
exit code `1` with a different non-zero value with a particular meaning.
However, all non-zero exit values will always constitute an error.

### Reading the .bazelrc file

By default, Bazel will read the [`.bazelrc` file](#bazelrc) from the base
workspace directory or the user's home directory. Whether or not this is
desirable is a choice for your script; if your script needs to be perfectly
hermetic (e.g. when doing release builds), you should disable reading the
.bazelrc file by using the option `--bazelrc=/dev/null`. If you want to perform
a build using the user's preferred settings, the default behavior is better.


### Command log

The Bazel output is also available in a command log file which you can find with
the following command:

```
% bazel info command_log
```

The command log file contains the interleaved stdout and stderr streams of the
most recent Bazel command. Note that running `bazel info` will overwrite the
contents of this file, since it then becomes the most recent Bazel command.
However, the location of the command log file will not change unless you change
the setting of the `--output_base` or `--output_user_root` options.

### Parsing output

The Bazel output is quite easy to parse for many purposes. Two options that may
be helpful for your script are `--noshow_progress` which suppresses progress
messages, and <code>--show_result <var>n</var></code>, which controls whether or
not "build up-to-date" messages are printed; these messages may be parsed to
discover which targets were successfully built, and the location of the output
files they created. Be sure to specify a very large value of _n_ if you rely on
these messages.


<a id="profiling"></a>

## Troubleshooting performance by profiling

The first step in analyzing the performance of your build is to profile your
build with the [`--profile`](user-manual.html#flag--profile) flag.

The file generated by the [`--profile`](user-manual.html#flag--profile) flag is
a binary file. Once you have generated this binary profile, you can analyze it
using Bazel's [`analyze-profile`](user-manual.html#analyze-profile') command. By
default, it will print out summary analysis information for each of the
specified profile datafiles. This includes cumulative statistics for different
task types for each build phase and an analysis of the critical execution path.

The first section of the default output describes an overview of the time spent
on the different build phases:

```
=== PHASE SUMMARY INFORMATION ===

Total launch phase time         6.00 ms    0.01%
Total init phase time            864 ms    1.11%
Total loading phase time       21.841 s   28.05%
Total analysis phase time       5.444 s    6.99%
Total preparation phase time     155 ms    0.20%
Total execution phase time     49.473 s   63.54%
Total finish phase time         83.9 ms    0.11%
Total run time                 77.866 s  100.00%
```

The following sections show the execution time of different tasks happening
during a particular phase:

```
=== INIT PHASE INFORMATION ===

Total init phase time                     864 ms

Total time (across all threads) spent on:
              Type    Total    Count     Average
          VFS_STAT    2.72%        1     23.5 ms
      VFS_READLINK   32.19%        1      278 ms

=== LOADING PHASE INFORMATION ===

Total loading phase time                21.841 s

Total time (across all threads) spent on:
              Type    Total    Count     Average
             SPAWN    3.26%      154      475 ms
          VFS_STAT   10.81%    65416     3.71 ms
[...]
STARLARK_BUILTIN_FN   13.12%    45138     6.52 ms

=== ANALYSIS PHASE INFORMATION ===

Total analysis phase time                5.444 s

Total time (across all threads) spent on:
              Type    Total    Count     Average
     SKYFRAME_EVAL    9.35%        1     4.782 s
       SKYFUNCTION   89.36%    43332     1.06 ms

=== EXECUTION PHASE INFORMATION ===

Total preparation time                    155 ms
Total execution phase time              49.473 s
Total time finalizing build              83.9 ms

Action dependency map creation           0.00 ms
Actual execution time                   49.473 s

Total time (across all threads) spent on:
              Type    Total    Count     Average
            ACTION    2.25%    12229     10.2 ms
[...]
       SKYFUNCTION    1.87%   236131     0.44 ms
```

The last section shows the critical path:

```
Critical path (32.078 s):
    Id        Time Percentage   Description
1109746     5.171 s   16.12%   Building [...]
1109745      164 ms    0.51%   Extracting interface [...]
1109744     4.615 s   14.39%   Building [...]
[...]
1109639     2.202 s    6.86%   Executing genrule [...]
1109637     2.00 ms    0.01%   Symlinking [...]
1109636      163 ms    0.51%   Executing genrule [...]
           4.00 ms    0.01%   [3 middleman actions]
```

You can use the following options to display more detailed information:

-   <a id="dump-text-format"></a>[`--dump=text`](user-manual.html#flag--dump)

    This option prints all recorded tasks in the order they occurred. Nested
    tasks are indented relative to the parent. For each task, output includes
    the following information:

    ```
    [task type] [task description]
    Thread: [thread id]    Id: [task id]     Parent: [parent task id or 0 for top-level tasks]
    Start time: [time elapsed from the profiling session start]       Duration: [task duration]
    [aggregated statistic for nested tasks, including count and total duration for each nested task]
    ```

-   <a id="dump-raw-format"></a>[`--dump=raw`](user-manual.html#flag--dump)

    This option is most useful for automated analysis with scripts. It outputs
    each task record on a single line using '|' delimiter between fields. Fields
    are printed in the following order:

    1.  thread id - integer positive number, identifies owner thread for the
        task
    2.  task id - integer positive number, identifies specific task
    3.  parent task id for nested tasks or 0 for root tasks
    4.  task start time in ns, relative to the start of the profiling session
    5.  task duration in ns. Please note that this will include duration of all
        subtasks.
    6.  aggregated statistic for immediate subtasks per type. This will include
        type name (lower case), number of subtasks for that type and their
        cumulative duration. Types are space-delimited and information for
        single type is comma-delimited.
    7.  task type (upper case)
    8.  task description

    Example:

    ```
    1|1|0|0|0||PHASE|Launch Bazel
    1|2|0|6000000|0||PHASE|Initialize command
    1|3|0|168963053|278111411||VFS_READLINK|/[...]
    1|4|0|571055781|23495512||VFS_STAT|/[...]
    1|5|0|869955040|0||PHASE|Load packages
    [...]
    ```

-   <a id="dump-html-format"></a>[`--html`](user-manual.html#flag--html)

    This option writes a file called `<profile-file>.html` in the directory of
    the profile file. Open it in your browser to see the visualization of the
    actions in your build. Note that the file can be quite large and may push
    the capabilities of your browser &ndash; please wait for the file to load.

    In most cases, the HTML output from [`--html`](user-manual.html#flag--html)
    is easier to read than the [`--dump`](user-manual.html#flag--dump) output.
    It includes a Gantt chart that displays time on the horizontal axis and
    threads of execution along the vertical axis. If you click on the Statistics
    link in the top right corner of the page, you will jump to a section that
    lists summary analysis information from your build.

    *   [`--html_details`](user-manual.html#flag--html_details)

        Additionally passing this option will render a more detailed execution
        chart and additional tables on the performance of built-in and
        user-defined Starlark functions. Beware that this increases the file
        size and the load on the browser considerably.

If Bazel appears to be hung, you can hit <kbd>Ctrl-&#92;</kbd> or send
Bazel a `SIGQUIT` signal (`kill -3 $(bazel info server_pid)`) to get a thread
dump in the file `$(bazel info output_base)/server/jvm.out`.

Since you may not be able to run `bazel info` if bazel is hung, the
`output_base` directory is usually the parent of the `bazel-<workspace>`
symlink in your workspace directory.
