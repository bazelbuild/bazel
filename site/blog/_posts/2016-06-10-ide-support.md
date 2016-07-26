---
layout: posts
title: IDE support
---

One of Bazel’s longest-standing feature requests is integration with IDEs.
With the 0.3 release, we finally have all machinery in place that allows
implementing integration with Bazel in IDEs. Simultaneous with that
Bazel release we are also making public two IDE plugins:

*   [Tulsi](http://tulsi.bazel.io): Bazel support for Xcode.
*   [e4b](https://github.com/bazelbuild/e4b): a sample Bazel plugin for Eclipse.

In this post, we will look into how Bazel enables IDE integration
and how an IDE plugin integrating with Bazel can be implemented.


## Principles of Bazel IDE support

Bazel BUILD files provide a description of a project’s source code: what
source files are part of the project, what artifacts (targets) should be
built from those files, what the dependencies between those files are, etc.
Bazel uses this information to perform a build, that is, it figures out the set
of actions needed to produce the artifacts (such as running a compiler or
linker) and executes those actions. Bazel accomplishes this by constructing a
_dependency graph_ between targets and visiting this graph to collect
those actions.

IDEs (as well as other tools working with source code) also need the same
information about the set of sources and their roles; but instead of building
the artifacts, IDEs use it to provide code navigation, autocompletion and
other code-aware features.

In the 0.3.0 Bazel release, we are adding a new concept to Bazel -
[_aspects_](/docs/skylark/aspects.html).
Aspects allow augmenting build dependency graphs with additional information
and actions. Applying an aspect to a build target creates a "shadow
dependency graph" reflecting all transitive dependencies of that target,
and the aspect's implementation determines the actions that Bazel executes
while traversing that graph.
The [documentation on aspects](/docs/skylark/aspects.html) explains this in more
detail.

## Architecture of a Bazel IDE plugin.

As an example of how aspects are useful for IDE integration, we will take
a look at a sample
[Eclipse plugin for Bazel support, e4b](https://github.com/bazelbuild/e4b).

e4b includes an aspect, defined in a file
[`e4b_aspect.bzl`](https://github.com/bazelbuild/e4b/blob/master/com.google.devtools.bazel.e4b/resources/tools/must/be/unique/e4b_aspect.bzl),
that when
applied to a particular target, generates a small JSON file with information
about that target relevant to Eclipse. Those JSON files are then consumed
by the e4b plugin inside Eclipse to build [Eclipse's representation
of a project](https://github.com/bazelbuild/e4b/blob/master/com.google.devtools.bazel.e4b/src/com/google/devtools/bazel/e4b/classpath/BazelClasspathContainer.java),
[`IClasspathContainer`](http://help.eclipse.org/juno/index.jsp?topic=%2Forg.eclipse.jdt.doc.isv%2Freference%2Fapi%2Forg%2Feclipse%2Fjdt%2Fcore%2FIClasspathContainer.html):

![e4bazel workflow](/assets/e4b-workflow.png)

Through the e4b plugin UI, the user specifies an initial set of targets
(typically a java or android binary, a selection of tests, all targets
in certain packages, etc). E4b plugin then invokes bazel as follows:

```
bazel build //java/com/company/example:main \
--aspects e4b_aspect.bzl%e4b_aspect \
--output_groups ide-info
```

(some details are omitted for clarity; see
[e4b source](https://github.com/bazelbuild/e4b/blob/master/com.google.devtools.bazel.e4b/src/com/google/devtools/bazel/e4b/command/BazelCommand.java) for complete
invocation)

The `--aspects` flag directs Bazel to apply `e4b_aspect`, exported from
`e4bazel.bzl` Skylark extension, to target `//java/com/company/example:main`.

The aspect is then applied transitively to the dependencies of the specified
targets, producing `.e4b-build.json` files for each target in the transitive
closure of dependencies. The e4b plugin reads those outputs and provides
a Classpath for Eclipse core to consume. If the input BUILD files change
so that a project model needs to be re-synced, the plugin still invokes
the exact same command: Bazel will rebuild only those files that are affected
by the change, so the plugin need only reexamine only those newly built
`.e4b-build.json` files. `ide-info` is an output group defined by e4b\_aspect;
the `--output_groups` flag ensures that only the artifacts belonging to that
group (and hence only to the aspect) are built, and therefore that no
unnecessary build steps are performed.

The aspect uses the
[`java` provider](/docs/skylark/lib/JavaSkylarkApiProvider.html) on the targets
it applies to to access a variety of information about Java targets.


