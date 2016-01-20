---
layout: documentation
title: Skyframe
---

# Skyframe

<p class="lead">The parallel evaluation and incrementality model of Bazel</p>

## Data model

The data model consists of the following items:

 - `SkyValue`. Also called nodes. `SkyValues` are immutable objects that contain all the data built
   over the course of the build and the inputs of the build. Examples are: input files, output
   files, targets and configured targets.
 - `SkyKey`. A short immutable name to reference a `SkyValue`, for example, `FILECONTENTS:/tmp/foo`
   or `PACKAGE://foo`.
 - `SkyFunction`. Builds nodes based on their keys and dependent nodes.
 - Node graph. A data structure containing the dependency relationship between nodes.
 - `Skyframe`. Code name for the incremental evaluation framework Bazel is based on.


## Evaluation

A build consists of evaluating the node that represents the build request (this is the state we are
striving for, but there is a lot of legacy code in the way). First its `SkyFunction` is found and
called with the key of the top-level `SkyKey`. The function then requests the evaluation of the
nodes it needs to evaluate the top-level node, which in turn result in other function invocations,
and so on, until the leaf nodes are reached (which are usually nodes representing input files in
the file system). Finally, we end up with the value of the top-level `SkyValue`, some side effects
(e.g. output files in the file system) and a directed acyclic graph of the dependencies between the
nodes that were involved in the build.

A `SkyFunction` can request `SkyKeys` in multiple passes if it cannot tell in advance all of the
nodes it needs to do its job. A simple example is evaluating an input file node that turns out to
be a symlink: the function tries to read the file, realizes that it's a symlink, and thus fetches
the file system node representing the target of the symlink. But that itself can be a symlink, in
which case the original function will need to fetch its target, too.

The functions are represented in the code by the interface `SkyFunction` and the services
provided to it by an interface called `SkyFunction.Environment`. These are the things functions can
do:

 - Request the evaluation of another node by way of calling `env.getValue`. If the node is
   available, its value is returned, otherwise, `null` is returned and the function itself is
   expected to return `null`. In the latter case, the dependent node is evaluated, and then the
   original node builder is invoked again, but this time the same `env.getValue` call will return a
   non-`null` value.
 - Request the evaluation of multiple other nodes by calling `env.getValues()`. This does
   essentially the same, except that the dependent nodes are evaluated in parallel.
 - Do computation during their invocation
 - Have side effects, for example, writing files to the file system. Care needs to be taken that two
   different functions do not step on each other's toes. In general, write side effects (where
   data flows outwards from Bazel) are okay, read side effects (where data flows inwards into Bazel
   without a registered dependency) are not, because they are an unregistered dependency and as
   such, can cause incorrect incremental builds.

`SkyFunction` implementations should not access data in any other way than requesting dependencies
(e.g. by directly reading the file system), because that results in Bazel not registering the data
dependency on the file that was read, thus resulting in incorrect incremental builds.

Once a function has enough data to do its job, it should return a non-`null` value indicating
completion.

This evaluation strategy has a number of benefits:

 - Hermeticity. If functions only request input data by way of depending on other nodes, Bazel
   can guarantee that if the input state is the same, the same data is returned. If all sky
   functions are deterministic, this means that the whole build will also be deterministic.
 - Correct and perfect incrementality. If all the input data of all functions is recorded, Bazel
   can invalidate only the exact set of nodes that need to be invalidated when the input data
   changes.
 - Parallelism. Since functions can only interact with each other by way of requesting
   dependencies, functions that do not depend on each other can be run in parallel and Bazel can
   guarantee that the result is the same as if they were run sequentially.

## Incrementality

Since functions can only access input data by depending on other nodes, Bazel can build up a
complete data flow graph from the input files to the output files, and use this information to only
rebuild those nodes that actually need to be rebuilt: the reverse transitive closure of the set of
changed input files.

In particular, two possible incrementality strategies exist: the bottom-up one and the top-down one.
Which one is optimal depends on how the dependency graph looks like.

 - During bottom-up invalidation, after a graph is built and the set of changed inputs is known,
   all the nodes are invalidated that transitively depend on changed files. This is optimal
   if we know that the same top-level node will be built again.
   Note that bottom-up invalidation requires running `stat()` on all input files of the previous
   build to determine if they were changed. This can be improved by using `inotify` or a similar
   mechanism to learn about changed files.

 - During top-down invalidation, the transitive closure of the top-level node is checked and only
   those nodes are kept whose transitive closure is clean. This is better if we know that the
   current node graph is large, but we only need a small subset of it in the next build: bottom-up
   invalidation would invalidate the larger graph of the first build, unlike top-down invalidation,
   which just walks the small graph of second build.

We currently only do bottom-up invalidation.

To get further incrementality, we use _change pruning_: if a node is invalidated, but upon rebuild,
it is discovered that its new value is the same as its old value, the nodes that were invalidated
due to a change in this node are "resurrected".

This is useful, for example, if one changes a comment in a C++ file: then the `.o` file generated
from it will be the same, thus, we don't need to call the linker again.

## Incremental Linking / Compilation

The main limitation of this model is that the invalidation of a node is an all-or-nothing affair:
when a dependency changes, the dependent node is always rebuilt from scratch, even if a better
algorithm would exist that would mutate the old value of the node based on the changes. A few
examples where this would be useful:

 - Incremental linking
 - When a single `.class` file changes in a `.jar`, we could theoretically modify the `.jar` file
   instead of building it from scratch again.

The reason why Bazel currently does not support these things in a principled way (we have some
measure of support for incremental linking, but it's not implemented within Skyframe) is twofold:
we only had limited performance gains and it was hard to guarantee that the result of the mutation
is the same as that of a clean rebuild would be, and Google values builds that are bit-for-bit
repeatable.

Until now, we could always achieve good enough performance by simply decomposing an expensive build
step and achieving partial re-evaluation that way: it splits all the classes in an app into
multiple groups and does dexing on them separately. This way, if classes in a group do not change,
the dexing does not have to be redone.

## Restarting SkyFunctions

Another inefficiency is that, currently, if a `SkyFunction` implementation cannot complete its job
because one of its dependencies is missing, it needs to be completely restarted instead of resuming
where it left off. This is currently not a big problem because we usually learn all the
dependencies after a small amount of work. The only exceptions are package loading and execution of
actions; these are both external processes that are expensive to restart. We allow package loading
to proceed fully, store the loaded package away, record the dependencies in the graph, and on
re-execution of the function return the already loaded package. I.e., we allow the function to keep
state between executions.

If this turns out to be a significant performance or code health problem, there are alternative ways
to add a more principled mechanism to keep state between executions:

 - Splitting each node into multiple ones so that each smaller node only has to do one round of
   dependency discovery (effectively continuation passing); this requires explicit code.
 - By reimplementing Skyframe on some sort of lightweight thread infrastructure (e.g.
   [Quasar](http://docs.paralleluniverse.co/quasar/)) so that function execution can be suspended
   and resumed without a large performance hit and without requiring this to be explicit in the
   code.
 - By maintaining state for each `SkyFunction` instance between restarts (this is the workaround we
   are using for package loading, but is not implemented as a first-class feature of the evaluation
   framework).

## Mapping to Bazel concepts

This is a rough overview of some of the `SkyFunction` implementations Bazel uses to perform a build:

 - **FileStateValue**. The result of an `lstat()`. For existent files, we also compute additional
   information in order to detect changes to the file. This is the lowest level node in the Skyframe
   graph and has no dependencies.
 - **FileValue**. Used by anything that cares about the actual contents and/or resolved path of a
   file. Depends on the corresponding `FileStateValue` and any symlinks that need to be resolved
   (e.g. the `FileValue` for `a/b` needs the resolved path of `a` and the resolved path of `a/b`).
   The distinction between `FileStateValue` is important because in some cases (for example,
   evaluating file system globs (e.g. `srcs=glob(["*/*.java"])`) the contents of the file are not
   actually needed.
 - **DirectoryListingValue**. Essentially the result of `readdir()`. Depends on the associated
   `FileValue` associated with the directory.
 - **PackageValue**. Represents the parsed version of a BUILD file. Depends on the `FileValue` of
   the associated `BUILD` file, and also transitively on any `DirectoryListingValue` that is used
   to resolve the globs in the package (the data structure representing the contents of a `BUILD`
   file internally)
 - **ConfiguredTargetValue**. Represents a configured target, which is a tuple of the set of actions
   generated during the analysis of a target and information provided to configured targets that
   depend on this one. Depends on the `PackageValue` the corresponding target is in, the
   `ConfiguredTargetValues` of direct dependencies, and a special node representing the build
   configuration.
 - **ArtifactValue**. Represents a file in the build, be it a source or an output artifacts
   (artifacts are almost equivalent to files, and are used to refer to files during the actual
   execution of build steps). For source files, it depends on the `FileValue` of the associated
   node, for output artifacts, it depends on the `ActionExecutionValue` of whatever action generates
   the artifact.
 - **ActionExecutionValue**. Represents the execution of an action. Depends on the `ArtifactValues`
   of its input files. The action it executes is currently contained within its sky key, which is
   contrary to the concept that sky keys should be small. We are working on solving this
   discrepancy (note that `ActionExecutionValue` and `ArtifactValue` are unused if we do not run the
   execution phase on Skyframe).
