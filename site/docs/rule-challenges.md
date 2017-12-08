---
layout: documentation
title: Challenges of Writing Rules
---

# Challenges of Writing Rules

We have heard feedback from various people that they have
difficulty to write efficient Bazel rules. There is no single root cause, but
it’s due to a combination of historical circumstances and intrinsic complexity
in the problem domain. This document attempts to give a high level overview of
the specific issues that we believe to be the main contributors.

* Assumption: Aim for Correctness, Throughput, Ease of Use & Latency
* Assumption: Large Scale Repositories
* Assumption: BUILD-like Description Language
* Intrinsic: Remote Execution and Caching are Hard
* Historic: Hard Separation between Loading, Analysis, and Execution is
  Outdated, but still affects the API
* Intrinsic: Using Change Information for Correct and Fast Incremental Builds
  requires Unusual Coding Patterns
* Intrinsic: Avoiding Quadratic Time and Memory Consumption is Hard

## Assumption: Aim for Correctness, Throughput, Ease of Use & Latency

We assume that the build system needs to be first and foremost correct with
respect to incremental builds, i.e., for a given source tree, the output of the
same build should always be the same, regardless of what the output tree looks
like. In the first approximation, this means Bazel needs to know every single
input that goes into a given build step, such that it can rerun that step if any
of the inputs change. There are limits to how correct Bazel can get, as it leaks
some information such as date / time of the build, and ignores certain types of
changes such as changes to file attributes. Sandboxing helps ensure correctness
by preventing reads to undeclared input files. Besides the intrinsic limits of
the system, there are a few known correctness issues, most of which are related
to Fileset or the C++ rules, which are both hard problems. We have long-term
efforts to fix these.

The second goal of the build system is to have high throughput; we are
permanently pushing the boundaries of what can be done within the current
machine allocation for a remote execution service. If the remote execution
service gets overloaded, nobody can get work done.

Ease of use comes next, i.e., of multiple correct approaches with the same (or
similar) footprint of the remote execution service, we choose the one that is
easier to use.

For the purpose of this document, latency denotes the time it takes from
starting a build to getting the intended result, whether that is a test log from
a passing or failing test, or an error message that a BUILD file has a
typo.

Note that these goals often overlap; latency is as much a function of throughput
of the remote execution service as is correctness relevant for ease of use.


## Assumption: Large Scale Repositories

The build system needs to operate at the scale of large repositories where large
scale means that it does not fit on a single hard drive, so it is impossible to
do a full checkout on virtually all developer machines. A medium-sized build
will need to read and parse tens of thousands of BUILD files, and evaluate
hundreds of thousands of globs. While it is theoretically possible to read all
BUILD files on a single machine, we have not yet been able to do so within a
reasonable amount of time and memory. As such, it is critical that BUILD files
can be loaded and parsed independently.


## Assumption: BUILD-like Description Language

For the purpose of this document, we assume a configuration language that is
roughly similar to BUILD files, i.e., declaration of library and binary rules
and their interdependencies. BUILD files can be read and parsed independently,
and we avoid even looking at source files whenever we can (except for
existence).


## Intrinsic: Remote Execution and Caching are Hard

Remote execution and caching improve build times in large repositories by
roughly two orders of magnitude compared to running the build on a single
machine. However, the scale at which it needs to perform is staggering: Google's
remote execution service is designed to handle a huge number of requests per
second, and the protocol carefully avoids unnecessary roundtrips as well as
unnecessary work on the service side.

At this time, the protocol requires that the build system knows all inputs to a
given action ahead of time; the build system then computes a unique action
fingerprint, and asks the scheduler for a cache hit. If a cache hit is found,
the scheduler replies with the digests of the output files; the files itself are
addressed by digest later on. However, this imposes restrictions on the Bazel
rules, which need to declare all input files ahead of time.


## Historic: Hard Separation between Loading, Analysis, and Execution is Outdated, but still affects the API

Technically, it is sufficient for a rule to know the input and output files of
an action just before the action is sent to remote execution. However, the
original Bazel code base had a strict separation of loading packages, then
analyzing rules using a configuration (command-line flags, essentially), and
only then running any actions. This distinction is still part of the rules API
today, even though the core of Bazel no longer requires it (more details below).

That means that the rules API requires a declarative description of the rule
interface (what attributes it has, types of attributes). There are some
exceptions where the API allows custom code to run during the loading phase to
compute implicit names of output files and implicit values of attributes. For
example, a java_library rule named ‘foo’ implicitly generates an output named
‘libfoo.jar’, which can be referenced from other rules in the build graph.

Furthermore, the analysis of a rule cannot read any source files or inspect the
output of an action; instead, it needs to generate a partial directed bipartite
graph of build steps and output file names that is only determined from the rule
itself and its dependencies.


## Intrinsic: Using Change Information for Correct and Fast Incremental Builds requires Unusual Coding Patterns

Above, we argued that in order to be correct, Bazel needs to know all the input
files that go into a build step in order to detect whether that build step is
still up-to-date. The same is true for package loading and rule analysis, and we
have designed [Skyframe] (https://bazel.build/designs/skyframe.html) to handle this
in general. Skyframe is a graph library and evaluation framework that takes a
goal node (such as ‘build //foo with these options’), and breaks it down into
its constituent parts, which are then evaluated and combined to yield this
result. As part of this process, Skyframe reads packages, analyzes rules, and
executes actions.

At each node, Skyframe tracks exactly which nodes any given node used to compute
its own output, all the way from the goal node down to the input files (which
are also Skyframe nodes). Having this graph explicitly represented in memory
allows the build system to identify exactly which nodes are affected by a given
change to an input file (including creation or deletion of an input file), doing
the minimal amount of work to restore the output tree to its intended state.

As part of this, each node performs a dependency discovery process; i.e., each
node can declare dependencies, and then use the contents of those dependencies
to declare even further dependencies. In principle, this maps well to a
thread-per-node model. However, medium-sized builds contain hundreds of
thousands of Skyframe nodes, which isn’t easily possible with current Java
technology (and for historical reasons, we’re currently tied to using Java, so
no lightweight threads and no continuations).

Instead, Bazel uses a fixed-size thread pool. However, that means that if a node
declares a dependency that isn’t available yet, we may have to abort that
evaluation and restart it (possibly in another thread), when the dependency is
available. This, in turn, means that nodes should not do this excessively; a
node that declares N dependencies serially can potentially be restarted N times,
costing O(N^2) time. Instead, we aim for up-front bulk declaration of
dependencies, which sometimes requires reorganizing the code, or even splitting
a node into multiple nodes to limit the number of restarts.

Note that this technology isn’t currently available in the rules API; instead,
the rules API is still defined using the legacy concepts of loading, analysis,
and execution phases. However, a fundamental restriction is that all accesses to
other nodes have to go through the framework so that it can track the
corresponding dependencies. Regardless of the language in which the build system
is implemented or in which the rules are written (they don’t have to be the
same), rule authors must not use standard libraries or patterns that bypass
Skyframe. For Java, that means avoiding java.io.File as well as any form of
reflection, and any library that does either. Libraries that support dependency
injection of these low-level interfaces still need to be setup correctly for
Skyframe.

This strongly suggests to avoid exposing rule authors to a full language runtime
in the first place. The danger of accidental use of such APIs is just too big -
several Bazel bugs in the past were caused by rules using unsafe APIs, even
though the rules were written by the Bazel team, i.e., by the domain experts.


## Intrinsic: Avoiding Quadratic Time and Memory Consumption is Hard

To make matters worse, apart from the requirements imposed by Skyframe, the
historical constraints of using Java, and the outdatedness of the rules API,
accidentally introducing quadratic time or memory consumption is a fundamental
problem in any build system based on library and binary rules. There are two
very common patterns that introduce quadratic memory consumption (and therefore
quadratic time consumption).

1. Chains of Library Rules -
Consider the case of a chain of library rules A depends on B, depends on C, and
so on. Then, we want to compute some property over the transitive closure of
these rules, such as the Java runtime classpath, or the C++ linker command for
each library. Naively, we might take a standard list implementation; however,
this already introduces quadratic memory consumption: the first library
contains one entry on the classpath, the second two, the third three, and so
on, for a total of 1+2+3+...+N = O(N^2) entries.

2. Binary Rules Depending on the Same Library Rules -
Consider the case where a set of binaries that depend on the same library
rules; for example, you might have a number of test rules that test the same
library code. Let’s say out of N rules, half the rules are binary rules, and
the other half library rules. Now consider that each binary makes a copy of
some property computed over the transitive closure of library rules, such as
the Java runtime classpath, or the C++ linker command line. For example, it
could expand the command line string representation of the C++ link action. N/2
copies of N/2 elements is O(N^2) memory.


### Custom Collections Classes to Avoid Quadratic Complexity

Bazel is heavily affected by both of these scenarios, so we introduced a set of
custom collection classes that effectively compress the information in memory by
avoiding the copy at each step. Almost all of these data structures have set
semantics, so we called the class NestedSet. The majority of changes to reduce
Bazel’s memory consumption over the past several years were changes to use
NestedSet instead of whatever was previously used.

Unfortunately, usage of NestedSet does not automatically solve all the issues;
in particular, even just iterating over a NestedSet in each rule re-introduces
quadratic time consumption. NestedSet also has some helper methods to facilitate
interoperability with normal collections classes; unfortunately, accidentally
passing a NestedSet to one of these methods leads to copying behavior, and
reintroduces quadratic memory consumption.
