Project: /_project.yaml
Book: /_book.yaml

# Artifact-Based Build Systems

{% include "_buttons.html" %}

This page covers artifact-based build systems and the philosophy behind their
creation. Bazel is an artifact-based build system. While task-based build
systems are good step above build scripts, they give too much power to
individual engineers by letting them define their own tasks.

Artifact-based build systems have a small number of tasks defined by the system
that engineers can configure in a limited way. Engineers still tell the system
**what** to build, but the build system determines **how** to build it. As with
task-based build systems, artifact-based build systems, such as Bazel, still
have buildfiles, but the contents of those buildfiles are very different. Rather
than being an imperative set of commands in a Turing-complete scripting language
describing how to produce an output, buildfiles in Bazel are a declarative
manifest describing a set of artifacts to build, their dependencies, and a
limited set of options that affect how they’re built. When engineers run `bazel`
on the command line, they specify a set of targets to build (the **what**), and
Bazel is responsible for configuring, running, and scheduling the compilation
steps (the **how**). Because the build system now has full control over what
tools to run when, it can make much stronger guarantees that allow it to be far
more efficient while still guaranteeing correctness.

## A functional perspective

It’s easy to make an analogy between artifact-based build systems and functional
programming. Traditional imperative programming languages (such as, Java, C, and
Python) specify lists of statements to be executed one after another, in the
same way that task-based build systems let programmers define a series of steps
to execute. Functional programming languages (such as, Haskell and ML), in
contrast, are structured more like a series of mathematical equations. In
functional languages, the programmer describes a computation to perform, but
leaves the details of when and exactly how that computation is executed to the
compiler.

This maps to the idea of declaring a manifest in an artifact-based build system
and letting the system figure out how to execute the build. Many problems can't
be easily expressed using functional programming, but the ones that do benefit
greatly from it: the language is often able to trivially parallelize such
programs and make strong guarantees about their correctness that would be
impossible in an imperative language. The easiest problems to express using
functional programming are the ones that simply involve transforming one piece
of data into another using a series of rules or functions. And that’s exactly
what a build system is: the whole system is effectively a mathematical function
that takes source files (and tools like the compiler) as inputs and produces
binaries as outputs. So, it’s not surprising that it works well to base a build
system around the tenets of functional programming.

## Understanding artifact-based build systems

Google's build system, Blaze, was the first artifact-based build system. Bazel
is the open-sourced version of Blaze.

Here’s what a buildfile (normally named `BUILD`) looks like in Bazel:

```python
java_binary(
    name = "MyBinary",
    srcs = ["MyBinary.java"],
    deps = [
        ":mylib",
    ],
)
java_library(
    name = "mylib",
    srcs = ["MyLibrary.java", "MyHelper.java"],
    visibility = ["//java/com/example/myproduct:__subpackages__"],
    deps = [
        "//java/com/example/common",
        "//java/com/example/myproduct/otherlib",
    ],
)
```

In Bazel, `BUILD` files define targets—the two types of targets here are
`java_binary` and `java_library`. Every target corresponds to an artifact that
can be created by the system: binary targets produce binaries that can be
executed directly, and library targets produce libraries that can be used by
binaries or other libraries. Every target has:

*   `name`: how the target is referenced on the command line and by other
    targets
*   `srcs`: the source files to be compiled to create the artifact for the target
*   `deps`: other targets that must be built before this target and linked into
    it

Dependencies can either be within the same package (such as `MyBinary`’s
dependency on `:mylib`) or on a different package in the same source hierarchy
(such as `mylib`’s dependency on `//java/com/example/common`).

As with task-based build systems, you perform builds using Bazel’s command-line
tool. To build the `MyBinary` target, you run `bazel build :MyBinary`. After
entering that command for the first time in a clean repository, Bazel:

1.  Parses every `BUILD` file in the workspace to create a graph of dependencies
    among artifacts.
1.  Uses the graph to determine the transitive dependencies of `MyBinary`; that
    is, every target that `MyBinary` depends on and every target that those
    targets depend on, recursively.
1.  Builds each of those dependencies, in order. Bazel starts by building each
    target that has no other dependencies and keeps track of which dependencies
    still need to be built for each target. As soon as all of a target’s
    dependencies are built, Bazel starts building that target. This process
    continues until every one of `MyBinary`’s transitive dependencies have been
    built.
1.  Builds `MyBinary` to produce a final executable binary that links in all of
    the dependencies that were built in step 3.

Fundamentally, it might not seem like what’s happening here is that much
different than what happened when using a task-based build system. Indeed, the
end result is the same binary, and the process for producing it involved
analyzing a bunch of steps to find dependencies among them, and then running
those steps in order. But there are critical differences. The first one appears
in step 3: because Bazel knows that each target only produces a Java library, it
knows that all it has to do is run the Java compiler rather than an arbitrary
user-defined script, so it knows that it’s safe to run these steps in parallel.
This can produce an order of magnitude performance improvement over building
targets one at a time on a multicore machine, and is only possible because the
artifact-based approach leaves the build system in charge of its own execution
strategy so that it can make stronger guarantees about parallelism.

The benefits extend beyond parallelism, though. The next thing that this
approach gives us becomes apparent when the developer types `bazel
build :MyBinary` a second time without making any changes: Bazel exits in less
than a second with a message saying that the target is up to date. This is
possible due to the functional programming paradigm we talked about
earlier—Bazel knows that each target is the result only of running a Java
compiler, and it knows that the output from the Java compiler depends only on
its inputs, so as long as the inputs haven’t changed, the output can be reused.
And this analysis works at every level; if `MyBinary.java` changes, Bazel knows
to rebuild `MyBinary` but reuse `mylib`. If a source file for
`//java/com/example/common` changes, Bazel knows to rebuild that library,
`mylib`, and `MyBinary`, but reuse `//java/com/example/myproduct/otherlib`.
Because Bazel knows about the properties of the tools it runs at every step,
it’s able to rebuild only the minimum set of artifacts each time while
guaranteeing that it won’t produce stale builds.

Reframing the build process in terms of artifacts rather than tasks is subtle
but powerful. By reducing the flexibility exposed to the programmer, the build
system can know more about what is being done at every step of the build. It can
use this knowledge to make the build far more efficient by parallelizing build
processes and reusing their outputs. But this is really just the first step, and
these building blocks of parallelism and reuse form the basis for a distributed
and highly scalable build system.

## Other nifty Bazel tricks

Artifact-based build systems fundamentally solve the problems with parallelism
and reuse that are inherent in task-based build systems. But there are still a
few problems that came up earlier that we haven’t addressed. Bazel has clever
ways of solving each of these, and we should discuss them before moving on.

### Tools as dependencies

One problem we ran into earlier was that builds depended on the tools installed
on our machine, and reproducing builds across systems could be difficult due to
different tool versions or locations. The problem becomes even more difficult
when your project uses languages that require different tools based on which
platform they’re being built on or compiled for (such as, Windows versus Linux),
and each of those platforms requires a slightly different set of tools to do the
same job.

Bazel solves the first part of this problem by treating tools as dependencies to
each target. Every `java_library` in the workspace implicitly depends on a Java
compiler, which defaults to a well-known compiler. Whenever Bazel builds a
`java_library`, it checks to make sure that the specified compiler is available
at a known location. Just like any other dependency, if the Java compiler
changes, every artifact that depends on it is rebuilt.

Bazel solves the second part of the problem, platform independence, by setting
[build configurations](/run/build#build-config-cross-compilation). Rather than
targets depending directly on their tools, they depend on types of configurations:

*   **Host configuration**: building tools that run during the build
*   **Target configuration**: building the binary you ultimately requested

### Extending the build system

Bazel comes with targets for several popular programming languages out of the
box, but engineers will always want to do more—part of the benefit of task-based
systems is their flexibility in supporting any kind of build process, and it
would be better not to give that up in an artifact-based build system.
Fortunately, Bazel allows its supported target types to be extended by
[adding custom rules](/extending/rules).

To define a rule in Bazel, the rule author declares the inputs that the rule
requires (in the form of attributes passed in the `BUILD` file) and the fixed
set of outputs that the rule produces. The author also defines the actions that
will be generated by that rule. Each action declares its inputs and outputs,
runs a particular executable or writes a particular string to a file, and can be
connected to other actions via its inputs and outputs. This means that actions
are the lowest-level composable unit in the build system—an action can do
whatever it wants so long as it uses only its declared inputs and outputs, and
Bazel takes care of scheduling actions and caching their results as appropriate.

The system isn’t foolproof given that there’s no way to stop an action developer
from doing something like introducing a nondeterministic process as part of
their action. But this doesn’t happen very often in practice, and pushing the
possibilities for abuse all the way down to the action level greatly decreases
opportunities for errors. Rules supporting many common languages and tools are
widely available online, and most projects will never need to define their own
rules. Even for those that do, rule definitions only need to be defined in one
central place in the repository, meaning most engineers will be able to use
those rules without ever having to worry about their implementation.

### Isolating the environment

Actions sound like they might run into the same problems as tasks in other
systems—isn’t it still possible to write actions that both write to the same
file and end up conflicting with one another? Actually, Bazel makes these
conflicts impossible by using _[sandboxing](/docs/sandboxing)_. On supported
systems, every action is isolated from every other action via a filesystem
sandbox. Effectively, each action can see only a restricted view of the
filesystem that includes the inputs it has declared and any outputs it has
produced. This is enforced by systems such as LXC on Linux, the same technology
behind Docker. This means that it’s impossible for actions to conflict with one
another because they are unable to read any files they don’t declare, and any
files that they write but don’t declare will be thrown away when the action
finishes. Bazel also uses sandboxes to restrict actions from communicating via
the network.

### Making external dependencies deterministic

There’s still one problem remaining: build systems often need to download
dependencies (whether tools or libraries) from external sources rather than
directly building them. This can be seen in the example via the
`@com_google_common_guava_guava//jar` dependency, which downloads a `JAR` file
from Maven.

Depending on files outside of the current workspace is risky. Those files could
change at any time, potentially requiring the build system to constantly check
whether they’re fresh. If a remote file changes without a corresponding change
in the workspace source code, it can also lead to unreproducible builds—a build
might work one day and fail the next for no obvious reason due to an unnoticed
dependency change. Finally, an external dependency can introduce a huge security
risk when it is owned by a third party: if an attacker is able to infiltrate
that third-party server, they can replace the dependency file with something of
their own design, potentially giving them full control over your build
environment and its output.

The fundamental problem is that we want the build system to be aware of these
files without having to check them into source control. Updating a dependency
should be a conscious choice, but that choice should be made once in a central
place rather than managed by individual engineers or automatically by the
system. This is because even with a “Live at Head” model, we still want builds
to be deterministic, which implies that if you check out a commit from last
week, you should see your dependencies as they were then rather than as they are
now.

Bazel and some other build systems address this problem by requiring a
workspacewide manifest file that lists a _cryptographic hash_ for every external
dependency in the workspace. The hash is a concise way to uniquely represent the
file without checking the entire file into source control. Whenever a new
external dependency is referenced from a workspace, that dependency’s hash is
added to the manifest, either manually or automatically. When Bazel runs a
build, it checks the actual hash of its cached dependency against the expected
hash defined in the manifest and redownloads the file only if the hash differs.

If the artifact we download has a different hash than the one declared in the
manifest, the build will fail unless the hash in the manifest is updated. This
can be done automatically, but that change must be approved and checked into
source control before the build will accept the new dependency. This means that
there’s always a record of when a dependency was updated, and an external
dependency can’t change without a corresponding change in the workspace source.
It also means that, when checking out an older version of the source code, the
build is guaranteed to use the same dependencies that it was using at the point
when that version was checked in (or else it will fail if those dependencies are
no longer available).

Of course, it can still be a problem if a remote server becomes unavailable or
starts serving corrupt data—this can cause all of your builds to begin failing
if you don’t have another copy of that dependency available. To avoid this
problem, we recommend that, for any nontrivial project, you mirror all of its
dependencies onto servers or services that you trust and control. Otherwise you
will always be at the mercy of a third party for your build system’s
availability, even if the checked-in hashes guarantee its security.
