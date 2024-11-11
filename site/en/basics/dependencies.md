Project: /_project.yaml
Book: /_book.yaml

# Dependency Management

{% include "_buttons.html" %}

In looking through the previous pages, one theme repeats over and over: managing
your own code is fairly straightforward, but managing its dependencies is much
more difficult. There are all sorts of dependencies: sometimes there’s a
dependency on a task (such as “push the documentation before I mark a release as
complete”), and sometimes there’s a dependency on an artifact (such as “I need
to have the latest version of the computer vision library to build my code”).
Sometimes, you have internal dependencies on another part of your codebase, and
sometimes you have external dependencies on code or data owned by another team
(either in your organization or a third party). But in any case, the idea of “I
need that before I can have this” is something that recurs repeatedly in the
design of build systems, and managing dependencies is perhaps the most
fundamental job of a build system.

## Dealing with Modules and Dependencies

Projects that use artifact-based build systems like Bazel are broken into a set
of modules, with modules expressing dependencies on one another via `BUILD`
files. Proper organization of these modules and dependencies can have a huge
effect on both the performance of the build system and how much work it takes to
maintain.

## Using Fine-Grained Modules and the 1:1:1 Rule

The first question that comes up when structuring an artifact-based build is
deciding how much functionality an individual module should encompass. In Bazel,
a _module_ is represented by a target specifying a buildable unit like a
`java_library` or a `go_binary`. At one extreme, the entire project could be
contained in a single module by putting one `BUILD` file at the root and
recursively globbing together all of that project’s source files. At the other
extreme, nearly every source file could be made into its own module, effectively
requiring each file to list in a `BUILD` file every other file it depends on.

Most projects fall somewhere between these extremes, and the choice involves a
trade-off between performance and maintainability. Using a single module for the
entire project might mean that you never need to touch the `BUILD` file except
when adding an external dependency, but it means that the build system must
always build the entire project all at once. This means that it won’t be able to
parallelize or distribute parts of the build, nor will it be able to cache parts
that it’s already built. One-module-per-file is the opposite: the build system
has the maximum flexibility in caching and scheduling steps of the build, but
engineers need to expend more effort maintaining lists of dependencies whenever
they change which files reference which.

Though the exact granularity varies by language (and often even within
language), Google tends to favor significantly smaller modules than one might
typically write in a task-based build system. A typical production binary at
Google often depends on tens of thousands of targets, and even a moderate-sized
team can own several hundred targets within its codebase. For languages like
Java that have a strong built-in notion of packaging, each directory usually
contains a single package, target, and `BUILD` file (Pants, another build system
based on Bazel, calls this the 1:1:1 rule). Languages with weaker packaging
conventions frequently define multiple targets per `BUILD` file.

The benefits of smaller build targets really begin to show at scale because they
lead to faster distributed builds and a less frequent need to rebuild targets.
The advantages become even more compelling after testing enters the picture, as
finer-grained targets mean that the build system can be much smarter about
running only a limited subset of tests that could be affected by any given
change. Because Google believes in the systemic benefits of using smaller
targets, we’ve made some strides in mitigating the downside by investing in
tooling to automatically manage `BUILD` files to avoid burdening developers.

Some of these tools, such as `buildifier` and `buildozer`, are available with
Bazel in the [`buildtools`
directory](https://github.com/bazelbuild/buildtools){: .external}.

## Minimizing Module Visibility

Bazel and other build systems allow each target to specify a visibility — a
property that determines which other targets may depend on it. A private target
can only be referenced within its own `BUILD` file. A target may grant broader
visibility to the targets of an explicitly defined list of `BUILD` files, or, in
the case of public visibility, to every target in the workspace.

As with most programming languages, it is usually best to minimize visibility as
much as possible. Generally, teams at Google will make targets public only if
those targets represent widely used libraries available to any team at Google.
Teams that require others to coordinate with them before using their code will
maintain an allowlist of customer targets as their target’s visibility. Each
team’s internal implementation targets will be restricted to only directories
owned by the team, and most `BUILD` files will have only one target that isn’t
private.

## Managing Dependencies

Modules need to be able to refer to one another. The downside of breaking a
codebase into fine-grained modules is that you need to manage the dependencies
among those modules (though tools can help automate this). Expressing these
dependencies usually ends up being the bulk of the content in a `BUILD` file.

### Internal dependencies

In a large project broken into fine-grained modules, most dependencies are
likely to be internal; that is, on another target defined and built in the same
source repository. Internal dependencies differ from external dependencies in
that they are built from source rather than downloaded as a prebuilt artifact
while running the build. This also means that there’s no notion of “version” for
internal dependencies—a target and all of its internal dependencies are always
built at the same commit/revision in the repository. One issue that should be
handled carefully with regard to internal dependencies is how to treat
transitive dependencies (Figure 1). Suppose target A depends on target B, which
depends on a common library target C. Should target A be able to use classes
defined in target C?

[![Transitive
dependencies](/images/transitive-dependencies.png)](/images/transitive-dependencies.png)

**Figure 1**. Transitive dependencies

As far as the underlying tools are concerned, there’s no problem with this; both
B and C will be linked into target A when it is built, so any symbols defined in
C are known to A. Bazel allowed this for many years, but as Google grew, we
began to see problems. Suppose that B was refactored such that it no longer
needed to depend on C. If B’s dependency on C was then removed, A and any other
target that used C via a dependency on B would break. Effectively, a target’s
dependencies became part of its public contract and could never be safely
changed. This meant that dependencies accumulated over time and builds at Google
started to slow down.

Google eventually solved this issue by introducing a “strict transitive
dependency mode” in Bazel. In this mode, Bazel detects whether a target tries to
reference a symbol without depending on it directly and, if so, fails with an
error and a shell command that can be used to automatically insert the
dependency. Rolling this change out across Google’s entire codebase and
refactoring every one of our millions of build targets to explicitly list their
dependencies was a multiyear effort, but it was well worth it. Our builds are
now much faster given that targets have fewer unnecessary dependencies, and
engineers are empowered to remove dependencies they don’t need without worrying
about breaking targets that depend on them.

As usual, enforcing strict transitive dependencies involved a trade-off. It made
build files more verbose, as frequently used libraries now need to be listed
explicitly in many places rather than pulled in incidentally, and engineers
needed to spend more effort adding dependencies to `BUILD` files. We’ve since
developed tools that reduce this toil by automatically detecting many missing
dependencies and adding them to a `BUILD` files without any developer
intervention. But even without such tools, we’ve found the trade-off to be well
worth it as the codebase scales: explicitly adding a dependency to `BUILD` file
is a one-time cost, but dealing with implicit transitive dependencies can cause
ongoing problems as long as the build target exists. Bazel [enforces strict
transitive
dependencies](https://blog.bazel.build/2017/06/28/sjd-unused_deps.html){: .external}
on Java code by default.

### External dependencies

If a dependency isn’t internal, it must be external. External dependencies are
those on artifacts that are built and stored outside of the build system. The
dependency is imported directly from an artifact repository (typically accessed
over the internet) and used as-is rather than being built from source. One of
the biggest differences between external and internal dependencies is that
external dependencies have versions, and those versions exist independently of
the project’s source code.

### Automatic versus manual dependency management

Build systems can allow the versions of external dependencies to be managed
either manually or automatically. When managed manually, the buildfile
explicitly lists the version it wants to download from the artifact repository,
often using a [semantic version string](https://semver.org/){: .external} such
as `1.1.4`. When managed automatically, the source file specifies a range of
acceptable versions, and the build system always downloads the latest one. For
example, Gradle allows a dependency version to be declared as “1.+” to specify
that any minor or patch version of a dependency is acceptable so long as the
major version is 1.

Automatically managed dependencies can be convenient for small projects, but
they’re usually a recipe for disaster on projects of nontrivial size or that are
being worked on by more than one engineer. The problem with automatically
managed dependencies is that you have no control over when the version is
updated. There’s no way to guarantee that external parties won’t make breaking
updates (even when they claim to use semantic versioning), so a build that
worked one day might be broken the next with no easy way to detect what changed
or to roll it back to a working state. Even if the build doesn’t break, there
can be subtle behavior or performance changes that are impossible to track down.

In contrast, because manually managed dependencies require a change in source
control, they can be easily discovered and rolled back, and it’s possible to
check out an older version of the repository to build with older dependencies.
Bazel requires that versions of all dependencies be specified manually. At even
moderate scales, the overhead of manual version management is well worth it for
the stability it provides.

### The One-Version Rule

Different versions of a library are usually represented by different artifacts,
so in theory there’s no reason that different versions of the same external
dependency couldn’t both be declared in the build system under different names.
That way, each target could choose which version of the dependency it wanted to
use. This causes a lot of problems in practice, so Google enforces a strict
[One-Version
Rule](https://opensource.google/docs/thirdparty/oneversion/){: .external} for
all third-party dependencies in our codebase.

The biggest problem with allowing multiple versions is the diamond dependency
issue. Suppose that target A depends on target B and on v1 of an external
library. If target B is later refactored to add a dependency on v2 of the same
external library, target A will break because it now depends implicitly on two
different versions of the same library. Effectively, it’s never safe to add a
new dependency from a target to any third-party library with multiple versions,
because any of that target’s users could already be depending on a different
version. Following the One-Version Rule makes this conflict impossible—if a
target adds a dependency on a third-party library, any existing dependencies
will already be on that same version, so they can happily coexist.

### Transitive external dependencies

Dealing with the transitive dependencies of an external dependency can be
particularly difficult. Many artifact repositories such as Maven Central, allow
artifacts to specify dependencies on particular versions of other artifacts in
the repository. Build tools like Maven or Gradle often recursively download each
transitive dependency by default, meaning that adding a single dependency in
your project could potentially cause dozens of artifacts to be downloaded in
total.

This is very convenient: when adding a dependency on a new library, it would be
a big pain to have to track down each of that library’s transitive dependencies
and add them all manually. But there’s also a huge downside: because different
libraries can depend on different versions of the same third-party library, this
strategy necessarily violates the One-Version Rule and leads to the diamond
dependency problem. If your target depends on two external libraries that use
different versions of the same dependency, there’s no telling which one you’ll
get. This also means that updating an external dependency could cause seemingly
unrelated failures throughout the codebase if the new version begins pulling in
conflicting versions of some of its dependencies.

Bazel did not use to automatically download transitive dependencies. It used to
employ a `WORKSPACE` file that required all transitive dependencies to be
listed, which led to a lot of pain when managing external dependencies. Bazel
has since added support for automatic transitive external dependency management
in the form of the `MODULE.bazel` file. See [external dependency
overview](/external/overview) for more details.

Yet again, the choice here is one between convenience and scalability. Small
projects might prefer not having to worry about managing transitive dependencies
themselves and might be able to get away with using automatic transitive
dependencies. This strategy becomes less and less appealing as the organization
and codebase grows, and conflicts and unexpected results become more and more
frequent. At larger scales, the cost of manually managing dependencies is much
less than the cost of dealing with issues caused by automatic dependency
management.

### Caching build results using external dependencies

External dependencies are most often provided by third parties that release
stable versions of libraries, perhaps without providing source code. Some
organizations might also choose to make some of their own code available as
artifacts, allowing other pieces of code to depend on them as third-party rather
than internal dependencies. This can theoretically speed up builds if artifacts
are slow to build but quick to download.

However, this also introduces a lot of overhead and complexity: someone needs to
be responsible for building each of those artifacts and uploading them to the
artifact repository, and clients need to ensure that they stay up to date with
the latest version. Debugging also becomes much more difficult because different
parts of the system will have been built from different points in the
repository, and there is no longer a consistent view of the source tree.

A better way to solve the problem of artifacts taking a long time to build is to
use a build system that supports remote caching, as described earlier. Such a
build system saves the resulting artifacts from every build to a location that
is shared across engineers, so if a developer depends on an artifact that was
recently built by someone else, the build system automatically downloads it
instead of building it. This provides all of the performance benefits of
depending directly on artifacts while still ensuring that builds are as
consistent as if they were always built from the same source. This is the
strategy used internally by Google, and Bazel can be configured to use a remote
cache.

### Security and reliability of external dependencies

Depending on artifacts from third-party sources is inherently risky. There’s an
availability risk if the third-party source (such as an artifact repository)
goes down, because your entire build might grind to a halt if it’s unable to
download an external dependency. There’s also a security risk: if the
third-party system is compromised by an attacker, the attacker could replace the
referenced artifact with one of their own design, allowing them to inject
arbitrary code into your build. Both problems can be mitigated by mirroring any
artifacts you depend on onto servers you control and blocking your build system
from accessing third-party artifact repositories like Maven Central. The
trade-off is that these mirrors take effort and resources to maintain, so the
choice of whether to use them often depends on the scale of the project. The
security issue can also be completely prevented with little overhead by
requiring the hash of each third-party artifact to be specified in the source
repository, causing the build to fail if the artifact is tampered with. Another
alternative that completely sidesteps the issue is to vendor your project’s
dependencies. When a project vendors its dependencies, it checks them into
source control alongside the project’s source code, either as source or as
binaries. This effectively means that all of the project’s external dependencies
are converted to internal dependencies. Google uses this approach internally,
checking every third-party library referenced throughout Google into a
`third_party` directory at the root of Google’s source tree. However, this works
at Google only because Google’s source control system is custom built to handle
an extremely large monorepo, so vendoring might not be an option for all
organizations.