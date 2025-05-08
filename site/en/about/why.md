Project: /_project.yaml
Book: /_book.yaml

# Why Bazel?

{% include "_buttons.html" %}

Bazel is a [fast](#fast), [correct](#correct), and [extensible](#extensible)
build tool with [integrated testing](#integrated-testing) that supports multiple
[languages](#multi-language), [repositories](#multi-repository), and
[platforms](#multi-platform) in an industry-leading [ecosystem](#ecosystem).

## Bazel is fast {:#fast}

Bazel knows exactly what input files each build command needs, avoiding
unnecessary work by re-running only when the set of input files have
changed between each build.
It runs build commands with as much parallelism as possible, either within the
same computer or on [remote build nodes](/remote/rbe). If the structure of build
allows for it, it can run thousands of build or test commands at the same time.

This is supported by multiple caching layers, in memory, on disk and on the
remote build farm, if available. At Google, we routinely achieve cache hit rates
north of 99%.

## Bazel is correct {:#correct}

Bazel ensures that your binaries are built *only* from your own
source code. Bazel actions run in individual sandboxes and Bazel tracks
every input file of the build, only and always re-running build
commands when it needs to. This keeps your binaries up-to-date so that the
[same source code always results in the same binary](/basics/hermeticity), bit
by bit.

Say goodbyte to endless `make clean` invocations and to chasing phantom bugs
that were in fact resolved in source code that never got built.

## Bazel is extensible {:#extensible}

Harness the full power of Bazel by writing your own rules and macros to
customize Bazel for your specific needs across a wide range of projects.

Bazel rules are written in [Starlark](/rules/language), our
in-house programming language that's a subset of Python. Starlark makes
rule-writing accessible to most developers, while also creating rules that can
be used across the ecosystem.

## Integrated testing {:#integrated-testing}

Bazel's [integrated test runner](/docs/user-manual#running-tests)
knows and runs only those tests needing to be re-run, using remote execution
(if available) to run them in parallel. Detect flakes early by using remote
execution to quickly run a test thousands of times.

Bazel [provides facilities](/remote/bep) to upload test results to a central
location, thereby facilitating efficient communication of test outcomes, be it
on CI or by individual developers.

## Multi-language support {:#multi-language}

Bazel supports many common programming languages including C++, Java,
Kotlin, Python, Go, and Rust. You can build multiple binaries (for example,
backend, web UI and mobile app) in the same Bazel invocation without being
constrained to one language's idiomatic build tool.

## Multi-repository support {:#multi-repository}

Bazel can [gather source code from multiple locations](/external/overview): you
don't need to vendor your dependencies (but you can!), you can instead point
Bazel to the location of your source code or prebuilt artifacts (e.g. a git
repository or Maven Central), and it takes care of the rest.

## Multi-platform support {:#multi-platform}

Bazel can simultaneously build projects for multiple platforms including Linux,
macOS, Windows, and Android. It also provides powerful
[cross-compilation capabilities](/extending/platforms) to build code for one
platform while running the build on another.

## Wide ecosystem {:#ecosystem}

[Industry leaders](/community/users) love Bazel, building a large
community of developers who use and contribute to Bazel. Find a tools, services
and documentation, including [consulting and SaaS offerings](/community/experts)
Bazel can use. Explore extensions like support for programming languages in
our [open source software repositories](/rules).