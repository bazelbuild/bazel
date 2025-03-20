Project: /_project.yaml
Book: /_book.yaml

# Hermeticity

{% include "_buttons.html" %}

This page covers hermeticity, the benefits of using hermetic builds, and
strategies for identifying non-hermetic behavior in your builds.

## Overview {:#overview}

When given the same input source code and product configuration, a hermetic
build system always returns the same output by isolating the build from changes
to the host system.

In order to isolate the build, hermetic builds are insensitive to libraries and
other software installed on the local or remote host machine. They depend on
specific versions of build tools, such as compilers, and dependencies, such as
libraries. This makes the build process self-contained as it doesn't rely on
services external to the build environment.

The two important aspects of hermeticity are:

* **Isolation**: Hermetic build systems treat tools as source code. They
  download copies of tools and manage their storage and use inside managed file
  trees. This creates isolation between the host machine and local user,
  including installed versions of languages.
* **Source identity**: Hermetic build systems try to ensure the sameness of
  inputs. Code repositories, such as Git, identify sets of code mutations with a
  unique hash code. Hermetic build systems use this hash to identify changes to
  the build's input.

## Benefits {:#benefits}

The major benefits of hermetic builds are:

* **Speed**: The output of an action can be cached, and the action need not be
  run again unless inputs change.
* **Parallel execution**: For given input and output, the build system can
  construct a graph of all actions to calculate efficient and parallel
  execution. The build system loads the rules and calculates an action graph
  and hash inputs to look up in the cache.
* **Multiple builds**: You can build multiple hermetic builds on the same
  machine, each build using different tools and versions.
* **Reproducibility**: Hermetic builds are good for troubleshooting because you
  know the exact conditions that produced the build.

## Identifying non-hermeticity {:#nonhermeticity}

If you are preparing to switch to Bazel, migration is easier if you improve
your existing builds' hermeticity in advance. Some common sources of
non-hermeticity in builds are:

* Arbitrary processing in `.mk` files
* Actions or tooling that create files non-deterministically, usually involving
  build IDs or timestamps
* System binaries that differ across hosts (such as `/usr/bin` binaries, absolute
  paths, system C++ compilers for native C++ rules autoconfiguration)
* Writing to the source tree during the build. This prevents the same source
  tree from being used for another target. The first build writes to the source
  tree, fixing the source tree for target A. Then trying to build target B may
  fail.

## Troubleshooting non-hermetic builds {:#troubleshooting-nonhermeticity}

Starting with local execution, issues that affect local cache hits reveal
non-hermetic actions.

* Ensure null sequential builds: If you run `make` and get a successful build,
  running the build again should not rebuild any targets. If you run each build
  step twice or on different systems, compare a hash of the file contents and
  get results that differ, the build is not reproducible.
* Run steps to
  [debug local cache hits](/remote/cache-remote#troubleshooting-cache-hits)
  from a variety of potential client machines to ensure that you catch any
  cases of client environment leaking into the actions.
* Execute a build within a docker container that contains nothing but the
  checked-out source tree and explicit list of host tools. Build breakages and
  error messages will catch implicit system dependencies.
* Discover and fix hermeticity problems using
  [remote execution rules](/remote/rules#overview).
* Enable strict [sandboxing](/docs/sandboxing)
  at the per-action level, since actions in a build can be stateful and affect
  the build or the output.
* [Workspace rules](/remote/workspace)
  allow developers to add dependencies to external workspaces, but they are
  rich enough to allow arbitrary processing to happen in the process. You can
  get a log of some potentially non-hermetic actions in Bazel workspace rules by
  adding the flag
  `--experimental_workspace_rules_log_file={{ '<var>' }}PATH{{ '</var>' }}` to
  your Bazel command.

Note: Make your build fully hermetic when mixing remote and local execution,
using Bazel’s “dynamic strategy” functionality. Running Bazel inside the remote
Docker container will enable the build to execute the same in both environments.

## Hermeticity with Bazel {:#hermeticity-bazel}

For more information about how other projects have had success using hermetic
builds with Bazel, see these  BazelCon talks:

*   [Building Real-time Systems with Bazel](https://www.youtube.com/watch?v=t_3bckhV_YI){: .external} (SpaceX)
*   [Bazel Remote Execution and Remote Caching](https://www.youtube.com/watch?v=_bPyEbAyC0s){: .external} (Uber and TwoSigma)
*   [Faster Builds With Remote Execution and Caching](https://www.youtube.com/watch?v=MyuJRUwT5LI){: .external}
*   [Fusing Bazel: Faster Incremental Builds](https://www.youtube.com/watch?v=rQd9Zd1ONOw){: .external}
*   [Remote Execution vs Local Execution](https://www.youtube.com/watch?v=C8wHmIln--g){: .external}
*   [Improving the Usability of Remote Caching](https://www.youtube.com/watch?v=u5m7V3ZRHLA){: .external} (IBM)
*   [Building Self Driving Cars with Bazel](https://www.youtube.com/watch?v=Gh4SJuYUoQI&list=PLxNYxgaZ8Rsf-7g43Z8LyXct9ax6egdSj&index=4&t=0s){: .external} (BMW)
*   [Building Self Driving Cars with Bazel + Q&A](https://www.youtube.com/watch?v=fjfFe98LTm8&list=PLxNYxgaZ8Rsf-7g43Z8LyXct9ax6egdSj&index=29){: .external} (GM Cruise)
