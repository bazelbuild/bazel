# [Bazel (Alpha)](http://bazel.io)

*{Fast, Correct} - Choose two*

Bazel is a build tool that builds code quickly and reliably. It is used to build
the majority of Google's software, and thus it has been designed to handle
build problems present in Google's development environment, including:

* **A massive, shared code repository, in which all software is built from
source.** Bazel has been built for speed, using both caching and parallelism
to achieve this. Bazel is critical to Google's ability to continue
to scale its software development practices as the company grows.

* **An emphasis on automated testing and releases.** Bazel has
been built for correctness and reproducibility, meaning that a build performed
on a continuous build machine or in a release pipeline will generate
bitwise-identical outputs to those generated on a developer's machine.

* **Language and platform diversity.** Bazel's architecture is general enough to
support many different programming languages within Google, and can be
used to build both client and server software targeting multiple
architectures from the same underlying codebase.

Find more background about Bazel in our [FAQ](docs/FAQ.md)

## Getting Started

  * How to [install Bazel](docs/install.md)
  * How to [get started using Bazel](docs/getting-started.md)
  * The Bazel command line is documented in the  [user manual](docs/bazel-user-manual.html)
  * The rule reference documentation is in the [build encyclopedia](docs/build-encyclopedia.html)
  * How to [use the query command](docs/query.html)
  * How to [extend Bazel](docs/skylark/index.md)
  * The test environment is described in the [test encyclopedia](docs/test-encyclopedia.html)

* About the Bazel project:

  * How to [contribute to Bazel](docs/contributing.md)
  * Our [governance plan](docs/governance.md)
  * Future plans are in the [roadmap](docs/roadmap.md)
  * For each feature, which level of [support](docs/support.md) to expect.
