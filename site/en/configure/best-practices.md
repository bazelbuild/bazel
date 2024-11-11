Project: /_project.yaml
Book: /_book.yaml

# Best Practices

{% include "_buttons.html" %}

This page assumes you are familiar with Bazel and provides guidelines and
advice on structuring your projects to take full advantage of Bazel's features.

The overall goals are:

- To use fine-grained dependencies to allow parallelism and incrementality.
- To keep dependencies well-encapsulated.
- To make code well-structured and testable.
- To create a build configuration that is easy to understand and maintain.

These guidelines are not requirements: few projects will be able to adhere to
all of them.  As the man page for lint says, "A special reward will be presented
to the first person to produce a real program that produces no errors with
strict checking." However, incorporating as many of these principles as possible
should make a project more readable, less error-prone, and faster to build.

This page uses the requirement levels described in
[this RFC](https://www.ietf.org/rfc/rfc2119.txt){: .external}.

## Running builds and tests {:#running-builds-tests}

A project should always be able to run `bazel build //...` and
`bazel test //...` successfully on its stable branch. Targets that are necessary
but do not build under certain circumstances (such as,require specific build
flags, don't build on a certain platform, require license agreements) should be
tagged as specifically as possible (for example, "`requires-osx`"). This
tagging allows targets to be filtered at a more fine-grained level than the
"manual" tag and allows someone inspecting the `BUILD` file to understand what
a target's restrictions are.

## Third-party dependencies {:#third-party-dependencies}

You may declare third-party dependencies:

*   Either declare them as remote repositories in the `MODULE.bazel` file.
*   Or put them in a directory called `third_party/` under your workspace directory.

## Depending on binaries {:#binaries}

Everything should be built from source whenever possible. Generally this means
that, instead of depending on a library `some-library.so`, you'd create a
`BUILD` file and build `some-library.so` from its sources, then depend on that
target.

Always building from source ensures that a build is not using a library that
was built with incompatible flags or a different architecture. There are also
some features like coverage, static analysis, or dynamic analysis that only
work on the source.

## Versioning {:#versioning}

Prefer building all code from head whenever possible. When versions must be
used, avoid including the version in the target name (for example, `//guava`,
not `//guava-20.0`). This naming makes the library easier to update (only one
target needs to be updated). It's also more resilient to diamond dependency
issues: if one library depends on `guava-19.0` and one depends on `guava-20.0`,
you could end up with a library that tries to depend on two different versions.
If you created a misleading alias to point both targets to one `guava` library,
then the `BUILD` files are misleading.

## Using the `.bazelrc` file {:#bazelrc-file}

For project-specific options, use the configuration file your
`{{ '<var>' }}workspace{{ '</var>' }}/.bazelrc` (see [bazelrc format](/run/bazelrc)).

If you want to support per-user options for your project that you **do not**
want to check into source control, include the line:

```
try-import %workspace%/user.bazelrc
```
(or any other file name) in your `{{ '<var>' }}workspace{{ '</var>' }}/.bazelrc`
and add `user.bazelrc` to your `.gitignore`.

## Packages {:#packages}

Every directory that contains buildable files should be a package. If a `BUILD`
file refers to files in subdirectories (such as, `srcs = ["a/b/C.java"]`) it's
a sign that a `BUILD` file should be added to that subdirectory. The longer
this structure exists, the more likely circular dependencies will be
inadvertently created, a target's scope will creep, and an increasing number
of reverse dependencies will have to be updated.
