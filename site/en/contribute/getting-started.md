Project: /_project.yaml
Book: /_book.yaml

# Getting started

This page covers how to start contributing code to Bazel. It covers how to set up
your coding environment, describes creating an IntelliJ project and takes you
through compiling and debugging your project.

After you set up your environment, there is a quick overview of the structure of
Bazel's code base, how to search and navigate the code, and how to monitor your
builds with Bazel's continuous integration system.

## How can I contribute to Bazel source code? {:#how-contribute-code}

Bazel is a large project and making a change to Bazel source code
can be difficult.

Before making a change, [create a GitHub
issue](http://github.com/bazelbuild/bazel/issues){: .external}
or email [bazel-dev@](mailto:bazel-dev@googlegroups.com){: .external}.

The most helpful contributions fix bugs or add features (as opposed
to stylistic, refactoring, or "cleanup" changes). Your change should
include tests and documentation, keeping in mind backward-compatibility,
portability, and the impact on memory usage and performance.

To learn about how to submit a change, see the
[patch acceptance process](/contribute/patch-acceptance).

## Installing Bazel {:#install-bazel}

Before you start developing, you'll need to:

1.  Install the latest version of Bazel on your system. For instructions, see
    [Compiling Bazel from source](/install/compile-source).

1.  Clone Bazel's Git repository from GitHub:

    ```
    git clone https://github.com/bazelbuild/bazel.git
    ```
1. Install any missing [prerequisites](/install/compile-source#bootstrap-unix).

1. Try to [build
   Bazel](/contribute/guide#building-programs-with-bazel):

   *  On Linux/macOS, in Bash/Terminal:

      ```
      cd bazel
      bazel build //src:bazel
      ```

   *  On Windows, in the Command Prompt:

      ```
      cd bazel
      bazel --output_user_root=c:\tmp build //src:bazel.exe
      ```

      For faster iteration times (but larger binaries), use
      `//src:bazel-dev.exe` instead.

This produces a working Bazel binary in `bazel-bin/src/bazel`
(or `bazel-bin/src/bazel.exe` on Windows).

## Creating an IntelliJ project {:#intellij-project}

The IDE that Bazel supports is IntelliJ.

To work with IntelliJ:

1. Install Bazel's [IntelliJ plug-in](https://ij.bazel.build).
1. Set the path to the Bazel binary in the plugin preferences
   (`Preferences` > `Other Settings` > `Bazel Settings`).
1. Import the Bazel workspace as a Bazel project
   (`File` > `Import Bazel Project...`) with the following settings:
   *  Use existing Bazel workspace: choose your cloned Git repository.
   *  Select `Import from workspace` and choose the `scripts/ij.bazelproject`
      file as the `Project view`.
1. Download [Google's Java Code Style Scheme file for IntelliJ](https://github.com/google/styleguide/blob/gh-pages/intellij-java-google-style.xml){: .external},
   import it (go to `Preferences` > `Editor` > `Code Style` > `Java`, click
   `Manage`, then `Import`) and use it when working on Bazel's code.

## Compiling Bazel {:#compile-bazel}

You need to compile Bazel in order to test it. To compile a development version
of Bazel, you need the latest released version of Bazel, which can be
[compiled from source](/install/compile-source).

You can build the Bazel binary using
`bazel build //src:bazel`, using `bazel` from your PATH.
The resulting binary can be found at `bazel-bin/src/bazel`. This is the
recommended way of rebuilding Bazel once you have bootstrapped it.

In addition to the Bazel binary, you might want to build the various tools Bazel
uses. They are located in `//src/java_tools/...`, `//src/objc_tools/...` and
`//src/tools/...` and their directories contain README files describing their
respective utility.

When modifying Bazel:

1. Build a distribution archive with `bazel build //:bazel-distfile`.
1. Unzip the archive in a new empty directory.
1. Run `bash compile.sh all` there.

This command rebuilds Bazel with `./compile.sh`, Bazel with
`compile.sh` and Bazel with the Bazel-built binary. It compares if the
constructed Bazel builts are identical and then runs all Bazel tests with
`bazel test //src/... //third_party/ijar/...`. This is also used internally
to ensure that we don't break Bazel when pushing new commits.

## Debugging Bazel {:#debug-bazel}

To create a debug configuration for both C++ and Java in your `.bazelrc`
use:

```
build:debug -c dbg
build:debug --javacopt="-g"
build:debug --copt="-g"
build:debug --strip="never"
```

Rebuild Bazel with `bazel build --config debug //src:bazel` and use
your favorite debugger to start debugging.

To debug the C++ client, run it from `gdb`or `lldb` as usual.
However, to debug Java code, attach to the server using the following:

*  Run Bazel with the debugging option `--host_jvm_debug` before the
   command (such as `bazel --host_jvm_debug build //src:bazel`).
*  Attach a debugger to the port 5005. For instance, with `jdb`,
   run `jdb -attach localhost:5005`.

Our IntelliJ plugin has built-in [debugging support](https://ij.bazel.build/docs/run-configurations.html).

## Using Bazel Continuous Integration {:#continuous-integration}

To get started with the Bazel CI system, see
[Bazel Continuous Integration](https://github.com/bazelbuild/continuous-integration/blob/master/buildkite/README.md){: .external}
To monitor the tests and builds of your Bazel contributions, use the
[Bazel CI Dashboard](https://ci.bazel.build/).

## Bazel's code description {:#code-description}

Bazel has a large codebase with code in multiple locations.

Bazel is organized as follows:

*  Client code is in `src/main/cpp` and provides the command-line interface.
*  Protocol buffers are in `src/main/protobuf`.
*  Server code is in `src/main/java` and `src/test/java`.
   *  Core code which is mostly composed of [SkyFrame](/reference/skyframe)
      and some utilities.
   *  Built-in rules are in `com.google.devtools.build.lib.rules` and in
     `com.google.devtools.build.lib.bazel.rules`. You might want to read about
     the [Challenges of Writing Rules](/docs/rule-challenges) first.
*  Java native interfaces are in `src/main/native`.
*  Various tooling for language support are described in the list in the
   [compiling Bazel](#compile-bazel) section.

For a detailed description of the code base, read about
[the Bazel codebase](codebase.md).

### Searching Bazel's source code {:#search-code}

To quickly search through Bazel's source code, use
[Bazel Code Search](https://source.bazel.build/). You can navigate Bazel's
repositories, branches, and files. You can also view history, diffs, and blame
information. To learn more, see the
[Bazel Code Search User Guide](/contribute/searching-codebase).
