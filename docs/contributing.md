# Contributing to Bazel

## How can I contribute to Bazel?

You should first read the [Bazel governance plan](governance.md). Then you can
read this page that explains how to set-up and work on the Bazel code and how
to submit a patch to Bazel.

## Setting-up your coding environment

For now we only have partial support for the Eclipse IDE for Java. We don't have
IDE support for others languages in Bazel right now.

### Creating an Eclipse project

To work with Eclipse, run from the root of the source tree `sh
scripts/setup-eclipse.sh` and it will create the `.project` and the `.classpath`
files (if a `.project` file is present, only the `.classpath` will get
overwritten). You can then import the project in Eclipse.

_You might see some errors in Eclipse concerning Truth assertions_

### Compiling Bazel {#compile-bazel}

To test out bazel, you need to compile it. There is currently two ways of
compiling it:

* `sh compile.sh` build a Bazel binary without Bazel, it should only be used to
  bootstrap Bazel itself. The resulting binary can be found at `output/bazel`.
* `bazel build //src:bazel` builds the Bazel binary using Bazel and the
  resulting binary can be found at `bazel-bin/src/bazel`.

In addition to the Bazel binary, you might want to build the various tools Bazel
uses:

* For Java support
  * JavaBuilder is the java compiler wrapper used by Bazel and its target can be
    found at `//src/java_tools/buildjar:JavaBuilder_deploy.jar`.
  * SingleJar is a tool to assemble a single jar composed of all classes from
    all jar dependencies (build the `*_deploy.jar` files), it can be found at
    `//src/java_tools/singlejar:SingleJar_deploy.jar`.
  * ijar is a tool to extracts the class interfaces of jars and is a third
    party software at `//third_party/ijar`.
* For Objective-C / iOS support
  * TODO(bazel-team): add tools description

When modifying Bazel, you want to make sure that the following still works:

* Bootstrap test with `sh bootstrap_test.sh all` after having removed the
  `output` directory: it rebuilds Bazel with `./compile.sh`, Bazel with the
  `compile.sh` Bazel and Bazel with the Bazel-built binary. It compares if the
  constructed Bazel builts are identical and then run all bazel tests with
  `bazel test //src/...`.
* ijar's tests with `bazel test //third_party/ijar/test/...`
* Building testing the examples in the `base_workspace` directory.
  `bazel test --nobuild_tests_only //...` in that directory should build and test
  everything (you might need to set-up extra tools following the instructions in
  the README files for each example).

## Bazel's code description

Bazel is organized in several parts:

* A client code in C++ that talk to the server code in java and provide the
  command-line interface. Its code is in `src/main/cpp`.
* Protocol buffers in `src/main/protobuf`.
* The server code in Java (in `src/main/java` and `src/test/java`)
  * Core code which is mostly composed of [SkyFrame](skyframe.md) and some
    utilities.
  * [Skylark](skylark/index.md) rules are defined in `tools/build_rules`. If you
    want to add rules, consider using [Skylark](skylark/index.md) first.
  * Builtin rules in `com.google.devtools.build.lib.rules` and in
    `com.google.devtools.build.lib.bazel.rules`.
* Java native interfaces in `src/main/native`.
* Various tooling for language support (see the list in the
  [compiling Bazel](#compile-bazel) section.

## Patch Acceptance Process

1. Discuss your plan and design, and get agreement on our [mailing
   list](https://groups.google.com/forum/#!forum/bazel-discuss).
2. Prepare a patch that implements the feature. Don't forget to add tests.
3. Upload to [Gerrit](https://bazel-review.googlesource.com); Gerrit upload
   requires that you have signed a
   [Contributor License Agreement](https://cla.developers.google.com/).
4. Complete a code review with a core contributor.
5. An engineer at Google applies the patch to our internal version control
   system.
6. The patch is exported as a Git commit, at which point the Gerrit code review
   is closed.

We do not currently accept pull requests on GitHub.

We will make changes to this process as necessary, and we're hoping to move
closer to a fully open development model in the future (also see
[Is Bazel developed fully in the open?](governance.md#isbazelopen)).
