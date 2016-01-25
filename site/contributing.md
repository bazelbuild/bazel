---
layout: community
title: Contributing to Bazel
---

# Contributing to Bazel

<p class="lead">We welcome contributions! This page covers setting up your
machine to develop Bazel and, when you've made a patch, how to submit it.</p>

## How can I contribute to Bazel?

In general, we prefer contributions that fix bugs or add features (as opposed to
stylistic, refactoring, or "cleanup" changes). Please check with us on the
[dev list](https://groups.google.com/forum/#!forum/bazel-dev) before investing
a lot of time in a patch.

## Patch Acceptance Process

<!-- Our markdown parser doesn't support nested lists. -->
<ol>
<li>Read the <a href="governance.html">Bazel governance plan</a>.</li>
<li>Discuss your plan and design, and get agreement on our <a href="https://groups.google.com/forum/#!forum/bazel-dev">mailing list</a>.
<li>Prepare a git commit that implements the feature. Don't forget to add tests.
<li>Create a new code review on <a href="https://bazel-review.googlesource.com">Gerrit</a>
   by running:
   <pre>$ git push https://bazel.googlesource.com/bazel HEAD:refs/for/master</pre>
   Gerrit upload requires that you:
   <ul>
     <li>Have signed a
       <a href="https://cla.developers.google.com">Contributor License Agreement</a>.
     <li>Have an automatically generated "Change Id" line in your commit message.
       If you haven't used Gerrit before, it will print a bash command to create
       the git hook and then you will need to run `git commit --amend` to add the
       line.
   </ul>
   The HTTP password required by Gerrit can be obtained from your
   <a href="https://bazel-review.googlesource.com/#/settings/http-password">Gerrit settings page</a>.
   See the
   <a href="https://gerrit-review.googlesource.com/Documentation/user-upload.html">Gerrit documentation</a>
   for more information about uploading changes.
<li>Complete a code review with a
   <a href="governance.html#core-contributors">core contributor</a>. Amend your existing
   commit and re-push to make changes to your patch.
<li>An engineer at Google applies the patch to our internal version control
   system.
<li>The patch is exported as a Git commit, at which point the Gerrit code review
   is closed.
</ol>

## Setting up your coding environment

For now we have partial support for the Eclipse and IntelliJ IDEs for Java. We
don't have IDE support for other languages in Bazel right now.

### Creating an Eclipse project

To work with Eclipse, run `sh scripts/setup-eclipse.sh` from the root of the
source tree and it will create the `.project` and the `.classpath` files (if a
`.project` file is present, only the `.classpath` will get overwritten). You
can then import the project in Eclipse.

_You might see some errors in Eclipse concerning Truth assertions._

### Creating an IntelliJ project

To work with IntelliJ, run `sh scripts/setup-intellij.sh` from the root of the
source tree and it will create the necessary project files. You can then open
the folder as a project in IntelliJ.

<a name="compile-bazel"></a>
### Compiling Bazel

To test out bazel, you need to compile it. There are currently two ways of
compiling it:

* `sh compile.sh` bootstraps Bazel from scratch, first compiling it without using
  Bazel, then rebuilding it again using the just built Bazel and optionally runs
  tests, too. The resulting binary can be found at `output/bazel`.
* `bazel build //src:bazel` builds the Bazel binary using Bazel and the
  resulting binary can be found at `bazel-bin/src/bazel`. This is the recommended
  way of rebuilding Bazel once you have bootstrapped it.

In addition to the Bazel binary, you might want to build the various tools Bazel
uses. They are located in `//src/java_tools`, `//src/objc_tools` and
`//src/tools` and their directories contain README files describing their
respective utility.

When modifying Bazel, you want to make sure that the following still works:

* Bootstrap test with `sh compile.sh all` after having removed the
  `output` directory: it rebuilds Bazel with `./compile.sh`, Bazel with the
  `compile.sh` Bazel and Bazel with the Bazel-built binary. It compares if the
  constructed Bazel builts are identical and then runs all bazel tests with
  `bazel test //src/... //third_party/ijar/...`. This is what we use at Google
  to ensure that we don't break Bazel when pushing new commits, too.

### Debugging Bazel

Start creating a debug configuration for both C++ and Java in your bazelrc with the following:

```
build:debug -c dbg
build:debug --javacopt="-g"
build:debug --copt="-g"
build:debug --strip="never"
```

Then you can rebuild Bazel with `bazel build --config debug //src:bazel` and use your favorite
debugger to start debugging.

For debugging the C++ client you can just run it from gdb or lldb as you normally would.
But if you want to debug the Java code, you must attach to the server using the following:

* Run Bazel with debugging option `--host_jvm_debug` before the
  command (e.g., `bazel --batch --host_jvm_debug build //src:bazel`).
* Attach a debugger to the port 5005. With `jdb` for instance,
  run `jdb -attach localhost:5005`. From within Eclipse, use the
  [remote Java application launch
  configuration](http://help.eclipse.org/luna/index.jsp?topic=%2Forg.eclipse.jdt.doc.user%2Ftasks%2Ftask-remotejava_launch_config.htm).
  For IntelliJ, you can refer to [Run/Debug Configuration: Remote](https://www.jetbrains.com/idea/help/run-debug-configuration-remote.html).

## Bazel's code description

Bazel is organized in several parts:

* Client code in `src/main/cpp` provides the command-line interface.
* Protocol buffers in `src/main/protobuf`.
* Server code in `src/main/java` and `src/test/java`.
  * Core code which is mostly composed of [SkyFrame](docs/skyframe.html) and some
    utilities.
  * [Skylark](docs/skylark/index.html) rules are defined in `tools/build_rules`.
    If you want to add rules, consider using [Skylark](docs/skylark/index.html)
    first.
  * Builtin rules in `com.google.devtools.build.lib.rules` and in
    `com.google.devtools.build.lib.bazel.rules`.
* Java native interfaces in `src/main/native`.
* Various tooling for language support (see the list in the
  [compiling Bazel](#compile-bazel) section).
