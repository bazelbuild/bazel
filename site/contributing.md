---
layout: contribute
title: Contributing to Bazel
---

# Contributing to Bazel

<p class="lead">We welcome contributions! This page covers setting up your
machine to develop Bazel and, when you've made a patch, how to submit it.</p>

## How can I contribute to Bazel?

In general, we prefer contributions that fix bugs or add features (as opposed to
stylistic, refactoring, or "cleanup" changes).

Please check with us on the
[dev list](https://groups.google.com/forum/#!forum/bazel-dev) before investing
a lot of time in a patch. Meet other Bazel contributors on [IRC](http://webchat.freenode.net)
(irc.freenode.net#bazel).

### Patch Acceptance Process


1. Read the [Bazel governance plan](governance.html).
2. Discuss your plan and design, and get agreement on our [mailing list](https://groups.google.com/forum/#!forum/bazel-dev).
3.  Prepare a git commit that implements the feature. Don't forget to add tests.
4.  Ensure you've signed a [Contributor License Agreement](https://cla.developers.google.com).
5.  Create a new code review. You can use a GitHub pull request, or a code
    review on [Gerrit](https://bazel-review.googlesource.com).
    *  To use GitHub, send a pull request. If you're new to GitHub, read [about
       pull requests](https://help.github.com/articles/about-pull-requests/).
    *  To use Gerrit, you must:
       *  Have an automatically generated "Change Id" line in your commit
          message. If you haven't used Gerrit before, it will print a bash
          command to create the git hook and then you will need to run
          `git commit --amend` to add the line.
       *  To create a code review on Gerrit, run:

          ```
          git push https://bazel.googlesource.com/bazel HEAD:refs/for/master
          ```

          The HTTP password required by Gerrit can be obtained from your
          [Gerrit settings page](https://bazel-review.googlesource.com/#/settings/http-password).
          See the [Gerrit documentation](https://gerrit-review.googlesource.com/Documentation/user-upload.html)
          for more information about uploading changes.
6.  Wait for a Bazel team member to assign you a reviewer.
    It should be done in 2 business days (excluding holidays in the USA and
    Germany). If you do not get a reviewer within that time frame, you can ask
    for one by sending a mail to [bazel-sheriff@googlegroups.com](mailto:bazel-sheriff@googlegroups.com).
    You can also assign yourself a reviewer if you know who the reviewer should
    be (e.g., because they reviewed an earlier related change).
7.  Complete a code review. Amend your existing commit and re-push to make
    changes to your patch.
8.  An engineer at Google applies the patch to our internal version control
    system. The patch is exported as a Git commit, at which point the GitHub
    or Gerrit code review is closed.

## Setting up your coding environment

For now we have support for IntelliJ, and partial support for the Eclipse IDE
for Java. We don't have IDE support for other languages in Bazel right now.

### Preparations

*  [Install Bazel](https://bazel.build/versions/master/docs/install.html) on your
   system. Note that for developing Bazel, you need the latest released version
   of Bazel.
*  Clone Bazel's Git repository from Gerrit:
   *  `git clone https://bazel.googlesource.com/bazel`
*  Try to build Bazel:
   *  `cd bazel && bazel build //src:bazel`
*  This should produce a working Bazel binary in `bazel-bin/src/bazel`.

If everything works fine, feel free to configure your favorite IDE in the
following steps.

### Creating an IntelliJ project

To work with IntelliJ:

*  Install Bazel's [IntelliJ plug-in](https://ij.bazel.build).
*  Set the path to the Bazel binary in the plugin preferences
   (`Preferences` > `Other Settings` > `Bazel Settings`).
*  Import the Bazel workspace as a Bazel project
   (`File` > `Import Bazel Project...`) with the following settings:
   *  Use existing bazel workspace: choose your cloned Git repository.
   *  Select `Import from workspace` and choose the `scripts/ij.bazelbuild`
   file as the `Project view`.
*  Download [Google's Java Code Style Scheme file for IntelliJ](https://github.com/google/styleguide/blob/gh-pages/intellij-java-google-style.xml),
   import it (go to `Preferences` > `Editor` > `Code Style` > `Java`, click `Manage`, then `Import`)
   and use it when working on Bazel's code.

### Creating an Eclipse project

To work with Eclipse:

*  [Install Bazel](https://bazel.build/versions/master/docs/install.html) on your system.
*  Clone Bazel's Git repository from Gerrit:
   *  `git clone https://bazel.googlesource.com/bazel`
*  Install the [e4b](https://github.com/bazelbuild/e4b) plugin.
*  Change the path to the Bazel binary in the plugin preferences.
*  Import the Bazel workspace as a Bazel project (`File` > `New` > `Other` >
   `Import Bazel Workspace`).
*  Select `src > main > java` and `src > test > java` as directories and add
   `//src/main/java/...` and `//src/test/java/...` as targets.
*  Download [Google's Java Code Style Scheme file for Eclipse](https://github.com/google/styleguide/blob/gh-pages/eclipse-java-google-style.xml) and use it when working on Bazel's code.

<a name="compile-bazel"></a>
### Compiling Bazel

To test out bazel, you need to compile it. To compile a development version of
Bazel, you need a the latest released version of bazel, which can be
[compiled from source](/versions/master/docs/install.html#compiling-from-source).

`bazel build //src:bazel` builds the Bazel binary using `bazel` from your PATH
and the resulting binary can be found at `bazel-bin/src/bazel`. This is the
recommended way of rebuilding Bazel once you have bootstrapped it.

In addition to the Bazel binary, you might want to build the various tools Bazel
uses. They are located in `//src/java_tools/...`, `//src/objc_tools/...` and
`//src/tools/...` and their directories contain README files describing their
respective utility.

When modifying Bazel, you want to make sure that the following still works:

*  Build a distribution archive with `bazel build //:bazel-distfile`. After
   unzipping it in a new empty directory, run `bash compile.sh all` there.
   It rebuilds Bazel with `./compile.sh`, Bazel with the
   `compile.sh` Bazel and Bazel with the Bazel-built binary. It compares if the
   constructed Bazel builts are identical and then runs all bazel tests with
   `bazel test //src/... //third_party/ijar/...`. This is what we use at Google
   to ensure that we don't break Bazel when pushing new commits, too.

### Debugging Bazel

Start creating a debug configuration for both C++ and Java in your `.bazelrc`
with the following:

```
build:debug -c dbg
build:debug --javacopt="-g"
build:debug --copt="-g"
build:debug --strip="never"
```

Then you can rebuild Bazel with `bazel build --config debug //src:bazel` and use
your favorite debugger to start debugging.

For debugging the C++ client you can just run it from gdb or lldb as you normally would.
But if you want to debug the Java code, you must attach to the server using the following:

*  Run Bazel with debugging option `--host_jvm_debug` before the
   command (e.g., `bazel --batch --host_jvm_debug build //src:bazel`).
*  Attach a debugger to the port 5005. With `jdb` for instance,
   run `jdb -attach localhost:5005`. From within Eclipse, use the
   [remote Java application launch
   configuration](http://help.eclipse.org/luna/index.jsp?topic=%2Forg.eclipse.jdt.doc.user%2Ftasks%2Ftask-remotejava_launch_config.htm).
*  Our IntelliJ plugin has built-in
  [debugging support](https://ij.bazel.build/docs/run-configurations.html)

## Bazel's code description

Bazel is organized in several parts:

*  Client code in `src/main/cpp` provides the command-line interface.
*  Protocol buffers in `src/main/protobuf`.
*  Server code in `src/main/java` and `src/test/java`.
   *  Core code which is mostly composed of [SkyFrame](designs/skyframe.html) and some
     utilities.
   *  Rules written in Bazel's extension language
     [Skylark](docs/skylark/index.html) are defined in `tools/build_rules`. If
     you want to add rules, consider using [Skylark](docs/skylark/index.html)
     first.
   *  Builtin rules in `com.google.devtools.build.lib.rules` and in
     `com.google.devtools.build.lib.bazel.rules`. You might want to read about
     the [Challenges of Writing Rules](docs/rule-challenges.html) first.
*  Java native interfaces in `src/main/native`.
*  Various tooling for language support (see the list in the
   [compiling Bazel](#compile-bazel) section).
