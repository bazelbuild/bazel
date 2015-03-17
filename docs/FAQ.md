
What is Bazel?
-------------

Bazel is a build system, i.e. a system that will run compilers and
tests to assemble your software, similar to Make, Ant, Gradle, Buck, Pants,
and Maven.


What is special about Bazel?
-----------------------

Bazel was designed to fit the way software is developed at Google. It
has the following features:

* Multi-language support: Bazel supports Java, Objective-C and C++ out
  of the box, and can be extended to support arbitrary programming
  languages.

* High-level build language: Projects are described in the BUILD
  language, a concise text format that describes a project as sets of
  small interconnected libraries, binaries and tests. By contrast, in
  systems like Make you have to describe individual files and compiler
  invocations.

* Multi-platform support: The same tool and the same BUILD files can
  be used to build software for different architectures, and even
  different platforms.  At Google, we use Bazel to build both server
  applications running on systems in our data centers and client apps
  running on mobile phones.

* Reproducibility: In BUILD files, each library, test, and binary must
  specify its direct dependencies completely.  Bazel uses this
  dependency information to know what must be rebuilt when you make
  changes to a source file, and which tasks can run in parallel.  This
  means that all builds are incremental and will always produce the
  same result.

* Scalable: Bazel can handle large builds; at Google, it is common for
  a server binary to have 100k source files, and builds where no files
  were changed take about ~200ms.


Why doesn't Google use ...?
-----------------------

* Make, Ninja: These systems give very exact control over what commands
  get invoked to build files, but it's up to the user to write rules
  that are correct.

  Users interact with Bazel on a higher level. For example, it has
  built-in rules for "Java test", "C++ binary", and notions such as
  "target platform" and "host platform". The rules have been battle
  tested to be foolproof.

* Ant and Maven: Ant and Maven are primarily geared toward Java, while
  Bazel handles multiple languages.  Bazel encourages subdividing
  codebases in smaller reusable units, and can rebuild only ones that
  need rebuilding. This speeds up development when working with larger
  codebases.

* Gradle: Bazel configuration files are much more structured than
  Gradle's, letting Bazel understand exactly what each action does.
  This allows for more parallelism and better reproducibility.

* Buck, Pants: Both systems were created ex-Googlers at Twitter and
  Facebook. They have been modeled on Bazel, but their feature set
  has not caught up to Bazel, so it's not an alternative for us.


What is Bazel's origin?
-----------------------

Bazel is a flavor of the tool that Google uses to build its server
software internally. It has expanded to also build the client apps
(iOS, Android) that connect to our servers.


Did you rewrite your internal tool as open-source? Is it a fork?
----------------------------------------------------------------

Bazel shares most of its code with the internal tool, and its rules
are used for millions of builds every day.


Why did Google build Bazel?
---------------------------

A long time ago, Google built its software using large, generated
Makefiles. These led to slow and unreliable builds, which began to
interfere with our developers' productivity and the company's
agility. Hence, we built Bazel.


Does Bazel require a build cluster?
-----------------------------------

Google's in-house flavor of Bazel does use [build
clusters](http://google-engtools.blogspot.com/2011/09/build-in-cloud-distributing-build-steps.html),
so Bazel does have hooks in the code base to plug in a remote build
cache or a remote execution system.

The code base we are opening up runs tasks locally. We are confident
that this is fast enough for most of our users.


How does the Google development process work?
----------------------------------------------

For our server code base, we use the following development workflow:

* All of our server code base is in a single, gigantic version control
  system.

* Everybody builds their software with Bazel.

* Different teams own different parts of the source tree, and make
  their components available as BUILD targets.

* Branching is primarily used for managing releases, so everybody
  develops their software at head.

Bazel is a cornerstone of this philosophy: since Bazel requires all
dependencies to be fully specified, we can predict which programs and
tests are affected by a change, and vet them before submission.

More background on the development process at Google can be found on
the [eng tools blog](http://google-engtools.blogspot.com/).


Why are you opening up Bazel?
-----------------------------

Building software should be fun and easy, and slow and unpredictable
builds take the fun out of programming.


Why would I want to use Bazel?
------------------------------

* Bazel may give you faster build times because it can recompile only
  the files that need to be recompiled. Similarly, it can skip
  re-running tests it knows haven't changed.

* Bazel produces deterministic results. This eliminates skews
  between incremental vs clean builds, laptop vs CI system, etc.

* Bazel can build different client and server apps with the same tool
  from the same workspace. For example, you can change a client/server
  protocol in a single commit, and test that the updated mobile app
  works with the updated server, building both with the same tool,
  reaping all the aforementioned benefits of Bazel.


Can I see examples?
-------------------

Yes, for a simple example, see

  https://github.com/google/bazel/blob/master/base_workspace/examples/cpp/BUILD

The bazel source code itself provides more complex examples, eg.

  https://github.com/google/bazel/blob/master/src/main/java/BUILD
  https://github.com/google/bazel/blob/master/src/test/java/BUILD


What is Bazel best at?
----------------------

Bazel shines at building and testing projects with the following properties:

* Projects with a large codebase
* Projects written in (multiple) compiled languages
* Projects that deploy on multiple platforms
* Projects that have extensive tests


On what platforms does Bazel run?
---------------------------------

Currently, Linux and MacOS. Porting to other Unix platforms should be
straightforward, provided a JDK is available for the platform.


What about Windows?
-------------------

We have experimented with a Windows port
[using MinGW/MSYS](windows.md), but have no plans to invest in this
port right now. Due to its Unix heritage, porting Bazel is significant
work. For example, Bazel uses symlinks extensively, which has varying
levels of support across Windows versions.


What should I not use Bazel for?
--------------------------------

* Bazel tries to be smart about caching. This means it is a bad match
  for build steps that should not be cached. For example, the following
  steps should not be controlled from Bazel:

  * A compilation step that fetches data from the internet.
  * A test step that connects to the QA instance of your site.
  * A deployment step that changes your site's cloud configuration.

* Bazel tries to minimize expensive compilation steps. If you are only
  using interpreted languages directly, such as JavaScript or Python,
  Bazel will likely not interest you.



How stable is Bazel's feature set?
--------------------

The core features (C++, Java, and shell rules) have extensive use
inside Google, so they are thoroughly tested and have very little
churn.  Similarly, we test new versions of Bazel across hundreds of
thousands of targets every day to find regressions, and we release new
versions multiple times every month.

In short, except for features marked as experimental, at any point in
time, Bazel should Just Work. Changes to non-experimental rules will
be backward compatible. A more detailed list of feature support
statuses can be found in our [support document](support.md).


How stable is Bazel as a binary?
--------------------

Inside Google, we make sure that Bazel crashes are very rare. This
should also hold for our open-source codebase.


How can I start using Bazel?
----------------------------

See our [getting started doc](getting-started.md)

TODO(bazel-team): link to an index of all our docs.


Why do I need to have a tools/ directory in my source tree?
----------------------------------------------------

Your project never works in isolation. Typically, it builds with a
certain version of the JDK/C++ compiler, with a certain test driver
framework, on a certain version of your operating system.

To guarantee builds are reproducible even when we upgrade our
workstations, we at Google check most of these tools into version
control, including the toolchains and Bazel itself. By convention, we
do this in a directory called "tools".

Bazel allows tools such as the JDK to live outside your workspace, but
the configuration data for this (where is the JDK, where is the C++
compiler?) still needs to be somewhere, and that place is also the
`tools/` directory.

Bazel comes with a `base_workspace/` directory, containing a minimal set
of configuration files, suitable for running toolchains from standard
system directories, e.g., `/usr/bin/`.


Doesn't Docker solve the reproducibility problems?
--------------------------------------------------

With Docker you can easily create sandboxes with fixed OS releases,
eg. Ubuntu 12.04, Fedora 21. This solves the problem of
reproducibility for the system environment (i.e. "which version of
/usr/bin/c++ do I need?").

It does not address reproducibility with regard to changes in the
source code.  Running Make with an imperfectly written Makefile inside a
Docker container can still yield unpredictable results.

Inside Google, we check tools into source control for reproducibility.
In this way, we can vet changes to tools ("upgrade GCC to 4.6.1") with
the same mechanism as changes to base libraries ("fix bounds check in
OpenSSL").


Will Bazel make my builds reproducible automatically?
-----------------------------------------------------

For Java and C++ binaries, yes, assuming you do not change the
toolchain. If you have build steps that involve custom recipes
(eg. executing binaries through a shell script inside a rule), you
will need to take some extra care:

  * Do not use dependencies that were not declared. Sandboxed
    execution (--spawn_strategy=sandboxed, only on Linux) can
    help find undeclared dependencies.

  * Avoid storing timestamps in generated files. ZIP files and other
    archives are especially prone to this.

  * Avoid connecting to the network. Sandboxed execution can help here
    too.

  * Avoid processes that use random numbers, in particular, dictionary
    traversal is randomized in many programming languages.

TODO(bazel-team): maybe this should be part of the skylark docs
instead, or the genrule docs?


Do you have binary releases?
----------------------------

No, but we should. Stay tuned.


I use Eclipse/IntelliJ. How does Bazel interoperate with IDEs?
--------------------------------------------------------------

We currently have no IDE integration API, although the "bazel query"
command could be used as a building block for for basic IDE support?


I use Jenkins/CircleCI/TravisCI. How does Bazel interoperate with CI systems?
-----------------------------------------------------------------------------

Bazel returns a non-zero exit code if the build or test invocation
fails, and this should be enough for basic CI integration.  Since
Bazel does not need clean builds for correctness, the CI system can
be configured to not clean before starting a build/test run.

TODO(bazel-team): link to exit codes.

What future features can we expect in Bazel?
--------------------------------------------

Our initial goal is to work on Google's internal use-cases. This
includes Google's principal languages (C++, Java, Go) and major
platforms (Linux, Android, iOS).  For practical reasons, not all of
these are currently open-sourced. For more details see our
[roadmap](roadmap.md).


What about Python?
------------------

It is possible to write Python rules as extensions (see below). See
the following files for an example of generating self-contained zip
files for python:

  https://github.com/google/bazel/blob/master/base_workspace/tools/build_rules/py_rules.bzl
  https://github.com/google/bazel/tree/master/base_workspace/examples/py

We are working on opening up a subset of our internal Python rules, so
they can be used as helper scripts as part of a build.

We currently have no plans to provide packaging up of self-contained
Python binaries.



Can I use Bazel for my LISP/Python/Haskell/Scala/Rust project?
-----------------------------------------------

We have an extension mechanism that allows you to add new rules
without recompiling Bazel.

For documentation: see [here](skylark/index.md).

At present, the extension mechanism is experimental though.


I need more functionality; can I add rules that are compiled into Bazel?
---------------------------------------------

If our extension mechanism is insufficient for your use case, email
the mailing list for advice: bazel-discuss@googlegroups.com.



Can I contribute to the Bazel code base?
----------------------------------------

You can propose patches to Bazel. You can find all the details in the
[contribution guidelines](contributing.md).


Why isn't all development done in the open?
-------------------------------------------

We still have to refactor the interfaces between the public code in
Bazel and our internal extensions frequently. This makes it hard to do
much development in the open. See our [governance plan](governance.md)
for more details.


How do I contact the team?
--------------------------

We are reachable at bazel-discuss@googlegroups.com.


Where do I report bugs?
-----------------------

Send e-mail to bazel-discuss@googlegroups.com, or file a bug at
[on github](https://github.com/google/bazel/issues).



What's up with the word "Blaze" in the codebase?
------------------------------------------------

This is an internal name for the system. Please refer to Bazel as
Bazel.


Why do other Google projects (Android, Chrome) use other build tools?
---------------------------------------------------------------------

Until now, Bazel was not available externally, so open source projects
such as Chromium, Android, etc. could not use it. In addition, lack of
Windows support is a problem for building Windows applications, such
as Chrome.


How do you pronounce "Bazel"?
-----------------------------

The same way as "basil" (the herb) in US English: "BAY-zel". It rhymes with
"hazel". IPA: /ˈbeɪzˌəl/
