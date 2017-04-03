---
layout: contribute
title: bazel init a.k.a ./configure for Bazel
---

__This design document has been replaced by
[Skylark Remote Repositories](/designs/2015/07/02/skylark-remote-repositories.html)
and is maintained here just for reference__

# Design Document: bazel init a.k.a ./configure for Bazel
_A configuration mechanism for Bazel_

**Design documents are not descriptions of the current functionality of Bazel.
Always go to the documentation for current information.**


**Status:** deprecated, replaced by [Skylark Remote Repositories](/designs/2015/07/02/skylark-remote-repositories.html)

**Author:** dmarting@google.com

**Design document published**: 06 March 2015

## I. Rationale

Bazel tooling needs special setup to work. For example, C++ crosstool
configuration requires path to GCC or Java configuration requires the
path to the JDK. Autodetecting those paths from Bazel would be broken
because each ruleset requires its own configuration (C++ CROSSTOOL
information is totally different from JDK detection or from go root
detection). Therefore, providing a general mechanism to configure
Bazel tooling seems natural. To have Bazel self-contained, we will
ship this mechanism as an additional command of Bazel. Because this
command deals with non-hermetic parts of Bazel, this command should
also group all non-hermetic steps (i.e. it should fetch the
dependencies from the remote repositories) so a user can run it and
get on a plane with everything needed.

## II. Considered use-cases

We consider the 3 following use-cases:

  - __UC1.__ The user wants to not worry about tools configuration and
    use the default one for golden languages (Java, C++, Shell) and
    wants to also activate an optional language (Go). No configuration
    information (aka `tools` package) should be checked into the
    version control system.
  - __UC2.__ The user wants to tweak Java configuration but not C++. Of
    course, the user wants his tweak to be shared with his team (i.e.
    `tools/jdk` should be checked into the version control system).
    However, the user does not want to have C++ information (i.e.
    `tools/cpp`) in the VCS.
  - __UC3.__ The user wants his build to be hermetic and he wants to
    set up everything in his `tools` directory (Google use-case).

### Notes

This document addresses the special case of the configuration of the
`tools` package, mechanisms presented here could be extended to any
dependency that needs to be configured (e.g., detecting the installed
libncurse) but that is out of the scope of this document.

Anywhere in this document we refer to the `tools` package as the
package that will receive the current `tools` package content, it does
not commit to keep that package name.

## III. Requirements

### `bazel init` should:
  - _a1._ Not be available in hermetic version (i.e. Google version of
    Bazel, a.k.a Blaze).
  - _a2._ Allow per-language configuration. I.e., Java and C++ tooling
    configuration should be separated.
  - _a3._ Allow Skylark add-ons to specify their configuration, this
    should be pluggable so we can actually activate configuration per
    rule set.
  - _a4._ Support at least 3 modes corresponding to each envisioned
    use-cases:
    * __UC1.__: __installed__ (default mode): a "hidden" `tools`
      package contains the detected tool paths (`gcc`, the JDKs, ...)
      as well as their configuration (basically the content of the
      current `//tools` package). This package should be constructed
      as much as possibly automatically with a way for the user to
      overwrite detected settings.
    * __UC2.__: __semi-hermetic__: the "hidden" `tools` package is used
      only for linking the actual tool paths but the configuration
      would be checked-in into the workspace (in a similar way that
      what is currently done in Bazel). The "hidden" `tools` package
      could contains several versions of the same tools (e.g., jdk-8,
      jdk-7, ...) and the workspace link to a specific one.
    * __UC3.__: __hermetic__: this is the Google way of thing: the
      user check-in everything that he thinks belong to the workspace
      and the init command should do nothing.
  - _a5._ Support explicit reconfiguration. If the configuration
    mechanism changes or the user wants to tune the configuration, it
    should support to modify the configuration, i.e., update the
    various paths or change the default options.

### `bazel init` could:

  - _b1._: Initialize a new workspace: as it would support configuring
    a whole tool directory, it might be quite close to actually
    initializing a new workspace.

## IV. User interface

To be efficient, when the `tools` directory is missing, `bazel build`
should display an informative error message to actually run `bazel
init`.

Configuration is basically just setting a list of build constants like
the path to the JDK, the list of C++ flags, etc...

When the user type `bazel init`, the configuration process starts with
the default configuration (e.g., configure for “gold features” such as
C++, Java, sh_, ...). It should try to autodetect as much as possible.
If a language configuration needs something it cannot autodetect, then
it can prompt the user for the missing information and the
configuration can fail if something is really wrong.

On default installation, `bazel init` should not prompt the user at
all. When the process finishes, the command should output a summary of
the configuration. The configuration is then stored in a "hidden"
directory which is similar to our current `tools` package. By default,
the labels in the configuration would direct to that package (always
mapped as a top-level package). The "hidden:" directory would live in
`$(output_base)/init/tools` and be mapped using the package path
mechanism. The `--overwrite` option would be needed to rerun the
automatic detection and overwrite everything including the eventual
user-set options.

For the hermetic mode, the user has to recreate the default tools
package inside the workspace. If the user has a package with the same
name in the workspace, then the "hidden" directory should be ignored
(--package_path).

To set a configuration option, the user would type `bazel init
java:jdk=/path/to/jdk` or to use the autodetection on a specific
option `bazel init java:jdk`. The list of settings group could be
obtained by `bazel init list` and the list of option with their value
for a specific language by `bazel init list group`. `bazel init list
all` could give the full configuration of all activated groups.

_Prospective idea:_ Bazel init should explore the BUILD file to find the
Skylark `load` statements, determine if there is an associated init
script and use it.

## V. Developer interface

This section presents the support for developer that wants to add
autoconfiguration for a ruleset. The developer adding a configuration
would provide with a configuration script for it. This script will be
in charge of creating the package in the tools directory during `bazel
init` (i.e., the script for Java support will construct the
//tools/jdk package in the "hidden" package path).

Because of skylark rules and the fact that the configuration script
should run before having access to the C++ and Java tooling, this
seems unreasonable to use a compiled language (Java or C++) for this
script. We could use the Skylark support to make it a subset of python
or we could use a bash script. Python support would be portable since
provided by Bazel itself and consistent with skylark. It also gives
immediate support for manipulating BUILD files. So keeping a
"skylark-like" syntax, the interface would look like:

```python
configuration(
	name,              # name of the tools package to configure
	autodetect_method, # the auto detection method
	generate_method,   # the actual package generation
	load_method,       # A method to load the attributes presented
	                   #            to the user from the package
	attrs = {          # List of attributes this script propose
		"jdk_path": String,
		"__some_other_path": String,  # not user-settable
		"jdk_version": Integer,
	})
```

Given that interface, an initial run of `bazel init` would do:

  1. Find all language configuration scripts
  2. Run `load_method` for each script
  3. Run `autodetect_method` for each script. Replace non loaded
     attribute (attribute still undefined after `load_method`) if and
     only if `--rerun` option is provided
  4. Run `generate_method` for each script
  5. Fetch all non up to date dependencies of remote repository

See Appendix B for examples of such methods.

## VI. Implementation plan

  1. Add the hidden tools directory and have it binded with package
     path when no tools directory exists. The hidden tools directory
     will have a WORKSPACE file and will have an automatic local
     repository with the "init" name so that we can actually bind
     targets from it into our workspace.
  2. Add `bazel init` that support the configuration for native
     packages in Java, that is: Java, C++, genrule and test. This
     would create the necessary mechanisms for supporting the
     developer and the basic user interface. This commands will be
     totally in Java for now and should trigger the fetch part of the
     remote repository.
  3. Design and implement the language extension ala Skylark using the
     design for the Java version of point 2.
  4. Convert the existing configuration into that language.
  5. Integrate the configuration with Skylark (i.e. Skylark rules
     writer can add configuration step). We should here decide on
     how it should be included (as a separate script? how do we
     ship a skylark rule set? can we have load statement loading
     a full set of rules?).
  6. Create configuration for the existing skylark build rules. If
     we support load statement with label, we can then create a
     repository for Skylark rules.

## Appendix A. Various comments

  1. We should get rid of the requirement for a `tools/defaults/BUILD` file.
  2. To works correctly, we need some local caching of the bazel
     repository so tools are available. We could have bazelrc specify a
     global path to the local cache (with `/etc/bazel.bazelrc` being loaded
     first to `~/.bazelrc`). We could use a `~/.bazel` directory to put an
     updatable tools cache also. This is needed because user probably want
     to initialize a workspace tooling on a plane
  3. This proposal would probably add a new top-level package. We
     should really take care of the naming convention for default top
     packages (i.e., `tools`, `tools/defaults`, `visibility`, `external`,
     `condition`...). We are going to make some user unhappy if they cannot
     have an `external` directory at the top of their workspace (I would
     just not use a build system that goes against my workspace structure).
     While it is still time to do it, we should rename them with a nice
     naming convention.  A good way to do it is to make top-package name
     constants, possibly settable in the WORKSPACE file (so we can actually
     keep the name we like but user that are bothered by that can change
     it).
  4. As we will remove the tools directory from the workspace, it
     makes sense to add another prelude_bazel file somewhere else. As the
     `WORKSPACE` file controls the workspace, it makes sense to have the
     `prelude_bazel` logic in it (and the load statement should support
     labels so that a user can actually specify remote repository labels).

## Appendix B. Skylark-like code examples of configuration functions

This is just a quick draft, please feel free to propose improvements:

```python
# env is the environment, attrs are the values set either from the command-line
# or from loading the package
def autodetect_method(env, attrs):
  if not attrs.java_version:  # If not given in the command line nor loaded
    attrs.java_version = 8
  if not attrs.jdk_path:
    if env.has("JDK_HOME"):
      attrs.jdk_path = env.get("JDK_HOME")
    elif env.os = "darwin":
      attrs.jdk_path = system("/usr/libexec/java_home -v 1." + attrs.java_version + "+")
    else:
      attrs.jdk_path = basename(basename(readlink(env.path.find(java))))
    if not attrs.jdk_path:
      fail("Could not find JDK home, please set it with `bazel init java:jdk_path=/path/to/jdk`")
   attrs.__some_other_path = first(glob(["/usr/bin/java", "/usr/local/bin/java"]))


# attrs is the list of attributes. It basically contains the list of rules
# we should generate in the corresponding package. Please note
# That all labels are replaced by relative ones as it should not be able
# to write out of the package.
def generate_method(attrs):
  scratch_file("BUILD.jdk", """
Content of the jdk BUILD file.
""")
  # Create binding using local_repository. This should not lie in
  # the WORKSPACE file but in a separate WORKSPACE file in the hidden
  # directory.
  local_repository(name = "jdk", path = attrs.jdk_path, build_file = "BUILD.jdk")
  bind("@jdk//jdk", "jdk")  # also add a filegroup("jdk", "//external:jdk")
  java_toolchain(name = "toolchain", source = attrs.java_version, target = attrs.java_version)
  # The magic __BAZEL_*__ variable could be set so we don’t
  # redownload the repository if possible. This install_target
  # should leverage the work already done on remote repositories.
  # This should build and copy the result into the tools directory with
  # The corresponding exports_files now.
  install_target(__BAZEL_REPOSITORY__, __BAZEL_VERSION__, "//src/java_tools/buildjar:JavaBuilder_deploy.jar")
  install_target(__BAZEL_REPOSITORY__, __BAZEL_VERSION__, "//src/java_tools/buildjar:JavaBuilder_deploy.jar")
  copy("https://ijar_url", "ijar")



# Load the package attributes.
# - attrs should be written and value will be replaced by the user-provided
# one if any
# - query is a query object restricted to the target package and resolving label
# relatively to the target package. This object should also be able to search
# for repository binding
# Note that the query will resolve in the actual tools directory, not the hidden
# one if it exists whereas the generation only happens in the hidden one.
def load_method(attrs, query):
  java_toolchain = query.getOne(kind("java_toolchain", "..."))
  if java_toolchain:
    attrs.jdk_version = max(java_toolchain.source, java_toolchain.target)
  jdk = query.getOne(attr("name", "jdk", kind("local_repository", "...")))
  if jdk:
    attrs.jdk_path = jdk.path
```
