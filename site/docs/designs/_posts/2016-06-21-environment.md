---
layout: documentation
title: Specifying environment variables
---

# Specifying environment variables for actions

This doc was written by [aehlig@google.com](mailto:aehlig@google.com).
Status: unimplemented.

## Current shortcomings

Currently, Bazel provides a cleaned set of environment variables to the
actions in order to obtain hermetic builds. This, however is not sufficent
for all use cases.

* Projects often want to use tools which are not part of the repository; however,
  their location varies from installation to installation. So, some sensible
  value for the `PATH` environment variable has to be set.

* Some set-ups depend on every program having access to specific variables,
  e.g., indicating the homebrew paths, or library paths.

* Commercial compilers sometimes need to be passed the location of a license
  server through the environment.

## Proposed solution

### New flag `--action_env`

We propose to add a new bazel flag, `--action_env` which has two
valid forms of usage,

* specifying a variable with unspecified value, `--action_env=VARIABLE`,
  and

* specifying a variable with a value, `--action_env=VARIABLE=VALUE`;
  in the latter case, the value can well be the empty string, but it is still
  considered a specified value.

This flag has a "latest wins" semantics in the sense that if the option is given
twice for the same variable, only the latest option will be used, regardless
whether specified or unspecified value. Options given for different variables
accumulate.

In every action executed
with [`use_default_shell_env`] (/docs/skylark/lib/ctx.html#action) being true,
precisely the environment variables specified by
`--action_env` options are set as the default environment.
(Note that, therefore, by default, the environment for actions is empty.)

* If the effective option for a variable has an unspecified value,
  the value from the invocation environment of Bazel is taken.

* If the effective option for a variable specifies a value, this value is
  taken, regardless of the environment in which Bazel is invoked.

Environment variables are considered an essential part of an action. In other
words, an action is expected to produce a different output, if the environment
it is invoked in differs; in particular, a previously cached value cannot be
taken if the effective environment changes.

Given that normally a rule writer cannot know which tools might need fancy
environment variables (think of the commercial compiler use case), the default
for the [`use_default_shell_env`] (/docs/skylark/lib/ctx.html#action)
parameter will become true.

### List of rc-files read by Bazel

The list of rc-files that Bazel takes options from will include, at
least, the following files, where files later in the list take precedence over
the ones earlier in the list for conflicting options; for the
`--action_env` option the already described "latest wins" semantics is
applied.

* A global rc-file. This file typically contains defaults for a whole group of
  machines, like all machines of a company. On UNIX-like systems, it will be
  located at `/etc/bazel.bazelrc`.

* A machine-wide rc-file. This file is typically set by the administrator of
  the machine or a group of machines with the same architecture. It typically
  contains settings that are specific to that architecture and hardware.
  On UNIX-like systems it will be next to be binary and called like the binary
  with `.bazelrc` appended to the file name.

* A user-specific file, located in `~/.bazelrc`. This file will be set by
  each user for options desired for all Bazel invocations.

* A project-specific file. This is the file `tools/bazel.rc` next to
  the `WORKSPACE` file. This file is considered project-specific and
  typically versioned in the same repository as the project.

* A file specific to user, project, and checkout. This is the file
  `.bazelrc` next to the `WORKSPACE` file. As it is specific to
  the user and the machine he or she is working on, projects are advised
  to ignore that file in the repository of the project (e.g., by adding
  it to their `.gitignore` file, if they version the project with git).

When looking for those rc-files, symbolic links are followed; files not
existing are silently assumed to be empty. Note that all those are regular
rc-files for Bazel, hence are not limited to the newly introduced
`--action_env` option. Also, the rule that options for more specific
invocations win over common options still applies; but, within each level of
specificness, precedence is given according to the mentioned order of rc-files.

## Example usages of environment specifications

The proposed solution allows for a variety of use cases, including the
following.

* Systems using commercial compilers can set the environment variables with
  information about the license server in the global rc file.

* Users requiring special variables, like the ones used by homebrew, can set
  them in their machine specific rc-file. In fact, once this proposal is
  implemented, the homebrew port for Bazel could itself install that
  machine-wide rc-file.

* Projects depending on the environment, e.g., because they use tools assumed to
  be already installed on the user's systm, have several options.

  * If they are optimistic about the environment, e.g., because they are not
    very version dependent on the tools used, can just specify which environment
    variables they depend on by adding declarations with unspecified values in
    the `tools/bazel.rc` file.

  * If dependencies are more delicate, projects can provide a configure script
    that does whatever analysis of the environment is necessary and then write
    `--action_env` options with specified values to the user-project
    local `.bazelrc`
    file. As the configure script will only run when manually invoked by the
    user and the syntax of the user-project local `.bazelrc` file is so that it
    can be easily
    be edited by a human, it is OK if that script only works in the majority of
    the cases, as a user requiring an unusual setup for that project can easily
    modify the user-project local `.bazelrc` by hand afterwards.

* Irrespectively of the approach chosen by the project, a user where the
  environment changes frequently (e.g., on clusters or other machines using a
  traditional layout) can fix the environment by adding `--action_env`
  options with specific values to the user-project local `.bazelrc`.

  To simplify this use case, Bazel might provide a script
  `bazel_freeze_environment` that reads the
  `tools/bazel.rc` and looks for `--action_env` options with
  unspecified values and writes corresponding ones with specified values to the
  user-project local `.bazelrc` file; the specified values are taken from the
  environment of
  the invocation of that script.

  To simplify "freeze on first use" approaches, there will be separate way of
  invoking the `bazel_freeze_environment` script so that it only adds
  `--action_env` options with specified values for variables not already
  mentioned in the user-project local `.bazelrc` file.

## Transition plan

Currently, some users of Bazel already make use of the fact that `PATH`,
`LD_LIBRARY_PATH`, and `TMPDIR` are being passed to actions. To allow those
projects a smooth
transition to the new set up, the global Bazel rc-file provided by upstream
will have the following content.

```
common --action_env=PATH
common --action_env=LD_LIBRARY_PATH
common --action_env=TMPDIR
```


## Bazel's own dependency on `PATH`

Bazel itself also uses external tools, like `cat`, `echo`, `sh`, but also
tools like `bash` where the location differs between installations. In
particular, a value for `PATH` needs to be provided. This will be covered
by the setting of the global bazel configuration file. Should the need arise, a
configure-like script can be added; at the moment it seems that this will not
be necessary.

## Reasons for the Design Choices, Risks, and Alternatives Considered

### Conflicting Interests on the environment influencing actions

There are conflicting requirements for the environment variables of an action.

* Users expect Bazel to "just work", i.e., the expectation is that if a tool
  works on the command line, it should also work when called from an action in
  a Bazel invocation from the same environment. A lot of compilers, however,
  depend, at least on some systems, on certain environment variables.
  An approach used by quite a few other build systems is to pass through the
  whole invocation environment.

* Bazel wants to provide correct and reproducible builds. Therefore, everything
  that potentially influences the outcome of an action needs to be controlled
  and tracked; a cached result cannot be used if anything potentially changing
  the outcome has changed.

* Users expect Bazel to not do rebuilds they (i.e., the users) know are
  unnecessary. And, while for a lot of users the environment variables that
  actually influence the build stay stable, the full environment constantly
  changes; take the `OLDPWD` environment variable as an example.

This design tries to reconcile these needs by allowing arbitrary environment
variables being set for actions, but only in an opt-in way. Variables need to
be explicitly mentioned, either in a configuration file or on the command line,
to be provided to an action.

### Generic Solutions versus Special Casing

As Bazel already has quite a number of concepts, there is the valid concern
that the complexity might increase too much and newly added concepts might
become a maintenance burden. Another concern is that more configuration
mechanisms make it harder for the user to know which one is the correct one
to use for his or her problem. The general desire is to have few, but powerful
enough mechanisms to control the build behaviour and avoid special casing.

* Putting the environment variables visible in actions in the hand of the
  user avoids the need of special casing more and more "important" environment
  variables.

* Building on the already existing mechanism to specify, inherit, and override
  command-line options reduces the amount newly introduced concepts. The main
  addition is a command-line option.

A corner case from that perspective is the `bazel_freeze_environment` script.
While it solves a valid use case, its only purpose is the management
of environment variables. At least it is strictly a user tool, in the sense
that Bazel itself does not depend on it: Bazel will happily read any
syntactically valid rc-file, regardless how it was created; so a user can
hand-code the user-project local `.bazelrc` file, use the help of
`bazel_freeze_environment`, or use a third-party tool to generate it.

### Source of Knowledge for Needed Environment Variables

Another aspect that went into the design is that different entities know
about environment variables that are essential for the build to work.

* Some variables are "obviously" relevant, like `PATH` or `TMPDIR`.
  However, there is no "obvious" value for them.

  * Both depend on the layout of the system in question. A special fast
    file system for temporary files might be provided at a designated
    location. Binaries might be installed under `/bin`, `/usr/bin`,
    `/usr/local/bin`, or even versioned paths to allow parallel installations
    of different versions of the same tool. For example, on Debian Gnu/Linux
    the `bash` is installed in `/bin`, whereas on FreeBSD it is usually
    installed in `/usr/local/bin` (but the prefix `/usr/local` is at the
    discretion of the system administrator).

  * The user might have custom-built versions of tools somewhere in the
    home directory, thus making the user the only one who knows an appropriate
    value for the `PATH` variable. Moreover, a user who works on several
    projects requiring different versions of the same tool may even require
    different values of the `PATH` variable for each project.

* The authors and users of a tool know about special variables the tools
  need to work. While the tool itself might serve a standard purpose, like
  compiling C code, the variables the tool depends on might be specific to
  that tool (like passing information about a license server).

* The maintainers of a porting or packaging system know about environment
  variables a tool might additionally need (e.g., in the homebrew case).
  These might not be needed if the same tool is packaged differently.

* The project authors know about environment variables special to their
  project that some of their actions need.

These different sources of information make it hard to designate a
single maintainer for the action environment. This makes approaches
undesireable that are based on a single source specifying the action
environment, like the `WORKSPACE` file, or the rule definitions. While
those approaches make it easy to predict the environment an action will
have, they all require the user to merge in the specifics of the system
and his or her personal settings for each checkout (including rebasing
these changes for each upstream change of that file). Collecting environment
variables via the rc-file mechanism allows setting each variable within
the appropriate scope (global, machine-dependent, user-spefic, project-specific,
specific to the user-project pair) in a conflict-free way by the entity
in charge of that scope.
