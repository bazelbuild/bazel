---
layout: contribute
title: Sandboxing
---

# Bazel Sandboxing 2.0

This doc was written by [philwo@google.com](mailto:philwo@google.com).

Status: launched

The section "Handling of environment variables" inspired and was then
superseded by the more detailed
[Specifying environment variables](/docs/designs/2016/06/21/environment.html)
design document.

## Current situation

Tools that use undeclared input files (files that are not explicitly listed in
the dependencies of an action) are a problem, as Bazel cannot keep track of them
and thus they can cause builds to become incorrect: When one of the undeclared
input files changes, Bazel will still believe that the build is up-to-date and
won't rebuild the action - resulting in an incorrect incremental build.

Bazel uses sandboxing to prevent tools (e.g. compilers, linkers, ...) from
accidentally working with input files that are not a declared dependency of an
action - the idea is to run each tool in an environment that contains only the
explicitly declared input files of the action. Thus, there simply are no other
files that a tool could access.

In theory this works well, but as nearly all Bazel users rely at least on some
tools provided by their operating system (e.g. `/usr/bin/zip`, `/usr/bin/gcc`),
which in turn require shared libraries, helper tools or data from other parts
of the installed OS, Bazel currently mounts a number of hard-coded directories
from the operating system into the sandbox in addition to the explicitly
declared inputs.

However, even with that some users continue to run into issues, making Bazel
hard to use - e.g. the compiler they want to use is in a directory that's not
part of the hard-coded list (such as `/usr/local` or `/opt`) or the tool needs
access to device files (e.g. the nVidia CUDA SDK).

## Proposal

We think that it's time to revisit how we do sandboxing in the default settings
of Bazel. Sandboxing was intended to protect the user from forgetting to
declare explicit dependencies between their targets and to protect from tests
or tools accidentally writing all over the hard-disk (e.g. a test that wants to
clean up its temporary work directory via rm -rf and unfortunately wipes the
whole disk), not so much for protecting against an operating system having any
influence on the build. For these users, the current sandboxing with its
hard-coded list of allowed directories is too strict.

On the other hand, some people absolutely do want 100% reproducible and
hermetic builds - and for them the current sandboxing actually isn't strict
enough, as it allows access to various files from the operating system.

We believe we have found a solution that satisfies the demands of all users:

 * Bazel sandboxing will by default recursively mount the root directory `/`
   into each sandbox in read-only mode, excluding the workspace directory (so
   that source files cannot be read from that well-known path) and with a new
   empty, writable execroot that contains the declared inputs of the action.
 * In addition, Bazel will allow to mount a 'base image' or 'base directory' as
   the root directory of the sandbox, thus completely removing any connection
   to the operating system the user is running Bazel under. For example, a
   project might decide that all builds should be done inside a standardized
   Ubuntu 16.04 LTS environment containing certain versions of gcc, etc., that
   is shipped as a base image. Now, even if the developer uses Arch Linux or
   CentOS on their machine, they can build using the same environment as
   everyone else, thus getting the exact same and reproducible outputs.

### Base images

Base images are simply `.tar.gz`'s of a directory structure that contains all
files necessary to execute binaries in, e.g. the output of “debootstrap” or
what you would usually “chroot” in and then run a tool inside. They should be
referred to via labels and could for example be downloaded from somewhere via
a `http_file` rule in the WORKSPACE.

We're investigating if we can reuse
[Docker images (OCI)](https://github.com/opencontainers/image-spec/blob/v0.1.0/serialization.md)
for this, which would make it easier for users to get started with this
feature.

### Handling of environment variables

As part of this project, we also propose to change the handling of environment
variables (e.g. `PATH`) in Bazel, as we believe they are an important part of
the configuration of the environment that the build runs in.

As an example, Bazel currently [resets PATH to a hard-coded string]
(https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/bazel/rules/BazelConfiguration.java),
which may not be suitable for the environment that it actually runs in - e.g.
if a user installs a tool called `babel` in `/usr/local/bin` and they call
`babel` in a shell script or Skylark rule they wrote, [they expect it to just
work] (https://github.com/bazelbuild/bazel/issues/884). We can argue that they
instead should check in their tool to the repository and not rely on `PATH`
lookup to find it, however this is sometimes not possible due to:

 * Users just don't think it's feasible and instead want to take whatever is
   installed on the system,
 * Bazel's restrictions in valid package label identifiers ([you can't check in
   nodejs](https://github.com/bazelbuild/bazel/issues/884#issuecomment-183378680)
   into your repository or even make it part of a filegroup, because it
   contains files that have characters like `$` that are currently illegal from
   Bazel's point of view, though that may change in the future),
 * Licensing restrictions that disallow users checking in certain tools (such
   as XCode).

The proposal how Bazel should decide whether an environment variable should be
included in the environment of a Spawn is:

 * If `use_default_shell_env` is `True`, set `PATH` and `TMPDIR` env vars
   (as we currently do).
 * If a rule declares its need for an environment variable, take it.
   * We already have an [“env” attribute in Skylark actions]
     (http://www.bazel.build/docs/skylark/lib/ctx.html#action) that allows one to
     set variables to hard-coded strings, we have `use_default_shell_env` in
     Skylark actions, which pulls in `PATH` and `TMPDIR`, but we don't have any
     way to just say "This rule needs this environment variable". Laurent
     suggested that we discuss this later, as adding yet another attribute is
     annoying - maybe there's some way we can fold all these use cases into one
     attribute.
   * We might want to add the same attribute to genrule as well then.
 * Don't include any other environment variables.

If Bazel decided that an environment variable is needed by a rule, the next
step is to figure out its value. The proposal how Bazel should decide the value
of an environment variable is:

 * If an environment variable is overridden in the `WORKSPACE.local` file
   ("machine-specific settings"), take it from there.
 * If an environment variable is overridden in the `WORKSPACE` file
   ("project-specific settings"), always take the value from there.
 * If not and we use a base image, take the environment variable from its
   specification (as in OCI).
 * If not, take it from the user's environment.

If an environment variable that is used by a rule changed compared to when it
was built last time, its target has to be rebuild for correctness.

Bazel should instead use `PATH` from the environment and for correctness
trigger a rebuild when it changes.

*Open question: Should the whitelist of environment variables be configurable,
e.g. in the WORKSPACE file?*

### What files does a sandboxed process have access to?

Ideally, we would want to execute SpawnActions in an environment that looks like this:

 * Allows read access to everything in /.
   * Except the workspace (e.g. /home/philwo/src/bazel).
   * Except the "real" execroot (e.g. /tmp/_bazel_philwo/6d3feea2bf88e88127079b36d7ddade1/execroot).
   * Except a user-configurable set of blacklisted files or directories (e.g. /var/secret).

 * Has a separate execroot just for this action: /tmp/_bazel_philwo/6d3feea2bf88e88127079b36d7ddade1/execroot-1
   * which only contains the input files listed for the action.
   * to which the output files will be written.
   * from which the output files will be moved to the real execroot after successful execution.

 * Processes can write wherever they naturally have permission to do so.
   * However, writes have no influence on the host system, instead they are redirected into a separate folder (see: overlayfs's upperdir).

#### Open issues

 * We don't know a way to hide the workspace, while still making selected input files out of it available inside the new execroot. Ideas we tried:
   * copying: Works, but too slow.
   * hard-linking: Does not work when workspace and output_base are on different filesystems.
   * bind mounting them: Works only on Linux, does not scale (the mount syscall becomes really slow once you're at >20000 active mounts).
   * building a custom FUSE filesystem: Might work, but lots of effort.

 * How to hide files from the sandboxed process?
   * There seems to be no good way to "hide" files on Linux or macOS.
   * The best we can do on both systems is to make them unreadable.

 * overlayfs is not usable for our purposes, because it requires root on all systems except Ubuntu.

#### What can we do today?

This is what Bazel is doing for sandboxing at the moment:

 * Allows read access to everything in /.
   * Except a [user-configurable](https://github.com/bazelbuild/bazel/blob/0f119a4db515105217244e4db5d4fed9371ef1a4/src/main/java/com/google/devtools/build/lib/sandbox/SandboxOptions.java#L96) set of blacklisted files or directories (e.g. /var/secret).

 * Has a separate execroot just for this action: /tmp/_bazel_philwo/6d3feea2bf88e88127079b36d7ddade1/execroot-1
   * which contains symlinks to the input files listed for the action (the targets are in the workspace or the "real" execroot).
   * to which the output files will be written.
   * from which the output files will be moved to the real execroot after successful execution.

 * Processes can only write to their private execroot and a private $TMPDIR.

### Related links

 * [Known issues in this area of work](https://github.com/bazelbuild/bazel/issues?q=is%3Aopen+is%3Aissue+label%3A%22category%3A+sandboxing%22)
