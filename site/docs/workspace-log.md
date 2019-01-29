---
layout: documentation
title: Finding non-hermetic behavior in WORKSPACE rules
---

# Finding non-hermetic behavior in WORKSPACE rules

In the following, we say that a host machine is the machine where Bazel runs.

When using remote execution, the actual build and/or test steps are not
happening on the host machine, but are instead sent off to the remote execution
system. However, the steps involved in resolving workspace rules are happening
on the host machine. If your workspace rules access information about the
host machine for use during execution, your build is likely to break due to
incompatibilities between the environments.

As part of [adapting Bazel rules for remote
execution](/remote-execution-rules.html), you need to find such workspace rules
and fix them. This page describes how to find potentially problematic workspace
rules using the workspace log.


## Finding non-hermetic rules

[Workspace rules](/be/workspace.html) allow the developer to add dependencies to
external workspaces, but they are rich enough to allow arbitrary processing to
happen in the process. All related commands are happening locally and can be a
potential source of non-hermeticity. Usually non-hermetic behavior is
introduced through
[`repository_ctx`](/skylark/lib/repository_ctx.html) which allows interacting
with the host machine.

Starting with Bazel 0.18, you can get a log of some potentially non-hermetic
actions by adding the flag `--experimental_workspace_rules_log_file=[PATH]` to
your Bazel command. Here `[PATH]` is a filename under which the log will be
created.

Things to note:

* the log captures the events as they are executed. If some steps are
  cached, they will not show up in the log, so to get a full result, don't
  forget to run `bazel clean --expunge` beforehand.

* Sometimes functions might be re-executed, in which case the related
  events will show up in the log multiple times.

* Workspace rules log currently only logs Skylark events. Some native rules
  may cause non-hermetic behavior but not show up in this log. Examples of those
  rules include
  [maven_jar](/be/workspace.html#maven_jar)
  and the deprecated
  [git_repository](/be/workspace.html#git_repository)
  and
  [http_file](/be/workspace.html#http_file).

  Note that these particular rules do not cause hermiticity concerns as long
  as a hash is specified.

To find what was executed during workspace initialization:

1.  Run `bazel clean --expunge`. This command will clean your local cache and
    any cached repositories, ensuring that all initialization will be re-run.

2.  Add `--experimental_workspace_rules_log_file=/tmp/workspacelog` to your
    Bazel command and run the build.

    This produces a binary proto file listing messages of type
    [WorkspaceEvent](https://source.bazel.build/bazel/+/master:src/main/java/com/google/devtools/build/lib/bazel/debug/workspace_log.proto?q=WorkspaceEvent)

3.  Download the Bazel source code and navigate to the Bazel folder by using
    the command below. You need the source code to be able to parse the
    workspace log with the
    [workspacelog parser](https://source.bazel.build/bazel/+/master:src/tools/workspacelog/).

        git clone https://github.com/bazelbuild/bazel.git
        cd bazel

4.  In the Bazel source code repo, convert the whole workspace log to text.

        bazel build src/tools/workspacelog:parser
        bazel-bin/src/tools/workspacelog/parser --log_path=/tmp/workspacelog > /tmp/workspacelog.txt

5.  The output may be quite verbose and include output from built in Bazel
    rules.

    To exclude specific rules from the output, use `--exclude_rule` option.
    For example:

        bazel build src/tools/workspacelog:parser
        bazel-bin/src/tools/workspacelog/parser --log_path=/tmp/workspacelog \
            --exclude_rule "//external:local_config_cc" \
            --exclude_rule "//external:dep" > /tmp/workspacelog.txt

5.  Open `/tmp/workspacelog.txt` and check for unsafe operations.

The log consists of
[WorkspaceEvent](https://source.bazel.build/bazel/+/master:src/main/java/com/google/devtools/build/lib/bazel/debug/workspace_log.proto?q=WorkspaceEvent)
messages outlining certain potentially non-hermetic actions performed on a
[`repository_ctx`](/skylark/lib/repository_ctx.html).

The actions that have been highlighted as potentially non-hermetic are as follows:

* `execute`: executes an arbitrary command on the host environment. Check if
  these may introduce any dependencies on the host environment.

* `download`, `download_and_extract`: to ensure hermetic builds, make sure
  that sha256 is specified

* `file`, `template`: this is not non-hermetic in itself, but may be a mechanism
  for introducing dependencies on the host environment into the repository.
  Ensure that you understand where the input comes from, and that it does not
  depend on the host environment.

* `os`: this is not non-hermetic in itself, but an easy way to get dependencies
  on the host environment. A hermetic build would generally not call this.
  In evaluating whether your usage is hermetic, keep in mind that this is
  running on the host and not on the workers. Getting environment specifics
  from the host is generally not a good idea for remote builds.

* `symlink`: this is normally safe, but look for red flags. Any symlinks to
  outside the repository or to an absolute path would cause problems on the
  remote worker. If the symlink is created based on host machine properties
  it would probably be problematic as well.

* `which`: checking for programs installed on the host is usually problematic
  since the workers may have different configurations.
