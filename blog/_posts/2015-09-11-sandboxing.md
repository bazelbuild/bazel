---
layout: posts
title: About Sandboxing
---

We've only added sandboxing to Bazel two weeks ago, and we've already seen a
flurry of fixes to almost all of the rules to conform with the additional
restrictions imposed by it.

## What is sandboxing?
Sandboxing is the technique of restricting the access rights of a process. In
the context of Bazel, we're mostly concerned with restricting file system
access. More specifically, Bazel's file system sandbox contains only known
inputs, such that compilers and other tools can't even see files they should
not access.

(We currently also mount a number of system directories into the sandbox to
allow running locally installed tools and make it easier to write shell
scripts. See below.)


## Why are we sandboxing in Bazel?
We believe that developers should never have to worry about correctness, and
that every build should result in the same output, regardless of the current
state of the output tree. If a compiler or tool reads a file without Bazel
knowing it, then Bazel won't rerun the action if that file has changed, leading
to incorrect incremental builds.

We would also like to support remote caching in Bazel, where incorrect reuse of
cache entries is even more of a problem than on the local machine. A bad cache
entry in a shared cache affects every developer on the project, and the
equivalent of 'bazel clean', namely wiping the entire remote cache, rather
defeats the purpose.

In addition, sandboxing is closely related to remote execution. If the build
works well with sandboxing, then it will likely work well with remote
execution - if we know all the inputs, we can just as well upload them to a
remote machine. Uploading all files (including local tools) can significantly
reduce maintenance costs for compile clusters compared to having to install the
tools on every machine in the cluster every time you want to try out a new
compiler or make a change to an existing tool.


## How does it work?
On Linux, we're using user namespaces, which are available in Linux 3.8 and
later. Specifically, we create a new mount namespace. We create a temporary
directory into which we mount all the files that the subprocess is allowed to
see. We then use `pivot_root` to make the temporary directory appear as the
root directory for all subprocesses.

We also mount `/proc`, `/dev/null`, `/dev/zero`, and a temporary filesystem
(tmpfs) on `/tmp`. We mount `/dev/random` and `/dev/urandom`, but recommend
against their usage, as it can lead to non-reproducible builds.

We currently also mount `/bin`, `/etc`, `/usr` (except `/usr/local`), and every
directory starting with `/lib`, to allow running local tools. In the future, we
are planning to provide a shell with a set of Linux utilities, and to require
that all other tools are specified as inputs.


## What about Mac and Windows?
We are planning to implement sandboxing for OS X (using OS X sandboxing, see
our [roadmap](/roadmap.html)) and eventually Windows as well.


## What about networking?
At some point, we'd like to also reduce network access, probably also using
namespaces, with a separate opt-out mechanism.


## How do I opt-out of sandboxing?
Preferably, you should make all your rules and scripts work properly with
sandboxing. If you need to opt out, you should talk to us first - at Google,
the vast majority of actions is fully sandboxed, so we have some experience
with how to make it work. For example, Bazel has a special mechanism to add
information about the current user, date, time, or the current source control
revision to generated binaries.

If you still need to opt out for individual rules, you can add the `local = 1`
attribute to `genrule` or `*_test` calls.

If you're writing a custom rule in Skylark, then you cannot currently opt out.
Instead, please [file a bug](https://github.com/bazelbuild/bazel/issues) and
we'll help you make it work.
