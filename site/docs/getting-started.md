---
layout: documentation
title: Getting Started
---

# Getting Started with Bazel

## Setup

Use the [installation instructions](/docs/install.html) to install a copy of
Bazel on your machine.

## Using a Workspace

All Bazel builds take place in a [_workspace_](/docs/build-ref.html#workspaces),
a directory on your filesystem that contains source code for the software you
want to build, as well symbolic links to directories that contain the build
outputs (for example, `bazel-bin` and `bazel-out`). The location of the
workspace directory is not significant, but it must contain a file called
`WORKSPACE` in the top-level directory; an empty file is a valid workspace.
The `WORKSPACE` file can be used to reference
[external dependencies](/docs/external.html) required to build the outputs.
One workspace can be shared among multiple projects if desired.

```bash
$ touch WORKSPACE
```

## Creating a Build File

To know which target can be build in your project, Bazel inspects `BUILD` files.
They are written in a Bazel's build language which is syntactically similar to
Python. Usually they are just a sequence of declarations of rules. Each rule
specifies its inputs, outputs, and a way to compute the outputs from the inputs.
The rule probably most familiar to people how have used `Makefile`s before (as
it is the only rule available there) is
the [genrule](/docs/be/general.html#genrule), which specifies how the output
can be gerated by invoking a shell command.

```
genrule(
  name = "hello",
  outs = ["hello_world.txt"],
  cmd = "echo Hello World > $@",
)
```

The shell command may contain the familiar
[Make variables](/docs/be/make-variables.html). With the quoted `BUILD` file,
you then ask Bazel to generate the target.

```
$ bazel build :hello
.
INFO: Found 1 target...
Target //:hello up-to-date:
  bazel-genfiles/hello_world.txt
INFO: Elapsed time: 2.255s, Critical Path: 0.07s
```

We note two things. First, targets are normally referred to by their
[label](/docs/build-ref.html#labels), which is specified by the
[name](/docs/be/general.html#genrule.name) attribute of the rule.
(Referencing them by the output file name is also possible, but not
the preferred way.)
Secondly, Bazel puts the generated
files to a separate directory (the `bazel-genfiles` directory actually
is a symbolic link) to not pollute your source tree.

Rules may use the output of other rules as input, as in the following
example. Again, the generated sources are referred to by their label.

```
genrule(
  name = "hello",
  outs = ["hello_world.txt"],
  cmd = "echo Hello World > $@",
)

genrule(
  name = "double",
  srcs = [":hello"],
  outs = ["double_hello.txt"],
  cmd = "cat $< $< > $@",
)
```

Finally note that, while the [genrule](/docs/be/general.html#genrule) might
seem familiar, it usually is _not_ the best rule to use. It is preferrable
to use one of the specialized [rules](/docs/be/overview.html#rules) for
various languages.

# Next Steps

Next, check out the tutorial on building [java](/docs/tutorial/java.html)
or [C++](/docs/tutorial/cpp.html) programs.
