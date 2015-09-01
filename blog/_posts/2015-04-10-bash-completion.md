---
layout: posts
title: Support for Bash Shell Completion
---

We just pushed a support for [shell completion in the Bourne-Again
Shell](https://www.gnu.org/software/bash/manual/html_node/Programmable-Completion.html).
It eases the use of Bazel by expanding its commands and the targets to build.

To use this new functionality, build the `//scripts:bash_completion` target
from the Bazel repository:
```
bazel build //scripts:bash_completion
```

This will create a `bazel-bin/scripts/bazel-complete.bash` completion script.
You can copy then copy this script to your completion directory
(`/etc/bash_completion.d` in Ubuntu). If you don't want to install it globally
or don't have such a directory, simply add the following line to your
`~/.bashrc` or `~/.bash_profile` (the latter is the recommended for OS X):
```
source /path/to/bazel/bazel-bin/scripts/bazel-complete.bash
```

After that you should be able to type the tab key after the `bazel`
command in your shell and see the list of possible completions.

If you are interested in supporting other shells, the script is made up
of two parts:

1. [`scripts/bazel-complete-header.bash`](https://github.com/bazelbuild/bazel/blob/master/scripts/bazel-complete-template.bash)
  is the completion logic.
2. `bazel info completion` dumps the list of commands of Bazel, their options
  and for commands and options that expect a value, a description of what is
  expected. This description is either:

* an enum of values enclosed into brackets, e.g., `{a,b,c}`;
* a type description, currently one of:

  * `label`, `label-bin`, `label-test`, `label-package` for
    a Bazel label for, respectively, a target, a runnable target,
    a test, and a package,
  * `path` for a filesystem path,
  * `info-key` for one of the information keys as listed by `bazel info`;

* a combination of possible values using `|` as a separator, e.g,
  `path|{or,an,enum}'`.

Let us know if you have any questions or issues on the
[mailing list](https://groups.google.com/forum/#!forum/bazel-discuss) or
[GitHub](https://github.com/bazelbuild/bazel).
