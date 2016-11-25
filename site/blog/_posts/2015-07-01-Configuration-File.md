---
layout: posts
title: Sharing your rc files
---

You can customize the options Bazel runs with in your `~/.bazelrc`, but
that doesn't scale when you share your workspace with others.

For instance, you could de-activate [Error Prone](http://errorprone.info)'s
[DepAnn](http://errorprone.info/bugpattern/DepAnn) checks by adding the
`--javacopts="-Xep:DepAnn:OFF"` flag in your `~/.bazelrc`. However, `~/.bazelrc`
is not really convenient as it a user file, not shared with
your team. You could instead add a rc file at `tools/bazel.rc` in your workspace
with the content of the bazelrc file you want to share with your team:

```
build --javacopts="-Xep:DepAnn:OFF"
```

This file, called a master rc file, is parsed before the user rc file. There is
three paths to master rc files that are read in the following order:

  1. `tools/bazel.rc` (depot master rc file),
  2. `/path/to/bazel.bazelrc` (alongside bazel rc file), and
  3. `/etc/bazel.bazelrc` (system-wide bazel rc file).

The complete documentation on rc file is [here](http://bazel.build/docs/bazel-user-manual.html#bazelrc).
