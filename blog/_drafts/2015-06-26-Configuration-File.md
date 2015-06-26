---
layout: posts
title: Sharing your rc files
---

Bazel users mostly use the default `tools` package provided by Bazel and
you add the various options you need in your `~/.bazelrc` but that doesn't
scale when you share your workspace with others.

However, you can configure your build by tweaking the BUILD files in the
`tools` package. For instance, you could activate [Error Prone](http://errorprone.info)
checks by adding the `--javacopts="-extra_checks:on"` flag in your `~/.bazelrc`.
However, `~/.bazelrc` is not really convenient as it a user file, not shared with
your team. You could instead add a rc file at `tools/bazel.rc` in your workspace
with the content of the bazelrc file you want to share with your team:

```
build --javacopts="-extra_checks:on"
```

This file, called a master rc file, is parsed before the user rc file. There is
three paths to master rc files that are read in the following order:

  1. `tools/bazel.rc` (depot master rc file),
  2. `/path/to/bazel.bazelrc` (alongside bazel rc file), and
  3. `/etc/bazel.bazelrc` (system-wide bazel rc file).

The complete documentation on rc file is [here](http://bazel.io/docs/bazel-user-manual.html#bazelrc).
