---
layout: posts
title: Announcing simplified workspace creation
---

To create a new workspace, you can now simply create an empty `WORKSPACE` file
in a directory.

Previously, you'd need to copy or symlink the `tools` directory into your
project, which was unpopular:

!["move my-project/ to be a subdirectory of base_workspace/" Ok. Ctrl-W.]({{ site_root }}/assets/ctrl-w-tweet.png)

[Miguel Alcon](https://github.com/mikelalcon) came up with a great idea for
making this process simpler. Now the `compile.sh` script will create a
`.bazelrc` file in your home directory which tells Bazel where `compile.sh` was
run from and, thus, where it can find its tools when you build.

To use this new functionality, get the latest version of the code from Github,
run `./compile.sh`, and then create a Bazel workspace by running
`touch WORKSPACE` in any directory.

Some caveats to watch out for:

* If you move the directory where Bazel was built you will need to
update your `~/.bazelrc` file.
* If you would like to use different tools than the ones `compile.sh`
finds/generates, you can create a `tools/` directory in your project and
Bazel will attempt to use that instead of the system-wide one.

See the [getting started]({{ site_root }}/docs/getting-started.html) docs for more info about
setting up your workspace.

Let us know if you have any questions or issues on the
[mailing list](https://groups.google.com/forum/#!forum/bazel-discuss) or
[GitHub](https://github.com/google/bazel).
