---
layout: posts
title: Build dashboard dogfood
---

We've added a basic dashboard where you can see and share build and test
results.  It's not ready for an official release yet, but if any adventurous
people would like to try it out (and please report any issues you find!), feel
free to give it a try.

<img src="/assets/dash.png" class="img-responsive" style="width: 800px; border: 1px solid black;"/>

First, you'll need to update your bazel (`git pull` and rerun `./compile.sh`).

Run `bazel build //src/tools/dash && bazel-bin/src/tools/dash/dash` and add
this line to your `~/.bazelrc`:

```
build --use_dash --dash_url=http://localhost:8080
```

Note that the `bazel build` will take a long time to build the first time (the
dashboard uses the AppEngine SDK, which is ~160MB and has to be downloaded).
The "dash" binary starts up a local server that listens on 8080.

With `--use_dash` specified, every build or test will publish info and logs to
http://localhost:8080/ (each build will print a unique URL to visit).

<img src="/assets/dash-shell.png"/>

See [the README](https://github.com/bazelbuild/bazel/blob/master/src/tools/dash/README.md)
for documentation.

This is very much a work in progress. Please let us know if you have any
questions, comments, or feedback.
