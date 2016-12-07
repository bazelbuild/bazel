---
layout: posts
title: Visualize your build
---

_Reposted from
[Kristina Chodorow's blog](http://www.kchodorow.com/blog/2015/04/24/have-you-ever-looked-at-your-build-i-mean-really-looked-at-your-build/)._

Bazel lets you see a graph of your build dependencies.  It _could_ help you
debug things, but honestly it's just really cool to see what your build is doing.

To try it out, you'll need a project that uses Bazel to build.  If you don't
have one handy,
[here's a tiny workspace](https://github.com/kchodorow/tiny-workspace) you can
use:

```bash
$ git clone https://github.com/kchodorow/tiny-workspace.git
$ cd tiny-workspace
```

Make sure you've
[downloaded and installed Bazel](http://bazel.build/docs/install.html) and have the
following line to your _~/.bazelrc_:

```
query --package_path %workspace%:[path to bazel]/base_workspace
```

Now run `bazel query` in your _tiny-workspace/_ directory, asking it to search
for all dependencies of `//:main` and format the output as a graph:

```bash
$ bazel query 'deps(//:main)' --output graph > graph.in
```

This creates a file called _graph.in_, which is a text representation of the
build graph.  You can use `dot` (install with `sudo apt-get install graphviz`)
to create a png from this:

```bash
$ dot -Tpng < graph.in > graph.png
```

If you open up _graph.png_, you should see something like this:

<img src="/assets/graph.png">

You can see `//:main` depends on one file (`//:main.cc`) and four targets
(`//:x`, `//tools/cpp:stl`, `//tools/default:crosstool`, and
`//tools/cpp:malloc`).  All of the `//tools` targets are implicit dependencies
of any C++ target: every C++ build you do needs the right compiler, flags, and
libraries available, but it crowds your result graph.  You can exclude these
implicit dependencies by removing them from your query results:

```bash
$ bazel query --noimplicit_deps 'deps(//:main)' --output graph > simplified_graph.in
```

Now the resulting graph is just:

<img src="/assets/simple-graph.png">

Much neater!

If you're interested in further refining your query, check out the
[docs on querying](/docs/query.html).
