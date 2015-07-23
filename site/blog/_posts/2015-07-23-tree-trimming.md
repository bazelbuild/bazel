---
layout: posts
title: Trimming your (build) tree
---

_Reposted from [@kchodorow's blog](http://www.kchodorow.com/blog/2015/07/23/trimming-the-build-tree-with-bazel/)._

[Jonathan Lange](https://twitter.com/mumak) wrote a [great blog
post](https://jml.io/2015/07/bazel-correct-reproducible-fast-builds.html) about
how Bazel caches tests. Basically: if you run a test, change your code, then run
a test again, the test will only be rerun if you changed something that could
actually change the outcome of the test.  Bazel takes this concept pretty far to
minimize the work your build needs to do, in some ways that aren't immediately
obvious.

Let's take an example. Say you're using Bazel to "build" rigatoni arrabiata,
which could be represented as having the following dependencies:

<img src="/assets/recipe.png"/>

Each food is a library which depends on the libraries below it.  Suppose you
change a dependency, like the garlic:

<img src="/assets/change-garlic.png"/>

Bazel will stat the files of the "garlic" library and notice this change, and
then make a note that the things that depend on "garlic" may have also changed:

<img src="/assets/dirty.png"/>

The fancy term for this is "invalidating the upward transitive closure" of the
build graph, aka "everything that depends on a thing might be dirty."  Note that
Bazel already knows that this change doesn't affect several of the libraries
(rigatoni, tomato-puree, and red-pepper), so they definitely don't have to be
rebuilt.

Bazel will then evaluate the "sauce" node and figures out if its output has
changed.  This is where the secret sauce (ha!) happens: if the output of the
"sauce" node hasn't changed, Bazel knows that it doesn't have to recompile
rigatoni-arrabiata (the top node), because none of its direct dependencies
changed!

<img src="/assets/dirty-unmark.png"/>

The sauce node is no longer “maybe dirty” and so its reverse dependencies
(rigatoni-arrabiata) can also be marked as clean.

In general, of course, changing the code for a library will change its compiled
form, so the "maybe dirty" node will end up being marked as "yes, dirty" and
re-evaluated (and so on up the tree).  However, Bazel's build graph lets you
compile the bare minimum for a well-structured library, and in some cases avoid
compilations altogether.
