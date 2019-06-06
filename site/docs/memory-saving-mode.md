---
layout: documentation
title: Memory-saving Mode
---

# Running Bazel with limited RAM

In certain situations, you may want Bazel to use minimal memory. You can set the
maximum heap via the startup flag
[`--host_jvm_args`](user-manual.html#flag--host_jvm_args),
like `--host_jvm_args=-Xmx2g`.

However, if your builds are big enough, Bazel may throw an `OutOfMemoryError`
(OOM) when it doesn't have enough memory. You can make Bazel use less memory, at
the cost of slower incremental builds, by passing the following command flags:
[`--discard_analysis_cache`](user-manual.html#flag--discard_analysis_cache),
`--nokeep_state_after_build`, and `--notrack_incremental_state`.

These flags will minimize the memory that Bazel uses in a build, at the cost of
making future builds slower than a standard incremental build would be.

You can also pass any one of these flags individually:

 * `--discard_analysis_cache` will reduce the memory used during execution (not
analysis). Incremental builds will not have to redo package loading, but will
have to redo analysis and execution (although the on-disk action cache can
prevent most re-execution).
 * `--notrack_incremental_state` will not store any edges in Bazel's internal
 dependency graph, so that it is unusable for incremental builds. The next build
 will discard that data, but it is preserved until then, for internal debugging,
 unless `--nokeep_state_after_build` is specified.
 * `--nokeep_state_after_build` will discard all data after the build, so that
 incremental builds have to build from scratch (except for the on-disk action
 cache). Alone, it does not affect the high-water mark of the current build.
