Project: /_project.yaml
Book: /_book.yaml


# Optimize Memory

{% include "_buttons.html" %}

This page describes how to limit and reduce the memory Bazel uses.

## Running Bazel with Limited RAM {:#running-bazel}

In certain situations, you may want Bazel to use minimal memory. You can set the
maximum heap via the startup flag
[`--host_jvm_args`](/docs/user-manual#host-jvm-args),
like `--host_jvm_args=-Xmx2g`.

### Trade incremental build speeds for memory {:#trade-incremental}

If your builds are too big, Bazel may throw an `OutOfMemoryError` (OOM) when
it doesn't have enough memory. You can make Bazel use less memory, at the cost
of slower incremental builds, by passing the following command flags:
[`--discard_analysis_cache`](/docs/user-manual#discard-analysis-cache),
[`--nokeep_state_after_build`](/reference/command-line-reference#flag--keep_state_after_build),
and
[`--notrack_incremental_state`](/reference/command-line-reference#flag--track_incremental_state).

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

### Trade build flexibility for memory with Skyfocus (Experimental) {:#trade-flexibility}

If you want to make Bazel use less memory *and* retain incremental build speeds,
you can tell Bazel the working set of files that you will be modifying, and
Bazel will only keep state needed to correctly incrementally rebuild changes to
those files. This feature is called **Skyfocus**.

To use Skyfocus, pass the `--experimental_enable_skyfocus` flag:

```sh
bazel build //pkg:target --experimental_enable_skyfocus
```

By default, the working set will be the set of files next to the target being
built. In the example, all files in `//pkg` will be kept in the working set, and
changes to files outside of the working set will be disallowed, until you issue
`bazel clean` or restart the Bazel server.

If you want to specify an exact set of files or directories, use the
`--experimental_working_set` flag, like so:

```sh
bazel build //pkg:target --experimental_enable_skyfocus
--experimental_working_set=path/to/another/dir,path/to/tests/dir
```

You can also pass `--experimental_skyfocus_dump_post_gc_stats` to show the
memory reduction amount:

Putting it altogether, you should see something like this:

```none
$ bazel test //pkg:target //tests/... --experimental_enable_skyfocus --experimental_working_set dir1,dir2,dir3/subdir --experimental_skyfocus_dump_post_gc_stats
INFO: --experimental_enable_skyfocus is enabled. Blaze will reclaim memory not needed to build the working set. Run 'blaze dump --skyframe=working_set' to show the working set, after this command.
WARNING: Changes outside of the working set will cause a build error.
INFO: Analyzed 149 targets (4533 packages loaded, 169438 targets configured).
INFO: Found 25 targets and 124 test targets...
INFO: Updated working set successfully.
INFO: Focusing on 334 roots, 3 leafs... (use --experimental_skyfocus_dump_keys to show them)
INFO: Heap: 1237MB -> 676MB (-45.31%)
INFO: Elapsed time: 192.670s ...
INFO: Build completed successfully, 62303 total actions
```

For this example, using Skyfocus allowed Bazel to drop 561MB (45%) of memory,
and incremental builds to handle changes to files under `dir1`, `dir2`, and
`dir3/subdir` will retain their fast speeds, with the tradeoff that Bazel cannot
rebuild changed files outside of these directories.

## Memory Profiling {:#memory-profiling}

Bazel comes with a built-in memory profiler that can help you check your rule’s
memory use. Read more about this process on the
[Memory Profiling section](/rules/performance#memory-profiling) of our
documentation on how to improve the performance of custom rules.