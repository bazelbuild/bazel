# Standalone disk cache garbage collection utility

This utility may be used to manually run a garbage collection on a disk cache,
if more control over when which garbage collection runs is desired than afforded
by the automatic garbage collection built into Bazel.

Usage (at least one of `--max_age` and `--max_size` must be specified):

```shell
$ bazel run //src/tools/diskcache:gc \
    --disk_cache=/path/to/disk/cache \
    --max_age=7d --max_size=2G
```
