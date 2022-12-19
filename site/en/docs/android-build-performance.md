Project: /_project.yaml
Book: /_book.yaml

# Android Build Performance

{% include "_buttons.html" %}

This page contains information on optimizing build performance for Android
apps specifically. For general build performance optimization with Bazel, see
[Optimizing Performance](/rules/performance).

## Recommended flags {:#recommended-flags}

The flags are in the
[`bazelrc` configuration syntax](/run/bazelrc#bazelrc-syntax-semantics), so
they can be pasted directly into a `bazelrc` file and invoked with
`--config=<configuration_name>` on the command line.

**Profiling performance**

Bazel writes a JSON trace profile by default to a file called
`command.profile.gz` in Bazel's output base.
See the [JSON Profile documentation](/rules/performance#performance-profiling) for
how to read and interact with the profile.

**Persistent workers for Android build actions**.

A subset of Android build actions has support for
[persistent workers](https://blog.bazel.build/2015/12/10/java-workers.html).

These actions' mnemonics are:

*   DexBuilder
*   Javac
*   Desugar
*   AaptPackage
*   AndroidResourceParser
*   AndroidResourceValidator
*   AndroidResourceCompiler
*   RClassGenerator
*   AndroidResourceLink
*   AndroidAapt2
*   AndroidAssetMerger
*   AndroidResourceMerger
*   AndroidCompiledResourceMerger

Enabling workers can result in better build performance by saving on JVM
startup costs from invoking each of these tools, but at the cost of increased
memory usage on the system by persisting them.

To enable workers for these actions, apply these flags with
`--config=android_workers` on the command line:

```
build:android_workers --strategy=DexBuilder=worker
build:android_workers --strategy=Javac=worker
build:android_workers --strategy=Desugar=worker

# A wrapper flag for these resource processing actions:
# - AndroidResourceParser
# - AndroidResourceValidator
# - AndroidResourceCompiler
# - RClassGenerator
# - AndroidResourceLink
# - AndroidAapt2
# - AndroidAssetMerger
# - AndroidResourceMerger
# - AndroidCompiledResourceMerger
build:android_workers --persistent_android_resource_processor
```

The default number of persistent workers created per action is `4`. We have
[measured improved build performance](https://github.com/bazelbuild/bazel/issues/8586#issuecomment-500070549){: .external}
by capping the number of instances for each action to `1` or `2`, although this
may vary depending on the system Bazel is running on, and the project being
built.

To cap the number of instances for an action, apply these flags:

```
build:android_workers --worker_max_instances=DexBuilder=2
build:android_workers --worker_max_instances=Javac=2
build:android_workers --worker_max_instances=Desugar=2
build:android_workers --worker_max_instances=AaptPackage=2
# .. and so on for each action you're interested in.
```

**Using AAPT2**

[`aapt2`](https://developer.android.com/studio/command-line/aapt2){: .external} has improved
performance over `aapt` and also creates smaller APKs. To use `aapt2`, use the
`--android_aapt=aapt2` flag or set `aapt2` on the `aapt_version` on
`android_binary` and `android_local_test`.

**SSD optimizations**

The `--experimental_multi_threaded_digest` flag is useful for optimizing digest
computation on SSDs.
