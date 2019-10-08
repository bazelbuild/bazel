---
layout: documentation
title: Optimizing Performance
---

# Optimizing Performance

When writing rules, the most common performance pitfall is to traverse or copy
data that is accumulated from dependencies. When aggregated over the whole
build, these operations can easily take O(N^2) time or space. To avoid this, it
is crucial to understand how to use depsets effectively.

This can be hard to get right, so Bazel also provides a memory profiler that
assists you in finding spots where you might have made a mistake. Be warned:
The cost of writing an inefficient rule may not be evident until it is in
widespread use.

## Contents
{:.no_toc}

* ToC
{:toc}
<!-- begin-block:internal
[TOC]
  ```
  STARTUP_FLAGS=\
  --host_jvm_args=-javaagent:$(BAZEL)/third_party/allocation_instrumenter/java-allocation-instrumenter-3.0.1.jar \
  --host_jvm_args=-DRULE_MEMORY_TRACKER=1
  ```
  **NOTE**: The bazel repository comes with an allocation instrumenter.
  Make sure to adjust '$(BAZEL)' for your repository location.

These start the server in memory tracking mode. If you forget these for even
one Bazel invocation the server will restart and you will have to start over.

### Using the Memory Tracker

Let's have a look at the target `foo` and see what it's up to. We add
`--nobuild` since it doesn't matter to memory consumption if we actually build
or not, we just have to run the analysis phase.

```
$ bazel $(STARTUP_FLAGS) build --nobuild //foo:foo
```

Let's see how much memory the whole Bazel instance consumes:

```
$ bazel $(STARTUP_FLAGS) info used-heap-size-after-gc
> 2594MB
```

Let's break it down by rule class by using `bazel dump --rules`:

```
$ bazel $(STARTUP_FLAGS) dump --rules
>

RULE                                 COUNT     ACTIONS          BYTES         EACH
genrule                             33,762      33,801    291,538,824        8,635
config_setting                      25,374           0     24,897,336          981
filegroup                           25,369      25,369     97,496,272        3,843
cc_library                           5,372      73,235    182,214,456       33,919
proto_library                        4,140     110,409    186,776,864       45,115
android_library                      2,621      36,921    218,504,848       83,366
java_library                         2,371      12,459     38,841,000       16,381
_gen_source                            719       2,157      9,195,312       12,789
_check_proto_library_deps              719         668      1,835,288        2,552
... (more output)
```

And finally let's have a look at where the memory is going by producing a
`pprof` file using `bazel dump --skylark_memory`:

```
$ bazel $(STARTUP_FLAGS) dump --skylark_memory=$HOME/prof.gz
> Dumping skylark heap to: /usr/local/google/home/$USER/prof.gz
```

Next, we use the `pprof` tool to investigate the heap. A good starting point is
getting a flame graph by using `pprof -flame $HOME/prof.gz`.

  You can get `pprof` from [https://github.com/google/pprof](https://github.com/google/pprof).

In this case we get a text dump of the hottest call sites annotated with lines:

```
$ pprof -text -lines $HOME/prof.gz
>
      flat  flat%   sum%        cum   cum%
  146.11MB 19.64% 19.64%   146.11MB 19.64%  android_library <native>:-1
  113.02MB 15.19% 34.83%   113.02MB 15.19%  genrule <native>:-1
   74.11MB  9.96% 44.80%    74.11MB  9.96%  glob <native>:-1
   55.98MB  7.53% 52.32%    55.98MB  7.53%  filegroup <native>:-1
   53.44MB  7.18% 59.51%    53.44MB  7.18%  sh_test <native>:-1
   26.55MB  3.57% 63.07%    26.55MB  3.57%  _generate_foo_files /foo/tc/tc.bzl:491
   26.01MB  3.50% 66.57%    26.01MB  3.50%  _build_foo_impl /foo/build_test.bzl:78
   22.01MB  2.96% 69.53%    22.01MB  2.96%  _build_foo_impl /foo/build_test.bzl:73
   ... (more output)
```
