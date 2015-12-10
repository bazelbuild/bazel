---
layout: posts
title: Persistent Worker Processes for Bazel
---

Bazel runs most build actions as a separate process. Many build actions invoke a compiler.  However, starting a compiler is often slow: they have to perform some initialization when they start up, read the standard library, header files, low-level libraries, and so on. That’s why some compilers and tools have a persistent mode, e.g. [sjavac](http://openjdk.java.net/jeps/199), [Nailgun](http://martiansoftware.com/nailgun/) and [gcc server](http://per.bothner.com/papers/GccSummit03/gcc-server.pdf). Keeping a single process for longer and passing multiple individual requests to the same server can significantly reduce the amount of duplicate work and cut down on compile times.

In Bazel, we have recently added experimental support for delegating work to [persistent worker processes](https://github.com/bazelbuild/bazel/tree/master/src/main/java/com/google/devtools/build/lib/worker) that run as child processes of and are managed by Bazel. Our Javac wrapper (called JavaBuilder) is the first compiler that supports running as a worker.

We’ve tried the persistent JavaBuilder for a variety of builds and are seeing a ~4x improvement in Java build times, as Javac can now benefit from JIT optimizations over multiple runs and we no longer have to start a new JVM for every compile action. For Bazel itself, we saw a reduction in build time for a clean build from ~58s to ~16s (on repeated builds).

<img src="/assets/fullbuild.png" alt="Full build" class="img-responsive">
<img src="/assets/incbuild.png" alt="Incremental build" class="img-responsive">

If you often build Java code, we’d like you to give it a try. Just pass `--strategy=Javac=worker` to enable it or add `build --strategy=Javac=worker` to the .bazelrc in your home directory or in your workspace. Check the WorkerOptions class for [flags to further tune the workers’ behavior](https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/worker/WorkerOptions.java) or run “bazel help” and look for the “Strategy options” category. Let us know how it works for you.

We’re currently using a simple [protobuf-based protocol](https://github.com/bazelbuild/bazel/blob/master/src/main/protobuf/worker_protocol.proto) to communicate with the worker process. Let us know if you want to add support for more compilers; in many cases, you can do that without any Bazel changes. However, the protocol is still subject to change based on your feedback.
