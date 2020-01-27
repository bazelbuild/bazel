# Remote worker

This program implements a remote execution worker that uses gRPC to accept work
requests. It can work as a remote execution worker, a cache worker, or both.
The simplest setup is as follows:

- First build the worker and run it.

        bazel build src/tools/remote:all
        bazel-bin/src/tools/remote/worker \
            --work_path=/tmp/test \
            --listen_port=8080

- Then you run Bazel pointing to the worker instance.

        bazel build \
            --spawn_strategy=remote --remote_cache=localhost:8080 \
            --remote_executor=localhost:8080 src/tools/generate_workspace:all

The above command will build generate_workspace with remote spawn strategy that
uses the local worker as the distributed caching and execution backend.

## Sandboxing

If you run the worker on Linux, you can also enable sandboxing for increased hermeticity:

        bazel-bin/src/tools/remote/worker \
            --work_path=/tmp/test \
            --listen_port=8080 \
            --sandboxing \
            --sandboxing_writable_path=/run/shm \
            --sandboxing_tmpfs_dir=/tmp \
            --sandboxing_block_network

As you can see, the specific behavior of the sandbox can be tuned via the flags

- --sandboxing_writable_path=<path> makes a path writable for running actions.
- --sandboxing_tmpfs_dir=<path> will mount a fresh, empty tmpfs for each running action on a path.
- --sandboxing_block_network will put each running action into its own network namespace that has
  no network connectivity except for its own "localhost". Note that due to a Linux kernel issue this
  might result in a loss of performance if you run many actions in parallel. For long running tests
  it probably won't matter much, though.
