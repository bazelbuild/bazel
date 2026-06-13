# Remote worker

This program implements a remote execution worker that uses gRPC to accept work
requests. It can work as a remote execution worker, a cache worker, or both.
The simplest setup is as follows:

- First build the worker and run it.

        bazel build src/tools/remote:all

        mkdir -p /tmp/worker/work /tmp/worker/cas

        bazel-bin/src/tools/remote/worker \
            --work_path=/tmp/worker/work \
            --cas_path=/tmp/worker/cas \
            --listen_port=8080

- Then you run Bazel pointing to the worker instance.

        bazel build \
            --remote_executor=grpc://localhost:8080 \
            src/tools/generate_workspace:all

The above command will build generate_workspace with remote spawn strategy that
uses the local worker as the distributed caching and execution backend.

## Simulating lost CAS entries

Real-world remote caches lose CAS entries, e.g. due to evictions or outages.
The worker can simulate this in a deterministic and consistent fashion:

        bazel-bin/src/tools/remote/worker \
            --work_path=/tmp/worker/work \
            --cas_path=/tmp/worker/cas \
            --listen_port=8080 \
            --lost_blob_percentage=2 \
            --lost_blob_seed=some-seed

With these flags, roughly 2% of all blobs are deleted from the CAS right after
their first upload. Since the loss is an actual deletion, all kinds of requests
observe it consistently: `FindMissingBlobs` reports the blob as missing, reads
fail with `NOT_FOUND`, executions fail with a `FAILED_PRECONDITION` that
carries a `MISSING` violation when staging the blob as an input, and action
cache hits referencing the blob are treated as stale. Whether a given blob is
lost is a deterministic function of its digest and `--lost_blob_seed`, so a
blob is only ever lost once by default and clients can always recover by
re-uploading or regenerating it.

With `--lost_blob_max_losses=N`, an affected blob is instead lost after each of
its first N uploads, so clients have to recover from the loss of the same blob
multiple times in a row (e.g. by rewinding the same action repeatedly). Bazel
tolerates losing the same input of the same action up to 20 times
(`ActionRewindStrategy.MAX_REPEATED_LOST_INPUTS`), so builds are expected to
converge for values well below that and to fail by design for values above it.

`--lost_blob_percentage` models a backend that drops blobs as they are produced.
To instead model a warm cache that has *evicted* part of its contents since a
previous build populated it, use `--evict_existing_percentage=N`:

        bazel-bin/src/tools/remote/worker \
            --work_path=/tmp/worker/work \
            --cas_path=/tmp/worker/cas \
            --listen_port=8080 \
            --evict_existing_percentage=10 \
            --lost_blob_seed=some-seed

On startup, before serving any request, the worker deletes roughly N% of the CAS
entries that already exist in `--cas_path`. Whether a given blob is evicted is a
deterministic function of its hash and `--lost_blob_seed`, independent of the
sample lost via `--lost_blob_percentage` (the two may be combined).

This is particularly useful for testing Bazel's recovery from lost inputs and
outputs via action rewinding (`--rewind_lost_inputs`). See
`src/test/shell/bazel/remote/rewinding_integration_test.sh` for an integration
test that uses `--evict_existing_percentage` to model a warm but evicted cache
and verifies that an incremental build over a synthetic target structure fails
when recovery is disabled, still fails when only whole-build retries
(`--experimental_remote_cache_eviction_retries`) are enabled, and succeeds with
action rewinding.

## Sandboxing

If you run the worker on Linux, you can also enable sandboxing for increased
hermeticity:

        mkdir --parents /tmp/worker/work /tmp/worker/cas /tmp/worker/tmpfs

        bazel-bin/src/tools/remote/worker \
            --work_path=/tmp/worker/work \
            --cas_path=/tmp/worker/cas \
            --listen_port=8080 \
            --sandboxing \
            --sandboxing_writable_path=/run/shm \
            --sandboxing_tmpfs_dir=/tmp/worker/tmpfs \
            --sandboxing_block_network

As you can see, the specific behavior of the sandbox can be tuned via the flags

- --sandboxing_writable_path=`<path>` makes a path writable for running actions.
- --sandboxing_tmpfs_dir=`<path>` will mount a fresh, empty tmpfs for each running action on a path.
- --sandboxing_block_network will put each running action into its own network
  namespace that has no network connectivity except for its own "localhost".
  Note that due to a Linux kernel issue this might result in a loss of
  performance if you run many actions in parallel. For long running tests it
  probably won't matter much, though.
