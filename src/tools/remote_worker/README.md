This program implements a remote execution worker that uses gRPC to accept work
requests. It can work as a remote execution worker, a cache worker, or both.
The simplest setup is as follows:

- First build remote_worker and run it.

        bazel build src/tools/remote_worker:all
        bazel-bin/src/tools/remote_worker/remote_worker --work_path=/tmp/test \
            --listen_port=8080

- Then you run Bazel pointing to the remote_worker instance.

        bazel --host_jvm_args=-Dbazel.DigestFunction=SHA1 build \
            --spawn_strategy=remote --remote_cache=localhost:8080 \
            --remote_executor=localhost:8080 src/tools/generate_workspace:all

The above command will build generate_workspace with remote spawn strategy that
uses the local worker as the distributed caching and execution backend.

You can also use a Hazelcast server for the distributed cache as follows:
Suppose your Hazelcast server is listening on address:port. Then, run the
remote worker with --hazelcast_node=address:port.
