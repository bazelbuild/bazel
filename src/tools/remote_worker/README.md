This program implements a remote execution worker that uses gRPC to accept work
requests. It also serves as a Hazelcast server for distributed caching.

- First build remote_worker and run it.

        bazel build src/tools/remote_worker:all
        bazel-bin/src/tools/remote_worker/remote_worker --work_path=/tmp/test \
            --listen_port=8080

- Then you run Bazel pointing to the remote_worker instance.

        bazel build --hazelcast_node=127.0.0.1:5701 --spawn_strategy=remote \
            --remote_worker=127.0.0.1:8080 src/tools/generate_workspace:all

The above command will build generate_workspace with remote spawn strategy that
uses Hazelcast as the distributed caching backend and executes work remotely on
the localhost remote_worker.
