# How to run a standalone Hazelcast server for testing distributed cache.

- First you need to run a standalone Hazelcast server with default
configuration. If you already have a separate Hazelcast cluster you can skip
this step.

    java -cp third_party/hazelcast/hazelcast-3.6.4.jar \
        com.hazelcast.core.server.StartServer

- Then you run Bazel pointing to the Hazelcast server.

    bazel build --hazelcast_node=127.0.0.1:5701 --spawn_strategy=remote \
        src/tools/generate_workspace:all

Above command will build generate_workspace with remote spawn strategy that uses
Hazelcast as the distributed caching backend.

# How to run a remote worker for testing remote execution.

- First run the remote worker. This will start a standalone Hazelcast server
with default configuration.

    bazel-bin/src/tools/remote_worker/remote_worker \
        --work_path=/tmp/remote --listen_port 8080

- Then run Bazel pointing to the Hazelcast server and remote worker.

        bazel build --hazelcast_node=127.0.0.1:5701 \
            --remote_worker=127.0.0.1:8080 \
            --spawn_strategy=remote src/tools/generate_workspace:all
