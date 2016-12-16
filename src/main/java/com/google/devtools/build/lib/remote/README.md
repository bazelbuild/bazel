# How to run a standalone Hazelcast server for testing distributed cache.

- First you need to run a standalone Hazelcast server with default
configuration. If you already have a separate Hazelcast cluster you can skip
this step.

```
    java -cp third_party/hazelcast/hazelcast-3.6.4.jar \
        com.hazelcast.core.server.StartServer
```

- Then you run Bazel pointing to the Hazelcast server.

```
    bazel --host_jvm_args=-Dbazel.DigestFunction=SHA1 build \
        --hazelcast_node=localhost:5701 --spawn_strategy=remote \
        src/tools/generate_workspace:all
```

Above command will build generate_workspace with remote spawn strategy that uses
Hazelcast as the distributed caching backend.

# How to run a remote worker for testing remote execution.

- First run the remote worker. This will start a standalone Hazelcast server
with default configuration.

```
    bazel-bin/src/tools/remote_worker/remote_worker \
        --work_path=/tmp/remote --listen_port 8080
```

- Then run Bazel pointing to the Hazelcast server and remote worker.

```
        bazel --host_jvm_args=-Dbazel.DigestFunction=SHA1 build \
            --hazelcast_node=localhost:5701 \
            --remote_worker=localhost:8080 \
            --spawn_strategy=remote src/tools/generate_workspace:all
```

# How to run a remote worker with a remote cache server.

- First you need to run a standalone Hazelcast server with default
configuration. If you already have a separate Hazelcast cluster you can skip
this step.

```
    java -cp third_party/hazelcast/hazelcast-3.6.4.jar \
        com.hazelcast.core.server.StartServer
```

- Then run the remote cache server:

```
    bazel-bin/src/tools/remote_worker/remote_cache --listen_port 8081
```

- The run the remote worker:

```
    bazel-bin/src/tools/remote_worker/remote_worker \
        --work_path=/tmp/remote --listen_port 8080
```

- Then run Bazel pointing to the cache server and remote worker.

```
        bazel --host_jvm_args=-Dbazel.DigestFunction=SHA1 build \
            --hazelcast_node=localhost:5701 \
            --remote_worker=localhost:8080 \
            --remote_cache=localhost:8081 \
            --spawn_strategy=remote src/tools/generate_workspace:all
```
