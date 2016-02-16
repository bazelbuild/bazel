How to run a standalone Hazelcast server for testing distributed cache.

* First you need to run a standalone Hazelcast server with JCache API in the
classpath. This will start Hazelcast with the default configuration.

java -cp third_party/hazelcast/hazelcast-3.5.4.jar \
    com.hazelcast.core.server.StartServer

* Then you run Bazel pointing to the Hazelcast server.

bazel build --hazelcast_node=127.0.0.1:5701 --spawn_strategy=remote \
    src/tools/generate_workspace:all

Above command will build generate_workspace with remote spawn strategy that uses
Hazelcast as the distributed caching backend.
