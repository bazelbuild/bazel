load("@rules_java//java:defs.bzl", "java_binary", "java_library", "java_test")

package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "ExampleWorker_lib",
    srcs = [
        "ExampleWorker.java",
        "ExampleWorkerMultiplexer.java",
        "ExampleWorkerMultiplexerOptions.java",
        "ExampleWorkerOptions.java",
        "ExampleWorkerProtocol.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:execution_requirements",
        "//src/main/java/com/google/devtools/build/lib/worker:work_request_handlers",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:worker_protocol_java_proto",
        "//third_party:gson",
        "//third_party:guava",
    ],
)

java_library(
    name = "testutil",
    testonly = 1,
    srcs = ["TestUtils.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:execution_requirements",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/worker",
        "//third_party:guava",
    ],
)

java_binary(
    name = "ExampleWorker",
    main_class = "com.google.devtools.build.lib.worker.ExampleWorker",
    runtime_deps = [":ExampleWorker_lib"],
)

java_binary(
    name = "ExampleWorkerMultiplexer",
    main_class = "com.google.devtools.build.lib.worker.ExampleWorkerMultiplexer",
    runtime_deps = [":ExampleWorker_lib"],
)

java_library(
    name = "WorkerTests_lib",
    srcs = glob(["*Test.java"]),
    deps = [
        ":testutil",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:execution_requirements",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/analysis:server_directories",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/exec:spawn_runner",
        "//src/main/java/com/google/devtools/build/lib/exec/local",
        "//src/main/java/com/google/devtools/build/lib/sandbox",
        "//src/main/java/com/google/devtools/build/lib/util:abrupt_exit_exception",
        "//src/main/java/com/google/devtools/build/lib/util:resource_converter",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/build/lib/worker",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:worker_protocol_java_proto",
        "//src/test/java/com/google/devtools/build/lib/actions/util",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//src/test/java/com/google/devtools/build/lib/vfs/util",
        "//third_party:apache_commons_pool2",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_test(
    name = "WorkerTests",
    size = "small",
    data = [
        "//src/test/java/com/google/devtools/build/lib:embedded_scripts",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":WorkerTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)

java_test(
    name = "WorkRequestHandlerTest",
    srcs = ["WorkRequestHandlerTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/worker:work_request_handlers",
        "//src/main/protobuf:worker_protocol_java_proto",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
)
