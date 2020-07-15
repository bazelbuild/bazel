load("//third_party/grpc:build_defs.bzl", "java_grpc_library")

proto_library(
    name = "GreetingService_proto",
    srcs = ["src/main/proto/greet.proto"],
)

java_proto_library(
    name = "GreetingService",
    deps = [":GreetingService_proto"],
)

java_grpc_library(
    name = "GreetingService_grpc",
    srcs = [":GreetingService_proto"],
    deps = [":GreetingService"],
)

java_binary(
    name = "server",
    srcs = glob(["src/**/*.java"]),
    main_class = "com.example.grpc.server.MyGrpcServer",
    deps = [
        ":GreetingService_grpc",
        "//examples/simple-grpc-server:GreetingService",
        "//third_party/grpc:grpc-jar",
       "//third_party:guava",
       "//third_party:netty",
    ],
)
