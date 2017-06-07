licenses(["notice"])

exports_files(["LICENSE"])

load("@//third_party/grpc:build_defs.bzl", "java_grpc_library")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//third_party:__pkg__"],
)

java_proto_library(
    name = "google_devtools_build_v1_build_status_java_proto",
    visibility = ["//visibility:public"],
    deps = [":google_devtools_build_v1_build_status_proto"],
)

java_proto_library(
    name = "google_devtools_build_v1_build_events_java_proto",
    visibility = ["//visibility:public"],
    deps = [":google_devtools_build_v1_build_events_proto"],
)

java_grpc_library(
    name = "google_devtools_build_v1_publish_build_event_java_grpc",
    srcs = [":google_devtools_build_v1_publish_build_event_proto"],
    deps = [":google_devtools_build_v1_publish_build_event_java_proto"],
    visibility = ["//visibility:public"],
)

java_proto_library(
    name = "google_devtools_build_v1_publish_build_event_java_proto",
    visibility = ["//visibility:public"],
    deps = [":google_devtools_build_v1_publish_build_event_proto"],
)

java_proto_library(
    name = "google_bytestream_bytestream_java_proto",
    visibility = ["//visibility:public"],
    deps = [":google_bytestream_bytestream_proto"],
)

java_proto_library(
    name = "google_longrunning_operations_java_proto",
    visibility = ["//visibility:public"],
    deps = [":google_longrunning_operations_proto"],
)

java_proto_library(
    name = "google_watch_v1_java_proto",
    visibility = ["//visibility:public"],
    deps = [":google_watch_v1_proto"],
)

java_proto_library(
    name = "google_rpc_status_java_proto",
    visibility = ["//visibility:public"],
    deps = [":google_rpc_status_proto"],
)

java_proto_library(
    name = "google_rpc_error_details_java_proto",
    visibility = ["//visibility:public"],
    deps = [":google_rpc_error_details_proto"],
)

java_proto_library(
    name = "google_rpc_code_java_proto",
    visibility = ["//visibility:public"],
    deps = [":google_rpc_code_proto"],
)

java_proto_library(
    name = "google_devtools_remoteexecution_v1test_remote_execution_java_proto",
    visibility = ["//visibility:public"],
    deps = [":google_devtools_remoteexecution_v1test_remote_execution_proto"],
)

java_grpc_library(
    name = "google_watch_v1_java_grpc",
    srcs = [":google_watch_v1_proto"],
    visibility = ["//visibility:public"],
    deps = [":google_watch_v1_java_proto"],
)

java_grpc_library(
    name = "google_devtools_remoteexecution_v1test_remote_execution_java_grpc",
    srcs = [":google_devtools_remoteexecution_v1test_remote_execution_proto"],
    visibility = ["//visibility:public"],
    deps = [
        ":google_devtools_remoteexecution_v1test_remote_execution_java_proto",
        ":google_longrunning_operations_java_proto",
    ],
)

proto_library(
    name = "google_devtools_remoteexecution_v1test_remote_execution_proto",
    srcs = ["google/devtools/remoteexecution/v1test/remote_execution.proto"],
    visibility = ["//visibility:public"],
    deps = [
        ":google_api_annotations_proto",
        ":google_longrunning_operations_proto",
        ":google_rpc_status_proto",
        "@com_google_protobuf//:well_known_types_any_proto",
        "@com_google_protobuf//:well_known_types_duration_proto",
    ],
)

proto_library(
    name = "google_rpc_code_proto",
    srcs = ["google/rpc/code.proto"],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "google_rpc_error_details_proto",
    srcs = ["google/rpc/error_details.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_protobuf//:well_known_types_any_proto",
        "@com_google_protobuf//:well_known_types_duration_proto",
    ],
)

proto_library(
    name = "google_watch_v1_proto",
    srcs = ["google/watcher/v1/watch.proto"],
    deps = [
        ":google_api_annotations_proto",
        "@com_google_protobuf//:well_known_types_any_proto",
        "@com_google_protobuf//:well_known_types_empty_proto",
    ],
)

proto_library(
    name = "google_bytestream_bytestream_proto",
    srcs = ["google/bytestream/bytestream.proto"],
    visibility = ["//visibility:public"],
    deps = [
        ":google_api_annotations_proto",
        "@com_google_protobuf//:well_known_types_wrappers_proto",
    ],
)

proto_library(
    name = "google_longrunning_operations_proto",
    srcs = ["google/longrunning/operations.proto"],
    visibility = ["//visibility:public"],
    deps = [
        ":google_api_annotations_proto",
        ":google_api_http_proto",
        ":google_rpc_status_proto",
        "@com_google_protobuf//:well_known_types_any_proto",
        "@com_google_protobuf//:well_known_types_empty_proto",
    ],
)

proto_library(
    name = "google_devtools_build_v1_build_status_proto",
    srcs = ["google/devtools/build/v1/build_status.proto"],
    visibility = ["//visibility:public"],
    deps = [
        ":google_api_annotations_proto",
        "@com_google_protobuf//:well_known_types_any_proto",
    ],
)

proto_library(
    name = "google_devtools_build_v1_build_events_proto",
    srcs = ["google/devtools/build/v1/build_events.proto"],
    visibility = ["//visibility:public"],
    deps = [
        ":google_api_annotations_proto",
        ":google_devtools_build_v1_build_status_proto",
        ":google_rpc_status_proto",
        "@com_google_protobuf//:well_known_types_any_proto",
        "@com_google_protobuf//:well_known_types_timestamp_proto",
        "@com_google_protobuf//:well_known_types_wrappers_proto",
    ],
)

proto_library(
    name = "google_devtools_build_v1_publish_build_event_proto",
    srcs = ["google/devtools/build/v1/publish_build_event.proto"],
    visibility = ["//visibility:public"],
    deps = [
        ":google_api_annotations_proto",
        ":google_api_auth_proto",
        ":google_devtools_build_v1_build_events_proto",
        "@com_google_protobuf//:well_known_types_any_proto",
        "@com_google_protobuf//:well_known_types_duration_proto",
        "@com_google_protobuf//:well_known_types_empty_proto",
    ],
)

proto_library(
    name = "google_api_annotations_proto",
    srcs = ["google/api/annotations.proto"],
    deps = [
        ":google_api_http_proto",
        "@com_google_protobuf//:well_known_types_descriptor_proto",
    ],
)

proto_library(
    name = "google_api_http_proto",
    srcs = ["google/api/http.proto"],
)

proto_library(
    name = "google_rpc_status_proto",
    srcs = ["google/rpc/status.proto"],
    deps = ["@com_google_protobuf//:well_known_types_any_proto"],
)

proto_library(
    name = "google_api_auth_proto",
    srcs = ["google/api/auth.proto"],
    deps = [":google_api_annotations_proto"],
)

