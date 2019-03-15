# load("@bazel_tools//tools/build_defs/repo:java.bzl", "java_import_external")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# java_import_external(
#     name = "remote_all_android_tools",
#     jar_urls = ["https://storage.googleapis.com/bazel-android-mirror/all_android_tools_deploy.jar"],
#     jar_sha256 = "806152c1801df848af7f6f55f824299f819b61e8b9b55ed4cf3d3e0850c432eb",
# )

http_archive(
    name = "legacy_android_tools",
    url = "https://github.com/jin/legacy_android_tools/archive/54eefc6aa7bf4c171fecb7b6022282c277a3824c.zip",
    strip_prefix = "legacy_android_tools-54eefc6aa7bf4c171fecb7b6022282c277a3824c",
    sha256 = "c4cd965983b22272d20f5038f6fd8d143d344fac1bb638d765ec7f62e652486f",
)
