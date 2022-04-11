load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_jar")

# This must be kept in sync with the top-level WORKSPACE file.
http_archive(
    name = "android_tools",
    sha256 = "ed5290594244c2eeab41f0104519bcef51e27c699ff4b379fcbd25215270513e",
    url = "https://mirror.bazel.build/bazel_android_tools/android_tools_pkg-0.23.0.tar.gz",
)

# This must be kept in sync with the top-level WORKSPACE file.
http_jar(
    name = "android_gmaven_r8",
    sha256 = "8626ca32fb47aba7fddd2c897615e2e8ffcdb4d4b213572a2aefb3f838f01972",
    url = "https://maven.google.com/com/android/tools/r8/3.3.28/r8-3.3.28.jar",
)
