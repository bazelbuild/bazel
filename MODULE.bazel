# To build Bazel with Bzlmod
# 1. Copy WORKSPACE.bzlmod to replace the original WORKSPACE file.
# 2. Run `bazel build --experimental_enable_bzlmod //src:bazel_nojdk`.

module(
    name = "bazel",
    version = "6.0.0-pre",
    repo_name = "io_bazel",
)

bazel_dep(name = "rules_license", version = "0.0.3")
bazel_dep(name = "bazel_skylib", version = "1.2.0")
bazel_dep(name = "protobuf", version = "3.19.6", repo_name = "com_google_protobuf")
bazel_dep(name = "grpc", version = "1.47.0", repo_name = "com_github_grpc_grpc")
bazel_dep(name = "platforms", version = "0.0.5")
bazel_dep(name = "rules_pkg", version = "0.7.0")
bazel_dep(name = "stardoc", version = "0.5.0", repo_name = "io_bazel_skydoc")
bazel_dep(name = "zstd-jni", version = "1.5.2-3")
bazel_dep(name = "zlib", version = "1.2.13")

# The following are required when building without WORKSPACE SUFFIX
bazel_dep(name = "rules_cc", version = "0.0.2")
bazel_dep(name = "rules_java", version = "5.5.0")
bazel_dep(name = "rules_proto", version = "4.0.0")

# TODO(pcloudy): Add remoteapis and googleapis as Bazel modules in the BCR.
bazel_dep(name = "remoteapis", version = "")
bazel_dep(name = "googleapis", version = "")
local_path_override(
  module_name = "remoteapis",
  path = "./third_party/remoteapis",
)
local_path_override(
  module_name = "googleapis",
  path = "./third_party/googleapis",
)

# TODO(pcloudy): Remove this when rules_jvm_external adopts Bzlmod.
single_version_override(
    module_name = "protobuf",
    patches = ["//third_party/protobuf:3.19.6.bzlmod.patch"],
    patch_strip = 1,
)
