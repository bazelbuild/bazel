load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def rules_jmh_deps():
  if "rules_jvm_external" not in native.existing_rules():
    http_archive(
      name = "rules_jvm_external",
      strip_prefix = "rules_jvm_external-1.2",
      sha256 = "e5c68b87f750309a79f59c2b69ead5c3221ffa54ff9496306937bfa1c9c8c86b",
      url = "https://github.com/bazelbuild/rules_jvm_external/archive/1.2.zip"
    )