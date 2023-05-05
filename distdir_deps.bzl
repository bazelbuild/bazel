# Copyright 2020 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""List the distribution dependencies we need to build Bazel.

Note for Bazel users: This is not the file that you are looking for.
This is internal source and is not intended to tell you what version
you should use for each dependency.
"""

DIST_DEPS = {
    ########################################
    #
    # Runtime language dependencies
    #
    ########################################
    "platforms": {
        "archive": "platforms-0.0.6.tar.gz",
        "sha256": "5308fc1d8865406a49427ba24a9ab53087f17f5266a7aabbfc28823f3916e1ca",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.6/platforms-0.0.6.tar.gz",
            "https://github.com/bazelbuild/platforms/releases/download/0.0.6/platforms-0.0.6.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
        "package_version": "0.0.6",
    },
    "bazelci_rules": {
        "archive": "bazelci_rules-1.0.0.tar.gz",
        "sha256": "eca21884e6f66a88c358e580fd67a6b148d30ab57b1680f62a96c00f9bc6a07e",
        "strip_prefix": "bazelci_rules-1.0.0",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/continuous-integration/releases/download/rules-1.0.0/bazelci_rules-1.0.0.tar.gz",
            "https://github.com/bazelbuild/continuous-integration/releases/download/rules-1.0.0/bazelci_rules-1.0.0.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
        ],
        "package_version": "1.0.0",
    },
    # Used in src/main/java/com/google/devtools/build/lib/bazel/rules/cpp/cc_configure.WORKSPACE.
    # Used in src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
    # Used in src/test/java/com/google/devtools/build/lib/blackbox/framework/blackbox.WORKSAPCE
    "rules_cc": {
        "archive": "rules_cc-0.0.6.tar.gz",
        "sha256": "3d9e271e2876ba42e114c9b9bc51454e379cbf0ec9ef9d40e2ae4cec61a31b40",
        "urls": ["https://github.com/bazelbuild/rules_cc/releases/download/0.0.6/rules_cc-0.0.6.tar.gz"],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
        "package_version": "0.0.6",
        "strip_prefix": "rules_cc-0.0.6",
    },
    "rules_java": {
        "archive": "rules_java-5.4.1.tar.gz",
        "sha256": "a1f82b730b9c6395d3653032bd7e3a660f9d5ddb1099f427c1e1fe768f92e395",
        "strip_prefix": "",
        "urls": [
            "https://github.com/bazelbuild/rules_java/releases/download/5.4.1/rules_java-5.4.1.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
        "license_kinds": [
            "@rules_license//licenses/spdx:Apache-2.0",
        ],
        "package_version": "5.4.1",
    },
    # Used in src/test/java/com/google/devtools/build/lib/blackbox/framework/blackbox.WORKSAPCE
    "rules_proto": {
        "archive": "5.3.0-21.7.tar.gz",
        "sha256": "dc3fb206a2cb3441b485eb1e423165b231235a1ea9b031b4433cf7bc1fa460dd",
        "strip_prefix": "rules_proto-5.3.0-21.7",
        "urls": [
            "https://github.com/bazelbuild/rules_proto/archive/refs/tags/5.3.0-21.7.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
        "license_kinds": [
            "@rules_license//licenses/spdx:Apache-2.0",
        ],
    },
    #################################################
    #
    # Dependencies which are part of the Bazel binary
    #
    #################################################
    "com_google_protobuf": {
        "archive": "v21.7.tar.gz",
        "sha256": "75be42bd736f4df6d702a0e4e4d30de9ee40eac024c4b845d17ae4cc831fe4ae",
        "strip_prefix": "protobuf-21.7",
        "urls": [
            "https://mirror.bazel.build/github.com/protocolbuffers/protobuf/archive/v21.7.tar.gz",
            "https://github.com/protocolbuffers/protobuf/archive/v21.7.tar.gz",
        ],
        "patch_args": ["-p1"],
        "patches": ["//third_party/protobuf:21.7.patch"],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
        "license_kinds": [
            "@rules_license//licenses/generic:notice",
        ],
        "license_text": "LICENSE",
        "package_version": "3.19.6",
    },
    "com_github_grpc_grpc": {
        "archive": "v1.48.1.tar.gz",
        "sha256": "320366665d19027cda87b2368c03939006a37e0388bfd1091c8d2a96fbc93bd8",
        "strip_prefix": "grpc-1.48.1",
        "urls": [
            "https://mirror.bazel.build/github.com/grpc/grpc/archive/v1.48.1.tar.gz",
            "https://github.com/grpc/grpc/archive/v1.48.1.tar.gz",
        ],
        "patch_args": ["-p1"],
        "patches": [
            "//third_party/grpc:grpc_1.48.1.patch",
            "//third_party/grpc:grpc_1.48.1.win_arm64.patch",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
        "package_version": "1.48.1",
    },
    "com_github_cncf_udpa": {
        "archive": "cb28da3451f158a947dfc45090fe92b07b243bc1.tar.gz",
        "sha256": "5bc8365613fe2f8ce6cc33959b7667b13b7fe56cb9d16ba740c06e1a7c4242fc",
        "urls": [
            "https://mirror.bazel.build/github.com/cncf/xds/archive/cb28da3451f158a947dfc45090fe92b07b243bc1.tar.gz",
            "https://github.com/cncf/xds/archive/cb28da3451f158a947dfc45090fe92b07b243bc1.tar.gz",
        ],
        "strip_prefix": "xds-cb28da3451f158a947dfc45090fe92b07b243bc1",
        "patch_args": ["-p1"],
        "patches": [
            "//third_party/cncf_udpa:cncf_udpa_0.0.1.patch",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "com_envoyproxy_protoc_gen_validate": {
        "archive": "4694024279bdac52b77e22dc87808bd0fd732b69.tar.gz",
        "sha256": "1e490b98005664d149b379a9529a6aa05932b8a11b76b4cd86f3d22d76346f47",
        "strip_prefix": "protoc-gen-validate-4694024279bdac52b77e22dc87808bd0fd732b69",
        "urls": [
            "https://mirror.bazel.build/github.com/envoyproxy/protoc-gen-validate/archive/4694024279bdac52b77e22dc87808bd0fd732b69.tar.gz",
            "https://github.com/envoyproxy/protoc-gen-validate/archive/4694024279bdac52b77e22dc87808bd0fd732b69.tar.gz",
        ],
        "patch_args": ["-p1"],
        "patches": [
            "//third_party/protoc_gen_validate:protoc_gen_validate.patch",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "bazel_gazelle": {
        "archive": "bazel-gazelle-v0.24.0.tar.gz",
        "sha256": "de69a09dc70417580aabf20a28619bb3ef60d038470c7cf8442fafcf627c21cb",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v0.24.0/bazel-gazelle-v0.24.0.tar.gz",
            "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.24.0/bazel-gazelle-v0.24.0.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
        "package_version": "0.24.0",
    },
    "com_google_googleapis": {
        "archive": "2f9af297c84c55c8b871ba4495e01ade42476c92.tar.gz",
        "sha256": "5bb6b0253ccf64b53d6c7249625a7e3f6c3bc6402abd52d3778bfa48258703a0",
        "strip_prefix": "googleapis-2f9af297c84c55c8b871ba4495e01ade42476c92",
        "urls": [
            "https://mirror.bazel.build/github.com/googleapis/googleapis/archive/2f9af297c84c55c8b871ba4495e01ade42476c92.tar.gz",
            "https://github.com/googleapis/googleapis/archive/2f9af297c84c55c8b871ba4495e01ade42476c92.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
        "license_kinds": [
            "@rules_license//licenses/spdx:Apache-2.0",
        ],
        "license_text": "LICENSE",
    },
    "upb": {
        "archive": "a5477045acaa34586420942098f5fecd3570f577.tar.gz",
        "sha256": "cf7f71eaff90b24c1a28b49645a9ff03a9a6c1e7134291ce70901cb63e7364b5",
        "strip_prefix": "upb-a5477045acaa34586420942098f5fecd3570f577",
        "urls": [
            "https://mirror.bazel.build/github.com/protocolbuffers/upb/archive/a5477045acaa34586420942098f5fecd3570f577.tar.gz",
            "https://github.com/protocolbuffers/upb/archive/a5477045acaa34586420942098f5fecd3570f577.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
        "license_kinds": [
            "@rules_license//licenses/generic:notice",
        ],
        "license_text": "LICENSE",
    },
    "c-ares": {
        "archive": "6654436a307a5a686b008c1d4c93b0085da6e6d8.tar.gz",
        "sha256": "ec76c5e79db59762776bece58b69507d095856c37b81fd35bfb0958e74b61d93",
        "urls": [
            "https://mirror.bazel.build/github.com/c-ares/c-ares/archive/6654436a307a5a686b008c1d4c93b0085da6e6d8.tar.gz",
            "https://github.com/c-ares/c-ares/archive/6654436a307a5a686b008c1d4c93b0085da6e6d8.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "re2": {
        "archive": "aecba11114cf1fac5497aeb844b6966106de3eb6.tar.gz",
        "sha256": "9f385e146410a8150b6f4cb1a57eab7ec806ced48d427554b1e754877ff26c3e",
        "urls": [
            "https://mirror.bazel.build/github.com/google/re2/archive/aecba11114cf1fac5497aeb844b6966106de3eb6.tar.gz",
            "https://github.com/google/re2/archive/aecba11114cf1fac5497aeb844b6966106de3eb6.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
    },
    "com_google_absl": {
        "archive": "20220623.1.tar.gz",
        "sha256": "91ac87d30cc6d79f9ab974c51874a704de9c2647c40f6932597329a282217ba8",
        "urls": [
            "https://mirror.bazel.build/github.com/abseil/abseil-cpp/archive/refs/tags/20220623.1.tar.gz",
            "https://github.com/abseil/abseil-cpp/archive/refs/tags/20220623.1.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
        "strip_prefix": "abseil-cpp-20220623.1",
        "license_kinds": [
            "@rules_license//licenses/generic:notice",
        ],
        "license_text": "LICENSE",
        "package_version": "20220623.1",
    },
    "zstd-jni": {
        "archive": "v1.5.2-3.zip",
        "patch_args": ["-p1"],
        "patches": [
            "//third_party:zstd-jni/Native.java.patch",
        ],
        "sha256": "366009a43cfada35015e4cc40a7efc4b7f017c6b8df5cac3f87d2478027b2056",
        "urls": [
            "https://mirror.bazel.build/github.com/luben/zstd-jni/archive/refs/tags/v1.5.2-3.zip",
            "https://github.com/luben/zstd-jni/archive/refs/tags/v1.5.2-3.zip",
        ],
        "used_in": [
            "additional_distfiles",
        ],
        "license_kinds": [
            "@rules_license//licenses/spdx:BSD-2-Clause",
        ],
        "license_text": "LICENSE",
        "package_version": "1.5.2-3",
    },
    ###################################################
    #
    # Build time dependencies for testing and packaging
    #
    ###################################################
    "android_gmaven_r8": {
        "archive": "r8-8.0.40.jar",
        "sha256": "ab1379835c7d3e5f21f80347c3c81e2f762e0b9b02748ae5232c3afa14adf702",
        "urls": [
            "https://maven.google.com/com/android/tools/r8/8.0.40/r8-8.0.40.jar",
        ],
        "used_in": [
            "test_WORKSPACE_files",
        ],
        "package_version": "8.0.40",
    },
    "bazel_skylib": {
        "archive": "bazel-skylib-1.3.0.tar.gz",
        "sha256": "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
        "package_version": "1.3.0",
    },
    "io_bazel_skydoc": {
        "archive": "1ef781ced3b1443dca3ed05dec1989eca1a4e1cd.tar.gz",
        "sha256": "5a725b777976b77aa122b707d1b6f0f39b6020f66cd427bb111a585599c857b1",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/stardoc/archive/1ef781ced3b1443dca3ed05dec1989eca1a4e1cd.tar.gz",
            "https://github.com/bazelbuild/stardoc/archive/1ef781ced3b1443dca3ed05dec1989eca1a4e1cd.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
        ],
        "strip_prefix": "stardoc-1ef781ced3b1443dca3ed05dec1989eca1a4e1cd",
    },
    "rules_license": {
        "archive": "rules_license-0.0.3.tar.gz",
        "sha256": "00ccc0df21312c127ac4b12880ab0f9a26c1cff99442dc6c5a331750360de3c3",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_license/releases/download/0.0.3/rules_license-0.0.3.tar.gz",
            "https://github.com/bazelbuild/rules_license/releases/download/0.0.3/rules_license-0.0.3.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
        "package_version": "0.0.3",
    },
    "rules_pkg": {
        "archive": "rules_pkg-0.8.0.tar.gz",
        "sha256": "eea0f59c28a9241156a47d7a8e32db9122f3d50b505fae0f33de6ce4d9b61834",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.8.0/rules_pkg-0.8.0.tar.gz",
            "https://github.com/bazelbuild/rules_pkg/releases/download/0.8.0/rules_pkg-0.8.0.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
        "package_version": "0.8.0",
    },
    "rules_jvm_external": {
        "archive": "4.5.zip",
        "sha256": "b17d7388feb9bfa7f2fa09031b32707df529f26c91ab9e5d909eb1676badd9a6",
        "strip_prefix": "rules_jvm_external-4.5",
        "patches": [
            "//third_party:rules_jvm_external.patch",
        ],
        "patch_args": ["-p1"],
        "urls": [
            "https://github.com/bazelbuild/rules_jvm_external/archive/4.5.zip",
        ],
        "used_in": [
            "additional_distfiles",
        ],
        "package_version": "4.5",
    },
    "rules_python": {
        "sha256": "ffc7b877c95413c82bfd5482c017edcf759a6250d8b24e82f41f3c8b8d9e287e",
        "strip_prefix": "rules_python-0.19.0",
        "urls": ["https://github.com/bazelbuild/rules_python/releases/download/0.19.0/rules_python-0.19.0.tar.gz"],
        "archive": "rules_python-0.19.0.tar.gz",
        "used_in": ["additional_distfiles", "test_WORKSPACE_files"],
    },
    "rules_testing": {
        "sha256": "4e21f9aa7996944ce91431f27bca374bff56e680acfe497276074d56bc5d9af2",
        "strip_prefix": "rules_testing-0.0.4",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_testing/releases/download/v0.0.4/rules_testing-v0.0.4.tar.gz",
            "https://github.com/bazelbuild/rules_testing/releases/download/v0.0.4/rules_testing-v0.0.4.tar.gz",
        ],
        "archive": "rules_testing-v0.0.4.tar.gz",
        "used_in": ["additional_distfiles", "test_WORKSPACE_files"],
        "package_version": "0.0.4",
    },
    "desugar_jdk_libs": {
        # Commit 5847d6a06302136d95a14b4cbd4b55a9c9f1436e of 2021-03-10
        "archive": "5847d6a06302136d95a14b4cbd4b55a9c9f1436e.zip",
        "sha256": "299452e6f4a4981b2e6d22357f7332713382a63e4c137f5fd6b89579f6d610cb",
        "strip_prefix": "desugar_jdk_libs-5847d6a06302136d95a14b4cbd4b55a9c9f1436e",
        "urls": [
            "https://mirror.bazel.build/github.com/google/desugar_jdk_libs/archive/5847d6a06302136d95a14b4cbd4b55a9c9f1436e.zip",
            "https://github.com/google/desugar_jdk_libs/archive/5847d6a06302136d95a14b4cbd4b55a9c9f1436e.zip",
        ],
        "used_in": [
            "additional_distfiles",
        ],
    },
    "remote_coverage_tools": {
        "archive": "coverage_output_generator-v2.6.zip",
        "sha256": "7006375f6756819b7013ca875eab70a541cf7d89142d9c511ed78ea4fefa38af",
        "urls": [
            "https://mirror.bazel.build/bazel_coverage_output_generator/releases/coverage_output_generator-v2.6.zip",
        ],
        "used_in": [
            "test_WORKSPACE_files",
        ],
        "package_version": "2.6",
    },
    "remote_java_tools": {
        "aliases": [
            "remote_java_tools_test",
            "remote_java_tools_for_testing",
        ],
        "archive": "java_tools-v12.3.zip",
        "sha256": "cbb62ecfef61568ded46260a8e8e8430755db7ec9638c0c7ff668a656f6c042f",
        "urls": [
            "https://mirror.bazel.build/bazel_java_tools/releases/java/v12.3/java_tools-v12.3.zip",
            "https://github.com/bazelbuild/java_tools/releases/download/java_v12.3/java_tools-v12.3.zip",
        ],
        "used_in": [
            "test_WORKSPACE_files",
        ],
        "package_version": "12.3",
    },
    "remote_java_tools_linux": {
        "aliases": [
            "remote_java_tools_test_linux",
            "remote_java_tools_linux_for_testing",
        ],
        "archive": "java_tools_linux-v12.3.zip",
        "sha256": "32157b5218b151009f5b99bf5e2f65e28823d269dfbba8cd57e7da5e7cdd291d",
        "urls": [
            "https://mirror.bazel.build/bazel_java_tools/releases/java/v12.3/java_tools_linux-v12.3.zip",
            "https://github.com/bazelbuild/java_tools/releases/download/java_v12.3/java_tools_linux-v12.3.zip",
        ],
        "used_in": [
            "test_WORKSPACE_files",
        ],
        "package_version": "12.3",
    },
    "remote_java_tools_windows": {
        "aliases": [
            "remote_java_tools_test_windows",
            "remote_java_tools_windows_for_testing",
        ],
        "archive": "java_tools_windows-v12.3.zip",
        "sha256": "ec6f91387d2353eacb0ca0492f35f68c5c7b0e7a80acd1fb825088b4b069fab1",
        "urls": [
            "https://mirror.bazel.build/bazel_java_tools/releases/java/v12.3/java_tools_windows-v12.3.zip",
            "https://github.com/bazelbuild/java_tools/releases/download/java_v12.3/java_tools_windows-v12.3.zip",
        ],
        "used_in": [
            "test_WORKSPACE_files",
        ],
        "package_version": "12.3",
    },
    "remote_java_tools_darwin_x86_64": {
        "aliases": [
            "remote_java_tools_test_darwin_x86_64",
            "remote_java_tools_darwin_x86_64_for_testing",
        ],
        "archive": "java_tools_darwin_x86_64-v12.3.zip",
        "sha256": "3c3fb1967a0f35c73ff509505de53ca4611518922a6b7c8c22a468aa7503132c",
        "urls": [
            "https://mirror.bazel.build/bazel_java_tools/releases/java/v12.3/java_tools_darwin_x86_64-v12.3.zip",
            "https://github.com/bazelbuild/java_tools/releases/download/java_v12.3/java_tools_darwin_x86_64-v12.3.zip",
        ],
        "used_in": [
            "test_WORKSPACE_files",
        ],
        "package_version": "12.3",
    },
    "remote_java_tools_darwin_arm64": {
        "aliases": [
            "remote_java_tools_test_darwin_arm64",
            "remote_java_tools_darwin_arm64_for_testing",
        ],
        "archive": "java_tools_darwin_arm64-v12.3.zip",
        "sha256": "29aa0c2de4e3cf45bc55d2995ba803ecbd1173a8d363860abbc309551db7931b",
        "urls": [
            "https://mirror.bazel.build/bazel_java_tools/releases/java/v12.3/java_tools_darwin_arm64-v12.3.zip",
            "https://github.com/bazelbuild/java_tools/releases/download/java_v12.3/java_tools_darwin_arm64-v12.3.zip",
        ],
        "used_in": [
            "test_WORKSPACE_files",
        ],
        "package_version": "12.3",
    },
    "remotejdk11_linux": {
        "aliases": [
            "remotejdk11_linux_for_testing",
            "openjdk11_linux_archive",
        ],
        "archive": "zulu11.56.19-ca-jdk11.0.15-linux_x64.tar.gz",
        "sha256": "e064b61d93304012351242bf0823c6a2e41d9e28add7ea7f05378b7243d34247",
        "strip_prefix": "zulu11.56.19-ca-jdk11.0.15-linux_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu11.56.19-ca-jdk11.0.15-linux_x64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu11.56.19-ca-jdk11.0.15-linux_x64.tar.gz",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk11_linux_aarch64": {
        "aliases": [
            "remotejdk11_linux_aarch64_for_testing",
        ],
        "archive": "zulu11.56.19-ca-jdk11.0.15-linux_aarch64.tar.gz",
        "sha256": "fc7c41a0005180d4ca471c90d01e049469e0614cf774566d4cf383caa29d1a97",
        "strip_prefix": "zulu11.56.19-ca-jdk11.0.15-linux_aarch64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu-embedded/bin/zulu11.56.19-ca-jdk11.0.15-linux_aarch64.tar.gz",
            "https://cdn.azul.com/zulu-embedded/bin/zulu11.56.19-ca-jdk11.0.15-linux_aarch64.tar.gz",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk11_linux_ppc64le": {
        "aliases": [
            "remotejdk11_linux_ppc64le_for_testing",
        ],
        "sha256": "a8fba686f6eb8ae1d1a9566821dbd5a85a1108b96ad857fdbac5c1e4649fc56f",
        "strip_prefix": "jdk-11.0.15+10",
        "urls": [
            "https://mirror.bazel.build/github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.15+10/OpenJDK11U-jdk_ppc64le_linux_hotspot_11.0.15_10.tar.gz",
            "https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.15+10/OpenJDK11U-jdk_ppc64le_linux_hotspot_11.0.15_10.tar.gz",
        ],
        "used_in": [],
    },
    "remotejdk11_linux_s390x": {
        "aliases": [
            "remotejdk11_linux_s390x_for_testing",
            "openjdk11_linux_s390x_archive",
        ],
        "sha256": "a58fc0361966af0a5d5a31a2d8a208e3c9bb0f54f345596fd80b99ea9a39788b",
        "strip_prefix": "jdk-11.0.15+10",
        "urls": [
            "https://mirror.bazel.build/github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.15+10/OpenJDK11U-jdk_s390x_linux_hotspot_11.0.15_10.tar.gz",
            "https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.15+10/OpenJDK11U-jdk_s390x_linux_hotspot_11.0.15_10.tar.gz",
        ],
        "used_in": [],
    },
    "remotejdk11_macos": {
        "aliases": [
            "remotejdk11_macos_for_testing",
            "openjdk11_darwin_archive",
        ],
        "archive": "zulu11.56.19-ca-jdk11.0.15-macosx_x64.tar.gz",
        "sha256": "2614e5c5de8e989d4d81759de4c333aa5b867b17ab9ee78754309ba65c7f6f55",
        "strip_prefix": "zulu11.56.19-ca-jdk11.0.15-macosx_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu11.56.19-ca-jdk11.0.15-macosx_x64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu11.56.19-ca-jdk11.0.15-macosx_x64.tar.gz",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk11_macos_aarch64": {
        "aliases": [
            "remotejdk11_macos_aarch64_for_testing",
            "openjdk11_darwin_aarch64_archive",
        ],
        "archive": "zulu11.56.19-ca-jdk11.0.15-macosx_aarch64.tar.gz",
        "sha256": "6bb0d2c6e8a29dcd9c577bbb2986352ba12481a9549ac2c0bcfd00ed60e538d2",
        "strip_prefix": "zulu11.56.19-ca-jdk11.0.15-macosx_aarch64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu11.56.19-ca-jdk11.0.15-macosx_aarch64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu11.56.19-ca-jdk11.0.15-macosx_aarch64.tar.gz",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk11_win": {
        "aliases": [
            "remotejdk11_win_for_testing",
            "openjdk11_windows_archive",
        ],
        "archive": "zulu11.56.19-ca-jdk11.0.15-win_x64.zip",
        "sha256": "a106c77389a63b6bd963a087d5f01171bd32aa3ee7377ecef87531390dcb9050",
        "strip_prefix": "zulu11.56.19-ca-jdk11.0.15-win_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu11.56.19-ca-jdk11.0.15-win_x64.zip",
            "https://cdn.azul.com/zulu/bin/zulu11.56.19-ca-jdk11.0.15-win_x64.zip",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk17_win_arm64": {
        "aliases": [
            "remotejdk17_win_arm64_for_testing",
            "openjdk17_windows_arm64_archive",
            "openjdk_win_arm64_vanilla",
        ],
        "archive": "zulu17.38.21-ca-jdk17.0.5-win_aarch64.zip",
        "sha256": "bc3476f2161bf99bc9a243ff535b8fc033b34ce9a2fa4b62fb8d79b6bfdc427f",
        "strip_prefix": "zulu17.38.21-ca-jdk17.0.5-win_aarch64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-win_aarch64.zip",
            "https://cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-win_aarch64.zip",
        ],
        "used_in": [
            "test_WORKSPACE_files",
        ],
    },
    "remotejdk11_win_arm64": {
        "aliases": [
            "remotejdk11_win_arm64_for_testing",
            "openjdk11_windows_arm64_archive",
        ],
        "archive": "microsoft-jdk-11.0.13.8.1-windows-aarch64.zip",
        "sha256": "b8a28e6e767d90acf793ea6f5bed0bb595ba0ba5ebdf8b99f395266161e53ec2",
        "strip_prefix": "jdk-11.0.13+8",
        "urls": [
            "https://mirror.bazel.build/aka.ms/download-jdk/microsoft-jdk-11.0.13.8.1-windows-aarch64.zip",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk17_linux": {
        "aliases": [
            "remotejdk17_linux_for_testing",
            "openjdk17_linux_archive",
            "openjdk_linux_vanilla",
        ],
        "archive": "zulu17.38.21-ca-jdk17.0.5-linux_x64.tar.gz",
        "sha256": "20c91a922eec795f3181eaa70def8b99d8eac56047c9a14bfb257c85b991df1b",
        "strip_prefix": "zulu17.38.21-ca-jdk17.0.5-linux_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-linux_x64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-linux_x64.tar.gz",
        ],
        "used_in": [
            "test_WORKSPACE_files",
        ],
    },
    "remotejdk17_linux_aarch64": {
        "aliases": [
            "remotejdk17_linux_aarch64_for_testing",
            "openjdk17_linux_aarch64_archive",
            "openjdk_linux_aarch64_vanilla",
        ],
        "archive": "zulu17.38.21-ca-jdk17.0.5-linux_aarch64.tar.gz",
        "sha256": "dbc6ae9163e7ff469a9ab1f342cd1bc1f4c1fb78afc3c4f2228ee3b32c4f3e43",
        "strip_prefix": "zulu17.38.21-ca-jdk17.0.5-linux_aarch64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-linux_aarch64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-linux_aarch64.tar.gz",
        ],
        "used_in": [
            "test_WORKSPACE_files",
        ],
    },
    "remotejdk17_linux_s390x": {
        "aliases": [
            "remotejdk17_linux_s390x_for_testing",
            "openjdk17_linux_s390x_archive",
            "openjdk_linux_s390x_vanilla",
        ],
        "archive": "OpenJDK17U-jdk_s390x_linux_hotspot_17.0.4.1_1.tar.gz",
        "sha256": "fdc82f4b06c880762503b0cb40e25f46cf8190d06011b3b768f4091d3334ef7f",
        "strip_prefix": "jdk-17.0.4.1+1",
        "urls": [
            "https://mirror.bazel.build/github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.4.1%2B1/OpenJDK17U-jdk_s390x_linux_hotspot_17.0.4.1_1.tar.gz",
            "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.4.1%2B1/OpenJDK17U-jdk_s390x_linux_hotspot_17.0.4.1_1.tar.gz",
        ],
        "used_in": [
            "test_WORKSPACE_files",
        ],
    },
    "remotejdk17_linux_ppc64le": {
        "aliases": [
            "remotejdk17_linux_ppc64le_for_testing",
            "openjdk_linux_ppc64le_vanilla",
        ],
        "archive": "OpenJDK17U-jdk_ppc64le_linux_hotspot_17.0.4.1_1.tar.gz",
        "sha256": "cbedd0a1428b3058d156e99e8e9bc8769e0d633736d6776a4c4d9136648f2fd1",
        "strip_prefix": "jdk-17.0.4.1+1",
        "urls": [
            "https://mirror.bazel.build/github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.4.1%2B1/OpenJDK17U-jdk_ppc64le_linux_hotspot_17.0.4.1_1.tar.gz",
            "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.4.1%2B1/OpenJDK17U-jdk_ppc64le_linux_hotspot_17.0.4.1_1.tar.gz",
        ],
        "used_in": [],
    },
    "remotejdk17_macos": {
        "aliases": [
            "remotejdk17_macos_for_testing",
            "openjdk17_darwin_archive",
            "openjdk_macos_x86_64_vanilla",
        ],
        "archive": "zulu17.38.21-ca-jdk17.0.5-macosx_x64.tar.gz",
        "sha256": "e6317cee4d40995f0da5b702af3f04a6af2bbd55febf67927696987d11113b53",
        "strip_prefix": "zulu17.38.21-ca-jdk17.0.5-macosx_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-macosx_x64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-macosx_x64.tar.gz",
        ],
        "used_in": [
            "test_WORKSPACE_files",
        ],
    },
    "remotejdk17_macos_aarch64": {
        "aliases": [
            "remotejdk17_macos_aarch64_for_testing",
            "openjdk17_darwin_aarch64_archive",
            "openjdk_macos_aarch64_vanilla",
        ],
        "archive": "zulu17.38.21-ca-jdk17.0.5-macosx_aarch64",
        "sha256": "515dd56ec99bb5ae8966621a2088aadfbe72631818ffbba6e4387b7ee292ab09",
        "strip_prefix": "zulu17.38.21-ca-jdk17.0.5-macosx_aarch64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-macosx_aarch64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-macosx_aarch64.tar.gz",
        ],
        "used_in": [
            "test_WORKSPACE_files",
        ],
    },
    "remotejdk17_win": {
        "aliases": [
            "remotejdk17_win_for_testing",
            "openjdk17_windows_archive",
            "openjdk_win_vanilla",
        ],
        "archive": "zulu17.38.21-ca-jdk17.0.5-win_x64.zip",
        "sha256": "9972c5b62a61b45785d3d956c559e079d9e91f144ec46225f5deeda214d48f27",
        "strip_prefix": "zulu17.38.21-ca-jdk17.0.5-win_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-win_x64.zip",
            "https://cdn.azul.com/zulu/bin/zulu17.38.21-ca-jdk17.0.5-win_x64.zip",
        ],
        "used_in": [
            "test_WORKSPACE_files",
        ],
    },
    "remotejdk20_linux_aarch64": {
        "aliases": [
            "remotejdk20_linux_for_testing",
        ],
        "archive": "zulu20.28.85-ca-jdk20.0.0-linux_aarch64.tar.gz",
        "sha256": "47ce58ead9a05d5d53b96706ff6fa0eb2e46755ee67e2b416925e28f5b55038a",
        "strip_prefix": "zulu20.28.85-ca-jdk20.0.0-linux_aarch64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu20.28.85-ca-jdk20.0.0-linux_aarch64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu20.28.85-ca-jdk20.0.0-linux_aarch64.tar.gz",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk20_linux": {
        "aliases": [
            "remotejdk20_linux_for_testing",
            "openjdk20_linux_archive",
        ],
        "archive": "zulu20.28.85-ca-jdk20.0.0-linux_x64.tar.gz",
        "sha256": "0386418db7f23ae677d05045d30224094fc13423593ce9cd087d455069893bac",
        "strip_prefix": "zulu20.28.85-ca-jdk20.0.0-linux_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu20.28.85-ca-jdk20.0.0-linux_x64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu20.28.85-ca-jdk20.0.0-linux_x64.tar.gz",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk20_macos": {
        "aliases": [
            "remotejdk20_macos_for_testing",
            "openjdk20_darwin_archive",
        ],
        "archive": "zulu20.28.85-ca-jdk20.0.0-macosx_x64.tar.gz",
        "sha256": "fde6cc17a194ea0d9b0c6c0cb6178199d8edfc282d649eec2c86a9796e843f86",
        "strip_prefix": "zulu20.28.85-ca-jdk20.0.0-macosx_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu20.28.85-ca-jdk20.0.0-macosx_x64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu20.28.85-ca-jdk20.0.0-macosx_x64.tar.gz",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk20_macos_aarch64": {
        "aliases": [
            "remotejdk20_macos_aarch64_for_testing",
            "openjdk20_darwin_aarch64_archive",
        ],
        "archive": "zulu20.28.85-ca-jdk20.0.0-macosx_aarch64.tar.gz",
        "sha256": "a2eff6a940c2df3a2352278027e83f5959f34dcfc8663034fe92be0f1b91ce6f",
        "strip_prefix": "zulu20.28.85-ca-jdk20.0.0-macosx_aarch64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu20.28.85-ca-jdk20.0.0-macosx_aarch64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu20.28.85-ca-jdk20.0.0-macosx_aarch64.tar.gz",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
    "remotejdk20_win": {
        "aliases": [
            "remotejdk20_win_for_testing",
            "openjdk20_windows_archive",
        ],
        "archive": "zulu20.28.85-ca-jdk20.0.0-win_x64.zip",
        "sha256": "ac5f6a7d84dbbb0bb4d376feb331cc4c49a9920562f2a5e85b7a6b4863b10e1e",
        "strip_prefix": "zulu20.28.85-ca-jdk20.0.0-win_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu20.28.85-ca-jdk20.0.0-win_x64.zip",
            "https://cdn.azul.com/zulu/bin/zulu20.28.85-ca-jdk20.0.0-win_x64.zip",
        ],
        "used_in": ["test_WORKSPACE_files"],
    },
}

# Add aliased names
DEPS_BY_NAME = {}

def _create_index():
    for repo_name in DIST_DEPS:
        repo = DIST_DEPS[repo_name]
        DEPS_BY_NAME[repo_name] = repo
        aliases = repo.get("aliases")
        if aliases:
            for alias in aliases:
                DEPS_BY_NAME[alias] = repo

_create_index()

def _gen_workspace_stanza_impl(ctx):
    if ctx.attr.template and (ctx.attr.preamble or ctx.attr.postamble):
        fail("Can not use template with either preamble or postamble")
    if ctx.attr.use_maybe and ctx.attr.repo_clause:
        fail("Can not use use_maybe with repo_clause")

    if ctx.attr.use_maybe:
        repo_clause = """
maybe(
    http_archive,
    name = "{repo}",
    sha256 = "{sha256}",
    strip_prefix = {strip_prefix},
    urls = {urls},
)
"""
    elif ctx.attr.repo_clause:
        repo_clause = ctx.attr.repo_clause
    else:
        repo_clause = """
http_archive(
    name = "{repo}",
    sha256 = "{sha256}",
    strip_prefix = {strip_prefix},
    urls = {urls},
)
"""

    repo_stanzas = {}
    for repo in ctx.attr.repos:
        info = DEPS_BY_NAME[repo]
        strip_prefix = info.get("strip_prefix")
        if strip_prefix:
            strip_prefix = "\"%s\"" % strip_prefix
        else:
            strip_prefix = "None"

        repo_stanzas["{%s}" % repo] = repo_clause.format(
            repo = repo,
            sha256 = str(info["sha256"]),
            strip_prefix = strip_prefix,
            urls = info["urls"],
        )

    if ctx.attr.template:
        ctx.actions.expand_template(
            output = ctx.outputs.out,
            template = ctx.file.template,
            substitutions = repo_stanzas,
        )
    else:
        content = "\n".join([p.strip() for p in ctx.attr.preamble.strip().split("\n")])
        content += "\n"
        content += "".join(repo_stanzas.values())
        content += "\n"
        content += "\n".join([p.strip() for p in ctx.attr.postamble.strip().split("\n")])
        content += "\n"
        ctx.actions.write(ctx.outputs.out, content)

    return [DefaultInfo(files = depset([ctx.outputs.out]))]

gen_workspace_stanza = rule(
    attrs = {
        "repos": attr.string_list(doc = "Set of repos to include."),
        "out": attr.output(mandatory = True),
        "preamble": attr.string(doc = "Preamble."),
        "postamble": attr.string(doc = "Set of rules to follow repos."),
        "template": attr.label(
            doc = "Template WORKSPACE file. May not be used with preamble or postamble." +
                  "Repo stanzas can be included using the syntax '{repo name}'.",
            allow_single_file = True,
            mandatory = False,
        ),
        "use_maybe": attr.bool(doc = "Use maybe() invocation instead of http_archive."),
        "repo_clause": attr.string(doc = "Use a custom clause for each repository."),
    },
    doc = "Use specifications from DIST_DEPS to generate WORKSPACE http_archive stanzas or to" +
          "drop them into a template.",
    implementation = _gen_workspace_stanza_impl,
)
