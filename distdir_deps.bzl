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

load("//src/tools/bzlmod:utils.bzl", "get_canonical_repo_name")

##################################################################################
#
# The list of repositories required while bootstrapping Bazel offline
#
##################################################################################
DIST_ARCHIVE_REPOS = [get_canonical_repo_name(repo) for repo in [
    "abseil-cpp",
    "apple_support",
    "bazel_skylib",
    "blake3",
    "c-ares",
    "com_github_grpc_grpc",
    "com_google_protobuf",
    "io_bazel_skydoc",
    "platforms",
    "rules_cc",
    "rules_go",
    "rules_java",
    "rules_jvm_external",
    "rules_license",
    "rules_pkg",
    "rules_proto",
    "rules_python",
    "upb",
    "zlib",
    "zstd-jni",
]] + [(get_canonical_repo_name("com_github_grpc_grpc") + suffix) for suffix in [
    # Extra grpc dependencies introduced via its module extension
    "~grpc_repo_deps_ext~bazel_gazelle",  # TODO: Should be a bazel_dep
    "~grpc_repo_deps_ext~bazel_skylib",  # TODO: Should be removed
    "~grpc_repo_deps_ext~com_envoyproxy_protoc_gen_validate",
    "~grpc_repo_deps_ext~com_github_cncf_udpa",
    "~grpc_repo_deps_ext~com_google_googleapis",
    "~grpc_repo_deps_ext~envoy_api",
    "~grpc_repo_deps_ext~rules_cc",  # TODO: Should be removed
]]

DIST_DEPS = {
    ########################################
    #
    # Runtime language dependencies
    #
    ########################################
    "platforms": {
        "archive": "platforms-0.0.8.tar.gz",
        "sha256": "8150406605389ececb6da07cbcb509d5637a3ab9a24bc69b1101531367d89d74",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.8/platforms-0.0.8.tar.gz",
            "https://github.com/bazelbuild/platforms/releases/download/0.0.8/platforms-0.0.8.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
        ],
        "package_version": "0.0.8",
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
        "archive": "rules_cc-0.0.9.tar.gz",
        "sha256": "2037875b9a4456dce4a79d112a8ae885bbc4aad968e6587dca6e64f3a0900cdf",
        "urls": ["https://github.com/bazelbuild/rules_cc/releases/download/0.0.9/rules_cc-0.0.9.tar.gz"],
        "used_in": [
            "additional_distfiles",
        ],
        "package_version": "0.0.9",
        "strip_prefix": "rules_cc-0.0.9",
    },
    "rules_java": {
        "aliases": [
            "rules_java_builtin",
            "rules_java_builtin_for_testing",
        ],
        "archive": "rules_java-7.0.6.tar.gz",
        "sha256": "e81e9deaae0d9d99ef3dd5f6c1b32338447fe16d5564155531ea4eb7ef38854b",
        "urls": ["https://github.com/bazelbuild/rules_java/releases/download/7.0.6/rules_java-7.0.6.tar.gz"],
        "workspace_file_content": "",
        "used_in": [
            "additional_distfiles",
        ],
        "license_kinds": [
            "@rules_license//licenses/spdx:Apache-2.0",
        ],
        "package_version": "7.0.6",
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
        "archive": "protobuf-all-21.7.zip",
        "sha256": "5493a21f5ed3fc502e66fec6b9449c06a551ced63002fa48903c40dfa8de7a4a",
        "strip_prefix": "protobuf-21.7",
        "urls": [
            "https://github.com/protocolbuffers/protobuf/releases/download/v21.7/protobuf-all-21.7.zip",
        ],
        "patch_args": ["-p1"],
        "patches": ["//third_party/protobuf:21.7.patch"],
        "used_in": [
            "additional_distfiles",
        ],
        "license_kinds": [
            "@rules_license//licenses/generic:notice",
        ],
        "license_text": "LICENSE",
        "package_version": "21.7",
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
    "blake3": {
        "archive": "1.3.3.zip",
        "sha256": "bb529ba133c0256df49139bd403c17835edbf60d2ecd6463549c6a5fe279364d",
        "urls": [
            "https://github.com/BLAKE3-team/BLAKE3/archive/refs/tags/1.3.3.zip",
        ],
        "used_in": [
            "additional_distfiles",
        ],
        "license_kinds": [
            "@rules_license//licenses/spdx:Apache-2.0",
        ],
        "license_text": "LICENSE",
        "package_version": "1.3.3",
    },
    ###################################################
    #
    # Build time dependencies for testing and packaging
    #
    ###################################################
    "android_gmaven_r8": {
        "archive": "r8-8.1.56.jar",
        "sha256": "57a696749695a09381a87bc2f08c3a8ed06a717a5caa3ef878a3077e0d3af19d",
        "urls": [
            "https://maven.google.com/com/android/tools/r8/8.1.56/r8-8.1.56.jar",
        ],
        "used_in": [
        ],
        "package_version": "8.0.40",
    },
    "bazel_skylib": {
        "archive": "bazel-skylib-1.4.1.tar.gz",
        "sha256": "b8a1527901774180afc798aeb28c4634bdccf19c4d98e7bdd1ce79d1fe9aaad7",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.4.1/bazel-skylib-1.4.1.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.4.1/bazel-skylib-1.4.1.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
        ],
        "package_version": "1.4.1",
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
        "archive": "rules_license-0.0.7.tar.gz",
        "sha256": "4531deccb913639c30e5c7512a054d5d875698daeb75d8cf90f284375fe7c360",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_license/releases/download/0.0.7/rules_license-0.0.7.tar.gz",
            "https://github.com/bazelbuild/rules_license/releases/download/0.0.7/rules_license-0.0.7.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
        ],
        "package_version": "0.0.7",
    },
    "rules_pkg": {
        "archive": "rules_pkg-0.9.1.tar.gz",
        "sha256": "8f9ee2dc10c1ae514ee599a8b42ed99fa262b757058f65ad3c384289ff70c4b8",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.9.1/rules_pkg-0.9.1.tar.gz",
            "https://github.com/bazelbuild/rules_pkg/releases/download/0.9.1/rules_pkg-0.9.1.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
        ],
        "package_version": "0.9.1",
    },
    "rules_jvm_external": {
        "archive": "rules_jvm_external-5.2.tar.gz",
        "sha256": "f86fd42a809e1871ca0aabe89db0d440451219c3ce46c58da240c7dcdc00125f",
        "strip_prefix": "rules_jvm_external-5.2",
        "patches": [
            "//third_party:rules_jvm_external_5.2.patch",
        ],
        "patch_args": ["-p1"],
        "urls": [
            "https://github.com/bazelbuild/rules_jvm_external/releases/download/5.2/rules_jvm_external-5.2.tar.gz",
        ],
        "used_in": [
            "additional_distfiles",
        ],
        "package_version": "5.2",
    },
    "rules_python": {
        "sha256": "0a8003b044294d7840ac7d9d73eef05d6ceb682d7516781a4ec62eeb34702578",
        "strip_prefix": "rules_python-0.24.0",
        "urls": ["https://github.com/bazelbuild/rules_python/releases/download/0.24.0/rules_python-0.24.0.tar.gz"],
        "archive": "rules_python-0.24.0.tar.gz",
        "used_in": ["additional_distfiles"],
    },
    "rules_testing": {
        "sha256": "4e21f9aa7996944ce91431f27bca374bff56e680acfe497276074d56bc5d9af2",
        "strip_prefix": "rules_testing-0.0.4",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_testing/releases/download/v0.0.4/rules_testing-v0.0.4.tar.gz",
            "https://github.com/bazelbuild/rules_testing/releases/download/v0.0.4/rules_testing-v0.0.4.tar.gz",
        ],
        "archive": "rules_testing-v0.0.4.tar.gz",
        "used_in": ["additional_distfiles"],
        "package_version": "0.0.4",
    },
    "desugar_jdk_libs": {
        # Commit 24dcd1dead0b64aae3d7c89ca9646b5dc4068009 of 2023-09-18
        "archive": "24dcd1dead0b64aae3d7c89ca9646b5dc4068009.zip",
        "sha256": "ef71be474fbb3b3b7bd70cda139f01232c63b9e1bbd08c058b00a8d538d4db17",
        "strip_prefix": "desugar_jdk_libs-24dcd1dead0b64aae3d7c89ca9646b5dc4068009",
        "urls": [
            "https://github.com/google/desugar_jdk_libs/archive/24dcd1dead0b64aae3d7c89ca9646b5dc4068009.zip",
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
        ],
        "package_version": "2.6",
    },
    "openjdk_linux_vanilla": {
        "archive": "zulu21.28.85-ca-jdk21.0.0-linux_x64.tar.gz",
        "sha256": "0c0eadfbdc47a7ca64aeab51b9c061f71b6e4d25d2d87674512e9b6387e9e3a6",
        "strip_prefix": "zulu21.28.85-ca-jdk21.0.0-linux_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu21.28.85-ca-jdk21.0.0-linux_x64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu21.28.85-ca-jdk21.0.0-linux_x64.tar.gz",
        ],
        "used_in": [
        ],
    },
    "openjdk_linux_aarch64_vanilla": {
        "archive": "zulu21.28.85-ca-jdk21.0.0-linux_aarch64.tar.gz",
        "sha256": "1fb64b8036c5d463d8ab59af06bf5b6b006811e6012e3b0eb6bccf57f1c55835",
        "strip_prefix": "zulu21.28.85-ca-jdk21.0.0-linux_aarch64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu21.28.85-ca-jdk21.0.0-linux_aarch64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu21.28.85-ca-jdk21.0.0-linux_aarch64.tar.gz",
        ],
        "used_in": [
        ],
    },
    # JDK21 unavailable so use JDK19 instead for linux s390x.
    "openjdk_linux_s390x_vanilla": {
        "archive": "OpenJDK19U-jdk_s390x_linux_hotspot_19.0.2_7.tar.gz",
        "sha256": "f2512f9a8e9847dd5d3557c39b485a8e7a1ef37b601dcbcb748d22e49f44815c",
        "strip_prefix": "jdk-19.0.2+7",
        "urls": [
            "https://mirror.bazel.build/github.com/adoptium/temurin19-binaries/releases/download/jdk-19.0.2%2B7/OpenJDK19U-jdk_s390x_linux_hotspot_19.0.2_7.tar.gz",
            "https://github.com/adoptium/temurin19-binaries/releases/download/jdk-19.0.2%2B7/OpenJDK19U-jdk_s390x_linux_hotspot_19.0.2_7.tar.gz",
        ],
        "used_in": [
        ],
    },
    # JDK21 unavailable so use JDK19 instead for linux ppc64le.
    "openjdk_linux_ppc64le_vanilla": {
        "archive": "OpenJDK20U-jdk_ppc64le_linux_hotspot_20_36.tar.gz",
        "sha256": "45dde71faf8cbb78fab3c976894259655c8d3de827347f23e0ebe5710921dded",
        "strip_prefix": "jdk-20+36",
        "urls": [
            "https://mirror.bazel.build/github.com/adoptium/temurin20-binaries/releases/download/jdk-20%2B36/OpenJDK20U-jdk_ppc64le_linux_hotspot_20_36.tar.gz",
            "https://github.com/adoptium/temurin20-binaries/releases/download/jdk-20%2B36/OpenJDK20U-jdk_ppc64le_linux_hotspot_20_36.tar.gz",
        ],
        "used_in": [],
    },
    "openjdk_linux_riscv64_vanilla": {
        "archive": "OpenJDK21U-jdk_riscv64_linux_hotspot_21.0.6_7.tar.gz",
        "sha256": "203796e4ba2689aa921b5e0cdc9e02984d88622f80fcb9acb05a118b05007be8",
        "strip_prefix": "jdk-21.0.6+7",
        "urls": [
            "https://mirror.bazel.build/github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.6%2B7/OpenJDK21U-jdk_riscv64_linux_hotspot_21.0.6_7.tar.gz",
            "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.6%2B7/OpenJDK21U-jdk_riscv64_linux_hotspot_21.0.6_7.tar.gz",
        ],
        "used_in": [],
    },
    "openjdk_macos_x86_64_vanilla": {
        "archive": "zulu21.28.85-ca-jdk21.0.0-macosx_x64.tar.gz",
        "sha256": "9639b87db586d0c89f7a9892ae47f421e442c64b97baebdff31788fbe23265bd",
        "strip_prefix": "zulu21.28.85-ca-jdk21.0.0-macosx_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu21.28.85-ca-jdk21.0.0-macosx_x64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu21.28.85-ca-jdk21.0.0-macosx_x64.tar.gz",
        ],
        "used_in": [
        ],
    },
    "openjdk_macos_aarch64_vanilla": {
        "archive": "zulu21.28.85-ca-jdk21.0.0-macosx_aarch64.tar.gz",
        "sha256": "2a7a99a3ea263dbd8d32a67d1e6e363ba8b25c645c826f5e167a02bbafaff1fa",
        "strip_prefix": "zulu21.28.85-ca-jdk21.0.0-macosx_aarch64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu21.28.85-ca-jdk21.0.0-macosx_aarch64.tar.gz",
            "https://cdn.azul.com/zulu/bin/zulu21.28.85-ca-jdk21.0.0-macosx_aarch64.tar.gz",
        ],
        "used_in": [
        ],
    },
    "openjdk_win_vanilla": {
        "archive": "zulu21.28.85-ca-jdk21.0.0-win_x64.zip",
        "sha256": "e9959d500a0d9a7694ac243baf657761479da132f0f94720cbffd092150bd802",
        "strip_prefix": "zulu21.28.85-ca-jdk21.0.0-win_x64",
        "urls": [
            "https://mirror.bazel.build/cdn.azul.com/zulu/bin/zulu21.28.85-ca-jdk21.0.0-win_x64.zip",
            "https://cdn.azul.com/zulu/bin/zulu21.28.85-ca-jdk21.0.0-win_x64.zip",
        ],
        "used_in": [
        ],
    },
    # JDK21 unavailable from zulu, we'll use Microsoft's OpenJDK build instead.
    "openjdk_win_arm64_vanilla": {
        "archive": "microsoft-jdk-21.0.0-windows-aarch64.zip",
        "sha256": "975603e684f2ec5a525b3b5336d6aa0b09b5b7d2d0d9e271bd6a9892ad550181",
        "strip_prefix": "microsoft-jdk-21.0.0-windows-aarch64",
        "urls": [
            "https://mirror.bazel.build/aka.ms/download-jdk/microsoft-jdk-21.0.0-windows-aarch64.zip",
            "https://aka.ms/download-jdk/microsoft-jdk-21.0.0-windows-aarch64.zip",
        ],
        "used_in": [
        ],
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
