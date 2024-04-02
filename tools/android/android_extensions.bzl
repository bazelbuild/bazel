# Copyright 2022 The Bazel Authors. All rights reserved.
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

"""Module extension to declare Android runtime dependencies for Bazel."""

load("//tools/build_defs/repo:http.bzl", "http_archive", "http_jar")

def _remote_android_tools_extensions_impl(_ctx):
    http_archive(
        name = "android_tools",
        sha256 = "2b661a761a735b41c41b3a78089f4fc1982626c76ddb944604ae3ff8c545d3c2",  # DO_NOT_REMOVE_THIS_ANDROID_TOOLS_UPDATE_MARKER
        url = "https://mirror.bazel.build/bazel_android_tools/android_tools_pkg-0.30.0.tar",
    )
    http_jar(
        name = "android_gmaven_r8",
        sha256 = "59753e70a74f918389cc87f1b7d66b5c0862932559167425708ded159e3de439",
        url = "https://maven.google.com/com/android/tools/r8/8.3.37/r8-8.3.37.jar",
    )

remote_android_tools_extensions = module_extension(
    implementation = _remote_android_tools_extensions_impl,
)

def _android_external_repository_impl(repo_ctx):
    repo_ctx.file(
        "BUILD",
        """
alias(
  name  = "has_androidsdk",
  actual = "%s",
  visibility = ["//visibility:public"],
)
alias(
  name  = "sdk",
  actual = "%s",
  visibility = ["//visibility:public"],
)
alias(
  name  = "dx_jar_import",
  actual = "%s",
  visibility = ["//visibility:public"],
)
alias(
  name = "android_sdk_for_testing",
  actual = "%s",
  visibility = ["//visibility:public"],
)
alias(
  name = "android_ndk_for_testing",
  actual = "%s",
  visibility = ["//visibility:public"],
)
""" % (
            repo_ctx.attr.has_androidsdk,
            repo_ctx.attr.sdk,
            repo_ctx.attr.dx_jar_import,
            repo_ctx.attr.android_sdk_for_testing,
            repo_ctx.attr.android_ndk_for_testing,
        ),
    )

    pass

android_external_repository = repository_rule(
    implementation = _android_external_repository_impl,
    attrs = {
        "has_androidsdk": attr.label(default = "@bazel_tools//tools/android:always_false"),
        "sdk": attr.label(default = "@bazel_tools//tools/android:poison_pill_android_sdk"),
        "dx_jar_import": attr.label(default = "@bazel_tools//tools/android:no_android_sdk_repository_error"),
        "android_sdk_for_testing": attr.label(default = "@bazel_tools//tools/android:empty"),
        "android_ndk_for_testing": attr.label(default = "@bazel_tools//tools/android:empty"),
    },
    local = True,
)

def _android_sdk_proxy_extensions_impl(module_ctx):
    root_modules = [m for m in module_ctx.modules if m.is_root]
    if len(root_modules) > 1:
        fail("Expected at most one root module, found {}".format(", ".join([x.name for x in root_modules])))

    if root_modules:
        module = root_modules[0]
    else:
        module = module_ctx.modules[0]

    kwargs = {}
    if module.tags.configure:
        kwargs["has_androidsdk"] = module.tags.configure[0].has_androidsdk
        kwargs["sdk"] = module.tags.configure[0].sdk
        kwargs["dx_jar_import"] = module.tags.configure[0].dx_jar_import
        kwargs["android_sdk_for_testing"] = module.tags.configure[0].android_sdk_for_testing
        kwargs["android_ndk_for_testing"] = module.tags.configure[0].android_ndk_for_testing

    android_external_repository(
        name = "android_external",
        **kwargs
    )

android_sdk_proxy_extensions = module_extension(
    implementation = _android_sdk_proxy_extensions_impl,
    tag_classes = {
        "configure": tag_class(attrs = {
            "has_androidsdk": attr.label(default = "@bazel_tools//tools/android:always_false"),
            "sdk": attr.label(default = "@bazel_tools//tools/android:poison_pill_android_sdk"),
            "dx_jar_import": attr.label(default = "@bazel_tools//tools/android:no_android_sdk_repository_error"),
            "android_sdk_for_testing": attr.label(default = "@bazel_tools//tools/android:empty"),
            "android_ndk_for_testing": attr.label(default = "@bazel_tools//tools/android:empty"),
        }),
    },
)
