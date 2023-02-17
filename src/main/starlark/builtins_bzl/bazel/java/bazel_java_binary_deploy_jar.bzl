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

"""Auxiliary rule to create the deploy archives for java_binary

This needs to be a separate rule because we need to add the runfiles manifest as an input to
the generating actions, so that the runfiles symlink tree is staged for the deploy jars.
"""

load(":common/java/java_binary_deploy_jar.bzl", "create_deploy_archives", "make_deploy_jars_rule")
load(":common/java/java_binary.bzl", "InternalDeployJarInfo")

def _bazel_deploy_jars_impl(ctx):
    info = ctx.attr.binary[InternalDeployJarInfo]

    runfiles_manifest = ctx.attr.binary.files_to_run.runfiles_manifest
    if runfiles_manifest:
        runfiles = depset(
            [runfiles_manifest],
            transitive = [ctx.attr.binary[OutputGroupInfo]._hidden_top_level_INTERNAL_],
        )
    else:
        runfiles = depset()

    create_deploy_archives(
        ctx,
        info.java_attrs,
        info.launcher_info,
        runfiles,
        info.main_class,
        info.coverage_main_class,
        info.strip_as_default,
        info.build_info_files,
        ctx.attr.binary.label.name,
        manifest_lines = info.manifest_lines,
    )

    return []

deploy_jars = make_deploy_jars_rule(implementation = _bazel_deploy_jars_impl)
