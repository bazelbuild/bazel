// Copyright 2026 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.rules.android;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.SpawnActionTemplate;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link AndroidStarlarkCommon#createDexMergerActions}. */
@RunWith(JUnit4.class)
public final class AndroidStarlarkCommonDexMergerTest extends BuildViewTestCase {

  @Test
  public void createDexMergerActions_registersSpawnActionTemplateForDirectoryArtifacts()
      throws Exception {
    scratch.file(
        "test/defs.bzl",
        """
        def _tool_impl(ctx):
            exe = ctx.actions.declare_file(ctx.label.name)
            ctx.actions.write(output = exe, content = "", is_executable = True)
            return [DefaultInfo(executable = exe)]

        dexmerger_tool = rule(
            implementation = _tool_impl,
            executable = True,
        )

        def _impl(ctx):
            in_dir = ctx.actions.declare_directory(ctx.label.name + "_in")
            ctx.actions.run(outputs = [in_dir], executable = ctx.executable._dexmerger)
            out_dir = ctx.actions.declare_directory(ctx.label.name + "_out")
            android_common.create_dex_merger_actions(
                ctx,
                input = in_dir,
                output = out_dir,
                dexopts = ["--opt"],
                dexmerger = ctx.attr._dexmerger[DefaultInfo].files_to_run,
                min_sdk_version = 21,
            )
            return [DefaultInfo(files = depset([out_dir]))]

        dex_merge_rule = rule(
            implementation = _impl,
            attrs = {
                "_dexmerger": attr.label(
                    default = "//test:dexmerger",
                    executable = True,
                    cfg = "exec",
                ),
            },
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load(":defs.bzl", "dex_merge_rule", "dexmerger_tool")

        dexmerger_tool(name = "dexmerger")

        dex_merge_rule(name = "merged")
        """);

    ConfiguredTarget target = getConfiguredTarget("//test:merged");
    Artifact outDir = Iterables.getOnlyElement(getFilesToBuild(target).toList());
    assertThat(outDir.isTreeArtifact()).isTrue();
    ActionAnalysisMetadata action = getGeneratingActionAnalysisMetadata(outDir);
    assertThat(action).isInstanceOf(SpawnActionTemplate.class);
    assertThat(action.getMnemonic()).isEqualTo("DexShardsToMerge");
  }

  @Test
  public void createDexMergerActions_nonDirectoryOutput_failsWithEvalError() throws Exception {
    scratch.file(
        "test/defs.bzl",
        """
        def _tool_impl(ctx):
            exe = ctx.actions.declare_file(ctx.label.name)
            ctx.actions.write(output = exe, content = "", is_executable = True)
            return [DefaultInfo(executable = exe)]

        dexmerger_tool = rule(
            implementation = _tool_impl,
            executable = True,
        )

        def _impl(ctx):
            in_dir = ctx.actions.declare_directory(ctx.label.name + "_in")
            ctx.actions.run(outputs = [in_dir], executable = ctx.executable._dexmerger)
            out_file = ctx.actions.declare_file(ctx.label.name + "_out.jar")
            android_common.create_dex_merger_actions(
                ctx,
                input = in_dir,
                output = out_file,
                dexopts = [],
                dexmerger = ctx.attr._dexmerger[DefaultInfo].files_to_run,
            )
            return [DefaultInfo(files = depset([out_file]))]

        dex_merge_rule = rule(
            implementation = _impl,
            attrs = {
                "_dexmerger": attr.label(
                    default = "//test:dexmerger",
                    executable = True,
                    cfg = "exec",
                ),
            },
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load(":defs.bzl", "dex_merge_rule", "dexmerger_tool")

        dexmerger_tool(name = "dexmerger")

        dex_merge_rule(name = "bad_output")
        """);

    checkError(
        "//test:bad_output",
        "Expected 'output' to be a directory artifact, but got test/bad_output_out.jar");
  }

  @Test
  public void createDexMergerActions_nonDirectoryInput_failsWithEvalError() throws Exception {
    scratch.file(
        "test/defs.bzl",
        """
        def _tool_impl(ctx):
            exe = ctx.actions.declare_file(ctx.label.name)
            ctx.actions.write(output = exe, content = "", is_executable = True)
            return [DefaultInfo(executable = exe)]

        dexmerger_tool = rule(
            implementation = _tool_impl,
            executable = True,
        )

        def _impl(ctx):
            in_file = ctx.actions.declare_file(ctx.label.name + "_in.jar")
            ctx.actions.write(output = in_file, content = "")
            out_dir = ctx.actions.declare_directory(ctx.label.name + "_out")
            android_common.create_dex_merger_actions(
                ctx,
                input = in_file,
                output = out_dir,
                dexopts = [],
                dexmerger = ctx.attr._dexmerger[DefaultInfo].files_to_run,
            )
            return [DefaultInfo(files = depset([out_dir]))]

        dex_merge_rule = rule(
            implementation = _impl,
            attrs = {
                "_dexmerger": attr.label(
                    default = "//test:dexmerger",
                    executable = True,
                    cfg = "exec",
                ),
            },
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load(":defs.bzl", "dex_merge_rule", "dexmerger_tool")

        dexmerger_tool(name = "dexmerger")

        dex_merge_rule(name = "bad_input")
        """);

    checkError(
        "//test:bad_input",
        "Expected 'input' to be a directory artifact, but got test/bad_input_in.jar");
  }
}
