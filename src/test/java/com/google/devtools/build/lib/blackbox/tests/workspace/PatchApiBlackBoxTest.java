// Copyright 2019 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.blackbox.tests.workspace;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.blackbox.framework.BuilderRunner;
import com.google.devtools.build.lib.blackbox.framework.PathUtils;
import com.google.devtools.build.lib.blackbox.framework.ProcessResult;
import com.google.devtools.build.lib.blackbox.junit.AbstractBlackBoxTest;
import java.io.IOException;
import java.nio.file.Path;
import org.junit.Test;

/**
 * End to end test of patch API we exposed in @bazel_tools//tools/build_defs/repo:utils.bzl. The
 * patch API is used in http_repository and git_repository.
 *
 * <p>The idea is to use a custom repository rules that use the API to patch existing files.
 */
public class PatchApiBlackBoxTest extends AbstractBlackBoxTest {

  private void setUpPatchTestRepo(
      ImmutableList<String> patchArgs, String patchTool, boolean hasPatchCmdsWin)
      throws IOException {
    StringBuilder patchArgsStr = new StringBuilder("\"-p1\"");
    if (patchArgs != null) {
      patchArgsStr = new StringBuilder();
      for (String arg : patchArgs) {
        patchArgsStr.append("\"").append(arg).append("\", ");
      }
    }
    context()
        .write(
            "patched_repo.bzl",
            """
            load(
                "@bazel_tools//tools/build_defs/repo:utils.bzl",
                "patch",
                "workspace_and_buildfile",
            )

            _common_attrs = {
                "files": attr.string_dict(default = {}),
                "patches": attr.label_list(default = []),
                "patch_tool": attr.string(default = ""),
                "patch_args": attr.string_list(default = []),
                "patch_cmds": attr.string_list(default = []),
                "patch_cmds_win": attr.string_list(default = []),
                "build_file": attr.label(allow_single_file = True),
                "build_file_content": attr.string(),
                "workspace_file": attr.label(),
                "workspace_file_content": attr.string(),
            }

            def _patched_repo_implementation(ctx):
                for file_name, label in ctx.attr.files.items():
                    ctx.template(file_name, ctx.path(Label(label)))
                workspace_and_buildfile(ctx)
                patch(ctx)

            patched_repo = repository_rule(
                implementation = _patched_repo_implementation,
                attrs = _common_attrs,
            )
            """);
    context()
        .write(
            WORKSPACE,
            "load(\":patched_repo.bzl\", \"patched_repo\")",
            "",
            "patched_repo(",
            "    name = \"test\",",
            "    files = {\"foo.sh\" : \"//:foo.sh\"},",
            "    patches = [\"//:remove-dragons.patch\"],",
            String.format("    patch_args = [%s],", patchArgsStr.toString()),
            (patchTool == null ? "" : String.format("    patch_tool = \"%s\",", patchTool)),
            "    patch_cmds = [",
            "      \"find . -name '*.sh' -exec sed -i.bak '1s|/usr/bin/env sh|/bin/sh|' {} +\",",
            "      \"chmod u+x ./foo.sh\",",
            "    ],",
            (hasPatchCmdsWin
                ? "    patch_cmds_win = [\"(Get-Content -path foo.sh) -replace '/usr/bin/env"
                    + " sh','/bin/sh' | Set-Content -Path foo.sh\"],"
                : ""),
            "    build_file_content =",
            "    \"\"\"",
            "filegroup(",
            "    name = \"foo\",",
            "    srcs = [\"foo.sh\"],",
            ")",
            "    \"\"\"",
            ")");
    context().write("BUILD");
    context().write("foo.sh", "#!/usr/bin/env sh", "", "echo Here be dragons...", "");
    context()
        .write(
            "remove-dragons.patch",
            "#!/usr/bin/env sh",
            "diff --git a/foo.sh b/foo.sh",
            "index 1f4c41e..9d548ff 100644",
            "--- a/foo.sh",
            "+++ b/foo.sh",
            "@@ -1,4 +1,4 @@",
            " #!/usr/bin/env sh",
            "",
            "-echo Here be dragons...",
            "+echo New version of foo.sh, no more dangerous animals...",
            "");
  }

  @Test
  public void testPatchApiUsingNativePatch() throws Exception {
    setUpPatchTestRepo(null, null, true);
    BuilderRunner bazel = WorkspaceTestUtils.bazel(context());
    bazel.build("@test//:foo");
    assertFooIsPatched(bazel);
  }

  @Test
  public void testPatchApiUsingNativePatchFailed() throws Exception {
    // Using -p2 should cause an error
    setUpPatchTestRepo(ImmutableList.of("-p2"), null, true);
    BuilderRunner bazel = WorkspaceTestUtils.bazel(context()).shouldFail();
    ProcessResult result = bazel.build("@test//:foo");
    assertThat(result.errString())
        .contains("Cannot determine file name with strip = 2 at line 4:\n--- a/foo.sh");
  }

  @Test
  public void testFallBackToPatchToolDueToPatchArgs() throws Exception {
    // Native patch doesn't support -b argument, should fallback to patch command line tool.
    setUpPatchTestRepo(ImmutableList.of("-p1", "-b"), null, true);
    BuilderRunner bazel = WorkspaceTestUtils.bazel(context());
    if (isWindows()) {
      // On Windows, we expect no patch tool in PATH after removing MSYS paths from PATH env var.
      bazel.shouldFail();
    }
    ProcessResult result = bazel.build("@test//:foo");
    if (isWindows()) {
      assertThat(result.errString())
          .contains("CreateProcessW(\"C:\\foo\\bar\\usr\\bin\\bash.exe\" -c \"patch '-p1' '-b'");
      assertThat(result.errString()).contains("The system cannot find the file specified.");
    } else {
      assertFooIsPatched(bazel);
      // foo.sh.orig should be generated due to "-b" argument.
      Path fooOrig = context().resolveExecRootPath(bazel, "external/test/foo.sh.orig");
      assertThat(fooOrig.toFile().exists()).isTrue();
    }
  }

  @Test
  public void testFallBackToPatchToolWhenItIsSpecified() throws Exception {
    // Should fallback to the specified patch tool.
    setUpPatchTestRepo(null, "patch", true);
    BuilderRunner bazel = WorkspaceTestUtils.bazel(context());
    if (isWindows()) {
      // On Windows, we expect no patch tool in PATH after removing MSYS paths from PATH env var.
      bazel.shouldFail();
    }
    ProcessResult result = bazel.build("@test//:foo");
    if (isWindows()) {
      assertThat(result.errString())
          .contains("CreateProcessW(\"C:\\foo\\bar\\usr\\bin\\bash.exe\" -c \"patch '-p1'");
      assertThat(result.errString()).contains("The system cannot find the file specified.");
    } else {
      assertFooIsPatched(bazel);
    }
  }

  @Test
  public void testFallBackToPatchCmdsWhenPatchCmdsWinNotSpecified() throws Exception {
    setUpPatchTestRepo(null, null, false);
    BuilderRunner bazel = WorkspaceTestUtils.bazel(context());
    if (isWindows()) {
      // On Windows, we expect no bash tool in PATH after removing MSYS paths from PATH env var.
      bazel.shouldFail();
    }
    ProcessResult result = bazel.build("@test//:foo");
    if (isWindows()) {
      assertThat(result.errString())
          .contains("CreateProcessW(\"C:\\foo\\bar\\usr\\bin\\bash.exe\" -c");
      assertThat(result.errString()).contains("The system cannot find the file specified.");
    } else {
      assertFooIsPatched(bazel);
    }
  }

  private void assertFooIsPatched(BuilderRunner bazel) throws Exception {
    Path foo = context().resolveExecRootPath(bazel, "external/test/foo.sh");
    assertThat(foo.toFile().exists()).isTrue();
    ImmutableList<String> patchedFoo =
        ImmutableList.of(
            "#!/bin/sh", "", "echo New version of foo.sh, no more dangerous animals...", "");
    assertThat(PathUtils.readFile(foo)).containsExactlyElementsIn(patchedFoo);
  }
}
