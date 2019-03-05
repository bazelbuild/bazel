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

import com.google.devtools.build.lib.blackbox.framework.BuilderRunner;
import com.google.devtools.build.lib.blackbox.framework.PathUtils;
import com.google.devtools.build.lib.blackbox.junit.AbstractBlackBoxTest;
import com.google.devtools.build.lib.util.OS;
import java.nio.file.Path;
import org.junit.Test;

/** End to end test of workspace-related functionality. */
public class WorkspaceBlackBoxTest extends AbstractBlackBoxTest {

  @Test
  public void testNotInMsys() throws Exception {
    context().write("repo_rule.bzl",
        "def _impl(rctx):",
        "  result = rctx.execute(['bash', '-c', 'which bash > out.txt'])",
        "  if result.return_code != 0:",
        "    fail('Execute bash failed: ' + result.stderr)",
        "  rctx.file('BUILD', 'exports_files([\"out.txt\"])')",
        "check_bash = repository_rule(implementation = _impl)");

    context().write(WORKSPACE, "workspace(name='subdir')",
        "load(':repo_rule.bzl', 'check_bash')",
        "check_bash(name = 'check_bash_target')");

    // To make repository rule target be computed, depend on it in debug_rule
    context().write("BUILD", "load(':rule.bzl', 'debug_rule')",
        "debug_rule(name = 'check', dep = '@check_bash_target//:out.txt')");

    context().write("rule.bzl",
        "def _impl(ctx):",
        "  out = ctx.actions.declare_file('does_not_matter')",
        "  ctx.actions.write(out, 'hi')",
        "  return [DefaultInfo(files = depset([out]))]",
        "",
        "debug_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "        \"dep\": attr.label(allow_single_file = True),",
        "    }",
        ")"
    );

    BuilderRunner bazel = WorkspaceTestUtils.bazel(context());
    // The build using "bash" should fail on Windows, and pass on Linux and Mac OS
    if (OS.WINDOWS.equals(OS.getCurrent())) {
      bazel.shouldFail();
    }
    bazel.build("check");
  }

  @Test
  public void testWorkspaceChanges() throws Exception {
    Path repoA = context().getTmpDir().resolve("a");
    new RepoWithRuleWritingTextGenerator(repoA).withOutputText("hi").setupRepository();

    Path repoB = context().getTmpDir().resolve("b");
    new RepoWithRuleWritingTextGenerator(repoB).withOutputText("bye").setupRepository();

    context()
        .write(
            WORKSPACE,
            String.format(
                "local_repository(name = 'x', path = '%s',)",
                PathUtils.pathForStarlarkFile(repoA)));
    BuilderRunner bazel = WorkspaceTestUtils.bazel(context());
    bazel.build("@x//:" + RepoWithRuleWritingTextGenerator.TARGET);

    Path xPath = context().resolveBinPath(bazel, "external/x/out");
    WorkspaceTestUtils.assertLinesExactly(xPath, "hi");

    context().write(WORKSPACE,
        String.format("local_repository(name = 'x', path = '%s',)",
            PathUtils.pathForStarlarkFile(repoB)));
    bazel.build("@x//:" + RepoWithRuleWritingTextGenerator.TARGET);

    WorkspaceTestUtils.assertLinesExactly(xPath, "bye");
  }

  @Test
  public void testPathWithSpace() throws Exception {
    context().write("a b/WORKSPACE");
    BuilderRunner bazel = WorkspaceTestUtils.bazel(context());
    bazel.info();
    bazel.help();
  }

  // TODO(ichern) move other tests from workspace_test.sh here.

}
