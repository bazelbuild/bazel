// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Integration test for warnings issued when an artifact is a directory.
 */
@RunWith(JUnit4.class)
public class DirectoryArtifactWarningTest extends BuildIntegrationTestCase {

  private void setupGenruleWithOutputArtifactDirectory() throws Exception {
    write(
        "x/BUILD",
        """
        genrule(
            name = "x",
            srcs = [],
            outs = ["dir"],
            cmd = "mkdir $(location dir)",
        )
        """);
  }

  @Test
  public void testOutputArtifactDirectoryError_forGenrule() throws Exception {
    setupGenruleWithOutputArtifactDirectory();

    assertThrows(BuildFailedException.class, () -> buildTarget("//x"));

    events.assertContainsError(
        "output 'x/dir' of //x:x is a directory but was not declared as such");
  }

  private void setupStarlarkRuleWithOutputArtifactDirectory() throws Exception {
    write(
        "x/defs.bzl",
        """
        def _impl(ctx):
            ctx.actions.run_shell(
                outputs = [ctx.outputs.out],
                command = "mkdir %s" % ctx.outputs.out.path,
            )

        my_rule = rule(
            implementation = _impl,
            attrs = {
                "out": attr.output(),
            },
        )
        """);
    write(
        "x/BUILD",
        """
        load("defs.bzl", "my_rule")

        my_rule(
            name = "x",
            out = "dir",
        )
        """);
  }

  @Test
  public void testOutputArtifactDirectoryError_forStarlarkRule() throws Exception {
    setupStarlarkRuleWithOutputArtifactDirectory();

    assertThrows(BuildFailedException.class, () -> buildTarget("//x"));

    events.assertContainsError(
        "output 'x/dir' of //x:x is a directory but was not declared as such");
  }

  @Test
  public void testInputArtifactDirectoryWarning_forGenrule() throws Exception {
    write(
        "x/BUILD",
        """
        genrule(
            name = "x",
            srcs = ["dir"],
            outs = ["out"],
            cmd = "touch $(location out)",
        )
        """);
    write("x/dir/empty");

    buildTarget("//x");

    events.assertContainsWarning(
        "input 'x/dir' to //x:x is a directory; "
            + "dependency checking of directories is unsound");
  }

  @Test
  public void testInputArtifactDirectoryWarning_forStarlarkRule() throws Exception {
    write(
        "x/defs.bzl",
        """
        def _impl(ctx):
            ctx.actions.run_shell(
                inputs = [ctx.file.src],
                outputs = [ctx.outputs.out],
                command = "touch %s" % ctx.outputs.out.path,
            )

        my_rule = rule(
            implementation = _impl,
            attrs = {
                "src": attr.label(allow_single_file = True),
                "out": attr.output(),
            },
        )
        """);
    write(
        "x/BUILD",
        """
        load("defs.bzl", "my_rule")

        my_rule(
            name = "x",
            src = "dir",
            out = "out",
        )
        """);
    write("x/dir/empty");

    buildTarget("//x");

    events.assertContainsWarning(
        "input 'x/dir' to //x:x is a directory; "
            + "dependency checking of directories is unsound");
  }
}
