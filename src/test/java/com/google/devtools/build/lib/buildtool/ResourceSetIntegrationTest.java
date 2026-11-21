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
package com.google.devtools.build.lib.buildtool;

import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Integration test for the arguments {@code resource_set} callbacks receive at execution. */
@RunWith(JUnit4.class)
public class ResourceSetIntegrationTest extends BuildIntegrationTestCase {

  @Before
  public final void useStandaloneStrategy() {
    addOptions("--spawn_strategy=standalone");
  }

  @Before
  public final void writeReportingCallback() throws Exception {
    write(
        "x/resources.bzl",
        """
        def report_inputs_size(os, inputs_size):
            fail("inputs_size = %d" % inputs_size)
        """);
  }

  @Test
  public void testInputsSize_expandsTreeArtifact() throws Exception {
    write(
        "x/defs.bzl",
        """
        load(":resources.bzl", "report_inputs_size")

        def _impl(ctx):
            tree = ctx.actions.declare_directory(ctx.label.name + ".d")
            ctx.actions.run_shell(
                outputs = [tree],
                command = "touch %s/1 %s/2 %s/3" % (tree.path, tree.path, tree.path),
            )
            out = ctx.actions.declare_file(ctx.label.name + ".out")
            ctx.actions.run_shell(
                inputs = [tree],
                outputs = [out],
                command = "touch " + out.path,
                resource_set = report_inputs_size,
            )
            return DefaultInfo(files = depset([out]))

        tree_consumer = rule(implementation = _impl)
        """);
    write(
        "x/BUILD",
        """
        load(":defs.bzl", "tree_consumer")

        tree_consumer(name = "x")
        """);

    assertThrows(BuildFailedException.class, () -> buildTarget("//x"));

    events.assertContainsError("inputs_size = 3");
  }

  @Test
  public void testInputsSize_expandsSourceDirectory() throws Exception {
    write(
        "x/defs.bzl",
        """
        load(":resources.bzl", "report_inputs_size")

        def _impl(ctx):
            out = ctx.actions.declare_file(ctx.label.name + ".out")
            ctx.actions.run_shell(
                inputs = ctx.files.srcs,
                outputs = [out],
                command = "touch " + out.path,
                resource_set = report_inputs_size,
            )
            return DefaultInfo(files = depset([out]))

        dir_consumer = rule(
            implementation = _impl,
            attrs = {"srcs": attr.label_list(allow_files = True)},
        )
        """);
    write(
        "x/BUILD",
        """
        load(":defs.bzl", "dir_consumer")

        dir_consumer(
            name = "x",
            srcs = ["dir"],
        )
        """);
    write("x/dir/1");
    write("x/dir/2");
    write("x/dir/sub/3");

    assertThrows(BuildFailedException.class, () -> buildTarget("//x"));

    events.assertContainsError("inputs_size = 3");
  }

  @Test
  public void testInputsSize_excludesTools() throws Exception {
    write(
        "x/defs.bzl",
        """
        load(":resources.bzl", "report_inputs_size")

        def _tool_impl(ctx):
            executable = ctx.actions.declare_file(ctx.label.name)
            ctx.actions.write(executable, "", is_executable = True)
            return DefaultInfo(
                executable = executable,
                runfiles = ctx.runfiles(files = ctx.files.data),
            )

        tool = rule(
            implementation = _tool_impl,
            attrs = {"data": attr.label_list(allow_files = True)},
            executable = True,
        )

        def _impl(ctx):
            out = ctx.actions.declare_file(ctx.label.name + ".out")
            ctx.actions.run(
                executable = ctx.executable.tool,
                inputs = ctx.files.srcs,
                outputs = [out],
                resource_set = report_inputs_size,
            )
            return DefaultInfo(files = depset([out]))

        tool_consumer = rule(
            implementation = _impl,
            attrs = {
                "srcs": attr.label_list(allow_files = True),
                "tool": attr.label(executable = True, cfg = "exec"),
            },
        )
        """);
    write(
        "x/BUILD",
        """
        load(":defs.bzl", "tool", "tool_consumer")

        tool(
            name = "tool",
            data = ["1", "2", "3"],
        )

        tool_consumer(
            name = "x",
            srcs = ["a", "b"],
            tool = ":tool",
        )
        """);
    write("x/1");
    write("x/2");
    write("x/3");
    write("x/a");
    write("x/b");

    assertThrows(BuildFailedException.class, () -> buildTarget("//x"));

    // The tool's executable and runfiles don't count.
    events.assertContainsError("inputs_size = 2");
  }
}
