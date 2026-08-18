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

package com.google.devtools.build.lib.starlark;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Execution-level tests for {@code ctx.actions.copy}. */
@RunWith(JUnit4.class)
public final class CopyActionIntegrationTest extends BuildIntegrationTestCase {

  private Path execPath(Artifact artifact) {
    return directories
        .getExecRoot(TestConstants.WORKSPACE_NAME)
        .getRelative(artifact.getExecPath());
  }

  private Artifact getOutput(String rootRelativePath) throws Exception {
    ImmutableList<Artifact> artifacts = getArtifacts("//test:target");
    return artifacts.stream()
        .filter(a -> a.getRootRelativePathString().equals(rootRelativePath))
        .findFirst()
        .orElseThrow(() -> new AssertionError("no output " + rootRelativePath));
  }

  @Test
  public void fileCopy_hasSourceContentAtOwnPath() throws Exception {
    write(
        "test/rule.bzl",
        """
        def _impl(ctx):
            src = ctx.actions.declare_file("src.txt")
            ctx.actions.write(output = src, content = "hello copy")
            out = ctx.actions.declare_file("out.txt")
            ctx.actions.copy(output = out, target_file = src)
            return [DefaultInfo(files = depset([out]))]

        copy_rule = rule(implementation = _impl)
        """);
    write(
        "test/BUILD",
        """
        load(":rule.bzl", "copy_rule")

        copy_rule(name = "target")
        """);

    buildTarget("//test:target");

    Path out = execPath(getOutput("test/out.txt"));
    assertThat(out.exists()).isTrue();
    assertThat(new String(FileSystemUtils.readContentAsLatin1(out))).isEqualTo("hello copy");
    // A copy is realized as real content, not a followable symlink: its realpath is itself.
    assertThat(out.isSymbolicLink()).isFalse();
    assertThat(out.resolveSymbolicLinks()).isEqualTo(out);
  }

  @Test
  public void copyOfCopy_resolvesToTerminalContent() throws Exception {
    // Regression test: a copy-of-a-copy must resolve to the terminal source, never to a byteless
    // intermediate.
    write(
        "test/rule.bzl",
        """
        def _impl(ctx):
            a = ctx.actions.declare_file("a.txt")
            ctx.actions.write(output = a, content = "chained")
            b = ctx.actions.declare_file("b.txt")
            ctx.actions.copy(output = b, target_file = a)
            c = ctx.actions.declare_file("c.txt")
            ctx.actions.copy(output = c, target_file = b)
            return [DefaultInfo(files = depset([c]))]

        copy_rule = rule(implementation = _impl)
        """);
    write(
        "test/BUILD",
        """
        load(":rule.bzl", "copy_rule")

        copy_rule(name = "target")
        """);

    buildTarget("//test:target");

    Path out = execPath(getOutput("test/c.txt"));
    assertThat(out.exists()).isTrue();
    assertThat(new String(FileSystemUtils.readContentAsLatin1(out))).isEqualTo("chained");
  }

  @Test
  public void directoryCopy_hasSourceChildren() throws Exception {
    write(
        "test/rule.bzl",
        """
        def _impl(ctx):
            src = ctx.actions.declare_directory("src_dir")
            ctx.actions.run_shell(
                outputs = [src],
                command = "echo one > %s/f1.txt; echo two > %s/f2.txt" % (src.path, src.path),
            )
            out = ctx.actions.declare_directory("out_dir")
            ctx.actions.copy(output = out, target_file = src)
            return [DefaultInfo(files = depset([out]))]

        copy_rule = rule(implementation = _impl)
        """);
    write(
        "test/BUILD",
        """
        load(":rule.bzl", "copy_rule")

        copy_rule(name = "target")
        """);

    buildTarget("//test:target");

    Path outDir = execPath(getOutput("test/out_dir"));
    assertThat(outDir.getRelative("f1.txt").exists()).isTrue();
    assertThat(outDir.getRelative("f2.txt").exists()).isTrue();
    assertThat(new String(FileSystemUtils.readContentAsLatin1(outDir.getRelative("f1.txt"))))
        .contains("one");
    assertThat(new String(FileSystemUtils.readContentAsLatin1(outDir.getRelative("f2.txt"))))
        .contains("two");
  }
}
