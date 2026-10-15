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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.writeIsoLatin1;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Integration test for {@code File.is_directory} on source artifacts with {@code
 * --incompatible_source_directory_is_directory} and {@code TrackSourceDirectoriesFlag} enabled.
 */
@RunWith(JUnit4.class)
public final class SourceDirectoryIsDirectoryIntegrationTest extends BuildIntegrationTestCase {

  private Path sourceDir;
  private Path sourceFile;

  @Before
  public void setUpWorkspace() throws Exception {
    addOptions("--incompatible_source_directory_is_directory");
    write(
        "foo/defs.bzl",
        """
        def _kinds_impl(ctx):
            out = ctx.actions.declare_file(ctx.label.name + ".txt")
            lines = [
                "%s is %s" % (f.basename, "directory" if f.is_directory else "file")
                for f in ctx.files.srcs
            ]
            ctx.actions.write(out, "\\n".join(lines))
            return DefaultInfo(files = depset([out]))

        kinds = rule(
            implementation = _kinds_impl,
            attrs = {"srcs": attr.label_list(allow_files = True)},
        )

        def _symlink_dir_impl(ctx):
            out = ctx.actions.declare_directory(ctx.label.name + ".link")
            ctx.actions.symlink(output = out, target_file = ctx.file.src)
            return DefaultInfo(files = depset([out]))

        symlink_dir = rule(
            implementation = _symlink_dir_impl,
            attrs = {"src": attr.label(allow_single_file = True)},
        )

        def _list_children(directory, expander):
            return [f.path for f in expander.expand(directory)]

        def _args_impl(ctx):
            out = ctx.actions.declare_file(ctx.label.name + ".args")
            args = ctx.actions.args()
            args.add_all([ctx.file.src])
            args.add_all([ctx.file.src], uniquify = True)
            args.add_all(
                [ctx.file.src],
                expand_directories = False,
                map_each = _list_children,
            )
            ctx.actions.write(out, args)
            return DefaultInfo(files = depset([out]))

        args_of = rule(
            implementation = _args_impl,
            attrs = {"src": attr.label(allow_single_file = True)},
        )

        def _copy_children(template_ctx, input_directories, output_directories, tools, **kwargs):
            for f in input_directories["in"].children:
                out = template_ctx.declare_file(
                    f.basename + ".out",
                    directory = output_directories["out"],
                )
                args = template_ctx.args()
                args.add_all([f, out])
                template_ctx.run(
                    inputs = [f],
                    outputs = [out],
                    executable = tools["cp"],
                    arguments = [args],
                )

        def _map_dir_impl(ctx):
            out = ctx.actions.declare_directory(ctx.label.name + ".out")
            ctx.actions.map_directory(
                implementation = _copy_children,
                input_directories = {"in": ctx.file.src},
                output_directories = {"out": out},
                tools = {"cp": ctx.attr._cp.files_to_run},
            )
            return DefaultInfo(files = depset([out]))

        map_dir = rule(
            implementation = _map_dir_impl,
            attrs = {
                "src": attr.label(allow_single_file = True),
                "_cp": attr.label(
                    default = ":cp",
                    allow_files = True,
                    cfg = "exec",
                    executable = True,
                ),
            },
        )
        """);
    write(
        "foo/BUILD",
        """
        load(":defs.bzl", "args_of", "kinds", "map_dir", "symlink_dir")

        args_of(
            name = "args",
            src = "dir",
        )

        map_dir(
            name = "map",
            src = "dir",
        )

        genrule(
            name = "cp_tool",
            outs = ["cp"],
            cmd = "echo 'cp $$1 $$2' > $@",
            executable = True,
        )

        kinds(
            name = "kinds",
            srcs = [
                "dir",
                "file",
            ],
        )

        symlink_dir(
            name = "link",
            src = "dir",
        )
        """);

    sourceDir = getWorkspace().getRelative("foo/dir");
    sourceDir.createDirectoryAndParents();
    writeIsoLatin1(sourceDir.getRelative("child"), "content");
    sourceFile = getWorkspace().getRelative("foo/file");
    writeIsoLatin1(sourceFile, "content");
  }

  private String buildKinds() throws Exception {
    buildTarget("//foo:kinds");
    Artifact out = Iterables.getOnlyElement(getArtifacts("//foo:kinds"));
    return readContentAsLatin1String(out);
  }

  @Test
  public void sourceDirectory_isDirectory() throws Exception {
    assertThat(buildKinds()).isEqualTo("dir is directory\nfile is file");
  }

  @Test
  public void directoryReplacedByFile_isReanalyzed() throws Exception {
    assertThat(buildKinds()).isEqualTo("dir is directory\nfile is file");

    sourceDir.deleteTree();
    writeIsoLatin1(sourceDir, "content");

    assertThat(buildKinds()).isEqualTo("dir is file\nfile is file");
  }

  @Test
  public void fileReplacedByDirectory_isReanalyzed() throws Exception {
    assertThat(buildKinds()).isEqualTo("dir is directory\nfile is file");

    sourceFile.delete();
    sourceFile.createDirectory();

    assertThat(buildKinds()).isEqualTo("dir is directory\nfile is directory");
  }

  @Test
  public void togglingFlag_isReanalyzed() throws Exception {
    assertThat(buildKinds()).isEqualTo("dir is directory\nfile is file");

    addOptions("--noincompatible_source_directory_is_directory");
    assertThat(buildKinds()).isEqualTo("dir is file\nfile is file");

    addOptions("--incompatible_source_directory_is_directory");
    assertThat(buildKinds()).isEqualTo("dir is directory\nfile is file");
  }

  @Test
  public void sourceFileContentChange_doesNotReanalyzeDependents() throws Exception {
    buildKinds();
    ConfiguredTarget before = getExistingConfiguredTarget("//foo:kinds");

    writeIsoLatin1(sourceFile, "changed");
    buildKinds();

    assertThat(getExistingConfiguredTarget("//foo:kinds")).isSameInstanceAs(before);
  }

  @Test
  public void symlinkToSourceDirectory_succeeds() throws Exception {
    buildTarget("//foo:link");

    Artifact link = Iterables.getOnlyElement(getArtifacts("//foo:link"));
    assertThat(link.getPath().isSymbolicLink()).isTrue();
  }

  private List<String> buildArgs() throws Exception {
    buildTarget("//foo:args");
    Artifact out = Iterables.getOnlyElement(getArtifacts("//foo:args"));
    return readContentAsLatin1String(out).lines().toList();
  }

  @Test
  public void argsAddAll_expandsSourceDirectory() throws Exception {
    sourceDir.getRelative("sub").createDirectory();
    writeIsoLatin1(sourceDir.getRelative("sub/nested"), "content");

    assertThat(buildArgs())
        .containsExactly(
            "foo/dir/child",
            "foo/dir/sub/nested",
            "foo/dir/child",
            "foo/dir/sub/nested",
            "foo/dir/child",
            "foo/dir/sub/nested")
        .inOrder();
  }

  @Test
  public void argsAddAll_sourceDirectoryChanged_isReexpanded() throws Exception {
    assertThat(buildArgs()).containsExactly("foo/dir/child", "foo/dir/child", "foo/dir/child");

    writeIsoLatin1(sourceDir.getRelative("added"), "content");

    assertThat(buildArgs())
        .containsExactly(
            "foo/dir/added",
            "foo/dir/child",
            "foo/dir/added",
            "foo/dir/child",
            "foo/dir/added",
            "foo/dir/child")
        .inOrder();
  }

  @Test
  public void argsAddAll_flagDisabled_addsDirectoryPath() throws Exception {
    addOptions("--noincompatible_source_directory_is_directory");

    assertThat(buildArgs()).containsExactly("foo/dir", "foo/dir", "foo/dir");
  }

  private String readOutput(String name) throws Exception {
    Path out = Iterables.getOnlyElement(getArtifacts("//foo:map")).getPath();
    return new String(FileSystemUtils.readContentAsLatin1(out.getRelative(name)));
  }

  @Test
  public void mapDirectory_expandsSourceDirectory() throws Exception {
    addOptions("--experimental_allow_map_directory");
    sourceDir.getRelative("sub").createDirectory();
    writeIsoLatin1(sourceDir.getRelative("sub/nested"), "nested content");

    buildTarget("//foo:map");

    assertThat(readOutput("child.out")).isEqualTo("content\n");
    assertThat(readOutput("nested.out")).isEqualTo("nested content\n");
  }

  @Test
  public void mapDirectory_sourceDirectoryChanged_isReexpanded() throws Exception {
    addOptions("--experimental_allow_map_directory");
    buildTarget("//foo:map");

    writeIsoLatin1(sourceDir.getRelative("added"), "added content");
    buildTarget("//foo:map");

    assertThat(readOutput("added.out")).isEqualTo("added content\n");
  }

  @Test
  public void mapDirectory_flagDisabled_rejectsSourceDirectory() throws Exception {
    addOptions("--experimental_allow_map_directory");
    addOptions("--noincompatible_source_directory_is_directory");

    assertThrows(ViewCreationFailedException.class, () -> buildTarget("//foo:map"));
    assertContainsError("Expected a directory in input_directories['in'], but got file foo/dir");
  }
}
