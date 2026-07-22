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

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.vfs.Path;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Integration test for {@code File.is_directory} on source artifacts with {@code
 * SourceDirectoryIsDirectoryFlag} and {@code TrackSourceDirectoriesFlag} enabled.
 */
@RunWith(JUnit4.class)
public final class SourceDirectoryIsDirectoryIntegrationTest extends BuildIntegrationTestCase {

  private Path sourceDir;
  private Path sourceFile;

  @Before
  public void setUpWorkspace() throws Exception {
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
        """);
    write(
        "foo/BUILD",
        """
        load(":defs.bzl", "kinds", "symlink_dir")

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
  public void symlinkToSourceDirectory_succeeds() throws Exception {
    buildTarget("//foo:link");

    Artifact link = Iterables.getOnlyElement(getArtifacts("//foo:link"));
    assertThat(link.getPath().isSymbolicLink()).isTrue();
  }
}
