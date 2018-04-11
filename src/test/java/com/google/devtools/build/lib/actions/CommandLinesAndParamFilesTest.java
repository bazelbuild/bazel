// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.CommandLinesAndParamFiles.ResolvedCommandLineAndParamFiles;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.nio.charset.StandardCharsets;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CommandLinesAndParamFiles}. */
@RunWith(JUnit4.class)
public class CommandLinesAndParamFilesTest {

  private final ArtifactExpander artifactExpander = null;
  private final PathFragment execPath = PathFragment.create("output.txt");

  @Test
  public void testSimpleCommandLine() throws Exception {
    ResolvedCommandLineAndParamFiles resolved =
        CommandLinesAndParamFiles.builder()
            .addCommandLine(CommandLine.of(ImmutableList.of("--foo", "--bar")))
            .build()
            .resolve(artifactExpander, execPath, 1024, 0);
    assertThat(resolved.arguments()).containsExactly("--foo", "--bar");
    assertThat(resolved.getParamFiles()).isEmpty();
  }

  @Test
  public void testSimpleParamFileUseAlways() throws Exception {
    ResolvedCommandLineAndParamFiles resolved =
        CommandLinesAndParamFiles.builder()
            .addCommandLine(
                CommandLine.of(ImmutableList.of("--foo", "--bar")),
                ParamFileInfo.builder(ParameterFileType.UNQUOTED).setUseAlways(true).build())
            .build()
            .resolve(artifactExpander, execPath, 1024, 0);
    assertThat(resolved.arguments()).containsExactly("@output.txt-0.params");
    assertThat(resolved.getParamFiles()).hasSize(1);
    assertThat(resolved.getParamFiles().get(0).arguments).containsExactly("--foo", "--bar");
  }

  @Test
  public void testMaybeUseParamsFiles() throws Exception {
    CommandLinesAndParamFiles commandLinesAndParamFiles =
        CommandLinesAndParamFiles.builder()
            .addCommandLine(
                CommandLine.of(ImmutableList.of("--foo", "--bar")),
                ParamFileInfo.builder(ParameterFileType.UNQUOTED).setUseAlways(false).build())
            .build();
    // Set max length to longer than command line, no param file needed
    ResolvedCommandLineAndParamFiles resolved =
        commandLinesAndParamFiles.resolve(artifactExpander, execPath, 1024, 0);
    assertThat(resolved.arguments()).containsExactly("--foo", "--bar");
    assertThat(resolved.getParamFiles()).isEmpty();

    // Set max length to 0, spill to param file is forced
    resolved = commandLinesAndParamFiles.resolve(artifactExpander, execPath, 0, 0);
    assertThat(resolved.arguments()).containsExactly("@output.txt-0.params");
    assertThat(resolved.getParamFiles()).hasSize(1);
    assertThat(resolved.getParamFiles().get(0).arguments).containsExactly("--foo", "--bar");
  }

  @Test
  public void testMixOfCommandLinesAndParamFiles() throws Exception {
    ResolvedCommandLineAndParamFiles resolved =
        CommandLinesAndParamFiles.builder()
            .addCommandLine(CommandLine.of(ImmutableList.of("a", "b")))
            .addCommandLine(
                CommandLine.of(ImmutableList.of("c", "d")),
                ParamFileInfo.builder(ParameterFileType.UNQUOTED).setUseAlways(true).build())
            .addCommandLine(CommandLine.of(ImmutableList.of("e", "f")))
            .addCommandLine(
                CommandLine.of(ImmutableList.of("g", "h")),
                ParamFileInfo.builder(ParameterFileType.UNQUOTED).setUseAlways(true).build())
            .build()
            .resolve(artifactExpander, execPath, 1024, 0);
    assertThat(resolved.arguments())
        .containsExactly("a", "b", "@output.txt-0.params", "e", "f", "@output.txt-1.params");
    assertThat(resolved.getParamFiles()).hasSize(2);
    assertThat(resolved.getParamFiles().get(0).arguments).containsExactly("c", "d");
    assertThat(resolved.getParamFiles().get(0).paramFileExecPath.getPathString())
        .isEqualTo("output.txt-0.params");
    assertThat(resolved.getParamFiles().get(1).arguments).containsExactly("g", "h");
    assertThat(resolved.getParamFiles().get(1).paramFileExecPath.getPathString())
        .isEqualTo("output.txt-1.params");
  }

  @Test
  public void testFirstParamFilePassesButSecondFailsLengthTest() throws Exception {
    ResolvedCommandLineAndParamFiles resolved =
        CommandLinesAndParamFiles.builder()
            .addCommandLine(
                CommandLine.of(ImmutableList.of("a", "b")),
                ParamFileInfo.builder(ParameterFileType.UNQUOTED).setUseAlways(false).build())
            .addCommandLine(
                CommandLine.of(ImmutableList.of("c", "d")),
                ParamFileInfo.builder(ParameterFileType.UNQUOTED).setUseAlways(false).build())
            .build()
            .resolve(artifactExpander, execPath, 4, 0);
    assertThat(resolved.arguments()).containsExactly("a", "b", "@output.txt-0.params");
    assertThat(resolved.getParamFiles()).hasSize(1);
    assertThat(resolved.getParamFiles().get(0).arguments).containsExactly("c", "d");
  }

  @Test
  public void testWriteParamFiles() throws Exception {
    CommandLinesAndParamFiles commandLinesAndParamFiles =
        CommandLinesAndParamFiles.builder()
            .addCommandLine(
                CommandLine.of(ImmutableList.of("--foo", "--bar")),
                ParamFileInfo.builder(ParameterFileType.UNQUOTED).setUseAlways(true).build())
            .addCommandLine(
                CommandLine.of(ImmutableList.of("--baz")),
                ParamFileInfo.builder(ParameterFileType.UNQUOTED).setUseAlways(true).build())
            .build();
    InMemoryFileSystem inMemoryFileSystem = new InMemoryFileSystem();
    Path execRoot = inMemoryFileSystem.getPath("/exec");
    execRoot.createDirectoryAndParents();
    ResolvedCommandLineAndParamFiles resolved =
        commandLinesAndParamFiles.resolve(
            artifactExpander, PathFragment.create("my/param/file/out"), 0, 0);
    resolved.writeParamFiles(execRoot);

    assertThat(
            FileSystemUtils.readLines(
                execRoot.getRelative("my/param/file/out-0.params"), StandardCharsets.ISO_8859_1))
        .containsExactly("--foo", "--bar");
    assertThat(
            FileSystemUtils.readLines(
                execRoot.getRelative("my/param/file/out-1.params"), StandardCharsets.ISO_8859_1))
        .containsExactly("--baz");
  }
}
