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
import com.google.devtools.build.lib.actions.CommandLines.CommandLineLimits;
import com.google.devtools.build.lib.actions.CommandLines.ExpandedCommandLines;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CommandLines}. */
@RunWith(JUnit4.class)
public class CommandLinesTest {

  private final ArtifactExpander artifactExpander = null;
  private final PathFragment execPath = PathFragment.create("output.txt");
  private static final CommandLineLimits NO_LIMIT = new CommandLineLimits(10000);

  @Test
  public void testSimpleCommandLine() throws Exception {
    CommandLines commandLines =
        CommandLines.builder()
            .addCommandLine(CommandLine.of(ImmutableList.of("--foo", "--bar")))
            .build();
    ExpandedCommandLines expanded = commandLines.expand(artifactExpander, execPath, NO_LIMIT, 0);
    assertThat(commandLines.allArguments()).containsExactly("--foo", "--bar");
    assertThat(expanded.arguments()).containsExactly("--foo", "--bar");
    assertThat(expanded.getParamFiles()).isEmpty();
  }

  @Test
  public void testFromArguments() throws Exception {
    CommandLines commandLines = CommandLines.of(ImmutableList.of("--foo", "--bar"));
    ExpandedCommandLines expanded = commandLines.expand(artifactExpander, execPath, NO_LIMIT, 0);
    assertThat(commandLines.allArguments()).containsExactly("--foo", "--bar");
    assertThat(expanded.arguments()).containsExactly("--foo", "--bar");
    assertThat(expanded.getParamFiles()).isEmpty();
  }

  @Test
  public void testConcat() throws Exception {
    CommandLines commandLines =
        CommandLines.concat(
            CommandLine.of(ImmutableList.of("--before")),
            CommandLines.of(ImmutableList.of("--foo", "--bar")));
    ExpandedCommandLines expanded = commandLines.expand(artifactExpander, execPath, NO_LIMIT, 0);
    assertThat(commandLines.allArguments()).containsExactly("--before", "--foo", "--bar");
    assertThat(expanded.arguments()).containsExactly("--before", "--foo", "--bar");
    assertThat(expanded.getParamFiles()).isEmpty();
  }

  @Test
  public void testSimpleParamFileUseAlways() throws Exception {
    CommandLines commandLines =
        CommandLines.builder()
            .addCommandLine(
                CommandLine.of(ImmutableList.of("--foo", "--bar")),
                ParamFileInfo.builder(ParameterFileType.UNQUOTED).setUseAlways(true).build())
            .build();
    ExpandedCommandLines expanded = commandLines.expand(artifactExpander, execPath, NO_LIMIT, 0);
    assertThat(commandLines.allArguments()).containsExactly("--foo", "--bar");
    assertThat(expanded.arguments()).containsExactly("@output.txt-0.params");
    assertThat(expanded.getParamFiles()).hasSize(1);
    assertThat(expanded.getParamFiles().get(0).arguments).containsExactly("--foo", "--bar");
  }

  @Test
  public void testMaybeUseParamsFiles() throws Exception {
    CommandLines commandLines =
        CommandLines.builder()
            .addCommandLine(
                CommandLine.of(ImmutableList.of("--foo", "--bar")),
                ParamFileInfo.builder(ParameterFileType.UNQUOTED).setUseAlways(false).build())
            .build();
    // Set max length to longer than command line, no param file needed
    ExpandedCommandLines expanded = commandLines.expand(artifactExpander, execPath, NO_LIMIT, 0);
    assertThat(expanded.arguments()).containsExactly("--foo", "--bar");
    assertThat(expanded.getParamFiles()).isEmpty();

    // Set max length to 0, spill to param file is forced
    expanded = commandLines.expand(artifactExpander, execPath, new CommandLineLimits(0), 0);
    assertThat(expanded.arguments()).containsExactly("@output.txt-0.params");
    assertThat(expanded.getParamFiles()).hasSize(1);
    assertThat(expanded.getParamFiles().get(0).arguments).containsExactly("--foo", "--bar");
  }

  @Test
  public void testMixOfCommandLinesAndParamFiles() throws Exception {
    CommandLines commandLines =
        CommandLines.builder()
            .addCommandLine(CommandLine.of(ImmutableList.of("a", "b")))
            .addCommandLine(
                CommandLine.of(ImmutableList.of("c", "d")),
                ParamFileInfo.builder(ParameterFileType.UNQUOTED).setUseAlways(true).build())
            .addCommandLine(CommandLine.of(ImmutableList.of("e", "f")))
            .addCommandLine(
                CommandLine.of(ImmutableList.of("g", "h")),
                ParamFileInfo.builder(ParameterFileType.UNQUOTED).setUseAlways(true).build())
            .build();
    ExpandedCommandLines expanded = commandLines.expand(artifactExpander, execPath, NO_LIMIT, 0);
    assertThat(commandLines.allArguments()).containsExactly("a", "b", "c", "d", "e", "f", "g", "h");
    assertThat(expanded.arguments())
        .containsExactly("a", "b", "@output.txt-0.params", "e", "f", "@output.txt-1.params");
    assertThat(expanded.getParamFiles()).hasSize(2);
    assertThat(expanded.getParamFiles().get(0).arguments).containsExactly("c", "d");
    assertThat(expanded.getParamFiles().get(0).paramFileExecPath.getPathString())
        .isEqualTo("output.txt-0.params");
    assertThat(expanded.getParamFiles().get(1).arguments).containsExactly("g", "h");
    assertThat(expanded.getParamFiles().get(1).paramFileExecPath.getPathString())
        .isEqualTo("output.txt-1.params");
  }

  @Test
  public void testFirstParamFilePassesButSecondFailsLengthTest() throws Exception {
    CommandLines commandLines =
        CommandLines.builder()
            .addCommandLine(
                CommandLine.of(ImmutableList.of("a", "b")),
                ParamFileInfo.builder(ParameterFileType.UNQUOTED).setUseAlways(false).build())
            .addCommandLine(
                CommandLine.of(ImmutableList.of("c", "d")),
                ParamFileInfo.builder(ParameterFileType.UNQUOTED).setUseAlways(false).build())
            .build();
    ExpandedCommandLines expanded =
        commandLines.expand(artifactExpander, execPath, new CommandLineLimits(4), 0);
    assertThat(commandLines.allArguments()).containsExactly("a", "b", "c", "d");
    assertThat(expanded.arguments()).containsExactly("a", "b", "@output.txt-0.params");
    assertThat(expanded.getParamFiles()).hasSize(1);
    assertThat(expanded.getParamFiles().get(0).arguments).containsExactly("c", "d");
  }
}
