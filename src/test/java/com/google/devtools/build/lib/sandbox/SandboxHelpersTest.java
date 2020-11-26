// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.sandbox;

import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.CommandLines.ParamFileActionInput;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.util.Arrays;
import java.util.function.Function;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SandboxHelpers}. */
@RunWith(JUnit4.class)
public class SandboxHelpersTest {

  private static final ArtifactExpander EMPTY_EXPANDER = (ignored1, ignored2) -> {};
  private static final Spawn SPAWN = new SpawnBuilder().build();

  private final Scratch scratch = new Scratch();
  private Path execRoot;

  @Before
  public void createExecRoot() throws IOException {
    execRoot = scratch.dir("/execRoot");
  }

  @Test
  public void processInputFiles_materializesParamFile() throws Exception {
    SandboxHelpers sandboxHelpers = new SandboxHelpers(/*delayVirtualInputMaterialization=*/ false);
    ParamFileActionInput paramFile =
        new ParamFileActionInput(
            PathFragment.create("paramFile"),
            ImmutableList.of("-a", "-b"),
            ParameterFileType.UNQUOTED,
            UTF_8);

    SandboxInputs inputs =
        sandboxHelpers.processInputFiles(inputMap(paramFile), SPAWN, EMPTY_EXPANDER, execRoot);

    assertThat(inputs.getFiles())
        .containsExactly(PathFragment.create("paramFile"), execRoot.getChild("paramFile"));
    assertThat(inputs.getSymlinks()).isEmpty();
    assertThat(FileSystemUtils.readLines(execRoot.getChild("paramFile"), UTF_8))
        .containsExactly("-a", "-b")
        .inOrder();
    assertThat(execRoot.getChild("paramFile").isExecutable()).isTrue();
  }

  @Test
  public void processInputFiles_materializesBinToolsFile() throws Exception {
    SandboxHelpers sandboxHelpers = new SandboxHelpers(/*delayVirtualInputMaterialization=*/ false);
    BinTools.PathActionInput tool =
        new BinTools.PathActionInput(
            scratch.file("tool", "#!/bin/bash", "echo hello"),
            PathFragment.create("_bin/say_hello"));

    SandboxInputs inputs =
        sandboxHelpers.processInputFiles(inputMap(tool), SPAWN, EMPTY_EXPANDER, execRoot);

    assertThat(inputs.getFiles())
        .containsExactly(
            PathFragment.create("_bin/say_hello"), execRoot.getRelative("_bin/say_hello"));
    assertThat(inputs.getSymlinks()).isEmpty();
    assertThat(FileSystemUtils.readLines(execRoot.getRelative("_bin/say_hello"), UTF_8))
        .containsExactly("#!/bin/bash", "echo hello")
        .inOrder();
    assertThat(execRoot.getRelative("_bin/say_hello").isExecutable()).isTrue();
  }

  @Test
  public void processInputFiles_delayVirtualInputMaterialization_doesNotStoreVirtualInput()
      throws Exception {
    SandboxHelpers sandboxHelpers = new SandboxHelpers(/*delayVirtualInputMaterialization=*/ true);
    ParamFileActionInput paramFile =
        new ParamFileActionInput(
            PathFragment.create("paramFile"),
            ImmutableList.of("-a", "-b"),
            ParameterFileType.UNQUOTED,
            UTF_8);

    SandboxInputs inputs =
        sandboxHelpers.processInputFiles(inputMap(paramFile), SPAWN, EMPTY_EXPANDER, execRoot);

    assertThat(inputs.getFiles()).isEmpty();
    assertThat(inputs.getSymlinks()).isEmpty();
    assertThat(execRoot.getChild("paramFile").exists()).isFalse();
  }

  @Test
  public void sandboxInputMaterializeVirtualInputs_delayMaterialization_writesCorrectFiles()
      throws Exception {
    SandboxHelpers sandboxHelpers = new SandboxHelpers(/*delayVirtualInputMaterialization=*/ true);
    ParamFileActionInput paramFile =
        new ParamFileActionInput(
            PathFragment.create("paramFile"),
            ImmutableList.of("-a", "-b"),
            ParameterFileType.UNQUOTED,
            UTF_8);
    BinTools.PathActionInput tool =
        new BinTools.PathActionInput(
            scratch.file("tool", "tool_code"), PathFragment.create("tools/tool"));
    SandboxInputs inputs =
        sandboxHelpers.processInputFiles(
            inputMap(paramFile, tool), SPAWN, EMPTY_EXPANDER, execRoot);

    inputs.materializeVirtualInputs(scratch.dir("/sandbox"));

    Path sandboxParamFile = scratch.resolve("/sandbox/paramFile");
    assertThat(FileSystemUtils.readLines(sandboxParamFile, UTF_8))
        .containsExactly("-a", "-b")
        .inOrder();
    assertThat(sandboxParamFile.isExecutable()).isTrue();
    Path sandboxToolFile = scratch.resolve("/sandbox/tools/tool");
    assertThat(FileSystemUtils.readLines(sandboxToolFile, UTF_8)).containsExactly("tool_code");
    assertThat(sandboxToolFile.isExecutable()).isTrue();
  }

  private static ImmutableMap<PathFragment, ActionInput> inputMap(ActionInput... inputs) {
    return Arrays.stream(inputs)
        .collect(toImmutableMap(ActionInput::getExecPath, Function.identity()));
  }

  @Test
  public void atomicallyWriteVirtualInput_writesParamFile() throws Exception {
    ParamFileActionInput paramFile =
        new ParamFileActionInput(
            PathFragment.create("paramFile"),
            ImmutableList.of("-a", "-b"),
            ParameterFileType.UNQUOTED,
            UTF_8);

    SandboxHelpers.atomicallyWriteVirtualInput(
        paramFile, scratch.resolve("/outputs/paramFile"), "-1234");

    assertThat(scratch.resolve("/outputs").readdir(Symlinks.NOFOLLOW))
        .containsExactly(new Dirent("paramFile", Dirent.Type.FILE));
    Path outputFile = scratch.resolve("/outputs/paramFile");
    assertThat(FileSystemUtils.readLines(outputFile, UTF_8)).containsExactly("-a", "-b").inOrder();
    assertThat(outputFile.isExecutable()).isTrue();
  }

  @Test
  public void atomicallyWriteVirtualInput_writesBinToolsFile() throws Exception {
    BinTools.PathActionInput tool =
        new BinTools.PathActionInput(
            scratch.file("tool", "tool_code"), PathFragment.create("tools/tool"));

    SandboxHelpers.atomicallyWriteVirtualInput(tool, scratch.resolve("/outputs/tool"), "-1234");

    assertThat(scratch.resolve("/outputs").readdir(Symlinks.NOFOLLOW))
        .containsExactly(new Dirent("tool", Dirent.Type.FILE));
    Path outputFile = scratch.resolve("/outputs/tool");
    assertThat(FileSystemUtils.readLines(outputFile, UTF_8)).containsExactly("tool_code");
    assertThat(outputFile.isExecutable()).isTrue();
  }

  @Test
  public void atomicallyWriteVirtualInput_writesArbitraryVirtualInput() throws Exception {
    VirtualActionInput input = ActionsTestUtil.createVirtualActionInput("file", "hello");

    SandboxHelpers.atomicallyWriteVirtualInput(input, scratch.resolve("/outputs/file"), "-1234");

    assertThat(scratch.resolve("/outputs").readdir(Symlinks.NOFOLLOW))
        .containsExactly(new Dirent("file", Dirent.Type.FILE));
    Path outputFile = scratch.resolve("/outputs/file");
    assertThat(FileSystemUtils.readLines(outputFile, UTF_8)).containsExactly("hello");
    assertThat(outputFile.isExecutable()).isTrue();
  }
}
