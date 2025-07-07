// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.actions;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext.LostInputsCheck;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.DiscoveredModulesPruner;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.ThreadStateReceiver;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.util.ActionTester;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.exec.util.FakeActionInputFileCache;
import com.google.devtools.build.lib.exec.util.TestExecutorBuilder;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.SyscallCache;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for ParamFileWriteAction. */
@RunWith(JUnit4.class)
public class ParamFileWriteActionTest extends BuildViewTestCase {
  private ArtifactRoot rootDir;
  private Artifact outputArtifact;
  private SpecialArtifact treeArtifact;

  @Before
  public void createArtifacts() throws Exception  {
    Path execRoot = scratch.getFileSystem().getPath("/exec");
    rootDir = ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, "out");
    outputArtifact = getBinArtifactWithNoOwner("destination.txt");
    outputArtifact.getPath().getParentDirectory().createDirectoryAndParents();
    treeArtifact = createTreeArtifact("artifact/myTreeFileArtifact");
  }

  @Test
  public void testOutputs() {
    Action action =
        createParameterFileWriteAction(
            NestedSetBuilder.emptySet(Order.STABLE_ORDER), createNormalCommandLine(), false);
    assertThat(Artifact.toRootRelativePaths(action.getOutputs())).containsExactly(
        "destination.txt");
  }

  @Test
  public void testInputs() {
    Action action =
        createParameterFileWriteAction(
            NestedSetBuilder.create(Order.STABLE_ORDER, treeArtifact),
            createTreeArtifactExpansionCommandLineDefault(),
            false);
    assertThat(Artifact.asExecPaths(action.getInputs()))
        .containsExactly("out/artifact/myTreeFileArtifact");
  }

  @Test
  public void testNonExecutableOutput() throws Exception {
    Action action =
        createParameterFileWriteAction(
            NestedSetBuilder.emptySet(Order.STABLE_ORDER), createNormalCommandLine(), false);
    ActionExecutionContext context = actionExecutionContext();
    action.execute(context);
    assertThat(outputArtifact.getPath().isExecutable()).isFalse();
  }

  @Test
  public void testExecutableOutput() throws Exception {
    Action action =
        createParameterFileWriteAction(
            NestedSetBuilder.emptySet(Order.STABLE_ORDER), createNormalCommandLine(), true);
    ActionExecutionContext context = actionExecutionContext();
    action.execute(context);
    assertThat(outputArtifact.getPath().isExecutable()).isTrue();
  }

  @Test
  public void testWriteCommandLineWithoutTreeArtifactExpansion() throws Exception {
    Action action =
        createParameterFileWriteAction(
            NestedSetBuilder.emptySet(Order.STABLE_ORDER), createNormalCommandLine(), false);
    ActionExecutionContext context = actionExecutionContext();
    ActionResult actionResult = action.execute(context);
    assertThat(actionResult.spawnResults()).isEmpty();
    String content = new String(FileSystemUtils.readContentAsLatin1(outputArtifact.getPath()));
    assertThat(content.trim()).isEqualTo("--flag1\n--flag2\n--flag3\nvalue1\nvalue2");
  }

  @Test
  public void testWriteCommandLineWithTreeArtifactExpansionDefault() throws Exception {
    Action action =
        createParameterFileWriteAction(
            NestedSetBuilder.create(Order.STABLE_ORDER, treeArtifact),
            createTreeArtifactExpansionCommandLineDefault(),
            false);
    ActionExecutionContext context = actionExecutionContext();
    ActionResult actionResult = action.execute(context);
    assertThat(actionResult.spawnResults()).isEmpty();
    String content = new String(FileSystemUtils.readContentAsLatin1(outputArtifact.getPath()));
    assertThat(content.trim())
        .isEqualTo(
            """
            --flag1
            out/artifact/myTreeFileArtifact/artifacts/treeFileArtifact1
            out/artifact/myTreeFileArtifact/artifacts/treeFileArtifact2\
            """);
  }

  private SpecialArtifact createTreeArtifact(String rootRelativePath) {
    return ActionsTestUtil.createTreeArtifactWithGeneratingAction(
        rootDir, rootDir.getExecPath().getRelative(rootRelativePath));
  }

  private ParameterFileWriteAction createParameterFileWriteAction(
      NestedSet<Artifact> inputTreeArtifacts, CommandLine commandLine, boolean executable) {
    return new ParameterFileWriteAction(
        ActionsTestUtil.NULL_ACTION_OWNER,
        inputTreeArtifacts,
        outputArtifact,
        commandLine,
        ParameterFileType.UNQUOTED,
        executable,
        AbstractFileWriteAction.MNEMONIC);
  }

  private static CommandLine createNormalCommandLine() {
    return CustomCommandLine.builder()
        .add("--flag1")
        .add("--flag2")
        .addAll("--flag3", ImmutableList.of("value1", "value2"))
        .build();
  }

  private CommandLine createTreeArtifactExpansionCommandLineDefault() {
    return CustomCommandLine.builder()
        .add("--flag1")
        .addExpandedTreeArtifactExecPaths(treeArtifact)
        .build();
  }

  private ActionExecutionContext actionExecutionContext() throws Exception {
    TreeFileArtifact child1 =
        TreeFileArtifact.createTreeOutput(treeArtifact, "artifacts/treeFileArtifact1");
    TreeFileArtifact child2 =
        TreeFileArtifact.createTreeOutput(treeArtifact, "artifacts/treeFileArtifact2");

    // We don't need the metadata to test the expansion of a tree artifact into the files in it, so
    // MISSING_FILE_MARKER will do
    TreeArtifactValue treeArtifactValue =
        TreeArtifactValue.newBuilder(treeArtifact)
            .putChild(child1, FileArtifactValue.MISSING_FILE_MARKER)
            .putChild(child2, FileArtifactValue.MISSING_FILE_MARKER)
            .build();

    FakeActionInputFileCache fakeActionInputFileCache = new FakeActionInputFileCache();
    fakeActionInputFileCache.putTreeArtifact(treeArtifact, treeArtifactValue);

    Executor executor = new TestExecutorBuilder(fileSystem, directories).build();
    return new ActionExecutionContext(
        executor,
        fakeActionInputFileCache,
        ActionInputPrefetcher.NONE,
        actionKeyContext,
        /* outputMetadataStore= */ null,
        /* rewindingEnabled= */ false,
        LostInputsCheck.NONE,
        new FileOutErr(),
        new StoredEventHandler(),
        /* clientEnv= */ ImmutableMap.of(),
        /* actionFileSystem= */ null,
        DiscoveredModulesPruner.DEFAULT,
        SyscallCache.NO_CACHE,
        ThreadStateReceiver.NULL_INSTANCE);
  }

  private enum KeyAttributes {
    COMMANDLINE,
    FILE_TYPE,
  }

  @Test
  public void testComputeKey() throws Exception {
    Artifact outputArtifact = getSourceArtifact("output");
    ActionTester.runTest(
        KeyAttributes.class,
        attributesToFlip -> {
          String arg = attributesToFlip.contains(KeyAttributes.COMMANDLINE) ? "foo" : "bar";
          CommandLine commandLine = CommandLine.of(ImmutableList.of(arg));
          ParameterFileType parameterFileType =
              attributesToFlip.contains(KeyAttributes.FILE_TYPE)
                  ? ParameterFileType.SHELL_QUOTED
                  : ParameterFileType.UNQUOTED;
          return new ParameterFileWriteAction(
              ActionsTestUtil.NULL_ACTION_OWNER,
              outputArtifact,
              commandLine,
              parameterFileType,
              false);
        },
        actionKeyContext);
  }
}
