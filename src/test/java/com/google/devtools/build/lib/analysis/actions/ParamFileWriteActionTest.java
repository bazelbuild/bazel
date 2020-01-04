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
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.util.ActionTester;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.exec.util.TestExecutorBuilder;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
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
    rootDir = ArtifactRoot.asDerivedRoot(execRoot, scratch.dir("/exec/out"));
    outputArtifact = getBinArtifactWithNoOwner("destination.txt");
    FileSystemUtils.createDirectoryAndParents(outputArtifact.getPath().getParentDirectory());
    treeArtifact = createTreeArtifact("artifact/myTreeFileArtifact");
  }


  @Test
  public void testOutputs() {
    Action action =
        createParameterFileWriteAction(
            NestedSetBuilder.emptySet(Order.STABLE_ORDER), createNormalCommandLine());
    assertThat(Artifact.toRootRelativePaths(action.getOutputs())).containsExactly(
        "destination.txt");
  }

  @Test
  public void testInputs() {
    Action action =
        createParameterFileWriteAction(
            NestedSetBuilder.create(Order.STABLE_ORDER, treeArtifact),
            createTreeArtifactExpansionCommandLineDefault());
    assertThat(Artifact.toExecPaths(action.getInputs()))
        .containsExactly("out/artifact/myTreeFileArtifact");
  }

  @Test
  public void testWriteCommandLineWithoutTreeArtifactExpansion() throws Exception {
    Action action =
        createParameterFileWriteAction(
            NestedSetBuilder.emptySet(Order.STABLE_ORDER), createNormalCommandLine());
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
            createTreeArtifactExpansionCommandLineDefault());
    ActionExecutionContext context = actionExecutionContext();
    ActionResult actionResult = action.execute(context);
    assertThat(actionResult.spawnResults()).isEmpty();
    String content = new String(FileSystemUtils.readContentAsLatin1(outputArtifact.getPath()));
    assertThat(content.trim())
        .isEqualTo(
            "--flag1\n"
                + "out/artifact/myTreeFileArtifact/artifacts/treeFileArtifact1\n"
                + "out/artifact/myTreeFileArtifact/artifacts/treeFileArtifact2");
  }

  @Test
  public void testWriteCommandLineWithTreeArtifactExpansionExpandedFunction() throws Exception {
    Action action =
        createParameterFileWriteAction(
            NestedSetBuilder.create(Order.STABLE_ORDER, treeArtifact),
            createTreeArtifactExpansionCommandLineExpandedFunction());
    ActionExecutionContext context = actionExecutionContext();
    ActionResult actionResult = action.execute(context);
    assertThat(actionResult.spawnResults()).isEmpty();
    String content = new String(FileSystemUtils.readContentAsLatin1(outputArtifact.getPath()));
    assertThat(content.trim())
        .isEqualTo(
            "--flag1=out/artifact/myTreeFileArtifact/artifacts/treeFileArtifact1\n"
                + "--flag1=out/artifact/myTreeFileArtifact/artifacts/treeFileArtifact2");
  }

  private SpecialArtifact createTreeArtifact(String rootRelativePath) {
    PathFragment relpath = PathFragment.create(rootRelativePath);
    return new SpecialArtifact(
        rootDir,
        rootDir.getExecPath().getRelative(relpath),
        ActionsTestUtil.NULL_ARTIFACT_OWNER,
        SpecialArtifactType.TREE);
  }

  private TreeFileArtifact createTreeFileArtifact(
      SpecialArtifact inputTreeArtifact, String parentRelativePath) {
    return ActionInputHelper.treeFileArtifactWithNoGeneratingActionSet(
        inputTreeArtifact,
        PathFragment.create(parentRelativePath),
        inputTreeArtifact.getArtifactOwner());
  }

  private ParameterFileWriteAction createParameterFileWriteAction(
      NestedSet<Artifact> inputTreeArtifacts, CommandLine commandLine) {
    return new ParameterFileWriteAction(
        ActionsTestUtil.NULL_ACTION_OWNER,
        inputTreeArtifacts,
        outputArtifact,
        commandLine,
        ParameterFileType.UNQUOTED,
        StandardCharsets.ISO_8859_1);
  }

  private CommandLine createNormalCommandLine() {
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

  private CommandLine createTreeArtifactExpansionCommandLineExpandedFunction() {
    return CustomCommandLine.builder()
        .addExpandedTreeArtifact(
            treeArtifact, artifact -> ImmutableList.of("--flag1=" + artifact.getExecPath()))
        .build();
  }

  private ActionExecutionContext actionExecutionContext() throws Exception {
    final Iterable<TreeFileArtifact> treeFileArtifacts = ImmutableList.of(
        createTreeFileArtifact(treeArtifact, "artifacts/treeFileArtifact1"),
        createTreeFileArtifact(treeArtifact, "artifacts/treeFileArtifact2"));

    ArtifactExpander artifactExpander = new ArtifactExpander() {
      @Override
      public void expand(Artifact artifact, Collection<? super Artifact> output) {
        for (TreeFileArtifact treeFileArtifact : treeFileArtifacts) {
          if (treeFileArtifact.getParent().equals(artifact)) {
            output.add(treeFileArtifact);
          }
        }
      }
    };

    BinTools binTools = BinTools.forUnitTesting(directories, analysisMock.getEmbeddedTools());
    Executor executor = new TestExecutorBuilder(fileSystem, directories, binTools).build();
    return new ActionExecutionContext(
        executor,
        /*actionInputFileCache=*/ null,
        ActionInputPrefetcher.NONE,
        actionKeyContext,
        /*metadataHandler=*/ null,
        LostInputsCheck.NONE,
        new FileOutErr(),
        new StoredEventHandler(),
        /*clientEnv=*/ ImmutableMap.of(),
        /*topLevelFilesets=*/ ImmutableMap.of(),
        artifactExpander,
        /*actionFileSystem=*/ null,
        /*skyframeDepsResult=*/ null);
  }

  private enum KeyAttributes {
    COMMANDLINE,
    FILE_TYPE,
    CHARSET,
  }

  @Test
  public void testComputeKey() throws Exception {
    final Artifact outputArtifact = getSourceArtifact("output");
    ActionTester.runTest(
        KeyAttributes.class,
        attributesToFlip -> {
          String arg = attributesToFlip.contains(KeyAttributes.COMMANDLINE) ? "foo" : "bar";
          CommandLine commandLine = CommandLine.of(ImmutableList.of(arg));
          ParameterFileType parameterFileType =
              attributesToFlip.contains(KeyAttributes.FILE_TYPE)
                  ? ParameterFileType.SHELL_QUOTED
                  : ParameterFileType.UNQUOTED;
          Charset charset =
              attributesToFlip.contains(KeyAttributes.CHARSET)
                  ? StandardCharsets.UTF_8
                  : StandardCharsets.US_ASCII;
          return new ParameterFileWriteAction(
              ActionsTestUtil.NULL_ACTION_OWNER,
              outputArtifact,
              commandLine,
              parameterFileType,
              charset);
        },
        actionKeyContext);
  }
}
