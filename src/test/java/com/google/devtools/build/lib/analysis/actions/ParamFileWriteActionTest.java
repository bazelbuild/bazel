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
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.exec.util.TestExecutorBuilder;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for ParamFileWriteAction. */
@RunWith(JUnit4.class)
public class ParamFileWriteActionTest extends BuildViewTestCase {
  private Root rootDir;
  private Artifact outputArtifact;
  private Artifact treeArtifact;

  @Before
  public void createArtifacts() throws Exception  {
    rootDir = Root.asDerivedRoot(scratch.dir("/exec/root"));
    outputArtifact = getBinArtifactWithNoOwner("destination.txt");
    FileSystemUtils.createDirectoryAndParents(outputArtifact.getPath().getParentDirectory());
    treeArtifact = createTreeArtifact("artifact/myTreeFileArtifact");
  }


  @Test
  public void testOutputs() {
    Action action = createParameterFileWriteAction(
        ImmutableList.<Artifact>of(), createNormalCommandLine());
    assertThat(Artifact.toRootRelativePaths(action.getOutputs())).containsExactly(
        "destination.txt");
  }

  @Test
  public void testInputs() {
    Action action = createParameterFileWriteAction(
        ImmutableList.of(treeArtifact),
        createTreeArtifactExpansionCommandLine());
    assertThat(Artifact.toExecPaths(action.getInputs())).containsExactly(
        "artifact/myTreeFileArtifact");
  }

  @Test
  public void testWriteCommandLineWithoutTreeArtifactExpansion() throws Exception {
    Action action = createParameterFileWriteAction(
        ImmutableList.<Artifact>of(), createNormalCommandLine());
    ActionExecutionContext context = actionExecutionContext();
    action.execute(context);
    String content = new String(FileSystemUtils.readContentAsLatin1(outputArtifact.getPath()));
    assertThat(content.trim()).isEqualTo("--flag1\n--flag2\n--flag3\nvalue1\nvalue2");
  }

  @Test
  public void testWriteCommandLineWithTreeArtifactExpansion() throws Exception {
    Action action = createParameterFileWriteAction(
        ImmutableList.of(treeArtifact),
        createTreeArtifactExpansionCommandLine());
    ActionExecutionContext context = actionExecutionContext();
    action.execute(context);
    String content = new String(FileSystemUtils.readContentAsLatin1(outputArtifact.getPath()));
    assertThat(content.trim())
        .isEqualTo(
            "--flag1\n"
                + "artifact/myTreeFileArtifact/artifacts/treeFileArtifact1:"
                + "artifact/myTreeFileArtifact/artifacts/treeFileArtifact2");
  }

  private Artifact createTreeArtifact(String rootRelativePath) {
    PathFragment relpath = PathFragment.create(rootRelativePath);
    return new SpecialArtifact(
        rootDir.getPath().getRelative(relpath),
        rootDir,
        rootDir.getExecPath().getRelative(relpath),
        ArtifactOwner.NULL_OWNER,
        SpecialArtifactType.TREE);
  }

  private TreeFileArtifact createTreeFileArtifact(
      Artifact inputTreeArtifact, String parentRelativePath) {
    return ActionInputHelper.treeFileArtifact(
        inputTreeArtifact,
        PathFragment.create(parentRelativePath));
  }

  private ParameterFileWriteAction createParameterFileWriteAction(
      Iterable<Artifact> inputTreeArtifacts, CommandLine commandLine) {
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
        .add("--flag3")
        .add("value1")
        .add("value2")
        .build();
  }

  private CommandLine createTreeArtifactExpansionCommandLine() {
    return CustomCommandLine.builder()
        .add("--flag1")
        .addJoinExpandedTreeArtifactExecPath(":", treeArtifact)
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

    Executor executor = new TestExecutorBuilder(directories, binTools).build();
    return new ActionExecutionContext(executor, null, null, new FileOutErr(),
        ImmutableMap.<String, String>of(), artifactExpander);
  }
}
