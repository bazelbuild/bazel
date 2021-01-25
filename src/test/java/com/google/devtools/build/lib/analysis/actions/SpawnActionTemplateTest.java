// Copyright 2016 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.actions.SpawnActionTemplate.OutputPathMapper;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link SpawnActionTemplate}.
 */
@RunWith(JUnit4.class)
public class SpawnActionTemplateTest {
  private static final OutputPathMapper IDENTITY_MAPPER = new OutputPathMapper() {
    @Override
    public PathFragment parentRelativeOutputPath(TreeFileArtifact inputTreeFileArtifact) {
      return inputTreeFileArtifact.getParentRelativePath();
    }
  };

  private ArtifactRoot root;

  @Before
  public void setRootDir() throws Exception  {
    Scratch scratch = new Scratch();
    Path execRoot = scratch.getFileSystem().getPath("/");
    root = ArtifactRoot.asDerivedRoot(execRoot, false, false, false, "root");
  }

  @Test
  public void testInputAndOutputTreeArtifacts() {
    SpawnActionTemplate actionTemplate = createSimpleSpawnActionTemplate();
    assertThat(actionTemplate.getInputs().toList()).containsExactly(createInputTreeArtifact());
    assertThat(actionTemplate.getOutputs()).containsExactly(createOutputTreeArtifact());
  }

  @Test
  public void testCommonToolsAndInputs() {
    SpecialArtifact inputTreeArtifact = createInputTreeArtifact();
    SpecialArtifact outputTreeArtifact = createOutputTreeArtifact();
    Artifact commonInput = createDerivedArtifact("common/input");
    Artifact commonTool = createDerivedArtifact("common/tool");
    Artifact executable = createDerivedArtifact("bin/cp");


    SpawnActionTemplate actionTemplate = builder(inputTreeArtifact, outputTreeArtifact)
        .setExecutionInfo(ImmutableMap.<String, String>of("local", ""))
        .setExecutable(executable)
        .setCommandLineTemplate(
            createSimpleCommandLineTemplate(inputTreeArtifact, outputTreeArtifact))
        .setOutputPathMapper(IDENTITY_MAPPER)
        .setMnemonics("ActionTemplate", "ExpandedAction")
        .addCommonTools(ImmutableList.of(commonTool))
        .addCommonInputs(ImmutableList.of(commonInput))
        .build(ActionsTestUtil.NULL_ACTION_OWNER);

    assertThat(actionTemplate.getTools().toList()).containsAtLeast(commonTool, executable);
    assertThat(actionTemplate.getInputs().toList())
        .containsAtLeast(commonInput, commonTool, executable);
  }

  @Test
  public void testBuilder_outputPathMapperRequired() {
    SpecialArtifact inputTreeArtifact = createInputTreeArtifact();
    SpecialArtifact outputTreeArtifact = createOutputTreeArtifact();
    SpawnActionTemplate.Builder builder = builder(inputTreeArtifact, outputTreeArtifact)
        .setExecutionInfo(ImmutableMap.<String, String>of("local", ""))
        .setExecutable(PathFragment.create("/bin/cp"))
        .setCommandLineTemplate(
            createSimpleCommandLineTemplate(inputTreeArtifact, outputTreeArtifact))
        .setMnemonics("ActionTemplate", "ExpandedAction");

    assertThrows(
        NullPointerException.class, () -> builder.build(ActionsTestUtil.NULL_ACTION_OWNER));
  }

  @Test
  public void testBuilder_executableRequired() {
    SpecialArtifact inputTreeArtifact = createInputTreeArtifact();
    SpecialArtifact outputTreeArtifact = createOutputTreeArtifact();
    SpawnActionTemplate.Builder builder = builder(inputTreeArtifact, outputTreeArtifact)
        .setExecutionInfo(ImmutableMap.<String, String>of("local", ""))
        .setOutputPathMapper(IDENTITY_MAPPER)
        .setCommandLineTemplate(
            createSimpleCommandLineTemplate(inputTreeArtifact, outputTreeArtifact))
        .setMnemonics("ActionTemplate", "ExpandedAction");

    assertThrows(
        NullPointerException.class, () -> builder.build(ActionsTestUtil.NULL_ACTION_OWNER));
  }

  @Test
  public void testBuilder_commandlineTemplateRequired() {
    SpecialArtifact inputTreeArtifact = createInputTreeArtifact();
    SpecialArtifact outputTreeArtifact = createOutputTreeArtifact();
    SpawnActionTemplate.Builder builder = builder(inputTreeArtifact, outputTreeArtifact)
        .setExecutionInfo(ImmutableMap.<String, String>of("local", ""))
        .setOutputPathMapper(IDENTITY_MAPPER)
        .setExecutable(PathFragment.create("/bin/cp"))
        .setMnemonics("ActionTemplate", "ExpandedAction");

    assertThrows(
        NullPointerException.class, () -> builder.build(ActionsTestUtil.NULL_ACTION_OWNER));
  }

  @Test
  public void getKey_same() throws Exception {
    ActionKeyContext keyContext = new ActionKeyContext();
    SpecialArtifact inputTreeArtifact = createInputTreeArtifact();
    SpecialArtifact outputTreeArtifact = createOutputTreeArtifact();
    Artifact executable = createDerivedArtifact("bin/cp");

    // Use two different builders because the same builder would share the underlying
    // SpawnActionBuilder.
    SpawnActionTemplate actionTemplate =
        builder(inputTreeArtifact, outputTreeArtifact)
            .setExecutable(executable)
            .setCommandLineTemplate(
                createSimpleCommandLineTemplate(inputTreeArtifact, outputTreeArtifact))
            .setOutputPathMapper(IDENTITY_MAPPER)
            .setMnemonics("ActionTemplate", "ExpandedAction")
            .build(ActionsTestUtil.NULL_ACTION_OWNER);
    SpawnActionTemplate actionTemplate2 =
        builder(inputTreeArtifact, outputTreeArtifact)
            .setExecutable(executable)
            .setCommandLineTemplate(
                createSimpleCommandLineTemplate(inputTreeArtifact, outputTreeArtifact))
            .setOutputPathMapper(IDENTITY_MAPPER)
            .setMnemonics("ActionTemplate", "ExpandedAction")
            .build(ActionsTestUtil.NULL_ACTION_OWNER);
    assertThat(actionTemplate2.getKey(keyContext, /*artifactExpander=*/ null))
        .isEqualTo(actionTemplate.getKey(keyContext, /*artifactExpander=*/ null));
  }

  @Test
  public void getKey_differs() throws Exception {
    ActionKeyContext keyContext = new ActionKeyContext();
    SpecialArtifact inputTreeArtifact = createInputTreeArtifact();
    SpecialArtifact outputTreeArtifact = createOutputTreeArtifact();
    Artifact executable = createDerivedArtifact("bin/cp");

    // Use two different builders because the same builder would share the underlying
    // SpawnActionBuilder.
    SpawnActionTemplate actionTemplate =
        builder(inputTreeArtifact, outputTreeArtifact)
            .setExecutable(executable)
            .setCommandLineTemplate(
                createSimpleCommandLineTemplate(inputTreeArtifact, outputTreeArtifact))
            .setOutputPathMapper(IDENTITY_MAPPER)
            .setMnemonics("ActionTemplate", "ExpandedAction")
            .build(ActionsTestUtil.NULL_ACTION_OWNER);
    SpawnActionTemplate actionTemplate2 =
        builder(inputTreeArtifact, outputTreeArtifact)
            .setExecutable(executable)
            .setCommandLineTemplate(
                createSimpleCommandLineTemplate(inputTreeArtifact, outputTreeArtifact))
            .setOutputPathMapper(IDENTITY_MAPPER)
            .setMnemonics("ActionTemplate", "ExpandedAction2")
            .build(ActionsTestUtil.NULL_ACTION_OWNER);
    assertThat(actionTemplate2.getKey(keyContext, /*artifactExpander=*/ null))
        .isNotEqualTo(actionTemplate.getKey(keyContext, /*artifactExpander=*/ null));
  }

  @Test
  public void testExpandedAction_inputAndOutputTreeFileArtifacts() {
    SpawnActionTemplate actionTemplate = createSimpleSpawnActionTemplate();
    SpecialArtifact inputTreeArtifact = createInputTreeArtifact();
    SpecialArtifact outputTreeArtifact = createOutputTreeArtifact();

    ImmutableSet<TreeFileArtifact> inputTreeFileArtifacts =
        createInputTreeFileArtifacts(inputTreeArtifact);

    List<SpawnAction> expandedActions =
        actionTemplate.generateActionsForInputArtifacts(
            inputTreeFileArtifacts, ActionsTestUtil.NULL_TEMPLATE_EXPANSION_ARTIFACT_OWNER);

    assertThat(expandedActions).hasSize(3);

    for (int i = 0; i < expandedActions.size(); ++i) {
      String baseName = "child" + i;
      assertThat(expandedActions.get(i).getInputs().toList())
          .containsExactly(
              TreeFileArtifact.createTreeOutput(inputTreeArtifact, "children/" + baseName));
      assertThat(expandedActions.get(i).getOutputs())
          .containsExactly(
              TreeFileArtifact.createTemplateExpansionOutput(
                  outputTreeArtifact,
                  "children/" + baseName,
                  ActionsTestUtil.NULL_TEMPLATE_EXPANSION_ARTIFACT_OWNER));
    }
  }

  @Test
  public void testExpandedAction_commonToolsAndInputs() {
    SpecialArtifact inputTreeArtifact = createInputTreeArtifact();
    SpecialArtifact outputTreeArtifact = createOutputTreeArtifact();
    Artifact commonInput = createDerivedArtifact("common/input");
    Artifact commonTool = createDerivedArtifact("common/tool");
    Artifact executable = createDerivedArtifact("bin/cp");

    SpawnActionTemplate actionTemplate =
        builder(inputTreeArtifact, outputTreeArtifact)
            .setExecutionInfo(ImmutableMap.of("local", ""))
            .setExecutable(executable)
            .setCommandLineTemplate(
                createSimpleCommandLineTemplate(inputTreeArtifact, outputTreeArtifact))
            .setOutputPathMapper(IDENTITY_MAPPER)
            .setMnemonics("ActionTemplate", "ExpandedAction")
            .addCommonTools(ImmutableList.of(commonTool))
            .addCommonInputs(ImmutableList.of(commonInput))
            .build(ActionsTestUtil.NULL_ACTION_OWNER);

    ImmutableSet<TreeFileArtifact> inputTreeFileArtifacts =
        createInputTreeFileArtifacts(inputTreeArtifact);
    List<SpawnAction> expandedActions =
        actionTemplate.generateActionsForInputArtifacts(
            inputTreeFileArtifacts, ActionsTestUtil.NULL_TEMPLATE_EXPANSION_ARTIFACT_OWNER);

    for (int i = 0; i < expandedActions.size(); ++i) {
      assertThat(expandedActions.get(i).getInputs().toList())
          .containsAtLeast(commonInput, commonTool, executable);
      assertThat(expandedActions.get(i).getTools().toList())
          .containsAtLeast(commonTool, executable);
    }
  }

  @Test
  public void testExpandedAction_arguments() throws Exception {
    SpawnActionTemplate actionTemplate = createSimpleSpawnActionTemplate();
    SpecialArtifact inputTreeArtifact = createInputTreeArtifact();
    SpecialArtifact outputTreeArtifact = createOutputTreeArtifact();

    ImmutableSet<TreeFileArtifact> inputTreeFileArtifacts =
        createInputTreeFileArtifacts(inputTreeArtifact);

    List<SpawnAction> expandedActions =
        actionTemplate.generateActionsForInputArtifacts(
            inputTreeFileArtifacts, ActionsTestUtil.NULL_TEMPLATE_EXPANSION_ARTIFACT_OWNER);

    assertThat(expandedActions).hasSize(3);

    for (int i = 0; i < expandedActions.size(); ++i) {
      String baseName = String.format("child%d", i);
      assertThat(expandedActions.get(i).getArguments())
          .containsExactly(
              "/bin/cp",
              inputTreeArtifact.getExecPathString() + "/children/" + baseName,
              outputTreeArtifact.getExecPathString() + "/children/" + baseName)
          .inOrder();
    }
  }

  @Test
  public void testExpandedAction_executionInfoAndEnvironment() {
    SpawnActionTemplate actionTemplate = createSimpleSpawnActionTemplate();
    SpecialArtifact inputTreeArtifact = createInputTreeArtifact();
    ImmutableSet<TreeFileArtifact> inputTreeFileArtifacts =
        createInputTreeFileArtifacts(inputTreeArtifact);

    List<SpawnAction> expandedActions =
        actionTemplate.generateActionsForInputArtifacts(
            inputTreeFileArtifacts, ActionsTestUtil.NULL_TEMPLATE_EXPANSION_ARTIFACT_OWNER);

    assertThat(expandedActions).hasSize(3);

    for (int i = 0; i < expandedActions.size(); ++i) {
      assertThat(expandedActions.get(i).getIncompleteEnvironmentForTesting())
          .containsExactly("env", "value");
      assertThat(expandedActions.get(i).getExecutionInfo()).containsExactly("local", "");
    }
  }

  @Test
  public void testExpandedAction_illegalOutputPath() throws Exception {
    SpecialArtifact inputTreeArtifact = createInputTreeArtifact();
    SpecialArtifact outputTreeArtifact = createOutputTreeArtifact();
    ImmutableSet<TreeFileArtifact> inputTreeFileArtifacts =
        createInputTreeFileArtifacts(inputTreeArtifact);

    SpawnActionTemplate.Builder builder = builder(inputTreeArtifact, outputTreeArtifact)
        .setExecutable(PathFragment.create("/bin/cp"))
        .setCommandLineTemplate(
            createSimpleCommandLineTemplate(inputTreeArtifact, outputTreeArtifact));

    OutputPathMapper mapper = new OutputPathMapper() {
      @Override
      public PathFragment parentRelativeOutputPath(TreeFileArtifact inputTreeFileArtifact) {
        return PathFragment.create("//absolute/" + inputTreeFileArtifact.getParentRelativePath());
      }
    };

    SpawnActionTemplate actionTemplate =
        builder.setOutputPathMapper(mapper).build(ActionsTestUtil.NULL_ACTION_OWNER);

    assertThrows(
        "Absolute output paths not allowed, expected IllegalArgumentException",
        IllegalArgumentException.class,
        () ->
            actionTemplate.generateActionsForInputArtifacts(
                inputTreeFileArtifacts, ActionsTestUtil.NULL_TEMPLATE_EXPANSION_ARTIFACT_OWNER));

    mapper = new OutputPathMapper() {
      @Override
      public PathFragment parentRelativeOutputPath(TreeFileArtifact inputTreeFileArtifact) {
        return PathFragment.create("../" + inputTreeFileArtifact.getParentRelativePath());
      }
    };

    SpawnActionTemplate actionTemplate2 =
        builder.setOutputPathMapper(mapper).build(ActionsTestUtil.NULL_ACTION_OWNER);

    assertThrows(
        "Output paths containing '..' not allowed, expected IllegalArgumentException",
        IllegalArgumentException.class,
        () ->
            actionTemplate2.generateActionsForInputArtifacts(
                inputTreeFileArtifacts, ActionsTestUtil.NULL_TEMPLATE_EXPANSION_ARTIFACT_OWNER));
  }

  private SpawnActionTemplate.Builder builder(
      SpecialArtifact inputTreeArtifact, SpecialArtifact outputTreeArtifact) {
    return new SpawnActionTemplate.Builder(inputTreeArtifact, outputTreeArtifact);
  }

  private SpawnActionTemplate createSimpleSpawnActionTemplate() {
    SpecialArtifact inputTreeArtifact = createInputTreeArtifact();
    SpecialArtifact outputTreeArtifact = createOutputTreeArtifact();

    return builder(inputTreeArtifact, outputTreeArtifact)
        .setExecutionInfo(ImmutableMap.of("local", ""))
        .setEnvironment(ImmutableMap.of("env", "value"))
        .setExecutable(PathFragment.create("/bin/cp"))
        .setCommandLineTemplate(
            createSimpleCommandLineTemplate(inputTreeArtifact, outputTreeArtifact))
        .setOutputPathMapper(IDENTITY_MAPPER)
        .setMnemonics("ActionTemplate", "ExpandedAction")
        .build(ActionsTestUtil.NULL_ACTION_OWNER);
  }

  private SpecialArtifact createInputTreeArtifact() {
    return createTreeArtifact("my/inputTree");
  }

  private SpecialArtifact createOutputTreeArtifact() {
    return createTreeArtifact("my/outputTree");
  }

  private SpecialArtifact createTreeArtifact(String rootRelativePath) {
    PathFragment relpath = PathFragment.create(rootRelativePath);
    SpecialArtifact result =
        new SpecialArtifact(
            root,
            root.getExecPath().getRelative(relpath),
            ActionsTestUtil.NULL_ARTIFACT_OWNER,
            SpecialArtifactType.TREE);
    result.setGeneratingActionKey(ActionsTestUtil.NULL_ACTION_LOOKUP_DATA);
    return result;
  }

  private Artifact createDerivedArtifact(String rootRelativePath) {
    return ActionsTestUtil.createArtifact(root, rootRelativePath);
  }

  private CustomCommandLine createSimpleCommandLineTemplate(
      Artifact inputTreeArtifact, Artifact outputTreeArtifact) {
    return CustomCommandLine.builder()
        .addPlaceholderTreeArtifactExecPath(inputTreeArtifact)
        .addPlaceholderTreeArtifactExecPath(outputTreeArtifact)
        .build();
  }

  private static ImmutableSet<TreeFileArtifact> createInputTreeFileArtifacts(
      SpecialArtifact inputTreeArtifact) {
    return ImmutableSet.of(
        TreeFileArtifact.createTreeOutput(inputTreeArtifact, "children/child0"),
        TreeFileArtifact.createTreeOutput(inputTreeArtifact, "children/child1"),
        TreeFileArtifact.createTreeOutput(inputTreeArtifact, "children/child2"));
  }
}
