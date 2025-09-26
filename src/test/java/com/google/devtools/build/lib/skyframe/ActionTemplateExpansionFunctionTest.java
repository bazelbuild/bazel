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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionConflictException;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionTemplate;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.BasicActionLookupValue;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.InjectedActionLookupKey;
import com.google.devtools.build.lib.actions.util.TestAction.DummyAction;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnActionTemplate;
import com.google.devtools.build.lib.analysis.actions.SpawnActionTemplate.OutputPathMapper;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.ActionTemplateExpansionValue.ActionTemplateExpansionKey;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.Differencer.DiffWithDelta.Delta;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.SequencedRecordingDifferencer;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ActionTemplateExpansionFunction}. */
@RunWith(JUnit4.class)
public final class ActionTemplateExpansionFunctionTest extends FoundationTestCase  {

  private final Map<Artifact, TreeArtifactValue> artifactValueMap = new LinkedHashMap<>();
  private final SequencedRecordingDifferencer differencer = new SequencedRecordingDifferencer();
  private final MemoizingEvaluator evaluator =
      new InMemoryMemoizingEvaluator(
          ImmutableMap.of(
              Artifact.ARTIFACT,
              new DummyArtifactFunction(artifactValueMap),
              SkyFunctions.ACTION_TEMPLATE_EXPANSION,
              new ActionTemplateExpansionFunction(new ActionKeyContext())),
          differencer);

  @Before
  public void setUp() {
    PrecomputedValue.BUILD_ID.set(differencer, UUID.randomUUID());
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(
        differencer,
        new PathPackageLocator(
            rootDirectory.getFileSystem().getPath("/outputbase"),
            ImmutableList.of(Root.fromPath(rootDirectory)),
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY));
  }

  @Test
  public void testActionTemplateExpansionFunction() throws Exception {
    SpecialArtifact inputTreeArtifact =
        createAndPopulateTreeArtifact("inputTreeArtifact", "child0", "child1", "child2");
    SpecialArtifact outputTreeArtifact = createTreeArtifact("outputTreeArtifact");

    SpawnActionTemplate spawnActionTemplate = ActionsTestUtil.createDummySpawnActionTemplate(
        inputTreeArtifact, outputTreeArtifact);
    List<Action> actions = evaluate(spawnActionTemplate);
    assertThat(actions).hasSize(3);

    ArtifactOwner owner = ActionTemplateExpansionValue.key(CTKEY, 0);
    int i = 0;
    for (Action action : actions) {
      String childName = "child" + i;
      assertThat(Artifact.asExecPaths(action.getInputs()))
          .contains("out/inputTreeArtifact/" + childName);
      assertThat(Artifact.asExecPaths(action.getOutputs()))
          .containsExactly("out/outputTreeArtifact/" + childName);
      assertThat(Iterables.getOnlyElement(action.getOutputs()).getArtifactOwner()).isEqualTo(owner);
      ++i;
    }
  }

  @Test
  public void testThrowsOnActionConflict() throws Exception {
    SpecialArtifact inputTreeArtifact =
        createAndPopulateTreeArtifact("inputTreeArtifact", "child0", "child1", "child2");
    SpecialArtifact outputTreeArtifact = createTreeArtifact("outputTreeArtifact");

    OutputPathMapper mapper = artifact -> PathFragment.create("conflict_path");
    SpawnActionTemplate spawnActionTemplate =
        new SpawnActionTemplate.Builder(inputTreeArtifact, outputTreeArtifact)
            .setExecutable(PathFragment.create("/bin/cp"))
            .setCommandLineTemplate(CustomCommandLine.builder().build())
            .setOutputPathMapper(mapper)
            .build(ActionsTestUtil.NULL_ACTION_OWNER);

    ActionConflictException e =
        assertThrows(ActionConflictException.class, () -> evaluate(spawnActionTemplate));
    assertThat(e)
        .hasMessageThat()
        .contains(
            "file 'outputTreeArtifact/conflict_path' is generated by these conflicting actions");
  }

  @Test
  public void testThrowsOnArtifactPrefixConflict() throws Exception {
    SpecialArtifact inputTreeArtifact =
        createAndPopulateTreeArtifact("inputTreeArtifact", "child0", "child1", "child2");
    SpecialArtifact outputTreeArtifact = createTreeArtifact("outputTreeArtifact");

    OutputPathMapper mapper =
        new OutputPathMapper() {
          private int i = 0;

          @Override
          public PathFragment parentRelativeOutputPath(TreeFileArtifact inputTreeFileArtifact) {
            PathFragment path =
                switch (i) {
                  case 0 -> PathFragment.create("path_prefix");
                  case 1 -> PathFragment.create("path_prefix/conflict");
                  default -> inputTreeFileArtifact.getParentRelativePath();
                };
            ++i;
            return path;
          }
        };
    SpawnActionTemplate spawnActionTemplate =
        new SpawnActionTemplate.Builder(inputTreeArtifact, outputTreeArtifact)
            .setExecutable(PathFragment.create("/bin/cp"))
            .setCommandLineTemplate(CustomCommandLine.builder().build())
            .setOutputPathMapper(mapper)
            .build(ActionsTestUtil.NULL_ACTION_OWNER);

    ActionConflictException e =
        assertThrows(ActionConflictException.class, () -> evaluate(spawnActionTemplate));
    assertThat(e).hasMessageThat().contains("is a prefix of the other");
  }

  @Test
  public void cannotDeclareNonTreeOutput() throws Exception {
    SpecialArtifact inputTree = createAndPopulateTreeArtifact("input", "child");
    SpecialArtifact outputTree = createTreeArtifact("output");

    ActionTemplate<DummyAction> template =
        new TestActionTemplate(inputTree, outputTree) {
          @Override
          public ImmutableList<DummyAction> generateActionsForInputArtifacts(
              ImmutableList<TreeFileArtifact> inputTreeFileArtifacts,
              ActionLookupKey artifactOwner,
              EventHandler eventHandler) {
            return ImmutableList.of();
          }

          @Override
          public ImmutableSet<Artifact> getOutputs() {
            return ImmutableSet.of(
                outputTree,
                DerivedArtifact.create(
                    outputTree.getRoot(),
                    outputTree.getRoot().getExecPath().getRelative("not_tree"),
                    outputTree.getArtifactOwner()));
          }
        };

    Exception e = assertThrows(RuntimeException.class, () -> evaluate(template));
    assertThat(e).hasCauseThat().isInstanceOf(IllegalStateException.class);
    assertThat(e)
        .hasCauseThat()
        .hasMessageThat()
        .contains(template + " declares an output which is not a tree artifact");
  }

  @Test
  public void cannotGenerateOutputWithWrongOwner() throws Exception {
    SpecialArtifact inputTree = createAndPopulateTreeArtifact("input", "child");
    SpecialArtifact outputTree = createTreeArtifact("output");

    ActionTemplate<DummyAction> template =
        new TestActionTemplate(inputTree, outputTree) {
          @Override
          public ImmutableList<DummyAction> generateActionsForInputArtifacts(
              ImmutableList<TreeFileArtifact> inputTreeFileArtifacts,
              ActionLookupKey artifactOwner,
              EventHandler eventHandler) {
            TreeFileArtifact input = Iterables.getOnlyElement(inputTreeFileArtifacts);
            TreeFileArtifact outputWithWrongOwner =
                TreeFileArtifact.createTemplateExpansionOutput(
                    outputTree, "child", ActionsTestUtil.NULL_TEMPLATE_EXPANSION_ARTIFACT_OWNER);
            assertThat(outputWithWrongOwner.getArtifactOwner()).isNotEqualTo(artifactOwner);
            return ImmutableList.of(new DummyAction(input, outputWithWrongOwner));
          }
        };

    Exception e = assertThrows(RuntimeException.class, () -> evaluate(template));
    assertThat(e).hasCauseThat().isInstanceOf(IllegalStateException.class);
    assertThat(e)
        .hasCauseThat()
        .hasMessageThat()
        .contains(template + " generated an action with an output owned by the wrong owner");
  }

  @Test
  public void cannotGenerateNonTreeFileArtifactOutput() throws Exception {
    SpecialArtifact inputTree = createAndPopulateTreeArtifact("input", "child");
    SpecialArtifact outputTree = createTreeArtifact("output");

    ActionTemplate<DummyAction> template =
        new TestActionTemplate(inputTree, outputTree) {
          @Override
          public ImmutableList<DummyAction> generateActionsForInputArtifacts(
              ImmutableList<TreeFileArtifact> inputTreeFileArtifacts,
              ActionLookupKey artifactOwner,
              EventHandler eventHandler) {
            TreeFileArtifact input = Iterables.getOnlyElement(inputTreeFileArtifacts);
            Artifact notTreeFileArtifact =
                DerivedArtifact.create(
                    input.getRoot(),
                    input.getRoot().getExecPath().getRelative("a.txt"),
                    artifactOwner);
            assertThat(notTreeFileArtifact.isTreeArtifact()).isFalse();
            return ImmutableList.of(new DummyAction(input, notTreeFileArtifact));
          }
        };

    Exception e = assertThrows(RuntimeException.class, () -> evaluate(template));
    assertThat(e).hasCauseThat().isInstanceOf(IllegalStateException.class);
    assertThat(e)
        .hasCauseThat()
        .hasMessageThat()
        .contains(template + " generated an action which outputs a non-TreeFileArtifact");
  }

  @Test
  public void cannotGenerateOutputUnderUndeclaredTree() throws Exception {
    SpecialArtifact inputTree = createAndPopulateTreeArtifact("input", "child");
    SpecialArtifact outputTree = createTreeArtifact("output");

    ActionTemplate<DummyAction> template =
        new TestActionTemplate(inputTree, outputTree) {
          @Override
          public ImmutableList<DummyAction> generateActionsForInputArtifacts(
              ImmutableList<TreeFileArtifact> inputTreeFileArtifacts,
              ActionLookupKey artifactOwner,
              EventHandler eventHandler) {
            TreeFileArtifact input = Iterables.getOnlyElement(inputTreeFileArtifacts);
            TreeFileArtifact outputUnderWrongTree =
                TreeFileArtifact.createTemplateExpansionOutput(
                    createTreeArtifact("undeclared"), "child", artifactOwner);
            return ImmutableList.of(new DummyAction(input, outputUnderWrongTree));
          }
        };

    Exception e = assertThrows(RuntimeException.class, () -> evaluate(template));
    assertThat(e).hasCauseThat().isInstanceOf(IllegalStateException.class);
    assertThat(e)
        .hasCauseThat()
        .hasMessageThat()
        .contains(
            template
                + " generated an action with an output File:[[<execution_root>]out]undeclared/child"
                + " under an undeclared tree not in [File:[[<execution_root>]out]output]");
  }

  @Test
  public void canGenerateOutputUnderAdditionalDeclaredTree() throws Exception {
    SpecialArtifact inputTree = createAndPopulateTreeArtifact("input", "child");
    SpecialArtifact outputTree = createTreeArtifact("output");
    SpecialArtifact additionalOutputTree = createTreeArtifact("additional_output");

    ActionTemplate<DummyAction> template =
        new TestActionTemplate(inputTree, outputTree) {
          @Override
          public ImmutableList<DummyAction> generateActionsForInputArtifacts(
              ImmutableList<TreeFileArtifact> inputTreeFileArtifacts,
              ActionLookupKey artifactOwner,
              EventHandler eventHandler) {
            TreeFileArtifact input = Iterables.getOnlyElement(inputTreeFileArtifacts);
            return ImmutableList.of(
                new DummyAction(
                    input,
                    TreeFileArtifact.createTemplateExpansionOutput(
                        outputTree, "child", artifactOwner)),
                new DummyAction(
                    input,
                    TreeFileArtifact.createTemplateExpansionOutput(
                        additionalOutputTree, "additional_child", artifactOwner)));
          }

          @Override
          public ImmutableSet<Artifact> getOutputs() {
            return ImmutableSet.of(outputTree, additionalOutputTree);
          }
        };

    evaluate(template);
  }

  @Test
  public void canUseMultipleInputTrees() throws Exception {
    SpecialArtifact inputTree1 = createAndPopulateTreeArtifact("input1", "child1", "child2");
    SpecialArtifact inputTree2 = createAndPopulateTreeArtifact("input2", "child1", "child2");
    SpecialArtifact outputTree = createTreeArtifact("output");

    ActionTemplate<DummyAction> template =
        new TestActionTemplate(ImmutableList.of(inputTree1, inputTree2), outputTree) {
          @Override
          public ImmutableList<DummyAction> generateActionsForInputArtifacts(
              ImmutableList<TreeFileArtifact> inputTreeFileArtifacts,
              ActionLookupKey artifactOwner,
              EventHandler eventHandler) {
            ImmutableList.Builder<DummyAction> actions = ImmutableList.builder();
            ImmutableListMultimap<SpecialArtifact, TreeFileArtifact> inputTreeArtifactsToChildren =
                ActionTemplate.getInputTreeArtifactsToChildren(inputTreeFileArtifacts);
            int i = 0;
            for (SpecialArtifact inputTreeArtifact : getInputTreeArtifacts()) {
              actions.add(
                  new DummyAction(
                      NestedSetBuilder.<Artifact>wrap(
                          Order.STABLE_ORDER, inputTreeArtifactsToChildren.get(inputTreeArtifact)),
                      TreeFileArtifact.createTemplateExpansionOutput(
                          outputTree, "child-" + i++, artifactOwner)));
            }
            return actions.build();
          }
        };
    evaluate(template);
  }

  private static final ActionLookupKey CTKEY = new InjectedActionLookupKey("key");

  private ImmutableList<Action> evaluate(ActionTemplate<?> actionTemplate) throws Exception {
    ActionLookupValue ctValue = createActionLookupValue(actionTemplate);

    differencer.inject(CTKEY, Delta.justNew(ctValue));
    ActionTemplateExpansionKey templateKey = ActionTemplateExpansionValue.key(CTKEY, 0);
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setParallelism(SkyframeExecutor.DEFAULT_THREAD_COUNT)
            .setEventHandler(NullEventHandler.INSTANCE)
            .build();
    EvaluationResult<ActionTemplateExpansionValue> result =
        evaluator.evaluate(ImmutableList.of(templateKey), evaluationContext);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    ActionTemplateExpansionValue actionTemplateExpansionValue = result.get(templateKey);
    ImmutableList.Builder<Action> actionList = ImmutableList.builder();
    for (int i = 0; i < actionTemplateExpansionValue.getActions().size(); i++) {
      actionList.add(actionTemplateExpansionValue.getAction(i));
    }
    return actionList.build();
  }

  private static ActionLookupValue createActionLookupValue(ActionTemplate<?> actionTemplate)
      throws ActionConflictException,
          InterruptedException,
          Actions.ArtifactGeneratedByOtherRuleException {
    ImmutableList<ActionAnalysisMetadata> actions = ImmutableList.of(actionTemplate);
    Actions.assignOwnersAndThrowIfConflict(new ActionKeyContext(), actions, CTKEY);
    return new BasicActionLookupValue(actions);
  }

  private SpecialArtifact createTreeArtifact(String path) {
    PathFragment execPath = PathFragment.create("out").getRelative(path);
    return SpecialArtifact.create(
        ArtifactRoot.asDerivedRoot(rootDirectory, RootType.OUTPUT, "out"),
        execPath,
        CTKEY,
        SpecialArtifactType.TREE);
  }

  private SpecialArtifact createAndPopulateTreeArtifact(String path, String... childRelativePaths)
      throws Exception {
    SpecialArtifact treeArtifact = createTreeArtifact(path);
    treeArtifact.setGeneratingActionKey(ActionLookupData.create(CTKEY, /*actionIndex=*/ 0));
    TreeArtifactValue.Builder tree = TreeArtifactValue.newBuilder(treeArtifact);

    for (String childRelativePath : childRelativePaths) {
      TreeFileArtifact treeFileArtifact =
          TreeFileArtifact.createTreeOutput(treeArtifact, childRelativePath);
      scratch.file(treeFileArtifact.getPath().toString(), childRelativePath);
      // We do not care about the FileArtifactValues in this test.
      tree.putChild(treeFileArtifact, FileArtifactValue.createForTesting(treeFileArtifact));
    }

    artifactValueMap.put(treeArtifact, tree.build());
    return treeArtifact;
  }

  /** Dummy ArtifactFunction that just returns injected values */
  private static final class DummyArtifactFunction implements SkyFunction {
    private final Map<Artifact, TreeArtifactValue> artifactValueMap;

    DummyArtifactFunction(Map<Artifact, TreeArtifactValue> artifactValueMap) {
      this.artifactValueMap = artifactValueMap;
    }
    @Override
    public SkyValue compute(SkyKey skyKey, Environment env) {
      return Preconditions.checkNotNull(artifactValueMap.get(skyKey));
    }
  }

  private abstract static class TestActionTemplate implements ActionTemplate<DummyAction> {
    private final ImmutableList<SpecialArtifact> inputTreeArtifacts;
    private final SpecialArtifact outputTreeArtifact;

    TestActionTemplate(SpecialArtifact inputTreeArtifact, SpecialArtifact outputTreeArtifact) {
      this(ImmutableList.of(inputTreeArtifact), outputTreeArtifact);
    }

    TestActionTemplate(
        ImmutableList<SpecialArtifact> inputTreeArtifacts, SpecialArtifact outputTreeArtifact) {
      for (SpecialArtifact inputTreeArtifact : inputTreeArtifacts) {
        Preconditions.checkArgument(inputTreeArtifact.isTreeArtifact(), inputTreeArtifact);
      }
      Preconditions.checkArgument(outputTreeArtifact.isTreeArtifact(), outputTreeArtifact);
      this.inputTreeArtifacts = inputTreeArtifacts;
      this.outputTreeArtifact = outputTreeArtifact;
    }

    @Override
    public ImmutableList<SpecialArtifact> getInputTreeArtifacts() {
      return inputTreeArtifacts;
    }

    @Override
    public ImmutableSet<Artifact> getOutputs() {
      return ImmutableSet.of(outputTreeArtifact);
    }

    @Override
    public ActionOwner getOwner() {
      return ActionsTestUtil.NULL_ACTION_OWNER;
    }

    @Override
    public boolean isShareable() {
      return false;
    }

    @Override
    public String getMnemonic() {
      return "TestActionTemplate";
    }

    @Override
    public String getKey(
        ActionKeyContext actionKeyContext, @Nullable InputMetadataProvider inputMetadataProvider) {
      Fingerprint fp = new Fingerprint();
      for (SpecialArtifact inputTreeArtifact : inputTreeArtifacts) {
        fp.addPath(inputTreeArtifact.getPath());
      }
      fp.addPath(outputTreeArtifact.getPath());
      return fp.hexDigestAndReset();
    }

    @Override
    public String prettyPrint() {
      return "TestActionTemplate for " + outputTreeArtifact;
    }

    @Override
    public String describe() {
      return prettyPrint();
    }

    @Override
    public NestedSet<Artifact> getTools() {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }

    @Override
    public NestedSet<Artifact> getInputs() {
      return NestedSetBuilder.wrap(Order.STABLE_ORDER, inputTreeArtifacts);
    }

    @Override
    public NestedSet<Artifact> getOriginalInputs() {
      return getInputs();
    }

    @Override
    public NestedSet<Artifact> getSchedulingDependencies() {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }

    @Override
    public Collection<String> getClientEnvironmentVariables() {
      return ImmutableList.of();
    }

    @Override
    public NestedSet<Artifact> getInputFilesForExtraAction(
        ActionExecutionContext actionExecutionContext) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }

    @Override
    public ImmutableSet<Artifact> getMandatoryOutputs() {
      return ImmutableSet.of();
    }

    @Override
    public NestedSet<Artifact> getMandatoryInputs() {
      return NestedSetBuilder.wrap(Order.STABLE_ORDER, inputTreeArtifacts);
    }

    @Override
    public String toString() {
      return prettyPrint();
    }
  }
}
