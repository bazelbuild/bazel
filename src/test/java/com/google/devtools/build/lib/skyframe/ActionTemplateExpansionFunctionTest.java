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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.ActionTemplate;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.ArtifactPrefixConflictException;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.InjectedActionLookupKey;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnActionTemplate;
import com.google.devtools.build.lib.analysis.actions.SpawnActionTemplate.OutputPathMapper;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.ActionTemplateExpansionValue.ActionTemplateExpansionKey;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.SequencedRecordingDifferencer;
import com.google.devtools.build.skyframe.SequentialBuildDriver;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Tests for {@link ActionTemplateExpansionFunction}. */
@RunWith(JUnit4.class)
public final class ActionTemplateExpansionFunctionTest extends FoundationTestCase  {
  private Map<Artifact, TreeArtifactValue> artifactValueMap;
  private SequentialBuildDriver driver;
  private SequencedRecordingDifferencer differencer;

  @Before
  public void setUp() throws Exception  {
    artifactValueMap = new LinkedHashMap<>();
    AtomicReference<PathPackageLocator> pkgLocator =
        new AtomicReference<>(
            new PathPackageLocator(
                rootDirectory.getFileSystem().getPath("/outputbase"),
                ImmutableList.of(Root.fromPath(rootDirectory)),
                BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY));
    differencer = new SequencedRecordingDifferencer();
    MemoizingEvaluator evaluator =
        new InMemoryMemoizingEvaluator(
            ImmutableMap.<SkyFunctionName, SkyFunction>builder()
                .put(Artifact.ARTIFACT, new DummyArtifactFunction(artifactValueMap))
                .put(
                    SkyFunctions.ACTION_TEMPLATE_EXPANSION,
                    new ActionTemplateExpansionFunction(new ActionKeyContext()))
                .build(),
            differencer);
    driver = new SequentialBuildDriver(evaluator);
    PrecomputedValue.BUILD_ID.set(differencer, UUID.randomUUID());
    PrecomputedValue.PATH_PACKAGE_LOCATOR.set(differencer, pkgLocator.get());
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

    OutputPathMapper mapper = new OutputPathMapper() {
      @Override
      public PathFragment parentRelativeOutputPath(TreeFileArtifact inputTreeFileArtifact) {
        return PathFragment.create("conflict_path");
      }
    };
    SpawnActionTemplate spawnActionTemplate =
        new SpawnActionTemplate.Builder(inputTreeArtifact, outputTreeArtifact)
            .setExecutable(PathFragment.create("/bin/cp"))
            .setCommandLineTemplate(CustomCommandLine.builder().build())
            .setOutputPathMapper(mapper)
            .build(ActionsTestUtil.NULL_ACTION_OWNER);

    assertThrows(ActionConflictException.class, () -> evaluate(spawnActionTemplate));
  }

  @Test
  public void testThrowsOnArtifactPrefixConflict() throws Exception {
    SpecialArtifact inputTreeArtifact =
        createAndPopulateTreeArtifact("inputTreeArtifact", "child0", "child1", "child2");
    SpecialArtifact outputTreeArtifact = createTreeArtifact("outputTreeArtifact");

    OutputPathMapper mapper = new OutputPathMapper() {
      private int i = 0;
      @Override
      public PathFragment parentRelativeOutputPath(TreeFileArtifact inputTreeFileArtifact) {
        PathFragment path;
        switch (i) {
          case 0:
            path = PathFragment.create("path_prefix");
            break;
          case 1:
            path = PathFragment.create("path_prefix/conflict");
            break;
          default:
            path = inputTreeFileArtifact.getParentRelativePath();
        }

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

    assertThrows(ArtifactPrefixConflictException.class, () -> evaluate(spawnActionTemplate));
  }

  private static final ActionLookupValue.ActionLookupKey CTKEY = new InjectedActionLookupKey("key");

  private List<Action> evaluate(SpawnActionTemplate spawnActionTemplate) throws Exception {
    ConfiguredTargetValue ctValue = createConfiguredTargetValue(spawnActionTemplate);

    differencer.inject(CTKEY, ctValue);
    ActionTemplateExpansionKey templateKey = ActionTemplateExpansionValue.key(CTKEY, 0);
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setNumThreads(SkyframeExecutor.DEFAULT_THREAD_COUNT)
            .setEventHander(NullEventHandler.INSTANCE)
            .build();
    EvaluationResult<ActionTemplateExpansionValue> result =
        driver.evaluate(ImmutableList.of(templateKey), evaluationContext);
    if (result.hasError()) {
      throw result.getError().getException();
    }
    ActionTemplateExpansionValue actionTemplateExpansionValue = result.get(templateKey);
    ImmutableList.Builder<Action> actionList = ImmutableList.builder();
    for (int i = 0; i < actionTemplateExpansionValue.getNumActions(); i++) {
      actionList.add(actionTemplateExpansionValue.getAction(i));
    }
    return actionList.build();
  }

  private static ConfiguredTargetValue createConfiguredTargetValue(
      ActionTemplate<?> actionTemplate) {
    return new NonRuleConfiguredTargetValue(
        Mockito.mock(ConfiguredTarget.class),
        Actions.GeneratingActions.fromSingleAction(actionTemplate, CTKEY),
        NestedSetBuilder.<Package>stableOrder().build());
  }

  private SpecialArtifact createTreeArtifact(String path) {
    PathFragment execPath = PathFragment.create("out").getRelative(path);
    return new SpecialArtifact(
        ArtifactRoot.asDerivedRoot(rootDirectory, "out"),
        execPath,
        CTKEY,
        SpecialArtifactType.TREE);
  }

  private SpecialArtifact createAndPopulateTreeArtifact(String path, String... childRelativePaths)
      throws Exception {
    SpecialArtifact treeArtifact = createTreeArtifact(path);
    Map<TreeFileArtifact, FileArtifactValue> treeFileArtifactMap = new LinkedHashMap<>();

    for (String childRelativePath : childRelativePaths) {
      TreeFileArtifact treeFileArtifact =
          ActionsTestUtil.createTreeFileArtifactWithNoGeneratingAction(
              treeArtifact, childRelativePath);
      scratch.file(treeFileArtifact.getPath().toString(), childRelativePath);
      // We do not care about the FileArtifactValues in this test.
      treeFileArtifactMap.put(
          treeFileArtifact, FileArtifactValue.createForTesting(treeFileArtifact));
    }

    artifactValueMap.put(
        treeArtifact, TreeArtifactValue.create(ImmutableMap.copyOf(treeFileArtifactMap)));

    return treeArtifact;
  }

  /** Dummy ArtifactFunction that just returns injected values */
  private static class DummyArtifactFunction implements SkyFunction {
    private final Map<Artifact, TreeArtifactValue> artifactValueMap;

    DummyArtifactFunction(Map<Artifact, TreeArtifactValue> artifactValueMap) {
      this.artifactValueMap = artifactValueMap;
    }
    @Override
    public SkyValue compute(SkyKey skyKey, Environment env) {
      return Preconditions.checkNotNull(artifactValueMap.get(skyKey));
    }

    @Override
    public String extractTag(SkyKey skyKey) {
      return null;
    }
  }
}
