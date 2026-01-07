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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
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
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.BasicActionLookupValue;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.TestAction.DummyAction;
import com.google.devtools.build.lib.analysis.actions.SpawnActionTemplate;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.skyframe.ArtifactFunction.SourceArtifactException;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileStatusWithDigestAdapter;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.skyframe.Differencer.DiffWithDelta.Delta;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link ArtifactFunction}.
 */
// Doesn't actually need any particular Skyframe, but is only relevant to Skyframe full mode.
@RunWith(JUnit4.class)
public class ArtifactFunctionTest extends ArtifactFunctionTestCase {

  @Before
  public final void setUp() {
    delegateActionExecutionFunction = new SimpleActionExecutionFunction();
  }

  private void assertFileArtifactValueMatches() throws Exception {
    Artifact output = createDerivedArtifact("output");
    Path path = output.getPath();
    file(path, "contents");
    assertValueMatches(path.stat(), path.getDigest(), evaluateFileArtifactValue(output));
  }

  @Test
  public void testBasicArtifact() throws Exception {
    fastDigest = false;
    assertFileArtifactValueMatches();
  }

  @Test
  public void testBasicArtifactWithXattr() throws Exception {
    fastDigest = true;
    assertFileArtifactValueMatches();
  }

  @Test
  public void testMissingNonMandatoryArtifact() throws Throwable {
    Artifact input = createSourceArtifact("input1");
    assertThat(evaluateArtifactValue(input)).isNotNull();
  }

  @Test
  public void testUnreadableInputWithFsWithAvailableDigest() throws Throwable {
    final byte[] expectedDigest = {1, 2, 3, 4};
    setupRoot(
        new CustomInMemoryFs() {
          @Override
          public byte[] getDigest(PathFragment path) throws IOException {
            return path.getBaseName().equals("unreadable") ? expectedDigest : super.getDigest(path);
          }
        });

    Artifact input = createSourceArtifact("unreadable");
    Path inputPath = input.getPath();
    file(inputPath, "dummynotused");
    inputPath.chmod(0);

    FileArtifactValue value = (FileArtifactValue) evaluateArtifactValue(input);

    FileStatus stat = inputPath.stat();
    assertThat(value.getSize()).isEqualTo(stat.getSize());
    assertThat(value.getDigest()).isEqualTo(expectedDigest);
  }

  /**
   * Tests that ArtifactFunction rethrows a transitive {@link IOException} as an {@link
   * SourceArtifactException}.
   */
  @Test
  public void testIOException_endToEnd() throws Throwable {
    IOException exception = new IOException("beep");
    setupRoot(
        new CustomInMemoryFs() {
          @Override
          public FileStatus statIfFound(PathFragment path, boolean followSymlinks)
              throws IOException {
            if (path.getBaseName().equals("bad")) {
              throw exception;
            }
            return super.statIfFound(path, followSymlinks);
          }
        });
    Artifact sourceArtifact = createSourceArtifact("bad");
    SourceArtifactException e =
        assertThrows(SourceArtifactException.class, () -> evaluateArtifactValue(sourceArtifact));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo("error reading file '" + sourceArtifact.getExecPathString() + "': beep");
  }

  @Test
  public void testActionTreeArtifactOutput() throws Throwable {
    SpecialArtifact artifact = createDerivedTreeArtifactWithAction("treeArtifact");
    TreeFileArtifact treeFileArtifact1 = createFakeTreeFileArtifact(artifact, "child1", "hello1");
    TreeFileArtifact treeFileArtifact2 = createFakeTreeFileArtifact(artifact, "child2", "hello2");

    TreeArtifactValue value = (TreeArtifactValue) evaluateArtifactValue(artifact);
    assertThat(value.getChildValues()).containsKey(treeFileArtifact1);
    assertThat(value.getChildValues()).containsKey(treeFileArtifact2);
    assertThat(value.getChildValues().get(treeFileArtifact1).getDigest()).isNotNull();
    assertThat(value.getChildValues().get(treeFileArtifact2).getDigest()).isNotNull();
  }

  @Test
  public void testSpawnActionTemplate() throws Throwable {
    // artifact1 is a tree artifact generated by normal action.
    SpecialArtifact artifact1 = createDerivedTreeArtifactWithAction("treeArtifact1");
    createFakeTreeFileArtifact(artifact1, "child1", "hello1");
    createFakeTreeFileArtifact(artifact1, "child2", "hello2");

    // artifact2 is a tree artifact generated by action template.
    SpecialArtifact artifact2 = createDerivedTreeArtifactOnly("treeArtifact2");
    SpawnActionTemplate actionTemplate =
        ActionsTestUtil.createDummySpawnActionTemplate(artifact1, artifact2);
    actions.add(actionTemplate);
    TreeFileArtifact treeFileArtifact1 =
        createFakeExpansionTreeFileArtifact(actionTemplate, artifact2, "child1", "hello1");
    TreeFileArtifact treeFileArtifact2 =
        createFakeExpansionTreeFileArtifact(actionTemplate, artifact2, "child2", "hello2");

    TreeArtifactValue value = (TreeArtifactValue) evaluateArtifactValue(artifact2);
    assertThat(value.getChildValues()).containsKey(treeFileArtifact1);
    assertThat(value.getChildValues()).containsKey(treeFileArtifact2);
    assertThat(value.getChildValues().get(treeFileArtifact1).getDigest()).isNotNull();
    assertThat(value.getChildValues().get(treeFileArtifact2).getDigest()).isNotNull();
  }

  @Test
  public void testConsecutiveSpawnActionTemplates() throws Throwable {
    // artifact1 is a tree artifact generated by normal action.
    SpecialArtifact artifact1 = createDerivedTreeArtifactWithAction("treeArtifact1");
    createFakeTreeFileArtifact(artifact1, "child1", "hello1");
    createFakeTreeFileArtifact(artifact1, "child2", "hello2");

    // artifact2 is a tree artifact generated by action template.
    SpecialArtifact artifact2 = createDerivedTreeArtifactOnly("treeArtifact2");
    SpawnActionTemplate template2 =
        ActionsTestUtil.createDummySpawnActionTemplate(artifact1, artifact2);
    actions.add(template2);
    createFakeExpansionTreeFileArtifact(template2, artifact2, "child1", "hello1");
    createFakeExpansionTreeFileArtifact(template2, artifact2, "child2", "hello2");

    // artifact3 is a tree artifact generated by action template.
    SpecialArtifact artifact3 = createDerivedTreeArtifactOnly("treeArtifact3");
    SpawnActionTemplate template3 =
        ActionsTestUtil.createDummySpawnActionTemplate(artifact2, artifact3);
    actions.add(template3);
    TreeFileArtifact treeFileArtifact1 =
        createFakeExpansionTreeFileArtifact(template3, artifact3, "child1", "hello1");
    TreeFileArtifact treeFileArtifact2 =
        createFakeExpansionTreeFileArtifact(template3, artifact3, "child2", "hello2");

    TreeArtifactValue value = (TreeArtifactValue) evaluateArtifactValue(artifact3);
    assertThat(value.getChildValues()).containsKey(treeFileArtifact1);
    assertThat(value.getChildValues()).containsKey(treeFileArtifact2);
    assertThat(value.getChildValues().get(treeFileArtifact1).getDigest()).isNotNull();
    assertThat(value.getChildValues().get(treeFileArtifact2).getDigest()).isNotNull();
  }

  @Test
  public void testActionTemplateGeneratesMultipleOutputTreesFromDifferentActions()
      throws Throwable {
    // `inputTree` is a tree artifact generated by normal action.
    SpecialArtifact inputTree = createDerivedTreeArtifactWithAction("treeArtifact1");
    createFakeTreeFileArtifact(inputTree, "child1", "hello1");
    createFakeTreeFileArtifact(inputTree, "child2", "hello2");
    SpecialArtifact outputTree1 = createDerivedTreeArtifactOnly("treeArtifact2");
    SpecialArtifact outputTree2 = createDerivedTreeArtifactOnly("treeArtifact3");
    ActionTemplate<DummyAction> template =
        new TestActionTemplate(
            ImmutableList.of(inputTree), ImmutableSet.of(outputTree1, outputTree2)) {
          @Override
          public ImmutableList<DummyAction> generateActionsForInputArtifacts(
              ImmutableList<TreeFileArtifact> inputTreeFileArtifacts,
              ActionLookupKey artifactOwner,
              EventHandler eventHandler) {
            ImmutableList.Builder<DummyAction> actions = ImmutableList.builder();
            for (SpecialArtifact outputTree : ImmutableSet.of(outputTree1, outputTree2)) {
              TreeFileArtifact output =
                  TreeFileArtifact.createTemplateExpansionOutput(
                      outputTree, "child", artifactOwner);
              actions.add(new DummyAction(NestedSetBuilder.emptySet(Order.STABLE_ORDER), output));
            }
            return actions.build();
          }
        };
    actions.add(template);
    TreeFileArtifact treeFileArtifact1 =
        createFakeExpansionTreeFileArtifact(template, outputTree1, "child", "hello");
    TreeFileArtifact treeFileArtifact2 =
        createFakeExpansionTreeFileArtifact(template, outputTree2, "child", "hello");
    TreeArtifactValue value = (TreeArtifactValue) evaluateArtifactValue(outputTree1);
    TreeArtifactValue value2 = (TreeArtifactValue) evaluateArtifactValue(outputTree2);

    assertThat(value.getChildValues()).containsKey(treeFileArtifact1);
    assertThat(value2.getChildValues()).containsKey(treeFileArtifact2);
    // The TreeArtifactValue for outputTree1 should not contain the child from outputTree2 and vice
    // versa.
    assertThat(value.getChildValues()).doesNotContainKey(treeFileArtifact2);
    assertThat(value2.getChildValues()).doesNotContainKey(treeFileArtifact1);
    assertThat(value.getChildValues().get(treeFileArtifact1).getDigest()).isNotNull();
    assertThat(value2.getChildValues().get(treeFileArtifact2).getDigest()).isNotNull();
  }

  private static void file(Path path, String contents) throws Exception {
    path.getParentDirectory().createDirectoryAndParents();
    writeFile(path, contents);
  }

  private Artifact createSourceArtifact(String path) {
    return ActionsTestUtil.createArtifactWithExecPath(
        ArtifactRoot.asSourceRoot(Root.fromPath(root)), PathFragment.create(path));
  }

  private DerivedArtifact createDerivedArtifact(String path) {
    PathFragment execPath = PathFragment.create("out").getRelative(path);
    DerivedArtifact output =
        DerivedArtifact.create(
            ArtifactRoot.asDerivedRoot(root, RootType.OUTPUT, "out"), execPath, ALL_OWNER);
    actions.add(new DummyAction(NestedSetBuilder.emptySet(Order.STABLE_ORDER), output));
    output.setGeneratingActionKey(ActionLookupData.create(ALL_OWNER, actions.size() - 1));
    return output;
  }

  private SpecialArtifact createDerivedTreeArtifactWithAction(String path) {
    SpecialArtifact treeArtifact = createDerivedTreeArtifactOnly(path);
    actions.add(new DummyAction(NestedSetBuilder.emptySet(Order.STABLE_ORDER), treeArtifact));
    treeArtifact.setGeneratingActionKey(ActionLookupData.create(ALL_OWNER, actions.size() - 1));
    return treeArtifact;
  }

  private SpecialArtifact createDerivedTreeArtifactOnly(String path) {
    PathFragment execPath = PathFragment.create("out").getRelative(path);
    return SpecialArtifact.create(
        ArtifactRoot.asDerivedRoot(root, RootType.OUTPUT, "out"),
        execPath,
        ALL_OWNER,
        SpecialArtifactType.TREE);
  }

  private static TreeFileArtifact createFakeTreeFileArtifact(
      SpecialArtifact treeArtifact, String parentRelativePath, String content) throws Exception {
    TreeFileArtifact treeFileArtifact =
        TreeFileArtifact.createTreeOutput(treeArtifact, parentRelativePath);
    Path path = treeFileArtifact.getPath();
    path.getParentDirectory().createDirectoryAndParents();
    writeFile(path, content);
    return treeFileArtifact;
  }

  @CanIgnoreReturnValue
  private TreeFileArtifact createFakeExpansionTreeFileArtifact(
      ActionTemplate<?> actionTemplate,
      SpecialArtifact outputTreeArtifact,
      String parentRelativePath,
      String content)
      throws Exception {
    int actionIndex = Iterables.indexOf(actions, actionTemplate::equals);
    Preconditions.checkState(actionIndex >= 0, "%s not registered", actionTemplate);
    TreeFileArtifact treeFileArtifact =
        TreeFileArtifact.createTemplateExpansionOutput(
            outputTreeArtifact,
            parentRelativePath,
            ActionTemplateExpansionValue.key(ALL_OWNER, actionIndex));
    Path path = treeFileArtifact.getPath();
    path.getParentDirectory().createDirectoryAndParents();
    writeFile(path, content);
    return treeFileArtifact;
  }

  private static void assertValueMatches(FileStatus file, byte[] digest, FileArtifactValue value)
      throws IOException {
    assertThat(value.getSize()).isEqualTo(file.getSize());
    if (digest == null) {
      assertThat(value.getDigest()).isNull();
      assertThat(value.getModifiedTime()).isEqualTo(file.getLastModifiedTime());
    } else {
      assertThat(value.getDigest()).isEqualTo(digest);
    }
  }

  private FileArtifactValue evaluateFileArtifactValue(Artifact artifact) throws Exception {
    SkyValue value = evaluateArtifactValue(artifact);
    assertThat(value).isInstanceOf(FileArtifactValue.class);
    return (FileArtifactValue) value;
  }

  private SkyValue evaluateArtifactValue(Artifact artifact) throws Exception {
    SkyKey key = Artifact.key(artifact);
    EvaluationResult<SkyValue> result = evaluate(ImmutableList.of(key).toArray(new SkyKey[0]));
    if (result.hasError()) {
      throw result.getError().getException();
    }
    SkyValue value = result.get(key);
    if (value instanceof ActionExecutionValue actionExecutionValue) {
      return actionExecutionValue.getExistingFileArtifactValue(artifact);
    }
    return value;
  }

  private void setGeneratingActions()
      throws InterruptedException, ActionConflictException,
          Actions.ArtifactGeneratedByOtherRuleException {
    if (evaluator.getExistingValue(ALL_OWNER) == null) {
      ImmutableList<ActionAnalysisMetadata> generatingActions = ImmutableList.copyOf(actions);
      Actions.assignOwnersAndThrowIfConflictToleratingSharedActions(
          actionKeyContext, generatingActions, ALL_OWNER);
      differencer.inject(
          ImmutableMap.of(ALL_OWNER, Delta.justNew(new BasicActionLookupValue(generatingActions))));
    }
  }

  private <E extends SkyValue> EvaluationResult<E> evaluate(SkyKey... keys)
      throws InterruptedException, ActionConflictException,
          Actions.ArtifactGeneratedByOtherRuleException {
    setGeneratingActions();
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setParallelism(SkyframeExecutor.DEFAULT_THREAD_COUNT)
            .setEventHandler(NullEventHandler.INSTANCE)
            .build();
    return evaluator.evaluate(Arrays.asList(keys), evaluationContext);
  }

  /**
   * Value builder for actions that just stats and stores the output file (which must either be
   * orphaned or exist).
   */
  private static final class SimpleActionExecutionFunction implements SkyFunction {
    SimpleActionExecutionFunction() {}

    @Override
    public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
      Map<Artifact, FileArtifactValue> artifactData = new HashMap<>();
      Map<Artifact, TreeArtifactValue> treeArtifactData = new HashMap<>();
      ActionLookupData actionLookupData = (ActionLookupData) skyKey.argument();
      ActionLookupValue actionLookupValue =
          (ActionLookupValue) env.getValue(actionLookupData.getActionLookupKey());
      Action action = actionLookupValue.getAction(actionLookupData.getActionIndex());
      Artifact output = Iterables.getOnlyElement(action.getOutputs());

      try {
        if (output.isTreeArtifact()) {
          SpecialArtifact parent = (SpecialArtifact) output;
          TreeFileArtifact treeFileArtifact1 =
              TreeFileArtifact.createTreeOutput((SpecialArtifact) output, "child1");
          TreeFileArtifact treeFileArtifact2 =
              TreeFileArtifact.createTreeOutput((SpecialArtifact) output, "child2");
          TreeArtifactValue tree =
              TreeArtifactValue.newBuilder(parent)
                  .putChild(
                      treeFileArtifact1, FileArtifactValue.createForTesting(treeFileArtifact1))
                  .putChild(
                      treeFileArtifact2, FileArtifactValue.createForTesting(treeFileArtifact2))
                  .build();
          treeArtifactData.put(output, tree);
        } else if (output.isRunfilesTree()) {
          artifactData.put(output, FileArtifactValue.RUNFILES_TREE_MARKER);
        } else {
          Path path = output.getPath();
          FileArtifactValue noDigest =
              ActionOutputMetadataStore.fileArtifactValueFromArtifact(
                  output,
                  FileStatusWithDigestAdapter.maybeAdapt(path.statIfFound(Symlinks.NOFOLLOW)),
                  SyscallCache.NO_CACHE,
                  null);
          FileArtifactValue withDigest =
              FileArtifactValue.createFromInjectedDigest(noDigest, path.getDigest());
          artifactData.put(output, withDigest);
        }
      } catch (IOException e) {
        throw new IllegalStateException(e);
      }
      return ActionsTestUtil.createActionExecutionValue(
          ImmutableMap.copyOf(artifactData), ImmutableMap.copyOf(treeArtifactData));
    }
  }

  private abstract static class TestActionTemplate implements ActionTemplate<DummyAction> {
    private final ImmutableList<SpecialArtifact> inputTreeArtifacts;
    private final ImmutableSet<SpecialArtifact> outputTreeArtifacts;

    TestActionTemplate(
        ImmutableList<SpecialArtifact> inputTreeArtifacts,
        ImmutableSet<SpecialArtifact> outputTreeArtifacts) {
      for (SpecialArtifact inputTreeArtifact : inputTreeArtifacts) {
        Preconditions.checkArgument(inputTreeArtifact.isTreeArtifact(), inputTreeArtifact);
      }
      for (SpecialArtifact outputTreeArtifact : outputTreeArtifacts) {
        Preconditions.checkArgument(outputTreeArtifact.isTreeArtifact(), outputTreeArtifact);
      }
      this.inputTreeArtifacts = inputTreeArtifacts;
      this.outputTreeArtifacts = outputTreeArtifacts;
    }

    @Override
    public ImmutableList<SpecialArtifact> getInputTreeArtifacts() {
      return inputTreeArtifacts;
    }

    @Override
    public ImmutableSet<Artifact> getOutputs() {
      return ImmutableSet.copyOf(outputTreeArtifacts);
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
      for (SpecialArtifact outputTreeArtifact : outputTreeArtifacts) {
        fp.addPath(outputTreeArtifact.getPath());
      }
      return fp.hexDigestAndReset();
    }

    @Override
    public String prettyPrint() {
      return "TestActionTemplate for " + outputTreeArtifacts;
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
    public NestedSet<Artifact> getAnalysisTimeInputs() {
      return getInputs();
    }

    @Override
    public NestedSet<Artifact> getSchedulingDependencies() {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }

    @Override
    public ImmutableList<String> getClientEnvironmentVariables() {
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
