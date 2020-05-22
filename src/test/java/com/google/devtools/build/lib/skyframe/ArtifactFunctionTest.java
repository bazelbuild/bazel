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
import static com.google.devtools.build.lib.actions.FileArtifactValue.createForTesting;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata.MiddlemanType;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.ActionTemplate;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.BasicActionLookupValue;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.TestAction.DummyAction;
import com.google.devtools.build.lib.analysis.actions.SpawnActionTemplate;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileStatusWithDigestAdapter;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
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

  private final Set<Artifact> omittedOutputs = new HashSet<>();

  @Before
  public final void setUp() {
    delegateActionExecutionFunction = new SimpleActionExecutionFunction(omittedOutputs);
  }

  private void assertFileArtifactValueMatches(boolean expectDigest) throws Throwable {
    Artifact output = createDerivedArtifact("output");
    Path path = output.getPath();
    file(path, "contents");
    assertValueMatches(path.stat(), expectDigest ? path.getDigest() : null, evaluateFAN(output));
  }

  @Test
  public void testBasicArtifact() throws Throwable {
    fastDigest = false;
    assertFileArtifactValueMatches(/*expectDigest=*/ true);
  }

  @Test
  public void testBasicArtifactWithXattr() throws Throwable {
    fastDigest = true;
    assertFileArtifactValueMatches(/*expectDigest=*/ true);
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
          public byte[] getDigest(Path path) throws IOException {
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

  @Test
  public void testMiddlemanArtifact() throws Throwable {
    Artifact output = createMiddlemanArtifact("output");
    Artifact input1 = createSourceArtifact("input1");
    Artifact input2 = createDerivedArtifact("input2");
    SpecialArtifact tree = createDerivedTreeArtifactWithAction("treeArtifact");
    TreeFileArtifact treeFile1 = createFakeTreeFileArtifact(tree, "child1", "hello1");
    TreeFileArtifact treeFile2 = createFakeTreeFileArtifact(tree, "child2", "hello2");
    file(treeFile1.getPath(), "src1");
    file(treeFile2.getPath(), "src2");
    Action action =
        new DummyAction(
            NestedSetBuilder.create(Order.STABLE_ORDER, input1, input2, tree),
            output,
            MiddlemanType.AGGREGATING_MIDDLEMAN);
    actions.add(action);
    file(input2.getPath(), "contents");
    file(input1.getPath(), "source contents");
    evaluate(
        Iterables.toArray(
            Artifact.keys(ImmutableSet.of(input2, input1, input2, tree)), SkyKey.class));
    SkyValue value = evaluateArtifactValue(output);
    ArrayList<Pair<Artifact, ?>> inputs = new ArrayList<>();
    inputs.addAll(((AggregatingArtifactValue) value).getFileArtifacts());
    inputs.addAll(((AggregatingArtifactValue) value).getTreeArtifacts());
    assertThat(inputs)
        .containsExactly(
            Pair.of(input1, createForTesting(input1)),
            Pair.of(input2, createForTesting(input2)),
            Pair.of(tree, ((TreeArtifactValue) evaluateArtifactValue(tree))));
  }

  /**
   * Tests that ArtifactFunction rethrows transitive {@link IOException}s as
   * {@link MissingInputFileException}s.
   */
  @Test
  public void testIOException_EndToEnd() throws Throwable {
    final IOException exception = new IOException("beep");
    setupRoot(
        new CustomInMemoryFs() {
          @Override
          public FileStatus statIfFound(Path path, boolean followSymlinks) throws IOException {
            if (path.getBaseName().equals("bad")) {
              throw exception;
            }
            return super.statIfFound(path, followSymlinks);
          }
        });
    IOException e =
        assertThrows(IOException.class, () -> evaluateArtifactValue(createSourceArtifact("bad")));
    assertThat(e).hasMessageThat().contains(exception.getMessage());
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
        createFakeExpansionTreeFileArtifact(actionTemplate, "child1", "hello1");
    TreeFileArtifact treeFileArtifact2 =
        createFakeExpansionTreeFileArtifact(actionTemplate, "child2", "hello2");

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
    createFakeExpansionTreeFileArtifact(template2, "child1", "hello1");
    createFakeExpansionTreeFileArtifact(template2, "child2", "hello2");

    // artifact3 is a tree artifact generated by action template.
    SpecialArtifact artifact3 = createDerivedTreeArtifactOnly("treeArtifact3");
    SpawnActionTemplate template3 =
        ActionsTestUtil.createDummySpawnActionTemplate(artifact2, artifact3);
    actions.add(template3);
    TreeFileArtifact treeFileArtifact1 =
        createFakeExpansionTreeFileArtifact(template3, "child1", "hello1");
    TreeFileArtifact treeFileArtifact2 =
        createFakeExpansionTreeFileArtifact(template3, "child2", "hello2");

    TreeArtifactValue value = (TreeArtifactValue) evaluateArtifactValue(artifact3);
    assertThat(value.getChildValues()).containsKey(treeFileArtifact1);
    assertThat(value.getChildValues()).containsKey(treeFileArtifact2);
    assertThat(value.getChildValues().get(treeFileArtifact1).getDigest()).isNotNull();
    assertThat(value.getChildValues().get(treeFileArtifact2).getDigest()).isNotNull();
  }

  @Test
  public void actionTemplateExpansionOutputsOmitted() throws Throwable {
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
        createFakeExpansionTreeFileArtifact(actionTemplate, "child1", "hello1");
    TreeFileArtifact treeFileArtifact2 =
        createFakeExpansionTreeFileArtifact(actionTemplate, "child2", "hello2");

    omittedOutputs.add(treeFileArtifact1);
    omittedOutputs.add(treeFileArtifact2);

    SkyValue value = evaluateArtifactValue(artifact2);
    assertThat(value).isEqualTo(FileArtifactValue.OMITTED_FILE_MARKER);
  }

  @Test
  public void cannotOmitSomeButNotAllActionTemplateExpansionOutputs() throws Throwable {
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
        createFakeExpansionTreeFileArtifact(actionTemplate, "child1", "hello1");
    TreeFileArtifact treeFileArtifact2 =
        createFakeExpansionTreeFileArtifact(actionTemplate, "child2", "hello2");

    omittedOutputs.add(treeFileArtifact1);

    Exception e = assertThrows(RuntimeException.class, () -> evaluateArtifactValue(artifact2));
    assertThat(e).hasCauseThat().isInstanceOf(IllegalStateException.class);
    assertThat(e)
        .hasCauseThat()
        .hasMessageThat()
        .matches(
            "Action template expansion has some but not all outputs omitted, present outputs: .*"
                + treeFileArtifact2.getParentRelativePath()
                + ".*");
  }

  @Test
  public void actionExecutionValueSerialization() throws Exception {
    ActionLookupData dummyData = ActionLookupData.create(ALL_OWNER, 0);
    DerivedArtifact artifact1 = createDerivedArtifact("one");
    FileArtifactValue metadata1 =
        ActionMetadataHandler.fileArtifactValueFromArtifact(artifact1, null, null);
    SpecialArtifact treeArtifact = createDerivedTreeArtifactOnly("tree");
    treeArtifact.setGeneratingActionKey(dummyData);
    TreeFileArtifact treeFileArtifact = TreeFileArtifact.createTreeOutput(treeArtifact, "subpath");
    Path path = treeFileArtifact.getPath();
    path.getParentDirectory().createDirectoryAndParents();
    writeFile(path, "contents");
    TreeArtifactValue treeArtifactValue =
        TreeArtifactValue.create(
            ImmutableMap.of(
                treeFileArtifact, FileArtifactValue.createForTesting(treeFileArtifact)));
    DerivedArtifact artifact3 = createDerivedArtifact("three");
    FilesetOutputSymlink filesetOutputSymlink =
        FilesetOutputSymlink.createForTesting(
            PathFragment.EMPTY_FRAGMENT, PathFragment.EMPTY_FRAGMENT, PathFragment.EMPTY_FRAGMENT);
    ActionExecutionValue actionExecutionValue =
        ActionExecutionValue.create(
            ImmutableMap.of(artifact1, metadata1, artifact3, FileArtifactValue.DEFAULT_MIDDLEMAN),
            ImmutableMap.of(treeArtifact, treeArtifactValue),
            ImmutableList.of(filesetOutputSymlink),
            null,
            true);
    new SerializationTester(actionExecutionValue)
        .addDependency(FileSystem.class, root.getFileSystem())
        .addDependency(
            Root.RootCodecDependencies.class,
            new Root.RootCodecDependencies(Root.absoluteRoot(root.getFileSystem())))
        .runTests();
  }

  private static void file(Path path, String contents) throws Exception {
    FileSystemUtils.createDirectoryAndParents(path.getParentDirectory());
    writeFile(path, contents);
  }

  private Artifact createSourceArtifact(String path) {
    return ActionsTestUtil.createArtifactWithExecPath(
        ArtifactRoot.asSourceRoot(Root.fromPath(root)), PathFragment.create(path));
  }

  private DerivedArtifact createDerivedArtifact(String path) {
    PathFragment execPath = PathFragment.create("out").getRelative(path);
    DerivedArtifact output =
        new DerivedArtifact(ArtifactRoot.asDerivedRoot(root, "out"), execPath, ALL_OWNER);
    actions.add(new DummyAction(NestedSetBuilder.emptySet(Order.STABLE_ORDER), output));
    output.setGeneratingActionKey(ActionLookupData.create(ALL_OWNER, actions.size() - 1));
    return output;
  }

  private Artifact createMiddlemanArtifact(String path) {
    ArtifactRoot middlemanRoot =
        ArtifactRoot.middlemanRoot(middlemanPath, middlemanPath.getRelative("out"));
    return new DerivedArtifact(
        middlemanRoot, middlemanRoot.getExecPath().getRelative(path), ALL_OWNER);
  }

  private SpecialArtifact createDerivedTreeArtifactWithAction(String path) {
    SpecialArtifact treeArtifact = createDerivedTreeArtifactOnly(path);
    actions.add(new DummyAction(NestedSetBuilder.emptySet(Order.STABLE_ORDER), treeArtifact));
    treeArtifact.setGeneratingActionKey(ActionLookupData.create(ALL_OWNER, actions.size() - 1));
    return treeArtifact;
  }

  private SpecialArtifact createDerivedTreeArtifactOnly(String path) {
    PathFragment execPath = PathFragment.create("out").getRelative(path);
    return new SpecialArtifact(
        ArtifactRoot.asDerivedRoot(root, "out"), execPath, ALL_OWNER, SpecialArtifactType.TREE);
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

  private TreeFileArtifact createFakeExpansionTreeFileArtifact(
      ActionTemplate<?> actionTemplate, String parentRelativePath, String content)
      throws Exception {
    int actionIndex = Iterables.indexOf(actions, actionTemplate::equals);
    Preconditions.checkState(actionIndex >= 0, "%s not registered", actionTemplate);
    TreeFileArtifact treeFileArtifact =
        TreeFileArtifact.createTemplateExpansionOutput(
            actionTemplate.getOutputTreeArtifact(),
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

  private FileArtifactValue evaluateFAN(Artifact artifact) throws Throwable {
    return ((FileArtifactValue) evaluateArtifactValue(artifact));
  }

  private SkyValue evaluateArtifactValue(Artifact artifact) throws Throwable {
    SkyKey key = Artifact.key(artifact);
    EvaluationResult<SkyValue> result = evaluate(ImmutableList.of(key).toArray(new SkyKey[0]));
    if (result.hasError()) {
      throw result.getError().getException();
    }
    SkyValue value = result.get(key);
    if (value instanceof ActionExecutionValue) {
      return ((ActionExecutionValue) value)
          .getExistingFileArtifactValue((DerivedArtifact) artifact);
    }
    return value;
  }

  private void setGeneratingActions() throws InterruptedException, ActionConflictException {
    if (evaluator.getExistingValue(ALL_OWNER) == null) {
      differencer.inject(
          ImmutableMap.of(
              ALL_OWNER,
              new BasicActionLookupValue(
                  Actions.assignOwnersAndFilterSharedActionsAndThrowActionConflict(
                      actionKeyContext,
                      ImmutableList.copyOf(actions),
                      ALL_OWNER,
                      /*outputFiles=*/ null))));
    }
  }

  private <E extends SkyValue> EvaluationResult<E> evaluate(SkyKey... keys)
      throws InterruptedException, ActionConflictException {
    setGeneratingActions();
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setNumThreads(SkyframeExecutor.DEFAULT_THREAD_COUNT)
            .setEventHander(NullEventHandler.INSTANCE)
            .build();
    return driver.evaluate(Arrays.asList(keys), evaluationContext);
  }

  /**
   * Value builder for actions that just stats and stores the output file (which must either be
   * orphaned or exist).
   */
  private static final class SimpleActionExecutionFunction implements SkyFunction {
    private final Set<Artifact> omittedOutputs;

    SimpleActionExecutionFunction(Set<Artifact> omittedOutputs) {
      this.omittedOutputs = omittedOutputs;
    }

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
        if (omittedOutputs.contains(output)) {
          Preconditions.checkState(!output.isTreeArtifact(), "Cannot omit %s", output);
          artifactData.put(output, FileArtifactValue.OMITTED_FILE_MARKER);
        } else if (output.isTreeArtifact()) {
          TreeFileArtifact treeFileArtifact1 =
              TreeFileArtifact.createTreeOutput((SpecialArtifact) output, "child1");
          TreeFileArtifact treeFileArtifact2 =
              TreeFileArtifact.createTreeOutput((SpecialArtifact) output, "child2");
          TreeArtifactValue treeArtifactValue =
              TreeArtifactValue.create(
                  ImmutableMap.of(
                      treeFileArtifact1, FileArtifactValue.createForTesting(treeFileArtifact1),
                      treeFileArtifact2, FileArtifactValue.createForTesting(treeFileArtifact2)));
          treeArtifactData.put(output, treeArtifactValue);
        } else if (action.getActionType() == MiddlemanType.NORMAL) {
          Path path = output.getPath();
          FileArtifactValue noDigest =
              ActionMetadataHandler.fileArtifactValueFromArtifact(
                  output,
                  FileStatusWithDigestAdapter.adapt(path.statIfFound(Symlinks.NOFOLLOW)),
                  null);
          FileArtifactValue withDigest =
              FileArtifactValue.createFromInjectedDigest(
                  noDigest, path.getDigest(), !output.isConstantMetadata());
          artifactData.put(output, withDigest);
       } else {
          artifactData.put(output, FileArtifactValue.DEFAULT_MIDDLEMAN);
        }
      } catch (IOException e) {
        throw new IllegalStateException(e);
      }
      return ActionExecutionValue.create(
          artifactData,
          treeArtifactData,
          /*outputSymlinks=*/ null,
          /*discoveredModules=*/ null,
          /*actionDependsOnBuildId=*/ false);
    }

    @Override
    public String extractTag(SkyKey skyKey) {
      return null;
    }
  }
}
