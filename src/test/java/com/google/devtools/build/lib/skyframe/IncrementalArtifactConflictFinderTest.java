// Copyright 2022 The Bazel Authors. All rights reserved.
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
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.actions.ArtifactPrefixConflictException;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.MapBasedActionGraph;
import com.google.devtools.build.lib.actions.MutableActionGraph;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.actions.util.TestAction.DummyAction;
import com.google.devtools.build.lib.skyframe.ArtifactConflictFinder.ActionConflictsAndStats;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link IncrementalArtifactConflictFinder}. */
@RunWith(JUnit4.class)
public class IncrementalArtifactConflictFinderTest {
  private final Scratch scratch = new Scratch();

  @Test
  public void testFindArtifactConflicts_sequential() throws Exception {

    ActionAnalysisMetadata action1 =
        new DummyAction(
            createTestSourceArtifactWithPath(PathFragment.create("in/a")),
            createTestDerivedArtifactWithPath(PathFragment.create("out/conflicting")));
    ActionLookupValue alv1 = createMockActionLookupValueThatContains(action1);
    ImmutableList<SkyValue> actionLookupValues1 = ImmutableList.of(alv1);

    ActionAnalysisMetadata action2 =
        new DummyAction(
            createTestSourceArtifactWithPath(PathFragment.create("in/b")),
            createTestDerivedArtifactWithPath(PathFragment.create("out/conflicting")));
    ActionLookupValue alv2 = createMockActionLookupValueThatContains(action2);
    ImmutableList<SkyValue> actionLookupValues2 = ImmutableList.of(alv2);

    MutableActionGraph actionGraph = new MapBasedActionGraph(new ActionKeyContext());
    IncrementalArtifactConflictFinder conflictFinder =
        IncrementalArtifactConflictFinder.createWithActionGraph(actionGraph);

    ActionConflictsAndStats expectNoConflict =
        conflictFinder.findArtifactConflicts(actionLookupValues1, /*strictConflictChecks=*/ true);
    assertThat(expectNoConflict.getConflicts()).isEmpty();

    ActionConflictsAndStats expectConflict =
        conflictFinder.findArtifactConflicts(actionLookupValues2, /*strictConflictChecks=*/ true);
    assertThat(expectConflict.getConflicts()).hasSize(1);
    assertThrows(
        ActionConflictException.class,
        () -> expectConflict.getConflicts().get(action2).rethrowTyped());
  }

  @Test
  public void testFindArtifactConflicts_multiThreaded() throws Exception {
    ActionAnalysisMetadata action1 =
        new DummyAction(
            createTestSourceArtifactWithPath(PathFragment.create("in/a")),
            createTestDerivedArtifactWithPath(PathFragment.create("out/conflicting")));
    ActionLookupValue alv1 = createMockActionLookupValueThatContains(action1);
    ImmutableList<SkyValue> actionLookupValues1 = ImmutableList.of(alv1);

    ActionAnalysisMetadata action2 =
        new DummyAction(
            createTestSourceArtifactWithPath(PathFragment.create("in/b")),
            createTestDerivedArtifactWithPath(PathFragment.create("out/conflicting")));
    ActionLookupValue alv2 = createMockActionLookupValueThatContains(action2);
    ImmutableList<SkyValue> actionLookupValues2 = ImmutableList.of(alv2);

    MutableActionGraph actionGraph = new MapBasedActionGraph(new ActionKeyContext());
    IncrementalArtifactConflictFinder conflictFinder =
        IncrementalArtifactConflictFinder.createWithActionGraph(actionGraph);

    ExecutorService executor = Executors.newFixedThreadPool(2);
    Future<ActionConflictsAndStats> future1 =
        executor.submit(
            () ->
                conflictFinder.findArtifactConflicts(
                    actionLookupValues1, /*strictConflictChecks=*/ true));
    Future<ActionConflictsAndStats> future2 =
        executor.submit(
            () ->
                conflictFinder.findArtifactConflicts(
                    actionLookupValues2, /*strictConflictChecks=*/ true));
    ActionConflictsAndStats expectNoConflict;
    ActionConflictsAndStats expectConflict;
    if (future1.get().getConflicts().isEmpty()) {
      expectConflict = future2.get();
      expectNoConflict = future1.get();
    } else {
      expectConflict = future1.get();
      expectNoConflict = future2.get();
    }
    executor.shutdownNow();

    assertThat(expectNoConflict.getConflicts()).isEmpty();
    assertThat(expectConflict.getConflicts()).hasSize(1);
    assertThrows(
        ActionConflictException.class,
        () -> Iterables.getOnlyElement(expectConflict.getConflicts().values()).rethrowTyped());
  }

  @Test
  public void testFindPrefixArtifactConflicts_singleRun() throws Exception {
    ActionAnalysisMetadata action1 =
        new DummyAction(
            createTestSourceArtifactWithPath(PathFragment.create("in/a")),
            createTestDerivedArtifactWithPath(PathFragment.create("out/foo")));
    ActionLookupValue alv1 = createMockActionLookupValueThatContains(action1);

    ActionAnalysisMetadata action2 =
        new DummyAction(
            createTestSourceArtifactWithPath(PathFragment.create("in/b")),
            createTestDerivedArtifactWithPath(PathFragment.create("out/foo/bar")));
    ActionLookupValue alv2 = createMockActionLookupValueThatContains(action2);

    ImmutableList<SkyValue> actionLookupValues = ImmutableList.of(alv1, alv2);
    MutableActionGraph actionGraph = new MapBasedActionGraph(new ActionKeyContext());
    IncrementalArtifactConflictFinder conflictFinder =
        IncrementalArtifactConflictFinder.createWithActionGraph(actionGraph);

    ActionConflictsAndStats result =
        conflictFinder.findArtifactConflicts(actionLookupValues, /*strictConflictChecks=*/ true);

    assertThat(result.getConflicts()).hasSize(2);
    assertThat(result.getConflicts().get(action1)).isEqualTo(result.getConflicts().get(action2));
    assertThrows(
        ArtifactPrefixConflictException.class,
        () -> result.getConflicts().get(action1).rethrowTyped());
  }

  @Test
  public void testFindPrefixArtifactConflicts_noStrictChecks_expectNoConflict() throws Exception {
    ActionAnalysisMetadata action1 =
        new PrefixConflictTolerantDummyAction(
            createTestSourceArtifactWithPath(PathFragment.create("in/a")),
            createTestDerivedArtifactWithPath(PathFragment.create("out/foo")));
    ActionLookupValue alv1 = createMockActionLookupValueThatContains(action1);

    ActionAnalysisMetadata action2 =
        new PrefixConflictTolerantDummyAction(
            createTestSourceArtifactWithPath(PathFragment.create("in/b")),
            createTestDerivedArtifactWithPath(PathFragment.create("out/foo/bar")));
    ActionLookupValue alv2 = createMockActionLookupValueThatContains(action2);

    ImmutableList<SkyValue> actionLookupValues = ImmutableList.of(alv1, alv2);
    MutableActionGraph actionGraph = new MapBasedActionGraph(new ActionKeyContext());
    IncrementalArtifactConflictFinder conflictFinder =
        IncrementalArtifactConflictFinder.createWithActionGraph(actionGraph);

    ActionConflictsAndStats result =
        conflictFinder.findArtifactConflicts(actionLookupValues, /*strictConflictChecks=*/ false);

    assertThat(result.getConflicts()).isEmpty();
  }

  @Test
  public void testFindPrefixArtifactConflicts_multithreaded() throws Exception {
    ActionAnalysisMetadata action1 =
        new DummyAction(
            createTestSourceArtifactWithPath(PathFragment.create("in/a")),
            createTestDerivedArtifactWithPath(PathFragment.create("out/foo")));
    ActionLookupValue alv1 = createMockActionLookupValueThatContains(action1);
    ImmutableList<SkyValue> actionLookupValues1 = ImmutableList.of(alv1);

    ActionAnalysisMetadata action2 =
        new DummyAction(
            createTestSourceArtifactWithPath(PathFragment.create("in/b")),
            createTestDerivedArtifactWithPath(PathFragment.create("out/foo/bar")));
    ActionLookupValue alv2 = createMockActionLookupValueThatContains(action2);
    ImmutableList<SkyValue> actionLookupValues2 = ImmutableList.of(alv2);

    MutableActionGraph actionGraph = new MapBasedActionGraph(new ActionKeyContext());
    IncrementalArtifactConflictFinder conflictFinder =
        IncrementalArtifactConflictFinder.createWithActionGraph(actionGraph);

    ExecutorService executor = Executors.newFixedThreadPool(2);
    Future<ActionConflictsAndStats> future1 =
        executor.submit(
            () ->
                conflictFinder.findArtifactConflicts(
                    actionLookupValues1, /*strictConflictChecks=*/ true));
    Future<ActionConflictsAndStats> future2 =
        executor.submit(
            () ->
                conflictFinder.findArtifactConflicts(
                    actionLookupValues2, /*strictConflictChecks=*/ true));
    ActionConflictsAndStats withConflict =
        future1.get().getConflicts().isEmpty() ? future2.get() : future1.get();
    executor.shutdownNow();

    assertThat(withConflict.getConflicts()).hasSize(2);
    assertThat(withConflict.getConflicts().get(action1))
        .isEqualTo(withConflict.getConflicts().get(action2));
    assertThrows(
        ArtifactPrefixConflictException.class,
        () -> withConflict.getConflicts().get(action1).rethrowTyped());
  }

  private DerivedArtifact createTestDerivedArtifactWithPath(PathFragment path) {
    return DerivedArtifact.create(
        ArtifactRoot.asDerivedRoot(scratch.getFileSystem().getPath("/"), RootType.Output, "out"),
        path,
        mock(ActionLookupKey.class));
  }

  private SourceArtifact createTestSourceArtifactWithPath(PathFragment path) {
    return new SourceArtifact(
        ArtifactRoot.asSourceRoot(Root.fromPath(scratch.getFileSystem().getPath("/"))),
        path,
        mock(ActionLookupKey.class));
  }

  private ActionLookupValue createMockActionLookupValueThatContains(ActionAnalysisMetadata action) {
    ActionLookupValue actionLookupValue = mock(ActionLookupValue.class);
    when(actionLookupValue.getActions()).thenReturn(ImmutableList.of(action));
    return actionLookupValue;
  }

  private static class PrefixConflictTolerantDummyAction extends DummyAction {

    public PrefixConflictTolerantDummyAction(Artifact input, Artifact output) {
      super(input, output);
    }

    @Override
    public boolean shouldReportPathPrefixConflict(ActionAnalysisMetadata action) {
      return false;
    }
  }
}
