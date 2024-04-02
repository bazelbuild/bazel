// Copyright 2023 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.createArtifact;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.BasicActionLookupValue;
import com.google.devtools.build.lib.actions.cache.ActionCache;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil.NullAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.skyframe.SkyframeFocuser.FocusResult;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.GraphTester.StringValue;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.NodeEntry;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.Version;
import java.util.Set;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

/** Tests for {@link SkyframeFocuser}. */
@RunWith(JUnit4.class)
public final class SkyframeFocuserTest extends BuildViewTestCase {
  @Rule public final MockitoRule mockito = MockitoJUnit.rule();

  @Mock private ActionCache mockActionCache;

  @Test
  public void testFocus_emptyInputsReturnsEmptyResult() throws InterruptedException {
    InMemoryGraph graph = skyframeExecutor.getEvaluator().getInMemoryGraph();
    FocusResult focusResult =
        SkyframeFocuser.focus(
            graph, mockActionCache, reporter, Sets.newHashSet(), Sets.newHashSet());

    assertThat(focusResult.getDeps()).isEmpty();
    assertThat(focusResult.getRdeps()).isEmpty();
  }

  @Test
  public void testFocus_keepsLeafs() throws InterruptedException {
    InMemoryGraph graph = skyframeExecutor.getEvaluator().getInMemoryGraph();
    SkyKey cat = SkyKeyWithSkyKeyInterner.create("cat");
    SkyKey dog = SkyKeyWithSkyKeyInterner.create("dog");
    ImmutableList<SkyKey> keys = ImmutableList.of(cat, dog);

    graph.createIfAbsentBatch(null, Reason.OTHER, keys);

    createEdgesAndMarkDone(graph, cat, ImmutableList.of(), ImmutableList.of());
    createEdgesAndMarkDone(graph, dog, ImmutableList.of(), ImmutableList.of());

    Set<SkyKey> roots = Sets.newHashSet();
    Set<SkyKey> leafs = Sets.newHashSet(cat, dog);

    FocusResult focusResult = SkyframeFocuser.focus(graph, mockActionCache, reporter, roots, leafs);

    assertThat(focusResult.getDeps()).isEmpty();
    assertThat(focusResult.getRdeps()).containsExactly(cat, dog);
    assertThat(graph.getValues().keySet()).containsExactly(cat, dog);
  }

  @Test
  public void testFocus_dropsUnreachableNodesFromLeafs() throws InterruptedException {
    InMemoryGraph graph = skyframeExecutor.getEvaluator().getInMemoryGraph();
    SkyKey cat = SkyKeyWithSkyKeyInterner.create("cat");
    SkyKey dog = SkyKeyWithSkyKeyInterner.create("dog");
    ImmutableList<SkyKey> keys = ImmutableList.of(cat, dog);

    graph.createIfAbsentBatch(null, Reason.OTHER, keys);

    createEdgesAndMarkDone(graph, cat, ImmutableList.of(), ImmutableList.of());
    createEdgesAndMarkDone(graph, dog, ImmutableList.of(), ImmutableList.of());

    Set<SkyKey> roots = Sets.newHashSet();
    Set<SkyKey> leafs = Sets.newHashSet(cat); // dog is unreachable

    FocusResult focusResult = SkyframeFocuser.focus(graph, mockActionCache, reporter, roots, leafs);

    assertThat(focusResult.getDeps()).isEmpty();
    assertThat(focusResult.getRdeps()).containsExactly(cat);
    assertThat(graph.getValues().keySet()).containsExactly(cat);
  }

  @Test
  public void testFocus_keepsReverseDepOfLeafs() throws InterruptedException {
    InMemoryGraph graph = skyframeExecutor.getEvaluator().getInMemoryGraph();
    SkyKey cat = SkyKeyWithSkyKeyInterner.create("cat");
    SkyKey dog = SkyKeyWithSkyKeyInterner.create("dog");
    ImmutableList<SkyKey> keys = ImmutableList.of(cat, dog);

    graph.createIfAbsentBatch(null, Reason.OTHER, keys);
    createEdgesAndMarkDone(graph, cat, ImmutableList.of(), ImmutableList.of(dog));
    createEdgesAndMarkDone(graph, dog, ImmutableList.of(), ImmutableList.of());

    Set<SkyKey> roots = Sets.newHashSet();
    Set<SkyKey> leafs = Sets.newHashSet(cat); // dog is cat's rdep

    FocusResult focusResult = SkyframeFocuser.focus(graph, mockActionCache, reporter, roots, leafs);

    assertThat(focusResult.getDeps()).isEmpty();
    assertThat(focusResult.getRdeps()).containsExactly(cat, dog);
    assertThat(graph.getValues().keySet()).containsExactly(cat, dog);
  }

  @Test
  public void testFocus_keepsRoots() throws InterruptedException {
    InMemoryGraph graph = skyframeExecutor.getEvaluator().getInMemoryGraph();
    SkyKey cat = SkyKeyWithSkyKeyInterner.create("cat");
    SkyKey dog = SkyKeyWithSkyKeyInterner.create("dog");
    ImmutableList<SkyKey> keys = ImmutableList.of(cat, dog);

    graph.createIfAbsentBatch(null, Reason.OTHER, keys);
    createEdgesAndMarkDone(graph, cat, ImmutableList.of(), ImmutableList.of());
    createEdgesAndMarkDone(graph, dog, ImmutableList.of(), ImmutableList.of());

    Set<SkyKey> roots = Sets.newHashSet(cat, dog);
    Set<SkyKey> leafs = Sets.newHashSet();

    FocusResult focusResult = SkyframeFocuser.focus(graph, mockActionCache, reporter, roots, leafs);

    assertThat(focusResult.getDeps()).containsExactly(cat, dog);
    assertThat(focusResult.getRdeps()).isEmpty();
    assertThat(graph.getValues().keySet()).containsExactly(cat, dog);
  }

  @Test
  public void testFocus_dropsUnreachableFromRoots() throws InterruptedException {
    InMemoryGraph graph = skyframeExecutor.getEvaluator().getInMemoryGraph();
    SkyKey cat = SkyKeyWithSkyKeyInterner.create("cat");
    SkyKey dog = SkyKeyWithSkyKeyInterner.create("dog");
    ImmutableList<SkyKey> keys = ImmutableList.of(cat, dog);

    graph.createIfAbsentBatch(null, Reason.OTHER, keys);
    createEdgesAndMarkDone(graph, cat, ImmutableList.of(), ImmutableList.of());
    createEdgesAndMarkDone(graph, dog, ImmutableList.of(), ImmutableList.of());

    Set<SkyKey> roots = Sets.newHashSet(cat);
    Set<SkyKey> leafs = Sets.newHashSet();

    FocusResult focusResult = SkyframeFocuser.focus(graph, mockActionCache, reporter, roots, leafs);

    assertThat(focusResult.getDeps()).containsExactly(cat);
    assertThat(focusResult.getRdeps()).isEmpty();
    assertThat(graph.getValues().keySet()).containsExactly(cat);
  }

  @Test
  public void testFocus_keepDirectDepsOfRdepTransitiveClosure() throws InterruptedException {
    InMemoryGraph graph = skyframeExecutor.getEvaluator().getInMemoryGraph();
    SkyKey cat = SkyKeyWithSkyKeyInterner.create("cat");
    SkyKey dog = SkyKeyWithSkyKeyInterner.create("dog");
    SkyKey civet = SkyKeyWithSkyKeyInterner.create("civet");
    SkyKey hamster = SkyKeyWithSkyKeyInterner.create("hamster");
    SkyKey fish = SkyKeyWithSkyKeyInterner.create("fish");
    SkyKey bird = SkyKeyWithSkyKeyInterner.create("bird");
    SkyKey monkey = SkyKeyWithSkyKeyInterner.create("monkey");
    ImmutableList<SkyKey> keys = ImmutableList.of(cat, dog, civet, hamster, fish, bird, monkey);
    graph.createIfAbsentBatch(null, Reason.OTHER, keys);

    // Graph:
    //
    // monkey (isolated)
    //
    //    /-> fish -> bird
    // cat -> dog -> civet*
    //          \-> hamster
    //
    // *Only civet in the working set.
    createEdgesAndMarkDone(graph, civet, ImmutableList.of(), ImmutableList.of(dog));
    createEdgesAndMarkDone(graph, hamster, ImmutableList.of(), ImmutableList.of(dog));
    createEdgesAndMarkDone(graph, dog, ImmutableList.of(civet, hamster), ImmutableList.of(cat));
    createEdgesAndMarkDone(graph, bird, ImmutableList.of(), ImmutableList.of(fish));
    createEdgesAndMarkDone(graph, fish, ImmutableList.of(bird), ImmutableList.of(cat));
    createEdgesAndMarkDone(graph, cat, ImmutableList.of(dog, fish), ImmutableList.of());
    createEdgesAndMarkDone(graph, monkey, ImmutableList.of(), ImmutableList.of());

    Set<SkyKey> roots = Sets.newHashSet(cat);
    Set<SkyKey> leafs = Sets.newHashSet(civet);

    FocusResult focusResult = SkyframeFocuser.focus(graph, mockActionCache, reporter, roots, leafs);

    assertThat(focusResult.getDeps()).containsExactly(hamster, fish);
    assertThat(focusResult.getRdeps()).containsExactly(civet, dog, cat);

    // no monkey (isolated) and bird (indirect dep)
    assertThat(graph.getValues().keySet()).containsExactly(hamster, fish, civet, dog, cat);
  }

  @Test
  public void testFocus_removeActionCacheEntries() throws InterruptedException {
    InMemoryGraph graph = skyframeExecutor.getEvaluator().getInMemoryGraph();
    SkyKey cat = SkyKeyWithSkyKeyInterner.create("cat");
    SkyKey dog = SkyKeyWithSkyKeyInterner.create("dog");
    SkyKey hamster = SkyKeyWithSkyKeyInterner.create("hamster");
    ImmutableList<SkyKey> keys = ImmutableList.of(cat, dog, hamster);
    graph.createIfAbsentBatch(null, Reason.OTHER, keys);

    ArtifactRoot artifactRoot =
        ArtifactRoot.asDerivedRoot(
            this.directories.getExecRoot("workspace"), RootType.Output, "blaze-out");

    Action catAction = new NullAction(createArtifact(artifactRoot, "cat"));
    Action dogAction = new NullAction(createArtifact(artifactRoot, "dog"));
    Action hamsterAction = new NullAction(createArtifact(artifactRoot, "hamster"));

    createEdgesAndMarkDone(
        graph,
        cat,
        ImmutableList.of(),
        ImmutableList.of(),
        new BasicActionLookupValue(ImmutableList.of(catAction)));
    createEdgesAndMarkDone(
        graph,
        dog,
        ImmutableList.of(),
        ImmutableList.of(hamster),
        new BasicActionLookupValue(ImmutableList.of(dogAction)));
    createEdgesAndMarkDone(
        graph,
        hamster,
        ImmutableList.of(dog),
        ImmutableList.of(),
        new BasicActionLookupValue(ImmutableList.of(hamsterAction)));

    Set<SkyKey> roots = Sets.newHashSet(hamster);
    Set<SkyKey> leafs = Sets.newHashSet(dog);

    FocusResult unused = SkyframeFocuser.focus(graph, mockActionCache, reporter, roots, leafs);

    verify(mockActionCache).remove(catAction.getPrimaryOutput().getExecPathString());
    verify(mockActionCache, never()).remove(dogAction.getPrimaryOutput().getExecPathString());
    verify(mockActionCache, never()).remove(hamsterAction.getPrimaryOutput().getExecPathString());
  }

  private static void createEdgesAndMarkDone(
      InMemoryGraph graph, SkyKey k, ImmutableList<SkyKey> deps, ImmutableList<SkyKey> rdeps)
      throws InterruptedException {
    createEdgesAndMarkDone(graph, k, deps, rdeps, new StringValue("unused"));
  }

  // Create dep and rdep edges for a node, and ensures that it's marked done.
  private static void createEdgesAndMarkDone(
      InMemoryGraph graph,
      SkyKey k,
      ImmutableList<SkyKey> deps,
      ImmutableList<SkyKey> rdeps,
      SkyValue value)
      throws InterruptedException {
    NodeEntry entry = graph.getIfPresent(k);
    assertThat(entry).isNotNull();
    if (rdeps.isEmpty()) {
      entry.addReverseDepAndCheckIfDone(null);
    } else {
      for (SkyKey rdep : rdeps) {
        entry.addReverseDepAndCheckIfDone(rdep);
      }
    }
    entry.markRebuilding();
    for (SkyKey dep : deps) {
      entry.addSingletonTemporaryDirectDep(dep);
      entry.signalDep(Version.constant(), dep);
    }
    entry.setValue(value, Version.constant(), null);
  }

  private static final class SkyKeyWithSkyKeyInterner extends AbstractSkyKey<String> {
    private static final SkyKeyInterner<SkyframeFocuserTest.SkyKeyWithSkyKeyInterner> interner =
        SkyKey.newInterner();

    static SkyKeyWithSkyKeyInterner create(String arg) {
      return interner.intern(new SkyframeFocuserTest.SkyKeyWithSkyKeyInterner(arg));
    }

    private SkyKeyWithSkyKeyInterner(String arg) {
      super(arg);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctionName.FOR_TESTING;
    }

    @Override
    public SkyKeyInterner<SkyframeFocuserTest.SkyKeyWithSkyKeyInterner> getSkyKeyInterner() {
      return interner;
    }
  }
}
