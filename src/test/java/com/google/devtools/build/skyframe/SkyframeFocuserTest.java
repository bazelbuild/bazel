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
package com.google.devtools.build.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.skyframe.GraphTester.StringValue;
import com.google.devtools.build.skyframe.InMemoryGraphTest.SkyKeyWithSkyKeyInterner;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import com.google.devtools.build.skyframe.SkyframeFocuser.FocusResult;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SkyframeFocuser}. */
@RunWith(JUnit4.class)
public final class SkyframeFocuserTest {

  private EventCollector eventCollector;
  private ExtendedEventHandler reporter;

  @Before
  public void setup() {
    eventCollector = new EventCollector();
    reporter = new Reporter(new EventBus(), eventCollector);
  }

  @Test
  public void testFocus_emptyInputsReturnsEmptyResult() throws InterruptedException {
    InMemoryGraphImpl graph = new InMemoryGraphImpl();
    FocusResult focusResult =
        SkyframeFocuser.focus(
            graph, reporter, Sets.newHashSet(), Sets.newHashSet(), (SkyKey k) -> ImmutableSet.of());

    assertThat(focusResult.getDeps()).isEmpty();
    assertThat(focusResult.getRdeps()).isEmpty();
  }

  @Test
  public void testFocus_keepsLeafs() throws InterruptedException {
    InMemoryGraphImpl graph = new InMemoryGraphImpl();
    SkyKey cat = SkyKeyWithSkyKeyInterner.create("cat");
    SkyKey dog = SkyKeyWithSkyKeyInterner.create("dog");
    ImmutableList<SkyKey> keys = ImmutableList.of(cat, dog);

    graph.createIfAbsentBatch(null, Reason.OTHER, keys);

    createEdgesAndMarkDone(graph, cat, ImmutableList.of(), ImmutableList.of());
    createEdgesAndMarkDone(graph, dog, ImmutableList.of(), ImmutableList.of());

    Set<SkyKey> roots = Sets.newHashSet();
    Set<SkyKey> leafs = Sets.newHashSet(cat, dog);

    FocusResult focusResult =
        SkyframeFocuser.focus(graph, reporter, roots, leafs, (SkyKey k) -> ImmutableSet.of());

    assertThat(focusResult.getDeps()).isEmpty();
    assertThat(focusResult.getRdeps()).containsExactly(cat, dog);
    assertThat(graph.getValues().keySet()).containsExactly(cat, dog);
  }

  @Test
  public void testFocus_dropsUnreachableNodesFromLeafs() throws InterruptedException {
    InMemoryGraphImpl graph = new InMemoryGraphImpl();
    SkyKey cat = SkyKeyWithSkyKeyInterner.create("cat");
    SkyKey dog = SkyKeyWithSkyKeyInterner.create("dog");
    ImmutableList<SkyKey> keys = ImmutableList.of(cat, dog);

    graph.createIfAbsentBatch(null, Reason.OTHER, keys);

    createEdgesAndMarkDone(graph, cat, ImmutableList.of(), ImmutableList.of());
    createEdgesAndMarkDone(graph, dog, ImmutableList.of(), ImmutableList.of());

    Set<SkyKey> roots = Sets.newHashSet();
    Set<SkyKey> leafs = Sets.newHashSet(cat); // dog is unreachable

    FocusResult focusResult =
        SkyframeFocuser.focus(graph, reporter, roots, leafs, (SkyKey k) -> ImmutableSet.of());

    assertThat(focusResult.getDeps()).isEmpty();
    assertThat(focusResult.getRdeps()).containsExactly(cat);
    assertThat(graph.getValues().keySet()).containsExactly(cat);
  }

  @Test
  public void testFocus_keepsReverseDepOfLeafs() throws InterruptedException {
    InMemoryGraphImpl graph = new InMemoryGraphImpl();
    SkyKey cat = SkyKeyWithSkyKeyInterner.create("cat");
    SkyKey dog = SkyKeyWithSkyKeyInterner.create("dog");
    ImmutableList<SkyKey> keys = ImmutableList.of(cat, dog);

    graph.createIfAbsentBatch(null, Reason.OTHER, keys);
    createEdgesAndMarkDone(graph, cat, ImmutableList.of(), ImmutableList.of(dog));
    createEdgesAndMarkDone(graph, dog, ImmutableList.of(), ImmutableList.of());

    Set<SkyKey> roots = Sets.newHashSet();
    Set<SkyKey> leafs = Sets.newHashSet(cat); // dog is cat's rdep

    FocusResult focusResult =
        SkyframeFocuser.focus(graph, reporter, roots, leafs, (SkyKey k) -> ImmutableSet.of());

    assertThat(focusResult.getDeps()).isEmpty();
    assertThat(focusResult.getRdeps()).containsExactly(cat, dog);
    assertThat(graph.getValues().keySet()).containsExactly(cat, dog);
  }

  @Test
  public void testFocus_keepsRoots() throws InterruptedException {
    InMemoryGraphImpl graph = new InMemoryGraphImpl();
    SkyKey cat = SkyKeyWithSkyKeyInterner.create("cat");
    SkyKey dog = SkyKeyWithSkyKeyInterner.create("dog");
    ImmutableList<SkyKey> keys = ImmutableList.of(cat, dog);

    graph.createIfAbsentBatch(null, Reason.OTHER, keys);
    createEdgesAndMarkDone(graph, cat, ImmutableList.of(), ImmutableList.of());
    createEdgesAndMarkDone(graph, dog, ImmutableList.of(), ImmutableList.of());

    Set<SkyKey> roots = Sets.newHashSet(cat, dog);
    Set<SkyKey> leafs = Sets.newHashSet();

    FocusResult focusResult =
        SkyframeFocuser.focus(graph, reporter, roots, leafs, (SkyKey k) -> ImmutableSet.of());

    assertThat(focusResult.getDeps()).containsExactly(cat, dog);
    assertThat(focusResult.getRdeps()).isEmpty();
    assertThat(graph.getValues().keySet()).containsExactly(cat, dog);
  }

  @Test
  public void testFocus_dropsUnreachableFromRoots() throws InterruptedException {
    InMemoryGraphImpl graph = new InMemoryGraphImpl();
    SkyKey cat = SkyKeyWithSkyKeyInterner.create("cat");
    SkyKey dog = SkyKeyWithSkyKeyInterner.create("dog");
    ImmutableList<SkyKey> keys = ImmutableList.of(cat, dog);

    graph.createIfAbsentBatch(null, Reason.OTHER, keys);
    createEdgesAndMarkDone(graph, cat, ImmutableList.of(), ImmutableList.of());
    createEdgesAndMarkDone(graph, dog, ImmutableList.of(), ImmutableList.of());

    Set<SkyKey> roots = Sets.newHashSet(cat);
    Set<SkyKey> leafs = Sets.newHashSet();

    FocusResult focusResult =
        SkyframeFocuser.focus(graph, reporter, roots, leafs, (SkyKey k) -> ImmutableSet.of());

    assertThat(focusResult.getDeps()).containsExactly(cat);
    assertThat(focusResult.getRdeps()).isEmpty();
    assertThat(graph.getValues().keySet()).containsExactly(cat);
  }

  @Test
  public void testFocus_keepDirectDepsOfRdepTransitiveClosure() throws InterruptedException {
    InMemoryGraphImpl graph = new InMemoryGraphImpl();
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

    Set<SkyKey> roots = Sets.newHashSet(cat);
    Set<SkyKey> leafs = Sets.newHashSet(civet);

    FocusResult focusResult =
        SkyframeFocuser.focus(graph, reporter, roots, leafs, (SkyKey k) -> ImmutableSet.of());

    assertThat(focusResult.getDeps()).containsExactly(hamster, fish);
    assertThat(focusResult.getRdeps()).containsExactly(civet, dog, cat);

    // no monkey (isolated) and bird (indirect dep)
    assertThat(graph.getValues().keySet()).containsExactly(hamster, fish, civet, dog, cat);
  }

  // Create dep and rdep edges for a node, and ensures that it's marked done.
  private static void createEdgesAndMarkDone(
      InMemoryGraphImpl graph, SkyKey k, ImmutableList<SkyKey> deps, ImmutableList<SkyKey> rdeps)
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
    entry.setValue(new StringValue("unused"), Version.constant(), null);
  }
}
