// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.collect.nestedset;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableSet;
import com.google.common.graph.GraphBuilder;
import com.google.common.graph.MutableGraph;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.HashSet;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ArtifactNestedSetKey}'s rewind graph construction. */
@RunWith(JUnit4.class)
public final class ArtifactNestedSetKeyTest {

  private static final FileSystem FS = new InMemoryFileSystem(DigestHashFunction.SHA256);
  private static final ArtifactRoot ROOT =
      ArtifactRoot.asDerivedRoot(FS.getPath("/execroot"), RootType.OUTPUT, "out");

  /**
   * Stands in for the {@link ActionLookupData} of an aggregator action such as {@code
   * RunfilesTreeAction}, for which {@code ActionRewindStrategy} takes the precise branch.
   */
  private static final SkyKey AGGREGATOR = actionKey(1000);

  /**
   * Stands in for the {@link ActionLookupData} of a non-aggregator action that insensitively
   * propagates inputs, such as a runfiles {@code SymlinkTreeAction} on Windows, for which {@code
   * ActionRewindStrategy} takes the imprecise branch.
   */
  private static final SkyKey PROPAGATOR = actionKey(1001);

  private int nextActionIndex = 0;

  @Test
  public void entireNestedSetAfterPrecisePartialWalk_addsRemainingLeaves() {
    DerivedArtifact lost = derivedArtifact("lost");
    DerivedArtifact untouched = derivedArtifact("untouched");
    ArtifactNestedSetKey key =
        keyOf(NestedSetBuilder.<Artifact>stableOrder().add(lost).add(untouched).build());

    MutableGraph<SkyKey> rewindGraph = newRewindGraph();
    ArtifactNestedSetKey.addNestedSetPathsToRewindGraph(
        rewindGraph, AGGREGATOR, key, ImmutableSet.of(lost), new HashSet<>());
    ArtifactNestedSetKey.addEntireNestedSetToRewindGraph(rewindGraph, key, new HashSet<>());

    assertThat(rewindGraph.nodes()).contains(Artifact.key(untouched));
    assertThat(rewindGraph.successors(key))
        .containsExactly(Artifact.key(lost), Artifact.key(untouched));
  }

  @Test
  public void entireNestedSetAfterPrecisePartialWalk_addsIntermediateNodes() {
    DerivedArtifact lost = derivedArtifact("lost");
    DerivedArtifact untouched = derivedArtifact("untouched");
    DerivedArtifact alsoUntouched = derivedArtifact("also_untouched");
    NestedSet<Artifact> inner =
        NestedSetBuilder.<Artifact>stableOrder().add(untouched).add(alsoUntouched).build();
    NestedSet<Artifact> outer =
        NestedSetBuilder.<Artifact>stableOrder().add(lost).addTransitive(inner).build();
    ArtifactNestedSetKey outerKey = keyOf(outer);
    ArtifactNestedSetKey innerKey = keyOf(inner);

    MutableGraph<SkyKey> rewindGraph = newRewindGraph();
    ArtifactNestedSetKey.addNestedSetPathsToRewindGraph(
        rewindGraph, AGGREGATOR, outerKey, ImmutableSet.of(lost), new HashSet<>());
    ArtifactNestedSetKey.addEntireNestedSetToRewindGraph(rewindGraph, outerKey, new HashSet<>());
    rewindGraph.putEdge(PROPAGATOR, outerKey);

    // Without the intermediate node, the leaves below it are rewound while a done
    // ArtifactNestedSetValue still depends on them.
    assertThat(rewindGraph.nodes()).contains(innerKey);
    assertThat(rewindGraph.successors(innerKey))
        .containsExactly(Artifact.key(untouched), Artifact.key(alsoUntouched));
  }

  @Test
  public void precisePartialWalkAfterEntireNestedSet_addsRemainingLeaves() {
    DerivedArtifact lost = derivedArtifact("lost");
    DerivedArtifact untouched = derivedArtifact("untouched");
    ArtifactNestedSetKey key =
        keyOf(NestedSetBuilder.<Artifact>stableOrder().add(lost).add(untouched).build());

    MutableGraph<SkyKey> rewindGraph = newRewindGraph();
    ArtifactNestedSetKey.addEntireNestedSetToRewindGraph(rewindGraph, key, new HashSet<>());
    ArtifactNestedSetKey.addNestedSetPathsToRewindGraph(
        rewindGraph, AGGREGATOR, key, ImmutableSet.of(lost), new HashSet<>());

    assertThat(rewindGraph.successors(key))
        .containsExactly(Artifact.key(lost), Artifact.key(untouched));
    assertThat(rewindGraph.successors(AGGREGATOR)).containsExactly(key);
  }

  private static MutableGraph<SkyKey> newRewindGraph() {
    return GraphBuilder.directed().allowsSelfLoops(false).build();
  }

  private static ArtifactNestedSetKey keyOf(NestedSet<Artifact> set) {
    return ArtifactNestedSetKey.create(set);
  }

  private DerivedArtifact derivedArtifact(String name) {
    DerivedArtifact artifact =
        DerivedArtifact.create(
            ROOT, ROOT.getExecPath().getRelative(name), ActionsTestUtil.NULL_ARTIFACT_OWNER);
    artifact.setGeneratingActionKey(actionKey(nextActionIndex++));
    return artifact;
  }

  private static ActionLookupData actionKey(int actionIndex) {
    return ActionLookupData.create(ActionsTestUtil.NULL_ARTIFACT_OWNER, actionIndex);
  }
}
