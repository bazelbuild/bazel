// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.util.LabelArtifactOwner;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsInOutputGroup;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsToBuild;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.SuccessfulArtifactFilter;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit test for {@link SuccessfulArtifactFilter}. */
@RunWith(JUnit4.class)
public class SuccessfulArtifactFilterTest {
  private final Scratch scratch = new Scratch();

  private ArtifactRoot root;
  private TopLevelArtifactContext ctx;
  private OutputGroupInfo groupProvider;

  @Before
  public void setUp() throws IOException {
    Path sourceDir = scratch.dir("/source");
    root = ArtifactRoot.asSourceRoot(Root.fromPath(sourceDir));
  }

  @SafeVarargs
  private void initializeOutputGroupInfo(Pair<String, NestedSet<Artifact>>... groups) {
    TreeMap<String, NestedSetBuilder<Artifact>> outputGroups = new TreeMap<>();
    for (var pair : groups) {
      outputGroups.put(pair.first, NestedSetBuilder.fromNestedSet(pair.second));
    }
    groupProvider = OutputGroupInfo.fromBuilders(outputGroups);
    ctx =
        new TopLevelArtifactContext(false, false, false, ImmutableSortedSet.copyOf(groupProvider));
  }

  @Test
  public void allOutputGroupsFiltered() {
    SourceArtifact group1FailedArtifact = newArtifact("g1_failed_output");
    SourceArtifact group2FailedArtifact = newArtifact("g2_failed_output");
    SourceArtifact group3FailedArtifact = newArtifact("g3_failed_output");
    SourceArtifact group1BuiltArtifact = newArtifact("g1_output");
    SourceArtifact group2BuiltArtifact = newArtifact("g2_output");
    SourceArtifact group3BuiltArtifact1 = newArtifact("g3_output1");
    SourceArtifact group3BuiltArtifact2 = newArtifact("g3_output2");
    ImmutableSet<Artifact> successfulArtifacts =
        ImmutableSet.of(
            group1BuiltArtifact, group2BuiltArtifact, group3BuiltArtifact1, group3BuiltArtifact2);
    // Arrange each output group with a different nested set structure.
    NestedSet<Artifact> group1Artifacts =
        NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(NestedSetBuilder.create(Order.STABLE_ORDER, group1BuiltArtifact))
            .addTransitive(NestedSetBuilder.create(Order.STABLE_ORDER, group1FailedArtifact))
            .build();
    NestedSet<Artifact> group2Artifacts =
        NestedSetBuilder.<Artifact>stableOrder()
            .add(group2BuiltArtifact)
            .addTransitive(NestedSetBuilder.create(Order.STABLE_ORDER, group2FailedArtifact))
            .build();
    NestedSet<Artifact> group3Artifacts1 =
        NestedSetBuilder.<Artifact>stableOrder()
            .add(group3BuiltArtifact1)
            .addTransitive(NestedSetBuilder.create(Order.STABLE_ORDER, group3FailedArtifact))
            .build();
    NestedSet<Artifact> group3Artifacts2 =
        NestedSetBuilder.<Artifact>stableOrder()
            .add(group3FailedArtifact)
            .addTransitive(NestedSetBuilder.create(Order.STABLE_ORDER, group3BuiltArtifact2))
            .build();
    NestedSet<Artifact> group3Artifacts =
        NestedSetBuilder.fromNestedSets(ImmutableList.of(group3Artifacts1, group3Artifacts2))
            .build();

    initializeOutputGroupInfo(
        Pair.of("g1", group1Artifacts),
        Pair.of("g2", group2Artifacts),
        Pair.of("g3", group3Artifacts));

    ArtifactsToBuild allArtifactsToBuild =
        TopLevelArtifactHelper.getAllArtifactsToBuild(groupProvider, null, ctx);
    ImmutableMap<String, ArtifactsInOutputGroup> outputGroups =
        allArtifactsToBuild.getAllArtifactsByOutputGroup();

    SuccessfulArtifactFilter filter = new SuccessfulArtifactFilter(successfulArtifacts);
    ImmutableMap<String, ArtifactsInOutputGroup> filteredOutputGroups =
        filter.filterArtifactsInOutputGroup(outputGroups);
    assertThat(filteredOutputGroups.get("g1").isIncomplete()).isTrue();
    assertThat(filteredOutputGroups.get("g2").isIncomplete()).isTrue();
    assertThat(filteredOutputGroups.get("g3").isIncomplete()).isTrue();
    Map<String, ImmutableSet<Artifact>> groupArtifacts =
        extractArtifactsByOutputGroup(filteredOutputGroups);
    assertThat(groupArtifacts.get("g1")).containsExactly(group1BuiltArtifact);
    assertThat(groupArtifacts.get("g2")).containsExactly(group2BuiltArtifact);
    assertThat(groupArtifacts.get("g3"))
        .containsExactly(group3BuiltArtifact1, group3BuiltArtifact2);
  }

  @Test
  public void emptyOutputGroupsNotReturned() {
    SourceArtifact group1FailedArtifact = newArtifact("g1_failed_output");
    SourceArtifact group2FailedArtifact = newArtifact("g2_failed_output");
    SourceArtifact group1BuiltArtifact = newArtifact("g1_output");
    ImmutableSet<Artifact> successfulArtifacts = ImmutableSet.of(group1BuiltArtifact);
    NestedSet<Artifact> group1Artifacts =
        NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(NestedSetBuilder.create(Order.STABLE_ORDER, group1BuiltArtifact))
            .addTransitive(NestedSetBuilder.create(Order.STABLE_ORDER, group1FailedArtifact))
            .build();
    NestedSet<Artifact> group2Artifacts =
        NestedSetBuilder.<Artifact>stableOrder().add(group2FailedArtifact).build();

    initializeOutputGroupInfo(Pair.of("g1", group1Artifacts), Pair.of("g2", group2Artifacts));

    ArtifactsToBuild allArtifactsToBuild =
        TopLevelArtifactHelper.getAllArtifactsToBuild(groupProvider, null, ctx);
    ImmutableMap<String, ArtifactsInOutputGroup> outputGroups =
        allArtifactsToBuild.getAllArtifactsByOutputGroup();

    SuccessfulArtifactFilter filter = new SuccessfulArtifactFilter(successfulArtifacts);
    ImmutableMap<String, ArtifactsInOutputGroup> filteredOutputGroups =
        filter.filterArtifactsInOutputGroup(outputGroups);
    assertThat(filteredOutputGroups.get("g1").isIncomplete()).isTrue();
    assertThat(filteredOutputGroups).containsKey("g1");
    assertThat(filteredOutputGroups).doesNotContainKey("g2");
  }

  @Test
  public void unfilteredNestedSetsReused() {
    SourceArtifact group1BuiltArtifact = newArtifact("output1");
    SourceArtifact group1BuiltArtifact2 = newArtifact("output2");
    SourceArtifact group1BuiltArtifact3 = newArtifact("output3");
    ImmutableSet<Artifact> successfulArtifacts =
        ImmutableSet.of(group1BuiltArtifact, group1BuiltArtifact2, group1BuiltArtifact3);
    NestedSet<Artifact> successfulArtifactSet =
        NestedSetBuilder.<Artifact>stableOrder()
            .add(group1BuiltArtifact)
            .addTransitive(
                NestedSetBuilder.create(
                    Order.STABLE_ORDER, group1BuiltArtifact2, group1BuiltArtifact3))
            .build();
    NestedSet<Artifact> setContainingSuccessfulSet1 =
        NestedSetBuilder.<Artifact>stableOrder()
            .add(group1BuiltArtifact)
            .addTransitive(successfulArtifactSet)
            .build();
    NestedSet<Artifact> setContainingSuccessfulSet2 =
        NestedSetBuilder.<Artifact>stableOrder()
            .add(group1BuiltArtifact)
            .addTransitive(successfulArtifactSet)
            .build();
    NestedSet<Artifact> outputGroup =
        NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(setContainingSuccessfulSet1)
            .addTransitive(setContainingSuccessfulSet2)
            .build();

    initializeOutputGroupInfo(Pair.of("out", outputGroup));

    ArtifactsToBuild allArtifactsToBuild =
        TopLevelArtifactHelper.getAllArtifactsToBuild(groupProvider, null, ctx);
    ImmutableMap<String, ArtifactsInOutputGroup> outputGroups =
        allArtifactsToBuild.getAllArtifactsByOutputGroup();

    SuccessfulArtifactFilter filter = new SuccessfulArtifactFilter(successfulArtifacts);
    ArtifactsInOutputGroup unfilteredArtifactsInOutputGroup = outputGroups.get("out");
    ArtifactsInOutputGroup filteredArtifactsInOutputGroup =
        filter.filterArtifactsInOutputGroup(outputGroups).get("out");
    assertThat(filteredArtifactsInOutputGroup.isIncomplete()).isFalse();
    assertThat(filteredArtifactsInOutputGroup).isSameInstanceAs(unfilteredArtifactsInOutputGroup);
  }

  @Test(timeout = 10_000)
  public void deeplyNestedSetFilteredQuickly() {
    SourceArtifact failedArtifact = newArtifact("failed_output");
    SourceArtifact builtArtifact = newArtifact("output");
    ImmutableSet<Artifact> successfulArtifacts = ImmutableSet.of(builtArtifact);
    // Arrange each output group with a different nested set structure.
    NestedSet<Artifact> baseSet =
        NestedSetBuilder.<Artifact>stableOrder().add(builtArtifact).add(failedArtifact).build();
    List<NestedSet<Artifact>> sets = new ArrayList<>();
    sets.add(baseSet);
    // Create a NestedSet DAG with ((500 * 499) / 2) nodes, but with only 500 unique nodes. It
    // should be feasible to filter this NestedSet using memoization in a small test and we should
    // timeout if we aren't using memoization.
    for (int i = 0; i < 500; i++) {
      NestedSetBuilder<Artifact> builder = NestedSetBuilder.stableOrder();
      builder.add(builtArtifact).add(failedArtifact);
      for (NestedSet<Artifact> set : sets) {
        builder.addTransitive(set);
      }
      sets.add(builder.build());
    }
    NestedSet<Artifact> maxSet = Iterables.getLast(sets);

    initializeOutputGroupInfo(Pair.of("group", maxSet));

    ArtifactsToBuild allArtifactsToBuild =
        TopLevelArtifactHelper.getAllArtifactsToBuild(groupProvider, null, ctx);
    ImmutableMap<String, ArtifactsInOutputGroup> outputGroups =
        allArtifactsToBuild.getAllArtifactsByOutputGroup();

    SuccessfulArtifactFilter filter = new SuccessfulArtifactFilter(successfulArtifacts);
    Map<String, ImmutableSet<Artifact>> groupArtifacts =
        extractArtifactsByOutputGroup(filter.filterArtifactsInOutputGroup(outputGroups));
    assertThat(groupArtifacts.get("group")).containsExactlyElementsIn(successfulArtifacts);
  }

  private SourceArtifact newArtifact(String name) {
    return new SourceArtifact(root, PathFragment.create(name), LabelArtifactOwner.NULL_OWNER);
  }

  private Map<String, ImmutableSet<Artifact>> extractArtifactsByOutputGroup(
      ImmutableMap<String, ArtifactsInOutputGroup> outputGroups) {
    Map<String, ImmutableSet<Artifact>> groupToDeclaredArtifacts = new HashMap<>();
    for (Map.Entry<String, ArtifactsInOutputGroup> entry : outputGroups.entrySet()) {
      groupToDeclaredArtifacts.put(entry.getKey(), entry.getValue().getArtifacts().toSet());
    }
    return groupToDeclaredArtifacts;
  }
}
