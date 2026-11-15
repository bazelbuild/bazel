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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.ProviderCollection;
import com.google.devtools.build.lib.analysis.RuleConfiguredObjectValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.buildtool.TransitiveTargetCounter.Counts;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.InMemoryNodeEntry;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueWithMetadata;
import com.google.devtools.build.skyframe.Version;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link TransitiveTargetCounter}. */
@RunWith(JUnit4.class)
public final class TransitiveTargetCounterTest {

  private static final BuildConfigurationKey CONFIG_KEY =
      BuildConfigurationKey.create(BuildOptions.builder().build());

  private static final BuildConfigurationKey TRANSITIONED_CONFIG_KEY =
      BuildConfigurationKey.create(
          BuildOptions.builder()
              .addStarlarkOption(Label.parseCanonicalUnchecked("//pkg:flag"), "transitioned")
              .build());

  private static final SkyValue NON_RULE_VALUE = new SkyValue() {};

  private static final SkyFunctionName OTHER_FUNCTION =
      SkyFunctionName.createHermetic("OTHER_TEST_FUNCTION");

  private static final AspectClass TEST_ASPECT_CLASS = () -> "test_aspect";

  private InMemoryGraph graph;

  @Before
  public void setUp() {
    graph = InMemoryGraph.create();
  }

  @Test
  public void count_emptyRoots_returnsEmptyMap() throws Exception {
    assertThat(TransitiveTargetCounter.count(graph, ImmutableSet.of())).isEmpty();
  }

  @Test
  public void count_singleRoot_countsSelfAndActions() throws Exception {
    ConfiguredTargetKey root = configuredTargetKey("//pkg:root");
    addDoneNode(graph, root, ruleValue(/* actionCount= */ 3));

    ImmutableMap<ConfiguredTargetKey, Counts> counts =
        TransitiveTargetCounter.count(graph, ImmutableSet.of(root));

    assertThat(counts).containsExactly(root, new Counts(1, 3));
  }

  @Test
  public void count_diamondDag_countsSharedDependencyOnce() throws Exception {
    // root (1 action) -> {left (2 actions), right (3 actions)} -> leaf (5 actions).
    // Also attach non-configurable and non-ActionLookupKey Skyframe deps to verify they are
    // ignored.
    ConfiguredTargetKey leaf = configuredTargetKey("//pkg:leaf");
    ConfiguredTargetKey left = configuredTargetKey("//pkg:left");
    ConfiguredTargetKey right = configuredTargetKey("//pkg:right");
    ConfiguredTargetKey root = configuredTargetKey("//pkg:root");
    ConfiguredTargetKey sourceFile = unconfiguredTargetKey("//pkg:src.cc");
    SkyKey otherKey = () -> OTHER_FUNCTION;

    addDoneNode(graph, sourceFile, ruleValue(/* actionCount= */ 99));
    addDoneNode(graph, otherKey, ruleValue(/* actionCount= */ 99));
    addDoneNode(graph, leaf, ruleValue(/* actionCount= */ 5), sourceFile, otherKey);
    addDoneNode(graph, left, ruleValue(/* actionCount= */ 2), leaf);
    addDoneNode(graph, right, ruleValue(/* actionCount= */ 3), leaf);
    addDoneNodeWithDepGroups(
        graph,
        root,
        ruleValue(/* actionCount= */ 1),
        ImmutableList.of(ImmutableList.of(left, sourceFile), ImmutableList.of(right, otherKey)));

    ImmutableMap<ConfiguredTargetKey, Counts> counts =
        TransitiveTargetCounter.count(graph, ImmutableSet.of(root));

    assertThat(counts).containsExactly(root, new Counts(4, 1 + 2 + 3 + 5));
  }

  @Test
  public void count_multipleRootsAndRootDependingOnRoot_countsEachTransitiveClosure()
      throws Exception {
    // r1 (1) -> r2 (2) -> shared (4)
    // r3 (8) -> shared (4)
    ConfiguredTargetKey shared = configuredTargetKey("//pkg:shared");
    ConfiguredTargetKey r2 = configuredTargetKey("//pkg:r2");
    ConfiguredTargetKey r1 = configuredTargetKey("//pkg:r1");
    ConfiguredTargetKey r3 = configuredTargetKey("//pkg:r3");

    addDoneNode(graph, shared, ruleValue(/* actionCount= */ 4));
    addDoneNode(graph, r2, ruleValue(/* actionCount= */ 2), shared);
    addDoneNode(graph, r1, ruleValue(/* actionCount= */ 1), r2);
    addDoneNode(graph, r3, ruleValue(/* actionCount= */ 8), shared);

    ImmutableMap<ConfiguredTargetKey, Counts> counts =
        TransitiveTargetCounter.count(graph, ImmutableSet.of(r1, r2, r3));

    assertThat(counts)
        .containsExactly(
            r1, new Counts(3, 1 + 2 + 4),
            r2, new Counts(2, 2 + 4),
            r3, new Counts(2, 8 + 4));
  }

  @Test
  public void count_aspectsAndNonRuleIntermediateNodes_countsRulesAndAspectsOnly()
      throws Exception {
    // root (rule, 1 action)
    //   -> aliasLike (non-rule ConfiguredTargetValue, e.g. AliasConfiguredTarget)
    //        -> realDep (rule, 2 actions)
    //   -> aspectOnDep (aspect, 3 actions)
    //        -> aspectOnlyDep (rule, 4 actions)
    //        -> realDep (shared with aliasLike, counted once!)
    ConfiguredTargetKey realDep = configuredTargetKey("//pkg:real_dep");
    ConfiguredTargetKey aspectOnlyDep = configuredTargetKey("//pkg:aspect_only_dep");
    ConfiguredTargetKey aliasLike = configuredTargetKey("//pkg:alias_like");
    AspectKey aspectOnDep = aspectKey(realDep);
    ConfiguredTargetKey root = configuredTargetKey("//pkg:root");
    ConfiguredTargetKey nonRuleRoot = configuredTargetKey("//pkg:output_file");

    addDoneNode(graph, realDep, ruleValue(/* actionCount= */ 2));
    addDoneNode(graph, aspectOnlyDep, ruleValue(/* actionCount= */ 4));
    addDoneNode(graph, aliasLike, NON_RULE_VALUE, realDep);
    addDoneNode(graph, aspectOnDep, ruleValue(/* actionCount= */ 3), realDep, aspectOnlyDep);
    addDoneNode(graph, root, ruleValue(/* actionCount= */ 1), aliasLike, aspectOnDep);
    addDoneNode(graph, nonRuleRoot, NON_RULE_VALUE, realDep);

    ImmutableMap<ConfiguredTargetKey, Counts> counts =
        TransitiveTargetCounter.count(graph, ImmutableSet.of(root, nonRuleRoot));

    // root reaches: root (1), realDep (2), aspectOnDep (3), aspectOnlyDep (4) -> 4 nodes, 10
    // actions (aliasLike is traversed through without incrementing targetCount).
    // nonRuleRoot reaches: realDep (2) -> 1 node, 2 actions.
    assertThat(counts)
        .containsExactly(
            root, new Counts(4, 1 + 2 + 3 + 4),
            nonRuleRoot, new Counts(1, 2));
  }

  @Test
  public void count_delegatedConfiguredTarget_countedOnce() throws Exception {
    // When a rule transition changes a target's configuration from CONFIG_KEY to
    // TRANSITIONED_CONFIG_KEY, preTransitionKey delegates to postTransitionKey and holds the exact
    // same ConfiguredTargetValue (whose ConfiguredTarget has TRANSITIONED_CONFIG_KEY).
    ConfiguredTargetKey leaf =
        ConfiguredTargetKey.builder()
            .setLabel(Label.parseCanonicalUnchecked("//pkg:leaf"))
            .setConfigurationKey(TRANSITIONED_CONFIG_KEY)
            .build();
    ConfiguredTargetKey preTransitionKey = configuredTargetKey("//pkg:transitioned");
    ConfiguredTargetKey postTransitionKey =
        ConfiguredTargetKey.builder()
            .setLabel(Label.parseCanonicalUnchecked("//pkg:transitioned"))
            .setConfigurationKey(TRANSITIONED_CONFIG_KEY)
            .build();
    ConfiguredTargetKey root = configuredTargetKey("//pkg:root");

    StubRuleConfiguredTargetValue transitionedValue =
        new StubRuleConfiguredTargetValue(/* actionCount= */ 3, TRANSITIONED_CONFIG_KEY);
    addDoneNode(
        graph,
        leaf,
        new StubRuleConfiguredTargetValue(/* actionCount= */ 2, TRANSITIONED_CONFIG_KEY));
    addDoneNode(graph, postTransitionKey, transitionedValue, leaf);
    addDoneNode(graph, preTransitionKey, transitionedValue, postTransitionKey);
    addDoneNode(
        graph,
        root,
        new StubRuleConfiguredTargetValue(/* actionCount= */ 1, CONFIG_KEY),
        preTransitionKey);

    ImmutableMap<ConfiguredTargetKey, Counts> counts =
        TransitiveTargetCounter.count(
            graph, ImmutableSet.of(root, preTransitionKey, postTransitionKey));

    assertThat(counts)
        .containsExactly(
            root, new Counts(3, 1 + 3 + 2),
            preTransitionKey, new Counts(2, 3 + 2),
            postTransitionKey, new Counts(2, 3 + 2));
  }

  @Test
  public void count_unconfiguredMissingAndUndoneNodes_handledGracefully() throws Exception {
    ConfiguredTargetKey unconfiguredRoot = unconfiguredTargetKey("//pkg:input.txt");
    ConfiguredTargetKey missingRoot = configuredTargetKey("//pkg:missing");
    ConfiguredTargetKey undoneRoot = configuredTargetKey("//pkg:undone");

    // Create `undoneRoot` in the graph without marking it done.
    InMemoryNodeEntry undoneEntry =
        (InMemoryNodeEntry)
            graph
                .createIfAbsentBatch(null, Reason.OTHER, ImmutableList.of(undoneRoot))
                .get(undoneRoot);
    undoneEntry.addReverseDepAndCheckIfDone(null);

    assertThat(TransitiveTargetCounter.count(graph, ImmutableSet.of(unconfiguredRoot)))
        .containsExactly(unconfiguredRoot, new Counts(0, 0));

    ImmutableMap<ConfiguredTargetKey, Counts> counts =
        TransitiveTargetCounter.count(
            graph, ImmutableSet.of(unconfiguredRoot, missingRoot, undoneRoot));

    assertThat(counts)
        .containsExactly(
            unconfiguredRoot, new Counts(0, 0),
            missingRoot, new Counts(0, 0),
            undoneRoot, new Counts(0, 0));
  }

  @Test
  public void count_doneErrorAndCycleNodes_ignoredWithoutFollowingDeps() throws Exception {
    // Under --keep_going, failed and cycle nodes are marked done with ErrorInfoValue
    // (entry.isDone() == true, entry.getValue() == null) while retaining their direct dep edges.
    ConfiguredTargetKey validDep = configuredTargetKey("//pkg:valid_dep");
    ConfiguredTargetKey cycleA = configuredTargetKey("//pkg:cycle_a");
    ConfiguredTargetKey cycleB = configuredTargetKey("//pkg:cycle_b");
    ConfiguredTargetKey validRoot = configuredTargetKey("//pkg:valid_root");
    SkyValue errorValue =
        ValueWithMetadata.error(
            ErrorInfo.fromCycle(CycleInfo.cycleInfoNoDetails()),
            NestedSetBuilder.emptySet(Order.STABLE_ORDER));

    addDoneNode(graph, validDep, ruleValue(/* actionCount= */ 4));
    addDoneNode(graph, cycleA, errorValue, cycleB, validDep);
    addDoneNode(graph, cycleB, errorValue, cycleA);
    addDoneNode(graph, validRoot, ruleValue(/* actionCount= */ 2), cycleA, validDep);

    ImmutableMap<ConfiguredTargetKey, Counts> counts =
        TransitiveTargetCounter.count(graph, ImmutableSet.of(validRoot, cycleA));

    assertThat(counts)
        .containsExactly(
            validRoot, new Counts(2, 2 + 4),
            cycleA, new Counts(0, 0));
  }

  @Test
  public void count_edgelessGraph_countsRootWithoutFollowingDeps() throws Exception {
    InMemoryGraph edgelessGraph = InMemoryGraph.createEdgeless(/* usePooledInterning= */ false);
    ConfiguredTargetKey dep = configuredTargetKey("//pkg:dep");
    ConfiguredTargetKey root = configuredTargetKey("//pkg:root");
    addDoneNode(edgelessGraph, dep, ruleValue(/* actionCount= */ 5));
    addDoneNode(edgelessGraph, root, ruleValue(/* actionCount= */ 2), dep);

    ImmutableMap<ConfiguredTargetKey, Counts> counts =
        TransitiveTargetCounter.count(edgelessGraph, ImmutableSet.of(root));

    assertThat(counts).containsExactly(root, new Counts(1, 2));
  }

  @Test
  public void count_multiPassPropagation_matchesExpectedCountsAcrossPasses() throws Exception {
    // Force 1 word (64 roots) per pass by passing maxMaskBytes = 0, with 150 roots spanning 3
    // passes ([0, 64), [64, 128), [128, 150)), including cross-pass root-to-root edges in both
    // directions.
    ConfiguredTargetKey sharedLeaf = configuredTargetKey("//pkg:shared_leaf");
    addDoneNode(graph, sharedLeaf, ruleValue(/* actionCount= */ 3));

    int numRoots = 150;
    List<ConfiguredTargetKey> roots = new ArrayList<>(numRoots);
    for (int i = 0; i < numRoots; i++) {
      roots.add(configuredTargetKey("//pkg:root_" + i));
    }

    // root_0 (pass 0) -> root_140 (pass 2) -> sharedLeaf
    // root_145 (pass 2) -> root_10 (pass 0) -> sharedLeaf
    addDoneNode(graph, roots.get(140), ruleValue(/* actionCount= */ 5), sharedLeaf);
    addDoneNode(graph, roots.get(0), ruleValue(/* actionCount= */ 2), roots.get(140));
    addDoneNode(graph, roots.get(10), ruleValue(/* actionCount= */ 7), sharedLeaf);
    addDoneNode(graph, roots.get(145), ruleValue(/* actionCount= */ 11), roots.get(10));

    for (int i = 0; i < numRoots; i++) {
      if (i == 0 || i == 10 || i == 140 || i == 145) {
        continue;
      }
      addDoneNode(graph, roots.get(i), ruleValue(/* actionCount= */ 1), sharedLeaf);
    }

    ImmutableMap<ConfiguredTargetKey, Counts> counts =
        TransitiveTargetCounter.count(graph, ImmutableSet.copyOf(roots), /* maxMaskBytes= */ 0);

    assertThat(counts).hasSize(numRoots);
    assertThat(counts.get(roots.get(0))).isEqualTo(new Counts(3, 2 + 5 + 3));
    assertThat(counts.get(roots.get(140))).isEqualTo(new Counts(2, 5 + 3));
    assertThat(counts.get(roots.get(145))).isEqualTo(new Counts(3, 11 + 7 + 3));
    assertThat(counts.get(roots.get(10))).isEqualTo(new Counts(2, 7 + 3));
    for (int i = 0; i < numRoots; i++) {
      if (i == 0 || i == 10 || i == 140 || i == 145) {
        continue;
      }
      assertThat(counts.get(roots.get(i))).isEqualTo(new Counts(2, 1 + 3));
    }
  }

  @Test
  public void count_executesOnDedicatedPoolAndNotCommonPool() throws Exception {
    Set<Thread> observedThreads = ConcurrentHashMap.newKeySet();
    // Build > 1024 nodes (> FOR_EACH_PARALLELISM_THRESHOLD) so ConcurrentHashMap.forEachValue
    // also forks subtasks, and record every thread that calls getActions().
    int chainLength = 1500;
    ConfiguredTargetKey prev = null;
    for (int i = 0; i < chainLength; i++) {
      ConfiguredTargetKey key = configuredTargetKey("//pkg:n_" + i);
      RuleConfiguredObjectValue value =
          new StubRuleConfiguredObjectValue(/* actionCount= */ 1) {
            @Override
            public ImmutableList<ActionAnalysisMetadata> getActions() {
              observedThreads.add(Thread.currentThread());
              return super.getActions();
            }
          };
      if (prev == null) {
        addDoneNode(graph, key, value);
      } else {
        addDoneNode(graph, key, value, prev);
      }
      prev = key;
    }

    long commonPoolStealsBefore = ForkJoinPool.commonPool().getStealCount();
    ImmutableMap<ConfiguredTargetKey, Counts> counts =
        TransitiveTargetCounter.count(graph, ImmutableSet.of(prev));
    long commonPoolStealsAfter = ForkJoinPool.commonPool().getStealCount();

    assertThat(counts).containsExactly(prev, new Counts(chainLength, chainLength));
    assertThat(observedThreads).isNotEmpty();
    for (Thread thread : observedThreads) {
      assertThat(thread.getName()).startsWith("transitive-target-counter-");
    }
    assertThat(commonPoolStealsAfter).isEqualTo(commonPoolStealsBefore);
  }

  private static ConfiguredTargetKey configuredTargetKey(String label) {
    return ConfiguredTargetKey.builder()
        .setLabel(Label.parseCanonicalUnchecked(label))
        .setConfigurationKey(CONFIG_KEY)
        .build();
  }

  private static ConfiguredTargetKey unconfiguredTargetKey(String label) {
    return ConfiguredTargetKey.builder().setLabel(Label.parseCanonicalUnchecked(label)).build();
  }

  private static AspectKey aspectKey(ConfiguredTargetKey baseKey) {
    return AspectKeyCreator.createAspectKey(
        AspectDescriptor.of(TEST_ASPECT_CLASS, AspectParameters.EMPTY), baseKey);
  }

  private static RuleConfiguredObjectValue ruleValue(int actionCount) {
    return new StubRuleConfiguredObjectValue(actionCount);
  }

  private static void addDoneNode(
      InMemoryGraph targetGraph, SkyKey key, SkyValue value, SkyKey... directDeps)
      throws InterruptedException {
    addDoneNodeWithDepGroups(
        targetGraph, key, value, ImmutableList.of(ImmutableList.copyOf(directDeps)));
  }

  private static void addDoneNodeWithDepGroups(
      InMemoryGraph targetGraph,
      SkyKey key,
      SkyValue value,
      ImmutableList<ImmutableList<SkyKey>> depGroups)
      throws InterruptedException {
    InMemoryNodeEntry entry =
        (InMemoryNodeEntry)
            targetGraph.createIfAbsentBatch(null, Reason.OTHER, ImmutableList.of(key)).get(key);
    entry.addReverseDepAndCheckIfDone(null);
    entry.markRebuilding();
    for (ImmutableList<SkyKey> group : depGroups) {
      if (group.size() == 1) {
        SkyKey dep = group.get(0);
        entry.addSingletonTemporaryDirectDep(dep);
        entry.signalDep(Version.constant(), dep);
      } else if (!group.isEmpty()) {
        entry.addTemporaryDirectDepGroup(group);
        for (SkyKey dep : group) {
          entry.signalDep(Version.constant(), dep);
        }
      }
    }
    entry.setValue(value, Version.constant(), /* maxTransitiveSourceVersion= */ null);
  }

  private static class StubRuleConfiguredObjectValue implements RuleConfiguredObjectValue {
    private final ImmutableList<ActionAnalysisMetadata> actions;

    StubRuleConfiguredObjectValue(int actionCount) {
      ImmutableList.Builder<ActionAnalysisMetadata> builder =
          ImmutableList.builderWithExpectedSize(actionCount);
      for (int i = 0; i < actionCount; i++) {
        builder.add(new ActionsTestUtil.NullAction());
      }
      this.actions = builder.build();
    }

    @Override
    public ImmutableList<ActionAnalysisMetadata> getActions() {
      return actions;
    }

    @Override
    public ProviderCollection getConfiguredObject() {
      throw new UnsupportedOperationException();
    }

    @Override
    public NestedSet<Package.Metadata> getTransitivePackages() {
      throw new UnsupportedOperationException();
    }
  }

  private static final class StubRuleConfiguredTargetValue extends StubRuleConfiguredObjectValue
      implements ConfiguredTargetValue {
    private final ConfiguredTarget configuredTarget;

    StubRuleConfiguredTargetValue(int actionCount, BuildConfigurationKey configKey) {
      super(actionCount);
      this.configuredTarget = mock(ConfiguredTarget.class);
      when(this.configuredTarget.getConfigurationKey()).thenReturn(configKey);
    }

    @Override
    public ConfiguredTarget getConfiguredTarget() {
      return configuredTarget;
    }

    @Override
    public ConfiguredTarget getConfiguredObject() {
      return configuredTarget;
    }

    @Override
    public void clear(boolean clearEverything) {
      throw new UnsupportedOperationException();
    }
  }
}
