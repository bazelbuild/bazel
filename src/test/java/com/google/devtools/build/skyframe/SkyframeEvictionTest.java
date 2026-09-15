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
package com.google.devtools.build.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.events.StoredEventHandler;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * The place where eventual tests for Skyframe eviction will live.
 *
 * <p>For now, since eviction is not implemented yet, it's just a fuzz test. Eventually, once
 * eviction is there, we'll add that to the mix of events that can happen.
 *
 * <p>We construct a Skyframe graph of depth {@code LAYERS} with {@code NODES_PER_LAYER} nodes per
 * layer. Each node has a randomly chosen dependency structure (see {@link #getBatches(LayerKey)})
 * with each edge pointing to a random node in one of the layers below.
 *
 * <p>The value of each leaf node is {@code <node id>_<generation>} where "generation" is
 * incremented on each invalidation. The value of each node is a set that contains the union of the
 * sets in its direct deps.
 */
@RunWith(JUnit4.class)
public final class SkyframeEvictionTest {

  // Number of layers in the test Skyframe graph
  private static final int LAYERS = 50;

  // Number of nodes in each layer
  private static final int NODES_PER_LAYER = 100;

  // Random seed (for reproducibility)
  private static final long BASE_SEED = 12345L;

  private static final SkyFunctionName LAYER_NODE_TYPE =
      SkyFunctionName.createNonHermetic("LAYER_NODE");

  private record LayerKey(int layer, int nodeNumber) implements SkyKey {
    @Override
    public SkyFunctionName functionName() {
      return LAYER_NODE_TYPE;
    }
  }

  private record LayerValue(ImmutableSortedSet<String> values) implements SkyValue {}

  // Returns how many SkyKeys the given SkyValue should depend on. Each item in the returned list
  // represents a dependency group and the integer is the number of random dependencies in that
  // group.
  private static ImmutableList<ImmutableList<LayerKey>> getBatches(LayerKey key) {
    Random random = new Random(BASE_SEED ^ ((long) key.layer() << 32) ^ key.nodeNumber());
    int roll = random.nextInt(100);
    ImmutableList<Integer> batchSizes;
    if (roll < 5) {
      batchSizes = ImmutableList.of(); // 1. No dependencies (5%)
    } else if (roll < 8) {
      batchSizes = ImmutableList.of(10); // 2. 10 dependencies in a single restart (3%)
    } else if (roll < 10) {
      batchSizes = ImmutableList.of(1, 1, 1, 1, 1, 1, 1, 1, 1, 1); // 3. 10 deps, 1 per restart (2%)
    } else if (roll < 40) {
      batchSizes = ImmutableList.of(1); // 4. One dependency (30%)
    } else if (roll < 65) {
      batchSizes = ImmutableList.of(2); // 5. Two dependencies in a single restart (25%)
    } else if (roll < 90) {
      batchSizes = ImmutableList.of(1, 1); // 6. Two dependencies, each in its own restart (25%)
    } else if (roll < 95) {
      batchSizes = ImmutableList.of(1, 1, 1); // 7. Three dependencies, 1 in each of 3 restarts (5%)
    } else {
      batchSizes = ImmutableList.of(2, 2); // 8. Four dependencies, 2 in each of 2 restarts (5%)
    }

    ImmutableList.Builder<ImmutableList<LayerKey>> batches = ImmutableList.builder();
    for (int size : batchSizes) {
      ImmutableList.Builder<LayerKey> batch = ImmutableList.builder();
      for (int i = 0; i < size; i++) {
        int targetLayer = key.layer() + 1 + random.nextInt(LAYERS - key.layer());
        batch.add(new LayerKey(targetLayer, random.nextInt(NODES_PER_LAYER)));
      }
      batches.add(batch.build());
    }
    return batches.build();
  }

  private final class LayerFunction implements SkyFunction {
    @Override
    @Nullable
    public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
      LayerKey key = (LayerKey) skyKey;
      if (key.layer() == LAYERS) {
        // Leaf node. The value is <node id>_<generation>.
        return new LayerValue(
            ImmutableSortedSet.of(key.nodeNumber() + "_" + leafVersions[key.nodeNumber()]));
      }

      // Non-leaf node. The value is the union of the strings in the direct deps.
      ImmutableSortedSet.Builder<String> union = ImmutableSortedSet.naturalOrder();
      for (ImmutableList<LayerKey> batch : getBatches(key)) {
        SkyframeLookupResult values = env.getValuesAndExceptions(batch);
        if (env.valuesMissing()) {
          return null;
        }
        for (LayerKey dep : batch) {
          LayerValue depValue = (LayerValue) values.get(dep);
          union.addAll(depValue.values());
        }
      }
      return new LayerValue(union.build());
    }
  }

  private static List<LayerKey> getTopLevelKeys() {
    List<LayerKey> keys = new ArrayList<>();
    for (int node = 0; node < NODES_PER_LAYER; node++) {
      keys.add(new LayerKey(1, node));
    }
    return keys;
  }

  private Map<LayerKey, ImmutableSortedSet<String>> computeExpectedValues() {
    // This is basically a non-incremental algorithm to do the same thing Skyframe should be doing:
    // Walk up the graph from leaf nodes and compute the unions all the way to the top level.
    Map<LayerKey, ImmutableSortedSet<String>> expectedValues = new HashMap<>();
    for (int layer = LAYERS; layer >= 1; layer--) {
      for (int node = 0; node < NODES_PER_LAYER; node++) {
        LayerKey key = new LayerKey(layer, node);
        if (layer == LAYERS) {
          expectedValues.put(key, ImmutableSortedSet.of(node + "_" + leafVersions[node]));
        } else {
          ImmutableSortedSet.Builder<String> union = ImmutableSortedSet.naturalOrder();
          for (ImmutableList<LayerKey> batch : getBatches(key)) {
            for (LayerKey dep : batch) {
              union.addAll(expectedValues.get(dep));
            }
          }
          expectedValues.put(key, union.build());
        }
      }
    }
    return expectedValues;
  }

  private final SequencedRecordingDifferencer differencer = new SequencedRecordingDifferencer();
  private InMemoryMemoizingEvaluator evaluator;
  private EvaluationContext evaluationContext;

  // The generation for each leaf node. Leaf node X in generation Y will yield the value "X_Y".
  private final int[] leafVersions = new int[NODES_PER_LAYER];

  @Before
  public void setUp() {
    ImmutableMap<SkyFunctionName, SkyFunction> functions =
        ImmutableMap.of(LAYER_NODE_TYPE, new LayerFunction());
    evaluator =
        new InMemoryMemoizingEvaluator(
            functions,
            differencer,
            new TrackingProgressReceiver(/* checkEvaluationResults= */ true),
            GraphInconsistencyReceiver.THROWING,
            EventFilter.FULL_STORAGE,
            new EmittedEventState(),
            /* keepEdges= */ true,
            /* usePooledInterning= */ true);
    evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setParallelism(100)
            .setEventHandler(new StoredEventHandler())
            .build();
  }

  private void evaluateAndAssert() throws Exception {
    List<LayerKey> topLevelKeys = getTopLevelKeys();
    Map<LayerKey, ImmutableSortedSet<String>> expectedValues = computeExpectedValues();
    EvaluationResult<LayerValue> result = evaluator.evaluate(topLevelKeys, evaluationContext);
    assertThat(result.hasError()).isFalse();
    for (LayerKey key : topLevelKeys) {
      LayerValue value = result.get(key);
      assertThat(value).isNotNull();
      assertThat(value.values()).isEqualTo(expectedValues.get(key));
    }
  }

  @Test
  public void randomGraphEvaluationWithInvalidation() throws Exception {
    evaluateAndAssert();

    // Make the seed different from the seed of the per-SkyFunction random generator in
    // getBatches().
    Random invalidationRandom = new Random(BASE_SEED - 1);
    for (int iteration = 0; iteration < 100; iteration++) {
      int numToInvalidate = invalidationRandom.nextInt(21);
      Set<SkyKey> keysToInvalidate = new HashSet<>();
      while (keysToInvalidate.size() < numToInvalidate) {
        int leafNode = invalidationRandom.nextInt(NODES_PER_LAYER);
        if (keysToInvalidate.add(new LayerKey(LAYERS, leafNode))) {
          leafVersions[leafNode]++;
        }
      }
      differencer.invalidate(keysToInvalidate);
      evaluateAndAssert();
    }
  }
}
