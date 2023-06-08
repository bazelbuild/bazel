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
package com.google.devtools.build.lib.analysis.producers;

import static com.google.devtools.build.lib.analysis.AspectResolutionHelpers.computeAspectCollection;
import static com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData.SPLIT_DEP_ORDERING;

import com.google.devtools.build.lib.analysis.AspectCollection;
import com.google.devtools.build.lib.analysis.DuplicateException;
import com.google.devtools.build.lib.analysis.InconsistentAspectOrderException;
import com.google.devtools.build.lib.analysis.PartiallyResolvedDependency;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.build.skyframe.state.StateMachine;
import java.util.Arrays;
import java.util.Set;

/** Computes aspect values and merges them. */
// TODO(b/261521010): This temporary scaffolding exists to reduce the changelist size. Delete this.
public final class AspectMergerForTesting implements StateMachine {
  /** Receives output of {@link AspectMerger}. */
  public interface ResultSink {
    void acceptAspectMergerResult(
        OrderedSetMultimap<PartiallyResolvedDependency, ConfiguredTargetAndData> map);

    void acceptAspectMergerError(InconsistentAspectOrderException error);

    void acceptAspectMergerError(DuplicateException error);
  }

  // -------------------- Input --------------------
  private final OrderedSetMultimap<PartiallyResolvedDependency, ConfiguredTargetAndData>
      dependencies;

  // -------------------- Output --------------------
  private final ResultSink sink;

  // -------------------- Internal State --------------------
  /**
   * Staging area for output, aligning with {@link #dependencies}.
   *
   * <p>While performance doesn't matter in this test code, it needs to fit interfaces that are used
   * in production code. ConfiguredTargetFunction is highly-cpu constrained and intensive.
   *
   * <p>Using indices and depending on the stable iteration order of `dependencies` saves us
   * allocation, GC, scanning and hashing costs that would be associated with constructing a
   * parallel map with the same keys.
   *
   * <ol>
   *   <li>The first index corresponds with {@link PartiallyResolvedDependency} key of {@link
   *       #dependencies}.
   *   <li>The second index corresponds with the multiple entries per key.
   * </ol>
   */
  private final ConfiguredTargetAndData[][] result;

  private boolean hasError = false;

  public AspectMergerForTesting(
      OrderedSetMultimap<PartiallyResolvedDependency, ConfiguredTargetAndData> dependencies,
      ResultSink sink) {
    this.dependencies = dependencies;
    this.sink = sink;
    this.result = new ConfiguredTargetAndData[dependencies.keySet().size()][];
  }

  @Override
  public StateMachine step(Tasks tasks, ExtendedEventHandler listener) {
    int keyIndex = 0;
    for (PartiallyResolvedDependency key : dependencies.keySet()) {
      Set<ConfiguredTargetAndData> targets = dependencies.get(key);

      AspectCollection aspects;
      try {
        // Using any target to calculate aspects is fine because the filtering is based on Target
        // properties and all values for a given key share a common one.
        aspects = computeAspectCollection(targets.iterator().next(), key.getPropagatingAspects());
      } catch (InconsistentAspectOrderException e) {
        sink.acceptAspectMergerError(e);
        return DONE;
      }

      var targetResult = new ConfiguredTargetAndData[targets.size()];
      result[keyIndex] = targetResult;

      var mergedTargetSink =
          new ConfiguredAspectProducer.ResultSink() {
            @Override
            public void acceptConfiguredAspectMergedTarget(
                int outputIndex, ConfiguredTargetAndData mergedTarget) {
              targetResult[outputIndex] = mergedTarget;
            }

            @Override
            public void acceptConfiguredAspectError(DuplicateException error) {
              hasError = true;
              sink.acceptAspectMergerError(error);
            }
          };

      int targetIndex = 0;
      for (ConfiguredTargetAndData target : targets) {
        tasks.enqueue(
            new ConfiguredAspectProducer(
                aspects, target, mergedTargetSink, targetIndex, /* transitivePackages= */ null));
        targetIndex++;
      }
      keyIndex++;
    }
    return this::emitResult;
  }

  private StateMachine emitResult(Tasks tasks, ExtendedEventHandler listener) {
    if (hasError) {
      return DONE;
    }

    var mergedDependencies =
        OrderedSetMultimap.<PartiallyResolvedDependency, ConfiguredTargetAndData>create();
    int keyIndex = 0;
    for (PartiallyResolvedDependency key : dependencies.keySet()) {
      ConfiguredTargetAndData[] targets = result[keyIndex];
      if (targets.length > 1) {
        Arrays.sort(targets, SPLIT_DEP_ORDERING);
      }
      mergedDependencies.putAll(key, Arrays.asList(targets));
      ++keyIndex;
    }
    sink.acceptAspectMergerResult(mergedDependencies);
    return DONE;
  }
}
