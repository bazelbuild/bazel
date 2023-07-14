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

import static com.google.devtools.build.lib.analysis.AspectResolutionHelpers.computePropagatingAspects;
import static com.google.devtools.build.lib.analysis.producers.DependencyError.isSecondErrorMoreImportant;
import static java.util.Arrays.asList;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.DependencyKind;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionCollector;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.build.skyframe.state.StateMachine;
import java.util.Collection;
import java.util.Map;

/**
 * Computes the full multimap of prerequisite values from a multimap of labels.
 *
 * <p>This class creates a child {@link DependencyProducer} for each ({@link DependencyKind}, {@link
 * Label}) multimap entry and collects the results. It outputs a multimap with the same entries,
 * replacing {@link Label} values with the corresponding computed {@link ConfiguredTargetAndData}
 * dependency values.
 */
public final class DependencyMapProducer implements StateMachine, DependencyProducer.ResultSink {
  /** Receiver for output of {@link DependencyMapProducer}. */
  public interface ResultSink extends TransitionCollector {
    void acceptDependencyMap(OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> value);

    void acceptDependencyMapError(DependencyError error);

    void acceptDependencyMapError(MissingEdgeError error);
  }

  // -------------------- Input --------------------
  private final PrerequisiteParameters parameters;
  private final OrderedSetMultimap<DependencyKind, Label> dependencyLabels;

  // -------------------- Output --------------------
  private final ResultSink sink;

  // -------------------- Internal State --------------------
  /**
   * This buffer receives results from child {@link DependencyProducer}s.
   *
   * <p>The indices break down the result by the following.
   *
   * <ol>
   *   <li>The entries of {@link #dependencyLabels}.
   *   <li>The configurations for that entry (more than one if there is a split transition).
   * </ol>
   *
   * <p>It would not be straightforward to replace this with a {@link OrderedSetMultimap} because
   * the child {@link DependencyProducer}s complete in an arbitrary order and the ordering of {@link
   * #dependencyLabels} must be preserved. Additionally, this is a fairly hot codepath and the
   * additional overhead of maps would consume significant resources.
   */
  private final ConfiguredTargetAndData[][] results;

  private DependencyError lastError;

  public DependencyMapProducer(
      PrerequisiteParameters parameters,
      OrderedSetMultimap<DependencyKind, Label> dependencyLabels,
      ResultSink sink) {
    this.parameters = parameters;
    this.dependencyLabels = dependencyLabels;
    this.sink = sink;
    this.results = new ConfiguredTargetAndData[dependencyLabels.size()][];
  }

  @Override
  public StateMachine step(Tasks tasks) {
    int index = 0;
    for (Map.Entry<DependencyKind, Collection<Label>> entry : dependencyLabels.asMap().entrySet()) {
      var kind = entry.getKey();
      ImmutableList<Aspect> aspects =
          computePropagatingAspects(kind, parameters.aspects(), parameters.associatedRule());
      for (var label : entry.getValue()) {
        tasks.enqueue(
            new DependencyProducer(
                parameters, kind, label, aspects, (DependencyProducer.ResultSink) this, index++));
      }
    }
    return this::buildAndEmitResult;
  }

  @Override
  public void acceptDependencyValues(int index, ConfiguredTargetAndData[] values) {
    results[index] = values;
  }

  @Override
  public void acceptDependencyError(DependencyError error) {
    emitErrorIfMostImportant(error);
  }

  @Override
  public void acceptDependencyError(MissingEdgeError error) {
    sink.acceptDependencyMapError(error);
  }

  @Override
  public void acceptTransition(
      DependencyKind kind, Label label, ConfigurationTransition transition) {
    sink.acceptTransition(kind, label, transition);
  }

  private StateMachine buildAndEmitResult(Tasks tasks) {
    if (lastError != null || parameters.transitiveState().hasRootCause()) {
      return DONE; // There was an error.
    }
    var output = new OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData>();
    int i = 0;
    for (DependencyKind kind : dependencyLabels.keys()) {
      ConfiguredTargetAndData[] result = results[i++];
      if (result == null) {
        return DONE; // There was an error.
      }
      // An empty `result` means the entry is skipped due to a missing exec group.
      if (result.length > 0) {
        output.putAll(kind, asList(result));
      }
    }
    sink.acceptDependencyMap(output);
    return DONE;
  }

  private void emitErrorIfMostImportant(DependencyError error) {
    if (lastError == null || isSecondErrorMoreImportant(lastError, error)) {
      lastError = error;
      sink.acceptDependencyMapError(error);
    }
  }
}
