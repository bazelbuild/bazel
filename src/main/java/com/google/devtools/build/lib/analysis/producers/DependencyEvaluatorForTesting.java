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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.analysis.BaseDependencySpecification;
import com.google.devtools.build.lib.analysis.InvalidVisibilityDependencyException;
import com.google.devtools.build.lib.analysis.PartiallyResolvedDependency;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.DependencyEvaluationException;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.skyframe.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ConfiguredValueCreationException;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.build.skyframe.state.StateMachine;
import com.google.devtools.build.skyframe.state.StateMachine.Tasks;
import java.util.Arrays;
import java.util.List;

/** Computes aspect values and merges them. */
// TODO(b/261521010): This temporary scaffolding exists to reduce the changelist size. Delete this.
public final class DependencyEvaluatorForTesting implements StateMachine {
  /** Receives output of {@link DependencyEvaluatorForTesting}. */
  public interface ResultSink {
    void acceptDependencyEvaluatorResult(
        OrderedSetMultimap<PartiallyResolvedDependency, ConfiguredTargetAndData> result);

    void acceptDependencyEvaluatorError(InvalidVisibilityDependencyException error);

    void acceptDependencyEvaluatorError(ConfiguredValueCreationException error);

    void acceptDependencyEvaluatorError(DependencyEvaluationException error);
  }

  // -------------------- Input --------------------
  private final PrerequisiteParameters parameters;
  private final ImmutableSet<PartiallyResolvedDependency> dependencies;
  private final ListMultimap<BaseDependencySpecification, BuildConfigurationValue> configurations;

  // -------------------- Output --------------------
  private final ResultSink sink;

  // -------------------- Internal State --------------------
  private final OrderedSetMultimap<PartiallyResolvedDependency, ConfiguredTargetAndData>
      prerequisiteValues = OrderedSetMultimap.create();
  private boolean hasError = false;

  public DependencyEvaluatorForTesting(
      PrerequisiteParameters parameters,
      ImmutableSet<PartiallyResolvedDependency> dependencies,
      ListMultimap<BaseDependencySpecification, BuildConfigurationValue> configurations,
      ResultSink sink) {
    this.parameters = parameters;
    this.dependencies = dependencies;
    this.configurations = configurations;
    this.sink = sink;
  }

  @Override
  public StateMachine step(Tasks tasks, ExtendedEventHandler listener) {
    for (PartiallyResolvedDependency key : dependencies) {
      if (!configurations.containsKey(key)) {
        // If we couldn't compute a configuration for this target, the target was in error (e.g.
        // it couldn't be loaded). Exclude it from the results.
        continue;
      }

      List<BuildConfigurationValue> depConfigs = configurations.get(key);
      // Constructs a fake attributeConfiguration for testing based on how many configurations there
      // are. This inference is only needed in testing.
      AttributeConfiguration attributeConfiguration;
      if (depConfigs.size() == 1) {
        BuildConfigurationValue configuration = depConfigs.get(0);
        if (configuration == null) {
          // If the configuration is null, it was explicitly added as the result of a
          // `NullTransition` by SkyframeExecutor.getConfigurations. The only way this can happen
          // for a PartiallyResolvedDependency (which is determined without inspecting child
          // targets) is for visibility dependencies.
          attributeConfiguration = AttributeConfiguration.ofVisibility();
        } else {
          attributeConfiguration = AttributeConfiguration.ofUnary(configuration.getKey());
        }
      } else {
        var split = ImmutableMap.<String, BuildConfigurationKey>builder();
        for (int i = 0; i < depConfigs.size(); i++) {
          // Uses some fake transition keys.
          split.put(Integer.toString(i), depConfigs.get(i).getKey());
        }
        attributeConfiguration = AttributeConfiguration.ofSplit(split.buildOrThrow());
      }

      tasks.enqueue(
          new PrerequisitesProducer(
              parameters,
              key.getLabel(),
              key.getExecutionPlatformLabel(),
              attributeConfiguration,
              key.getPropagatingAspects(),
              new PrerequisitesProducer.ResultSink() {
                @Override
                public void acceptPrerequisitesValue(ConfiguredTargetAndData[] prerequisites) {
                  prerequisiteValues.putAll(key, Arrays.asList(prerequisites));
                }

                @Override
                public void acceptPrerequisitesError(InvalidVisibilityDependencyException error) {
                  hasError = true;
                  sink.acceptDependencyEvaluatorError(error);
                }

                @Override
                public void acceptPrerequisitesCreationError(
                    ConfiguredValueCreationException error) {
                  hasError = true;
                  sink.acceptDependencyEvaluatorError(error);
                }

                @Override
                public void acceptPrerequisitesAspectError(DependencyEvaluationException error) {
                  hasError = true;
                  sink.acceptDependencyEvaluatorError(error);
                }
              }));
    }
    return this::emitResult;
  }

  private StateMachine emitResult(Tasks tasks, ExtendedEventHandler listener) {
    if (!hasError) {
      sink.acceptDependencyEvaluatorResult(prerequisiteValues);
    }
    return DONE;
  }
}
