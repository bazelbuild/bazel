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

import static com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition.PATCH_TRANSITION_KEY;

import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimaps;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.StarlarkTransitionCache;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionUtil;
import com.google.devtools.build.lib.analysis.starlark.StarlarkBuildSettingsDetailsValue;
import com.google.devtools.build.lib.analysis.starlark.StarlarkTransition;
import com.google.devtools.build.lib.analysis.starlark.StarlarkTransition.TransitionException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.skyframe.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.PlatformMappingValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.StateMachine;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import javax.annotation.Nullable;

/**
 * Applies a configuration transition to a build options instance.
 *
 * <p>postwork - replay events/throw errors from transition implementation function and validate the
 * outputs of the transition. This only applies to Starlark transitions.
 */
final class TransitionApplier
    implements StateMachine, StateMachine.ValueOrExceptionSink<TransitionException> {
  interface ResultSink {
    void acceptTransitionedConfigurations(
        ImmutableMap<String, BuildConfigurationKey> transitionedOptions);

    void acceptTransitionError(TransitionException e);

    void acceptTransitionError(OptionsParsingException e);
  }

  // -------------------- Input --------------------
  private final BuildConfigurationKey fromConfiguration;
  private final ConfigurationTransition transition;
  private final StarlarkTransitionCache transitionCache;

  // -------------------- Output --------------------
  private final ResultSink sink;
  private final ExtendedEventHandler eventHandler;

  // -------------------- Sequencing --------------------
  private final StateMachine runAfter;

  // -------------------- Internal State --------------------
  private StarlarkBuildSettingsDetailsValue buildSettingsDetailsValue;

  TransitionApplier(
      BuildConfigurationKey fromConfiguration,
      ConfigurationTransition transition,
      StarlarkTransitionCache transitionCache,
      ResultSink sink,
      ExtendedEventHandler eventHandler,
      StateMachine runAfter) {
    this.fromConfiguration = fromConfiguration;
    this.transition = transition;
    this.transitionCache = transitionCache;
    this.sink = sink;
    this.eventHandler = eventHandler;
    this.runAfter = runAfter;
  }

  @Override
  public StateMachine step(Tasks tasks) throws InterruptedException {
    boolean doesStarlarkTransition;
    try {
      doesStarlarkTransition = StarlarkTransition.doesStarlarkTransition(transition);
    } catch (TransitionException e) {
      sink.acceptTransitionError(e);
      return runAfter;
    }
    if (!doesStarlarkTransition) {
      return convertOptionsToKeys(
          transition.apply(
              TransitionUtil.restrict(transition, fromConfiguration.getOptions()), eventHandler));
    }

    ImmutableSet<Label> starlarkBuildSettings =
        StarlarkTransition.getAllStarlarkBuildSettings(transition);
    if (starlarkBuildSettings.isEmpty()) {
      // Quick escape if transition doesn't use any Starlark build settings.
      buildSettingsDetailsValue = StarlarkBuildSettingsDetailsValue.EMPTY;
      return applyStarlarkTransition(tasks);
    }
    tasks.lookUp(
        StarlarkBuildSettingsDetailsValue.key(starlarkBuildSettings),
        TransitionException.class,
        (ValueOrExceptionSink<TransitionException>) this);
    return this::applyStarlarkTransition;
  }

  @Override
  public void acceptValueOrException(@Nullable SkyValue value, @Nullable TransitionException e) {
    if (value != null) {
      buildSettingsDetailsValue = (StarlarkBuildSettingsDetailsValue) value;
      return;
    }
    if (e != null) {
      sink.acceptTransitionError(e);
      return;
    }
    throw new IllegalArgumentException("No result received.");
  }

  private StateMachine applyStarlarkTransition(Tasks tasks) throws InterruptedException {
    if (buildSettingsDetailsValue == null) {
      return runAfter; // There was an error.
    }

    Map<String, BuildOptions> transitionedOptions;
    try {
      transitionedOptions =
          transitionCache.computeIfAbsent(
              fromConfiguration.getOptions(), transition, buildSettingsDetailsValue, eventHandler);
    } catch (TransitionException e) {
      sink.acceptTransitionError(e);
      return runAfter;
    }
    return convertOptionsToKeys(transitionedOptions);
  }

  private StateMachine convertOptionsToKeys(Map<String, BuildOptions> transitionedOptions) {
    // If there is a single, unchanged value, just outputs the original configuration, stripping any
    // transition key.
    if (transitionedOptions.size() == 1) {
      BuildOptions options = transitionedOptions.values().iterator().next();
      if (options.checksum().equals(fromConfiguration.getOptionsChecksum())) {
        sink.acceptTransitionedConfigurations(
            ImmutableMap.of(PATCH_TRANSITION_KEY, fromConfiguration));
        return runAfter;
      }
    }

    // Otherwise, applies a platform mapping to the results.
    return new PlatformMappingApplier(transitionedOptions);
  }

  /**
   * Applies the platform mapping to each option.
   *
   * <p>The output preserves the iteration order of the input.
   */
  private class PlatformMappingApplier implements StateMachine {
    // -------------------- Input --------------------
    private final Map<String, BuildOptions> options;

    // -------------------- Internal State --------------------
    private final Map<String, PlatformMappingValue> platformMappingValues = new HashMap<>();

    private PlatformMappingApplier(Map<String, BuildOptions> options) {
      this.options = options;
    }

    @Override
    public StateMachine step(Tasks tasks) {
      // Deduplicates the platform mapping paths and collates the transition keys.
      ImmutableListMultimap<Optional<PathFragment>, String> index =
          Multimaps.index(
              options.keySet(),
              transitionKey ->
                  Optional.ofNullable(getPlatformMappingsPath(options.get(transitionKey))));
      for (Map.Entry<Optional<PathFragment>, Collection<String>> entry : index.asMap().entrySet()) {
        Collection<String> transitionKeys = entry.getValue();
        tasks.lookUp(
            PlatformMappingValue.Key.create(entry.getKey().orElse(null)),
            rawValue -> {
              var value = (PlatformMappingValue) rawValue;
              // Maps the value from all transition keys with the same platform mappings path.
              for (String key : transitionKeys) {
                platformMappingValues.put(key, value);
              }
            });
      }
      return this::applyMappings;
    }

    private StateMachine applyMappings(Tasks tasks) {
      var result =
          ImmutableMap.<String, BuildConfigurationKey>builderWithExpectedSize(options.size());
      for (Map.Entry<String, BuildOptions> entry : options.entrySet()) {
        String transitionKey = entry.getKey();
        BuildConfigurationKey newConfigurationKey;
        try {
          newConfigurationKey =
              BuildConfigurationKey.withPlatformMapping(
                  platformMappingValues.get(transitionKey), entry.getValue());
        } catch (OptionsParsingException e) {
          sink.acceptTransitionError(e);
          return runAfter;
        }
        result.put(transitionKey, newConfigurationKey);
      }
      sink.acceptTransitionedConfigurations(result.buildOrThrow());
      return runAfter;
    }
  }

  @Nullable
  private static PathFragment getPlatformMappingsPath(BuildOptions fromOptions) {
    return fromOptions.hasNoConfig()
        ? null
        : fromOptions.get(PlatformOptions.class).platformMappings;
  }
}
