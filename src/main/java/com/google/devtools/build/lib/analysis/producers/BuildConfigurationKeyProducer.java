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

import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Multimaps;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.config.PlatformMappingValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.state.StateMachine;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Creates the needed {@link BuildConfigurationKey} instances for the given options.
 *
 * <p>This includes merging in platform mappings.
 *
 * <p>The output preserves the iteration order of the input.
 */
// Logic here must be kept in sync with SkyframeExecutor.createBuildConfigurationKey.
public class BuildConfigurationKeyProducer implements StateMachine {
  /** Interface for clients to accept results of this computation. */
  public interface ResultSink {

    void acceptTransitionError(OptionsParsingException e);

    void acceptTransitionedConfigurations(
        ImmutableMap<String, BuildConfigurationKey> transitionedOptions);
  }

  // -------------------- Input --------------------
  private final ResultSink sink;
  private final StateMachine runAfter;
  private final Map<String, BuildOptions> options;

  // -------------------- Internal State --------------------
  private final Map<String, PlatformMappingValue> platformMappingValues = new HashMap<>();

  public BuildConfigurationKeyProducer(
      ResultSink sink, StateMachine runAfter, Map<String, BuildOptions> options) {
    this.sink = sink;
    this.runAfter = runAfter;
    this.options = options;
  }

  @Override
  public StateMachine step(Tasks tasks) {
    // Deduplicates the platform mapping paths and collates the transition keys.
    ImmutableListMultimap<Optional<PathFragment>, String> index =
        Multimaps.index(
            options.keySet(), transitionKey -> getPlatformMappingsPath(options.get(transitionKey)));
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
        PlatformMappingValue mappingValue = platformMappingValues.get(transitionKey);
        BuildOptions mappedOptions = mappingValue.map(entry.getValue());
        newConfigurationKey = BuildConfigurationKey.create(mappedOptions);
      } catch (OptionsParsingException e) {
        sink.acceptTransitionError(e);
        return runAfter;
      }
      result.put(transitionKey, newConfigurationKey);
    }
    sink.acceptTransitionedConfigurations(result.buildOrThrow());
    return runAfter;
  }

  private static Optional<PathFragment> getPlatformMappingsPath(BuildOptions fromOptions) {
    return fromOptions.hasNoConfig()
        ? Optional.empty()
        : Optional.ofNullable(fromOptions.get(PlatformOptions.class).platformMappings);
  }
}
