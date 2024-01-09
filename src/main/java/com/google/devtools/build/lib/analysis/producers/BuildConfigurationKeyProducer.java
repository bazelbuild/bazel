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
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.config.PlatformMappingValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.StateMachine;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Map;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * Creates the needed {@link BuildConfigurationKey} instances for the given options.
 *
 * <p>This includes merging in platform mappings.
 *
 * <p>The output preserves the iteration order of the input.
 */
// Logic here must be kept in sync with SkyframeExecutor.createBuildConfigurationKey.
public class BuildConfigurationKeyProducer implements StateMachine, Consumer<SkyValue> {

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
  // There is only ever a single PlatformMappingValue in use, as the `--platform_mappings` flag
  // can not be changed in a transition.
  private PlatformMappingValue platformMappingValue;

  public BuildConfigurationKeyProducer(
      ResultSink sink, StateMachine runAfter, Map<String, BuildOptions> options) {
    this.sink = sink;
    this.runAfter = runAfter;
    this.options = options;
  }

  @Override
  public StateMachine step(Tasks tasks) {
    // Use any configuration, since all configurations will have the same platform mapping.
    Optional<PathFragment> platformMappingsPath =
        options.values().stream()
            .filter(opts -> opts.contains(PlatformOptions.class))
            .filter(opts -> opts.get(PlatformOptions.class).platformMappings != null)
            .map(opts -> opts.get(PlatformOptions.class).platformMappings)
            .findAny();
    PlatformMappingValue.Key platformMappingValueKey =
        PlatformMappingValue.Key.create(platformMappingsPath.orElse(null));
    // No need to consider error handling: this flag cannot be changed and so errors were
    // detected and handled in SkyframeExecutor.createBuildConfigurationKey.
    tasks.lookUp(platformMappingValueKey, this);
    return this::applyMappings;
  }

  @Override
  public void accept(SkyValue skyValue) {
    if (skyValue instanceof PlatformMappingValue) {
      this.platformMappingValue = (PlatformMappingValue) skyValue;
    }
  }

  private StateMachine applyMappings(Tasks tasks) {
    var result =
        ImmutableMap.<String, BuildConfigurationKey>builderWithExpectedSize(options.size());
    for (Map.Entry<String, BuildOptions> entry : options.entrySet()) {
      String transitionKey = entry.getKey();
      BuildConfigurationKey newConfigurationKey;
      try {
        BuildOptions mappedOptions = this.platformMappingValue.map(entry.getValue());
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
}
