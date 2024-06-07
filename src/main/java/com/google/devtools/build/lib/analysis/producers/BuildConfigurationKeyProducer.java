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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.config.NativeAndStarlarkFlags;
import com.google.devtools.build.lib.skyframe.config.PlatformMappingException;
import com.google.devtools.build.lib.skyframe.config.PlatformMappingValue;
import com.google.devtools.build.lib.skyframe.toolchains.PlatformLookupUtil.InvalidPlatformException;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.StateMachine;
import com.google.devtools.build.skyframe.state.StateMachine.ValueOrExceptionSink;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import javax.annotation.Nullable;

/**
 * Creates the needed {@link BuildConfigurationKey} instances for the given options.
 *
 * <p>This includes merging in platform mappings.
 *
 * <p>The output preserves the iteration order of the input.
 */
public class BuildConfigurationKeyProducer
    implements StateMachine,
        ValueOrExceptionSink<PlatformMappingException>,
        PlatformFlagsProducer.ResultSink {

  /** Interface for clients to accept results of this computation. */
  public interface ResultSink {

    void acceptTransitionError(OptionsParsingException e);

    void acceptPlatformMappingError(PlatformMappingException e);

    void acceptPlatformFlagsError(InvalidPlatformException error);

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
  private final Map<Label, NativeAndStarlarkFlags> platformFlags = new HashMap<>();

  public BuildConfigurationKeyProducer(
      ResultSink sink, StateMachine runAfter, Map<String, BuildOptions> options) {
    this.sink = sink;
    this.runAfter = runAfter;
    this.options = options;
  }

  @Override
  public StateMachine step(Tasks tasks) {
    findPlatformMappings(tasks);
    findTargetPlatformInfos(tasks);
    return this::applyFlags;
  }

  private void findPlatformMappings(Tasks tasks) {
    // Use any configuration, since all configurations will have the same platform mapping.
    Optional<PathFragment> platformMappingsPath =
        options.values().stream()
            .filter(opts -> opts.contains(PlatformOptions.class))
            .filter(opts -> opts.get(PlatformOptions.class).platformMappings != null)
            .map(opts -> opts.get(PlatformOptions.class).platformMappings)
            .findAny();
    PlatformMappingValue.Key platformMappingValueKey =
        PlatformMappingValue.Key.create(platformMappingsPath.orElse(null));
    tasks.lookUp(platformMappingValueKey, PlatformMappingException.class, this);
  }

  private void findTargetPlatformInfos(Tasks tasks) {
    this.options.values().stream()
        .filter(opts -> opts.contains(PlatformOptions.class))
        .flatMap(opts -> opts.get(PlatformOptions.class).platforms.stream())
        .map(targetPlatform -> new PlatformFlagsProducer(targetPlatform, this, StateMachine.DONE))
        .forEach(tasks::enqueue);
  }

  // Handles results from the PlatformMappingValueKey lookup.
  @Override
  public void acceptValueOrException(
      @Nullable SkyValue value, @Nullable PlatformMappingException exception) {
    if (exception != null) {
      sink.acceptPlatformMappingError(exception);
      return;
    }
    if (value instanceof PlatformMappingValue platformMappingValue) {
      this.platformMappingValue = platformMappingValue;
      return;
    }

    throw new IllegalStateException("No value or exception was provided");
  }

  // Handle results from PlatformFlagsProducer.
  @Override
  public void acceptPlatformFlags(Label platform, NativeAndStarlarkFlags flags) {
    BuildConfigurationKeyProducer.this.platformFlags.put(platform, flags);
  }

  @Override
  public void acceptPlatformFlagsError(Label unusedPlatform, InvalidPlatformException error) {
    // The requested platform is in the exception already, so it's fine to drop.
    sink.acceptPlatformFlagsError(error);
  }

  @Override
  public void acceptPlatformFlagsError(Label unusedPlatform, OptionsParsingException error) {
    // TODO: blaze-configurability-team - See if this should include the requested platform in the
    // error message.
    sink.acceptTransitionError(error);
  }

  private StateMachine applyFlags(Tasks tasks) {
    if (this.platformMappingValue == null) {
      return DONE; // There was an error.
    }

    var result =
        ImmutableMap.<String, BuildConfigurationKey>builderWithExpectedSize(options.size());
    for (Map.Entry<String, BuildOptions> entry : options.entrySet()) {
      String transitionKey = entry.getKey();
      try {
        BuildConfigurationKey newConfigurationKey = applyFlagsForOptions(entry.getValue());
        result.put(transitionKey, newConfigurationKey);
      } catch (OptionsParsingException e) {
        sink.acceptTransitionError(e);
        return runAfter;
      }
    }
    sink.acceptTransitionedConfigurations(result.buildOrThrow());
    return runAfter;
  }

  /**
   * Apply discovered flags from platforms or platform mappings to the given options, and return the
   * {@link BuildConfigurationKey}.
   *
   * <p>Platform-based flags and platform mappings are mutually exclusive: only one will be applied
   * if they are present. Trying to mix and match would be possible but confusing, especially if
   * they try to change the same flag.
   */
  private BuildConfigurationKey applyFlagsForOptions(BuildOptions options)
      throws OptionsParsingException {
    // Does the target platform provide any flags?
    if (options.contains(PlatformOptions.class)) {
      List<Label> targetPlatforms = options.get(PlatformOptions.class).platforms;
      if (targetPlatforms != null && targetPlatforms.size() == 1) {
        // TODO: https://github.com/bazelbuild/bazel/issues/19807 - We define this flag to only use
        // the first value and ignore any subsequent ones. Remove this check as part of cleanup.
        Label targetPlatform = targetPlatforms.get(0);

        if (this.platformFlags.containsKey(targetPlatform)) {
          NativeAndStarlarkFlags platformBasedFlags = this.platformFlags.get(targetPlatform);
          OptionsParsingResult parsingResult = platformBasedFlags.parse();
          BuildOptions updatedOptions = options.applyParsingResult(parsingResult);
          return BuildConfigurationKey.create(updatedOptions);
        }
      }
    }

    // Is there a platform mapping?
    if (this.platformMappingValue != null) {
      BuildOptions mappedOptions = this.platformMappingValue.map(options);
      return BuildConfigurationKey.create(mappedOptions);
    }

    // Just use the original options.
    return BuildConfigurationKey.create(options);
  }
}
