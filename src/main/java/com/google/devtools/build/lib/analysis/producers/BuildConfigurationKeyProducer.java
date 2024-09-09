// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.platform.PlatformValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.config.ParsedFlagsValue;
import com.google.devtools.build.lib.skyframe.config.PlatformMappingException;
import com.google.devtools.build.lib.skyframe.config.PlatformMappingValue;
import com.google.devtools.build.lib.skyframe.toolchains.PlatformLookupUtil.InvalidPlatformException;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.StateMachine;
import com.google.devtools.build.skyframe.state.StateMachine.ValueOrExceptionSink;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.List;
import java.util.Optional;
import javax.annotation.Nullable;

/**
 * Creates the needed {@link BuildConfigurationKey} instances for a single {@link BuildOptions}.
 *
 * <p>This includes merging in platform mappings and platform-based flags.
 *
 * @param <C> The type of the context variable that the producer will pass via the {@link
 *     ResultSink} so that consumers can identify which options are which.
 */
public class BuildConfigurationKeyProducer<C>
    implements StateMachine,
        ValueOrExceptionSink<PlatformMappingException>,
        PlatformProducer.ResultSink {

  /** Interface for clients to accept results of this computation. */
  public interface ResultSink<C> {

    void acceptOptionsParsingError(OptionsParsingException e);

    void acceptPlatformMappingError(PlatformMappingException e);

    void acceptPlatformFlagsError(InvalidPlatformException error);

    void acceptTransitionedConfiguration(C context, BuildConfigurationKey transitionedOptionKey);
  }

  // -------------------- Input --------------------
  private final ResultSink<C> sink;
  private final StateMachine runAfter;
  private final BuildConfigurationKeyCache cache;
  private final C context;
  private final BuildOptions options;

  // -------------------- Internal State --------------------
  private PlatformMappingValue platformMappingValue;
  private Optional<ParsedFlagsValue> platformFlags = Optional.empty();

  BuildConfigurationKeyProducer(
      ResultSink<C> sink,
      StateMachine runAfter,
      BuildConfigurationKeyCache cache,
      C context,
      BuildOptions options) {
    this.sink = sink;
    this.runAfter = runAfter;
    this.cache = cache;
    this.context = context;
    this.options = options;
  }

  @Override
  public StateMachine step(Tasks tasks) throws InterruptedException {
    BuildConfigurationKey result = cache.get(this.options);
    if (result != null) {
      this.sink.acceptTransitionedConfiguration(this.context, result);
      return this.runAfter;
    }

    // Short-circuit if there are no platform options.
    if (!this.options.contains(PlatformOptions.class)) {
      return finishConfigurationKeyProcessing(BuildConfigurationKey.create(this.options));
    }

    // Find platform mappings and platform-based flags for merging.
    findPlatformMapping(tasks);
    findTargetPlatformInfo(tasks);
    return this::applyFlags;
  }

  private void findPlatformMapping(Tasks tasks) {
    PathFragment platformMappingsPath = options.get(PlatformOptions.class).platformMappings;
    PlatformMappingValue.Key platformMappingValueKey =
        PlatformMappingValue.Key.create(platformMappingsPath);
    tasks.lookUp(platformMappingValueKey, PlatformMappingException.class, this);
  }

  // Handles results from the PlatformMappingValueKey lookup.
  @Override
  public void acceptValueOrException(
      @Nullable SkyValue value, @Nullable PlatformMappingException exception) {
    if (value == null && exception == null) {
      throw new IllegalStateException("No value or exception was provided");
    }
    if (value != null && exception != null) {
      throw new IllegalStateException("Both value and exception were provided");
    }

    if (exception != null) {
      sink.acceptPlatformMappingError(exception);
    } else if (value instanceof PlatformMappingValue platformMappingValue) {
      this.platformMappingValue = platformMappingValue;
    }
  }

  private void findTargetPlatformInfo(Tasks tasks) {
    List<Label> targetPlatforms = options.get(PlatformOptions.class).platforms;
    if (targetPlatforms.size() == 1) {
      // TODO: https://github.com/bazelbuild/bazel/issues/19807 - We define this flag to only use
      // the first value and ignore any subsequent ones. Remove this check as part of cleanup.
      tasks.enqueue(new PlatformProducer(targetPlatforms.getFirst(), this, StateMachine.DONE));
    }
  }

  @Override
  public void acceptPlatformValue(PlatformValue value) {
    this.platformFlags = value.parsedFlags();
  }

  @Override
  public void acceptPlatformInfoError(InvalidPlatformException error) {
    sink.acceptPlatformFlagsError(error);
  }

  @Override
  public void acceptOptionsParsingError(OptionsParsingException error) {
    sink.acceptOptionsParsingError(error);
  }

  private StateMachine applyFlags(Tasks tasks) {
    if (this.platformMappingValue == null) {
      return DONE; // There was an error.
    }

    try {
      BuildConfigurationKey newConfigurationKey = applyFlagsForOptions(options);
      return finishConfigurationKeyProcessing(newConfigurationKey);
    } catch (OptionsParsingException e) {
      sink.acceptOptionsParsingError(e);
      return runAfter;
    }
  }

  private StateMachine finishConfigurationKeyProcessing(BuildConfigurationKey newConfigurationKey) {
    cache.put(this.options, newConfigurationKey);
    sink.acceptTransitionedConfiguration(this.context, newConfigurationKey);
    return this.runAfter;
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
    if (this.platformFlags.isPresent()) {
      BuildOptions updatedOptions = this.platformFlags.get().mergeWith(options);
      return BuildConfigurationKey.create(updatedOptions);
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
