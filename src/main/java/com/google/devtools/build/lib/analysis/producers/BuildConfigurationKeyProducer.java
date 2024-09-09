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
 * Creates the needed {@link BuildConfigurationKey} instance for a single {@link BuildOptions},
 * including merging in any platform-based flags or a platform mapping.
 *
 * <p>Platform-based flags and platform mappings are mutually exclusive: only one will be applied if
 * they are present. Trying to mix and match would be possible but confusing, especially if they try
 * to change the same flag. The logic is:
 *
 * <ul>
 *   <li>If {@link PlatformOptions#platforms} specifies a target platform, look up the {@link
 *       PlatformValue}. If it specifies {@linkplain PlatformValue#parsedFlags flags}, use {@link
 *       ParsedFlagsValue#mergeWith}.
 *   <li>If {@link PlatformOptions#platforms} does not specify a target platform, or if the target
 *       platform does not specify {@linkplain PlatformValue#parsedFlags flags}, look up the {@link
 *       PlatformMappingValue} and use {@link PlatformMappingValue#map}.
 * </ul>
 *
 * @param <C> The type of the context variable that the producer will pass via the {@link
 *     ResultSink} so that consumers can identify which options are which.
 */
public final class BuildConfigurationKeyProducer<C>
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
  private PlatformValue targetPlatformValue;
  private PlatformMappingValue platformMappingValue;

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
    BuildConfigurationKey result = cache.get(options);
    if (result != null) {
      sink.acceptTransitionedConfiguration(context, result);
      return runAfter;
    }

    // Short-circuit if there are no platform options.
    if (!options.contains(PlatformOptions.class)) {
      return finishConfigurationKeyProcessing(BuildConfigurationKey.create(options));
    }

    List<Label> targetPlatforms = options.get(PlatformOptions.class).platforms;
    if (targetPlatforms.size() == 1) {
      // TODO: https://github.com/bazelbuild/bazel/issues/19807 - We define this flag to only use
      //  the first value and ignore any subsequent ones. Remove this check as part of cleanup.
      tasks.enqueue(
          new PlatformProducer(targetPlatforms.getFirst(), this, this::checkTargetPlatformFlags));
      return runAfter;
    } else {
      return mergeFromPlatformMapping(tasks);
    }
  }

  private StateMachine checkTargetPlatformFlags(Tasks tasks) {
    if (targetPlatformValue == null) {
      return DONE; // Error.
    }
    Optional<ParsedFlagsValue> parsedFlags = targetPlatformValue.parsedFlags();
    if (parsedFlags.isPresent()) {
      BuildOptions updatedOptions = parsedFlags.get().mergeWith(options);
      return finishConfigurationKeyProcessing(BuildConfigurationKey.create(updatedOptions));
    } else {
      return mergeFromPlatformMapping(tasks);
    }
  }

  private StateMachine mergeFromPlatformMapping(Tasks tasks) {
    PathFragment platformMappingsPath = options.get(PlatformOptions.class).platformMappings;
    tasks.lookUp(
        PlatformMappingValue.Key.create(platformMappingsPath),
        PlatformMappingException.class,
        this);
    return this::applyPlatformMapping;
  }

  private StateMachine applyPlatformMapping(Tasks tasks) {
    if (platformMappingValue == null) {
      return DONE; // Error.
    }
    try {
      BuildOptions updatedOptions = platformMappingValue.map(options);
      return finishConfigurationKeyProcessing(BuildConfigurationKey.create(updatedOptions));
    } catch (OptionsParsingException e) {
      sink.acceptOptionsParsingError(e);
      return runAfter;
    }
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
    } else {
      this.platformMappingValue = (PlatformMappingValue) value;
    }
  }

  @Override
  public void acceptPlatformValue(PlatformValue value) {
    this.targetPlatformValue = value;
  }

  @Override
  public void acceptPlatformInfoError(InvalidPlatformException error) {
    sink.acceptPlatformFlagsError(error);
  }

  @Override
  public void acceptOptionsParsingError(OptionsParsingException error) {
    sink.acceptOptionsParsingError(error);
  }

  private StateMachine finishConfigurationKeyProcessing(BuildConfigurationKey newConfigurationKey) {
    cache.put(this.options, newConfigurationKey);
    sink.acceptTransitionedConfiguration(this.context, newConfigurationKey);
    return this.runAfter;
  }
}
