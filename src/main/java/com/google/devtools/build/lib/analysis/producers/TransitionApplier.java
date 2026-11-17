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

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.StarlarkTransitionCache;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionUtil;
import com.google.devtools.build.lib.analysis.platform.PlatformValue;
import com.google.devtools.build.lib.analysis.starlark.StarlarkBuildSettingsDetailsValue;
import com.google.devtools.build.lib.analysis.starlark.StarlarkTransition.StarlarkTransitionVisitor;
import com.google.devtools.build.lib.analysis.starlark.StarlarkTransition.TransitionException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.config.ParsedFlagsValue;
import com.google.devtools.build.lib.skyframe.toolchains.PlatformLookupUtil.InvalidPlatformException;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.StateMachine;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

/**
 * Applies a configuration transition to a build options instance.
 *
 * <p>postwork - replay events/throw errors from transition implementation function and validate the
 * outputs of the transition. This only applies to Starlark transitions.
 */
final class TransitionApplier
    implements StateMachine, StateMachine.ValueOrExceptionSink<TransitionException> {
  interface ResultSink extends BuildConfigurationKeyMapProducer.ResultSink {
    void acceptTransitionError(TransitionException e);
  }

  // -------------------- Input --------------------
  private final Label label;
  private final BuildConfigurationKey fromConfiguration;
  private final StarlarkBuildSettingsDetailsValue fromConfigurationFlagDetails;
  private final ConfigurationTransition transition;
  private final StarlarkTransitionCache transitionCache;

  // -------------------- Output --------------------
  private final ResultSink sink;
  private final ExtendedEventHandler eventHandler;

  // -------------------- Sequencing --------------------
  private final StateMachine runAfter;

  // -------------------- Internal State --------------------
  private boolean appliesExecTransition;
  private StarlarkBuildSettingsDetailsValue buildSettingsDetailsValue;

  /**
   * The flags of the platforms that transitions in a composition change to before another one is
   * applied, or null if there are none.
   */
  @Nullable private Map<Label, StarlarkTransitionCache.PlatformFlags> platformFlags;

  /** The platforms looked up for {@link #platformFlags} whose flag details are still missing. */
  private final Map<Label, PlatformValue> lookedUpPlatforms = new HashMap<>();

  private boolean platformLookupFailed;

  /** The events of the last incomplete application of the transition, if any. */
  @Nullable private StoredEventHandler incompleteTransitionEvents;

  /**
   * @param fromConfigurationFlagDetails the details of {@code fromConfiguration}'s Starlark flags
   *     or {@link StarlarkBuildSettingsDetailsValue#EMPTY}, see {@link
   *     BuildConfigurationKeyProducer}
   */
  TransitionApplier(
      Label label,
      BuildConfigurationKey fromConfiguration,
      StarlarkBuildSettingsDetailsValue fromConfigurationFlagDetails,
      ConfigurationTransition transition,
      StarlarkTransitionCache transitionCache,
      ResultSink sink,
      ExtendedEventHandler eventHandler,
      StateMachine runAfter) {
    this.fromConfiguration = fromConfiguration;
    this.fromConfigurationFlagDetails = fromConfigurationFlagDetails;
    this.transition = transition;
    this.transitionCache = transitionCache;
    this.sink = sink;
    this.eventHandler = eventHandler;
    this.runAfter = runAfter;
    this.label = label;
  }

  @Override
  public StateMachine step(Tasks tasks) throws InterruptedException {
    AtomicBoolean doesStarlarkTransition = new AtomicBoolean(false);
    AtomicBoolean stampDependent = new AtomicBoolean(false);
    try {
      transition.visit(
          (StarlarkTransitionVisitor)
              t -> {
                doesStarlarkTransition.set(true);
                if (t.isExecTransition()) {
                  appliesExecTransition = true;
                } else if (t.readsStampSetting() || t.setsStampSetting()) {
                  stampDependent.set(true);
                }
              });
    } catch (TransitionException e) {
      sink.acceptTransitionError(e);
      return runAfter;
    }
    if (!doesStarlarkTransition.get()) {
      BuildOptions fromOptions = fromConfiguration.getOptions();
      Map<String, BuildOptions> transitionedOptions =
          transition.apply(TransitionUtil.restrict(transition, fromOptions), eventHandler);
      return new BuildConfigurationKeyMapProducer(
          this.sink,
          this.runAfter,
          transitionedOptions,
          Maps.filterValues(
                  transitionedOptions,
                  toOptions ->
                      StarlarkTransitionCache.changesTargetPlatform(
                          transition, fromOptions, toOptions))
              .keySet(),
          /* forBaseline= */ false,
          this.fromConfigurationFlagDetails,
          this.label);
    }
    if (stampDependent.get()
        && fromConfiguration.getOptions().get(CoreOptions.class).getStampBinaries()) {
      // Request the STAMP_SETTING_MARKER dep. It's a precomputed value so should already be done,
      // but return a reference to the next step anyway as a state machine best practice.
      tasks.lookUp(PrecomputedValue.STAMP_SETTING_MARKER.getKey(), val -> {});
      return this::handleStarlarkTransition;
    }
    return handleStarlarkTransition(tasks);
  }

  private StateMachine handleStarlarkTransition(Tasks tasks) throws InterruptedException {
    ImmutableSet<Label> starlarkBuildSettings =
        transitionCache.getAllStarlarkBuildSettings(
            transition,
            fromConfiguration.getOptions().get(CoreOptions.class).getCommandLineFlagAliasesMap());
    ImmutableSet<Label> hostFlags;

    // If the transition is or composes the exec transition, we want to look up the host flag
    // declared by users in the blazerc/MODULE.bazel files with alias pointing to the starlark
    // definition. This is useful to determine exec propagation for flags with scope that starts
    // with "exec:--".
    if (appliesExecTransition) {
      hostFlags = fromConfiguration.getOptions().get(CoreOptions.class).getHostFlagAliases();
    } else if (starlarkBuildSettings.isEmpty()) {
      // Quick escape if transition doesn't use any Starlark build settings.
      buildSettingsDetailsValue = StarlarkBuildSettingsDetailsValue.EMPTY;
      return applyStarlarkTransition(tasks);
    } else {
      hostFlags = ImmutableSet.of();
    }
    tasks.lookUp(
        StarlarkBuildSettingsDetailsValue.key(starlarkBuildSettings, hostFlags),
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
    if (buildSettingsDetailsValue == null || platformLookupFailed) {
      return runAfter; // There was an error.
    }

    StarlarkTransitionCache.Result result;
    try {
      result =
          transitionCache.computeIfAbsent(
              fromConfiguration.getOptions(),
              transition,
              buildSettingsDetailsValue,
              platformFlags == null ? ImmutableMap.of() : platformFlags,
              eventHandler);
    } catch (TransitionException e) {
      sink.acceptTransitionError(e);
      return runAfter;
    } catch (InterruptedException e) {
      // Workaround for https://github.com/bazelbuild/bazel/issues/29132. Is there some way for
      // Skfyrame to handle this automaticaly without needing special checking here?
      return runAfter;
    }

    switch (result) {
      case StarlarkTransitionCache.MissingPlatforms(var platforms, var events) -> {
        incompleteTransitionEvents = events;
        ImmutableMap<String, Label> flagAliases =
            fromConfiguration.getOptions().get(CoreOptions.class).getCommandLineFlagAliasesMap();
        for (Label platform : platforms) {
          tasks.lookUp(
              PlatformValue.key(platform, flagAliases),
              InvalidPlatformException.class,
              OptionsParsingException.class,
              (value, invalidPlatformException, optionsParsingException) ->
                  acceptPlatformValue(
                      platform, value, invalidPlatformException, optionsParsingException));
        }
        return this::lookUpPlatformFlagDetails;
      }
      case StarlarkTransitionCache.Applied(var options, var platformChangedKeys) -> {
        return new BuildConfigurationKeyMapProducer(
            this.sink,
            this.runAfter,
            options,
            platformChangedKeys,
            /* forBaseline= */ false,
            this.fromConfigurationFlagDetails,
            this.label);
      }
    }
  }

  private void acceptPlatformValue(
      Label platform,
      @Nullable SkyValue value,
      @Nullable InvalidPlatformException invalidPlatformException,
      @Nullable OptionsParsingException optionsParsingException) {
    if (value != null) {
      lookedUpPlatforms.put(platform, (PlatformValue) value);
      return;
    }
    failPlatformLookup();
    if (invalidPlatformException != null) {
      sink.acceptPlatformFlagsError(invalidPlatformException);
    } else {
      sink.acceptOptionsParsingError(optionsParsingException);
    }
  }

  private StateMachine lookUpPlatformFlagDetails(Tasks tasks) {
    if (platformLookupFailed) {
      return runAfter;
    }
    for (Map.Entry<Label, PlatformValue> entry : lookedUpPlatforms.entrySet()) {
      Label platform = entry.getKey();
      Optional<ParsedFlagsValue> flags = entry.getValue().parsedFlags();
      ImmutableSet<Label> starlarkFlags =
          appliesExecTransition && flags.isPresent()
              ? flags.get().parsingResult().getStarlarkOptions().keySet().stream()
                  .map(Label::parseCanonicalUnchecked)
                  .collect(toImmutableSet())
              : ImmutableSet.of();
      if (starlarkFlags.isEmpty()) {
        putPlatformFlags(platform, flags, StarlarkBuildSettingsDetailsValue.EMPTY);
        continue;
      }
      tasks.lookUp(
          StarlarkBuildSettingsDetailsValue.key(
              starlarkFlags,
              fromConfiguration.getOptions().get(CoreOptions.class).getHostFlagAliases()),
          TransitionException.class,
          (value, e) -> {
            if (value != null) {
              putPlatformFlags(platform, flags, (StarlarkBuildSettingsDetailsValue) value);
            } else {
              failPlatformLookup();
              sink.acceptTransitionError(e);
            }
          });
    }
    lookedUpPlatforms.clear();
    return this::applyStarlarkTransition;
  }

  private void putPlatformFlags(
      Label platform, Optional<ParsedFlagsValue> flags, StarlarkBuildSettingsDetailsValue details) {
    if (platformFlags == null) {
      platformFlags = new HashMap<>();
    }
    platformFlags.put(platform, new StarlarkTransitionCache.PlatformFlags(flags, details));
  }

  private void failPlatformLookup() {
    platformLookupFailed = true;
    if (incompleteTransitionEvents != null) {
      incompleteTransitionEvents.replayOn(eventHandler);
      incompleteTransitionEvents = null;
    }
  }
}
