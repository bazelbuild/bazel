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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.StarlarkTransitionCache;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionUtil;
import com.google.devtools.build.lib.analysis.starlark.StarlarkBuildSettingsDetailsValue;
import com.google.devtools.build.lib.analysis.starlark.StarlarkTransition;
import com.google.devtools.build.lib.analysis.starlark.StarlarkTransition.TransitionException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.StateMachine;
import java.util.Map;
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
    void acceptTransitionedOptions(Map<String, BuildOptions> transitionedOptions);

    void acceptTransitionError(TransitionException e);
  }

  // -------------------- Input --------------------
  private final BuildOptions fromOptions;
  private final ConfigurationTransition transition;
  private final StarlarkTransitionCache transitionCache;

  // -------------------- Output --------------------
  private final ResultSink sink;

  // -------------------- Internal State --------------------
  private StarlarkBuildSettingsDetailsValue buildSettingsDetailsValue;

  TransitionApplier(
      BuildOptions fromOptions,
      ConfigurationTransition transition,
      StarlarkTransitionCache transitionCache,
      ResultSink sink) {
    this.fromOptions = fromOptions;
    this.transition = transition;
    this.transitionCache = transitionCache;
    this.sink = sink;
  }

  @Override
  public StateMachine step(Tasks tasks, ExtendedEventHandler listener) throws InterruptedException {
    boolean doesStarlarkTransition;
    try {
      doesStarlarkTransition = StarlarkTransition.doesStarlarkTransition(transition);
    } catch (TransitionException e) {
      sink.acceptTransitionError(e);
      return DONE;
    }
    if (!doesStarlarkTransition) {
      sink.acceptTransitionedOptions(
          transition.apply(TransitionUtil.restrict(transition, fromOptions), listener));
      return DONE;
    }

    ImmutableSet<Label> starlarkBuildSettings =
        StarlarkTransition.getAllStarlarkBuildSettings(transition);
    if (starlarkBuildSettings.isEmpty()) {
      // Quick escape if transition doesn't use any Starlark build settings.
      buildSettingsDetailsValue = StarlarkBuildSettingsDetailsValue.EMPTY;
      return applyStarlarkTransition(tasks, listener);
    }
    tasks.lookUp(
        StarlarkBuildSettingsDetailsValue.key(starlarkBuildSettings),
        TransitionException.class,
        this);
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

  private StateMachine applyStarlarkTransition(Tasks tasks, ExtendedEventHandler listener)
      throws InterruptedException {
    if (buildSettingsDetailsValue == null) {
      return DONE; // There was an error.
    }

    try {
      sink.acceptTransitionedOptions(
          transitionCache.computeIfAbsent(
              fromOptions, transition, buildSettingsDetailsValue, listener));
    } catch (TransitionException e) {
      sink.acceptTransitionError(e);
    }
    return DONE;
  }
}
