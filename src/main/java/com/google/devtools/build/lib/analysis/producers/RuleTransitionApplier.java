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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.StarlarkTransitionCache;
import com.google.devtools.build.lib.analysis.config.transitions.ComposingTransition;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.analysis.starlark.StarlarkTransition.TransitionException;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleTransitionData;
import com.google.devtools.build.lib.skyframe.BuildConfigurationKey;
import com.google.devtools.build.skyframe.state.StateMachine;
import com.google.devtools.common.options.OptionsParsingException;
import javax.annotation.Nullable;

/** Applies the rule and trimming transitions to a {@link BuildConfigurationKey}. */
final class RuleTransitionApplier implements StateMachine, TransitionApplier.ResultSink {
  interface ResultSink {
    void acceptRuleTransitionResult(BuildConfigurationKey key);

    void acceptRuleTransitionError(TransitionException error);

    void acceptRuleTransitionError(OptionsParsingException error);
  }

  // -------------------- Input --------------------
  private final BuildConfigurationKey configurationKey;
  private final Rule rule;
  @Nullable private final TransitionFactory<RuleTransitionData> trimmingTransitionFactory;
  private final StarlarkTransitionCache transitionCache;

  // -------------------- Output --------------------
  private final ResultSink sink;

  // -------------------- Sequencing --------------------
  private final StateMachine runAfter;

  RuleTransitionApplier(
      BuildConfigurationKey configurationKey,
      Rule rule,
      @Nullable TransitionFactory<RuleTransitionData> trimmingTransitionFactory,
      StarlarkTransitionCache transitionCache,
      ResultSink sink,
      StateMachine runAfter) {
    this.configurationKey = configurationKey;
    this.rule = rule;
    this.trimmingTransitionFactory = trimmingTransitionFactory;
    this.transitionCache = transitionCache;
    this.sink = sink;
    this.runAfter = runAfter;
  }

  @Override
  public StateMachine step(Tasks tasks, ExtendedEventHandler listener) {
    ConfigurationTransition transition = computeTransition(rule, trimmingTransitionFactory);
    if (transition == null) {
      sink.acceptRuleTransitionResult(configurationKey);
      return runAfter;
    }

    tasks.enqueue(
        new TransitionApplier(
            configurationKey, transition, transitionCache, (TransitionApplier.ResultSink) this));

    return runAfter;
  }

  @Override
  public void acceptTransitionedConfigurations(
      ImmutableMap<String, BuildConfigurationKey> transitionResult) {
    checkState(transitionResult.size() == 1, "Expected exactly one result: %s", transitionResult);
    sink.acceptRuleTransitionResult(
        checkNotNull(
            transitionResult.get(ConfigurationTransition.PATCH_TRANSITION_KEY),
            "Transition result missing patch transition entry: %s",
            transitionResult));
  }

  @Override
  public void acceptTransitionError(TransitionException e) {
    sink.acceptRuleTransitionError(e);
  }

  @Override
  public void acceptTransitionError(OptionsParsingException e) {
    sink.acceptRuleTransitionError(e);
  }

  @Nullable
  private static ConfigurationTransition computeTransition(
      Rule rule, @Nullable TransitionFactory<RuleTransitionData> trimmingTransitionFactory) {
    var transitionData = RuleTransitionData.create(rule);

    ConfigurationTransition transition = null;

    TransitionFactory<RuleTransitionData> transitionFactory =
        rule.getRuleClassObject().getTransitionFactory();
    if (transitionFactory != null) {
      transition = transitionFactory.create(transitionData);
    }

    if (trimmingTransitionFactory != null) {
      var trimmingTransition = trimmingTransitionFactory.create(transitionData);
      if (transition != null) {
        transition = ComposingTransition.of(transition, trimmingTransition);
      } else {
        transition = trimmingTransition;
      }
    }

    return transition;
  }
}
