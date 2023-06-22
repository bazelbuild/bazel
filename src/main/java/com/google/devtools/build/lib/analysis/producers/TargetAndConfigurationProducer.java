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
import static com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil.configurationId;

import com.google.auto.value.AutoOneOf;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.InconsistentNullConfigException;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.TransitiveDependencyState;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.StarlarkTransitionCache;
import com.google.devtools.build.lib.analysis.config.transitions.ComposingTransition;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.analysis.starlark.StarlarkTransition.TransitionException;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleTransitionData;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.ConfiguredValueCreationException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.StateMachine;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.function.Consumer;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Location;

/**
 * Computes the target and configuration for a configured target key.
 *
 * <p>If the key has a configuration and the target is configurable, attempts to apply a rule side
 * transition. If the configuration changes, delegates to a target with the new configuration. If
 * the target is not configurable, directly delegates to the null configuration.
 */
public final class TargetAndConfigurationProducer
    implements StateMachine, Consumer<SkyValue>, TargetProducer.ResultSink {
  /** Accepts results of this producer. */
  public interface ResultSink {
    void acceptTargetAndConfiguration(TargetAndConfiguration value, ConfiguredTargetKey fullKey);

    void acceptTargetAndConfigurationDelegatedValue(ConfiguredTargetValue value);

    void acceptTargetAndConfigurationError(TargetAndConfigurationError error);
  }

  /** Tagged union of possible errors. */
  @AutoOneOf(TargetAndConfigurationError.Kind.class)
  public abstract static class TargetAndConfigurationError {
    /** Tags the error type. */
    public enum Kind {
      CONFIGURED_VALUE_CREATION,
      INCONSISTENT_NULL_CONFIG
    }

    public abstract Kind kind();

    public abstract ConfiguredValueCreationException configuredValueCreation();

    public abstract InconsistentNullConfigException inconsistentNullConfig();

    private static TargetAndConfigurationError of(ConfiguredValueCreationException e) {
      return AutoOneOf_TargetAndConfigurationProducer_TargetAndConfigurationError
          .configuredValueCreation(e);
    }

    private static TargetAndConfigurationError of(InconsistentNullConfigException e) {
      return AutoOneOf_TargetAndConfigurationProducer_TargetAndConfigurationError
          .inconsistentNullConfig(e);
    }
  }

  // -------------------- Input --------------------
  private final ConfiguredTargetKey preRuleTransitionKey;
  @Nullable private final TransitionFactory<RuleTransitionData> trimmingTransitionFactory;
  private final StarlarkTransitionCache transitionCache;

  private final TransitiveDependencyState transitiveState;

  // -------------------- Output --------------------
  private final ResultSink sink;

  // -------------------- Internal State --------------------
  private Target target;

  public TargetAndConfigurationProducer(
      ConfiguredTargetKey preRuleTransitionKey,
      @Nullable TransitionFactory<RuleTransitionData> trimmingTransitionFactory,
      StarlarkTransitionCache transitionCache,
      TransitiveDependencyState transitiveState,
      ResultSink sink) {
    this.preRuleTransitionKey = preRuleTransitionKey;
    this.trimmingTransitionFactory = trimmingTransitionFactory;
    this.transitionCache = transitionCache;
    this.transitiveState = transitiveState;
    this.sink = sink;
  }

  @Override
  public StateMachine step(Tasks tasks, ExtendedEventHandler listener) {
    return new TargetProducer(
        preRuleTransitionKey.getLabel(),
        transitiveState,
        (TargetProducer.ResultSink) this,
        /* runAfter= */ this::determineConfiguration);
  }

  @Override
  public void acceptTarget(Target target) {
    this.target = target;
  }

  @Override
  public void acceptTargetError(NoSuchPackageException e) {
    emitError(e.getMessage(), /* location= */ null, e.getDetailedExitCode());
  }

  @Override
  public void acceptTargetError(NoSuchTargetException e, Location location) {
    emitError(e.getMessage(), location, e.getDetailedExitCode());
  }

  private StateMachine determineConfiguration(Tasks tasks, ExtendedEventHandler listener) {
    if (target == null) {
      return DONE; // There was an error.
    }

    // TODO(b/261521010): after removing the rule transition from dependency resolution, remove
    // this. It won't be possible afterwards because null configuration keys will only be used for
    // visibility dependencies.
    BuildConfigurationKey configurationKey = preRuleTransitionKey.getConfigurationKey();
    if (configurationKey == null) {
      if (target.isConfigurable()) {
        // We somehow ended up in a target that requires a non-null configuration but with a key
        // that doesn't have a configuration. This is always an error, but we need to bubble this
        // up to the parent to provide more context.
        sink.acceptTargetAndConfigurationError(
            TargetAndConfigurationError.of(new InconsistentNullConfigException()));
        return DONE;
      }
      sink.acceptTargetAndConfiguration(
          new TargetAndConfiguration(target, /* configuration= */ null), preRuleTransitionKey);
      return DONE;
    }

    if (!target.isConfigurable()) {
      // If target is not configurable, but requested with a configuration. Delegates to a key with
      // the null configuration. This is expected to be uncommon. The common case of a
      // non-configurable target is an input file, but those are usually package local and requested
      // correctly with the null configuration.
      delegateTo(
          tasks,
          ConfiguredTargetKey.builder()
              .setLabel(preRuleTransitionKey.getLabel())
              .setExecutionPlatformLabel(preRuleTransitionKey.getExecutionPlatformLabel())
              .build()
              .toKey());
      return DONE;
    }

    return new RuleTransitionApplier();
  }

  /** Applies any requested rule transition before producing the final configuration. */
  private class RuleTransitionApplier
      implements StateMachine,
          TransitionApplier.ResultSink,
          ValueOrExceptionSink<InvalidConfigurationException> {
    // -------------------- Internal State --------------------
    private BuildConfigurationKey configurationKey;
    private ConfiguredTargetKey fullKey;

    @Override
    public StateMachine step(Tasks tasks, ExtendedEventHandler listener) {
      ConfigurationTransition transition =
          computeTransition(target.getAssociatedRule(), trimmingTransitionFactory);
      if (transition == null) {
        this.configurationKey = preRuleTransitionKey.getConfigurationKey();
        return processTransitionedKey(tasks, listener);
      }

      return new TransitionApplier(
          preRuleTransitionKey.getConfigurationKey(),
          transition,
          transitionCache,
          (TransitionApplier.ResultSink) this,
          /* runAfter= */ this::processTransitionedKey);
    }

    @Override
    public void acceptTransitionedConfigurations(
        ImmutableMap<String, BuildConfigurationKey> transitionResult) {
      checkState(transitionResult.size() == 1, "Expected exactly one result: %s", transitionResult);
      this.configurationKey =
          checkNotNull(
              transitionResult.get(ConfigurationTransition.PATCH_TRANSITION_KEY),
              "Transition result missing patch transition entry: %s",
              transitionResult);
    }

    @Override
    public void acceptTransitionError(TransitionException e) {
      emitTransitionErrorMessage(e.getMessage());
    }

    @Override
    public void acceptTransitionError(OptionsParsingException e) {
      emitTransitionErrorMessage(e.getMessage());
    }

    private StateMachine processTransitionedKey(Tasks tasks, ExtendedEventHandler listener) {
      if (configurationKey == null) {
        return DONE; // There was an error.
      }

      if (!configurationKey.equals(preRuleTransitionKey.getConfigurationKey())) {
        delegateTo(
            tasks,
            ConfiguredTargetKey.builder()
                .setLabel(preRuleTransitionKey.getLabel())
                .setExecutionPlatformLabel(preRuleTransitionKey.getExecutionPlatformLabel())
                .setConfigurationKey(configurationKey)
                .build()
                .toKey());
        return DONE;
      } else {
        fullKey = preRuleTransitionKey;
      }

      // This key owns the configuration and the computation completes normally.
      tasks.lookUp(
          configurationKey,
          InvalidConfigurationException.class,
          (ValueOrExceptionSink<InvalidConfigurationException>) this);
      return DONE;
    }

    @Override
    public void acceptValueOrException(
        @Nullable SkyValue value, @Nullable InvalidConfigurationException error) {
      if (value != null) {
        sink.acceptTargetAndConfiguration(
            new TargetAndConfiguration(target, (BuildConfigurationValue) value), fullKey);
        return;
      }
      emitTransitionErrorMessage(error.getMessage());
    }

    private void emitTransitionErrorMessage(String message) {
      // The target must be a rule because these errors happen during the Rule transition.
      Rule rule = target.getAssociatedRule();
      emitError(message, rule.getLocation(), /* exitCode= */ null);
    }
  }

  private void delegateTo(Tasks tasks, ActionLookupKey delegate) {
    tasks.lookUp(delegate, (Consumer<SkyValue>) this);
  }

  @Override
  public void accept(SkyValue value) {
    sink.acceptTargetAndConfigurationDelegatedValue((ConfiguredTargetValue) value);
  }

  private void emitError(
      String message, @Nullable Location location, @Nullable DetailedExitCode exitCode) {
    sink.acceptTargetAndConfigurationError(
        TargetAndConfigurationError.of(
            new ConfiguredValueCreationException(
                location,
                message,
                preRuleTransitionKey.getLabel(),
                configurationId(preRuleTransitionKey.getConfigurationKey()),
                /* rootCauses= */ null,
                exitCode)));
  }

  // Public for Cquery.
  @Nullable
  public static ConfigurationTransition computeTransition(
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
