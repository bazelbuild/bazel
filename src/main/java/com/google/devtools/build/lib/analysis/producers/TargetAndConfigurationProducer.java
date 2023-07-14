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
import com.google.devtools.build.lib.analysis.config.ConfigurationTransitionEvent;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.StarlarkTransitionCache;
import com.google.devtools.build.lib.analysis.config.transitions.ComposingTransition;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.analysis.starlark.StarlarkTransition.TransitionException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleTransitionData;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
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
    implements StateMachine,
        StateMachine.ValueOrExceptionSink<InvalidConfigurationException>,
        Consumer<SkyValue>,
        TargetProducer.ResultSink {
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
      NO_SUCH_THING,
      INCONSISTENT_NULL_CONFIG
    }

    public abstract Kind kind();

    public abstract ConfiguredValueCreationException configuredValueCreation();

    public abstract NoSuchThingException noSuchThing();

    public abstract InconsistentNullConfigException inconsistentNullConfig();

    private static TargetAndConfigurationError of(ConfiguredValueCreationException e) {
      return AutoOneOf_TargetAndConfigurationProducer_TargetAndConfigurationError
          .configuredValueCreation(e);
    }

    private static TargetAndConfigurationError of(NoSuchThingException e) {
      return AutoOneOf_TargetAndConfigurationProducer_TargetAndConfigurationError.noSuchThing(e);
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
  private final ExtendedEventHandler eventHandler;

  // -------------------- Internal State --------------------
  private Target target;

  public TargetAndConfigurationProducer(
      ConfiguredTargetKey preRuleTransitionKey,
      @Nullable TransitionFactory<RuleTransitionData> trimmingTransitionFactory,
      StarlarkTransitionCache transitionCache,
      TransitiveDependencyState transitiveState,
      ResultSink sink,
      ExtendedEventHandler eventHandler) {
    this.preRuleTransitionKey = preRuleTransitionKey;
    this.trimmingTransitionFactory = trimmingTransitionFactory;
    this.transitionCache = transitionCache;
    this.transitiveState = transitiveState;
    this.sink = sink;
    this.eventHandler = eventHandler;
  }

  @Override
  public StateMachine step(Tasks tasks) {
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
    eventHandler.handle(Event.error(e.getMessage()));
    sink.acceptTargetAndConfigurationError(TargetAndConfigurationError.of(e));
  }

  @Override
  public void acceptTargetError(NoSuchTargetException e, Location location) {
    eventHandler.handle(Event.error(location, e.getMessage()));
    sink.acceptTargetAndConfigurationError(TargetAndConfigurationError.of(e));
  }

  private StateMachine determineConfiguration(Tasks tasks) {
    if (target == null) {
      return DONE; // A target could not be determined.
    }

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
              .build());
      return DONE;
    }

    if (!preRuleTransitionKey.shouldApplyRuleTransition()) {
      lookUpConfigurationValue(tasks);
      return DONE;
    }

    ConfigurationTransition transition =
        computeTransition(target.getAssociatedRule(), trimmingTransitionFactory);
    if (transition == null) {
      lookUpConfigurationValue(tasks);
      return DONE;
    }

    return new RuleTransitionApplier(transition);
  }

  private void delegateTo(Tasks tasks, ActionLookupKey delegate) {
    tasks.lookUp(delegate, (Consumer<SkyValue>) this);
  }

  @Override
  public void accept(SkyValue value) {
    sink.acceptTargetAndConfigurationDelegatedValue((ConfiguredTargetValue) value);
  }

  private void lookUpConfigurationValue(Tasks tasks) {
    tasks.lookUp(
        preRuleTransitionKey.getConfigurationKey(),
        InvalidConfigurationException.class,
        (ValueOrExceptionSink<InvalidConfigurationException>) this);
  }

  @Override
  public void acceptValueOrException(
      @Nullable SkyValue value, @Nullable InvalidConfigurationException error) {
    if (value != null) {
      sink.acceptTargetAndConfiguration(
          new TargetAndConfiguration(target, (BuildConfigurationValue) value),
          preRuleTransitionKey);
      return;
    }
    emitError(
        error.getMessage(), TargetUtils.getLocationMaybe(target), error.getDetailedExitCode());
  }

  /**
   * Applies the requested rule transition.
   *
   * <p>When the rule transition results in a new configuration, performs an idempotency check and
   * constructs a delegate {@link ConfiguredTargetKey} with the appropriate {@link
   * ConfiguredTargetKey#shouldApplyRuleTransition} value. Otherwise, just looks up the
   * configuration.
   */
  private class RuleTransitionApplier implements StateMachine, TransitionApplier.ResultSink {
    // -------------------- Input --------------------
    private final ConfigurationTransition transition;
    // -------------------- Internal State --------------------
    private BuildConfigurationKey configurationKey;

    private RuleTransitionApplier(ConfigurationTransition transition) {
      this.transition = transition;
    }

    @Override
    public StateMachine step(Tasks tasks) {
      return new TransitionApplier(
          preRuleTransitionKey.getConfigurationKey(),
          transition,
          transitionCache,
          (TransitionApplier.ResultSink) this,
          eventHandler,
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

    private StateMachine processTransitionedKey(Tasks tasks) {
      if (configurationKey == null) {
        return DONE; // There was an error.
      }

      BuildConfigurationKey parentConfiguration = preRuleTransitionKey.getConfigurationKey();
      if (configurationKey.equals(parentConfiguration)) {
        // This key owns the configuration and the computation completes normally.
        lookUpConfigurationValue(tasks);
        return DONE;
      }

      eventHandler.post(
          ConfigurationTransitionEvent.create(
              parentConfiguration.getOptionsChecksum(), configurationKey.getOptionsChecksum()));

      return new IdempotencyChecker();
    }

    /**
     * Checks the transition for idempotency before applying delegation.
     *
     * <p>If the transition is non-idempotent, marks {@link
     * ConfiguredTargetKey#shouldApplyRuleTransition} false in the delegate key.
     */
    private class IdempotencyChecker implements StateMachine, TransitionApplier.ResultSink {
      /* At first glance, it seems like setting `shouldApplyRuleTransition=false` should be benign
       * in both cases, but it would be an error in the idempotent case.
       *
       * Idempotent Case
       *
       * If we were to mark the idempotent case with `shouldApplyRuleTransition=false`, it would
       * lead to action conflicts. Let `//foo[123]` be a key that rule transitions to `//foo[abc]`
       * and suppose the outcome is marked `//foo[abc] shouldApplyRuleTransition=false`.
       *
       * A different parent might directly request `//foo[abc] shouldApplyRuleTransition=true`.
       * Since the rule transition is a idempotent, it would result in the same actions as
       * `//foo[abc] shouldApplyRuleTransition=false` with a different key, causing action
       * conflicts.
       *
       * Non-idempotent Case
       *
       * In the example of //foo[abc] shouldApplyRuleTransition=false and //foo[abc]
       * shouldApplyRuleTransition=true, there should be no action conflicts because the
       * `shouldApplyRuleTransition=false` is the result of a non-idempotent rule transition and
       * `shouldApplyRuleTransition=true` will produce a different configuration. */

      // -------------------- Internal State --------------------
      private BuildConfigurationKey configurationKey2;

      @Override
      public StateMachine step(Tasks tasks) {
        return new TransitionApplier(
            configurationKey,
            transition,
            transitionCache,
            (TransitionApplier.ResultSink) this,
            eventHandler,
            /* runAfter= */ this::checkIdempotencyAndDelegate);
      }

      @Override
      public void acceptTransitionedConfigurations(
          ImmutableMap<String, BuildConfigurationKey> transitionResult) {
        checkState(
            transitionResult.size() == 1, "Expected exactly one result: %s", transitionResult);
        this.configurationKey2 =
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

      private StateMachine checkIdempotencyAndDelegate(Tasks tasks) {
        if (configurationKey2 == null) {
          return DONE; // There was an error.
        }

        ConfiguredTargetKey.Builder keyBuilder =
            ConfiguredTargetKey.builder()
                .setLabel(preRuleTransitionKey.getLabel())
                .setExecutionPlatformLabel(preRuleTransitionKey.getExecutionPlatformLabel())
                .setConfigurationKey(configurationKey);

        if (!configurationKey.equals(configurationKey2)) {
          // The transition was not idempotent. Explicitly informs the delegate to avoid applying a
          // rule transition.
          keyBuilder.setShouldApplyRuleTransition(false);
        }
        delegateTo(tasks, keyBuilder.build());
        return DONE;
      }
    }

    private void emitTransitionErrorMessage(String message) {
      emitError(message, TargetUtils.getLocationMaybe(target), /* exitCode= */ null);
    }
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
