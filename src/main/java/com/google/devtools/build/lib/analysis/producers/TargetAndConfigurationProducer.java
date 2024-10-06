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
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.ExecGroupCollection;
import com.google.devtools.build.lib.analysis.InconsistentNullConfigException;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.TransitiveDependencyState;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.ConfigConditions;
import com.google.devtools.build.lib.analysis.config.ConfigurationTransitionEvent;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.StarlarkTransitionCache;
import com.google.devtools.build.lib.analysis.config.ToolchainTypeRequirement;
import com.google.devtools.build.lib.analysis.config.transitions.ComposingTransitionFactory;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformValue;
import com.google.devtools.build.lib.analysis.starlark.StarlarkTransition.TransitionException;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.ConfigurationId;
import com.google.devtools.build.lib.causes.AnalysisFailedCause;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleTransitionData;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.server.FailureDetails.Analysis;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.ConfiguredValueCreationException;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.config.PlatformMappingException;
import com.google.devtools.build.lib.skyframe.toolchains.PlatformLookupUtil.InvalidPlatformException;
import com.google.devtools.build.lib.skyframe.toolchains.ToolchainContextKey;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.StateMachine;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
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
  private final PatchTransition toolchainTaggedTrimmingTransition;
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
      PatchTransition toolchainTaggedTrimmingTransition,
      StarlarkTransitionCache transitionCache,
      TransitiveDependencyState transitiveState,
      ResultSink sink,
      ExtendedEventHandler eventHandler) {
    this.preRuleTransitionKey = preRuleTransitionKey;
    this.trimmingTransitionFactory = trimmingTransitionFactory;
    this.toolchainTaggedTrimmingTransition = toolchainTaggedTrimmingTransition;
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

    return new RuleTransitionApplier();
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

  /** Applies any requested rule transition before producing the final configuration. */
  private class RuleTransitionApplier
      implements StateMachine,
          TransitionApplier.ResultSink,
          ConfigConditionsProducer.ResultSink,
          PlatformProducer.ResultSink {
    // -------------------- Internal State --------------------
    @Nullable private PlatformInfo platformInfo;
    private ConfigConditions configConditions;
    private ConfigurationTransition ruleTransition;
    private BuildConfigurationKey configurationKey;

    @Override
    public StateMachine step(Tasks tasks) throws InterruptedException {

      UnloadedToolchainContextsInputs unloadedToolchainContextsInputs =
          getUnloadedToolchainContextsInputs(
              target, preRuleTransitionKey.getExecutionPlatformLabel());

      if (unloadedToolchainContextsInputs.targetToolchainContextKey() != null) {
        PlatformConfiguration platformConfiguration =
            new PlatformConfiguration(preRuleTransitionKey.getConfigurationKey().getOptions());
        tasks.enqueue(
            new PlatformProducer(
                platformConfiguration.getTargetPlatform(),
                (PlatformProducer.ResultSink) this,
                this::computeConfigConditions));
      } else {
        this.platformInfo = null;
        computeConfigConditions(tasks);
      }

      return DONE;
    }

    // TODO: @aranguyen b/297077082
    public UnloadedToolchainContextsInputs getUnloadedToolchainContextsInputs(
        Target target, @Nullable Label parentExecutionPlatformLabel) throws InterruptedException {
      Rule rule = target.getAssociatedRule();
      if (rule == null) {
        return UnloadedToolchainContextsInputs.empty();
      }

      var platformOptions =
          preRuleTransitionKey.getConfigurationKey().getOptions().get(PlatformOptions.class);
      if (platformOptions == null) {
        return UnloadedToolchainContextsInputs.empty();
      }
      PlatformConfiguration platformConfiguration = new PlatformConfiguration(platformOptions);
      var defaultExecConstraintLabels =
          getExecutionPlatformConstraints(rule, platformConfiguration);
      var ruleClass = rule.getRuleClassObject();
      boolean useAutoExecGroups =
          rule.getRuleClassObject()
              .getAutoExecGroupsMode()
              .isEnabled(
                  RawAttributeMapper.of(rule),
                  preRuleTransitionKey
                      .getConfigurationKey()
                      .getOptions()
                      .get(CoreOptions.class)
                      .useAutoExecGroups);

      var processedExecGroups =
          ExecGroupCollection.process(
              ruleClass.getExecGroups(),
              defaultExecConstraintLabels,
              ruleClass.getToolchainTypes(),
              useAutoExecGroups);

      if (!rule.useToolchainResolution()) {
        return UnloadedToolchainContextsInputs.create(processedExecGroups, null);
      }

      return UnloadedToolchainContextsInputs.create(
          processedExecGroups,
          createDefaultToolchainContextKey(
              computeToolchainConfigurationKey(
                  preRuleTransitionKey.getConfigurationKey().getOptions(),
                  toolchainTaggedTrimmingTransition),
              defaultExecConstraintLabels,
              /* debugTarget= */ platformConfiguration.debugToolchainResolution(rule.getLabel()),
              /* useAutoExecGroups= */ useAutoExecGroups,
              ruleClass.getToolchainTypes(),
              parentExecutionPlatformLabel));
    }

    public ToolchainContextKey createDefaultToolchainContextKey(
        BuildConfigurationKey configurationKey,
        ImmutableSet<Label> defaultExecConstraintLabels,
        boolean debugTarget,
        boolean useAutoExecGroups,
        ImmutableSet<ToolchainTypeRequirement> toolchainTypes,
        @Nullable Label parentExecutionPlatformLabel) {
      ToolchainContextKey.Builder toolchainContextKeyBuilder =
          ToolchainContextKey.key()
              .configurationKey(configurationKey)
              .execConstraintLabels(defaultExecConstraintLabels)
              .debugTarget(debugTarget);

      // Add toolchain types only if automatic exec groups are not created for this target.
      if (!useAutoExecGroups) {
        toolchainContextKeyBuilder.toolchainTypes(toolchainTypes);
      }

      if (parentExecutionPlatformLabel != null) {
        // Find out what execution platform the parent used, and force that.
        // This should only be set for direct toolchain dependencies.
        toolchainContextKeyBuilder.forceExecutionPlatform(parentExecutionPlatformLabel);
      }
      return toolchainContextKeyBuilder.build();
    }

    private BuildConfigurationKey computeToolchainConfigurationKey(
        BuildOptions buildOptions, PatchTransition toolchainTaggedTrimmingTransition)
        throws InterruptedException {
      // The toolchain context's options are the parent rule's options with manual trimming
      // auto-applied. This means toolchains don't inherit feature flags. This helps build
      // performance: if the toolchain context had the exact same configuration of its parent and
      // that
      // included feature flags, all the toolchain's dependencies would apply this transition
      // individually. That creates a lot more potentially expensive applications of that transition
      // (especially since manual trimming applies to every configured target in the build).
      //
      // In other words: without this modification:
      // parent rule -> toolchain context -> toolchain
      //     -> toolchain dep 1 # applies manual trimming to remove feature flags
      //     -> toolchain dep 2 # applies manual trimming to remove feature flags
      //     ...
      //
      // With this modification:
      // parent rule -> toolchain context # applies manual trimming to remove feature flags
      //     -> toolchain
      //         -> toolchain dep 1
      //         -> toolchain dep 2
      //         ...
      //
      // None of this has any effect on rules that don't utilize manual trimming.
      BuildOptions toolchainOptions =
          toolchainTaggedTrimmingTransition.patch(
              new BuildOptionsView(
                  buildOptions, toolchainTaggedTrimmingTransition.requiresOptionFragments()),
              eventHandler);
      return BuildConfigurationKey.create(toolchainOptions);
    }

    private ImmutableSet<Label> getExecutionPlatformConstraints(
        Rule rule, @Nullable PlatformConfiguration platformConfiguration) {
      if (platformConfiguration == null) {
        return ImmutableSet.of(); // See NoConfigTransition.
      }
      NonconfigurableAttributeMapper mapper = NonconfigurableAttributeMapper.of(rule);
      ImmutableSet.Builder<Label> execConstraintLabels = new ImmutableSet.Builder<>();

      execConstraintLabels.addAll(rule.getRuleClassObject().getExecutionPlatformConstraints());
      if (rule.getRuleClassObject()
          .hasAttr(RuleClass.EXEC_COMPATIBLE_WITH_ATTR, BuildType.LABEL_LIST)) {
        execConstraintLabels.addAll(
            mapper.get(RuleClass.EXEC_COMPATIBLE_WITH_ATTR, BuildType.LABEL_LIST));
      }

      execConstraintLabels.addAll(
          platformConfiguration.getAdditionalExecutionConstraintsFor(rule.getLabel()));

      return execConstraintLabels.build();
    }

    @CanIgnoreReturnValue
    public StateMachine computeConfigConditions(Tasks tasks) {
      // TODO @aranguyen b/297077082
      tasks.enqueue(
          new ConfigConditionsProducer(
              target,
              preRuleTransitionKey.getLabel(),
              preRuleTransitionKey.getConfigurationKey(),
              platformInfo,
              transitiveState,
              (ConfigConditionsProducer.ResultSink) this,
              this::computeTransition));
      return DONE;
    }

    @Override
    public void acceptConfigConditions(ConfigConditions configConditions) {
      this.configConditions = configConditions;
    }

    @Override
    public void acceptConfigConditionsError(ConfiguredValueCreationException e) {
      emitErrorMessage(e.getMessage());
    }

    // Keep in sync with CqueryTransitionResolver.getRuleTransition.
    public StateMachine computeTransition(Tasks tasks) {
      if (configConditions == null) {
        return DONE;
      }

      TransitionFactory<RuleTransitionData> transitionFactory =
          target.getAssociatedRule().getRuleClassObject().getTransitionFactory();
      if (trimmingTransitionFactory != null) {
        transitionFactory =
            ComposingTransitionFactory.of(transitionFactory, trimmingTransitionFactory);
      }

      var transitionData =
          RuleTransitionData.create(
              target.getAssociatedRule(),
              configConditions.asProviders(),
              preRuleTransitionKey.getConfigurationKey().getOptionsChecksum());

      ConfigurationTransition transition = transitionFactory.create(transitionData);
      this.ruleTransition = transition;

      return new TransitionApplier(
          preRuleTransitionKey.getConfigurationKey(),
          ruleTransition,
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
      emitErrorMessage(e.getMessage());
    }

    @Override
    public void acceptOptionsParsingError(OptionsParsingException e) {
      emitErrorMessage(e.getMessage());
    }

    @Override
    public void acceptPlatformMappingError(PlatformMappingException e) {
      emitErrorMessage(e.getMessage());
    }

    @Override
    public void acceptPlatformFlagsError(InvalidPlatformException e) {
      emitErrorMessage(e.getMessage());
    }

    @Override
    public void acceptPlatformValue(PlatformValue value) {
      this.platformInfo = value.platformInfo();
    }

    @Override
    public void acceptPlatformInfoError(InvalidPlatformException error) {
      emitErrorMessage(error.getMessage());
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
            ruleTransition,
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
        emitErrorMessage(e.getMessage());
      }

      @Override
      public void acceptOptionsParsingError(OptionsParsingException e) {
        emitErrorMessage(e.getMessage());
      }

      @Override
      public void acceptPlatformMappingError(PlatformMappingException e) {
        emitErrorMessage(e.getMessage());
      }

      @Override
      public void acceptPlatformFlagsError(InvalidPlatformException e) {
        emitErrorMessage(e.getMessage());
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

    private void emitErrorMessage(String message) {
      emitError(message, TargetUtils.getLocationMaybe(target), /* exitCode= */ null);
    }
  }

  private void emitError(
      String message, @Nullable Location location, @Nullable DetailedExitCode exitCode) {
    Cause cause =
        new AnalysisFailedCause(
            preRuleTransitionKey.getLabel(),
            configurationIdMessage(preRuleTransitionKey.getConfigurationKey().getOptionsChecksum()),
            exitCode != null ? exitCode : createDetailedExitCode(message));
    sink.acceptTargetAndConfigurationError(
        TargetAndConfigurationError.of(
            new ConfiguredValueCreationException(
                location,
                message,
                target.getLabel(),
                configurationId(preRuleTransitionKey.getConfigurationKey()),
                NestedSetBuilder.create(Order.STABLE_ORDER, cause),
                exitCode != null ? exitCode : createDetailedExitCode(message))));
  }

  public static ConfigurationId configurationIdMessage(@Nullable String optionsCheckSum) {
    if (optionsCheckSum == null) {
      return ConfigurationId.newBuilder().setId("none").build();
    }
    return ConfigurationId.newBuilder().setId(optionsCheckSum).build();
  }

  public static DetailedExitCode createDetailedExitCode(String message) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setAnalysis(
                Analysis.newBuilder().setCode(Analysis.Code.CONFIGURED_VALUE_CREATION_FAILED))
            .build());
  }
}
