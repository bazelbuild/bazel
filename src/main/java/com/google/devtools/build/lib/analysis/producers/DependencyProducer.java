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
import static com.google.devtools.build.lib.analysis.DependencyKind.OUTPUT_FILE_RULE_DEPENDENCY;
import static com.google.devtools.build.lib.analysis.DependencyKind.VISIBILITY_DEPENDENCY;
import static com.google.devtools.build.lib.analysis.DependencyResolver.getExecutionPlatformLabel;
import static com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition.PATCH_TRANSITION_KEY;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.AnalysisRootCauseEvent;
import com.google.devtools.build.lib.analysis.DependencyKind;
import com.google.devtools.build.lib.analysis.DependencyKind.ToolchainDependencyKind;
import com.google.devtools.build.lib.analysis.DependencyResolver;
import com.google.devtools.build.lib.analysis.DependencyResolver.ExecutionPlatformResult;
import com.google.devtools.build.lib.analysis.InvalidVisibilityDependencyException;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.DependencyEvaluationException;
import com.google.devtools.build.lib.analysis.starlark.StarlarkTransition.TransitionException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeTransitionData;
import com.google.devtools.build.lib.skyframe.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ConfiguredValueCreationException;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.StateMachine;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * Evaluates dependencies.
 *
 * <p>A dependency is described by a {@link DependencyKind}, a {@link Label} and possibly a list of
 * {@link Aspect}s. This class determines the {@link AttributeConfiguration}, based on the parent's
 * configuration. This may include using the {@link TransitionApplier} to perform an attribute
 * configuration transition.
 *
 * <p>It then delegates computation of the {@link ConfiguredTargetAndData} prerequisite values to
 * {@link PrerequisitesProducer} with the determined configuration(s).
 */
final class DependencyProducer
    implements StateMachine, TransitionApplier.ResultSink, PrerequisitesProducer.ResultSink {
  private static final ConfiguredTargetAndData[] EMPTY_OUTPUT = new ConfiguredTargetAndData[0];

  interface ResultSink {
    /**
     * Accepts dependency values for a given kind and label.
     *
     * <p>Multiple values may occur if there is a split transition.
     *
     * <p>For a skipped dependency, outputs an empty array. See comments in {@link
     * DependencyResolver#getExecutionPlatformLabel} for when this happens.
     */
    void acceptDependencyValues(int index, ConfiguredTargetAndData[] values);

    void acceptDependencyError(DependencyError error);
  }

  // -------------------- Input --------------------
  private final PrerequisiteParameters parameters;
  private final DependencyKind kind;
  private final Label toLabel;
  private final ImmutableList<Aspect> propagatingAspects;

  // -------------------- Output --------------------
  private final ResultSink sink;
  private final int index;

  // -------------------- Internal State --------------------
  private ImmutableMap<String, BuildConfigurationKey> transitionedConfigurations;

  DependencyProducer(
      PrerequisiteParameters parameters,
      DependencyKind kind,
      Label toLabel,
      ImmutableList<Aspect> propagatingAspects,
      ResultSink sink,
      int index) {
    this.parameters = parameters;
    this.kind = checkNotNull(kind);
    this.toLabel = toLabel;
    this.propagatingAspects = propagatingAspects;
    this.sink = sink;
    this.index = index;
  }

  @Override
  public StateMachine step(Tasks tasks, ExtendedEventHandler listener) {
    @Nullable Attribute attribute = kind.getAttribute();

    if (kind == VISIBILITY_DEPENDENCY
        || (attribute != null && attribute.getName().equals("visibility"))) {
      // This is always a null transition because visibility targets are not configurable.
      return computePrerequisites(
          AttributeConfiguration.ofVisibility(), /* executionPlatformLabel= */ null);
    }

    // The logic of `DependencyResolver.computeDependencyLabels` implies that
    // `parameters.configurationKey()` is non-null for everything that follows.
    BuildConfigurationKey configurationKey = checkNotNull(parameters.configurationKey());

    if (DependencyKind.isToolchain(kind)) {
      // There's no attribute so no attribute transition.

      // This dependency is a toolchain. Its package has not been loaded and therefore we can't
      // determine which aspects and which rule configuration transition we should use, so just
      // use sensible defaults. Not depending on their package makes the error message reporting
      // a missing toolchain a bit better.
      // TODO(lberki): This special-casing is weird. Find a better way to depend on toolchains.
      // This logic needs to stay in sync with the dep finding logic in
      // //third_party/bazel/src/main/java/com/google/devtools/build/lib/analysis/Util.java#findImplicitDeps.
      return computePrerequisites(
          AttributeConfiguration.ofUnary(configurationKey),
          parameters.getExecutionPlatformLabel(
              ((ToolchainDependencyKind) kind).getExecGroupName()));
    }

    if (kind == OUTPUT_FILE_RULE_DEPENDENCY) {
      // There's no attribute so no attribute transition.
      return computePrerequisites(
          AttributeConfiguration.ofUnary(configurationKey), /* executionPlatformLabel= */ null);
    }

    var transitionData = AttributeTransitionData.builder().attributes(parameters.attributeMap());
    ExecutionPlatformResult executionPlatformResult =
        getExecutionPlatformLabel(kind, parameters.toolchainContexts(), parameters.aspects());
    switch (executionPlatformResult.kind()) {
      case LABEL:
        transitionData.executionPlatform(executionPlatformResult.label());
        break;
      case NULL_LABEL:
        transitionData.executionPlatform(null);
        break;
      case SKIP:
        sink.acceptDependencyValues(index, EMPTY_OUTPUT);
        return DONE;
      case ERROR:
        return new ExecGroupErrorEmitter(executionPlatformResult.error());
    }
    return new TransitionApplier(
        configurationKey,
        attribute.getTransitionFactory().create(transitionData.build()),
        parameters.transitionCache(),
        (TransitionApplier.ResultSink) this,
        /* runAfter= */ this::processTransitionResult);
  }

  @Override
  public void acceptTransitionedConfigurations(
      ImmutableMap<String, BuildConfigurationKey> transitionedConfigurations) {
    this.transitionedConfigurations = transitionedConfigurations;
  }

  @Override
  public void acceptTransitionError(TransitionException e) {
    sink.acceptDependencyError(DependencyError.of(e));
  }

  @Override
  public void acceptTransitionError(OptionsParsingException e) {
    sink.acceptDependencyError(DependencyError.of(e));
  }

  private StateMachine processTransitionResult(Tasks tasks, ExtendedEventHandler listener) {
    if (transitionedConfigurations == null) {
      return DONE; // There was a previously reported error.
    }

    AttributeConfiguration configuration;
    if (transitionedConfigurations.size() == 1
        && transitionedConfigurations.keySet().iterator().next().equals(PATCH_TRANSITION_KEY)) {
      // Drops the transition key if it was a patch transition.
      configuration =
          AttributeConfiguration.ofUnary(transitionedConfigurations.get(PATCH_TRANSITION_KEY));
    } else {
      configuration = AttributeConfiguration.ofSplit(transitionedConfigurations);
    }
    return computePrerequisites(configuration, /* executionPlatformLabel= */ null);
  }

  private StateMachine computePrerequisites(
      AttributeConfiguration configuration, @Nullable Label executionPlatformLabel) {
    return new PrerequisitesProducer(
        parameters,
        toLabel,
        executionPlatformLabel,
        configuration,
        propagatingAspects,
        (PrerequisitesProducer.ResultSink) this);
  }

  @Override
  public void acceptPrerequisitesValue(ConfiguredTargetAndData[] value) {
    sink.acceptDependencyValues(index, value);
  }

  @Override
  public void acceptPrerequisitesError(InvalidVisibilityDependencyException error) {
    sink.acceptDependencyError(DependencyError.of(error));
  }

  @Override
  public void acceptPrerequisitesCreationError(ConfiguredValueCreationException error) {
    sink.acceptDependencyError(DependencyError.of(error));
  }

  @Override
  public void acceptPrerequisitesAspectError(DependencyEvaluationException error) {
    sink.acceptDependencyError(DependencyError.of(error));
  }

  /**
   * Emits errors from {@link ExecutionPlatformResult#error}.
   *
   * <p>Exists to fetch the {@link BuildConfigurationValue}, needed to construct {@link
   * AnalysisRootCauseEvent}.
   */
  private class ExecGroupErrorEmitter implements StateMachine, Consumer<SkyValue> {
    // -------------------- Input --------------------
    private final String message;

    // -------------------- Internal State --------------------
    private BuildConfigurationValue configuration;

    private ExecGroupErrorEmitter(String message) {
      this.message = message;
    }

    @Override
    public StateMachine step(Tasks tasks, ExtendedEventHandler listener) {
      // The configuration value should already exist as a dependency so this lookup is safe enough
      // for error handling.
      tasks.lookUp(parameters.configurationKey(), (Consumer<SkyValue>) this);
      return this::postEvent;
    }

    @Override
    public void accept(SkyValue value) {
      this.configuration = (BuildConfigurationValue) value;
    }

    private StateMachine postEvent(Tasks tasks, ExtendedEventHandler listener) {
      listener.post(AnalysisRootCauseEvent.withConfigurationValue(configuration, toLabel, message));
      sink.acceptDependencyError(
          DependencyError.of(
              new DependencyEvaluationException(
                  new ConfiguredValueCreationException(
                      parameters.location(),
                      message,
                      toLabel,
                      parameters.eventId(),
                      /* rootCauses= */ null,
                      /* detailedExitCode= */ null),
                  // This error originates in dependency resolution, attached to the current target,
                  // so no dependency has reported the error.
                  /* depReportedOwnError= */ false)));
      return DONE;
    }
  }
}
