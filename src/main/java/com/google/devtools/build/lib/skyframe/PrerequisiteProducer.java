// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.devtools.build.lib.analysis.config.BuildConfigurationValue.configurationId;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.AnalysisRootCauseEvent;
import com.google.devtools.build.lib.analysis.AspectResolver;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.Dependency;
import com.google.devtools.build.lib.analysis.DependencyKey;
import com.google.devtools.build.lib.analysis.DependencyKind;
import com.google.devtools.build.lib.analysis.DependencyResolver;
import com.google.devtools.build.lib.analysis.DuplicateException;
import com.google.devtools.build.lib.analysis.ExecGroupCollection;
import com.google.devtools.build.lib.analysis.InconsistentAspectOrderException;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.ToolchainCollection;
import com.google.devtools.build.lib.analysis.TransitiveDependencyState;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.ConfigConditions;
import com.google.devtools.build.lib.analysis.config.ConfigurationResolver;
import com.google.devtools.build.lib.analysis.config.DependencyEvaluationException;
import com.google.devtools.build.lib.analysis.config.StarlarkTransitionCache;
import com.google.devtools.build.lib.analysis.config.ToolchainTypeRequirement;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionCollector;
import com.google.devtools.build.lib.analysis.constraints.IncompatibleTargetChecker.IncompatibleTargetException;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.producers.DependencyContext;
import com.google.devtools.build.lib.analysis.producers.DependencyContextError;
import com.google.devtools.build.lib.analysis.producers.DependencyContextProducer;
import com.google.devtools.build.lib.analysis.producers.DependencyContextProducerWithCompatibilityCheck;
import com.google.devtools.build.lib.analysis.producers.TargetAndConfigurationProducer;
import com.google.devtools.build.lib.analysis.producers.UnloadedToolchainContextsInputs;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetEvaluationExceptions.ReportedException;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetEvaluationExceptions.UnreportedException;
import com.google.devtools.build.lib.skyframe.toolchains.ToolchainContextKey;
import com.google.devtools.build.lib.skyframe.toolchains.ToolchainException;
import com.google.devtools.build.lib.skyframe.toolchains.UnloadedToolchainContext;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.DetailedExitCode.DetailedExitCodeComparator;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState;
import com.google.devtools.build.skyframe.SkyFunction.LookupEnvironment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import com.google.devtools.build.skyframe.state.Driver;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Helper logic for {@link ConfiguredTargetFunction} and {@link AspectFunction}: performs the
 * analysis phase through computation of prerequisites.
 *
 * <p>For the {@link ConfiguredTargetFunction} this includes:
 *
 * <ul>
 *   <li>getting this target's {@link Target} and {@link BuildConfigurationValue}
 *   <li>getting this target's {@code select()} keys ({@link ConfigConditions}), which are used to
 *       evaluate all rule attributes with {@code select()} and determine exact dependencies
 *   <li>figuring out which toolchains this target needs
 *   <li>getting the {@link ConfiguredTargetValue}s of this target's prerequisites (through
 *       recursive calls to {@link ConfiguredTargetFunction}
 * </ul>
 *
 * <p>Figuring out which toolchains are needed and computing the {@link ConfigConditions} is
 * performed by the {@link DependencyContextProducerWithCompatibilityCheck}, which additionally
 * checks for directly incompatible targets using the {@link
 * IncompatibleTargetChecker.IncompatibleTargetProducer}.
 *
 * <p>Cumulatively, this is enough information to run the target's rule logic.
 *
 * <p>This class also provides getters for the above data for subsequent analysis logic to use.
 *
 * <p>See {@link ConfiguredTargetFunction} for more review on analysis implementation.
 *
 * <p>{@link AspectFunction} shares the logic computing a target's prerequisites via the {@link
 * PrerequisiteProducer#computeDependencies}.
 */
public final class PrerequisiteProducer {
  /**
   * Memoizies computation steps of {@link #evaluate} so they do not need to be repeated on {@code
   * Skyframe} restart.
   */
  @VisibleForTesting
  public static class State implements SkyKeyComputeState, DependencyContextProducer.ResultSink {
    /** Must be set before calling {@link #evaluate}. */
    public TargetAndConfiguration targetAndConfiguration;

    /** Set once {@link #dependencyContextProducer} starts. */
    @VisibleForTesting public ExecGroupCollection.Builder execGroupCollectionBuilder;

    /**
     * Computes the dependency context, comprised of the unloaded toolchain contexts and the config
     * conditions.
     *
     * <p>One of {@link #dependencyContext} or {@link #dependencyContextError} will be set upon
     * completion.
     */
    @Nullable // Non-null when in-flight.
    Driver dependencyContextProducer;

    @VisibleForTesting // package-private
    @Nullable
    public DependencyContext dependencyContext;

    @Nullable DependencyContextError dependencyContextError;

    /** Null if not yet computed or if {@link #resolveConfigurationsResult} is non-null. */
    @Nullable private OrderedSetMultimap<DependencyKind, DependencyKey> dependentNodeMapResult;

    /** Null if not yet computed or if {@link #computeDependenciesResult} is non-null. */
    @Nullable private OrderedSetMultimap<DependencyKind, Dependency> resolveConfigurationsResult;

    /** Null if not yet computed or if {@link #computeDependenciesResult} is non-null. */
    @Nullable
    private Map<ConfiguredTargetKey, ConfiguredTargetAndData>
        resolveConfiguredTargetDependenciesResult;

    /**
     * Non-null if all the work in {@link #computeDependencies} is already done. This field contains
     * the result.
     */
    @Nullable
    private OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> computeDependenciesResult;

    final TransitiveDependencyState transitiveState;

    /**
     * Stores events emitted by memoized computations.
     *
     * <p>Both the {@link #computeDependencies} and the {@link TargetAndConfigurationProducer} may
     * perform Starlark transitions that emit events. Skyframe uses only the events emitted to
     * {@code env.getListener()} on a call to {@link #evaluate} that had no missing deps. Since the
     * computations are memoized, they do not re-emit events when Skyframe restarts. Therefore
     * events are stored and replayed when subsequent Skyframe restarts occur.
     */
    final StoredEventHandler storedEvents = new StoredEventHandler();

    public State(boolean storeTransitivePackages) {
      this.transitiveState =
          new TransitiveDependencyState(storeTransitivePackages, /* prerequisitePackages= */ null);
    }

    public NestedSetBuilder<Cause> transitiveRootCauses() {
      return transitiveState.transitiveRootCauses();
    }

    public NestedSet<Package> transitivePackages() {
      return transitiveState.transitivePackages();
    }

    @Override
    public void acceptDependencyContext(DependencyContext value) {
      this.dependencyContext = value;
    }

    @Override
    public void acceptDependencyContextError(DependencyContextError error) {
      this.dependencyContextError = error;
    }
  }

  /**
   * Thrown if this is an invalid target because it's a rule with a null configuration or a
   * non-null-configured dep of a null-configured target.
   */
  static class InconsistentNullConfigException extends Exception {}

  /** Lets calling logic provide a semaphore to restrict the number of concurrent analysis calls. */
  public interface SemaphoreAcquirer {
    void acquireSemaphore() throws InterruptedException;
  }

  private final TargetAndConfiguration targetAndConfiguration;
  private OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> depValueMap = null;
  private ConfigConditions configConditions = null;
  private PlatformInfo platformInfo = null;
  @Nullable private ToolchainCollection<UnloadedToolchainContext> unloadedToolchainContexts = null;

  public PrerequisiteProducer(TargetAndConfiguration targetAndConfiguration) {
    this.targetAndConfiguration = Preconditions.checkNotNull(targetAndConfiguration);
  }

  /** Return this target's {@link TargetAndConfiguration}. */
  TargetAndConfiguration getTargetAndConfiguration() {
    return targetAndConfiguration;
  }

  /**
   * Return this target's fully resolved dependencies.
   *
   * <p>{@link #evaluate} must be called before this info is available.
   */
  public OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> getDepValueMap() {
    return Preconditions.checkNotNull(depValueMap);
  }

  /**
   * Return the keys in this target's {@code select()}s.
   *
   * <p>{@link #evaluate} must be called before this info is available.
   */
  ConfigConditions getConfigConditions() {
    return Preconditions.checkNotNull(configConditions);
  }

  /**
   * Return this target's platform metadata, or null if it doesn't use platforms.
   *
   * <p>{@link #evaluate} must be called before this info is available.
   */
  @Nullable
  PlatformInfo getPlatformInfo() {
    return platformInfo;
  }

  /**
   * Return this target's toolchain requirements, or null if it doesn't use toolchains.
   *
   * <p>{@link #evaluate} must be called before this info is available.
   */
  @Nullable
  ToolchainCollection<UnloadedToolchainContext> getUnloadedToolchainContexts() {
    return unloadedToolchainContexts;
  }

  /**
   * Runs the analysis phase for this target through prerequisite evaluation.
   *
   * <p>See {@link PrerequisiteProducer} javadoc for details.
   *
   * <p>This is the main entry point to {@link PrerequisiteProducer}. This method runs its share of
   * the analysis phase, after which all the data is computes is accessible to calling code through
   * related getters.
   *
   * <p>After instantiating this class, this method should be called once. It returns false when any
   * Skyframe dependencies need to be evaluated, else true.
   */
  public boolean evaluate(
      State state,
      ConfiguredTargetKey configuredTargetKey,
      RuleClassProvider ruleClassProvider,
      StarlarkTransitionCache transitionCache,
      SemaphoreAcquirer semaphoreLocker,
      TransitionCollector transitionCollector,
      LookupEnvironment env,
      ExtendedEventHandler listener)
      throws ReportedException,
          UnreportedException,
          IncompatibleTargetException,
          InterruptedException {
    // TODO(janakr): this call may tie up this thread indefinitely, reducing the parallelism of
    //  Skyframe. This is a strict improvement over the prior state of the code, in which we ran
    //  with #processors threads, but ideally we would call #tryAcquire here, and if we failed,
    //  would exit this SkyFunction and restart it when permits were available.
    semaphoreLocker.acquireSemaphore();
    try {
      var dependencyContext =
          getDependencyContext(state, configuredTargetKey, ruleClassProvider, env, listener);
      if (dependencyContext == null) {
        return false;
      }
      this.unloadedToolchainContexts = dependencyContext.unloadedToolchainContexts();
      this.platformInfo =
          unloadedToolchainContexts != null ? unloadedToolchainContexts.getTargetPlatform() : null;
      this.configConditions = dependencyContext.configConditions();

      // TODO(ulfjack): ConfiguredAttributeMapper (indirectly used from computeDependencies) isn't
      // safe to use if there are missing config conditions, so we stop here, but only if there are
      // config conditions - though note that we can't check if configConditions is non-empty - it
      // may be empty for other reasons. It would be better to continue here so that we can collect
      // more root causes during computeDependencies.
      // Note that this doesn't apply to AspectFunction, because aspects can't have configurable
      // attributes.
      NestedSetBuilder<Cause> transitiveRootCauses = state.transitiveRootCauses();
      if (!transitiveRootCauses.isEmpty()
          && !Objects.equals(configConditions, ConfigConditions.EMPTY)) {
        NestedSet<Cause> causes = transitiveRootCauses.build();
        listener.handle(
            Event.error(
                targetAndConfiguration.getTarget().getLocation(),
                "Cannot compute config conditions"));
        throw new ReportedException(
            new ConfiguredValueCreationException(
                targetAndConfiguration,
                "Cannot compute config conditions",
                causes,
                getPrioritizedDetailedExitCode(causes)));
      }

      // Calculate the dependencies of this target.
      depValueMap =
          computeDependencies(
              state,
              /* aspects= */ ImmutableList.of(),
              ruleClassProvider,
              transitionCache,
              transitionCollector,
              env,
              listener);
      if (!transitiveRootCauses.isEmpty()) {
        NestedSet<Cause> causes = transitiveRootCauses.build();
        // TODO(bazel-team): consider reporting the error in this class vs. exporting it for
        // BuildTool to handle. Calling code needs to be untangled for that to work and pass tests.
        throw new UnreportedException(
            new ConfiguredValueCreationException(
                targetAndConfiguration,
                "Analysis failed",
                causes,
                getPrioritizedDetailedExitCode(causes)));
      }
      if (depValueMap == null) {
        return false;
      }
    } catch (DependencyEvaluationException
        | ConfiguredValueCreationException
        | AspectCreationException
        | ToolchainException e) {
      // We handle exceptions in a dedicated method to keep this method concise and readable.
      handleException(listener, targetAndConfiguration.getTarget(), e);
    }
    return true;
  }

  @VisibleForTesting
  @Nullable // Null when a Skyframe restart is needed.
  public static DependencyContext getDependencyContext(
      State state,
      ConfiguredTargetKey configuredTargetKey,
      RuleClassProvider ruleClassProvider,
      LookupEnvironment env,
      ExtendedEventHandler listener)
      throws InterruptedException,
          ToolchainException,
          ConfiguredValueCreationException,
          IncompatibleTargetException,
          DependencyEvaluationException {
    if (state.dependencyContext != null) {
      return state.dependencyContext;
    }
    if (state.dependencyContextProducer == null) {
      var targetAndConfiguration = state.targetAndConfiguration;
      var unloadedToolchainContextsInputs =
          getUnloadedToolchainContextsInputs(
              targetAndConfiguration,
              configuredTargetKey.getExecutionPlatformLabel(),
              ruleClassProvider,
              listener);
      state.execGroupCollectionBuilder = unloadedToolchainContextsInputs;
      state.dependencyContextProducer =
          new Driver(
              new DependencyContextProducerWithCompatibilityCheck(
                  targetAndConfiguration,
                  configuredTargetKey,
                  unloadedToolchainContextsInputs,
                  state.transitiveState,
                  (DependencyContextProducer.ResultSink) state));
    }
    if (state.dependencyContextProducer.drive(env, listener)) {
      state.dependencyContextProducer = null;
    }

    // During error bubbling, the state machine might not be done, but still emit an error.
    var error = state.dependencyContextError;
    if (error != null) {
      switch (error.kind()) {
        case TOOLCHAIN:
          throw error.toolchain();
        case CONFIGURED_VALUE_CREATION:
          throw error.configuredValueCreation();
        case INCOMPATIBLE_TARGET:
          throw error.incompatibleTarget();
        case VALIDATION:
          var targetAndConfiguration = state.targetAndConfiguration;
          BuildConfigurationValue configuration = targetAndConfiguration.getConfiguration();
          Label label = targetAndConfiguration.getLabel();
          var validationException = error.validation();
          BuildEventId configurationEventId = configurationId(configuration);
          listener.post(
              AnalysisRootCauseEvent.withConfigurationValue(
                  configuration, label, validationException.getMessage()));
          throw new DependencyEvaluationException(
              new ConfiguredValueCreationException(
                  targetAndConfiguration.getTarget().getLocation(),
                  validationException.getMessage(),
                  label,
                  configurationEventId,
                  null,
                  null),
              // These errors occur within DependencyResolver, which is attached to the current
              // target. i.e. no dependent ConfiguredTargetFunction call happens to report its own
              // error.
              /* depReportedOwnError= */ false);
      }
      throw new IllegalStateException("unreachable");
    }

    return state.dependencyContext; // Null if not yet done.
  }

  /**
   * Handles all exceptions that {@link #evaluate} may throw.
   *
   * <p>This is its own method because there's a lot of logic here and when directly inlined it
   * makes it harder to follow the calling method's control flow.
   */
  private void handleException(ExtendedEventHandler listener, Target target, Exception untyped)
      throws ReportedException {

    if (untyped instanceof DependencyEvaluationException) {
      DependencyEvaluationException e = (DependencyEvaluationException) untyped;
      String errorMessage = e.getMessage();
      if (!e.depReportedOwnError()) {
        listener.handle(Event.error(e.getLocation(), e.getMessage()));
      }

      ConfiguredValueCreationException cvce = null;
      if (e.getCause() instanceof ConfiguredValueCreationException) {
        cvce = (ConfiguredValueCreationException) e.getCause();

        // Check if this is caused by an unresolved toolchain, and report it as such.
        if (unloadedToolchainContexts != null) {
          ImmutableSet<Label> requiredToolchains =
              unloadedToolchainContexts.getResolvedToolchains();
          ImmutableSet<Label> toolchainDependencyErrors =
              cvce.getRootCauses().toList().stream()
                  .map(Cause::getLabel)
                  .filter(requiredToolchains::contains)
                  .collect(toImmutableSet());

          if (!toolchainDependencyErrors.isEmpty()) {
            errorMessage = "errors encountered resolving toolchains for " + target.getLabel();
            listener.handle(Event.error(target.getLocation(), errorMessage));
          }
        }
      }

      throw new ReportedException(
          cvce != null
              ? cvce
              : new ConfiguredValueCreationException(
                  targetAndConfiguration, errorMessage, null, e.getDetailedExitCode()));
    } else if (untyped instanceof ConfiguredValueCreationException) {
      ConfiguredValueCreationException e = (ConfiguredValueCreationException) untyped;
      if (!e.getMessage().isEmpty()) {
        // Report the error to the user.
        listener.handle(Event.error(e.getLocation(), e.getMessage()));
      }
      throw new ReportedException(e);
    } else if (untyped instanceof AspectCreationException) {
      AspectCreationException e = (AspectCreationException) untyped;
      if (!e.getMessage().isEmpty()) {
        // Report the error to the user.
        listener.handle(Event.error(null, e.getMessage()));
      }
      throw new ReportedException(
          new ConfiguredValueCreationException(
              targetAndConfiguration, e.getMessage(), e.getCauses(), e.getDetailedExitCode()));
    } else if (untyped instanceof ToolchainException) {
      ToolchainException e = (ToolchainException) untyped;
      ConfiguredValueCreationException cvce =
          e.asConfiguredValueCreationException(targetAndConfiguration);
      listener.handle(Event.error(target.getLocation(), cvce.getMessage()));
      throw new ReportedException(cvce);
    } else {
      throw new IllegalStateException("unexpected exception with no appropriate handler", untyped);
    }
  }

  /**
   * Returns the target-specific execution platform constraints, based on the rule definition and
   * any constraints added by the target, including those added for the target on the command line.
   */
  public static ImmutableSet<Label> getExecutionPlatformConstraints(
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

  /**
   * Computes the direct dependencies of a node in the configured target graph (a configured target
   * or an aspects).
   *
   * <p>Returns null if Skyframe hasn't evaluated the required dependencies yet. In this case, the
   * caller should also return null to Skyframe.
   *
   * <p>REQUIRES: {@code state.dependencyContext} is populated.
   *
   * @param state the compute state
   * @param ruleClassProvider rule class provider to supply a trimming transition factory
   * @param transitionCollector a callback that observes attribute transitions for Cquery
   * @param env the Skyframe environment
   */
  // TODO(b/213351014): Make the control flow of this helper function more readable. This will
  //   involve making a corresponding change to State to match the control flow.
  @Nullable
  public static OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> computeDependencies(
      State state,
      Iterable<Aspect> aspects,
      RuleClassProvider ruleClassProvider,
      StarlarkTransitionCache transitionCache,
      TransitionCollector transitionCollector,
      LookupEnvironment env,
      ExtendedEventHandler listener)
      throws DependencyEvaluationException,
          ConfiguredValueCreationException,
          AspectCreationException,
          InterruptedException {
    // Replays stored events unless a Skyframe restart is immediately needed and the events would
    // be unused anyway.
    boolean shouldReplayStoredEvents = true;
    var transitiveRootCauses = state.transitiveRootCauses();
    try {
      if (state.computeDependenciesResult != null) {
        return state.computeDependenciesResult;
      }

      TargetAndConfiguration ctgValue = state.targetAndConfiguration;
      OrderedSetMultimap<DependencyKind, Dependency> depValueNames;
      if (state.resolveConfigurationsResult != null) {
        depValueNames = state.resolveConfigurationsResult;
      } else {
        // Create the map from attributes to set of (target, transition) pairs.
        OrderedSetMultimap<DependencyKind, DependencyKey> initialDependencies;
        DependencyContext dependencyContext = state.dependencyContext;
        if (state.dependentNodeMapResult != null) {
          initialDependencies = state.dependentNodeMapResult;
        } else {
          BuildConfigurationValue configuration = ctgValue.getConfiguration();
          Label label = ctgValue.getLabel();
          try {
            initialDependencies =
                new SkyframeDependencyResolver(env, listener)
                    .dependentNodeMap(
                        ctgValue,
                        aspects,
                        dependencyContext.configConditions().asProviders(),
                        dependencyContext.toolchainContexts(),
                        transitiveRootCauses,
                        ((ConfiguredRuleClassProvider) ruleClassProvider)
                            .getTrimmingTransitionFactory(),
                        transitionCollector);
          } catch (DependencyResolver.Failure e) {
            listener.post(
                AnalysisRootCauseEvent.withConfigurationValue(
                    configuration, label, e.getMessage()));
            throw new DependencyEvaluationException(
                new ConfiguredValueCreationException(
                    e.getLocation(),
                    e.getMessage(),
                    label,
                    configurationId(configuration),
                    null,
                    null),
                // These errors occur within DependencyResolver, which is attached to the current
                // target. i.e. no dependent ConfiguredTargetFunction call happens to report its own
                // error.
                /* depReportedOwnError= */ false);
          } catch (InconsistentAspectOrderException e) {
            throw new DependencyEvaluationException(e);
          }
          if (!env.valuesMissing()) {
            state.dependentNodeMapResult = initialDependencies;
          }
        }
        // Trim each dep's configuration so it only includes the fragments needed by its transitive
        // closure.
        ConfigurationResolver configResolver =
            new ConfigurationResolver(
                env, ctgValue, dependencyContext.configConditions().asProviders(), transitionCache);
        StoredEventHandler storedEventHandler = new StoredEventHandler();
        try {
          depValueNames =
              configResolver.resolveConfigurations(initialDependencies, storedEventHandler);
        } catch (ConfiguredValueCreationException e) {
          storedEventHandler.replayOn(state.storedEvents);
          throw e;
        }
        if (!env.valuesMissing()) {
          state.resolveConfigurationsResult = depValueNames;
          storedEventHandler.replayOn(state.storedEvents);

          // We won't need this anymore.
          state.dependentNodeMapResult = null;
        }
      }

      // Return early in case packages were not loaded yet. In theory, we could start configuring
      // dependent targets in loaded packages. However, that creates an artificial sync boundary
      // between loading all dependent packages (fast) and configuring some dependent targets (can
      // have a long tail).
      if (env.valuesMissing()) {
        shouldReplayStoredEvents = false;
        return null;
      }

      var transitivePackages =
          state.transitiveState.storeTransitivePackages()
              ? NestedSetBuilder.<Package>stableOrder()
              : null;
      // Resolve configured target dependencies and handle errors.
      Map<ConfiguredTargetKey, ConfiguredTargetAndData> depValues;
      if (state.resolveConfiguredTargetDependenciesResult != null) {
        depValues = state.resolveConfiguredTargetDependenciesResult;
      } else {
        depValues =
            resolveConfiguredTargetDependencies(
                env, ctgValue, depValueNames.values(), transitivePackages, transitiveRootCauses);
        if (env.valuesMissing()) {
          shouldReplayStoredEvents = false;
          return null;
        }
        state.resolveConfiguredTargetDependenciesResult = depValues;
      }

      // Resolve required aspects.
      OrderedSetMultimap<Dependency, ConfiguredAspect> depAspects =
          AspectResolver.resolveAspectDependencies(
              env, depValues, depValueNames.values(), transitivePackages);
      if (env.valuesMissing()) {
        shouldReplayStoredEvents = false;
        return null;
      }

      // Merge the dependent configured targets and aspects into a single map.
      OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> mergeAspectsResult;
      try {
        mergeAspectsResult = AspectResolver.mergeAspects(depValueNames, depValues, depAspects);
      } catch (DuplicateException e) {
        throw new DependencyEvaluationException(
            new ConfiguredValueCreationException(ctgValue, e.getMessage()),
            /*depReportedOwnError=*/ false);
      }
      state.computeDependenciesResult = mergeAspectsResult;

      if (transitivePackages != null) {
        state.transitiveState.setTransitivePackagesBatch(transitivePackages.build());
      }

      // We won't need these anymore.
      state.resolveConfigurationsResult = null;
      state.resolveConfiguredTargetDependenciesResult = null;

      return mergeAspectsResult;
    } catch (InterruptedException e) {
      // In practice, this comes from resolveConfigurations: other InterruptedExceptions are
      // declared for Skyframe value retrievals, which don't throw in reality.
      if (!transitiveRootCauses.isEmpty()) {
        // Allow caller to throw, don't prioritize interrupt: we may be error bubbling.
        Thread.currentThread().interrupt();
        return null;
      }
      throw e;
    } finally {
      if (shouldReplayStoredEvents) {
        state.storedEvents.replayOn(listener);
      }
    }
  }

  static ToolchainContextKey createDefaultToolchainContextKey(
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

  @VisibleForTesting // private
  public static UnloadedToolchainContextsInputs getUnloadedToolchainContextsInputs(
      TargetAndConfiguration targetAndConfiguration,
      @Nullable Label parentExecutionPlatformLabel,
      RuleClassProvider ruleClassProvider,
      ExtendedEventHandler listener)
      throws InterruptedException {
    var target = targetAndConfiguration.getTarget();
    if (!(target instanceof Rule)) {
      return UnloadedToolchainContextsInputs.empty();
    }

    Rule rule = (Rule) target;
    var configuration = targetAndConfiguration.getConfiguration();
    boolean useAutoExecGroups =
        rule.isAttrDefined("$use_auto_exec_groups", Type.BOOLEAN)
            ? (boolean) rule.getAttr("$use_auto_exec_groups")
            : configuration.useAutoExecGroups();
    var platformConfig = configuration.getFragment(PlatformConfiguration.class);
    var defaultExecConstraintLabels = getExecutionPlatformConstraints(rule, platformConfig);
    var ruleClass = rule.getRuleClassObject();
    var processedExecGroups =
        ExecGroupCollection.process(
            ruleClass.getExecGroups(),
            defaultExecConstraintLabels,
            ruleClass.getToolchainTypes(),
            useAutoExecGroups);

    if (platformConfig == null || !rule.useToolchainResolution()) {
      return UnloadedToolchainContextsInputs.create(
          processedExecGroups, /* targetToolchainContextKey= */ null);
    }

    return UnloadedToolchainContextsInputs.create(
        processedExecGroups,
        createDefaultToolchainContextKey(
            computeToolchainConfigurationKey(
                configuration,
                ((ConfiguredRuleClassProvider) ruleClassProvider)
                    .getToolchainTaggedTrimmingTransition(),
                listener),
            defaultExecConstraintLabels,
            /* debugTarget= */ platformConfig.debugToolchainResolution(rule.getLabel()),
            /* useAutoExecGroups= */ useAutoExecGroups,
            ruleClass.getToolchainTypes(),
            parentExecutionPlatformLabel));
  }

  private static BuildConfigurationKey computeToolchainConfigurationKey(
      BuildConfigurationValue configuration,
      PatchTransition toolchainTaggedTrimmingTransition,
      ExtendedEventHandler listener)
      throws InterruptedException {
    // The toolchain context's options are the parent rule's options with manual trimming
    // auto-applied. This means toolchains don't inherit feature flags. This helps build
    // performance: if the toolchain context had the exact same configuration of its parent and that
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
                configuration.getOptions(),
                toolchainTaggedTrimmingTransition.requiresOptionFragments()),
            listener);
    return BuildConfigurationKey.withoutPlatformMapping(toolchainOptions);
  }

  /**
   * Resolves the targets referenced in depValueNames and returns their {@link
   * ConfiguredTargetAndData} instances.
   *
   * <p>Returns null if not all instances are available yet.
   */
  @Nullable
  private static Map<ConfiguredTargetKey, ConfiguredTargetAndData>
      resolveConfiguredTargetDependencies(
          LookupEnvironment env,
          TargetAndConfiguration ctgValue,
          Collection<Dependency> deps,
          @Nullable NestedSetBuilder<Package> transitivePackages,
          NestedSetBuilder<Cause> transitiveRootCauses)
          throws DependencyEvaluationException, InterruptedException {
    boolean missedValues = env.valuesMissing();
    ConfiguredValueCreationException rootError = null;
    DetailedExitCode detailedExitCode = null;
    // Naively we would like to just fetch all requested ConfiguredTargets, together with their
    // Packages. However, some ConfiguredTargets are AliasConfiguredTargets, which means that their
    // associated Targets (and therefore associated Packages) don't correspond to their own Labels.
    // We don't know the associated Package until we fetch the ConfiguredTarget. Therefore, we have
    // to do a potential second pass, in which we fetch all the Packages for AliasConfiguredTargets.
    ImmutableSet<SkyKey> packageKeys =
        ImmutableSet.copyOf(
            Iterables.transform(deps, input -> input.getLabel().getPackageIdentifier()));
    Iterable<SkyKey> depKeys =
        Iterables.concat(
            Iterables.transform(deps, dep -> dep.getConfiguredTargetKey().toKey()), packageKeys);
    SkyframeLookupResult depValuesOrExceptions = env.getValuesAndExceptions(depKeys);
    boolean depValuesMissingForDebugging = env.valuesMissing();
    Map<ConfiguredTargetKey, ConfiguredTargetAndData> result =
        Maps.newHashMapWithExpectedSize(deps.size());
    Set<SkyKey> aliasPackagesToFetch = new HashSet<>();
    List<Dependency> aliasDepsToRedo = new ArrayList<>();
    SkyframeLookupResult aliasPackageValues = null;
    Collection<Dependency> depsToProcess = deps;
    for (int i = 0; i < 2; i++) {
      for (Dependency dep : depsToProcess) {
        ConfiguredTargetKey key = dep.getConfiguredTargetKey();
        ConfiguredTargetValue depValue;
        try {
          depValue =
              (ConfiguredTargetValue)
                  depValuesOrExceptions.getOrThrow(
                      key.toKey(), ConfiguredValueCreationException.class);
        } catch (ConfiguredValueCreationException e) {
          transitiveRootCauses.addTransitive(e.getRootCauses());
          detailedExitCode =
              DetailedExitCodeComparator.chooseMoreImportantWithFirstIfTie(
                  e.getDetailedExitCode(), detailedExitCode);
          if (e.getDetailedExitCode().equals(detailedExitCode)) {
            rootError = e;
          }
          continue;
        }
        if (depValue == null) {
          if (!depValuesMissingForDebugging) {
            BugReport.logUnexpected(
                "Unexpected exception: dep %s had null value, even though there were no values"
                    + " missing in the initial fetch. That means it had an unexpected exception"
                    + " type (not ConfiguredValueCreationException)",
                dep);
            depValuesMissingForDebugging = true;
          }
          missedValues = true;
          continue;
        }

        ConfiguredTarget depCt = depValue.getConfiguredTarget();
        Label depLabel = depCt.getLabel();
        SkyKey packageKey = depLabel.getPackageIdentifier();
        PackageValue pkgValue;
        if (i == 0) {
          if (!packageKeys.contains(packageKey)) {
            aliasPackagesToFetch.add(packageKey);
            aliasDepsToRedo.add(dep);
            continue;
          } else {
            pkgValue = (PackageValue) depValuesOrExceptions.get(packageKey);
            if (pkgValue == null) {
              // In a race, the getValuesAndExceptions call above may have retrieved the package
              // before it was done but the configured target after it was done. Since
              // SkyFunctionEnvironment may cache absent values, re-requesting it on this evaluation
              // may be useless, just treat it as missing.
              missedValues = true;
              continue;
            }
          }
        } else {
          // We were doing AliasConfiguredTarget mop-up.
          pkgValue = (PackageValue) aliasPackageValues.get(packageKey);
          if (pkgValue == null) {
            // This is unexpected: on the second iteration, all packages should be present, since
            // the configured targets that depend on them are present. But since that is not a
            // guarantee Skyframe makes, we tolerate their absence.
            missedValues = true;
            continue;
          }
        }

        try {
          BuildConfigurationValue depConfiguration = dep.getConfiguration();
          BuildConfigurationKey depKey = depValue.getConfiguredTarget().getConfigurationKey();
          if (depKey != null && !depKey.equals(depConfiguration.getKey())) {
            depConfiguration = (BuildConfigurationValue) env.getValue(depKey);
          }
          result.put(
              key,
              new ConfiguredTargetAndData(
                  depValue.getConfiguredTarget(),
                  pkgValue.getPackage().getTarget(depLabel.getName()),
                  depConfiguration,
                  dep.getTransitionKeys()));
        } catch (NoSuchTargetException e) {
          throw new IllegalStateException("Target already verified for " + dep, e);
        }
        if (transitivePackages != null) {
          transitivePackages.addTransitive(
              Preconditions.checkNotNull(depValue.getTransitivePackages()));
        }
      }

      if (aliasDepsToRedo.isEmpty()) {
        break;
      }
      aliasPackageValues = env.getValuesAndExceptions(aliasPackagesToFetch);
      depsToProcess = aliasDepsToRedo;
    }

    if (rootError != null) {
      throw new DependencyEvaluationException(
          new ConfiguredValueCreationException(
              ctgValue, rootError.getMessage(), transitiveRootCauses.build(), detailedExitCode),
          /*depReportedOwnError=*/ true);
    }
    return missedValues ? null : result;
  }

  static DetailedExitCode getPrioritizedDetailedExitCode(NestedSet<Cause> causes) {
    DetailedExitCode prioritizedDetailedExitCode = null;
    for (Cause c : causes.toList()) {
      prioritizedDetailedExitCode =
          DetailedExitCodeComparator.chooseMoreImportantWithFirstIfTie(
              prioritizedDetailedExitCode, c.getDetailedExitCode());
    }
    return prioritizedDetailedExitCode;
  }
}
