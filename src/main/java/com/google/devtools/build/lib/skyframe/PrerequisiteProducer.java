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

import static com.google.devtools.build.lib.packages.ExecGroup.DEFAULT_EXEC_GROUP_NAME;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
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
import com.google.devtools.build.lib.analysis.ToolchainContext;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.ConfigConditions;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.ConfigurationResolver;
import com.google.devtools.build.lib.analysis.config.DependencyEvaluationException;
import com.google.devtools.build.lib.analysis.config.ToolchainTypeRequirement;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.constraints.IncompatibleTargetChecker.IncompatibleTargetException;
import com.google.devtools.build.lib.analysis.constraints.IncompatibleTargetChecker.IncompatibleTargetProducer;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.causes.LoadingFailedCause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper.ValidationException;
import com.google.devtools.build.lib.packages.ExecGroup;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetEvaluationExceptions.ReportedException;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetEvaluationExceptions.UnreportedException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.DetailedExitCode.DetailedExitCodeComparator;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import com.google.devtools.build.skyframe.state.Driver;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Location;

/**
 * Helper logic for {@link ConfiguredTargetFunction}: performs the analysis phase through
 * computation of prerequisites.
 *
 * <p>This includes:
 *
 * <ul>
 *   <li>getting this target's {@link Target} and {@link BuildConfigurationValue}
 *   <li>getting this target's {@code select()} keys (config conditions), which are used to evaluate
 *       all rule attributes with {@code select()} and determine exact dependencies
 *   <li>figuring out which toolchains this target needs
 *   <li>getting the {@link ConfiguredTargetValue}s of this target's prerequisites (through
 *       recursive calls to {@link ConfiguredTargetFunction}
 * </ul>
 *
 * <p>Cumulatively, this is enough information to run the target's rule logic.
 *
 * <p>This class also provides getters for the above data for subsequent analysis logic to use.
 *
 * <p>See {@link ConfiguredTargetFunction} for more review on analysis implementation.
 */
public final class PrerequisiteProducer {
  static class State implements SkyKeyComputeState, IncompatibleTargetProducer.ResultSink {
    @Nullable TargetAndConfiguration targetAndConfiguration;

    /**
     * Drives the stateful computation of {@link #incompatibleTarget}.
     *
     * <p>Non-null only while the computation is in-flight.
     */
    @Nullable private Driver incompatibleTargetProducer;

    /**
     * If a value is present, it means the target was directly incompatible.
     *
     * <p>Either this or {@link #validationException} will be non-null after the {@link
     * #incompatibleTargetProducer} completes.
     */
    @Nullable private Optional<RuleConfiguredTargetValue> incompatibleTarget;

    /**
     * If this is set, an exception occurred during validation of the {@code target_compatible_with}
     * attribute.
     *
     * <p>Either this or {@link #incompatibleTarget} will be non-null after the {@link
     * #incompatibleTargetProducer} completes.
     */
    @Nullable private ValidationException validationException;

    /** Null if not yet computed or if {@link #resolveConfigurationsResult} is non-null. */
    @Nullable private OrderedSetMultimap<DependencyKind, DependencyKey> dependentNodeMapResult;

    /** Null if not yet computed or if {@link #computeDependenciesResult} is non-null. */
    @Nullable private OrderedSetMultimap<DependencyKind, Dependency> resolveConfigurationsResult;

    /** Null if not yet computed or if {@link #computeDependenciesResult} is non-null. */
    @Nullable
    private Map<SkyKey, ConfiguredTargetAndData> resolveConfiguredTargetDependenciesResult;

    /**
     * Non-null if all the work in {@link #computeDependencies} is already done. This field contains
     * the result.
     */
    @Nullable
    private OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> computeDependenciesResult;

    /**
     * Non-null if either {@link #resolveConfigurationsResult} or {@link #computeDependenciesResult}
     * are non-null. This field contains events (from {@link
     * ConfigurationResolver#resolveConfigurations}) that should be replayed.
     *
     * <p>When {@link #resolveConfigurationsResult} or {@link #computeDependenciesResult} are
     * non-null (e.g. populated on a previous call to {@link #computeDependencies} on a previous
     * call to {@link #evaluate}), we don't freshly do the work that would cause these events to be
     * freshly emitted. So instead we replay these events from the actual call to {@link
     * ConfigurationResolver#resolveConfigurations} we did in the past. This is important because
     * Skyframe retains and uses only the events emitted to {@code env.getListener()} on a call to
     * {@link #evaluate} that had no missing deps. That is, if our earlier {@link #evaluate}'s call
     * to {@link ConfigurationResolver#resolveConfigurations} emitted events to {@code
     * env.getListener()}, and that {@link #evaluate} call returned null, then those events would be
     * thrown away.
     */
    @Nullable private StoredEventHandler storedEventHandlerFromResolveConfigurations;

    @Override
    public void accept(Optional<RuleConfiguredTargetValue> incompatibleTarget) {
      this.incompatibleTarget = incompatibleTarget;
    }

    @Override
    public void acceptValidationException(ValidationException e) {
      this.validationException = e;
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

  private TargetAndConfiguration targetAndConfiguration = null;
  private OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> depValueMap = null;
  private ConfigConditions configConditions = null;
  private PlatformInfo platformInfo = null;
  private ToolchainCollection<UnloadedToolchainContext> unloadedToolchainContexts = null;
  private ExecGroupCollection.Builder execGroupCollectionBuilder = null;

  /**
   * Return this target's {@link TargetAndConfiguration}.
   *
   * <p>{@link #evaluate} must be called before this info is available.
   */
  TargetAndConfiguration getTargetAndConfiguration() {
    return Preconditions.checkNotNull(targetAndConfiguration);
  }

  /**
   * Return this target's fully resolved dependencies.
   *
   * <p>{@link #evaluate} must be called before this info is available.
   */
  OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> getDepValueMap() {
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
   * Return this target's {@link ExecGroupCollection}, as far as this phase of analysis computes.
   *
   * <p>{@link #evaluate} must be called before this info is available.
   */
  ExecGroupCollection.Builder getExecGroupCollectionsBuilder() {
    return Preconditions.checkNotNull(execGroupCollectionBuilder);
  }

  /**
   * Run's the analysis phase for this target through prerequisite evaluation.
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
      ConfiguredTargetKey configuredTargetKey,
      State state,
      NestedSetBuilder<Cause> transitiveRootCauses,
      @Nullable NestedSetBuilder<Package> transitivePackages,
      RuleClassProvider ruleClassProvider,
      SkyframeBuildView view,
      SemaphoreAcquirer semaphoreLocker,
      Environment env)
      throws ReportedException,
          UnreportedException,
          InconsistentNullConfigException,
          IncompatibleTargetException,
          InterruptedException {
    this.targetAndConfiguration =
        computeTargetAndConfiguration(
            configuredTargetKey, state, transitiveRootCauses, transitivePackages, env);
    if (targetAndConfiguration == null) {
      return false;
    }
    Target target = targetAndConfiguration.getTarget();
    if ((target.isConfigurable() && configuredTargetKey.getConfigurationKey() == null)
        || (!target.isConfigurable() && configuredTargetKey.getConfigurationKey() != null)) {
      // We somehow ended up in a target that requires a non-null configuration as a dependency of
      // one that requires a null configuration or the other way round. This is always an error, but
      // we need to analyze the dependencies of the latter target to realize that. Short-circuit the
      // evaluation to avoid doing useless work and running code with a null configuration that's
      // not prepared for it.
      throw new InconsistentNullConfigException();
    }

    // TODO(janakr): this call may tie up this thread indefinitely, reducing the parallelism of
    //  Skyframe. This is a strict improvement over the prior state of the code, in which we ran
    //  with #processors threads, but ideally we would call #tryAcquire here, and if we failed,
    //  would exit this SkyFunction and restart it when permits were available.
    semaphoreLocker.acquireSemaphore();
    try {
      // Check target compatibility before requesting toolchains - known missing toolchains are a
      // valid reason for declaring a target incompatible and should not result in the target
      // failing to build rather than being skipped as incompatible.
      // Non-rule targets and those that are part of the toolchain resolution system do not support
      // target compatibility checking anyway.
      if (targetAndConfiguration.getTarget() instanceof Rule
          && ((Rule) targetAndConfiguration.getTarget()).useToolchainResolution()) {
        platformInfo = loadTargetPlatformInfo(env);
        // loadTargetPlatformInfo may return null even when no deps are missing.
        if (env.valuesMissing()) {
          return false;
        }

        configConditions =
            computeConfigConditions(
                env,
                targetAndConfiguration,
                transitivePackages,
                platformInfo,
                transitiveRootCauses);
        if (configConditions == null) {
          return false;
        }

        if (!checkForIncompatibleTarget(
            env,
            state,
            targetAndConfiguration,
            configConditions,
            platformInfo,
            transitivePackages)) {
          return false;
        }
      }
      var unloadedToolchainContextsInputs =
          getUnloadedToolchainContextsInputs(
              targetAndConfiguration,
              configuredTargetKey.getExecutionPlatformLabel(),
              ruleClassProvider,
              env.getListener());
      // Determine what toolchains are needed by this target.
      Optional<ToolchainCollection<UnloadedToolchainContext>> result =
          computeUnloadedToolchainContexts(env, unloadedToolchainContextsInputs);
      if (result == null) {
        return false;
      }
      this.unloadedToolchainContexts = result.orElse(null);
      this.execGroupCollectionBuilder = unloadedToolchainContextsInputs;
      this.platformInfo =
          unloadedToolchainContexts != null ? unloadedToolchainContexts.getTargetPlatform() : null;

      // Get the configuration targets that trigger this rule's configurable attributes.
      // Has been computed as part of checking for incompatible targets at the beginning of this
      // function unless the current target is not a rule participating in toolchain resolution.
      if (configConditions == null) {
        configConditions =
            computeConfigConditions(
                env,
                targetAndConfiguration,
                transitivePackages,
                platformInfo,
                transitiveRootCauses);
      }
      if (configConditions == null) {
        return false;
      }
      // TODO(ulfjack): ConfiguredAttributeMapper (indirectly used from computeDependencies) isn't
      // safe to use if there are missing config conditions, so we stop here, but only if there are
      // config conditions - though note that we can't check if configConditions is non-empty - it
      // may be empty for other reasons. It would be better to continue here so that we can collect
      // more root causes during computeDependencies.
      // Note that this doesn't apply to AspectFunction, because aspects can't have configurable
      // attributes.
      if (!transitiveRootCauses.isEmpty()
          && !Objects.equals(configConditions, ConfigConditions.EMPTY)) {
        NestedSet<Cause> causes = transitiveRootCauses.build();
        env.getListener()
            .handle(Event.error(target.getLocation(), "Cannot compute config conditions"));
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
              transitivePackages,
              transitiveRootCauses,
              env,
              ImmutableList.of(),
              configConditions.asProviders(),
              unloadedToolchainContexts == null
                  ? null
                  : unloadedToolchainContexts.asToolchainContexts(),
              ruleClassProvider,
              view);
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
      handleException(env, target, e);
    }
    return true;
  }

  // May return null even when no deps are missing, use env.valuesMissing() to check.
  @Nullable
  private PlatformInfo loadTargetPlatformInfo(Environment env)
      throws InterruptedException, ToolchainException {
    PlatformConfiguration platformConfiguration =
        targetAndConfiguration.getConfiguration().getFragment(PlatformConfiguration.class);
    if (platformConfiguration == null) {
      // No restart required in this case.
      return null;
    }
    Label targetPlatformLabel = platformConfiguration.getTargetPlatform();
    ConfiguredTargetKey targetPlatformKey =
        ConfiguredTargetKey.builder()
            .setLabel(targetPlatformLabel)
            .setConfiguration(targetAndConfiguration.getConfiguration())
            .build();

    Map<ConfiguredTargetKey, PlatformInfo> platformInfoMap =
        PlatformLookupUtil.getPlatformInfo(ImmutableList.of(targetPlatformKey), env);
    if (platformInfoMap == null) {
      return null;
    }

    return platformInfoMap.get(targetPlatformKey);
  }

  /**
   * Checks if a target is incompatible because of its "target_compatible_with" attribute.
   *
   * @return false if a {@code Skyframe} restart is needed.
   */
  private static boolean checkForIncompatibleTarget(
      Environment env,
      State state,
      TargetAndConfiguration targetAndConfiguration,
      @Nullable ConfigConditions configConditions,
      @Nullable PlatformInfo targetPlatformInfo,
      @Nullable NestedSetBuilder<Package> transitivePackages)
      throws InterruptedException, IncompatibleTargetException, DependencyEvaluationException {
    if (state.incompatibleTarget == null) {
      BuildConfigurationValue configuration = targetAndConfiguration.getConfiguration();
      if (state.incompatibleTargetProducer == null) {
        state.incompatibleTargetProducer =
            new Driver(
                new IncompatibleTargetProducer(
                    targetAndConfiguration.getTarget(),
                    configuration,
                    configConditions,
                    targetPlatformInfo,
                    transitivePackages,
                    state));
      }
      if (!state.incompatibleTargetProducer.drive(env, env.getListener())) {
        return false;
      }
      state.incompatibleTargetProducer = null;
      if (state.validationException != null) {
        Label label = targetAndConfiguration.getLabel();
        Location location = targetAndConfiguration.getTarget().getLocation();
        env.getListener()
            .post(
                new AnalysisRootCauseEvent(
                    configuration, label, state.validationException.getMessage()));
        throw new DependencyEvaluationException(
            new ConfiguredValueCreationException(
                location,
                state.validationException.getMessage(),
                label,
                configuration.getEventId(),
                null,
                null),
            // These errors occur within DependencyResolver, which is attached to the current
            // target. i.e. no dependent ConfiguredTargetFunction call happens to report its own
            // error.
            /* depReportedOwnError= */ false);
      }
      if (state.incompatibleTarget.isPresent()) {
        throw new IncompatibleTargetException(state.incompatibleTarget.get());
      }
    }
    return true;
  }

  /**
   * Handles all exceptions that {@link #evaluate} may throw.
   *
   * <p>This is its own method because there's a lot of logic here and when directly inlined it
   * makes it harder to follow the calling method's control flow.
   */
  private void handleException(Environment env, Target target, Exception untyped)
      throws ReportedException {

    if (untyped instanceof DependencyEvaluationException) {
      DependencyEvaluationException e = (DependencyEvaluationException) untyped;
      String errorMessage = e.getMessage();
      if (!e.depReportedOwnError()) {
        env.getListener().handle(Event.error(e.getLocation(), e.getMessage()));
      }

      ConfiguredValueCreationException cvce = null;
      if (e.getCause() instanceof ConfiguredValueCreationException) {
        cvce = (ConfiguredValueCreationException) e.getCause();

        // Check if this is caused by an unresolved toolchain, and report it as such.
        if (unloadedToolchainContexts != null) {
          ImmutableSet<Label> requiredToolchains =
              unloadedToolchainContexts.getResolvedToolchains();
          Set<Label> toolchainDependencyErrors =
              cvce.getRootCauses().toList().stream()
                  .map(Cause::getLabel)
                  .filter(requiredToolchains::contains)
                  .collect(ImmutableSet.toImmutableSet());

          if (!toolchainDependencyErrors.isEmpty()) {
            errorMessage = "errors encountered resolving toolchains for " + target.getLabel();
            env.getListener().handle(Event.error(target.getLocation(), errorMessage));
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
        env.getListener().handle(Event.error(e.getLocation(), e.getMessage()));
      }
      throw new ReportedException(e);
    } else if (untyped instanceof AspectCreationException) {
      AspectCreationException e = (AspectCreationException) untyped;
      if (!e.getMessage().isEmpty()) {
        // Report the error to the user.
        env.getListener().handle(Event.error(null, e.getMessage()));
      }
      throw new ReportedException(
          new ConfiguredValueCreationException(
              targetAndConfiguration, e.getMessage(), e.getCauses(), e.getDetailedExitCode()));
    } else if (untyped instanceof ToolchainException) {
      ToolchainException e = (ToolchainException) untyped;
      ConfiguredValueCreationException cvce =
          e.asConfiguredValueCreationException(targetAndConfiguration);
      env.getListener().handle(Event.error(target.getLocation(), cvce.getMessage()));
      throw new ReportedException(cvce);
    } else {
      throw new IllegalStateException("unexpected exception with no appropriate handler", untyped);
    }
  }

  @Nullable
  private static TargetAndConfiguration computeTargetAndConfiguration(
      ConfiguredTargetKey configuredTargetKey,
      State state,
      NestedSetBuilder<Cause> transitiveRootCauses,
      @Nullable NestedSetBuilder<Package> transitivePackages,
      Environment env)
      throws InterruptedException, ReportedException {
    if (state.targetAndConfiguration != null) {
      return state.targetAndConfiguration;
    }
    Label label = configuredTargetKey.getLabel();
    BuildConfigurationValue configuration = null;
    ImmutableSet<SkyKey> packageAndMaybeConfiguration;
    SkyKey packageKey = PackageValue.key(label.getPackageIdentifier());
    SkyKey configurationKeyMaybe = configuredTargetKey.getConfigurationKey();
    if (configurationKeyMaybe == null) {
      packageAndMaybeConfiguration = ImmutableSet.of(packageKey);
    } else {
      packageAndMaybeConfiguration = ImmutableSet.of(packageKey, configurationKeyMaybe);
    }
    SkyframeLookupResult packageAndMaybeConfigurationValues =
        env.getValuesAndExceptions(packageAndMaybeConfiguration);
    if (env.valuesMissing()) {
      return null;
    }
    PackageValue packageValue = (PackageValue) packageAndMaybeConfigurationValues.get(packageKey);
    if (packageValue == null) {
      return null;
    }
    Package pkg = packageValue.getPackage();
    if (configurationKeyMaybe != null) {
      configuration =
          (BuildConfigurationValue) packageAndMaybeConfigurationValues.get(configurationKeyMaybe);
    }
    // TODO(ulfjack): This tries to match the logic in TransitiveTargetFunction /
    // TargetMarkerFunction. Maybe we can merge the two?
    Target target;
    try {
      target = pkg.getTarget(label.getName());
    } catch (NoSuchTargetException e) {
      if (!e.getMessage().isEmpty()) {
        env.getListener().handle(Event.error(pkg.getBuildFile().getLocation(), e.getMessage()));
      }
      throw new ReportedException(
          new ConfiguredValueCreationException(
              pkg.getBuildFile().getLocation(),
              e.getMessage(),
              label,
              configuration.getEventId(),
              null,
              e.getDetailedExitCode()));
    }
    if (pkg.containsErrors()) {
      FailureDetail failureDetail = pkg.contextualizeFailureDetailForTarget(target);
      transitiveRootCauses.add(new LoadingFailedCause(label, DetailedExitCode.of(failureDetail)));
    }
    if (transitivePackages != null) {
      transitivePackages.add(pkg);
    }
    state.targetAndConfiguration = new TargetAndConfiguration(target, configuration);
    return state.targetAndConfiguration;
  }

  @AutoValue
  abstract static class UnloadedToolchainContextsInputs extends ExecGroupCollection.Builder {
    @Nullable // Null if no toolchain resolution is required.
    abstract ToolchainContextKey targetToolchainContextKey();

    static UnloadedToolchainContextsInputs create(
        ImmutableMap<String, ExecGroup> processedExecGroups,
        @Nullable ToolchainContextKey targetToolchainContextKey) {
      return new AutoValue_PrerequisiteProducer_UnloadedToolchainContextsInputs(
          processedExecGroups, targetToolchainContextKey);
    }

    static UnloadedToolchainContextsInputs empty() {
      return new AutoValue_PrerequisiteProducer_UnloadedToolchainContextsInputs(
          ImmutableMap.of(), /* targetToolchainContextKey= */ null);
    }
  }

  /**
   * Computes the unloaded toolchain contexts.
   *
   * @return null if a Skyframe restart is needed, {@link Optional#empty} if there is no toolchain
   *     resolution and the unloaded toolchain contexts otherwise.
   */
  @VisibleForTesting // package private
  @Nullable
  public static Optional<ToolchainCollection<UnloadedToolchainContext>>
      computeUnloadedToolchainContexts(Environment env, UnloadedToolchainContextsInputs inputs)
          throws InterruptedException, ToolchainException {
    var targetToolchainContextKey = inputs.targetToolchainContextKey();
    // Short circuit and end now if this target doesn't require toolchain resolution.
    if (targetToolchainContextKey == null) {
      return Optional.empty();
    }

    var toolchainContextKeys = new HashMap<String, ToolchainContextKey>();
    toolchainContextKeys.put(DEFAULT_EXEC_GROUP_NAME, targetToolchainContextKey);

    var keyBuilder =
        ToolchainContextKey.key()
            .configurationKey(targetToolchainContextKey.configurationKey())
            .debugTarget(targetToolchainContextKey.debugTarget());
    for (Map.Entry<String, ExecGroup> entry : inputs.execGroups().entrySet()) {
      var execGroup = entry.getValue();
      toolchainContextKeys.put(
          entry.getKey(),
          keyBuilder
              .toolchainTypes(execGroup.toolchainTypes())
              .execConstraintLabels(execGroup.execCompatibleWith())
              .build());
    }

    SkyframeLookupResult values = env.getValuesAndExceptions(toolchainContextKeys.values());
    boolean valuesMissing = env.valuesMissing();

    ToolchainCollection.Builder<UnloadedToolchainContext> toolchainContexts =
        valuesMissing
            ? null
            : ToolchainCollection.builderWithExpectedSize(toolchainContextKeys.size());
    for (Map.Entry<String, ToolchainContextKey> unloadedToolchainContextKey :
        toolchainContextKeys.entrySet()) {
      UnloadedToolchainContext unloadedToolchainContext =
          (UnloadedToolchainContext)
              values.getOrThrow(unloadedToolchainContextKey.getValue(), ToolchainException.class);
      if (valuesMissing != env.valuesMissing()) {
        BugReport.logUnexpected(
            "Value for: '%s' was missing, this should never happen",
            unloadedToolchainContextKey.getValue());
        break;
      }
      if (unloadedToolchainContext != null && unloadedToolchainContext.errorData() != null) {
        throw new NoMatchingPlatformException(unloadedToolchainContext.errorData());
      }
      if (!valuesMissing) {
        toolchainContexts.addContext(
            unloadedToolchainContextKey.getKey(), unloadedToolchainContext);
      }
    }

    if (valuesMissing) {
      return null;
    }

    return Optional.of(toolchainContexts.build());
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
   * @param state the compute state
   * @param env the Skyframe environment
   * @param configConditions the configuration conditions for evaluating the attributes of the node
   * @param toolchainContexts the toolchain context for this target
   * @param ruleClassProvider rule class provider for determining the right configuration fragments
   *     to apply to deps
   * @param buildView the build's {@link SkyframeBuildView}
   */
  // TODO(b/213351014): Make the control flow of this helper function more readable. This will
  //   involve making a corresponding change to State to match the control flow.
  @Nullable
  static OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> computeDependencies(
      State state,
      @Nullable NestedSetBuilder<Package> transitivePackages,
      NestedSetBuilder<Cause> transitiveRootCauses,
      Environment env,
      Iterable<Aspect> aspects,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      @Nullable ToolchainCollection<ToolchainContext> toolchainContexts,
      RuleClassProvider ruleClassProvider,
      SkyframeBuildView buildView)
      throws DependencyEvaluationException,
          ConfiguredValueCreationException,
          AspectCreationException,
          InterruptedException {
    if (state.computeDependenciesResult != null) {
      state.storedEventHandlerFromResolveConfigurations.replayOn(env.getListener());
      return state.computeDependenciesResult;
    }
    try {
      TargetAndConfiguration ctgValue = state.targetAndConfiguration;
      OrderedSetMultimap<DependencyKind, Dependency> depValueNames;
      if (state.resolveConfigurationsResult != null) {
        depValueNames = state.resolveConfigurationsResult;
      } else {
        // Create the map from attributes to set of (target, transition) pairs.
        OrderedSetMultimap<DependencyKind, DependencyKey> initialDependencies;
        if (state.dependentNodeMapResult != null) {
          initialDependencies = state.dependentNodeMapResult;
        } else {
          BuildConfigurationValue configuration = ctgValue.getConfiguration();
          Label label = ctgValue.getLabel();
          try {
            initialDependencies =
                new SkyframeDependencyResolver(env)
                    .dependentNodeMap(
                        ctgValue,
                        aspects,
                        configConditions,
                        toolchainContexts,
                        transitiveRootCauses,
                        ((ConfiguredRuleClassProvider) ruleClassProvider)
                            .getTrimmingTransitionFactory());
          } catch (DependencyResolver.Failure e) {
            env.getListener()
                .post(new AnalysisRootCauseEvent(configuration, label, e.getMessage()));
            throw new DependencyEvaluationException(
                new ConfiguredValueCreationException(
                    e.getLocation(), e.getMessage(), label, configuration.getEventId(), null, null),
                // These errors occur within DependencyResolver, which is attached to the current
                // target. i.e. no dependent ConfiguredTargetFunction call happens to report its own
                // error.
                /*depReportedOwnError=*/ false);
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
                env,
                ctgValue,
                configConditions,
                buildView.getStarlarkTransitionCache());
        StoredEventHandler storedEventHandler = new StoredEventHandler();
        try {
          depValueNames =
              configResolver.resolveConfigurations(initialDependencies, storedEventHandler);
        } catch (ConfiguredValueCreationException e) {
          storedEventHandler.replayOn(env.getListener());
          throw e;
        }
        if (!env.valuesMissing()) {
          state.resolveConfigurationsResult = depValueNames;
          state.storedEventHandlerFromResolveConfigurations = storedEventHandler;

          // We won't need this anymore.
          state.dependentNodeMapResult = null;
        }
      }

      // Return early in case packages were not loaded yet. In theory, we could start configuring
      // dependent targets in loaded packages. However, that creates an artificial sync boundary
      // between loading all dependent packages (fast) and configuring some dependent targets (can
      // have a long tail).
      if (env.valuesMissing()) {
        return null;
      }

      // Resolve configured target dependencies and handle errors.
      Map<SkyKey, ConfiguredTargetAndData> depValues;
      if (state.resolveConfiguredTargetDependenciesResult != null) {
        depValues = state.resolveConfiguredTargetDependenciesResult;
      } else {
        depValues =
            resolveConfiguredTargetDependencies(
                env, ctgValue, depValueNames.values(), transitivePackages, transitiveRootCauses);
        if (env.valuesMissing()) {
          return null;
        }
        state.resolveConfiguredTargetDependenciesResult = depValues;
      }

      // Resolve required aspects.
      OrderedSetMultimap<Dependency, ConfiguredAspect> depAspects =
          AspectResolver.resolveAspectDependencies(
              env, depValues, depValueNames.values(), transitivePackages);
      if (env.valuesMissing()) {
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
      state.storedEventHandlerFromResolveConfigurations.replayOn(env.getListener());

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
    }
  }

  /**
   * Returns the targets that key the configurable attributes used by this rule.
   *
   * <p>If the configured targets supplying those providers aren't yet resolved by the dependency
   * resolver, returns null.
   */
  @Nullable
  static ConfigConditions computeConfigConditions(
      Environment env,
      TargetAndConfiguration ctgValue,
      @Nullable NestedSetBuilder<Package> transitivePackages,
      @Nullable PlatformInfo platformInfo,
      NestedSetBuilder<Cause> transitiveRootCauses)
      throws ConfiguredValueCreationException, InterruptedException {
    Target target = ctgValue.getTarget();
    if (!(target instanceof Rule)) {
      return ConfigConditions.EMPTY;
    }
    RawAttributeMapper attrs = RawAttributeMapper.of(((Rule) target));
    if (!attrs.has(RuleClass.CONFIG_SETTING_DEPS_ATTRIBUTE)) {
      return ConfigConditions.EMPTY;
    }

    // Collect the labels of the configured targets we need to resolve.
    List<Label> configLabels =
        attrs.get(RuleClass.CONFIG_SETTING_DEPS_ATTRIBUTE, BuildType.LABEL_LIST);
    if (configLabels.isEmpty()) {
      return ConfigConditions.EMPTY;
    }

    // Collect the actual deps without a configuration transition (since by definition config
    // conditions evaluate over the current target's configuration). If the dependency is
    // (erroneously) something that needs the null configuration, its analysis will be
    // short-circuited. That error will be reported later.
    ImmutableList.Builder<Dependency> depsBuilder = ImmutableList.builder();
    for (Label configurabilityLabel : configLabels) {
      Dependency configurabilityDependency =
          Dependency.builder()
              .setLabel(configurabilityLabel)
              .setConfiguration(ctgValue.getConfiguration())
              .build();
      depsBuilder.add(configurabilityDependency);
    }

    ImmutableList<Dependency> configConditionDeps = depsBuilder.build();

    Map<SkyKey, ConfiguredTargetAndData> configValues;
    try {
      configValues =
          resolveConfiguredTargetDependencies(
              env, ctgValue, configConditionDeps, transitivePackages, transitiveRootCauses);
      if (configValues == null) {
        return null;
      }
    } catch (DependencyEvaluationException e) {
      // One of the config dependencies doesn't exist, and we need to report that. Unfortunately,
      // there's not enough information to know which configurable attribute has the problem.
      throw new ConfiguredValueCreationException(
          // The precise error is reported by the dependency that failed to load.
          // TODO(gregce): beautify this error: https://github.com/bazelbuild/bazel/issues/11984.
          ctgValue, "errors encountered resolving select() keys for " + target.getLabel());
    }

    ImmutableMap.Builder<Label, ConfiguredTargetAndData> asConfiguredTargets =
        ImmutableMap.builder();
    ImmutableMap.Builder<Label, ConfigMatchingProvider> asConfigConditions = ImmutableMap.builder();

    // Get the configured targets as ConfigMatchingProvider interfaces.
    for (Dependency entry : configConditionDeps) {
      SkyKey baseKey = entry.getConfiguredTargetKey();
      // The code above guarantees that selectKeyTarget is non-null here.
      ConfiguredTargetAndData selectKeyTarget = configValues.get(baseKey);
      asConfiguredTargets.put(entry.getLabel(), selectKeyTarget);
      try {
        asConfigConditions.put(
            entry.getLabel(), ConfigConditions.fromConfiguredTarget(selectKeyTarget, platformInfo));
      } catch (ConfigConditions.InvalidConditionException e) {
        String message =
            String.format(
                    "%s is not a valid select() condition for %s.\n",
                    selectKeyTarget.getTargetLabel(), target.getLabel())
                + String.format(
                    "To inspect the select(), run: bazel query --output=build %s.\n",
                    target.getLabel())
                + "For more help, see https://bazel.build/reference/be/functions#select.\n\n";
        throw new ConfiguredValueCreationException(ctgValue, message);
      }
    }

    return ConfigConditions.create(
        asConfiguredTargets.buildOrThrow(), asConfigConditions.buildOrThrow());
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
  private static Map<SkyKey, ConfiguredTargetAndData> resolveConfiguredTargetDependencies(
      Environment env,
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
            Iterables.transform(
                deps, input -> PackageValue.key(input.getLabel().getPackageIdentifier())));
    Iterable<SkyKey> depKeys =
        Iterables.concat(
            Iterables.transform(deps, Dependency::getConfiguredTargetKey), packageKeys);
    SkyframeLookupResult depValuesOrExceptions = env.getValuesAndExceptions(depKeys);
    boolean depValuesMissingForDebugging = env.valuesMissing();
    Map<SkyKey, ConfiguredTargetAndData> result = Maps.newHashMapWithExpectedSize(deps.size());
    Set<SkyKey> aliasPackagesToFetch = new HashSet<>();
    List<Dependency> aliasDepsToRedo = new ArrayList<>();
    SkyframeLookupResult aliasPackageValues = null;
    Collection<Dependency> depsToProcess = deps;
    for (int i = 0; i < 2; i++) {
      for (Dependency dep : depsToProcess) {
        SkyKey key = dep.getConfiguredTargetKey();
        ConfiguredTargetValue depValue;
        try {
          depValue =
              (ConfiguredTargetValue)
                  depValuesOrExceptions.getOrThrow(key, ConfiguredValueCreationException.class);
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
        SkyKey packageKey = PackageValue.key(depLabel.getPackageIdentifier());
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
