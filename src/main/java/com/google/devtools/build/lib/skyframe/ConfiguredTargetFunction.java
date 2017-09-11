// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Supplier;
import com.google.common.base.Verify;
import com.google.common.base.VerifyException;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.LinkedListMultimap;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Actions.GeneratingActions;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.AspectCollection;
import com.google.devtools.build.lib.analysis.AspectCollection.AspectDeps;
import com.google.devtools.build.lib.analysis.CachingAnalysisEnvironment;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.Dependency;
import com.google.devtools.build.lib.analysis.DependencyResolver.InconsistentAspectOrderException;
import com.google.devtools.build.lib.analysis.LabelAndConfiguration;
import com.google.devtools.build.lib.analysis.MergedConfiguredTarget;
import com.google.devtools.build.lib.analysis.MergedConfiguredTarget.DuplicateException;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.ToolchainContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.PatchTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.skyframe.AspectFunction.AspectCreationException;
import com.google.devtools.build.lib.skyframe.AspectValue.AspectKey;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor.BuildViewProvider;
import com.google.devtools.build.lib.skyframe.ToolchainUtil.ToolchainContextException;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;
import com.google.devtools.build.skyframe.ValueOrException2;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Semaphore;
import javax.annotation.Nullable;

/**
 * SkyFunction for {@link ConfiguredTargetValue}s.
 *
 * <p>This class, together with {@link AspectFunction} drives the analysis phase. For more
 * information, see {@link com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory}.
 *
 * @see com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory
 */
public final class ConfiguredTargetFunction implements SkyFunction {
  // This construction is a bit funky, but guarantees that the Object reference here is globally
  // unique.
  static final ImmutableMap<Label, ConfigMatchingProvider> NO_CONFIG_CONDITIONS =
      ImmutableMap.<Label, ConfigMatchingProvider>of();

  /**
   * Exception class that signals an error during the evaluation of a dependency.
   */
  public static class DependencyEvaluationException extends Exception {
    public DependencyEvaluationException(InvalidConfigurationException cause) {
      super(cause);
    }

    public DependencyEvaluationException(ConfiguredValueCreationException cause) {
      super(cause);
    }

    public DependencyEvaluationException(InconsistentAspectOrderException cause) {
      super(cause);
    }

    @Override
    public synchronized Exception getCause() {
      return (Exception) super.getCause();
    }
  }

  private final BuildViewProvider buildViewProvider;
  private final RuleClassProvider ruleClassProvider;
  private final Semaphore cpuBoundSemaphore;
  private final Supplier<Boolean> removeActionsAfterEvaluation;

  ConfiguredTargetFunction(
      BuildViewProvider buildViewProvider,
      RuleClassProvider ruleClassProvider,
      Semaphore cpuBoundSemaphore,
      Supplier<Boolean> removeActionsAfterEvaluation) {
    this.buildViewProvider = buildViewProvider;
    this.ruleClassProvider = ruleClassProvider;
    this.cpuBoundSemaphore = cpuBoundSemaphore;
    this.removeActionsAfterEvaluation = Preconditions.checkNotNull(removeActionsAfterEvaluation);
  }

  @Override
  public SkyValue compute(SkyKey key, Environment env) throws ConfiguredTargetFunctionException,
      InterruptedException {
    SkyframeBuildView view = buildViewProvider.getSkyframeBuildView();
    NestedSetBuilder<Package> transitivePackages = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Label> transitiveLoadingRootCauses = NestedSetBuilder.stableOrder();
    ConfiguredTargetKey configuredTargetKey = (ConfiguredTargetKey) key.argument();
    LabelAndConfiguration lc = LabelAndConfiguration.of(
        configuredTargetKey.getLabel(), configuredTargetKey.getConfiguration());

    BuildConfiguration configuration = lc.getConfiguration();

    PackageValue packageValue =
        (PackageValue) env.getValue(PackageValue.key(lc.getLabel().getPackageIdentifier()));
    if (packageValue == null) {
      return null;
    }

    // TODO(ulfjack): This tries to match the logic in TransitiveTargetFunction /
    // TargetMarkerFunction. Maybe we can merge the two?
    Package pkg = packageValue.getPackage();
    Target target;
    try {
      target = pkg.getTarget(lc.getLabel().getName());
    } catch (NoSuchTargetException e) {
      throw new ConfiguredTargetFunctionException(
          new ConfiguredValueCreationException(e.getMessage(), lc.getLabel()));
    }
    if (pkg.containsErrors()) {
      transitiveLoadingRootCauses.add(lc.getLabel());
    }
    transitivePackages.add(pkg);
    // TODO(bazel-team): This is problematic - we create the right key, but then end up with a value
    // that doesn't match; we can even have the same value multiple times. However, I think it's
    // only triggered in tests (i.e., in normal operation, the configuration passed in is already
    // null).
    if (!target.isConfigurable()) {
      configuration = null;
    }

    // This line is only needed for accurate error messaging. Say this target has a circular
    // dependency with one of its deps. With this line, loading this target fails so Bazel
    // associates the corresponding error with this target, as expected. Without this line,
    // the first TransitiveTargetValue call happens on its dep (in trimConfigurations), so Bazel
    // associates the error with the dep, which is misleading.
    if (configuration != null && configuration.trimConfigurations()
        && env.getValue(TransitiveTargetValue.key(lc.getLabel())) == null) {
      return null;
    }

    TargetAndConfiguration ctgValue = new TargetAndConfiguration(target, configuration);

    SkyframeDependencyResolver resolver = view.createDependencyResolver(env);

    ToolchainContext toolchainContext = null;

    // TODO(janakr): this acquire() call may tie up this thread indefinitely, reducing the
    // parallelism of Skyframe. This is a strict improvement over the prior state of the code, in
    // which we ran with #processors threads, but ideally we would call #tryAcquire here, and if we
    // failed, would exit this SkyFunction and restart it when permits were available.
    cpuBoundSemaphore.acquire();
    try {
      // Get the configuration targets that trigger this rule's configurable attributes.
      ImmutableMap<Label, ConfigMatchingProvider> configConditions = getConfigConditions(
          ctgValue.getTarget(), env, resolver, ctgValue, transitivePackages,
          transitiveLoadingRootCauses);
      if (env.valuesMissing()) {
        return null;
      }
      // TODO(ulfjack): ConfiguredAttributeMapper (indirectly used from computeDependencies) isn't
      // safe to use if there are missing config conditions, so we stop here, but only if there are
      // config conditions - though note that we can't check if configConditions is non-empty - it
      // may be empty for other reasons. It would be better to continue here so that we can collect
      // more root causes during computeDependencies.
      // Note that this doesn't apply to AspectFunction, because aspects can't have configurable
      // attributes.
      if (!transitiveLoadingRootCauses.isEmpty() && configConditions != NO_CONFIG_CONDITIONS) {
        throw new ConfiguredTargetFunctionException(
            new ConfiguredValueCreationException(transitiveLoadingRootCauses.build()));
      }

      // Determine what toolchains are needed by this target.
      if (target instanceof Rule) {
        Rule rule = ((Rule) target);
        ImmutableSet<Label> requiredToolchains = rule.getRuleClassObject().getRequiredToolchains();
        toolchainContext =
            ToolchainUtil.createToolchainContext(
                env, rule.toString(), requiredToolchains, configuration);
        if (env.valuesMissing()) {
          return null;
        }
      }

      // Calculate the dependencies of this target.
      OrderedSetMultimap<Attribute, ConfiguredTarget> depValueMap =
          computeDependencies(
              env,
              resolver,
              ctgValue,
              ImmutableList.<Aspect>of(),
              configConditions,
              toolchainContext,
              ruleClassProvider,
              view.getHostConfiguration(configuration),
              transitivePackages,
              transitiveLoadingRootCauses);
      if (env.valuesMissing()) {
        return null;
      }
      if (!transitiveLoadingRootCauses.isEmpty()) {
        throw new ConfiguredTargetFunctionException(
            new ConfiguredValueCreationException(transitiveLoadingRootCauses.build()));
      }
      Preconditions.checkNotNull(depValueMap);
      ConfiguredTargetValue ans =
          createConfiguredTarget(
              view,
              env,
              target,
              configuration,
              depValueMap,
              configConditions,
              toolchainContext,
              transitivePackages);
      return ans;
    } catch (DependencyEvaluationException e) {
      if (e.getCause() instanceof ConfiguredValueCreationException) {
        ConfiguredValueCreationException cvce = (ConfiguredValueCreationException) e.getCause();

        // Check if this is caused by an unresolved toolchain, and report it as such.
        if (toolchainContext != null) {
          ImmutableSet.Builder<Label> causes = new ImmutableSet.Builder<Label>();
          if (cvce.getAnalysisRootCause() != null) {
            causes.add(cvce.getAnalysisRootCause());
          }
          if (!cvce.getRootCauses().isEmpty()) {
            causes.addAll(cvce.getRootCauses());
          }
          Set<Label> toolchainDependencyErrors =
              toolchainContext.filterToolchainLabels(causes.build());
          if (!toolchainDependencyErrors.isEmpty()) {
            env.getListener()
                .handle(
                    Event.error(
                        String.format(
                            "While resolving toolchains for target %s: %s",
                            target.getLabel(), e.getCause().getMessage())));
          }
        }

        throw new ConfiguredTargetFunctionException(cvce);
      } else if (e.getCause() instanceof InconsistentAspectOrderException) {
        InconsistentAspectOrderException cause = (InconsistentAspectOrderException) e.getCause();
        throw new ConfiguredTargetFunctionException(
            new ConfiguredValueCreationException(cause.getMessage(), target.getLabel()));
      } else if (e.getCause() instanceof InvalidConfigurationException) {
        InvalidConfigurationException cause = (InvalidConfigurationException) e.getCause();
        env.getListener().handle(Event.error(cause.getMessage()));
        throw new ConfiguredTargetFunctionException(
            new ConfiguredValueCreationException(cause.getMessage(), target.getLabel()));
      } else {
        // Unknown exception type.
        throw new ConfiguredTargetFunctionException(
            new ConfiguredValueCreationException(e.getMessage(), target.getLabel()));
      }
    } catch (AspectCreationException e) {
      // getAnalysisRootCause may be null if the analysis of the aspect itself failed.
      Label analysisRootCause = target.getLabel();
      if (e.getAnalysisRootCause() != null) {
        analysisRootCause = e.getAnalysisRootCause();
      }
      throw new ConfiguredTargetFunctionException(
          new ConfiguredValueCreationException(e.getMessage(), analysisRootCause));
    } catch (ToolchainContextException e) {
      // We need to throw a ConfiguredValueCreationException, so either find one or make one.
      ConfiguredValueCreationException cvce;
      if (e.getCause() instanceof ConfiguredValueCreationException) {
        cvce = (ConfiguredValueCreationException) e.getCause();
      } else {
        cvce = new ConfiguredValueCreationException(e.getCause().getMessage(), target.getLabel());
      }

      env.getListener()
          .handle(
              Event.error(
                  String.format(
                      "While resolving toolchains for target %s: %s",
                      target.getLabel(), e.getCause().getMessage())));
      throw new ConfiguredTargetFunctionException(cvce);
    } finally {
      cpuBoundSemaphore.release();
    }
  }

  /**
   * Computes the direct dependencies of a node in the configured target graph (a configured target
   * or an aspects).
   *
   * <p>Returns null if Skyframe hasn't evaluated the required dependencies yet. In this case, the
   * caller should also return null to Skyframe.
   *
   * @param env the Skyframe environment
   * @param resolver the dependency resolver
   * @param ctgValue the label and the configuration of the node
   * @param aspects
   * @param configConditions the configuration conditions for evaluating the attributes of the node
   * @param toolchainContext context information for required toolchains
   * @param ruleClassProvider rule class provider for determining the right configuration fragments
   *     to apply to deps
   * @param hostConfiguration the host configuration. There's a noticeable performance hit from
   *     instantiating this on demand for every dependency that wants it, so it's best to compute
   *     the host configuration as early as possible and pass this reference to all consumers
   */
  @Nullable
  static OrderedSetMultimap<Attribute, ConfiguredTarget> computeDependencies(
      Environment env,
      SkyframeDependencyResolver resolver,
      TargetAndConfiguration ctgValue,
      Iterable<Aspect> aspects,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      @Nullable ToolchainContext toolchainContext,
      RuleClassProvider ruleClassProvider,
      BuildConfiguration hostConfiguration,
      NestedSetBuilder<Package> transitivePackages,
      NestedSetBuilder<Label> transitiveLoadingRootCauses)
      throws DependencyEvaluationException, ConfiguredTargetFunctionException,
          AspectCreationException, InterruptedException {
    // Create the map from attributes to set of (target, configuration) pairs.
    OrderedSetMultimap<Attribute, Dependency> depValueNames;
    try {
      depValueNames =
          resolver.dependentNodeMap(
              ctgValue,
              hostConfiguration,
              aspects,
              configConditions,
              toolchainContext,
              transitiveLoadingRootCauses);
    } catch (EvalException e) {
      // EvalException can only be thrown by computed Skylark attributes in the current rule.
      env.getListener().handle(Event.error(e.getLocation(), e.getMessage()));
      throw new DependencyEvaluationException(
          new ConfiguredValueCreationException(e.print(), ctgValue.getLabel()));
    } catch (InvalidConfigurationException e) {
      throw new DependencyEvaluationException(e);
    } catch (InconsistentAspectOrderException e) {
      env.getListener().handle(Event.error(e.getLocation(), e.getMessage()));
      throw new DependencyEvaluationException(e);
    }

    // Trim each dep's configuration so it only includes the fragments needed by its transitive
    // closure.
    if (ctgValue.getConfiguration() != null) {
      depValueNames = getDynamicConfigurations(env, ctgValue, depValueNames, hostConfiguration,
          ruleClassProvider);
      // It's important that we don't use "if (env.missingValues()) { return null }" here (or
      // in the following lines). See the comments in getDynamicConfigurations' Skyframe call
      // for explanation.
      if (depValueNames == null) {
        return null;
      }
    }

    // Resolve configured target dependencies and handle errors.
    Map<SkyKey, ConfiguredTarget> depValues = resolveConfiguredTargetDependencies(env,
        depValueNames.values(), transitivePackages, transitiveLoadingRootCauses);
    if (depValues == null) {
      return null;
    }

    // Resolve required aspects.
    OrderedSetMultimap<Dependency, ConfiguredAspect> depAspects = resolveAspectDependencies(
        env, depValues, depValueNames.values(), transitivePackages);
    if (depAspects == null) {
      return null;
    }

    // Merge the dependent configured targets and aspects into a single map.
    try {
      return mergeAspects(depValueNames, depValues, depAspects);
    } catch (DuplicateException e) {
      env.getListener().handle(
          Event.error(ctgValue.getTarget().getLocation(), e.getMessage()));

      throw new ConfiguredTargetFunctionException(
          new ConfiguredValueCreationException(e.getMessage(), ctgValue.getLabel()));
    }
  }

  /**
   * Helper class for {@link #getDynamicConfigurations} - encapsulates a set of config fragments and
   * a dynamic transition. This can be used to determine the exact build options needed to
   * set a dynamic configuration.
   */
  @Immutable
  private static final class FragmentsAndTransition {
    // Treat this as immutable. The only reason this isn't an ImmutableSet is because it
    // gets bound to a NestedSet.toSet() reference, which returns a Set interface.
    final Set<Class<? extends BuildConfiguration.Fragment>> fragments;
    final Attribute.Transition transition;
    private final int hashCode;

    FragmentsAndTransition(Set<Class<? extends BuildConfiguration.Fragment>> fragments,
        Attribute.Transition transition) {
      this.fragments = fragments;
      this.transition = transition;
      hashCode = Objects.hash(this.fragments, this.transition);
    }

    @Override
    public boolean equals(Object o) {
      if (o == this) {
        return true;
      } else if (o == null) {
        return false;
      } else {
        FragmentsAndTransition other = (FragmentsAndTransition) o;
        return other.transition.equals(transition) && other.fragments.equals(fragments);
      }
    }

    @Override
    public int hashCode() {
      return hashCode;
    }
  }

  /**
   * Helper class for {@link #getDynamicConfigurations} - encapsulates an <attribute, label> pair
   * that can be used to map from an input dependency to a trimmed dependency.
   */
  @Immutable
  private static final class AttributeAndLabel {
    final Attribute attribute;
    final Label label;
    Integer hashCode;

    AttributeAndLabel(Attribute attribute, Label label) {
      this.attribute = attribute;
      this.label = label;
    }

    @Override
    public boolean equals(Object o) {
      if (!(o instanceof AttributeAndLabel)) {
        return false;
      }
      AttributeAndLabel other = (AttributeAndLabel) o;
      return Objects.equals(other.attribute, attribute) && other.label.equals(label);
    }

    @Override
    public int hashCode() {
      if (hashCode == null) {
        // Not every <Attribute, Label> pair gets hashed. So only evaluate for the instances that
        // need it. This can significantly reduce the number of evaluations.
        hashCode = Objects.hash(this.attribute, this.label);
      }
      return hashCode;
    }
  }

  /**
   * Variation of {@link Multimap#put} that triggers an exception if a value already exists.
   */
  @VisibleForTesting
  static <K, V> void putOnlyEntry(Multimap<K, V> map, K key, V value) {
    // Performance note: while "Verify.verify(!map.containsKey(key, value), String.format(...)))"
    // is simpler code, profiling shows a substantial performance penalty to that approach
    // (~10% extra analysis phase time on a simple cc_binary). Most of that is from the cost of
    // evaluating value.toString() on every call. This approach essentially eliminates the overhead.
    if (map.containsKey(key)) {
      throw new VerifyException(
          String.format("couldn't insert %s: map already has key %s",
              value.toString(), key.toString()));
    }
    map.put(key, value);
  }

  /**
   * Creates a dynamic configuration for each dep that's custom-fitted specifically for that dep.
   *
   * <p>More specifically: given a set of {@link Dependency} instances holding dynamic config
   * transition requests (e.g. {@link Dependency#hasStaticConfiguration()} == false}), returns
   * equivalent dependencies containing dynamically created configurations applying those
   * transitions. If {@link BuildConfiguration.Options#trimConfigurations()} is true, these
   * configurations only contain the fragments needed by the dep and its transitive closure. Else
   * the configurations unconditionally include all fragments.
   *
   * <p>This method is heavily performance-optimized. Because it, in aggregate, reads over every
   * edge in the configured target graph, small inefficiencies can have observable impact on
   * analysis time. Keep this in mind when making modifications and performance-test any changes you
   * make.
   *
   * @param env Skyframe evaluation environment
   * @param ctgValue the label and the configuration of the node
   * @param originalDeps the set of configuration transition requests for this target's attributes
   * @param hostConfiguration the host configuration
   * @param ruleClassProvider the rule class provider for determining the right configuration
   *    fragments to apply to deps
   *
   * @return a mapping from each attribute to the {@link BuildConfiguration}s and {@link Label}s
   *    to use for that attribute's deps. Returns null if not all Skyframe dependencies are
   *    available yet.
   */
  @Nullable
  static OrderedSetMultimap<Attribute, Dependency> getDynamicConfigurations(
      Environment env,
      TargetAndConfiguration ctgValue,
      OrderedSetMultimap<Attribute, Dependency> originalDeps,
      BuildConfiguration hostConfiguration,
      RuleClassProvider ruleClassProvider)
      throws DependencyEvaluationException, InterruptedException {

    // Maps each Skyframe-evaluated BuildConfiguration to the dependencies that need that
    // configuration. For cases where Skyframe isn't needed to get the configuration (e.g. when
    // we just re-used the original rule's configuration), we should skip this outright.
    Multimap<SkyKey, Map.Entry<Attribute, Dependency>> keysToEntries = LinkedListMultimap.create();

    // Stores the result of applying a dynamic transition to the current configuration using a
    // particular subset of fragments. By caching this, we save from redundantly computing the
    // same transition for every dependency edge that requests that transition. This can have
    // real effect on analysis time for commonly triggered transitions.
    //
    // Split transitions may map to multiple values. All other transitions map to one.
    Map<FragmentsAndTransition, List<BuildOptions>> transitionsMap = new LinkedHashMap<>();

    // The fragments used by the current target's configuration.
    Set<Class<? extends BuildConfiguration.Fragment>> ctgFragments =
        ctgValue.getConfiguration().fragmentClasses();
    BuildOptions ctgOptions = ctgValue.getConfiguration().getOptions();

    // Stores the dynamically configured versions of each dependency. This method must preserve the
    // original label ordering of each attribute. For example, if originalDeps.get("data") is
    // [":a", ":b"], the dynamic variant must also be [":a", ":b"] in the same order. Because we may
    // not actualize the results in order (some results need Skyframe-evaluated configurations while
    // others can be computed trivially), we dump them all into this map, then as a final step
    // iterate through the original list and pluck out values from here for the final value.
    //
    // For split transitions, originaldeps.get("data") = [":a", ":b"] can produce the output
    // [":a"<config1>, ":a"<config2>, ..., ":b"<config1>, ":b"<config2>, ...]. All instances of ":a"
    // still appear before all instances of ":b". But the [":a"<config1>, ":a"<config2>"] subset may
    // be in any (deterministic) order. In particular, this may not be the same order as
    // SplitTransition.split. If needed, this code can be modified to use that order, but that
    // involves more runtime in performance-critical code, so we won't make that change without a
    // clear need.
    //
    // This map is used heavily by all builds. Inserts and gets should be as fast as possible.
    Multimap<AttributeAndLabel, Dependency> dynamicDeps = LinkedHashMultimap.create();

    // Performance optimization: This method iterates over originalDeps twice. By storing
    // AttributeAndLabel instances in this list, we avoid having to recreate them the second time
    // (particularly avoid recomputing their hash codes). Profiling shows this shaves 25% off this
    // method's execution time (at the time of this comment).
    ArrayList<AttributeAndLabel> attributesAndLabels = new ArrayList<>(originalDeps.size());

    for (Map.Entry<Attribute, Dependency> depsEntry : originalDeps.entries()) {
      Dependency dep = depsEntry.getValue();
      AttributeAndLabel attributeAndLabel =
          new AttributeAndLabel(depsEntry.getKey(), dep.getLabel());
      attributesAndLabels.add(attributeAndLabel);
      // Certain targets (like output files) trivially re-use their input configuration. Likewise,
      // deps with null configurations (e.g. source files), can be trivially computed. So we skip
      // all logic in this method for these cases and just reinsert their original configurations
      // back at the end (note that null-configured targets will have a static
      // NullConfigurationDependency instead of dynamic
      // Dependency(label, transition=Attribute.Configuration.Transition.NULL)).
      //
      // A *lot* of targets have null deps, so this produces real savings. Profiling tests over a
      // simple cc_binary show this saves ~1% of total analysis phase time.
      if (dep.hasStaticConfiguration()) {
        continue;
      }

      // Figure out the required fragments for this dep and its transitive closure.
      Set<Class<? extends BuildConfiguration.Fragment>> depFragments =
          getTransitiveFragments(env, dep.getLabel(), ctgValue.getConfiguration());
      if (depFragments == null) {
        return null;
      }
      // TODO(gregce): remove the below call once we have confidence dynamic configurations always
      // provide needed fragments. This unnecessarily drags performance on the critical path (up
      // to 0.5% of total analysis time as profiled over a simple cc_binary).
      if (ctgValue.getConfiguration().trimConfigurations()) {
        checkForMissingFragments(env, ctgValue, attributeAndLabel.attribute.getName(), dep,
            depFragments);
      }

      boolean sameFragments = depFragments.equals(ctgFragments);
      Attribute.Transition transition = dep.getTransition();

      if (sameFragments) {
        if (transition == Attribute.ConfigurationTransition.NONE) {
          // The dep uses the same exact configuration.
          putOnlyEntry(
              dynamicDeps,
              attributeAndLabel,
              Dependency.withConfigurationAndAspects(
                  dep.getLabel(), ctgValue.getConfiguration(), dep.getAspects()));
          continue;
        } else if (transition == HostTransition.INSTANCE) {
          // The current rule's host configuration can also be used for the dep. We short-circuit
          // the standard transition logic for host transitions because these transitions are
          // uniquely frequent. It's possible, e.g., for every node in the configured target graph
          // to incur multiple host transitions. So we aggressively optimize to avoid hurting
          // analysis time.
          putOnlyEntry(
              dynamicDeps,
              attributeAndLabel,
              Dependency.withConfigurationAndAspects(
                  dep.getLabel(), hostConfiguration, dep.getAspects()));
          continue;
        }
      }

      // Apply the transition or use the cached result if it was already applied.
      FragmentsAndTransition transitionKey = new FragmentsAndTransition(depFragments, transition);
      List<BuildOptions> toOptions = transitionsMap.get(transitionKey);
      if (toOptions == null) {
        toOptions = getDynamicTransitionOptions(ctgOptions, transition, depFragments,
            ruleClassProvider, !sameFragments);
        transitionsMap.put(transitionKey, toOptions);
      }

      // If the transition doesn't change the configuration, trivially re-use the original
      // configuration.
      if (sameFragments && toOptions.size() == 1
          && Iterables.getOnlyElement(toOptions).equals(ctgOptions)) {
        putOnlyEntry(
            dynamicDeps,
            attributeAndLabel,
            Dependency.withConfigurationAndAspects(
                dep.getLabel(), ctgValue.getConfiguration(), dep.getAspects()));
        continue;
      }

      // If we get here, we have to get the configuration from Skyframe.
      for (BuildOptions options : toOptions) {
        keysToEntries.put(BuildConfigurationValue.key(depFragments, options), depsEntry);
      }
    }

    // Get all BuildConfigurations we need from Skyframe. While not every value might be available,
    // we don't call env.valuesMissing() here because that could be true from the earlier
    // resolver.dependentNodeMap call in computeDependencies, which also calls Skyframe. This method
    // doesn't need those missing values, but it still has to be called after
    // resolver.dependentNodeMap because it consumes that method's output. The reason the missing
    // values don't matter is because resolver.dependentNodeMap still returns "partial" results
    // and this method runs over whatever's available.
    //
    // While there would be no *correctness* harm in nulling out early, there's significant
    // *performance* harm. Profiling shows that putting "if (env.valuesMissing()) { return null; }"
    // here (or even after resolver.dependentNodeMap) produces a ~30% performance hit on the
    // analysis phase. That's because resolveConfiguredTargetDependencies and
    // resolveAspectDependencies don't get a chance to make their own Skyframe requests before
    // bailing out of this ConfiguredTargetFunction call. Ideally we could batch all requests
    // from all methods into a single Skyframe call, but there are enough subtle data flow
    // dependencies in ConfiguredTargetFucntion to make that impractical.
    Map<SkyKey, ValueOrException<InvalidConfigurationException>> depConfigValues =
        env.getValuesOrThrow(keysToEntries.keySet(), InvalidConfigurationException.class);

    // Now fill in the remaining unresolved deps with the now-resolved configurations.
    try {
      for (Map.Entry<SkyKey, ValueOrException<InvalidConfigurationException>> entry :
          depConfigValues.entrySet()) {
        SkyKey key = entry.getKey();
        ValueOrException<InvalidConfigurationException> valueOrException = entry.getValue();
        if (valueOrException.get() == null) {
          // Instead of env.missingValues(), check for missing values here. This guarantees we only
          // null out on missing values from *this specific Skyframe request*.
          return null;
        }
        BuildConfigurationValue trimmedConfig = (BuildConfigurationValue) valueOrException.get();
        for (Map.Entry<Attribute, Dependency> info : keysToEntries.get(key)) {
          Dependency originalDep = info.getValue();
          AttributeAndLabel attr = new AttributeAndLabel(info.getKey(), originalDep.getLabel());
          Dependency resolvedDep = Dependency.withConfigurationAndAspects(originalDep.getLabel(),
              trimmedConfig.getConfiguration(), originalDep.getAspects());
          if (attr.attribute.hasSplitConfigurationTransition()) {
            dynamicDeps.put(attr, resolvedDep);
          } else {
            putOnlyEntry(dynamicDeps, attr, resolvedDep);
          }
        }
      }
    } catch (InvalidConfigurationException e) {
      throw new DependencyEvaluationException(e);
    }

    return sortDynamicallyConfiguredDeps(originalDeps, dynamicDeps, attributesAndLabels);
  }

  /**
   * Returns the configuration fragments required by a dep and its transitive closure.
   * Returns null if Skyframe dependencies aren't yet available.
   *
   * @param env Skyframe evaluation environment
   * @param dep label of the dep to check
   * @param parentConfig configuration of the rule depending on the dep
   */
  @Nullable
  private static Set<Class<? extends BuildConfiguration.Fragment>> getTransitiveFragments(
      Environment env, Label dep, BuildConfiguration parentConfig) throws InterruptedException {
    if (!parentConfig.trimConfigurations()) {
      return parentConfig.getAllFragments().keySet();
    }
    SkyKey fragmentsKey = TransitiveTargetValue.key(dep);
    TransitiveTargetValue transitiveDepInfo = (TransitiveTargetValue) env.getValue(fragmentsKey);
    if (transitiveDepInfo == null) {
      // This should only be possible for tests. In actual runs, this was already called
      // as a routine part of the loading phase.
      // TODO(bazel-team): check this only occurs in a test context.
      return null;
    }
    return transitiveDepInfo.getTransitiveConfigFragments().toSet();
  }

  /**
   * Applies a dynamic configuration transition over a set of build options.
   *
   * @return the build options for the transitioned configuration. If trimResults is true,
   *     only options needed by the required fragments are included. Else the same options as the
   *     original input are included (with different possible values, of course).
   */
  static List<BuildOptions> getDynamicTransitionOptions(BuildOptions fromOptions,
      Attribute.Transition transition,
      Iterable<Class<? extends BuildConfiguration.Fragment>> requiredFragments,
      RuleClassProvider ruleClassProvider, boolean trimResults) {
    List<BuildOptions> result;
    if (transition == Attribute.ConfigurationTransition.NONE) {
      result = ImmutableList.<BuildOptions>of(fromOptions);
    } else if (transition instanceof PatchTransition) {
      // TODO(bazel-team): safety-check that this never mutates fromOptions.
      result = ImmutableList.<BuildOptions>of(((PatchTransition) transition).apply(fromOptions));
    } else if (transition instanceof Attribute.SplitTransition) {
      @SuppressWarnings("unchecked") // Attribute.java doesn't have the BuildOptions symbol.
      List<BuildOptions> toOptions =
          ((Attribute.SplitTransition<BuildOptions>) transition).split(fromOptions);
      if (toOptions.isEmpty()) {
        // When the split returns an empty list, it's signaling it doesn't apply to this instance.
        // So return the original options.
        result = ImmutableList.<BuildOptions>of(fromOptions);
      } else {
        result = toOptions;
      }
    } else {
      throw new IllegalStateException(String.format(
          "unsupported dynamic transition type: %s", transition.getClass().getName()));
    }

    if (!trimResults) {
      return result;
    } else {
      ImmutableList.Builder<BuildOptions> trimmedOptions = ImmutableList.builder();
      for (BuildOptions toOptions : result) {
        trimmedOptions.add(toOptions.trim(
            BuildConfiguration.getOptionsClasses(requiredFragments, ruleClassProvider)));
      }
      return trimmedOptions.build();
    }
  }

  /**
   * Diagnostic helper method for dynamic configurations: checks the config fragments required by
   * a dep against the fragments in its actual configuration. If any are missing, triggers a
   * descriptive "missing fragments" error.
   */
  private static void checkForMissingFragments(Environment env, TargetAndConfiguration ctgValue,
      String attribute, Dependency dep,
      Set<Class<? extends BuildConfiguration.Fragment>> expectedDepFragments)
      throws DependencyEvaluationException {
    Set<String> ctgFragmentNames = new HashSet<>();
    for (BuildConfiguration.Fragment fragment :
        ctgValue.getConfiguration().getAllFragments().values()) {
      ctgFragmentNames.add(fragment.getClass().getSimpleName());
    }
    Set<String> depFragmentNames = new HashSet<>();
    for (Class<? extends BuildConfiguration.Fragment> fragmentClass : expectedDepFragments) {
     depFragmentNames.add(fragmentClass.getSimpleName());
    }
    Set<String> missing = Sets.difference(depFragmentNames, ctgFragmentNames);
    if (!missing.isEmpty()) {
      String msg = String.format(
          "%s: dependency %s from attribute \"%s\" is missing required config fragments: %s",
          ctgValue.getLabel(), dep.getLabel(), attribute, Joiner.on(", ").join(missing));
      env.getListener().handle(Event.error(msg));
      throw new DependencyEvaluationException(new InvalidConfigurationException(msg));
    }
  }

  /**
   * Determines the output ordering of each <attribute, depLabel> ->
   * [dep<config1>, dep<config2>, ...] collection produced by a split transition.
   */
  @VisibleForTesting
  static final Comparator<Dependency> DYNAMIC_SPLIT_DEP_ORDERING =
      new Comparator<Dependency>() {
        @Override
        public int compare(Dependency d1, Dependency d2) {
          return d1.getConfiguration().getMnemonic().compareTo(d2.getConfiguration().getMnemonic());
        }
      };

  /**
   * Helper method for {@link #getDynamicConfigurations}: returns a copy of the output deps
   * using the same key and value ordering as the input deps.
   *
   * @param originalDeps the input deps with the ordering to preserve
   * @param dynamicDeps the unordered output deps
   * @param attributesAndLabels collection of <attribute, depLabel> pairs guaranteed to match
   *   the ordering of originalDeps.entries(). This is a performance optimization: see
   *   {@link #getDynamicConfigurations#attributesAndLabels} for details.
   */
  private static OrderedSetMultimap<Attribute, Dependency> sortDynamicallyConfiguredDeps(
      OrderedSetMultimap<Attribute, Dependency> originalDeps,
      Multimap<AttributeAndLabel, Dependency> dynamicDeps,
      ArrayList<AttributeAndLabel> attributesAndLabels) {
    Iterator<AttributeAndLabel> iterator = attributesAndLabels.iterator();
    OrderedSetMultimap<Attribute, Dependency> result = OrderedSetMultimap.create();
    for (Map.Entry<Attribute, Dependency> depsEntry : originalDeps.entries()) {
      AttributeAndLabel attrAndLabel = iterator.next();
      if (depsEntry.getValue().hasStaticConfiguration()) {
        result.put(attrAndLabel.attribute, depsEntry.getValue());
      } else {
        Collection<Dependency> dynamicAttrDeps = dynamicDeps.get(attrAndLabel);
        Verify.verify(!dynamicAttrDeps.isEmpty());
        if (dynamicAttrDeps.size() > 1) {
          List<Dependency> sortedSplitList = new ArrayList<>(dynamicAttrDeps);
          Collections.sort(sortedSplitList, DYNAMIC_SPLIT_DEP_ORDERING);
          dynamicAttrDeps = sortedSplitList;
        }
        result.putAll(depsEntry.getKey(), dynamicAttrDeps);
      }
    }
    return result;
  }

  /**
   * Merges the each direct dependency configured target with the aspects associated with it.
   *
   * <p>Note that the combination of a configured target and its associated aspects are not
   * represented by a Skyframe node. This is because there can possibly be many different
   * combinations of aspects for a particular configured target, so it would result in a
   * combinatiorial explosion of Skyframe nodes.
   */
  private static OrderedSetMultimap<Attribute, ConfiguredTarget> mergeAspects(
      OrderedSetMultimap<Attribute, Dependency> depValueNames,
      Map<SkyKey, ConfiguredTarget> depConfiguredTargetMap,
      OrderedSetMultimap<Dependency, ConfiguredAspect> depAspectMap)
      throws DuplicateException  {
    OrderedSetMultimap<Attribute, ConfiguredTarget> result = OrderedSetMultimap.create();

    for (Map.Entry<Attribute, Dependency> entry : depValueNames.entries()) {
      Dependency dep = entry.getValue();
      SkyKey depKey = ConfiguredTargetValue.key(dep.getLabel(), dep.getConfiguration());
      ConfiguredTarget depConfiguredTarget = depConfiguredTargetMap.get(depKey);

      result.put(entry.getKey(),
          MergedConfiguredTarget.of(depConfiguredTarget, depAspectMap.get(dep)));
    }

    return result;
  }

  /**
   * Given a list of {@link Dependency} objects, returns a multimap from the
   * {@link Dependency}s too the {@link ConfiguredAspect} instances that should be merged into it.
   *
   * <p>Returns null if the required aspects are not computed yet.
   */
  @Nullable
  private static OrderedSetMultimap<Dependency, ConfiguredAspect> resolveAspectDependencies(
      Environment env,
      Map<SkyKey, ConfiguredTarget> configuredTargetMap,
      Iterable<Dependency> deps,
      NestedSetBuilder<Package> transitivePackages)
      throws AspectCreationException, InterruptedException {
    OrderedSetMultimap<Dependency, ConfiguredAspect> result = OrderedSetMultimap.create();
    Set<SkyKey> allAspectKeys = new HashSet<>();
    for (Dependency dep : deps) {
      allAspectKeys.addAll(getAspectKeys(dep).values());
    }

    Map<SkyKey, ValueOrException2<AspectCreationException, NoSuchThingException>> depAspects =
        env.getValuesOrThrow(allAspectKeys,
            AspectCreationException.class, NoSuchThingException.class);

    for (Dependency dep : deps) {
      Map<AspectDescriptor, SkyKey> aspectToKeys = getAspectKeys(dep);

      for (AspectDeps depAspect : dep.getAspects().getVisibleAspects()) {
        SkyKey aspectKey = aspectToKeys.get(depAspect.getAspect());

        AspectValue aspectValue;
        try {
          // TODO(ulfjack): Catch all thrown AspectCreationException and NoSuchThingException
          // instances and merge them into a single Exception to get full root cause data.
          aspectValue = (AspectValue) depAspects.get(aspectKey).get();
        } catch (NoSuchThingException e) {
          throw new AspectCreationException(
              String.format(
                  "Evaluation of aspect %s on %s failed: %s",
                  depAspect.getAspect().getAspectClass().getName(),
                  dep.getLabel(),
                  e.toString()));
        }

        if (aspectValue == null) {
          // Dependent aspect has either not been computed yet or is in error.
          return null;
        }

        // Validate that aspect is applicable to "bare" configured target.
        ConfiguredTarget associatedTarget = configuredTargetMap
            .get(ConfiguredTargetValue.key(dep.getLabel(), dep.getConfiguration()));
        if (!aspectMatchesConfiguredTarget(associatedTarget, aspectValue.getAspect())) {
          continue;
        }

        result.put(dep, aspectValue.getConfiguredAspect());
        transitivePackages.addTransitive(aspectValue.getTransitivePackages());
      }
    }
    return result;
  }

  private static Map<AspectDescriptor, SkyKey> getAspectKeys(Dependency dep) {
    HashMap<AspectDescriptor, SkyKey> result = new HashMap<>();
    AspectCollection aspects = dep.getAspects();
    for (AspectDeps aspectDeps : aspects.getVisibleAspects()) {
      buildAspectKey(aspectDeps, result, dep);
    }
    return result;
  }

  private static AspectKey buildAspectKey(AspectDeps aspectDeps,
      HashMap<AspectDescriptor, SkyKey> result, Dependency dep) {
    if (result.containsKey(aspectDeps.getAspect())) {
      return (AspectKey) result.get(aspectDeps.getAspect()).argument();
    }

    ImmutableList.Builder<AspectKey> dependentAspects = ImmutableList.builder();
    for (AspectDeps path : aspectDeps.getDependentAspects()) {
      dependentAspects.add(buildAspectKey(path, result, dep));
    }
    AspectKey aspectKey = AspectValue.createAspectKey(
        dep.getLabel(), dep.getConfiguration(),
        dependentAspects.build(),
        aspectDeps.getAspect(),
        dep.getAspectConfiguration(aspectDeps.getAspect()));
    result.put(aspectKey.getAspectDescriptor(), aspectKey.getSkyKey());
    return aspectKey;
  }

  static boolean aspectMatchesConfiguredTarget(final ConfiguredTarget dep, Aspect aspect) {
    if (!aspect.getDefinition().applyToFiles() && !(dep.getTarget() instanceof Rule)) {
      return false;
    }
    if (dep.getTarget().getAssociatedRule() == null) {
      // even aspects that 'apply to files' cannot apply to input files.
      return false;
    }
    return dep.satisfies(aspect.getDefinition().getRequiredProviders());
  }

  /**
   * Returns the set of {@link ConfigMatchingProvider}s that key the configurable attributes used by
   * this rule.
   *
   * <p>>If the configured targets supplying those providers aren't yet resolved by the dependency
   * resolver, returns null.
   */
  @Nullable
  static ImmutableMap<Label, ConfigMatchingProvider> getConfigConditions(
      Target target,
      Environment env,
      SkyframeDependencyResolver resolver,
      TargetAndConfiguration ctgValue,
      NestedSetBuilder<Package> transitivePackages,
      NestedSetBuilder<Label> transitiveLoadingRootCauses)
      throws DependencyEvaluationException, InterruptedException {
    if (!(target instanceof Rule)) {
      return NO_CONFIG_CONDITIONS;
    }

    Map<Label, ConfigMatchingProvider> configConditions = new LinkedHashMap<>();

    // Collect the labels of the configured targets we need to resolve.
    OrderedSetMultimap<Attribute, Label> configLabelMap = OrderedSetMultimap.create();
    RawAttributeMapper attributeMap = RawAttributeMapper.of(((Rule) target));
    for (Attribute a : ((Rule) target).getAttributes()) {
      for (Label configLabel : attributeMap.getConfigurabilityKeys(a.getName(), a.getType())) {
        if (!BuildType.Selector.isReservedLabel(configLabel)) {
          configLabelMap.put(a, target.getLabel().resolveRepositoryRelative(configLabel));
        }
      }
    }
    if (configLabelMap.isEmpty()) {
      return NO_CONFIG_CONDITIONS;
    }

    // Collect the corresponding Skyframe configured target values. Abort early if they haven't
    // been computed yet.
    Collection<Dependency> configValueNames = null;
    try {
      configValueNames = resolver.resolveRuleLabels(
          ctgValue, configLabelMap, transitiveLoadingRootCauses);
    } catch (InconsistentAspectOrderException e) {
      throw new DependencyEvaluationException(e);
    }
    if (env.valuesMissing()) {
      return null;
    }

    // No need to get new configs from Skyframe - config_setting rules always use the current
    // target's config.
    // TODO(bazel-team): remove the need for this special transformation. We can probably do this by
    // simply passing this through trimConfigurations.
    ImmutableList.Builder<Dependency> staticConfigs = ImmutableList.builder();
    for (Dependency dep : configValueNames) {
      staticConfigs.add(Dependency.withConfigurationAndAspects(dep.getLabel(),
          ctgValue.getConfiguration(), dep.getAspects()));
    }
    configValueNames = staticConfigs.build();

    Map<SkyKey, ConfiguredTarget> configValues = resolveConfiguredTargetDependencies(
        env, configValueNames, transitivePackages, transitiveLoadingRootCauses);
    if (configValues == null) {
      return null;
    }

    // Get the configured targets as ConfigMatchingProvider interfaces.
    for (Dependency entry : configValueNames) {
      SkyKey baseKey = ConfiguredTargetValue.key(entry.getLabel(), entry.getConfiguration());
      ConfiguredTarget value = configValues.get(baseKey);
      // The code above guarantees that value is non-null here.
      ConfigMatchingProvider provider = value.getProvider(ConfigMatchingProvider.class);
      if (provider != null) {
        configConditions.put(entry.getLabel(), provider);
      } else {
        // Not a valid provider for configuration conditions.
        String message =
            entry.getLabel() + " is not a valid configuration key for " + target.getLabel();
        env.getListener().handle(Event.error(TargetUtils.getLocationMaybe(target), message));
        throw new DependencyEvaluationException(new ConfiguredValueCreationException(
            message, target.getLabel()));
      }
    }

    return ImmutableMap.copyOf(configConditions);
  }

  /**
   * * Resolves the targets referenced in depValueNames and returns their ConfiguredTarget
   * instances.
   *
   * <p>Returns null if not all instances are available yet.
   */
  @Nullable
  private static Map<SkyKey, ConfiguredTarget> resolveConfiguredTargetDependencies(
      Environment env,
      Collection<Dependency> deps,
      NestedSetBuilder<Package> transitivePackages,
      NestedSetBuilder<Label> transitiveLoadingRootCauses)
      throws DependencyEvaluationException, InterruptedException {
    boolean missedValues = env.valuesMissing();
    boolean failed = false;
    Iterable<SkyKey> depKeys = Iterables.transform(deps,
        input -> ConfiguredTargetValue.key(input.getLabel(), input.getConfiguration()));
    Map<SkyKey, ValueOrException<ConfiguredValueCreationException>> depValuesOrExceptions =
            env.getValuesOrThrow(depKeys, ConfiguredValueCreationException.class);
    Map<SkyKey, ConfiguredTarget> result =
        Maps.newHashMapWithExpectedSize(depValuesOrExceptions.size());
    for (Map.Entry<SkyKey, ValueOrException<ConfiguredValueCreationException>> entry
        : depValuesOrExceptions.entrySet()) {
      try {
        ConfiguredTargetValue depValue = (ConfiguredTargetValue) entry.getValue().get();
        if (depValue == null) {
          missedValues = true;
        } else {
          result.put(entry.getKey(), depValue.getConfiguredTarget());
          transitivePackages.addTransitive(depValue.getTransitivePackages());
        }
      } catch (ConfiguredValueCreationException e) {
        // TODO(ulfjack): If there is an analysis root cause, we drop all loading root causes.
        if (e.getAnalysisRootCause() != null) {
          throw new DependencyEvaluationException(e);
        }
        transitiveLoadingRootCauses.addTransitive(e.loadingRootCauses);
        failed = true;
      }
    }
    if (missedValues) {
      return null;
    } else if (failed) {
      throw new DependencyEvaluationException(
          new ConfiguredValueCreationException(transitiveLoadingRootCauses.build()));
    } else {
      return result;
    }
  }


  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(((ConfiguredTargetKey) skyKey.argument()).getLabel());
  }

  @Nullable
  private ConfiguredTargetValue createConfiguredTarget(
      SkyframeBuildView view,
      Environment env,
      Target target,
      BuildConfiguration configuration,
      OrderedSetMultimap<Attribute, ConfiguredTarget> depValueMap,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      @Nullable ToolchainContext toolchainContext,
      NestedSetBuilder<Package> transitivePackages)
      throws ConfiguredTargetFunctionException, InterruptedException {
    StoredEventHandler events = new StoredEventHandler();
    BuildConfiguration ownerConfig =
        ConfiguredTargetFactory.getArtifactOwnerConfiguration(env, configuration);
    if (env.valuesMissing()) {
      return null;
    }
    CachingAnalysisEnvironment analysisEnvironment = view.createAnalysisEnvironment(
        new ConfiguredTargetKey(target.getLabel(), ownerConfig), false,
        events, env, configuration);
    if (env.valuesMissing()) {
      return null;
    }

    Preconditions.checkNotNull(depValueMap);
    ConfiguredTarget configuredTarget =
        view.createConfiguredTarget(
            target,
            configuration,
            analysisEnvironment,
            depValueMap,
            configConditions,
            toolchainContext);

    events.replayOn(env.getListener());
    if (events.hasErrors()) {
      analysisEnvironment.disable(target);
      throw new ConfiguredTargetFunctionException(new ConfiguredValueCreationException(
          "Analysis of target '" + target.getLabel() + "' failed; build aborted",
          target.getLabel()));
    }
    Preconditions.checkState(!analysisEnvironment.hasErrors(),
        "Analysis environment hasError() but no errors reported");
    if (env.valuesMissing()) {
      return null;
    }

    analysisEnvironment.disable(target);
    Preconditions.checkNotNull(configuredTarget, target);

    GeneratingActions generatingActions;
    // Check for conflicting actions within this configured target (that indicates a bug in the
    // rule implementation).
    try {
      generatingActions = Actions.filterSharedActionsAndThrowActionConflict(
          analysisEnvironment.getRegisteredActions());
    } catch (ActionConflictException e) {
      throw new ConfiguredTargetFunctionException(e);
    }
    return new ConfiguredTargetValue(
        configuredTarget,
        generatingActions,
        transitivePackages.build(),
        removeActionsAfterEvaluation.get());
  }

  /**
   * An exception indicating that there was a problem during the construction of
   * a ConfiguredTargetValue.
   */
  public static final class ConfiguredValueCreationException extends Exception {
    private final NestedSet<Label> loadingRootCauses;
    // TODO(ulfjack): Collect all analysis root causes, not just the first one.
    @Nullable private final Label analysisRootCause;

    public ConfiguredValueCreationException(String message, Label currentTarget) {
      super(message);
      this.loadingRootCauses = NestedSetBuilder.<Label>emptySet(Order.STABLE_ORDER);
      this.analysisRootCause = Preconditions.checkNotNull(currentTarget);
    }

    public ConfiguredValueCreationException(String message, NestedSet<Label> rootCauses) {
      super(message);
      this.loadingRootCauses = rootCauses;
      this.analysisRootCause = null;
    }

    public ConfiguredValueCreationException(NestedSet<Label> rootCauses) {
      this("Loading failed", rootCauses);
    }

    public ConfiguredValueCreationException(String message) {
      this(message, NestedSetBuilder.<Label>emptySet(Order.STABLE_ORDER));
    }

    public NestedSet<Label> getRootCauses() {
      return loadingRootCauses;
    }

    public Label getAnalysisRootCause() {
      return analysisRootCause;
    }
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link ConfiguredTargetFunction#compute}.
   */
  public static final class ConfiguredTargetFunctionException extends SkyFunctionException {
    public ConfiguredTargetFunctionException(NoSuchThingException e) {
      super(e, Transience.PERSISTENT);
    }

    private ConfiguredTargetFunctionException(ConfiguredValueCreationException e) {
      super(e, Transience.PERSISTENT);
    }

    private ConfiguredTargetFunctionException(ActionConflictException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
