// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.config;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Verify;
import com.google.common.base.VerifyException;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.LinkedListMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.analysis.ConfigurationsCollector;
import com.google.devtools.build.lib.analysis.ConfigurationsResult;
import com.google.devtools.build.lib.analysis.Dependency;
import com.google.devtools.build.lib.analysis.DependencyKind;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NullTransition;
import com.google.devtools.build.lib.analysis.skylark.StarlarkTransition;
import com.google.devtools.build.lib.analysis.skylark.StarlarkTransition.TransitionException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.PlatformMappingValue;
import com.google.devtools.build.lib.skyframe.TransitiveTargetKey;
import com.google.devtools.build.lib.skyframe.TransitiveTargetValue;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.ValueOrException;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Turns configuration transition requests into actual configurations.
 *
 * <p>This involves:
 * <ol>
 *   <li>Patching a source configuration's options with the transition
 *   <li>If {@link BuildConfiguration#trimConfigurations} is true, trimming configuration fragments
 *       to only those needed by the destination target and its transitive dependencies
 *   <li>Getting the destination configuration from Skyframe
 *   </ol>
 *
 * <p>For the work of determining the transition requests themselves, see
 * {@link TransitionResolver}.
 */
public final class ConfigurationResolver {
  /**
   * Translates a set of {@link Dependency} objects with configuration transition requests to the
   * same objects with resolved configurations.
   *
   * <p>If {@link BuildConfiguration#trimConfigurations()} is true, these configurations only
   * contain the fragments needed by the dep and its transitive closure. Else they unconditionally
   * include all fragments.
   *
   * <p>This method is heavily performance-optimized. Because {@link
   * com.google.devtools.build.lib.skyframe.ConfiguredTargetFunction} calls it over every edge in
   * the configured target graph, small inefficiencies can have observable impact on analysis time.
   * Keep this in mind when making modifications and performance-test any changes you make.
   *
   * @param env Skyframe evaluation environment
   * @param ctgValue the label and configuration of the source target
   * @param originalDeps the transition requests for each dep and each dependency kind
   * @param hostConfiguration the host configuration
   * @param ruleClassProvider provider for determining the right configuration fragments for deps
   * @return a mapping from each dependency kind in the source target to the {@link
   *     BuildConfiguration}s and {@link Label}s for the deps under that dependency kind . Returns
   *     null if not all Skyframe dependencies are available.
   */
  @Nullable
  public static OrderedSetMultimap<DependencyKind, Dependency> resolveConfigurations(
      SkyFunction.Environment env,
      TargetAndConfiguration ctgValue,
      OrderedSetMultimap<DependencyKind, Dependency> originalDeps,
      BuildConfiguration hostConfiguration,
      RuleClassProvider ruleClassProvider,
      BuildOptions defaultBuildOptions)
      throws DependencyEvaluationException, InterruptedException {

    // Maps each Skyframe-evaluated BuildConfiguration to the dependencies that need that
    // configuration paired with a transition key corresponding to the BuildConfiguration. For cases
    // where Skyframe isn't needed to get the configuration (e.g. when we just re-used the original
    // rule's configuration), we should skip this outright.
    Multimap<SkyKey, Pair<Map.Entry<DependencyKind, Dependency>, String>> keysToEntries =
        LinkedListMultimap.create();

    // Stores the result of applying a transition to the current configuration using a
    // particular subset of fragments. By caching this, we save from redundantly computing the
    // same transition for every dependency edge that requests that transition. This can have
    // real effect on analysis time for commonly triggered transitions.
    //
    // Split transitions may map to one or multiple values. Patch transitions map to exactly one.
    Map<FragmentsAndTransition, Map<String, BuildOptions>> transitionsMap = new LinkedHashMap<>();

    BuildConfiguration currentConfiguration = ctgValue.getConfiguration();

    // Stores the configuration-resolved versions of each dependency. This method must preserve the
    // original label ordering of each attribute. For example, if originalDeps.get("data") is
    // [":a", ":b"], the resolved variant must also be [":a", ":b"] in the same order. Because we
    // may not actualize the results in order (some results need Skyframe-evaluated configurations
    // while others can be computed trivially), we dump them all into this map, then as a final step
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
    Multimap<DependencyEdge, Dependency> resolvedDeps = LinkedHashMultimap.create();

    // Performance optimization: This method iterates over originalDeps twice. By storing
    // DependencyEdge instances in this list, we avoid having to recreate them the second time
    // (particularly avoid recomputing their hash codes). Profiling shows this shaves 25% off this
    // method's execution time (at the time of this comment).
    ArrayList<DependencyEdge> attributesAndLabels = new ArrayList<>(originalDeps.size());

    for (Map.Entry<DependencyKind, Dependency> depsEntry : originalDeps.entries()) {
      Dependency dep = depsEntry.getValue();
      DependencyEdge dependencyEdge = new DependencyEdge(depsEntry.getKey(), dep.getLabel());
      attributesAndLabels.add(dependencyEdge);
      // DependencyResolver should never emit a Dependency with an explicit configuration
      Preconditions.checkState(!dep.hasExplicitConfiguration());

      // The null configuration can be trivially computed (it's, well, null), so special-case that
      // transition here and skip the rest of the logic. A *lot* of targets have null deps, so
      // this produces real savings. Profiling tests over a simple cc_binary show this saves ~1% of
      // total analysis phase time.
      ConfigurationTransition transition = dep.getTransition();
      if (transition == NullTransition.INSTANCE) {
        putOnlyEntry(
            resolvedDeps, dependencyEdge, Dependency.withNullConfiguration(dep.getLabel()));
        continue;
      }

      // Figure out the required fragments for this dep and its transitive closure.
      Set<Class<? extends Fragment>> depFragments =
          getTransitiveFragments(env, dep.getLabel(), ctgValue.getConfiguration());
      if (depFragments == null) {
        return null;
      }
      // TODO(gregce): remove the below call once we have confidence trimmed configurations always
      // provide needed fragments. This unnecessarily drags performance on the critical path (up
      // to 0.5% of total analysis time as profiled over a simple cc_binary).
      if (ctgValue.getConfiguration().trimConfigurations()) {
        checkForMissingFragments(
            env, ctgValue, dependencyEdge.dependencyKind.getAttribute(), dep, depFragments);
      }

      boolean sameFragments =
          depFragments.equals(currentConfiguration.fragmentClasses().fragmentClasses());

      if (sameFragments) {
        if (transition == NoTransition.INSTANCE) {
          if (ctgValue.getConfiguration().trimConfigurationsRetroactively()
              && !dep.getAspects().isEmpty()) {
            String message =
                ctgValue.getLabel()
                    + " has aspects attached, but these are not supported in retroactive"
                    + " trimming mode.";
            env.getListener()
                .handle(Event.error(TargetUtils.getLocationMaybe(ctgValue.getTarget()), message));
            throw new DependencyEvaluationException(new InvalidConfigurationException(message));
          }
          // The dep uses the same exact configuration. Let's re-use the current configuration and
          // skip adding a Skyframe dependency edge on it.
          putOnlyEntry(
              resolvedDeps,
              dependencyEdge,
              Dependency.withConfigurationAndAspects(
                  dep.getLabel(), ctgValue.getConfiguration(), dep.getAspects()));
          continue;
        } else if (transition.isHostTransition()) {
          // The current rule's host configuration can also be used for the dep. We short-circuit
          // the standard transition logic for host transitions because these transitions are
          // uniquely frequent. It's possible, e.g., for every node in the configured target graph
          // to incur multiple host transitions. So we aggressively optimize to avoid hurting
          // analysis time.
          if (hostConfiguration.trimConfigurationsRetroactively() && !dep.getAspects().isEmpty()) {
            String message =
                ctgValue.getLabel()
                    + " has aspects attached, but these are not supported in retroactive"
                    + " trimming mode.";
            env.getListener()
                .handle(Event.error(TargetUtils.getLocationMaybe(ctgValue.getTarget()), message));
            throw new DependencyEvaluationException(new InvalidConfigurationException(message));
          }
          putOnlyEntry(
              resolvedDeps,
              dependencyEdge,
              Dependency.withConfigurationAndAspects(
                  dep.getLabel(), hostConfiguration, dep.getAspects()));
          continue;
        }
      }

      // Apply the transition or use the cached result if it was already applied.
      FragmentsAndTransition transitionKey = new FragmentsAndTransition(depFragments, transition);
      Map<String, BuildOptions> toOptions = transitionsMap.get(transitionKey);
      if (toOptions == null) {
        try {
          HashMap<PackageValue.Key, PackageValue> buildSettingPackages =
              StarlarkTransition.getBuildSettingPackages(env, transition);
          if (buildSettingPackages == null) {
            return null;
          }
          toOptions =
              applyTransition(
                  currentConfiguration.getOptions(),
                  transition,
                  buildSettingPackages,
                  env.getListener());
        } catch (TransitionException e) {
          throw new DependencyEvaluationException(e);
        }
        transitionsMap.put(transitionKey, toOptions);
      }
      // If the transition doesn't change the configuration, trivially re-use the original
      // configuration.
      if (sameFragments
          && toOptions.size() == 1
          && Iterables.getOnlyElement(toOptions.values())
              .equals(currentConfiguration.getOptions())) {
        if (ctgValue.getConfiguration().trimConfigurationsRetroactively()
            && !dep.getAspects().isEmpty()) {
          String message =
              ctgValue.getLabel()
                  + " has aspects attached, but these are not supported in retroactive"
                  + " trimming mode.";
          env.getListener()
              .handle(Event.error(TargetUtils.getLocationMaybe(ctgValue.getTarget()), message));
          throw new DependencyEvaluationException(new InvalidConfigurationException(message));
        }
        putOnlyEntry(
            resolvedDeps,
            dependencyEdge,
            Dependency.withConfigurationAspectsAndTransitionKey(
                dep.getLabel(), ctgValue.getConfiguration(), dep.getAspects(), null));
        continue;
      }

      // If we get here, we have to get the configuration from Skyframe.
      PathFragment platformMappingPath =
          currentConfiguration.getOptions().get(PlatformOptions.class).platformMappings;
      PlatformMappingValue platformMappingValue =
          (PlatformMappingValue) env.getValue(PlatformMappingValue.Key.create(platformMappingPath));
      if (platformMappingValue == null) {
        return null;
      }

      try {
        for (Map.Entry<String, BuildOptions> optionsEntry : toOptions.entrySet()) {
          if (sameFragments) {
            keysToEntries.put(
                BuildConfigurationValue.keyWithPlatformMapping(
                    platformMappingValue,
                    defaultBuildOptions,
                    currentConfiguration.fragmentClasses(),
                    BuildOptions.diffForReconstruction(
                        defaultBuildOptions, optionsEntry.getValue())),
                new Pair<>(depsEntry, optionsEntry.getKey()));

          } else {
            keysToEntries.put(
                BuildConfigurationValue.keyWithPlatformMapping(
                    platformMappingValue,
                    defaultBuildOptions,
                    depFragments,
                    BuildOptions.diffForReconstruction(
                        defaultBuildOptions, optionsEntry.getValue())),
                new Pair<>(depsEntry, optionsEntry.getKey()));
          }
        }
      } catch (OptionsParsingException e) {
        throw new DependencyEvaluationException(new InvalidConfigurationException(e));
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
    // dependencies in ConfiguredTargetFunction to make that impractical.
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
        BuildConfiguration trimmedConfig =
            ((BuildConfigurationValue) valueOrException.get()).getConfiguration();
        for (Pair<Map.Entry<DependencyKind, Dependency>, String> info : keysToEntries.get(key)) {
          Dependency originalDep = info.first.getValue();
          if (trimmedConfig.trimConfigurationsRetroactively()
              && !originalDep.getAspects().isEmpty()) {
            String message =
                ctgValue.getLabel()
                    + " has aspects attached, but these are not supported in retroactive"
                    + " trimming mode.";
            env.getListener()
                .handle(Event.error(TargetUtils.getLocationMaybe(ctgValue.getTarget()), message));
            throw new DependencyEvaluationException(new InvalidConfigurationException(message));
          }
          DependencyEdge attr = new DependencyEdge(info.first.getKey(), originalDep.getLabel());
          Dependency resolvedDep =
              Dependency.withConfigurationAspectsAndTransitionKey(
                  originalDep.getLabel(), trimmedConfig, originalDep.getAspects(), info.second);
          Attribute attribute = attr.dependencyKind.getAttribute();
          if (attribute != null && attribute.getTransitionFactory().isSplit()) {
            resolvedDeps.put(attr, resolvedDep);
          } else {
            putOnlyEntry(resolvedDeps, attr, resolvedDep);
          }
        }
      }
    } catch (InvalidConfigurationException e) {
      throw new DependencyEvaluationException(e);
    }

    return sortResolvedDeps(originalDeps, resolvedDeps, attributesAndLabels);
  }

  /**
   * Encapsulates a set of config fragments and a config transition. This can be used to determine
   * the exact build options needed to set a configuration.
   */
  @ThreadSafety.Immutable
  private static final class FragmentsAndTransition {
    // Treat this as immutable. The only reason this isn't an ImmutableSet is because it
    // gets bound to a NestedSet.toSet() reference, which returns a Set interface.
    final Set<Class<? extends Fragment>> fragments;
    final ConfigurationTransition transition;
    private final int hashCode;

    FragmentsAndTransition(
        Set<Class<? extends Fragment>> fragments, ConfigurationTransition transition) {
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
        if (!(o instanceof FragmentsAndTransition)) {
          return false;
        }
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
   * Encapsulates an <attribute, label> pair that can be used to map from an input dependency to a
   * trimmed dependency.
   */
  @ThreadSafety.Immutable
  private static final class DependencyEdge {
    final DependencyKind dependencyKind;
    final Label label;
    Integer hashCode;

    DependencyEdge(DependencyKind dependencyKind, Label label) {
      this.dependencyKind = dependencyKind;
      this.label = label;
    }

    @Override
    public boolean equals(Object o) {
      if (!(o instanceof DependencyEdge)) {
        return false;
      }
      DependencyEdge other = (DependencyEdge) o;
      return Objects.equals(other.dependencyKind, dependencyKind) && other.label.equals(label);
    }

    @Override
    public int hashCode() {
      if (hashCode == null) {
        // Not every <Attribute, Label> pair gets hashed. So only evaluate for the instances that
        // need it. This can significantly reduce the number of evaluations.
        hashCode = Objects.hash(this.dependencyKind, this.label);
      }
      return hashCode;
    }

    @Override
    public String toString() {
      Attribute attribute = dependencyKind.getAttribute();

      return "DependencyEdge{attribute="
          + (attribute == null ? "(null)" : attribute)
          + ", label="
          + label
          + "}";
    }
  }

  /**
   * Variation of {@link Multimap#put} that triggers an exception if a value already exists.
   */
  @VisibleForTesting
  public static <K, V> void putOnlyEntry(Multimap<K, V> map, K key, V value) {
    // Performance note: while "Verify.verify(!map.containsKey(key, value), String.format(...)))"
    // is simpler code, profiling shows a substantial performance penalty to that approach
    // (~10% extra analysis phase time on a simple cc_binary). Most of that is from the cost of
    // evaluating value.toString() on every call. This approach essentially eliminates the overhead.
    if (map.containsKey(key)) {
      throw new VerifyException(
          String.format("couldn't insert %s: map already has values for key %s: %s",
              value.toString(), key.toString(), map.get(key).toString()));
    }
    map.put(key, value);
  }

  /**
   * Returns the configuration fragments required by a dep and its transitive closure. Returns null
   * if Skyframe dependencies aren't yet available.
   *
   * @param env Skyframe evaluation environment
   * @param dep label of the dep to check
   * @param parentConfig configuration of the rule depending on the dep
   */
  @Nullable
  private static Set<Class<? extends Fragment>> getTransitiveFragments(
      SkyFunction.Environment env, Label dep, BuildConfiguration parentConfig)
      throws InterruptedException {
    if (!parentConfig.trimConfigurations()) {
      return parentConfig.getFragmentsMap().keySet();
    }
    SkyKey fragmentsKey = TransitiveTargetKey.of(dep);
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
   * Applies a configuration transition over a set of build options.
   *
   * <p>prework - load all default values for read build settings in Starlark transitions (by
   * design, {@link BuildOptions} never holds default values of build settings)
   *
   * <p>postwork - replay events/throw errors from transition implementation function and validate
   * the outputs of the transition
   *
   * @return the build options for the transitioned configuration.
   */
  @VisibleForTesting
  public static Map<String, BuildOptions> applyTransition(
      BuildOptions fromOptions,
      ConfigurationTransition transition,
      Map<PackageValue.Key, PackageValue> buildSettingPackages,
      ExtendedEventHandler eventHandler)
      throws TransitionException {
    boolean doesStarlarkTransition = StarlarkTransition.doesStarlarkTransition(transition);
    if (doesStarlarkTransition) {
      fromOptions =
          addDefaultStarlarkOptions(
              fromOptions,
              StarlarkTransition.getDefaultInputValues(buildSettingPackages, transition));
    }

    // TODO(bazel-team): Add safety-check that this never mutates fromOptions.
    StoredEventHandler handlerWithErrorStatus = new StoredEventHandler();
    Map<String, BuildOptions> result = transition.apply(fromOptions, handlerWithErrorStatus);

    if (doesStarlarkTransition) {
      // We use a temporary StoredEventHandler instead of the caller's event handler because
      // StarlarkTransition.validate assumes no errors occurred. We need a StoredEventHandler to be
      // able to check that, and fail out early if there are errors.
      //
      // TODO(bazel-team): harden StarlarkTransition.validate so we can eliminate this step.
      // StarlarkRuleTransitionProviderTest#testAliasedBuildSetting_outputReturnMismatch shows the
      // effect.
      handlerWithErrorStatus.replayOn(eventHandler);
      if (handlerWithErrorStatus.hasErrors()) {
        throw new TransitionException("Errors encountered while applying Starlark transition");
      }
      result = StarlarkTransition.validate(transition, buildSettingPackages, result);
    }
    return result;
  }

  private static BuildOptions addDefaultStarlarkOptions(
      BuildOptions fromOptions, ImmutableMap<Label, Object> buildSettingDefaults) {
    BuildOptions.Builder optionsWithDefaults = null;
    for (Map.Entry<Label, Object> buildSettingDefault : buildSettingDefaults.entrySet()) {
      Label buildSetting = buildSettingDefault.getKey();
      if (!fromOptions.getStarlarkOptions().containsKey(buildSetting)) {
        if (optionsWithDefaults == null) {
          optionsWithDefaults = fromOptions.toBuilder();
        }
        optionsWithDefaults.addStarlarkOption(buildSetting, buildSettingDefault.getValue());
      }
    }
    return optionsWithDefaults == null ? fromOptions : optionsWithDefaults.build();
  }

  /**
   * Checks the config fragments required by a dep against the fragments in its actual
   * configuration. If any are missing, triggers a descriptive "missing fragments" error.
   */
  private static void checkForMissingFragments(
      SkyFunction.Environment env,
      TargetAndConfiguration ctgValue,
      Attribute attribute,
      Dependency dep,
      Set<Class<? extends Fragment>> expectedDepFragments)
      throws DependencyEvaluationException {
    Set<String> ctgFragmentNames = new HashSet<>();
    for (Fragment fragment : ctgValue.getConfiguration().getFragmentsMap().values()) {
      ctgFragmentNames.add(fragment.getClass().getSimpleName());
    }
    Set<String> depFragmentNames = new HashSet<>();
    for (Class<? extends Fragment> fragmentClass : expectedDepFragments) {
     depFragmentNames.add(fragmentClass.getSimpleName());
    }
    Set<String> missing = Sets.difference(depFragmentNames, ctgFragmentNames);
    if (!missing.isEmpty()) {
      String msg =
          String.format(
              "%s: dependency %s from attribute \"%s\" is missing required config fragments: %s",
              ctgValue.getLabel(),
              dep.getLabel(),
              attribute == null ? "(null)" : attribute.getName(),
              Joiner.on(", ").join(missing));
      env.getListener().handle(Event.error(msg));
      throw new DependencyEvaluationException(new InvalidConfigurationException(msg));
    }
  }

  /**
   * Determines the output ordering of each <attribute, depLabel> ->
   * [dep<config1>, dep<config2>, ...] collection produced by a split transition.
   */
  @VisibleForTesting
  public static final Comparator<Dependency> SPLIT_DEP_ORDERING =
      new Comparator<Dependency>() {
        @Override
        public int compare(Dependency d1, Dependency d2) {
          return d1.getConfiguration().getMnemonic().compareTo(d2.getConfiguration().getMnemonic());
        }
      };

  /**
   * Returns a copy of the output deps using the same key and value ordering as the input deps.
   *
   * @param originalDeps the input deps with the ordering to preserve
   * @param resolvedDeps the unordered output deps
   * @param attributesAndLabels collection of <attribute, depLabel> pairs guaranteed to match the
   *     ordering of originalDeps.entries(). This is a performance optimization: see {@link
   *     #resolveConfigurations#attributesAndLabels} for details.
   */
  private static OrderedSetMultimap<DependencyKind, Dependency> sortResolvedDeps(
      OrderedSetMultimap<DependencyKind, Dependency> originalDeps,
      Multimap<DependencyEdge, Dependency> resolvedDeps,
      ArrayList<DependencyEdge> attributesAndLabels) {
    Iterator<DependencyEdge> iterator = attributesAndLabels.iterator();
    OrderedSetMultimap<DependencyKind, Dependency> result = OrderedSetMultimap.create();
    for (Map.Entry<DependencyKind, Dependency> depsEntry : originalDeps.entries()) {
      DependencyEdge edge = iterator.next();
      if (depsEntry.getValue().hasExplicitConfiguration()) {
        result.put(edge.dependencyKind, depsEntry.getValue());
      } else {
        Collection<Dependency> resolvedDepWithSplit = resolvedDeps.get(edge);
        Verify.verify(!resolvedDepWithSplit.isEmpty());
        if (resolvedDepWithSplit.size() > 1) {
          List<Dependency> sortedSplitList = new ArrayList<>(resolvedDepWithSplit);
          Collections.sort(sortedSplitList, SPLIT_DEP_ORDERING);
          resolvedDepWithSplit = sortedSplitList;
        }
        result.putAll(depsEntry.getKey(), resolvedDepWithSplit);
      }
    }
    return result;
  }

  /**
   * This method allows resolution of configurations outside of a skyfunction call.
   *
   * <p>Unlike {@link #resolveConfigurations}, this doesn't expect the current context to be
   * evaluating dependencies of a parent target. So this method is also suitable for top-level
   * targets.
   *
   * <p>Resolution consists of two steps:
   *
   * <ol>
   *   <li>Apply the per-target transitions specified in {@code targetsToEvaluate}. This can be
   *       used, e.g., to apply {@link
   *       com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory}s over global
   *       top-level configurations.
   *   <li>(Optionally) trim configurations to only the fragments the targets actually need. This is
   *       triggered by {@link BuildConfiguration#trimConfigurations}.
   * </ol>
   *
   * <p>Preserves the original input order (but merges duplicate nodes that might occur due to
   * top-level configuration transitions) . Uses original (untrimmed, pre-transition) configurations
   * for targets that can't be evaluated (e.g. due to loading phase errors).
   *
   * <p>This is suitable for feeding {@link
   * com.google.devtools.build.lib.skyframe.ConfiguredTargetValue} keys: as general principle {@link
   * com.google.devtools.build.lib.analysis.ConfiguredTarget}s should have exactly as much
   * information in their configurations as they need to evaluate and no more (e.g. there's no need
   * for Android settings in a C++ configured target).
   *
   * @param defaultContext the original targets and starting configurations before applying rule
   *     transitions and trimming. When actual configurations can't be evaluated, these values are
   *     returned as defaults. See TODO below.
   * @param targetsToEvaluate the inputs repackaged as dependencies, including rule-specific
   *     transitions
   * @param eventHandler the error event handler
   * @param configurationsCollector the collector which finds configurations for dependencies
   */
  // TODO(bazel-team): error out early for targets that fail - failed configuration evaluations
  //   should never make it through analysis (and especially not seed ConfiguredTargetValues)
  // TODO(gregce): merge this more with resolveConfigurations? One crucial difference is
  //   resolveConfigurations can null-return on missing deps since it executes inside Skyfunctions.
  // Keep this in sync with {@link PrepareAnalysisPhaseFunction#resolveConfigurations}.
  public static TopLevelTargetsAndConfigsResult getConfigurationsFromExecutor(
      Iterable<TargetAndConfiguration> defaultContext,
      Multimap<BuildConfiguration, Dependency> targetsToEvaluate,
      ExtendedEventHandler eventHandler,
      ConfigurationsCollector configurationsCollector)
      throws InvalidConfigurationException {

    Map<Label, Target> labelsToTargets = new HashMap<>();
    for (TargetAndConfiguration targetAndConfig : defaultContext) {
      labelsToTargets.put(targetAndConfig.getLabel(), targetAndConfig.getTarget());
    }

    // Maps <target, originalConfig> pairs to <target, finalConfig> pairs for targets that
    // could be successfully Skyframe-evaluated.
    Map<TargetAndConfiguration, TargetAndConfiguration> successfullyEvaluatedTargets =
        new LinkedHashMap<>();
    boolean hasError = false;
    if (!targetsToEvaluate.isEmpty()) {
      for (BuildConfiguration fromConfig : targetsToEvaluate.keySet()) {
        ConfigurationsResult configurationsResult =
            configurationsCollector.getConfigurations(
                eventHandler, fromConfig.getOptions(), targetsToEvaluate.get(fromConfig));
        hasError |= configurationsResult.hasError();
        for (Map.Entry<Dependency, BuildConfiguration> evaluatedTarget :
            configurationsResult.getConfigurationMap().entries()) {
          Target target = labelsToTargets.get(evaluatedTarget.getKey().getLabel());
          successfullyEvaluatedTargets.put(
              new TargetAndConfiguration(target, fromConfig),
              new TargetAndConfiguration(target, evaluatedTarget.getValue()));
        }
      }
    }

    LinkedHashSet<TargetAndConfiguration> result = new LinkedHashSet<>();
    for (TargetAndConfiguration originalInput : defaultContext) {
      if (successfullyEvaluatedTargets.containsKey(originalInput)) {
        // The configuration was successfully evaluated.
        result.add(successfullyEvaluatedTargets.get(originalInput));
      } else {
        // Either the configuration couldn't be determined (e.g. loading phase error) or it's null.
        result.add(originalInput);
      }
    }
    return new TopLevelTargetsAndConfigsResult(result, hasError);
  }

  /**
   * The result of {@link #getConfigurationsFromExecutor} which also registers if an error was
   * recorded.
   */
  public static class TopLevelTargetsAndConfigsResult {
    private final Collection<TargetAndConfiguration> configurations;
    private final boolean hasError;

    public TopLevelTargetsAndConfigsResult(
        Collection<TargetAndConfiguration> configurations, boolean hasError) {
      this.configurations = configurations;
      this.hasError = hasError;
    }

    public boolean hasError() {
      return hasError;
    }

    public Collection<TargetAndConfiguration> getTargetsAndConfigs() {
      return configurations;
    }
  }
}

