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
package com.google.devtools.build.lib.query2.cquery;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.devtools.build.lib.analysis.producers.TargetAndConfigurationProducer.computeTransition;

import com.google.auto.value.AutoValue;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimaps;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.DependencyKind;
import com.google.devtools.build.lib.analysis.DependencyKind.NonAttributeDependencyKind;
import com.google.devtools.build.lib.analysis.DependencyKind.ToolchainDependencyKind;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.StarlarkTransitionCache;
import com.google.devtools.build.lib.analysis.config.transitions.ComposingTransition;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NullTransition;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.constraints.IncompatibleTargetChecker.IncompatibleTargetException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetEvaluationExceptions.ReportedException;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetEvaluationExceptions.UnreportedException;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.DependencyResolver;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import java.util.Collection;
import java.util.Map;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * TransitionResolver resolves the dependencies of a ConfiguredTarget, reporting which
 * configurations its dependencies are actually needed in according to the transitions applied to
 * them. This class has been extracted from TransitionsOutputFormatterCallback.java so that it can
 * be used in both ProtoOutputFormatterCallback and TransitionsOutputFormatterCallback
 */
public class CqueryTransitionResolver {

  /**
   * ResolvedTransition represents a single edge in the dependency graph, between some target and a
   * target it depends on, reachable via a single attribute.
   */
  @AutoValue
  @Immutable
  public abstract static class ResolvedTransition {

    static ResolvedTransition create(
        Label label,
        ImmutableCollection<BuildOptions> buildOptions,
        String attributeName,
        String transitionName) {
      return new AutoValue_CqueryTransitionResolver_ResolvedTransition(
          label, buildOptions, attributeName, transitionName);
    }

    /** The label of the target being depended on. */
    abstract Label label();

    /**
     * The configuration(s) this edge results in. This is a collection because a split transition
     * may lead to a single attribute requesting a dependency in multiple configurations.
     *
     * <p>If a target is depended on via two attributes, separate ResolvedTransitions should be
     * used, rather than combining the two into a single ResolvedTransition with multiple options.
     *
     * <p>If no transition was applied to an attribute, this collection will be empty.
     */
    abstract ImmutableCollection<BuildOptions> options();

    /** The name of the attribute via which the dependency was requested. */
    abstract String attributeName();

    /** The name of the transition applied to the attribute. */
    abstract String transitionName();
  }

  private final ExtendedEventHandler eventHandler;
  private final ConfiguredTargetAccessor accessor;
  private final CqueryThreadsafeCallback cqueryThreadsafeCallback;
  private final RuleClassProvider ruleClassProvider;
  private final StarlarkTransitionCache transitionCache;

  public CqueryTransitionResolver(
      ExtendedEventHandler eventHandler,
      ConfiguredTargetAccessor accessor,
      CqueryThreadsafeCallback cqueryThreadsafeCallback,
      RuleClassProvider ruleClassProvider,
      StarlarkTransitionCache transitionCache) {
    this.eventHandler = eventHandler;
    this.accessor = accessor;
    this.cqueryThreadsafeCallback = cqueryThreadsafeCallback;
    this.ruleClassProvider = ruleClassProvider;
    this.transitionCache = transitionCache;
  }

  /**
   * Return the set of dependencies of a ConfiguredTarget, including information about the
   * configuration transitions applied to the dependencies.
   *
   * @see ResolvedTransition for more details.
   * @param configuredTarget the configured target whose dependencies are being looked up.
   */
  public ImmutableSet<ResolvedTransition> dependencies(ConfiguredTarget configuredTarget)
      throws EvaluateException, InterruptedException {
    if (!(configuredTarget instanceof RuleConfiguredTarget)) {
      return ImmutableSet.of();
    }

    Target target = accessor.getTarget(configuredTarget);
    BuildConfigurationValue configuration =
        cqueryThreadsafeCallback.getConfiguration(configuredTarget.getConfigurationKey());

    var targetAndConfiguration = new TargetAndConfiguration(target, configuration);
    var attributeTransitionCollector =
        HashBasedTable.<DependencyKind, Label, ConfigurationTransition>create();
    var state =
        DependencyResolver.State.createForCquery(
            targetAndConfiguration, attributeTransitionCollector::put);

    var producer = new DependencyResolver(targetAndConfiguration);
    try {
      if (!producer.evaluate(
          state,
          ConfiguredTargetKey.fromConfiguredTarget(configuredTarget),
          ruleClassProvider,
          transitionCache,
          /* semaphoreLocker= */ () -> {},
          accessor.getLookupEnvironment(),
          eventHandler)) {
        throw new EvaluateException("DependencyResolver.evaluate did not complete");
      }
    } catch (ReportedException | UnreportedException | IncompatibleTargetException e) {
      throw new EvaluateException(e.getMessage());
    }

    if (!state.transitiveRootCauses().isEmpty()) {
      throw new EvaluateException(
          "expected empty: " + state.transitiveRootCauses().build().toList());
    }

    OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> deps = producer.getDepValueMap();

    var resolved = ImmutableSet.<ResolvedTransition>builder();
    for (Map.Entry<DependencyKind, Collection<ConfiguredTargetAndData>> entry :
        deps.asMap().entrySet()) {
      DependencyKind kind = entry.getKey();
      if (kind instanceof NonAttributeDependencyKind) {
        continue; // No attribute edge to report.
      }

      // There can be multiple labels under a given kind. Groups the targets by label.
      ImmutableListMultimap<Label, ConfiguredTargetAndData> targetsByLabel =
          Multimaps.index(
              entry.getValue(),
              prerequisite -> prerequisite.getConfiguredTarget().getOriginalLabel());
      String dependencyName = getDependencyName(kind);
      Map<Label, ConfigurationTransition> attributeTransitions =
          attributeTransitionCollector.row(kind);

      for (Map.Entry<Label, Collection<ConfiguredTargetAndData>> labelEntry :
          targetsByLabel.asMap().entrySet()) {
        Label label = labelEntry.getKey();
        Collection<ConfiguredTargetAndData> targets = labelEntry.getValue();

        ConfigurationTransition noOrNullTransition =
            getTransitionIfNoOrNull(configuration, targets);
        if (noOrNullTransition != null) {
          resolved.add(
              ResolvedTransition.create(
                  label,
                  /* buildOptions= */ ImmutableList.of(),
                  dependencyName,
                  noOrNullTransition.getName()));
          continue;
        }

        // The rule transition does not vary across a split so using the first target is sufficient.
        ConfigurationTransition ruleTransition =
            getRuleTransition(targets.iterator().next().getConfiguredTarget());

        var toOptions =
            targets.stream().map(t -> t.getConfiguration().getOptions()).collect(toImmutableList());

        resolved.add(
            ResolvedTransition.create(
                label,
                toOptions,
                dependencyName,
                getTransitionName(attributeTransitions.get(label), ruleTransition)));
      }
    }
    return resolved.build();
  }

  static class EvaluateException extends Exception {
    private EvaluateException(String message) {
      super(message);
    }
  }

  @Nullable
  private static ConfigurationTransition getTransitionIfNoOrNull(
      BuildConfigurationValue fromConfiguration, Collection<ConfiguredTargetAndData> targets) {
    ConfiguredTargetAndData first = targets.iterator().next();
    if (targets.size() == 1 && Objects.equals(fromConfiguration, first.getConfiguration())) {
      return NoTransition.INSTANCE;
    }
    // If any target has a null configuration, they all do, so it's sufficient to check the first.
    if (first.getConfiguration() == null) {
      return NullTransition.INSTANCE;
    }
    return null;
  }

  private static String getDependencyName(DependencyKind kind) {
    if (DependencyKind.isToolchain(kind)) {
      ToolchainDependencyKind tdk = (ToolchainDependencyKind) kind;
      if (tdk.isDefaultExecGroup()) {
        return "[toolchain dependency]";
      }
      return String.format("[toolchain dependency: %s]", tdk.getExecGroupName());
    }
    return kind.getAttribute().getName();
  }

  @Nullable
  private ConfigurationTransition getRuleTransition(ConfiguredTarget configuredTarget) {
    if (configuredTarget instanceof RuleConfiguredTarget) {
      return computeTransition(
          accessor.getTarget(configuredTarget).getAssociatedRule(),
          ((ConfiguredRuleClassProvider) ruleClassProvider).getTrimmingTransitionFactory());
    }
    return null;
  }

  private static String getTransitionName(
      @Nullable ConfigurationTransition attributeTransition,
      @Nullable ConfigurationTransition ruleTransition) {
    ConfigurationTransition transition = NoTransition.INSTANCE;
    if (attributeTransition != null) {
      transition = ComposingTransition.of(transition, attributeTransition);
    }
    if (ruleTransition != null) {
      transition = ComposingTransition.of(transition, ruleTransition);
    }
    return transition.getName();
  }
}
