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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.DependencyKey;
import com.google.devtools.build.lib.analysis.DependencyKind;
import com.google.devtools.build.lib.analysis.DependencyKind.NonAttributeDependencyKind;
import com.google.devtools.build.lib.analysis.DependencyKind.ToolchainDependencyKind;
import com.google.devtools.build.lib.analysis.DependencyResolver;
import com.google.devtools.build.lib.analysis.InconsistentAspectOrderException;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.ToolchainCollection;
import com.google.devtools.build.lib.analysis.ToolchainContext;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NullTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionUtil;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.RuleTransitionData;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import java.util.Map;
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
  private final DependencyResolver dependencyResolver;
  private final ConfiguredTargetAccessor accessor;
  private final CqueryThreadsafeCallback cqueryThreadsafeCallback;
  @Nullable private final TransitionFactory<RuleTransitionData> trimmingTransitionFactory;

  public CqueryTransitionResolver(
      ExtendedEventHandler eventHandler,
      DependencyResolver dependencyResolver,
      ConfiguredTargetAccessor accessor,
      CqueryThreadsafeCallback cqueryThreadsafeCallback,
      @Nullable TransitionFactory<RuleTransitionData> trimmingTransitionFactory) {
    this.eventHandler = eventHandler;
    this.dependencyResolver = dependencyResolver;
    this.accessor = accessor;
    this.cqueryThreadsafeCallback = cqueryThreadsafeCallback;
    this.trimmingTransitionFactory = trimmingTransitionFactory;
  }

  /**
   * Return the set of dependencies of a ConfiguredTarget, including information about the
   * configuration transitions applied to the dependencies.
   *
   * @see ResolvedTransition for more details.
   * @param keyedConfiguredTarget the configured target whose dependencies are being looked up.
   */
  public ImmutableSet<ResolvedTransition> dependencies(ConfiguredTarget keyedConfiguredTarget)
      throws DependencyResolver.Failure, InconsistentAspectOrderException, InterruptedException {
    ImmutableSet.Builder<ResolvedTransition> resolved = new ImmutableSet.Builder<>();

    if (!(keyedConfiguredTarget instanceof RuleConfiguredTarget)) {
      return resolved.build();
    }

    Target target = accessor.getTarget(keyedConfiguredTarget);
    BuildConfigurationValue config =
        cqueryThreadsafeCallback.getConfiguration(keyedConfiguredTarget.getConfigurationKey());

    ImmutableMap<Label, ConfigMatchingProvider> configConditions =
        keyedConfiguredTarget.getConfigConditions();

    // Get a ToolchainContext to use for dependency resolution.
    ToolchainCollection<ToolchainContext> toolchainContexts =
        accessor.getToolchainContexts(target, config);
    // We don't actually use fromOptions in our implementation of
    // DependencyResolver but passing to avoid passing a null and since we have the information
    // anyway.
    OrderedSetMultimap<DependencyKind, DependencyKey> deps =
        dependencyResolver.dependentNodeMap(
            new TargetAndConfiguration(target, config),
            /*aspect=*/ null,
            configConditions,
            toolchainContexts,
            trimmingTransitionFactory);
    for (Map.Entry<DependencyKind, DependencyKey> attributeAndDep : deps.entries()) {
      DependencyKey dep = attributeAndDep.getValue();

      if (attributeAndDep.getKey() instanceof NonAttributeDependencyKind) {
        // No attribute edge to report.
        continue;
      }

      String dependencyName;
      if (DependencyKind.isToolchain(attributeAndDep.getKey())) {
        ToolchainDependencyKind tdk = (ToolchainDependencyKind) attributeAndDep.getKey();
        if (tdk.isDefaultExecGroup()) {
          dependencyName = "[toolchain dependency]";
        } else {
          dependencyName = String.format("[toolchain dependency: %s]", tdk.getExecGroupName());
        }
      } else {
        dependencyName = attributeAndDep.getKey().getAttribute().getName();
      }

      if (attributeAndDep.getValue().getTransition() == NoTransition.INSTANCE
          || attributeAndDep.getValue().getTransition() == NullTransition.INSTANCE) {
        resolved.add(
            ResolvedTransition.create(
                dep.getLabel(),
                ImmutableList.of(),
                dependencyName,
                attributeAndDep.getValue().getTransition().getName()));
        continue;
      }
      BuildOptions fromOptions = config.getOptions();
      // TODO(bazel-team): support transitions on Starlark-defined build flags. These require
      // Skyframe loading to get flag default values. See ConfigurationResolver.applyTransition
      // for an example of the required logic.
      ImmutableSet<BuildOptions> toOptions =
          ImmutableSet.copyOf(
              dep.getTransition()
                  .apply(TransitionUtil.restrict(dep.getTransition(), fromOptions), eventHandler)
                  .values());
      resolved.add(
          ResolvedTransition.create(
              dep.getLabel(), toOptions, dependencyName, dep.getTransition().getName()));
    }
    return resolved.build();
  }
}
