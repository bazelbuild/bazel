// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.Dependency;
import com.google.devtools.build.lib.analysis.DependencyKind;
import com.google.devtools.build.lib.analysis.DependencyResolver;
import com.google.devtools.build.lib.analysis.InconsistentAspectOrderException;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.ToolchainCollection;
import com.google.devtools.build.lib.analysis.ToolchainContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions.OptionsDiff;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NullTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import java.io.OutputStream;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * Output formatter that prints {@link ConfigurationTransition} information for rule configured
 * targets in the results of a cquery call.
 */
class TransitionsOutputFormatterCallback extends CqueryThreadsafeCallback {

  protected final BuildConfiguration hostConfiguration;

  private final HashMap<Label, Target> partialResultMap;
  @Nullable private final TransitionFactory<Rule> trimmingTransitionFactory;

  @Override
  public String getName() {
    return "transitions";
  }

  /**
   * @param accessor provider of query result configured targets.
   * @param hostConfiguration host configuration for this query.
   */
  TransitionsOutputFormatterCallback(
      ExtendedEventHandler eventHandler,
      CqueryOptions options,
      OutputStream out,
      SkyframeExecutor skyframeExecutor,
      TargetAccessor<ConfiguredTarget> accessor,
      BuildConfiguration hostConfiguration,
      @Nullable TransitionFactory<Rule> trimmingTransitionFactory) {
    super(eventHandler, options, out, skyframeExecutor, accessor);
    this.hostConfiguration = hostConfiguration;
    this.trimmingTransitionFactory = trimmingTransitionFactory;
    this.partialResultMap = Maps.newHashMap();
  }

  @Override
  public void processOutput(Iterable<ConfiguredTarget> partialResult) throws InterruptedException {
    CqueryOptions.Transitions verbosity = options.transitions;
    if (verbosity.equals(CqueryOptions.Transitions.NONE)) {
      eventHandler.handle(
          Event.error(
              "Instead of using --output=transitions, set the --transitions"
                  + " flag explicitly to 'lite' or 'full'"));
      return;
    }
    partialResult.forEach(
        ct -> partialResultMap.put(ct.getLabel(), accessor.getTargetFromConfiguredTarget(ct)));
    for (ConfiguredTarget configuredTarget : partialResult) {
      Target target = partialResultMap.get(configuredTarget.getLabel());
      BuildConfiguration config =
          skyframeExecutor.getConfiguration(
              eventHandler, configuredTarget.getConfigurationKey());
      addResult(
          getRuleClassTransition(configuredTarget, target)
              + configuredTarget.getLabel()
              + " ("
              + (config != null && config.isHostConfiguration() ? "HOST" : config)
              + ")");
      if (!(configuredTarget instanceof RuleConfiguredTarget)) {
        continue;
      }
      OrderedSetMultimap<DependencyKind, Dependency> deps;
      ImmutableMap<Label, ConfigMatchingProvider> configConditions =
          ((RuleConfiguredTarget) configuredTarget).getConfigConditions();

      // Get a ToolchainContext to use for dependency resolution.
      ToolchainCollection<ToolchainContext> toolchainContexts =
          accessor.getToolchainContexts(target, config);
      try {
        // We don't actually use fromOptions in our implementation of
        // DependencyResolver but passing to avoid passing a null and since we have the information
        // anyway.
        deps =
            new FormatterDependencyResolver(eventHandler)
                .dependentNodeMap(
                    new TargetAndConfiguration(target, config),
                    hostConfiguration,
                    /*aspect=*/ null,
                    configConditions,
                    toolchainContexts,
                    trimmingTransitionFactory);
      } catch (EvalException | InconsistentAspectOrderException e) {
        throw new InterruptedException(e.getMessage());
      }
      for (Map.Entry<DependencyKind, Dependency> attributeAndDep : deps.entries()) {
        // DependencyResolver should only ever return Dependency instances with transitions and not
        // with explicit configurations
        Preconditions.checkState(!attributeAndDep.getValue().hasExplicitConfiguration());
        if (attributeAndDep.getValue().getTransition() == NoTransition.INSTANCE
            || attributeAndDep.getValue().getTransition() == NullTransition.INSTANCE) {
          continue;
        }
        Dependency dep = attributeAndDep.getValue();
        BuildOptions fromOptions = config.getOptions();
        // TODO(bazel-team): support transitions on Starlark-defined build flags. These require
        // Skyframe loading to get flag default values. See ConfigurationResolver.applyTransition
        // for an example of the required logic.
        Collection<BuildOptions> toOptions =
            dep.getTransition().apply(fromOptions, eventHandler).values();
        String hostConfigurationChecksum = hostConfiguration.checksum();
        String dependencyName;
        if (attributeAndDep.getKey() == DependencyResolver.TOOLCHAIN_DEPENDENCY) {
          dependencyName = "[toolchain dependency]";
        } else {
          dependencyName = attributeAndDep.getKey().getAttribute().getName();
        }
        addResult(
            "  "
                .concat(dependencyName)
                .concat("#")
                .concat(dep.getLabel().toString())
                .concat("#")
                .concat(dep.getTransition().getName())
                .concat(" ( -> ")
                .concat(
                    toOptions.stream()
                        .map(
                            options -> {
                              String checksum = options.computeChecksum();
                              return checksum.equals(hostConfigurationChecksum) ? "HOST" : checksum;
                            })
                        .collect(Collectors.joining(", ")))
                .concat(")"));
        if (verbosity == CqueryOptions.Transitions.LITE) {
          continue;
        }
        OptionsDiff diff = new OptionsDiff();
        for (BuildOptions options : toOptions) {
          diff = BuildOptions.diff(diff, fromOptions, options);
        }
        diff.getPrettyPrintList().forEach(singleDiff -> addResult("    " + singleDiff));
      }
    }
  }

  private String getRuleClassTransition(ConfiguredTarget ct, Target target) {
    String output = "";
    if (ct instanceof RuleConfiguredTarget) {
      TransitionFactory<Rule> factory =
          target.getAssociatedRule().getRuleClassObject().getTransitionFactory();
      if (factory != null) {
        output =
            factory.create(target.getAssociatedRule()).getClass().getSimpleName().concat(" -> ");
      }
    }
    return output;
  }

  private class FormatterDependencyResolver extends DependencyResolver {
    private final ExtendedEventHandler eventHandler;

    private FormatterDependencyResolver(ExtendedEventHandler eventHandler) {
      this.eventHandler = eventHandler;
    }

    @Override
    protected void invalidPackageGroupReferenceHook(TargetAndConfiguration node, Label label) {
      eventHandler.handle(
          Event.error(
              TargetUtils.getLocationMaybe(node.getTarget()),
              String.format("label '%s' does not refer to a package group", label)));
    }

    @Override
    protected Map<Label, Target> getTargets(
        OrderedSetMultimap<DependencyKind, Label> labelMap,
        TargetAndConfiguration fromNode,
        NestedSetBuilder<Cause> rootCauses) {
      return labelMap.values().stream()
          .distinct()
          .filter(Objects::nonNull)
          .filter(partialResultMap::containsKey)
          .collect(Collectors.toMap(Function.identity(), partialResultMap::get));
    }
  }
}

