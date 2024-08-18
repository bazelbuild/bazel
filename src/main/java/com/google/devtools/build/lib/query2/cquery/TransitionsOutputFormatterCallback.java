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

import static java.util.stream.Collectors.joining;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.OptionsDiff;
import com.google.devtools.build.lib.analysis.config.StarlarkTransitionCache;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.analysis.constraints.IncompatibleTargetChecker;
import com.google.devtools.build.lib.analysis.producers.BuildConfigurationKeyCache;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.RuleTransitionData;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.common.CqueryNode;
import com.google.devtools.build.lib.query2.cquery.CqueryTransitionResolver.EvaluateException;
import com.google.devtools.build.lib.query2.cquery.CqueryTransitionResolver.ResolvedTransition;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import java.io.OutputStream;
import java.util.HashMap;

/**
 * Output formatter that prints {@link ConfigurationTransition} information for rule configured
 * targets in the results of a cquery call.
 */
class TransitionsOutputFormatterCallback extends CqueryThreadsafeCallback {

  private final HashMap<Label, Target> partialResultMap;
  private final RuleClassProvider ruleClassProvider;
  private final LabelPrinter labelPrinter;
  private final StarlarkTransitionCache transitionCache;
  private final BuildConfigurationKeyCache buildConfigurationKeyCache;

  @Override
  public String getName() {
    return "transitions";
  }

  /**
   * @param accessor provider of query result configured targets.
   */
  TransitionsOutputFormatterCallback(
      ExtendedEventHandler eventHandler,
      CqueryOptions options,
      OutputStream out,
      SkyframeExecutor skyframeExecutor,
      TargetAccessor<CqueryNode> accessor,
      RuleClassProvider ruleClassProvider,
      LabelPrinter labelPrinter) {
    super(eventHandler, options, out, skyframeExecutor, accessor, /* uniquifyResults= */ false);
    this.ruleClassProvider = ruleClassProvider;
    this.partialResultMap = Maps.newHashMap();
    this.labelPrinter = labelPrinter;
    this.transitionCache = skyframeExecutor.getSkyframeBuildView().getStarlarkTransitionCache();
    this.buildConfigurationKeyCache =
        skyframeExecutor.getSkyframeBuildView().getBuildConfigurationKeyCache();
  }

  @Override
  public void processOutput(Iterable<CqueryNode> partialResult) throws InterruptedException {
    CqueryOptions.Transitions verbosity = options.transitions;
    if (verbosity.equals(CqueryOptions.Transitions.NONE)) {
      eventHandler.handle(
          Event.error(
              "Instead of using --output=transitions, set the --transitions"
                  + " flag explicitly to 'lite' or 'full'"));
      return;
    }
    partialResult.forEach(
        kct -> partialResultMap.put(kct.getOriginalLabel(), accessor.getTarget(kct)));
    for (CqueryNode keyedConfiguredTarget : partialResult) {
      Target target = partialResultMap.get(keyedConfiguredTarget.getOriginalLabel());
      BuildConfigurationValue config =
          getConfiguration(keyedConfiguredTarget.getConfigurationKey());
      addResult(
          getRuleClassTransition(keyedConfiguredTarget, target)
              + String.format(
                  "%s (%s)",
                  labelPrinter.toString(keyedConfiguredTarget.getOriginalLabel()),
                  shortId(config)));
      ImmutableSet<ResolvedTransition> dependencies;
      try {
        // We don't actually use fromOptions in our implementation of
        // DependencyResolver but passing to avoid passing a null and since we have the information
        // anyway.
        dependencies =
            new CqueryTransitionResolver(
                    eventHandler,
                    accessor,
                    this,
                    ruleClassProvider,
                    transitionCache,
                    buildConfigurationKeyCache)
                .dependencies(keyedConfiguredTarget);
      } catch (EvaluateException e) {
        eventHandler.handle(
            Event.error(
                String.format(
                    "Failed to evaluate %s: %s", keyedConfiguredTarget.getOriginalLabel(), e)));
        return;
      } catch (IncompatibleTargetChecker.IncompatibleTargetException e) {
        eventHandler.handle(
            Event.warn(
                String.format(
                    "Skipping dependencies of incompatible target %s",
                    keyedConfiguredTarget.getOriginalLabel())));
        return;
      }
      for (ResolvedTransition dep : dependencies) {
        addResult(
            "  "
                .concat(dep.attributeName())
                .concat("#")
                .concat(labelPrinter.toString(dep.label()))
                .concat("#")
                .concat(dep.transitionName())
                .concat(" -> ")
                .concat(dep.options().stream().map(BuildOptions::shortId).collect(joining(", "))));
        if (verbosity == CqueryOptions.Transitions.LITE) {
          continue;
        }
        OptionsDiff diff = new OptionsDiff();
        for (BuildOptions options : dep.options()) {
          diff = OptionsDiff.diff(diff, config.getOptions(), options);
        }
        diff.getPrettyPrintList().forEach(singleDiff -> addResult("    " + singleDiff));
      }
    }
  }

  private static String getRuleClassTransition(CqueryNode ct, Target target) {
    Rule rule = target.getAssociatedRule();
    if (rule == null) {
      return "";
    }

    TransitionFactory<RuleTransitionData> factory =
        rule.getRuleClassObject().getTransitionFactory();
    return factory
        .create(
            RuleTransitionData.create(
                target.getAssociatedRule(),
                /* configConditions= */ null,
                ct.getConfigurationKey().getOptionsChecksum()))
        .getName()
        .concat(" -> ");
  }
}
