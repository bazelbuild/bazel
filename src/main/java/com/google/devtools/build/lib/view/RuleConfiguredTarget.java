// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.view;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.UnmodifiableIterator;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.view.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.view.config.RunUnder;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A generic implementation of RuleConfiguredTarget. Do not use directly. Use {@link
 * RuleConfiguredTargetBuilder} instead.
 */
public final class RuleConfiguredTarget extends AbstractConfiguredTarget {

  /**
   * The configuration transition for an attribute through which a prerequisite
   * is requested.
   */
  public enum Mode {
    TARGET,
    HOST,
    DATA,
    SPLIT,
    DONT_CHECK
  }

  private final ImmutableMap<Class<? extends TransitiveInfoProvider>, Object> providers;
  private final ImmutableList<Artifact> mandatoryStampFiles;
  private final Set<ConfigMatchingProvider> configConditions;

  RuleConfiguredTarget(RuleContext ruleContext,
      ImmutableList<Artifact> mandatoryStampFiles,
      ImmutableMap<String, Object> skylarkProviders,
      Map<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> providers) {
    super(ruleContext);
    // We don't use ImmutableMap.Builder here to allow augmenting the initial list of 'default'
    // providers by passing them in.
    Map<Class<? extends TransitiveInfoProvider>, Object> providerBuilder = new LinkedHashMap<>();
    providerBuilder.putAll(providers);
    Preconditions.checkState(providerBuilder.containsKey(RunfilesProvider.class));
    Preconditions.checkState(providerBuilder.containsKey(FileProvider.class));
    Preconditions.checkState(providerBuilder.containsKey(FilesToRunProvider.class));

    providerBuilder.put(SkylarkProviders.class, new SkylarkProviders(skylarkProviders));

    this.providers = ImmutableMap.copyOf(providerBuilder);
    this.mandatoryStampFiles = mandatoryStampFiles;
    this.configConditions = ruleContext.getConfigConditions();

    // If this rule is the run_under target, then check that we have an executable; note that
    // run_under is only set in the target configuration, and the target must also be analyzed for
    // the target configuration.
    RunUnder runUnder = getConfiguration().getRunUnder();
    if (runUnder != null && getLabel().equals(runUnder.getLabel())) {
      if (getProvider(FilesToRunProvider.class).getExecutable() == null) {
        ruleContext.ruleError("run_under target " + runUnder.getLabel() + " is not executable");
      }
    }

    // Make sure that all declared output files are also created as artifacts. The
    // CachingAnalysisEnvironment makes sure that they all have generating actions.
    if (!ruleContext.hasErrors()) {
      for (OutputFile out : ruleContext.getRule().getOutputFiles()) {
        ruleContext.createOutputArtifact(out);
      }
    }
  }

  /**
   * The configuration conditions that trigger this rule's configurable attributes.
   */
  Set<ConfigMatchingProvider> getConfigConditions() {
    return configConditions;
  }

  @Override
  public <P extends TransitiveInfoProvider> P getProvider(Class<P> provider) {
    AnalysisUtils.checkProvider(provider);
    return provider.cast(providers.get(provider));
  }

  /**
   * Returns a value provided by this target. Only meant to use from Skylark.
   */
  @Override
  public Object get(String providerKey) {
    return getProvider(SkylarkProviders.class).skylarkProviders.get(providerKey);
  }

  public ImmutableList<Artifact> getMandatoryStampFiles() {
    return mandatoryStampFiles;
  }

  @Override
  public final Rule getTarget() {
    return (Rule) super.getTarget();
  }

  /**
   * A helper class for transitive infos provided by Skylark rule implementations.
   */
  @Immutable
  private static final class SkylarkProviders implements TransitiveInfoProvider {
    private final ImmutableMap<String, Object> skylarkProviders;

    private SkylarkProviders(ImmutableMap<String, Object> skylarkProviders) {
      this.skylarkProviders = skylarkProviders;
    }
  }

  @Override
  public UnmodifiableIterator<TransitiveInfoProvider> iterator() {
    List<TransitiveInfoProvider> tip = Lists.newArrayList();
    for (Map.Entry<Class<? extends TransitiveInfoProvider>, Object> entry : providers.entrySet()) {
      tip.add(entry.getKey().cast(entry.getValue()));
    }
    return ImmutableList.copyOf(tip).iterator();
  }

  @Override
  public String errorMessage(String name) {
    return String.format("target (rule class of '%s') doesn't have provider '%s'.",
        getTarget().getRuleClass(), name);
  }

  @Override
  public ImmutableCollection<String> getKeys() {
    return ImmutableList.<String>builder().addAll(super.getKeys())
        .addAll(getProvider(SkylarkProviders.class).skylarkProviders.keySet()).build();
  }
}
