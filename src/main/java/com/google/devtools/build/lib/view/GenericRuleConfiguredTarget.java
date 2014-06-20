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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.view.RunfilesCollector.State;
import com.google.devtools.build.lib.view.config.RunUnder;

import java.util.Map;

/**
 * A generic implementation of RuleConfiguredTarget. Do not use directly. Use {@link
 * GenericRuleConfiguredTargetBuilder} instead.
 */
public final class GenericRuleConfiguredTarget extends RuleConfiguredTarget {

  private final Map<Class<? extends TransitiveInfoProvider>, Object> providers;
  private final RunfilesSupport runfilesSupport;
  private final Artifact executable;
  private final ImmutableList<Artifact> mandatoryStampFiles;

  GenericRuleConfiguredTarget(RuleContext ruleContext, RunfilesSupport runfilesSupport,
      Artifact executable,
      ImmutableList<Artifact> mandatoryStampFiles,
      TransitiveInfo... infos) {
    super(ruleContext);
    // We don't use ImmutableMap.Builder here to allow augmenting the initial list of 'default'
    // providers by passing them in.
    Map<Class<? extends TransitiveInfoProvider>, Object> providerBuilder = Maps.newHashMap();
    // Eventually, we want to remove these interfaces from RuleConfiguredTarget.
    providerBuilder.put(VisibilityProvider.class, this);
    providerBuilder.put(LicensesProvider.class, this);
    providerBuilder.put(FilesToRunProvider.class, this);
    providerBuilder.put(ExtraActionArtifactsProvider.class, this);
    for (TransitiveInfo info : infos) {
      providerBuilder.put(info.key, info.value);
    }
    Preconditions.checkState(providerBuilder.containsKey(RunfilesProvider.class));
    Preconditions.checkState(providerBuilder.containsKey(FileProvider.class));

    this.providers = ImmutableMap.copyOf(providerBuilder);
    // Ensure that that executable is the same as in runfilesSupport if both defined
    Preconditions.checkState(executable == null || runfilesSupport == null
        || runfilesSupport.getExecutable() == null
        || executable.equals(runfilesSupport.getExecutable()));
    this.runfilesSupport = runfilesSupport;
    this.executable = executable;
    this.mandatoryStampFiles = mandatoryStampFiles;

    // If this rule is the run_under target, then check that we have an executable; note that
    // run_under is only set in the target configuration, and the target must also be analyzed for
    // the target configuration.
    RunUnder runUnder = getConfiguration().getRunUnder();
    if (runUnder != null && getLabel().equals(runUnder.getLabel())) {
      if (getExecutable() == null) {
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

  @Override
  public Iterable<Class<? extends TransitiveInfoProvider>> getImplementedProviders() {
    return providers.keySet();
  }

  @Override
  public <P extends TransitiveInfoProvider> P getProvider(Class<P> provider) {
    AnalysisUtils.checkProvider(provider);
    return provider.cast(providers.get(provider));
  }

  @Override
  public Runfiles getTransitiveRunfiles(State state) {
    return getProvider(RunfilesProvider.class).getTransitiveRunfiles(state);
  }

  @Override
  public NestedSet<Artifact> getFilesToBuild() {
    return getProvider(FileProvider.class).getFilesToBuild();
  }

  @Override
  public Artifact getExecutable() {
    if (executable != null) {
      return executable;
    } else {
      return runfilesSupport != null ? runfilesSupport.getExecutable() : null;
    }
  }

  @Override
  public RunfilesSupport getRunfilesSupport() {
    return runfilesSupport;
  }

  @Override
  public ImmutableList<Artifact> getMandatoryStampFiles() {
    return mandatoryStampFiles;
  }

  static final class TransitiveInfo {
    final Class<? extends TransitiveInfoProvider> key;
    final TransitiveInfoProvider value;

    public TransitiveInfo(Class<? extends TransitiveInfoProvider> key,
        TransitiveInfoProvider value) {
      Preconditions.checkState(key.isAssignableFrom(value.getClass()));
      this.key = key;
      this.value = value;
    }
  }
}
