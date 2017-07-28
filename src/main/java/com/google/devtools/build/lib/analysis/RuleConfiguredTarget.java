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
package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.RunUnder;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.ClassObjectConstructor;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.rules.SkylarkApiProvider;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * A {@link ConfiguredTarget} that is produced by a rule.
 *
 * <p>Created by {@link RuleConfiguredTargetBuilder}. There is an instance of this class for every
 * analyzed rule. For more information about how analysis works, see {@link
 * com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory}.
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

  private final TransitiveInfoProviderMap providers;
  private final ImmutableMap<Label, ConfigMatchingProvider> configConditions;

  RuleConfiguredTarget(RuleContext ruleContext, TransitiveInfoProviderMap providers) {
    super(ruleContext);
    // We don't use ImmutableMap.Builder here to allow augmenting the initial list of 'default'
    // providers by passing them in.
    TransitiveInfoProviderMapBuilder providerBuilder =
        new TransitiveInfoProviderMapBuilder().addAll(providers);
    Preconditions.checkState(providerBuilder.contains(RunfilesProvider.class));
    Preconditions.checkState(providerBuilder.contains(FileProvider.class));
    Preconditions.checkState(providerBuilder.contains(FilesToRunProvider.class));

    // Initialize every SkylarkApiProvider
    for (int i = 0; i < providers.getProviderCount(); i++) {
      Object obj = providers.getProviderInstanceAt(i);
      if (obj instanceof SkylarkApiProvider) {
        ((SkylarkApiProvider) obj).init(this);
      }
    }


    this.providers = providerBuilder.build();
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
  ImmutableMap<Label, ConfigMatchingProvider> getConfigConditions() {
    return configConditions;
  }

  @Nullable
  @Override
  public <P extends TransitiveInfoProvider> P getProvider(Class<P> providerClass) {
    // TODO(bazel-team): Should aspects be allowed to override providers on the configured target
    // class?
    return providers.getProvider(providerClass);
  }

  @Override
  public final Rule getTarget() {
    return (Rule) super.getTarget();
  }

  @Override
  public String errorMessage(String name) {
    return Printer.format("%r (rule '%s') doesn't have provider '%s'",
        this, getTarget().getRuleClass(), name);
  }

  @Override
  protected void addExtraSkylarkKeys(Consumer<String> result) {
    for (int i = 0; i < providers.getProviderCount(); i++) {
      Object classAt = providers.getProviderKeyAt(i);
      if (classAt instanceof String) {
        result.accept((String) classAt);
      }
    }
  }

  @Override
  protected SkylarkClassObject rawGetSkylarkProvider(ClassObjectConstructor.Key providerKey) {
    return providers.getProvider(providerKey);
  }

  @Override
  protected Object rawGetSkylarkProvider(String providerKey) {
    return providers.getProvider(providerKey);
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    printer.append("<target " + getLabel() + ">");
  }
}
