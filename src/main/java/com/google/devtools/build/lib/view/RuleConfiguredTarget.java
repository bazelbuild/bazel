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
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin;
import com.google.devtools.build.lib.syntax.SkylarkCallable;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.view.config.RunUnder;

import java.util.Map;

//TODO(bazel-team): Remove ActionOwner from here; use ruleContext.getActionOwner() instead.
/**
 * A generic implementation of RuleConfiguredTarget. Do not use directly. Use {@link
 * RuleConfiguredTargetBuilder} instead.
 */
@SkylarkBuiltin(name = "", doc = "")
public final class RuleConfiguredTarget extends AbstractConfiguredTarget implements
    ActionOwner {

  /**
   * The configuration transition for an attribute through which a prerequisite
   * is requested.
   */
  public enum Mode {
    TARGET,
    HOST,
    DATA,
    DONT_CHECK
  }

  private final ImmutableMap<Class<? extends TransitiveInfoProvider>, Object> providers;
  private final ImmutableList<Artifact> mandatoryStampFiles;

  RuleConfiguredTarget(RuleContext ruleContext,
      ImmutableList<Artifact> mandatoryStampFiles,
      ImmutableMap<String, Object> skylarkProviders,
      TransitiveInfo... infos) {
    super(ruleContext);
    // We don't use ImmutableMap.Builder here to allow augmenting the initial list of 'default'
    // providers by passing them in.
    Map<Class<? extends TransitiveInfoProvider>, Object> providerBuilder = Maps.newHashMap();
    // Eventually, we want to remove these interfaces from RuleConfiguredTarget.
    providerBuilder.put(VisibilityProvider.class, this);
    providerBuilder.put(LicensesProvider.class, this);
    providerBuilder.put(ExtraActionArtifactsProvider.class, this);
    for (TransitiveInfo info : infos) {
      providerBuilder.put(info.key, info.value);
    }
    Preconditions.checkState(providerBuilder.containsKey(RunfilesProvider.class));
    Preconditions.checkState(providerBuilder.containsKey(FileProvider.class));
    Preconditions.checkState(providerBuilder.containsKey(FilesToRunProvider.class));

    checkSkylarkProviders(skylarkProviders);
    providerBuilder.put(SkylarkProviders.class, new SkylarkProviders(skylarkProviders));

    this.providers = ImmutableMap.copyOf(providerBuilder);
    this.mandatoryStampFiles = mandatoryStampFiles;

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
   * This is an extra safety check. All objects passed in the providers have to be
   * immutable in Skylark.
   */
  // TODO(bazel-team): Skylark will support only immutable objects. We will still need this check
  // here but we don't have to worry about nice error messages to Skylark users. Also this method
  // is going to be extended as we migrate more and more Transitive Info Providers.
  private void checkSkylarkProviders(ImmutableMap<String, Object> providers) {
    for (Map.Entry<String, Object> entry : providers.entrySet()) {
      Object value = entry.getValue();
      if (value instanceof ImmutableList) {
        // TODO(bazel-team): it is not efficient to iterate through the list.
        // We will have to come up with a way to have generic type information here,
        // but first we need to enforce type safety in Skylark.
        for (Object nestedSetValue : ((Iterable<?>) value)) {
          Preconditions.checkArgument(isSimpleSkylarkObjectImmutable(nestedSetValue),
              String.format("Transitive Info Provider '%s' contains mutable objects",
                  entry.getKey()));
        }
      } else {
        Preconditions.checkArgument(
            // Java transitive Info Providers are still accessible from Skylark, e.g.
            // RunfilesProvider. Those are safe.
            value instanceof TransitiveInfoProvider
            || value instanceof SkylarkNestedSet,
            String.format("Transitive Info Provider '%s' is mutable", entry.getKey()));
      }
    }
  }

  public boolean isSimpleSkylarkObjectImmutable(Object object) {
    if (object instanceof String
        || object instanceof Integer
        || object instanceof Label
        || object instanceof Artifact) {
      return true;
    } else {
      return false;
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

  /**
   * Returns a value provided by this target. Only meant to use from Skylark.
   */
  @SkylarkCallable(
      doc = "Returns the value provided by this target associated with the provider_key.")
  @Override
  public Object get(String providerKey) {
    return getProvider(SkylarkProviders.class).get(providerKey);
  }

  public ImmutableList<Artifact> getMandatoryStampFiles() {
    return mandatoryStampFiles;
  }

  // ActionOwner functions

  @Override
  public final String getTargetKind() {
    return getTarget().getTargetKind();
  }

  @Override
  public final String getConfigurationName() {
    return getConfiguration().getShortName();
  }

  @Override
  public final String getConfigurationMnemonic() {
    return getConfiguration().getMnemonic();
  }

  @Override
  public final String getConfigurationShortCacheKey() {
    return getConfiguration().shortCacheKey();
  }

  @Override
  public final Location getLocation() {
    return getTarget().getLocation();
  }

  @Override
  public final String getAdditionalProgressInfo() {
    return getConfiguration().isHostConfiguration() ? "for host" : null;
  }

  @Override
  public final Rule getTarget() {
    return (Rule) super.getTarget();
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

  /**
   * A helper class for transitive infos provided by Skylark rule implementations.
   */
  @Immutable
  static final class SkylarkProviders implements TransitiveInfoProvider {
    private final ImmutableMap<String, Object> skylarkProviders;

    private SkylarkProviders(ImmutableMap<String, Object> skylarkProviders) {
      this.skylarkProviders = skylarkProviders;
    }

    Object get(String providerKey) {
      return skylarkProviders.get(providerKey);
    }
  }
}
