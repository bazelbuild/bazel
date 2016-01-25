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

import static com.google.devtools.build.lib.analysis.ExtraActionUtils.createExtraActionProvider;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.UnmodifiableIterator;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TreeMap;

import javax.annotation.Nullable;

/**
 * Extra information about a configured target computed on request of a dependent.
 *
 * <p>Analogous to {@link ConfiguredTarget}: contains a bunch of transitive info providers, which
 * are merged with the providers of the associated configured target before they are passed to
 * the configured target factories that depend on the configured target to which this aspect is
 * added.
 *
 * <p>Aspects are created alongside configured targets on request from dependents.
 */
@Immutable
public final class ConfiguredAspect implements Iterable<TransitiveInfoProvider> {
  private final String name;
  private final ImmutableMap<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider>
      providers;

  private ConfiguredAspect(
      String name,
      ImmutableMap<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> providers) {
    this.name = name;
    this.providers = providers;
  }

  /**
   * Returns the aspect name.
   */
  public String getName() {
    return name;
  }

  /**
   * Returns the providers created by the aspect.
   */
  public ImmutableMap<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider>
      getProviders() {
    return providers;
  }


  @Nullable
  @VisibleForTesting
  public <P extends TransitiveInfoProvider> P getProvider(Class<P> providerClass) {
    AnalysisUtils.checkProvider(providerClass);

    return providerClass.cast(providers.get(providerClass));
  }

  @Override
  public UnmodifiableIterator<TransitiveInfoProvider> iterator() {
    return providers.values().iterator();
  }

  /**
   * Builder for {@link ConfiguredAspect}.
   */
  public static class Builder {
    private final Map<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider>
        providers = new LinkedHashMap<>();
    private final Map<String, NestedSetBuilder<Artifact>> outputGroupBuilders = new TreeMap<>();
    private final ImmutableMap.Builder<String, Object> skylarkProviderBuilder =
        ImmutableMap.builder();
    private final String name;
    private final RuleContext ruleContext;

    public Builder(String name, RuleContext ruleContext) {
      this.name = name;
      this.ruleContext = ruleContext;
    }

    /**
     * Adds a provider to the aspect.
     */
    public Builder addProvider(
        Class<? extends TransitiveInfoProvider> key, TransitiveInfoProvider value) {
      Preconditions.checkNotNull(key);
      Preconditions.checkNotNull(value);
      AnalysisUtils.checkProvider(key);
      Preconditions.checkState(!providers.containsKey(key));
      Preconditions.checkArgument(!SkylarkProviders.class.equals(key),
          "Do not provide SkylarkProviders directly");
      providers.put(key, value);
      return this;
    }

    public Builder addProviders(
        Map<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> providers) {
      for (Map.Entry<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> provider :
          providers.entrySet()) {
        addProvider(provider.getKey(), provider.getValue());
      }
      return this;
    }

    /**
     * Adds a set of files to an output group.
     */
    public Builder addOutputGroup(String name, NestedSet<Artifact> artifacts) {
      NestedSetBuilder<Artifact> nestedSetBuilder = outputGroupBuilders.get(name);
      if (nestedSetBuilder == null) {
        nestedSetBuilder = NestedSetBuilder.<Artifact>stableOrder();
        outputGroupBuilders.put(name, nestedSetBuilder);
      }
      nestedSetBuilder.addTransitive(artifacts);
      return this;
    }

    public Builder addSkylarkTransitiveInfo(String name, Object value, Location loc)
        throws EvalException {
      SkylarkProviderValidationUtil.validateAndThrowEvalException(name, value, loc);
      skylarkProviderBuilder.put(name, value);
      return this;
    }

    public ConfiguredAspect build() {
      if (!outputGroupBuilders.isEmpty()) {
        ImmutableMap.Builder<String, NestedSet<Artifact>> outputGroups = ImmutableMap.builder();
        for (Map.Entry<String, NestedSetBuilder<Artifact>> entry : outputGroupBuilders.entrySet()) {
          outputGroups.put(entry.getKey(), entry.getValue().build());
        }

        if (providers.containsKey(OutputGroupProvider.class)) {
          throw new IllegalStateException(
              "OutputGroupProvider was provided explicitly; do not use addOutputGroup");
        }
        addProvider(OutputGroupProvider.class, new OutputGroupProvider(outputGroups.build()));
      }

      ImmutableMap<String, Object> skylarkProvidersMap = skylarkProviderBuilder.build();
      if (!skylarkProvidersMap.isEmpty()) {
        providers.put(SkylarkProviders.class, new SkylarkProviders(skylarkProvidersMap));
      }

      addProvider(
          ExtraActionArtifactsProvider.class,
          createExtraActionProvider(
              ImmutableSet.<Action>of() /* actionsWithoutExtraAction */, ruleContext));

      return new ConfiguredAspect(name, ImmutableMap.copyOf(providers));
    }
  }
}
