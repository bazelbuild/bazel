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
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.ClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.Arrays;
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
 *
 * <p>For more information about aspects, see
 * {@link com.google.devtools.build.lib.packages.AspectClass}.
 *
 * @see com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory
 * @see com.google.devtools.build.lib.packages.AspectClass
 */
@Immutable
public final class ConfiguredAspect implements Iterable<TransitiveInfoProvider> {
  private final TransitiveInfoProviderMap providers;
  private final AspectDescriptor descriptor;

  private ConfiguredAspect(AspectDescriptor descriptor, TransitiveInfoProviderMap providers) {
    this.descriptor = descriptor;
    this.providers = providers;
  }

  /**
   * Returns the aspect name.
   */
  public String getName() {
    return descriptor.getAspectClass().getName();
  }

  /**
   *  The aspect descriptor originating this ConfiguredAspect.
   */
  public AspectDescriptor getDescriptor() {
    return descriptor;
  }

  /** Returns the providers created by the aspect. */
  public TransitiveInfoProviderMap getProviders() {
    return providers;
  }

  @Nullable
  @VisibleForTesting
  public <P extends TransitiveInfoProvider> P getProvider(Class<P> providerClass) {
    return providers.getProvider(providerClass);
  }

  @Override
  public UnmodifiableIterator<TransitiveInfoProvider> iterator() {
    return providers.values().iterator();
  }

  public static ConfiguredAspect forAlias(ConfiguredAspect real) {
    return new ConfiguredAspect(real.descriptor, real.getProviders());
  }

  public static ConfiguredAspect forNonapplicableTarget(AspectDescriptor descriptor) {
    return new ConfiguredAspect(descriptor, TransitiveInfoProviderMap.of());
  }

  /**
   * Builder for {@link ConfiguredAspect}.
   */
  public static class Builder {
    private final TransitiveInfoProviderMap.Builder providers = TransitiveInfoProviderMap.builder();
    private final Map<String, NestedSetBuilder<Artifact>> outputGroupBuilders = new TreeMap<>();
    private final ImmutableMap.Builder<String, Object> skylarkProviderBuilder =
        ImmutableMap.builder();
    private final RuleContext ruleContext;
    private final AspectDescriptor descriptor;

    public Builder(
        AspectClass aspectClass,
        AspectParameters parameters,
        RuleContext context) {
      this(new AspectDescriptor(aspectClass, parameters), context);
    }

    public Builder(AspectDescriptor descriptor, RuleContext ruleContext) {
      this.descriptor = descriptor;
      this.ruleContext = ruleContext;
    }

    public <T extends TransitiveInfoProvider> Builder addProvider(
        Class<? extends T> providerClass, T provider) {
      Preconditions.checkNotNull(provider);
      checkProviderClass(providerClass);
      providers.put(providerClass, provider);
      return this;
    }

    /** Adds a provider to the aspect. */
    public Builder addProvider(TransitiveInfoProvider provider) {
      Preconditions.checkNotNull(provider);
      addProvider(TransitiveInfoProviderMap.getEffectiveProviderClass(provider), provider);
      return this;
    }

    private void checkProviderClass(Class<? extends TransitiveInfoProvider> providerClass) {
      Preconditions.checkNotNull(providerClass);
      Preconditions.checkArgument(
          !SkylarkProviders.class.equals(providerClass),
          "Do not provide SkylarkProviders directly");
    }

    /** Adds providers to the aspect. */
    public Builder addProviders(TransitiveInfoProviderMap providers) {
      for (Map.Entry<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> entry :
          providers.entrySet()) {
        addProvider(entry.getKey(), entry.getKey().cast(entry.getValue()));
      }
      return this;
    }

    /** Adds providers to the aspect. */
    public Builder addProviders(TransitiveInfoProvider... providers) {
      return addProviders(Arrays.asList(providers));
    }

    /** Adds providers to the aspect. */
    public Builder addProviders(Iterable<TransitiveInfoProvider> providers) {
      for (TransitiveInfoProvider provider : providers) {
        addProvider(provider);
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

    public Builder addSkylarkTransitiveInfo(String name, Object value) {
      SkylarkProviderValidationUtil.checkSkylarkObjectSafe(value);
      skylarkProviderBuilder.put(name, value);
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

        if (providers.contains(OutputGroupProvider.class)) {
          throw new IllegalStateException(
              "OutputGroupProvider was provided explicitly; do not use addOutputGroup");
        }
        addProvider(new OutputGroupProvider(outputGroups.build()));
      }

      ImmutableMap<String, Object> skylarkProvidersMap = skylarkProviderBuilder.build();
      if (!skylarkProvidersMap.isEmpty()) {
        providers.add(
            new SkylarkProviders(skylarkProvidersMap,
                ImmutableMap.<ClassObjectConstructor.Key, SkylarkClassObject>of()));
      }

      addProvider(
          createExtraActionProvider(
              ImmutableSet.<ActionAnalysisMetadata>of() /* actionsWithoutExtraAction */,
              ruleContext));

      return new ConfiguredAspect(descriptor, providers.build());
    }
  }
}
