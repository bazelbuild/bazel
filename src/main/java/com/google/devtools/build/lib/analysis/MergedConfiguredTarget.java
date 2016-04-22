// Copyright 2016 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A single dependency with its configured target and aspects merged together.
 *
 * <p>This is an ephemeral object created only for the analysis of a single configured target.
 * After that configured target is analyzed, this is thrown away.
 */
public class MergedConfiguredTarget extends AbstractConfiguredTarget {
  private final ConfiguredTarget base;
  private final ImmutableMap<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider>
      providers;

  private MergedConfiguredTarget(ConfiguredTarget base,
      ImmutableMap<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> providers) {
    super(base.getTarget(), base.getConfiguration());
    this.base = base;
    this.providers = providers;
  }

  /**
   * Returns a value provided by this target. Only meant to use from Skylark.
   */
  @Override
  public Object get(String providerKey) {
    return getProvider(SkylarkProviders.class).getValue(providerKey);
  }

  @Override
  public <P extends TransitiveInfoProvider> P getProvider(Class<P> providerClass) {
    AnalysisUtils.checkProvider(providerClass);

    Object provider = providers.get(providerClass);
    if (provider == null) {
      provider = base.getProvider(providerClass);
    }

    return providerClass.cast(provider);
  }

  /**
   * Creates an instance based on a configured target and a set of aspects.
   */
  public static ConfiguredTarget of(ConfiguredTarget base,
      Iterable<ConfiguredAspect> aspects) {
    if (Iterables.isEmpty(aspects)) {
      // If there are no aspects, don't bother with creating a proxy object
      return base;
    }

    Set<Class<? extends TransitiveInfoProvider>> providers = new HashSet<>();

    ImmutableSet<Class<? extends TransitiveInfoProvider>> baseProviders =
        ImmutableSet.copyOf(providers);

    // Merge output group providers.
    OutputGroupProvider mergedOutputGroupProvider =
        OutputGroupProvider.merge(getAllProviders(base, aspects, OutputGroupProvider.class));

    // Merge Skylark providers.
    SkylarkProviders mergedSkylarkProviders =
        SkylarkProviders.merge(getAllProviders(base, aspects, SkylarkProviders.class));

    // Merge extra-actions provider.
    ExtraActionArtifactsProvider mergedExtraActionProviders = ExtraActionArtifactsProvider.merge(
        getAllProviders(base, aspects, ExtraActionArtifactsProvider.class));

    Map<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> aspectProviders =
        new LinkedHashMap<>();
    if (mergedOutputGroupProvider != null) {
      aspectProviders.put(OutputGroupProvider.class, mergedOutputGroupProvider);
    }
    if (mergedSkylarkProviders != null) {
      aspectProviders.put(SkylarkProviders.class, mergedSkylarkProviders);
    }
    if (mergedExtraActionProviders != null) {
      aspectProviders.put(ExtraActionArtifactsProvider.class, mergedExtraActionProviders);
    }

    for (ConfiguredAspect aspect : aspects) {
      for (Map.Entry<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> entry :
          aspect.getProviders().entrySet()) {
        if (OutputGroupProvider.class.equals(entry.getKey())
            || SkylarkProviders.class.equals(entry.getKey())
            || ExtraActionArtifactsProvider.class.equals(entry.getKey())) {
          continue;
        }

        if (base.getProvider(entry.getKey()) != null
            || aspectProviders.containsKey(entry.getKey())) {
          throw new IllegalStateException("Provider " + entry.getKey() + " provided twice");
        }

        aspectProviders.put(entry.getKey(), entry.getValue());
      }
    }
    return new MergedConfiguredTarget(base, ImmutableMap.copyOf(aspectProviders));
  }

  private static <T extends TransitiveInfoProvider> List<T> getAllProviders(
      ConfiguredTarget base, Iterable<ConfiguredAspect> aspects, Class<T> providerClass) {
    T baseProvider = base.getProvider(providerClass);
    List<T> providers = new ArrayList<>();
    if (baseProvider != null) {
      providers.add(baseProvider);
    }

    for (ConfiguredAspect configuredAspect : aspects) {
      final T aspectProvider = configuredAspect.getProvider(providerClass);
      if (aspectProvider == null) {
        continue;
      }
      providers.add(aspectProvider);
    }
    return providers;
  }
}
