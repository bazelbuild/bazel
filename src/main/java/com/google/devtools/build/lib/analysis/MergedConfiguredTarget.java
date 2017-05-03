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

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A single dependency with its configured target and aspects merged together.
 *
 * <p>This is an ephemeral object created only for the analysis of a single configured target. After
 * that configured target is analyzed, this is thrown away.
 */
public final class MergedConfiguredTarget extends AbstractConfiguredTarget {
  private final ConfiguredTarget base;
  private final TransitiveInfoProviderMap providers;

  /**
   * This exception is thrown when configured targets and aspects
   * being merged provide duplicate things that they shouldn't
   * (output groups or providers).
   */
  public static final class DuplicateException extends Exception {
    public DuplicateException(String message) {
      super(message);
    }
  }

  private MergedConfiguredTarget(ConfiguredTarget base, TransitiveInfoProviderMap providers) {
    super(base.getTarget(), base.getConfiguration());
    this.base = base;
    this.providers = providers;
  }

  @Override
  public <P extends TransitiveInfoProvider> P getProvider(Class<P> providerClass) {
    AnalysisUtils.checkProvider(providerClass);

    P provider = providers.getProvider(providerClass);
    if (provider == null) {
      provider = base.getProvider(providerClass);
    }

    return provider;
  }

  @Override
  public ImmutableCollection<String> getKeys() {
    return ImmutableList.<String>builder()
        .addAll(super.getKeys())
        .addAll(getProvider(SkylarkProviders.class).getKeys())
        .build();
  }


  /** Creates an instance based on a configured target and a set of aspects. */
  public static ConfiguredTarget of(ConfiguredTarget base, Iterable<ConfiguredAspect> aspects)
      throws DuplicateException {
    if (Iterables.isEmpty(aspects)) {
      // If there are no aspects, don't bother with creating a proxy object
      return base;
    }

    // Merge output group providers.
    OutputGroupProvider mergedOutputGroupProvider =
        OutputGroupProvider.merge(getAllOutputGroupProviders(base, aspects));

    // Merge Skylark providers.
    SkylarkProviders mergedSkylarkProviders =
        SkylarkProviders.merge(getAllProviders(base, aspects, SkylarkProviders.class));

    // Merge extra-actions provider.
    ExtraActionArtifactsProvider mergedExtraActionProviders = ExtraActionArtifactsProvider.merge(
        getAllProviders(base, aspects, ExtraActionArtifactsProvider.class));

    TransitiveInfoProviderMap.Builder aspectProviders = TransitiveInfoProviderMap.builder();
    if (mergedOutputGroupProvider != null) {
      aspectProviders.add(mergedOutputGroupProvider);
    }
    if (mergedSkylarkProviders != null) {
      aspectProviders.add(mergedSkylarkProviders);
    }
    if (mergedExtraActionProviders != null) {
      aspectProviders.add(mergedExtraActionProviders);
    }

    for (ConfiguredAspect aspect : aspects) {
      for (Map.Entry<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> entry :
          aspect.getProviders().entrySet()) {
        Class<? extends TransitiveInfoProvider> providerClass = entry.getKey();
        if (OutputGroupProvider.class.equals(providerClass)
            || SkylarkProviders.class.equals(providerClass)
            || ExtraActionArtifactsProvider.class.equals(providerClass)) {
          continue;
        }

        if (base.getProvider(providerClass) != null || aspectProviders.contains(providerClass)) {
          throw new IllegalStateException("Provider " + providerClass + " provided twice");
        }

        aspectProviders.add(entry.getValue());
      }
    }
    return new MergedConfiguredTarget(base, aspectProviders.build());
  }

  private static ImmutableList<OutputGroupProvider> getAllOutputGroupProviders(
      ConfiguredTarget base, Iterable<ConfiguredAspect> aspects) {
    OutputGroupProvider baseProvider = OutputGroupProvider.get(base);
    ImmutableList.Builder<OutputGroupProvider> providers = ImmutableList.builder();
    if (baseProvider != null) {
      providers.add(baseProvider);
    }

    for (ConfiguredAspect configuredAspect : aspects) {
      OutputGroupProvider aspectProvider = OutputGroupProvider.get(configuredAspect);;
      if (aspectProvider == null) {
        continue;
      }
      providers.add(aspectProvider);
    }
    return providers.build();
  }

  private static <T extends TransitiveInfoProvider> List<T> getAllProviders(
      ConfiguredTarget base, Iterable<ConfiguredAspect> aspects, Class<T> providerClass) {
    T baseProvider = base.getProvider(providerClass);
    List<T> providers = new ArrayList<>();
    if (baseProvider != null) {
      providers.add(baseProvider);
    }

    for (ConfiguredAspect configuredAspect : aspects) {
      T aspectProvider = configuredAspect.getProvider(providerClass);
      if (aspectProvider == null) {
        continue;
      }
      providers.add(aspectProvider);
    }
    return providers;
  }
}
