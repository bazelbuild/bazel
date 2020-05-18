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
package com.google.devtools.build.lib.analysis.configuredtargets;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.DuplicateException;
import com.google.devtools.build.lib.analysis.ExtraActionArtifactsProvider;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RequiredConfigFragmentsProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMap;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMapBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.Provider.Key;
import com.google.devtools.build.lib.skylarkbuildapi.ActionApi;
import com.google.devtools.build.lib.syntax.Printer;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * A single dependency with its configured target and aspects merged together.
 *
 * <p>This is an ephemeral object created only for the analysis of a single configured target. After
 * that configured target is analyzed, this is thrown away.
 */
@Immutable
public final class MergedConfiguredTarget extends AbstractConfiguredTarget {
  private final ConfiguredTarget base;
  private final ImmutableList<ConfiguredAspect> aspects;
  /**
   * Providers that come from any source that isn't a pure pointer to the base rule's providers.
   *
   * <p>Examples include providers from aspects and merged providers that appear in both the base
   * rule and aspects.
   */
  private final TransitiveInfoProviderMap nonBaseProviders;

  private MergedConfiguredTarget(
      ConfiguredTarget base,
      Iterable<ConfiguredAspect> aspects,
      TransitiveInfoProviderMap nonBaseProviders) {
    super(base.getLabel(), base.getConfigurationKey());
    this.base = base;
    this.aspects = ImmutableList.copyOf(aspects);
    this.nonBaseProviders = nonBaseProviders;
  }

  @Override
  public <P extends TransitiveInfoProvider> P getProvider(Class<P> providerClass) {
    AnalysisUtils.checkProvider(providerClass);

    P provider = nonBaseProviders.getProvider(providerClass);
    if (provider == null) {
      provider = base.getProvider(providerClass);
    }

    return provider;
  }

  @Override
  protected void addExtraSkylarkKeys(Consumer<String> result) {
    if (base instanceof AbstractConfiguredTarget) {
      ((AbstractConfiguredTarget) base).addExtraSkylarkKeys(result);
    }
    for (int i = 0; i < nonBaseProviders.getProviderCount(); i++) {
      Object classAt = nonBaseProviders.getProviderKeyAt(i);
      if (classAt instanceof String) {
        result.accept((String) classAt);
      }
    }
    result.accept(AbstractConfiguredTarget.ACTIONS_FIELD_NAME);
  }

  @Override
  protected Info rawGetStarlarkProvider(Provider.Key providerKey) {
    Info provider = nonBaseProviders.get(providerKey);
    if (provider == null) {
      provider = base.get(providerKey);
    }
    return provider;
  }

  @Override
  protected Object rawGetStarlarkProvider(String providerKey) {
    if (providerKey.equals(AbstractConfiguredTarget.ACTIONS_FIELD_NAME)) {
      ImmutableList.Builder<ActionAnalysisMetadata> actions = ImmutableList.builder();
      // Only expose actions which are SkylarkValues.
      // TODO(cparsons): Expose all actions to Starlark.
      for (ConfiguredAspect aspect : aspects) {
        actions.addAll(
            aspect.getActions().stream().filter(action -> action instanceof ActionApi).iterator());
      }
      if (base instanceof RuleConfiguredTarget) {
        actions.addAll(
            ((RuleConfiguredTarget) base)
                .getActions().stream().filter(action -> action instanceof ActionApi).iterator());
      }
      return actions.build();
    }
    Object provider = nonBaseProviders.get(providerKey);
    if (provider == null) {
      provider = base.get(providerKey);
    }
    return provider;
  }

  /** Creates an instance based on a configured target and a set of aspects. */
  public static ConfiguredTarget of(ConfiguredTarget base, Iterable<ConfiguredAspect> aspects)
      throws DuplicateException {
    if (Iterables.isEmpty(aspects)) {
      // If there are no aspects, don't bother with creating a proxy object
      return base;
    }

    TransitiveInfoProviderMapBuilder nonBaseProviders = new TransitiveInfoProviderMapBuilder();

    // Merge output group providers.
    OutputGroupInfo mergedOutputGroupInfo =
        OutputGroupInfo.merge(getAllOutputGroupProviders(base, aspects));
    if (mergedOutputGroupInfo != null) {
      nonBaseProviders.put(mergedOutputGroupInfo);
    }

    // Merge extra-actions provider.
    ExtraActionArtifactsProvider mergedExtraActionProviders = ExtraActionArtifactsProvider.merge(
        getAllProviders(base, aspects, ExtraActionArtifactsProvider.class));
    if (mergedExtraActionProviders != null) {
      nonBaseProviders.add(mergedExtraActionProviders);
    }

    // Merge required config fragments provider.
    List<RequiredConfigFragmentsProvider> requiredConfigFragmentProviders =
        getAllProviders(base, aspects, RequiredConfigFragmentsProvider.class);
    if (!requiredConfigFragmentProviders.isEmpty()) {
      nonBaseProviders.add(RequiredConfigFragmentsProvider.merge(requiredConfigFragmentProviders));
    }

    for (ConfiguredAspect aspect : aspects) {
      TransitiveInfoProviderMap providers = aspect.getProviders();
      for (int i = 0; i < providers.getProviderCount(); ++i) {
        Object providerKey = providers.getProviderKeyAt(i);
        if (OutputGroupInfo.SKYLARK_CONSTRUCTOR.getKey().equals(providerKey)
            || ExtraActionArtifactsProvider.class.equals(providerKey)
            || RequiredConfigFragmentsProvider.class.equals(providerKey)) {
          continue;
        }

        if (providerKey instanceof Class<?>) {
          @SuppressWarnings("unchecked")
          Class<? extends TransitiveInfoProvider> providerClass =
              (Class<? extends TransitiveInfoProvider>) providerKey;
          if (base.getProvider(providerClass) != null || nonBaseProviders.contains(providerClass)) {
            throw new DuplicateException("Provider " + providerKey + " provided twice");
          }
          nonBaseProviders.put(
              providerClass, (TransitiveInfoProvider) providers.getProviderInstanceAt(i));
        } else if (providerKey instanceof String) {
          String legacyId = (String) providerKey;
          if (base.get(legacyId) != null || nonBaseProviders.contains(legacyId)) {
            throw new DuplicateException("Provider " + legacyId + " provided twice");
          }
          nonBaseProviders.put(legacyId, providers.getProviderInstanceAt(i));
        } else if (providerKey instanceof Provider.Key) {
          Provider.Key key = (Key) providerKey;
          if (base.get(key) != null || nonBaseProviders.contains(key)) {
            throw new DuplicateException("Provider " + key + " provided twice");
          }
          nonBaseProviders.put((Info) providers.getProviderInstanceAt(i));
        }
      }
    }
    return new MergedConfiguredTarget(base, aspects, nonBaseProviders.build());
  }

  private static ImmutableList<OutputGroupInfo> getAllOutputGroupProviders(
      ConfiguredTarget base, Iterable<ConfiguredAspect> aspects) {
    OutputGroupInfo baseProvider = OutputGroupInfo.get(base);
    ImmutableList.Builder<OutputGroupInfo> providers = ImmutableList.builder();
    if (baseProvider != null) {
      providers.add(baseProvider);
    }

    for (ConfiguredAspect configuredAspect : aspects) {
      OutputGroupInfo aspectProvider = OutputGroupInfo.get(configuredAspect);
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

  @Override
  public void repr(Printer printer) {
    printer.append("<merged target " + getLabel() + ">");
  }

  @VisibleForTesting
  public ConfiguredTarget getBaseConfiguredTargetForTesting() {
    return base;
  }
}
