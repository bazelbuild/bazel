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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionLookupKey;
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
import com.google.devtools.build.lib.analysis.test.AnalysisFailure;
import com.google.devtools.build.lib.analysis.test.AnalysisFailureInfo;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.skyframe.BuildConfigurationKey;
import com.google.devtools.build.lib.starlarkbuildapi.ActionApi;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.function.Consumer;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.Printer;

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
    // TODO(b/281522692): it's unsound to pass a null key here, but the type system doesn't
    // currently provide a better way to do this.
    super(/* actionLookupKey= */ null);
    this.base = base;
    this.aspects = ImmutableList.copyOf(aspects);
    this.nonBaseProviders = nonBaseProviders;
  }

  @Override
  public ActionLookupKey getLookupKey() {
    throw new IllegalStateException(
        "MergedConfiguredTarget is ephemeral. It does not exist in the Skyframe graph and it does"
            + " not have a key.");
  }

  @Override
  public Label getLabel() {
    return base.getLabel();
  }

  @Override
  @Nullable
  public BuildConfigurationKey getConfigurationKey() {
    return base.getConfigurationKey();
  }

  @Override
  @Nullable
  public <P extends TransitiveInfoProvider> P getProvider(Class<P> providerClass) {
    AnalysisUtils.checkProvider(providerClass);

    P provider = nonBaseProviders.getProvider(providerClass);
    if (provider != null) {
      return provider;
    }
    provider = base.getProvider(providerClass);
    if (provider != null) {
      return provider;
    }
    if (providerClass.isAssignableFrom(getClass())) {
      return providerClass.cast(this);
    }
    return null;
  }

  @Override
  protected void addExtraStarlarkKeys(Consumer<String> result) {
    if (base instanceof AbstractConfiguredTarget) {
      ((AbstractConfiguredTarget) base).addExtraStarlarkKeys(result);
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
      // Only expose actions which are StarlarkValues.
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
  public static ConfiguredTarget of(ConfiguredTarget base, Collection<ConfiguredAspect> aspects)
      throws DuplicateException {
    if (aspects.isEmpty()) {
      return base; // If there are no aspects, don't bother with creating a proxy object.
    }

    TransitiveInfoProviderMapBuilder nonBaseProviders = new TransitiveInfoProviderMapBuilder();

    // Merge output group providers.
    OutputGroupInfo mergedOutputGroupInfo =
        OutputGroupInfo.merge(getAllOutputGroupProviders(base, aspects));
    if (mergedOutputGroupInfo != null) {
      nonBaseProviders.put(mergedOutputGroupInfo);
    }

    // Merge analysis failures.
    ImmutableList<NestedSet<AnalysisFailure>> analysisFailures = getAnalysisFailures(base, aspects);
    if (!analysisFailures.isEmpty()) {
      nonBaseProviders.put(AnalysisFailureInfo.forAnalysisFailureSets(analysisFailures));
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
        if (OutputGroupInfo.STARLARK_CONSTRUCTOR.getKey().equals(providerKey)
            || AnalysisFailureInfo.STARLARK_CONSTRUCTOR.getKey().equals(providerKey)
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
          Provider.Key key = (Provider.Key) providerKey;
          // If InstrumentedFilesInfo is on both the base target and an aspect, ignore the one from
          // the base. Otherwise, sharing implementation between a rule which returns
          // InstrumentedFilesInfo (e.g. *_library) and a related aspect (e.g. *_proto_library) can
          // add an implicit brittle assumption that the underlying rule (e.g. proto_library) does
          // not return InstrumentedFilesInfo.
          if ((!InstrumentedFilesInfo.STARLARK_CONSTRUCTOR.getKey().equals(key)
                  && base.get(key) != null)
              || nonBaseProviders.contains(key)) {
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

  private static ImmutableList<NestedSet<AnalysisFailure>> getAnalysisFailures(
      ConfiguredTarget base, Iterable<ConfiguredAspect> aspects) {
    ImmutableList.Builder<NestedSet<AnalysisFailure>> analysisFailures = ImmutableList.builder();
    AnalysisFailureInfo baseFailureInfo = base.get(AnalysisFailureInfo.STARLARK_CONSTRUCTOR);
    if (baseFailureInfo != null) {
      analysisFailures.add(baseFailureInfo.getCausesNestedSet());
    }
    for (ConfiguredAspect configuredAspect : aspects) {
      AnalysisFailureInfo aspectFailureInfo =
          configuredAspect.get(AnalysisFailureInfo.STARLARK_CONSTRUCTOR);
      if (aspectFailureInfo != null) {
        analysisFailures.add(aspectFailureInfo.getCausesNestedSet());
      }
    }
    return analysisFailures.build();
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

  @Override
  public Dict<String, Object> getProvidersDictForQuery() {
    return toProvidersDictForQuery(nonBaseProviders);
  }

  @Override
  public String getRuleClassString() {
    if (!(base instanceof AbstractConfiguredTarget)) {
      return super.getRuleClassString();
    }
    AbstractConfiguredTarget act = (AbstractConfiguredTarget) base;
    return act.getRuleClassString();
  }

  public ConfiguredTarget getBaseConfiguredTarget() {
    return base;
  }

  @Override
  public ConfiguredTarget unwrapIfMerged() {
    return base.unwrapIfMerged();
  }
}
