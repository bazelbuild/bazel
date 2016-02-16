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

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.UnmodifiableIterator;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.RunUnder;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.SkylarkApiProvider;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.ArrayList;
import java.util.HashSet;
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
  private final ImmutableList<ConfiguredAspect> configuredAspects;

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

    // Initialize every SkylarkApiProvider
    for (Object provider : skylarkProviders.values()) {
      if (provider instanceof SkylarkApiProvider) {
        ((SkylarkApiProvider) provider).init(this);
      }
    }

    providerBuilder.put(SkylarkProviders.class, new SkylarkProviders(skylarkProviders));

    this.providers = ImmutableMap.copyOf(providerBuilder);
    this.mandatoryStampFiles = mandatoryStampFiles;
    this.configConditions = ruleContext.getConfigConditions();
    this.configuredAspects = ImmutableList.of();

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
   * Merge a configured target with its associated aspects.
   *
   * <p>If aspects are present, the configured target must be created from a rule (instead of e.g.
   * an input or an output file).
   */
  public static ConfiguredTarget mergeAspects(
      ConfiguredTarget base, Iterable<ConfiguredAspect> aspects) {
    if (Iterables.isEmpty(aspects)) {
      // If there are no aspects, don't bother with creating a proxy object
      return base;
    } else {
      // Aspects can only be attached to rules for now. This invariant is upheld by
      // DependencyResolver#requiredAspects()
      return new RuleConfiguredTarget((RuleConfiguredTarget) base, aspects);
    }
  }

  /**
   * Creates an instance based on a configured target and a set of aspects.
   */
  private RuleConfiguredTarget(RuleConfiguredTarget base, Iterable<ConfiguredAspect> aspects) {
    super(base.getTarget(), base.getConfiguration());

    Set<Class<? extends TransitiveInfoProvider>> providers = new HashSet<>();

    providers.addAll(base.providers.keySet());

    // Merge output group providers.
    OutputGroupProvider mergedOutputGroupProvider =
        OutputGroupProvider.merge(getAllProviders(base, aspects, OutputGroupProvider.class));

    // Merge Skylark providers.
    SkylarkProviders mergedSkylarkProviders =
        SkylarkProviders.merge(getAllProviders(base, aspects, SkylarkProviders.class));

    // Merge extra-actions provider.
    ExtraActionArtifactsProvider mergedExtraActionProviders = ExtraActionArtifactsProvider.merge(
        getAllProviders(base, aspects, ExtraActionArtifactsProvider.class));

    // Validate that all other providers are only provided once.
    for (ConfiguredAspect configuredAspect : aspects) {
      for (TransitiveInfoProvider aspectProvider : configuredAspect) {
        Class<? extends TransitiveInfoProvider> aClass = aspectProvider.getClass();
        if (OutputGroupProvider.class.equals(aClass)) {
          continue;
        }
        if (SkylarkProviders.class.equals(aClass)) {
          continue;
        }
        if (ExtraActionArtifactsProvider.class.equals(aClass)) {
          continue;
        }
        if (!providers.add(aClass)) {
          throw new IllegalStateException("Provider " + aClass + " provided twice");
        }
      }
    }

    if (base.getProvider(OutputGroupProvider.class) == mergedOutputGroupProvider
        && base.getProvider(SkylarkProviders.class) == mergedSkylarkProviders
        && base.getProvider(ExtraActionArtifactsProvider.class) == mergedExtraActionProviders) {
      this.providers = base.providers;
    } else {
      ImmutableMap.Builder<Class<? extends TransitiveInfoProvider>, Object> builder =
          new ImmutableMap.Builder<>();
      for (Class<? extends TransitiveInfoProvider> aClass : base.providers.keySet()) {
        if (OutputGroupProvider.class.equals(aClass)) {
          continue;
        }
        if (SkylarkProviders.class.equals(aClass)) {
          continue;
        }
        if (ExtraActionArtifactsProvider.class.equals(aClass)) {
          continue;
        }
        builder.put(aClass, base.providers.get(aClass));
      }
      if (mergedOutputGroupProvider != null) {
        builder.put(OutputGroupProvider.class, mergedOutputGroupProvider);
      }
      if (mergedSkylarkProviders != null) {
        builder.put(SkylarkProviders.class, mergedSkylarkProviders);
      }
      if (mergedExtraActionProviders != null) {
        builder.put(ExtraActionArtifactsProvider.class, mergedExtraActionProviders);
      }
      this.providers = builder.build();
    }
    this.mandatoryStampFiles = base.mandatoryStampFiles;
    this.configConditions = base.configConditions;
    this.configuredAspects = ImmutableList.copyOf(aspects);
  }

  private static <T extends TransitiveInfoProvider> List<T> getAllProviders(
      RuleConfiguredTarget base, Iterable<ConfiguredAspect> aspects, Class<T> providerClass) {
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

  /**
   * The configuration conditions that trigger this rule's configurable attributes.
   */
  Set<ConfigMatchingProvider> getConfigConditions() {
    return configConditions;
  }

  @Override
  public <P extends TransitiveInfoProvider> P getProvider(Class<P> providerClass) {
    AnalysisUtils.checkProvider(providerClass);
    // TODO(bazel-team): Should aspects be allowed to override providers on the configured target
    // class?
    Object provider = providers.get(providerClass);
    if (provider == null) {
      for (ConfiguredAspect configuredAspect : configuredAspects) {
        provider = configuredAspect.getProviders().get(providerClass);
        if (provider != null) {
          break;
        }
      }
    }

    return providerClass.cast(provider);
  }

  /**
   * Returns a value provided by this target. Only meant to use from Skylark.
   */
  @Override
  public Object get(String providerKey) {
    return getProvider(SkylarkProviders.class).getValue(providerKey);
  }

  public ImmutableList<Artifact> getMandatoryStampFiles() {
    return mandatoryStampFiles;
  }

  @Override
  public final Rule getTarget() {
    return (Rule) super.getTarget();
  }

  @Override
  public UnmodifiableIterator<TransitiveInfoProvider> iterator() {
    Map<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> allProviders =
        new LinkedHashMap<>();
    for (int i = configuredAspects.size() - 1; i >= 0; i--) {
      for (TransitiveInfoProvider tip : configuredAspects.get(i)) {
        allProviders.put(tip.getClass(), tip);
      }
    }

    for (Map.Entry<Class<? extends TransitiveInfoProvider>, Object> entry : providers.entrySet()) {
      allProviders.put(entry.getKey(), entry.getKey().cast(entry.getValue()));
    }

    return ImmutableList.copyOf(allProviders.values()).iterator();
  }

  @Override
  public String errorMessage(String name) {
    return String.format("target (rule class of '%s') doesn't have provider '%s'.",
        getTarget().getRuleClass(), name);
  }

  @Override
  public ImmutableCollection<String> getKeys() {
    return ImmutableList.<String>builder().addAll(super.getKeys())
        .addAll(getProvider(SkylarkProviders.class).getKeys()).build();
  }
}
