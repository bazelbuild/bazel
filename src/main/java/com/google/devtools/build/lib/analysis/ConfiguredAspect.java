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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.analysis.ExtraActionUtils.createExtraActionProvider;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.starlark.StarlarkApiProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Provider;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.TreeMap;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/**
 * Extra information about a configured target computed on request of a dependent.
 *
 * <p>Analogous to {@link ConfiguredTarget}: contains a bunch of transitive info providers, which
 * are merged with the providers of the associated configured target before they are passed to the
 * configured target factories that depend on the configured target to which this aspect is added.
 *
 * <p>Aspects are created alongside configured targets on request from dependents.
 *
 * <p>For more information about aspects, see {@link
 * com.google.devtools.build.lib.packages.AspectClass}.
 *
 * @see com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory
 * @see com.google.devtools.build.lib.packages.AspectClass
 */
public interface ConfiguredAspect extends ProviderCollection {

  ImmutableList<ActionAnalysisMetadata> getActions();

  /** Returns the providers created by the aspect. */
  TransitiveInfoProviderMap getProviders();

  @Override
  @Nullable
  default <P extends TransitiveInfoProvider> P getProvider(Class<P> providerClass) {
    AnalysisUtils.checkProvider(providerClass);
    return getProviders().getProvider(providerClass);
  }

  @Override
  default Info get(Provider.Key key) {
    return getProviders().get(key);
  }

  @Override
  default Object get(String legacyKey) {
    if (OutputGroupInfo.STARLARK_NAME.equals(legacyKey)) {
      return get(OutputGroupInfo.STARLARK_CONSTRUCTOR.getKey());
    }
    return getProviders().get(legacyKey);
  }

  static ConfiguredAspect forAlias(ConfiguredAspect real) {
    return BasicConfiguredAspect.create(real.getActions(), real.getProviders());
  }

  static ConfiguredAspect forNonapplicableTarget() {
    return BasicConfiguredAspect.create(
        ImmutableList.of(), new TransitiveInfoProviderMapBuilder().build());
  }

  static Builder builder(RuleContext ruleContext) {
    return new Builder(ruleContext);
  }

  /** Builder for {@link ConfiguredAspect}. */
  final class Builder {
    private final TransitiveInfoProviderMapBuilder providers =
        new TransitiveInfoProviderMapBuilder();
    private final TreeMap<String, NestedSetBuilder<Artifact>> outputGroupBuilders = new TreeMap<>();
    private final RuleContext ruleContext;

    public Builder(RuleContext ruleContext) {
      this.ruleContext = ruleContext;
    }

    @CanIgnoreReturnValue
    public <T extends TransitiveInfoProvider> Builder addProvider(
        Class<? extends T> providerClass, T provider) {
      checkNotNull(provider);
      checkProviderClass(providerClass);
      providers.put(providerClass, provider);
      return this;
    }

    /** Adds a provider to the aspect. */
    @CanIgnoreReturnValue
    public Builder addProvider(TransitiveInfoProvider provider) {
      checkNotNull(provider);
      addProvider(TransitiveInfoProviderEffectiveClassHelper.get(provider), provider);
      return this;
    }

    private static void checkProviderClass(Class<? extends TransitiveInfoProvider> providerClass) {
      checkNotNull(providerClass);
    }

    /** Adds a set of files to an output group. */
    @CanIgnoreReturnValue
    public Builder addOutputGroup(String name, NestedSet<Artifact> artifacts) {
      outputGroupBuilders
          .computeIfAbsent(name, k -> NestedSetBuilder.stableOrder())
          .addTransitive(artifacts);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addStarlarkTransitiveInfo(String name, Object value) {
      providers.put(name, value);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addStarlarkDeclaredProvider(Info declaredProvider) throws EvalException {
      Provider constructor = declaredProvider.getProvider();
      if (!constructor.isExported()) {
        throw Starlark.errorf(
            "aspect function returned an instance of a provider (defined at %s) that is not a"
                + " global",
            constructor.getLocation());
      }
      addDeclaredProvider(declaredProvider);
      return this;
    }

    private void addDeclaredProvider(Info declaredProvider) {
      providers.put(declaredProvider);
    }

    @CanIgnoreReturnValue
    public Builder addNativeDeclaredProvider(Info declaredProvider) {
      Provider constructor = declaredProvider.getProvider();
      Preconditions.checkState(constructor.isExported());
      addDeclaredProvider(declaredProvider);
      return this;
    }

    @Nullable
    public ConfiguredAspect build() throws ActionConflictException, InterruptedException {
      if (!outputGroupBuilders.isEmpty()) {
        if (providers.contains(OutputGroupInfo.STARLARK_CONSTRUCTOR.getKey())) {
          throw new IllegalStateException(
              "OutputGroupInfo was provided explicitly; do not use addOutputGroup");
        }
        addDeclaredProvider(OutputGroupInfo.fromBuilders(outputGroupBuilders));
      }

      addProvider(
          createExtraActionProvider(/*actionsWithoutExtraAction=*/ ImmutableSet.of(), ruleContext));

      AnalysisEnvironment analysisEnvironment = ruleContext.getAnalysisEnvironment();
      ImmutableList<ActionAnalysisMetadata> actions = analysisEnvironment.getRegisteredActions();
      try {
        Actions.assignOwnersAndThrowIfConflictToleratingSharedActions(
            analysisEnvironment.getActionKeyContext(), actions, ruleContext.getOwner());
      } catch (Actions.ArtifactGeneratedByOtherRuleException e) {
        ruleContext.ruleError(e.getMessage());
        return null;
      }

      maybeAddRequiredConfigFragmentsProvider();

      TransitiveInfoProviderMap providerMap = providers.build();

      // Initialize every StarlarkApiProvider
      for (int i = 0; i < providerMap.getProviderCount(); i++) {
        Object obj = providerMap.getProviderInstanceAt(i);
        if (obj instanceof StarlarkApiProvider) {
          ((StarlarkApiProvider) obj).init(providerMap);
        }
      }

      return BasicConfiguredAspect.create(actions, providerMap);
    }

    /**
     * Adds {@link RequiredConfigFragmentsProvider} if {@link
     * CoreOptions#includeRequiredConfigFragmentsProvider} isn't {@link
     * CoreOptions.IncludeConfigFragmentsEnum#OFF}.
     *
     * <p>See {@link com.google.devtools.build.lib.analysis.config.RequiredFragmentsUtil} for a
     * description of the meaning of this provider's content. That class contains methods that
     * populate the results of {@link RuleContext#getRequiredConfigFragments}.
     */
    private void maybeAddRequiredConfigFragmentsProvider() {
      if (ruleContext.shouldIncludeRequiredConfigFragmentsProvider()) {
        addProvider(ruleContext.getRequiredConfigFragments());
      }
    }
  }

  /** Basic implementation of {@link ConfiguredAspect}. */
  @AutoValue
  abstract class BasicConfiguredAspect implements ConfiguredAspect {

    private static BasicConfiguredAspect create(
        ImmutableList<ActionAnalysisMetadata> actions, TransitiveInfoProviderMap providers) {
      return new AutoValue_ConfiguredAspect_BasicConfiguredAspect(actions, providers);
    }

    BasicConfiguredAspect() {}
  }
}
