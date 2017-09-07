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

package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.CachingAnalysisEnvironment;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.DependencyResolver.InconsistentAspectOrderException;
import com.google.devtools.build.lib.analysis.MergedConfiguredTarget;
import com.google.devtools.build.lib.analysis.MergedConfiguredTarget.DuplicateException;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.ToolchainContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.SkylarkAspect;
import com.google.devtools.build.lib.packages.SkylarkAspectClass;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.AspectValue.AspectKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetFunction.ConfiguredTargetFunctionException;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetFunction.ConfiguredValueCreationException;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetFunction.DependencyEvaluationException;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor.BuildViewProvider;
import com.google.devtools.build.lib.skyframe.SkylarkImportLookupFunction.SkylarkImportFailedException;
import com.google.devtools.build.lib.skyframe.ToolchainUtil.ToolchainContextException;
import com.google.devtools.build.lib.syntax.Type.ConversionException;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * The Skyframe function that generates aspects.
 *
 * This class, together with {@link ConfiguredTargetFunction} drives the analysis phase. For more
 * information, see {@link com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory}.
 *
 * {@link AspectFunction} takes a SkyKey containing an {@link AspectKey} [a tuple of
 * (target label, configurations, aspect class and aspect parameters)],
 * loads an {@link Aspect} from aspect class and aspect parameters,
 * gets a {@link ConfiguredTarget} for label and configurations, and then creates
 * a {@link ConfiguredAspect} for a given {@link AspectKey}.
 *
 * See {@link com.google.devtools.build.lib.packages.AspectClass} documentation
 * for an overview of aspect-related classes
 *
 * @see com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory
 * @see com.google.devtools.build.lib.packages.AspectClass
 */
public final class AspectFunction implements SkyFunction {
  private final BuildViewProvider buildViewProvider;
  private final RuleClassProvider ruleClassProvider;
  private final Supplier<Boolean> removeActionsAfterEvaluation;

  AspectFunction(
      BuildViewProvider buildViewProvider,
      RuleClassProvider ruleClassProvider,
      Supplier<Boolean> removeActionsAfterEvaluation) {
    this.buildViewProvider = buildViewProvider;
    this.ruleClassProvider = ruleClassProvider;
    this.removeActionsAfterEvaluation = Preconditions.checkNotNull(removeActionsAfterEvaluation);
  }

  /**
   * Load Skylark aspect from an extension file. Is to be called from a SkyFunction.
   *
   * @return {@code null} if dependencies cannot be satisfied.
   */
  @Nullable
  static SkylarkAspect loadSkylarkAspect(
      Environment env, Label extensionLabel, String skylarkValueName)
      throws AspectCreationException, InterruptedException {
    SkyKey importFileKey = SkylarkImportLookupValue.key(extensionLabel, false);
    try {
      SkylarkImportLookupValue skylarkImportLookupValue =
          (SkylarkImportLookupValue) env.getValueOrThrow(
              importFileKey, SkylarkImportFailedException.class);
      if (skylarkImportLookupValue == null) {
        return null;
      }

      Object skylarkValue = skylarkImportLookupValue.getEnvironmentExtension().getBindings()
          .get(skylarkValueName);
      if (skylarkValue == null) {
        throw new ConversionException(
            String.format(
                "%s is not exported from %s", skylarkValueName, extensionLabel.toString()));
      }
      if (!(skylarkValue instanceof SkylarkAspect)) {
        throw new ConversionException(
            String.format(
                "%s from %s is not an aspect", skylarkValueName, extensionLabel.toString()));
      }
      return (SkylarkAspect) skylarkValue;
    } catch (SkylarkImportFailedException | ConversionException e) {
      env.getListener().handle(Event.error(e.getMessage()));
      throw new AspectCreationException(e.getMessage());
    }
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws AspectFunctionException, InterruptedException {
    SkyframeBuildView view = buildViewProvider.getSkyframeBuildView();
    NestedSetBuilder<Package> transitivePackages = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Label> transitiveRootCauses = NestedSetBuilder.stableOrder();
    AspectKey key = (AspectKey) skyKey.argument();
    ConfiguredAspectFactory aspectFactory;
    Aspect aspect;
    if (key.getAspectClass() instanceof NativeAspectClass) {
      NativeAspectClass nativeAspectClass = (NativeAspectClass) key.getAspectClass();
      aspectFactory = (ConfiguredAspectFactory) nativeAspectClass;
      aspect = Aspect.forNative(nativeAspectClass, key.getParameters());
    } else if (key.getAspectClass() instanceof SkylarkAspectClass) {
      SkylarkAspectClass skylarkAspectClass = (SkylarkAspectClass) key.getAspectClass();
      SkylarkAspect skylarkAspect;
      try {
        skylarkAspect =
            loadSkylarkAspect(
                env, skylarkAspectClass.getExtensionLabel(), skylarkAspectClass.getExportedName());
      } catch (AspectCreationException e) {
        throw new AspectFunctionException(e);
      }
      if (skylarkAspect == null) {
        return null;
      }

      aspectFactory = new SkylarkAspectFactory(skylarkAspect);
      aspect = Aspect.forSkylark(
          skylarkAspect.getAspectClass(),
          skylarkAspect.getDefinition(key.getParameters()),
          key.getParameters());
    } else {
      throw new IllegalStateException();
    }

    // Keep this in sync with the same code in ConfiguredTargetFunction.
    PackageValue packageValue =
        (PackageValue) env.getValue(PackageValue.key(key.getLabel().getPackageIdentifier()));
    if (packageValue == null) {
      return null;
    }
    Package pkg = packageValue.getPackage();
    if (pkg.containsErrors()) {
      throw new AspectFunctionException(
          new BuildFileContainsErrorsException(key.getLabel().getPackageIdentifier()));
    }


    ConfiguredTargetValue configuredTargetValue;
    try {
      configuredTargetValue =
          (ConfiguredTargetValue) env.getValueOrThrow(
              ConfiguredTargetValue.key(key.getLabel(), key.getBaseConfiguration()),
              ConfiguredValueCreationException.class);
    } catch (ConfiguredValueCreationException e) {
      throw new AspectFunctionException(new AspectCreationException(e.getRootCauses()));
    }
    if (configuredTargetValue == null) {
      // TODO(bazel-team): remove this check when top-level targets also use dynamic configurations.
      // Right now the key configuration may be dynamic while the original target's configuration
      // is static, resulting in a Skyframe cache miss even though the original target is, in fact,
      // precomputed.
      return null;
    }


    if (configuredTargetValue.getConfiguredTarget() == null) {
      return null;
    }

    ConfiguredTarget associatedTarget = configuredTargetValue.getConfiguredTarget();

    Target target = associatedTarget.getTarget();

    if (configuredTargetValue.getConfiguredTarget().getProvider(AliasProvider.class) != null) {
      return createAliasAspect(env, target, aspect, key,
          configuredTargetValue.getConfiguredTarget());
    }


    ImmutableList.Builder<Aspect> aspectPathBuilder = ImmutableList.builder();

    if (!key.getBaseKeys().isEmpty()) {
      // We transitively collect all required aspects to reduce the number of restarts.
      // Semantically it is enough to just request key.getBaseKeys().
      ImmutableList.Builder<SkyKey> aspectPathSkyKeysBuilder = ImmutableList.builder();
      ImmutableMap<AspectDescriptor, SkyKey> aspectKeys =
          getSkyKeysForAspectsAndCollectAspectPath(key.getBaseKeys(), aspectPathSkyKeysBuilder);

      Map<SkyKey, SkyValue> values = env.getValues(aspectKeys.values());
      if (env.valuesMissing()) {
        return null;
      }
      ImmutableList<SkyKey> aspectPathSkyKeys = aspectPathSkyKeysBuilder.build();
      for (SkyKey aspectPathSkyKey : aspectPathSkyKeys) {
        aspectPathBuilder.add(((AspectValue) values.get(aspectPathSkyKey)).getAspect());
      }
      try {
        associatedTarget = getBaseTarget(
            associatedTarget, key.getBaseKeys(), values);
      } catch (DuplicateException e) {
        env.getListener().handle(
            Event.error(associatedTarget.getTarget().getLocation(), e.getMessage()));

        throw new AspectFunctionException(
            new AspectCreationException(e.getMessage(), associatedTarget.getLabel()));

      }
    }
    aspectPathBuilder.add(aspect);

    SkyframeDependencyResolver resolver = view.createDependencyResolver(env);

    // When getting the dependencies of this hybrid aspect+base target, use the aspect's
    // configuration. The configuration of the aspect will always be a superset of the target's
    // (dynamic configuration mode: target is part of the aspect's config fragment requirements;
    // static configuration mode: target is the same configuration as the aspect), so the fragments
    // required by all dependencies (both those of the aspect and those of the base target)
    // will be present this way.
    TargetAndConfiguration originalTargetAndAspectConfiguration =
        new TargetAndConfiguration(target, key.getAspectConfiguration());
    ImmutableList<Aspect> aspectPath = aspectPathBuilder.build();
    try {
      // Get the configuration targets that trigger this rule's configurable attributes.
      ImmutableMap<Label, ConfigMatchingProvider> configConditions =
          ConfiguredTargetFunction.getConfigConditions(
              target, env, resolver, originalTargetAndAspectConfiguration,
              transitivePackages, transitiveRootCauses);
      if (configConditions == null) {
        // Those targets haven't yet been resolved.
        return null;
      }

      // Determine what toolchains are needed by this target.
      ToolchainContext toolchainContext;
      try {
        ImmutableSet<Label> requiredToolchains = aspect.getDefinition().getRequiredToolchains();
        toolchainContext =
            ToolchainUtil.createToolchainContext(
                env,
                String.format(
                    "aspect %s applied to %s",
                    aspect.getDescriptor().getDescription(), target.toString()),
                requiredToolchains,
                key.getAspectConfiguration());
      } catch (ToolchainContextException e) {
        // TODO(katre): better error handling
        throw new AspectCreationException(e.getMessage());
      }
      if (env.valuesMissing()) {
        return null;
      }

      OrderedSetMultimap<Attribute, ConfiguredTarget> depValueMap;
      try {
        depValueMap =
            ConfiguredTargetFunction.computeDependencies(
                env,
                resolver,
                originalTargetAndAspectConfiguration,
                aspectPath,
                configConditions,
                toolchainContext,
                ruleClassProvider,
                view.getHostConfiguration(originalTargetAndAspectConfiguration.getConfiguration()),
                transitivePackages,
                transitiveRootCauses);
      } catch (ConfiguredTargetFunctionException e) {
        throw new AspectCreationException(e.getMessage());
      }
      if (depValueMap == null) {
        return null;
      }
      if (!transitiveRootCauses.isEmpty()) {
        throw new AspectFunctionException(
            new AspectCreationException("Loading failed", transitiveRootCauses.build()));
      }

      return createAspect(
          env,
          key,
          aspectPath,
          aspect,
          aspectFactory,
          associatedTarget,
          key.getAspectConfiguration(),
          configConditions,
          toolchainContext,
          depValueMap,
          transitivePackages);
    } catch (DependencyEvaluationException e) {
      if (e.getCause() instanceof ConfiguredValueCreationException) {
        ConfiguredValueCreationException cause = (ConfiguredValueCreationException) e.getCause();
        throw new AspectFunctionException(new AspectCreationException(
            cause.getMessage(), cause.getAnalysisRootCause()));
      } else if (e.getCause() instanceof InconsistentAspectOrderException) {
        InconsistentAspectOrderException cause = (InconsistentAspectOrderException) e.getCause();
        throw new AspectFunctionException(new AspectCreationException(
            cause.getMessage()));
      } else {
        // Cast to InvalidConfigurationException as a consistency check. If you add any
        // DependencyEvaluationException constructors, you may need to change this code, too.
        InvalidConfigurationException cause = (InvalidConfigurationException) e.getCause();
        throw new AspectFunctionException(new AspectCreationException(cause.getMessage()));
      }
    } catch (AspectCreationException e) {
      throw new AspectFunctionException(e);
    }
  }

  /**
   * Merges aspects defined by {@code aspectKeys} into the {@code target} using
   * previously computed {@code values}.
   *
   * @return A {@link ConfiguredTarget} that is a result of a merge.
   * @throws DuplicateException if there is a duplicate provider provided by aspects.
   */
  private ConfiguredTarget getBaseTarget(ConfiguredTarget target,
      ImmutableList<AspectKey> aspectKeys,
      Map<SkyKey, SkyValue> values)
      throws DuplicateException {
    ArrayList<ConfiguredAspect> aspectValues = new ArrayList<>();
    for (AspectKey aspectKey : aspectKeys) {
      SkyKey skyAspectKey = aspectKey.getSkyKey();
      AspectValue aspectValue = (AspectValue) values.get(skyAspectKey);
      ConfiguredAspect configuredAspect = aspectValue.getConfiguredAspect();
      aspectValues.add(configuredAspect);
    }
    return MergedConfiguredTarget.of(target, aspectValues);
  }

  /**
   *  Collect all SkyKeys that are needed for a given list of AspectKeys,
   *  including transitive dependencies.
   *
   *  Also collects all propagating aspects in correct order.
   */
  private ImmutableMap<AspectDescriptor, SkyKey> getSkyKeysForAspectsAndCollectAspectPath(
      ImmutableList<AspectKey> keys,
      ImmutableList.Builder<SkyKey> aspectPathBuilder) {
    HashMap<AspectDescriptor, SkyKey> result = new HashMap<>();
    for (AspectKey key : keys) {
      buildSkyKeys(key, result, aspectPathBuilder);
    }
    return ImmutableMap.copyOf(result);
  }

  private void buildSkyKeys(AspectKey key, HashMap<AspectDescriptor, SkyKey> result,
      ImmutableList.Builder<SkyKey> aspectPathBuilder) {
    if (result.containsKey(key.getAspectDescriptor())) {
      return;
    }
    ImmutableList<AspectKey> baseKeys = key.getBaseKeys();
    SkyKey skyKey = key.getSkyKey();
    result.put(key.getAspectDescriptor(), skyKey);
    for (AspectKey baseKey : baseKeys) {
      buildSkyKeys(baseKey, result, aspectPathBuilder);
    }
    // Post-order list of aspect SkyKeys gives the order of propagating aspects:
    // the aspect comes after all aspects it transitively sees.
    aspectPathBuilder.add(skyKey);
  }

  private SkyValue createAliasAspect(
      Environment env,
      Target originalTarget,
      Aspect aspect,
      AspectKey originalKey,
      ConfiguredTarget configuredTarget)
      throws InterruptedException {
    ImmutableList<Label> aliasChain = configuredTarget.getProvider(AliasProvider.class)
        .getAliasChain();
    // Find the next alias in the chain: either the next alias (if there are two) or the name of
    // the real configured target.
    Label aliasLabel = aliasChain.size() > 1 ? aliasChain.get(1) : configuredTarget.getLabel();

    SkyKey depKey = ActionLookupValue.key(originalKey.withLabel(aliasLabel));

    // Compute the AspectValue of the target the alias refers to (which can itself be either an
    // alias or a real target)
    AspectValue real = (AspectValue) env.getValue(depKey);
    if (env.valuesMissing()) {
      return null;
    }

    NestedSet<Package> transitivePackages = NestedSetBuilder.<Package>stableOrder()
        .addTransitive(real.getTransitivePackages())
        .add(originalTarget.getPackage())
        .build();

    return new AspectValue(
        originalKey,
        aspect,
        originalTarget.getLabel(),
        originalTarget.getLocation(),
        ConfiguredAspect.forAlias(real.getConfiguredAspect()),
        ImmutableList.<ActionAnalysisMetadata>of(),
        transitivePackages,
        removeActionsAfterEvaluation.get());
  }

  @Nullable
  private AspectValue createAspect(
      Environment env,
      AspectKey key,
      ImmutableList<Aspect> aspectPath,
      Aspect aspect,
      ConfiguredAspectFactory aspectFactory,
      ConfiguredTarget associatedTarget,
      BuildConfiguration aspectConfiguration,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      ToolchainContext toolchainContext,
      OrderedSetMultimap<Attribute, ConfiguredTarget> directDeps,
      NestedSetBuilder<Package> transitivePackages)
      throws AspectFunctionException, InterruptedException {

    SkyframeBuildView view = buildViewProvider.getSkyframeBuildView();

    StoredEventHandler events = new StoredEventHandler();
    CachingAnalysisEnvironment analysisEnvironment = view.createAnalysisEnvironment(
        key, false, events, env, aspectConfiguration);
    if (env.valuesMissing()) {
      return null;
    }

    ConfiguredAspect configuredAspect;
    if (ConfiguredTargetFunction.aspectMatchesConfiguredTarget(associatedTarget, aspect)) {
      configuredAspect =
          view.getConfiguredTargetFactory()
              .createAspect(
                  analysisEnvironment,
                  associatedTarget,
                  aspectPath,
                  aspectFactory,
                  aspect,
                  directDeps,
                  configConditions,
                  toolchainContext,
                  aspectConfiguration,
                  view.getHostConfiguration(aspectConfiguration));
    } else {
      configuredAspect = ConfiguredAspect.forNonapplicableTarget(aspect.getDescriptor());
    }

    events.replayOn(env.getListener());
    if (events.hasErrors()) {
      analysisEnvironment.disable(associatedTarget.getTarget());
      throw new AspectFunctionException(new AspectCreationException(
          "Analysis of target '" + associatedTarget.getLabel() + "' failed; build aborted"));
    }
    Preconditions.checkState(!analysisEnvironment.hasErrors(),
        "Analysis environment hasError() but no errors reported");

    if (env.valuesMissing()) {
      return null;
    }

    analysisEnvironment.disable(associatedTarget.getTarget());
    Preconditions.checkNotNull(configuredAspect);

    return new AspectValue(
        key,
        aspect,
        associatedTarget.getLabel(),
        associatedTarget.getTarget().getLocation(),
        configuredAspect,
        ImmutableList.copyOf(analysisEnvironment.getRegisteredActions()),
        transitivePackages.build(),
        removeActionsAfterEvaluation.get());
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    AspectKey aspectKey = (AspectKey) skyKey.argument();
    return Label.print(aspectKey.getLabel());
  }

  /**
   * An exception indicating that there was a problem creating an aspect.
   */
  public static final class AspectCreationException extends Exception {
    /** Targets in the transitive closure that failed to load. May be empty. */
    private final NestedSet<Label> loadingRootCauses;

    /**
     * The target for which analysis failed, if any. We can't represent aspects with labels, so if
     * the aspect analysis fails, this will be {@code null}.
     */
    @Nullable private final Label analysisRootCause;

    public AspectCreationException(String message, Label analysisRootCause) {
      super(message);
      this.loadingRootCauses = NestedSetBuilder.<Label>emptySet(Order.STABLE_ORDER);
      this.analysisRootCause = analysisRootCause;
    }

    public AspectCreationException(String message, NestedSet<Label> loadingRootCauses) {
      super(message);
      this.loadingRootCauses = loadingRootCauses;
      this.analysisRootCause = null;
    }

    public AspectCreationException(NestedSet<Label> loadingRootCauses) {
      this("Loading failed", loadingRootCauses);
    }

    public AspectCreationException(String message) {
      this(message, (Label) null);
    }

    public NestedSet<Label> getRootCauses() {
      return loadingRootCauses;
    }

    @Nullable public Label getAnalysisRootCause() {
      return analysisRootCause;
    }
  }

  /**
   * Used to indicate errors during the computation of an {@link AspectValue}.
   */
  private static final class AspectFunctionException extends SkyFunctionException {
    public AspectFunctionException(NoSuchThingException e) {
      super(e, Transience.PERSISTENT);
    }

    public AspectFunctionException(AspectCreationException e) {
      super(e, Transience.PERSISTENT);
    }

    public AspectFunctionException(InconsistentAspectOrderException cause) {
      super(cause, Transience.PERSISTENT);
    }
  }
}
