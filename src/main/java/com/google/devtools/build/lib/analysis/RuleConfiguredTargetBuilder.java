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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.constraints.ConstraintSemantics;
import com.google.devtools.build.lib.analysis.constraints.EnvironmentCollection;
import com.google.devtools.build.lib.analysis.constraints.SupportedEnvironments;
import com.google.devtools.build.lib.analysis.constraints.SupportedEnvironmentsProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.ClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.rules.test.ExecutionInfoProvider;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.rules.test.TestActionBuilder;
import com.google.devtools.build.lib.rules.test.TestEnvironmentProvider;
import com.google.devtools.build.lib.rules.test.TestProvider;
import com.google.devtools.build.lib.rules.test.TestProvider.TestParams;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TreeMap;

/**
 * Builder class for analyzed rule instances.
 *
 * <p>This is used to tell Bazel which {@link TransitiveInfoProvider}s are produced by the analysis
 * of a configured target. For more information about analysis, see
 * {@link com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory}.
 *
 * @see com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory
 */
public final class RuleConfiguredTargetBuilder {
  private final RuleContext ruleContext;
  private final TransitiveInfoProviderMapBuilder providersBuilder =
      new TransitiveInfoProviderMapBuilder();
  private final Map<String, NestedSetBuilder<Artifact>> outputGroupBuilders = new TreeMap<>();

  /** These are supported by all configured targets and need to be specially handled. */
  private NestedSet<Artifact> filesToBuild = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  private RunfilesSupport runfilesSupport;
  private Artifact executable;
  private ImmutableSet<ActionAnalysisMetadata> actionsWithoutExtraAction = ImmutableSet.of();

  public RuleConfiguredTargetBuilder(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
    add(LicensesProvider.class, LicensesProviderImpl.of(ruleContext));
    add(VisibilityProvider.class, new VisibilityProviderImpl(ruleContext.getVisibility()));
  }

  /**
   * Constructs the RuleConfiguredTarget instance based on the values set for this Builder.
   */
  public ConfiguredTarget build() {
    if (ruleContext.getConfiguration().enforceConstraints()) {
      checkConstraints();
    }
    if (ruleContext.hasErrors()) {
      return null;
    }

    FilesToRunProvider filesToRunProvider = new FilesToRunProvider(
        getFilesToRun(runfilesSupport, filesToBuild), runfilesSupport, executable);
    addProvider(new FileProvider(filesToBuild));
    addProvider(filesToRunProvider);

    if (runfilesSupport != null) {
      // If a binary is built, build its runfiles, too
      addOutputGroup(
          OutputGroupProvider.HIDDEN_TOP_LEVEL, runfilesSupport.getRunfilesMiddleman());
    } else if (providersBuilder.contains(RunfilesProvider.class)) {
      // If we don't have a RunfilesSupport (probably because this is not a binary rule), we still
      // want to build the files this rule contributes to runfiles of dependent rules so that we
      // report an error if one of these is broken.
      //
      // Note that this is a best-effort thing: there is .getDataRunfiles() and all the language-
      // specific *RunfilesProvider classes, which we don't add here for reasons that are lost in
      // the mists of time.
      addOutputGroup(
          OutputGroupProvider.HIDDEN_TOP_LEVEL,
          providersBuilder
              .getProvider(RunfilesProvider.class)
              .getDefaultRunfiles()
              .getAllArtifacts());
    }

    // Create test action and artifacts if target was successfully initialized
    // and is a test.
    if (TargetUtils.isTestRule(ruleContext.getTarget())) {
      Preconditions.checkState(runfilesSupport != null);
      add(TestProvider.class, initializeTestProvider(filesToRunProvider));
    }

    ExtraActionArtifactsProvider extraActionsProvider =
        createExtraActionProvider(actionsWithoutExtraAction, ruleContext);
    add(ExtraActionArtifactsProvider.class, extraActionsProvider);

    if (!outputGroupBuilders.isEmpty()) {
      ImmutableMap.Builder<String, NestedSet<Artifact>> outputGroups = ImmutableMap.builder();
      for (Map.Entry<String, NestedSetBuilder<Artifact>> entry : outputGroupBuilders.entrySet()) {
        outputGroups.put(entry.getKey(), entry.getValue().build());
      }

      OutputGroupProvider outputGroupProvider = new OutputGroupProvider(outputGroups.build());
      addProvider(OutputGroupProvider.class, outputGroupProvider);
    }

    TransitiveInfoProviderMap providers = providersBuilder.build();

    return new RuleConfiguredTarget(ruleContext, providers);
  }


  /**
   * Like getFilesToBuild(), except that it also includes the runfiles middleman, if any. Middlemen
   * are expanded in the SpawnStrategy or by the Distributor.
   */
  private NestedSet<Artifact> getFilesToRun(
      RunfilesSupport runfilesSupport, NestedSet<Artifact> filesToBuild) {
    if (runfilesSupport == null) {
      return filesToBuild;
    } else {
      NestedSetBuilder<Artifact> allFilesToBuild = NestedSetBuilder.stableOrder();
      allFilesToBuild.addTransitive(filesToBuild);
      allFilesToBuild.add(runfilesSupport.getRunfilesMiddleman());
      return allFilesToBuild.build();
    }
  }

  /**
   * Invokes Blaze's constraint enforcement system: checks that this rule's dependencies
   * support its environments and reports appropriate errors if violations are found. Also
   * publishes this rule's supported environments for the rules that depend on it.
   */
  private void checkConstraints() {
    if (!ruleContext.getRule().getRuleClassObject().supportsConstraintChecking()) {
      return;
    }
    EnvironmentCollection supportedEnvironments =
        ConstraintSemantics.getSupportedEnvironments(ruleContext);
    if (supportedEnvironments != null) {
      EnvironmentCollection.Builder refinedEnvironments = new EnvironmentCollection.Builder();
      Map<Label, Target> removedEnvironmentCulprits = new LinkedHashMap<>();
      ConstraintSemantics.checkConstraints(ruleContext, supportedEnvironments, refinedEnvironments,
          removedEnvironmentCulprits);
      add(SupportedEnvironmentsProvider.class,
          new SupportedEnvironments(supportedEnvironments, refinedEnvironments.build(),
              removedEnvironmentCulprits));
    }
  }

  private TestProvider initializeTestProvider(FilesToRunProvider filesToRunProvider) {
    int explicitShardCount = ruleContext.attributes().get("shard_count", Type.INTEGER);
    if (explicitShardCount < 0
        && ruleContext.getRule().isAttributeValueExplicitlySpecified("shard_count")) {
      ruleContext.attributeError("shard_count", "Must not be negative.");
    }
    if (explicitShardCount > 50) {
      ruleContext.attributeError("shard_count",
          "Having more than 50 shards is indicative of poor test organization. "
          + "Please reduce the number of shards.");
    }
    TestActionBuilder testActionBuilder =
        new TestActionBuilder(ruleContext)
            .setInstrumentedFiles(providersBuilder.getProvider(InstrumentedFilesProvider.class));

    TestEnvironmentProvider environmentProvider =
        (TestEnvironmentProvider)
            providersBuilder.getProvider(TestEnvironmentProvider.SKYLARK_CONSTRUCTOR.getKey());
    if (environmentProvider != null) {
      testActionBuilder.addExtraEnv(environmentProvider.getEnvironment());
    }

    TestParams testParams =
        testActionBuilder
            .setFilesToRunProvider(filesToRunProvider)
            .setExecutionRequirements(
                (ExecutionInfoProvider) providersBuilder
                    .getProvider(ExecutionInfoProvider.SKYLARK_CONSTRUCTOR.getKey()))
            .setShardCount(explicitShardCount)
            .build();
    ImmutableList<String> testTags = ImmutableList.copyOf(ruleContext.getRule().getRuleTags());
    return new TestProvider(testParams, testTags);
  }

  /** Add a specific provider. */
  public <T extends TransitiveInfoProvider> RuleConfiguredTargetBuilder addProvider(
      TransitiveInfoProvider provider) {
    providersBuilder.add(provider);
    maybeAddSkylarkProvider(provider);
    return this;
  }

  /** Add a collection of specific providers. */
  public <T extends TransitiveInfoProvider> RuleConfiguredTargetBuilder addProviders(
      Iterable<TransitiveInfoProvider> providers) {
    providersBuilder.addAll(providers);
    for (TransitiveInfoProvider provider : providers) {
      maybeAddSkylarkProvider(provider);
    }
    return this;
  }

  /** Add a collection of specific providers. */
  public <T extends TransitiveInfoProvider> RuleConfiguredTargetBuilder addProviders(
      TransitiveInfoProviderMap providers) {
    providersBuilder.addAll(providers);
    return this;
  }


  /**
   * Add a specific provider with a given value.
   *
   * @deprecated use {@link #addProvider}
   */
  @Deprecated
  public <T extends TransitiveInfoProvider> RuleConfiguredTargetBuilder add(Class<T> key, T value) {
    return addProvider(key, value);
  }

  /** Add a specific provider with a given value. */
  public <T extends TransitiveInfoProvider> RuleConfiguredTargetBuilder addProvider(
      Class<? extends T> key, T value) {
    Preconditions.checkNotNull(key);
    Preconditions.checkNotNull(value);
    providersBuilder.put(key, value);
    maybeAddSkylarkProvider(value);
    return this;
  }

  protected <T extends TransitiveInfoProvider> void maybeAddSkylarkProvider(T value) {
    if (value instanceof TransitiveInfoProvider.WithLegacySkylarkName) {
      addSkylarkTransitiveInfo(
          ((TransitiveInfoProvider.WithLegacySkylarkName) value).getSkylarkName(),
          value);
    }
  }

  /**
   * Add a Skylark transitive info. The provider value must be safe (i.e. a String, a Boolean,
   * an Integer, an Artifact, a Label, None, a Java TransitiveInfoProvider or something composed
   * from these in Skylark using lists, sets, structs or dicts). Otherwise an EvalException is
   * thrown.
   */
  public RuleConfiguredTargetBuilder addSkylarkTransitiveInfo(
      String name, Object value, Location loc) throws EvalException {
    providersBuilder.put(name, value);
    return this;
  }

  /**
   * Adds a "declared provider" defined in Skylark to the rule.
   * Use this method for declared providers defined in Skyark.
   *
   * Has special handling for {@link OutputGroupProvider}: that provider is not added
   * from Skylark directly, instead its outpuyt groups are added.
   *
   * Use {@link #addNativeDeclaredProvider(SkylarkClassObject)} in definitions of
   * native rules.
   */
  public RuleConfiguredTargetBuilder addSkylarkDeclaredProvider(
      SkylarkClassObject provider, Location loc) throws EvalException {
    ClassObjectConstructor constructor = provider.getConstructor();
    if (!constructor.isExported()) {
      throw new EvalException(constructor.getLocation(),
          "All providers must be top level values");
    }
    if (OutputGroupProvider.SKYLARK_CONSTRUCTOR.getKey().equals(constructor.getKey())) {
      OutputGroupProvider outputGroupProvider = (OutputGroupProvider) provider;
      for (String outputGroup : outputGroupProvider) {
        addOutputGroup(outputGroup, outputGroupProvider.getOutputGroup(outputGroup));
      }
    } else {
      providersBuilder.put(provider);
    }
    return this;
  }

  /**
   * Adds "declared providers" defined in native code to the rule. Use this method for declared
   * providers in definitions of native rules.
   *
   * <p>Use {@link #addSkylarkDeclaredProvider(SkylarkClassObject, Location)} for Skylark rule
   * implementations.
   */
  public RuleConfiguredTargetBuilder addNativeDeclaredProviders(
      Iterable<SkylarkClassObject> providers) {
    for (SkylarkClassObject provider : providers) {
      addNativeDeclaredProvider(provider);
    }
    return this;
  }

  /**
   * Adds a "declared provider" defined in native code to the rule.
   * Use this method for declared providers in definitions of native rules.
   *
   * Use {@link #addSkylarkDeclaredProvider(SkylarkClassObject, Location)}
   * for Skylark rule implementations.
   */
  public RuleConfiguredTargetBuilder addNativeDeclaredProvider(SkylarkClassObject provider) {
    ClassObjectConstructor constructor = provider.getConstructor();
    Preconditions.checkState(constructor.isExported());
    providersBuilder.put(provider);
    return this;
  }

  /**
   * Add a Skylark transitive info. The provider value must be safe.
   */
  public RuleConfiguredTargetBuilder addSkylarkTransitiveInfo(
      String name, Object value) {
    providersBuilder.put(name, value);
    return this;
  }

  /**
   * Set the runfiles support for executable targets.
   */
  public RuleConfiguredTargetBuilder setRunfilesSupport(
      RunfilesSupport runfilesSupport, Artifact executable) {
    this.runfilesSupport = runfilesSupport;
    this.executable = executable;
    return this;
  }

  /**
   * Set the files to build.
   */
  public RuleConfiguredTargetBuilder setFilesToBuild(NestedSet<Artifact> filesToBuild) {
    this.filesToBuild = filesToBuild;
    return this;
  }

  private NestedSetBuilder<Artifact> getOutputGroupBuilder(String name) {
    NestedSetBuilder<Artifact> result = outputGroupBuilders.get(name);
    if (result != null) {
      return result;
    }

    result = NestedSetBuilder.stableOrder();
    outputGroupBuilders.put(name, result);
    return result;
  }

  /**
   * Adds a set of files to an output group.
   */
  public RuleConfiguredTargetBuilder addOutputGroup(String name, NestedSet<Artifact> artifacts) {
    getOutputGroupBuilder(name).addTransitive(artifacts);
    return this;
  }

  /**
   * Adds a file to an output group.
   */
  public RuleConfiguredTargetBuilder addOutputGroup(String name, Artifact artifact) {
    getOutputGroupBuilder(name).add(artifact);
    return this;
  }

  /**
   * Adds multiple output groups.
   */
  public RuleConfiguredTargetBuilder addOutputGroups(Map<String, NestedSet<Artifact>> groups) {
    for (Map.Entry<String, NestedSet<Artifact>> group : groups.entrySet()) {
      getOutputGroupBuilder(group.getKey()).addTransitive(group.getValue());
    }

    return this;
  }

  /**
   * Set the extra action pseudo actions.
   */
  public RuleConfiguredTargetBuilder setActionsWithoutExtraAction(
      ImmutableSet<ActionAnalysisMetadata> actions) {
    this.actionsWithoutExtraAction = actions;
    return this;
  }
}
