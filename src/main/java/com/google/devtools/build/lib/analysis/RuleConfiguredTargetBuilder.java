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
import static com.google.devtools.build.lib.packages.RuleClass.Builder.SKYLARK_BUILD_SETTING_DEFAULT_ATTR_NAME;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Actions.GeneratingActions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions.IncludeConfigFragmentsEnum;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.constraints.ConstraintSemantics;
import com.google.devtools.build.lib.analysis.constraints.EnvironmentCollection;
import com.google.devtools.build.lib.analysis.constraints.SupportedEnvironments;
import com.google.devtools.build.lib.analysis.constraints.SupportedEnvironmentsProvider;
import com.google.devtools.build.lib.analysis.constraints.SupportedEnvironmentsProvider.RemovedEnvironmentCulprit;
import com.google.devtools.build.lib.analysis.test.AnalysisTestActionBuilder;
import com.google.devtools.build.lib.analysis.test.AnalysisTestResultInfo;
import com.google.devtools.build.lib.analysis.test.ExecutionInfo;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.analysis.test.TestActionBuilder;
import com.google.devtools.build.lib.analysis.test.TestEnvironmentInfo;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.analysis.test.TestProvider.TestParams;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildSetting;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.Type.LabelClass;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Location;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * Builder class for analyzed rule instances.
 *
 * <p>This is used to tell Bazel which {@link TransitiveInfoProvider}s are produced by the analysis
 * of a configured target. For more information about analysis, see
 * {@link com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory}.
 *
 * @see com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory
 */
public final class RuleConfiguredTargetBuilder {
  private final RuleContext ruleContext;
  private final TransitiveInfoProviderMapBuilder providersBuilder =
      new TransitiveInfoProviderMapBuilder();
  private final Map<String, NestedSetBuilder<Artifact>> outputGroupBuilders = new TreeMap<>();
  private final ImmutableList.Builder<Artifact> additionalTestActionTools =
      new ImmutableList.Builder<>();

  /** These are supported by all configured targets and need to be specially handled. */
  private NestedSet<Artifact> filesToBuild = NestedSetBuilder.emptySet(Order.STABLE_ORDER);

  private NestedSetBuilder<Artifact> filesToRunBuilder = NestedSetBuilder.stableOrder();
  private RunfilesSupport runfilesSupport;
  private Runfiles persistentTestRunnerRunfiles;
  private Artifact executable;
  private ImmutableSet<ActionAnalysisMetadata> actionsWithoutExtraAction = ImmutableSet.of();

  public RuleConfiguredTargetBuilder(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
    add(LicensesProvider.class, LicensesProviderImpl.of(ruleContext));
    add(VisibilityProvider.class, new VisibilityProviderImpl(ruleContext.getVisibility()));
  }

  /**
   * Constructs the RuleConfiguredTarget instance based on the values set for this Builder.
   * Returns null if there were rule errors reported.
   */
  @Nullable
  public ConfiguredTarget build() throws ActionConflictException {
    // If allowing analysis failures, the current target may not propagate all of the
    // expected providers; be lenient on such cases (for example, avoid precondition checks).
    boolean allowAnalysisFailures = ruleContext.getConfiguration().allowAnalysisFailures();

    if (ruleContext.getConfiguration().enforceConstraints()) {
      checkConstraints();
    }
    if (ruleContext.hasErrors() && !allowAnalysisFailures) {
      return null;
    }

    maybeAddRequiredConfigFragmentsProvider();

    NestedSetBuilder<Artifact> runfilesMiddlemenBuilder = NestedSetBuilder.stableOrder();
    if (runfilesSupport != null) {
      runfilesMiddlemenBuilder.add(runfilesSupport.getRunfilesMiddleman());
      runfilesMiddlemenBuilder.addTransitive(runfilesSupport.getRunfiles().getExtraMiddlemen());
    }
    NestedSet<Artifact> runfilesMiddlemen = runfilesMiddlemenBuilder.build();
    FilesToRunProvider filesToRunProvider =
        new FilesToRunProvider(
            buildFilesToRun(runfilesMiddlemen, filesToBuild), runfilesSupport, executable);
    addProvider(new FileProvider(filesToBuild));
    addProvider(filesToRunProvider);

    if (runfilesSupport != null) {
      // If a binary is built, build its runfiles, too
      addOutputGroup(OutputGroupInfo.HIDDEN_TOP_LEVEL, runfilesMiddlemen);
    } else if (providersBuilder.contains(RunfilesProvider.class)) {
      // If we don't have a RunfilesSupport (probably because this is not a binary rule), we still
      // want to build the files this rule contributes to runfiles of dependent rules so that we
      // report an error if one of these is broken.
      //
      // Note that this is a best-effort thing: there is .getDataRunfiles() and all the language-
      // specific *RunfilesProvider classes, which we don't add here for reasons that are lost in
      // the mists of time.
      addOutputGroup(
          OutputGroupInfo.HIDDEN_TOP_LEVEL,
          providersBuilder
              .getProvider(RunfilesProvider.class)
              .getDefaultRunfiles()
              .getAllArtifacts());
    }

    collectTransitiveValidationOutputGroups();

    // Create test action and artifacts if target was successfully initialized
    // and is a test.
    if (TargetUtils.isTestRule(ruleContext.getTarget())) {
      if (runfilesSupport != null) {
        add(TestProvider.class, initializeTestProvider(filesToRunProvider));
      } else {
        if (!allowAnalysisFailures) {
          throw new IllegalStateException("Test rules must have runfiles");
        }
      }
    }

    ExtraActionArtifactsProvider extraActionsProvider =
        createExtraActionProvider(actionsWithoutExtraAction, ruleContext);
    add(ExtraActionArtifactsProvider.class, extraActionsProvider);

    if (!outputGroupBuilders.isEmpty()) {
      ImmutableMap.Builder<String, NestedSet<Artifact>> outputGroups = ImmutableMap.builder();
      for (Map.Entry<String, NestedSetBuilder<Artifact>> entry : outputGroupBuilders.entrySet()) {
        outputGroups.put(entry.getKey(), entry.getValue().build());
      }

      OutputGroupInfo outputGroupInfo = new OutputGroupInfo(outputGroups.build());
      addNativeDeclaredProvider(outputGroupInfo);
    }

    if (ruleContext.getConfiguration().evaluatingForAnalysisTest()) {
      if (ruleContext.getRule().isAnalysisTest()) {
        ruleContext.ruleError(
            String.format(
                "analysis_test rule '%s' cannot be transitively "
                    + "depended on by another analysis test rule",
                ruleContext.getLabel()));
        return null;
      }
      addProvider(new TransitiveLabelsInfo(transitiveLabels()));
    }

    if (ruleContext.getRule().hasAnalysisTestTransition()) {
      NestedSet<Label> labels = transitiveLabels();
      int depCount = labels.toList().size();
      if (depCount > ruleContext.getConfiguration().analysisTestingDepsLimit()) {
        ruleContext.ruleError(
            String.format(
                "analysis test rule excedeed maximum dependency edge count. "
                    + "Count: %s. Limit is %s. This limit is imposed on analysis test rules which "
                    + "use analysis_test_transition attribute transitions. Exceeding this limit "
                    + "indicates either the analysis_test has too many dependencies, or the "
                    + "underlying toolchains may be large. Try decreasing the number of test "
                    + "dependencies, (Analysis tests should not be very large!) or, if possible, "
                    + "try not using configuration transitions. If underlying toolchain size is "
                    + "to blame, it might be worth considering increasing "
                    + "--analysis_testing_deps_limit. (Beware, however, that large values of "
                    + "this flag can lead to no safeguards against giant "
                    + "test suites that can lead to Out Of Memory exceptions in the build server.)",
                depCount, ruleContext.getConfiguration().analysisTestingDepsLimit()));
        return null;
      }
    }

    if (ruleContext.getRule().isBuildSetting()) {
      BuildSetting buildSetting = ruleContext.getRule().getRuleClassObject().getBuildSetting();
      Object defaultValue =
          ruleContext
              .attributes()
              .get(SKYLARK_BUILD_SETTING_DEFAULT_ATTR_NAME, buildSetting.getType());
      addProvider(
          BuildSettingProvider.class,
          new BuildSettingProvider(buildSetting, defaultValue, ruleContext.getLabel()));
    }

    TransitiveInfoProviderMap providers = providersBuilder.build();

    if (ruleContext.getRule().isAnalysisTest()) {
      // If the target is an analysis test that returned AnalysisTestResultInfo, register a
      // test pass/fail action on behalf of the target.
      AnalysisTestResultInfo testResultInfo =
          providers.get(AnalysisTestResultInfo.SKYLARK_CONSTRUCTOR);

      if (testResultInfo == null) {
        ruleContext.ruleError(
            "rules with analysis_test=true must return an instance of AnalysisTestResultInfo");
        return null;
      }

      AnalysisTestActionBuilder.writeAnalysisTestAction(ruleContext, testResultInfo);
    }

    AnalysisEnvironment analysisEnvironment = ruleContext.getAnalysisEnvironment();
    GeneratingActions generatingActions =
        Actions.assignOwnersAndFilterSharedActionsAndThrowActionConflict(
            analysisEnvironment.getActionKeyContext(),
            analysisEnvironment.getRegisteredActions(),
            ruleContext.getOwner(),
            ((Rule) ruleContext.getTarget()).getOutputFiles());
    return new RuleConfiguredTarget(
        ruleContext,
        providers,
        generatingActions.getActions(),
        generatingActions.getArtifactsByOutputLabel());
  }

  /**
   * Adds {@link RequiredConfigFragmentsProvider} if {@link
   * CoreOptions#includeRequiredConfigFragmentsProvider} isn't {@link
   * CoreOptions.IncludeConfigFragmentsEnum#OFF}.
   *
   * <p>See {@link ConfiguredTargetFactory#getRequiredConfigFragments} for a description of the
   * meaning of this provider's content. That method populates {@link
   * RuleContext#getRequiredConfigFragments}, which we read here. We add to that any additional
   * config state that is only known to be required after the rule's analysis function has finished.
   * In particular, if the current rule is a {@code config_setting}, we add as a direct requirement
   * the config state that it matches.
   */
  private void maybeAddRequiredConfigFragmentsProvider() {
    if (ruleContext
            .getConfiguration()
            .getOptions()
            .get(CoreOptions.class)
            .includeRequiredConfigFragmentsProvider
        == IncludeConfigFragmentsEnum.OFF) {
      return;
    }

    ImmutableSet.Builder<String> requiredFragments = ImmutableSet.builder();
    requiredFragments.addAll(ruleContext.getRequiredConfigFragments());

    if (providersBuilder.contains(ConfigMatchingProvider.class)) {
      // config_setting discovers extra requirements through its "values = {'some_option': ...}"
      // references. Make sure those are included here.
      requiredFragments.addAll(
          providersBuilder.getProvider(ConfigMatchingProvider.class).getRequiredFragmentOptions());
    }

    if (ruleContext.getRule().isAttrDefined("feature_flags", BuildType.LABEL_KEYED_STRING_DICT)) {
      // Sad hack to make android_binary rules list the feature flags they set.
      // TODO(gregce): move this to AndroidBinary.java. This requires some more coordination to
      // make sure we don't add a RequiredConfigFragmentsProvider both there and here, which is
      // technically a violation of TransitiveInfoProviderMapBuilder.put.
      requiredFragments.addAll(
          ruleContext
              .attributes()
              .get("feature_flags", BuildType.LABEL_KEYED_STRING_DICT)
              .keySet()
              .stream()
              .map(label -> label.toString())
              .collect(Collectors.toList()));
    }

    addProvider(new RequiredConfigFragmentsProvider(requiredFragments.build()));
  }

  private NestedSet<Label> transitiveLabels() {
    NestedSetBuilder<Label> nestedSetBuilder = NestedSetBuilder.stableOrder();

    for (String attributeName : ruleContext.attributes().getAttributeNames()) {
      Type<?> attributeType =
          ruleContext.attributes().getAttributeDefinition(attributeName).getType();
      if (attributeType.getLabelClass() == LabelClass.DEPENDENCY) {
        for (TransitiveLabelsInfo labelsInfo :
            ruleContext.getPrerequisites(
                attributeName, TransitionMode.DONT_CHECK, TransitiveLabelsInfo.class)) {
          nestedSetBuilder.addTransitive(labelsInfo.getLabels());
        }
      }
    }
    nestedSetBuilder.add(ruleContext.getLabel());
    return nestedSetBuilder.build();
  }

  /**
   * Collects the validation action output groups from every dependency-type attribute on this rule.
   * This is done within {@link RuleConfiguredTargetBuilder} so that every rule always and
   * automatically propagates the validation action output group.
   *
   * <p>Note that in addition to {@link LabelClass.DEPENDENCY}, there is also {@link
   * LabelClass.FILESET_ENTRY}, however the fileset implementation takes care of propagating the
   * validation action output group itself.
   */
  private void collectTransitiveValidationOutputGroups() {

    for (String attributeName : ruleContext.attributes().getAttributeNames()) {

      Attribute attribute = ruleContext.attributes().getAttributeDefinition(attributeName);

      // Validation actions in the host configuration, or for tools, or from implicit deps should
      // not fail the overall build, since those dependencies should have their own builds
      // and tests that should surface any failing validations.
      if (!attribute.getTransitionFactory().isHost()
          && !attribute.getTransitionFactory().isTool()
          && !attribute.isImplicit()
          && attribute.getType().getLabelClass() == LabelClass.DEPENDENCY) {

        for (OutputGroupInfo outputGroup :
            ruleContext.getPrerequisites(
                attributeName, TransitionMode.DONT_CHECK, OutputGroupInfo.SKYLARK_CONSTRUCTOR)) {

          NestedSet<Artifact> validationArtifacts =
              outputGroup.getOutputGroup(OutputGroupInfo.VALIDATION);

          if (!validationArtifacts.isEmpty()) {
            addOutputGroup(OutputGroupInfo.VALIDATION, validationArtifacts);
          }
        }
      }
    }
  }

  /**
   * Compute the artifacts to put into the {@link FilesToRunProvider} for this target. These are the
   * filesToBuild, any artifacts added by the rule with {@link #addFilesToRun}, and the runfiles'
   * middlemen if they exists.
   */
  private NestedSet<Artifact> buildFilesToRun(
      NestedSet<Artifact> runfilesMiddlemen, NestedSet<Artifact> filesToBuild) {
    filesToRunBuilder.addTransitive(filesToBuild);
    filesToRunBuilder.addTransitive(runfilesMiddlemen);
    if (executable != null && ruleContext.getRule().getRuleClassObject().isSkylark()) {
      filesToRunBuilder.add(executable);
    }
    return filesToRunBuilder.build();
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
    ConstraintSemantics<RuleContext> constraintSemantics = ruleContext.getConstraintSemantics();
    EnvironmentCollection supportedEnvironments =
        constraintSemantics.getSupportedEnvironments(ruleContext);
    if (supportedEnvironments != null) {
      EnvironmentCollection.Builder refinedEnvironments = new EnvironmentCollection.Builder();
      Map<Label, RemovedEnvironmentCulprit> removedEnvironmentCulprits = new LinkedHashMap<>();
      constraintSemantics.checkConstraints(ruleContext, supportedEnvironments, refinedEnvironments,
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
            .setInstrumentedFiles(
                (InstrumentedFilesInfo)
                    providersBuilder.getProvider(
                        InstrumentedFilesInfo.SKYLARK_CONSTRUCTOR.getKey()));

    TestEnvironmentInfo environmentProvider =
        (TestEnvironmentInfo)
            providersBuilder.getProvider(TestEnvironmentInfo.PROVIDER.getKey());
    if (environmentProvider != null) {
      testActionBuilder.addExtraEnv(environmentProvider.getEnvironment());
    }

    TestParams testParams =
        testActionBuilder
            .setFilesToRunProvider(filesToRunProvider)
            .setPersistentTestRunnerRunfiles(persistentTestRunnerRunfiles)
            .addTools(additionalTestActionTools.build())
            .setExecutionRequirements(
                (ExecutionInfo) providersBuilder.getProvider(ExecutionInfo.PROVIDER.getKey()))
            .setShardCount(explicitShardCount)
            .build();
    ImmutableList<String> testTags = ImmutableList.copyOf(ruleContext.getRule().getRuleTags());
    return new TestProvider(testParams, testTags);
  }

  /**
   * Add files required to run the target. Artifacts from {@link #setFilesToBuild} and the runfiles
   * middleman, if any, are added automatically.
   */
  public RuleConfiguredTargetBuilder addFilesToRun(NestedSet<Artifact> files) {
    filesToRunBuilder.addTransitive(files);
    return this;
  }

  /** Add a specific provider. */
  public <T extends TransitiveInfoProvider> RuleConfiguredTargetBuilder addProvider(
      TransitiveInfoProvider provider) {
    providersBuilder.add(provider);
    return this;
  }

  /** Add a collection of specific providers. */
  public <T extends TransitiveInfoProvider> RuleConfiguredTargetBuilder addProviders(
      Iterable<TransitiveInfoProvider> providers) {
    providersBuilder.addAll(providers);
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
    return this;
  }

  /**
   * Add a Starlark transitive info. The provider value must be safe (i.e. a String, a Boolean, an
   * Integer, an Artifact, a Label, None, a Java TransitiveInfoProvider or something composed from
   * these in Starlark using lists, sets, structs or dicts). Otherwise an EvalException is thrown.
   */
  public RuleConfiguredTargetBuilder addSkylarkTransitiveInfo(
      String name, Object value, Location loc) throws EvalException {
    providersBuilder.put(name, value);
    return this;
  }

  /**
   * Adds a "declared provider" defined in Starlark to the rule. Use this method for declared
   * providers defined in Skyark.
   *
   * <p>Has special handling for {@link OutputGroupInfo}: that provider is not added from Skylark
   * directly, instead its output groups are added.
   *
   * <p>Use {@link #addNativeDeclaredProvider(Info)} in definitions of native rules.
   */
  public RuleConfiguredTargetBuilder addSkylarkDeclaredProvider(Info provider)
      throws EvalException {
    Provider constructor = provider.getProvider();
    if (!constructor.isExported()) {
      throw new EvalException(constructor.getLocation(),
          "All providers must be top level values");
    }
    if (OutputGroupInfo.SKYLARK_CONSTRUCTOR.getKey().equals(constructor.getKey())) {
      OutputGroupInfo outputGroupInfo = (OutputGroupInfo) provider;
      for (String outputGroup : outputGroupInfo) {
        addOutputGroup(outputGroup, outputGroupInfo.getOutputGroup(outputGroup));
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
   * <p>Use {@link #addSkylarkDeclaredProvider(Info)} for Starlark rule implementations.
   */
  public RuleConfiguredTargetBuilder addNativeDeclaredProviders(Iterable<Info> providers) {
    for (Info provider : providers) {
      addNativeDeclaredProvider(provider);
    }
    return this;
  }

  /**
   * Adds a "declared provider" defined in native code to the rule. Use this method for declared
   * providers in definitions of native rules.
   *
   * <p>Use {@link #addSkylarkDeclaredProvider(Info)} for Starlark rule implementations.
   */
  public RuleConfiguredTargetBuilder addNativeDeclaredProvider(Info provider) {
    Provider constructor = provider.getProvider();
    Preconditions.checkState(constructor.isExported());
    providersBuilder.put(provider);
    return this;
  }

  /**
   * Returns true if a provider matching the given provider key has already been added to the
   * configured target builder.
   */
  public boolean containsProviderKey(Provider.Key providerKey) {
    return providersBuilder.contains(providerKey);
  }

  /**
   * Returns true if a provider matching the given legacy key has already been added to the
   * configured target builder.
   */
  public boolean containsLegacyKey(String legacyId) {
    return providersBuilder.contains(legacyId);
  }

  /** Add a Starlark transitive info. The provider value must be safe. */
  public RuleConfiguredTargetBuilder addSkylarkTransitiveInfo(String name, Object value) {
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

  public RuleConfiguredTargetBuilder setPersistentTestRunnerRunfiles(Runfiles testSupportRunfiles) {
    this.persistentTestRunnerRunfiles = testSupportRunfiles;
    return this;
  }

  public RuleConfiguredTargetBuilder addTestActionTools(List<Artifact> tools) {
    this.additionalTestActionTools.addAll(tools);
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

  /**
   * Contains a nested set of transitive dependencies of the target which propagated this object.
   *
   * <p>This is automatically provided by all targets which are being evaluated in analysis testing.
   *
   * <p>For large builds, this object will become <i>very large</i>, but analysis tests are required
   * to be very small. The small-size of analysis tests are enforced by evaluating the size of this
   * object.
   */
  private static class TransitiveLabelsInfo implements TransitiveInfoProvider {
    private final NestedSet<Label> labels;

    public TransitiveLabelsInfo(NestedSet<Label> labels) {
      this.labels = labels;
    }

    public NestedSet<Label> getLabels() {
      return labels;
    }
  }
}
