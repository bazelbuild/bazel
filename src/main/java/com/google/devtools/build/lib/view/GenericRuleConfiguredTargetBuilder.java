// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.view;

import com.google.common.base.Preconditions;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.License;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.ExtraActionArtifactsProvider.ExtraArtifactSet;
import com.google.devtools.build.lib.view.GenericRuleConfiguredTarget.TransitiveInfo;
import com.google.devtools.build.lib.view.LicensesProvider.TargetLicense;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RunfilesCollector.State;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.lib.view.extra.ExtraActionMapProvider;
import com.google.devtools.build.lib.view.extra.ExtraActionSpec;
import com.google.devtools.build.lib.view.test.ExecutionRequirementProvider;
import com.google.devtools.build.lib.view.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.view.test.TestHelper;
import com.google.devtools.build.lib.view.test.TestProvider;
import com.google.devtools.build.lib.view.test.TestProvider.TestParams;

import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

/**
 * Builder class for analyzed rule instances (i.e., instances of {@link RuleConfiguredTarget}).
 */
public final class GenericRuleConfiguredTargetBuilder {
  private final RuleContext ruleContext;
  private final List<TransitiveInfo> infos = Lists.newArrayList();

  /** These are supported by all configured targets and need to be specially handled. */
  private NestedSet<Artifact> filesToBuild = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  private RunfilesSupport runfilesSupport;
  private Artifact executable;
  private ImmutableList<Artifact> mandatoryStampFiles;
  private ImmutableList<Action> extraActionPseudoActions;
  private ImmutableList<Artifact> extraActionPseudoArtifacts;

  public GenericRuleConfiguredTargetBuilder(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
  }

  /**
   * Constructs the RuleConfiguredTarget instance based on the values set for this Builder.
   */
  public RuleConfiguredTarget build() {
    if (ruleContext.hasErrors()) {
      return null;
    }
    add(FileProvider.class, new FileProviderImpl(ruleContext.getLabel(), filesToBuild));
    add(LicensesProvider.class, initializeLicensesProvider());
    // Create test action and artifacts if target was successfully initialized
    // and is a test.
    if (TargetUtils.isTestRule(ruleContext.getTarget())) {
      Preconditions.checkState(runfilesSupport != null);
      add(TestProvider.class, initializeTestProvider());
    }
    add(ExtraActionArtifactsProvider.class, initializeExtraActions());
    return new GenericRuleConfiguredTarget(
        ruleContext, runfilesSupport, executable, mandatoryStampFiles,
        infos.toArray(new TransitiveInfo[0]));
  }

  private TestProvider initializeTestProvider() {
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
    final TestParams testParams = new TestHelper(ruleContext)
        .setRunfilesSupport(runfilesSupport)
        .setFilesToRun(RuleContext.getFilesToRun(runfilesSupport, filesToBuild))
        .setInstrumentedFiles(findProvider(InstrumentedFilesProvider.class))
        .setExecutionRequirements(findProvider(ExecutionRequirementProvider.class))
        .setShardCount(explicitShardCount)
        .build();
    final ImmutableList<String> testTags =
        ImmutableList.copyOf(ruleContext.getRule().getRuleTags());
    return new TestProvider(testParams, testTags);
  }

  private LicensesProvider initializeLicensesProvider() {
    if (!ruleContext.getConfiguration().checkLicenses()) {
      return LicensesProviderImpl.EMPTY;
    }

    NestedSetBuilder<TargetLicense> builder = NestedSetBuilder.linkOrder();
    BuildConfiguration configuration = ruleContext.getConfiguration();
    Rule rule = ruleContext.getRule();
    if (configuration.isHostConfiguration() && rule.getToolOutputLicense() != null) {
      if (rule.getToolOutputLicense() != License.NO_LICENSE) {
        builder.add(new TargetLicense(rule.getLabel(), rule.getToolOutputLicense()));
      }
    } else {
      if (rule.getLicense() != License.NO_LICENSE) {
        builder.add(new TargetLicense(rule.getLabel(), rule.getLicense()));
      }

      for (TransitiveInfoCollection dep : ruleContext.getConfiguredTargetMap().values()) {
        LicensesProvider provider = dep.getProvider(LicensesProvider.class);
        if (provider != null) {
          builder.addTransitive(provider.getTransitiveLicenses());
        }
      }
    }

    return new LicensesProviderImpl(builder.build());
  }

  /**
   * Scans {@code action_listeners} associated with this build to see if any
   * {@code extra_actions} should be added to this configured target. If any
   * action_listeners are present, a partial visit of the artifact/action graph
   * is performed (for as long as actions found are owned by this {@link
   * ConfiguredTarget}). Any actions that match the {@code action_listener}
   * get an {@code extra_action} associated. The output artifacts of the
   * extra_action are reported to the {@link AnalysisEnvironment} for
   * bookkeeping.
   */
  private ExtraActionArtifactsProvider initializeExtraActions() {
    BuildConfiguration configuration = ruleContext.getConfiguration();
    if (configuration.isHostConfiguration()) {
      return ExtraActionArtifactsProvider.EMPTY;
    }

    ImmutableList<Artifact> extraActionArtifacts = ImmutableList.of();
    NestedSetBuilder<ExtraArtifactSet> builder = NestedSetBuilder.stableOrder();

    List<Label> actionListenerLabels = configuration.getActionListeners();
    if (!actionListenerLabels.isEmpty()
        && ruleContext.getRule().getAttributeDefinition(":action_listener") != null) {
      ExtraActionsVisitor visitor = new ExtraActionsVisitor(ruleContext,
          computeMnemonicsToExtraActionMap());

      Set<Artifact> outputs = new LinkedHashSet<>(ruleContext.getOutputArtifacts());
      Iterables.addAll(outputs, filesToBuild);
      visitor.visitWhiteNodes(outputs);

      if (extraActionPseudoArtifacts != null) {
        visitor.visitWhiteNodes(extraActionPseudoArtifacts);
      }

      if (extraActionPseudoActions != null) {
        visitor.visitBlackNodes(extraActionPseudoActions);
      }
      extraActionArtifacts = visitor.getAndResetExtraArtifacts();
      if (!extraActionArtifacts.isEmpty()) {
        builder.add(ExtraArtifactSet.of(ruleContext.getLabel(), extraActionArtifacts));
      }
    }

    // Add extra action artifacts from dependencies
    for (TransitiveInfoCollection dep : ruleContext.getConfiguredTargetMap().values()) {
      ExtraActionArtifactsProvider provider =
          dep.getProvider(ExtraActionArtifactsProvider.class);
      if (provider != null) {
        builder.addTransitive(provider.getTransitiveExtraActionArtifacts());
      }
    }

    if (mandatoryStampFiles != null && !mandatoryStampFiles.isEmpty()) {
      builder.add(ExtraArtifactSet.of(ruleContext.getLabel(), mandatoryStampFiles));
    }

    if (extraActionArtifacts.isEmpty() && builder.isEmpty()) {
      return ExtraActionArtifactsProvider.EMPTY;
    }
    return new ExtraActionArtifactsProvider(extraActionArtifacts, builder.build());
  }

  /**
   * Populates the configuration specific mnemonicToExtraActionMap
   * based on all action_listers selected by the user (via the blaze option
   * --experimental_action_listener=<target>).
   */
  private Multimap<String, ExtraActionSpec> computeMnemonicsToExtraActionMap() {
    // We copy the multimap here every time. This could be expensive.
    Multimap<String, ExtraActionSpec> mnemonicToExtraActionMap = HashMultimap.create();
    for (TransitiveInfoCollection actionListener :
        ruleContext.getPrerequisites(":action_listener", Mode.HOST)) {
      ExtraActionMapProvider provider = actionListener.getProvider(ExtraActionMapProvider.class);
      if (provider == null) {
        ruleContext.ruleError(String.format(
            "Unable to match experimental_action_listeners to this rule. "
            + "Specified target %s is not an action_listener rule",
            actionListener.getLabel().toString()));
      } else {
        mnemonicToExtraActionMap.putAll(provider.getExtraActionMap());
      }
    }
    return mnemonicToExtraActionMap;
  }

  private <T extends TransitiveInfoProvider> T findProvider(Class<T> clazz) {
    for (TransitiveInfo info : infos) {
      if (info.key.equals(clazz)) {
        return clazz.cast(info.value);
      }
    }
    return null;
  }

  /**
   * Add a specific provider with a given value.
   */
  public <T extends TransitiveInfoProvider> GenericRuleConfiguredTargetBuilder add(
      Class<T> key, T value) {
    AnalysisUtils.checkProvider(key);
    infos.add(new TransitiveInfo(key, value));
    return this;
  }

  /**
   * Add a specific provider with a given value.
   */
  public GenericRuleConfiguredTargetBuilder addProvider(
      Class<? extends TransitiveInfoProvider> key, TransitiveInfoProvider value) {
    AnalysisUtils.checkProvider(key);
    infos.add(new TransitiveInfo(key, value));
    return this;
  }

  /**
   * Add multiple providers with given values.
   */
  public GenericRuleConfiguredTargetBuilder addProviders(
      Map<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> providers) {
    for (Entry<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> provider :
        providers.entrySet()) {
      addProvider(provider.getKey(), provider.getValue());
    }
    return this;
  }

  /**
   * Set the runfiles support for executable targets.
   */
  public GenericRuleConfiguredTargetBuilder setRunfilesSupport(RunfilesSupport runfilesSupport) {
    this.runfilesSupport = runfilesSupport;
    return this;
  }

  /**
   * Set the files to build.
   */
  public GenericRuleConfiguredTargetBuilder setFilesToBuild(NestedSet<Artifact> filesToBuild) {
    this.filesToBuild = filesToBuild;
    return this;
  }

  /**
   * Set the baseline coverage Artifacts.
   */
  public GenericRuleConfiguredTargetBuilder setBaselineCoverageArtifacts(
      Collection<Artifact> artifacts) {
    return add(BaselineCoverageArtifactsProvider.class,
        new BaselineCoverageArtifactsProvider(ImmutableList.copyOf(artifacts)));
  }

  /**
   * Set the executable.
   */
  public GenericRuleConfiguredTargetBuilder setExecutable(Artifact executable) {
    this.executable = executable;
    return this;
  }

  /**
   * Set the mandatory stamp files.
   */
  public GenericRuleConfiguredTargetBuilder setMandatoryStampFiles(ImmutableList<Artifact> files) {
    this.mandatoryStampFiles = files;
    return this;
  }

  /**
   * Set the extra action pseudo actions.
   */
  public GenericRuleConfiguredTargetBuilder setExtraActionPseudoActions(
      ImmutableList<Action> actions) {
    this.extraActionPseudoActions = actions;
    return this;
  }

  /**
   * Set the extra action pseudo artifacts.
   */
  public GenericRuleConfiguredTargetBuilder setExtraActionPseudoArtifacts(
      ImmutableList<Artifact> artifacts) {
    this.extraActionPseudoArtifacts = artifacts;
    return this;
  }

  /**
   * A runfile provider returning the same set of runfiles for every state.
   */
  public static final class StatelessRunfilesProvider implements RunfilesProvider {
    private final Runfiles runfiles;

    public StatelessRunfilesProvider(Runfiles runfiles) {
      this.runfiles = runfiles;
    }

    @Override
    public Runfiles getTransitiveRunfiles(State state) {
      return runfiles;
    }
  }

  private static final class FileProviderImpl implements FileProvider {
    private final Label label;
    private final NestedSet<Artifact> filesToBuild;

    public FileProviderImpl(Label label, NestedSet<Artifact> filesToBuild) {
      this.label = label;
      this.filesToBuild = filesToBuild;
    }

    @Override
    public Label getLabel() {
      return label;
    }

    @Override
    public NestedSet<Artifact> getFilesToBuild() {
      return filesToBuild;
    }
  }
}
