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

import static com.google.devtools.build.lib.analysis.ToolchainCollection.DEFAULT_EXEC_GROUP_NAME;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimaps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionRegistry;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.AliasProvider.TargetMode;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoKey;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions.IncludeConfigFragmentsEnum;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentCollection;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.constraints.ConstraintSemantics;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.stringtemplate.TemplateContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.ImmutableSortedKeyListMultimap;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.ConfigurationFragmentPolicy;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.FileTarget;
import com.google.devtools.build.lib.packages.FilesetEntry;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.PackageSpecification.PackageGroupContents;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.RequiredProviders;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.SymbolGenerator;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.Type.LabelClass;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.server.FailureDetails.Analysis;
import com.google.devtools.build.lib.server.FailureDetails.Analysis.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.SaneAnalysisException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.syntax.Location;

/**
 * The totality of data available during the analysis of a rule.
 *
 * <p>These objects should not outlast the analysis phase. Do not pass them to {@link
 * com.google.devtools.build.lib.actions.Action} objects or other persistent objects. There are
 * internal tests to ensure that RuleContext objects are not persisted that check the name of this
 * class, so update those tests if you change this class's name.
 *
 * @see com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory
 */
public final class RuleContext extends TargetContext
    implements ActionConstructionContext, ActionRegistry, RuleErrorConsumer {

  public boolean isAllowTagsPropagation() throws InterruptedException {
    return this.getAnalysisEnvironment()
        .getStarlarkSemantics()
        .getBool(BuildLanguageOptions.EXPERIMENTAL_ALLOW_TAGS_PROPAGATION);
  }

  /** Custom dependency validation logic. */
  public interface PrerequisiteValidator {
    /**
     * Checks whether the rule in {@code contextBuilder} is allowed to depend on {@code
     * prerequisite} through the attribute {@code attribute}.
     *
     * <p>Can be used for enforcing any organization-specific policies about the layout of the
     * workspace.
     */
    void validate(
        Builder contextBuilder, ConfiguredTargetAndData prerequisite, Attribute attribute);
  }

  /** The configured version of FilesetEntry. */
  @Immutable
  public static final class ConfiguredFilesetEntry {
    private final FilesetEntry entry;
    private final TransitiveInfoCollection src;
    private final ImmutableList<TransitiveInfoCollection> files;

    ConfiguredFilesetEntry(FilesetEntry entry, TransitiveInfoCollection src) {
      this.entry = entry;
      this.src = src;
      this.files = null;
    }

    ConfiguredFilesetEntry(FilesetEntry entry, ImmutableList<TransitiveInfoCollection> files) {
      this.entry = entry;
      this.src = null;
      this.files = files;
    }

    public FilesetEntry getEntry() {
      return entry;
    }

    public TransitiveInfoCollection getSrc() {
      return src;
    }

    /**
     * Targets from FilesetEntry.files, or null if the user omitted it.
     */
    @Nullable
    public ImmutableList<TransitiveInfoCollection> getFiles() {
      return files;
    }
  }

  private static final String HOST_CONFIGURATION_PROGRESS_TAG = "for host";

  private final Rule rule;
  /**
   * A list of all aspects applied to the target. If this <code>RuleContext</code>
   * is for a rule implementation, <code>aspects</code> is an empty list.
   *
   * Otherwise, the last aspect in <code>aspects</code> list is the aspect which
   * this <code>RuleCointext</code> is for.
   */
  private final ImmutableList<Aspect> aspects;
  private final ImmutableList<AspectDescriptor> aspectDescriptors;
  private final ListMultimap<String, ConfiguredTargetAndData> targetMap;
  private final ListMultimap<String, ConfiguredFilesetEntry> filesetEntryMap;
  private final ImmutableMap<Label, ConfigMatchingProvider> configConditions;
  private final AspectAwareAttributeMapper attributes;
  private final ImmutableSet<String> enabledFeatures;
  private final ImmutableSet<String> disabledFeatures;
  private final String ruleClassNameForLogging;
  private final BuildConfiguration hostConfiguration;
  private final ConfigurationFragmentPolicy configurationFragmentPolicy;
  private final ImmutableList<Class<? extends Fragment>> universalFragments;
  private final RuleErrorConsumer reporter;
  @Nullable private final ToolchainCollection<ResolvedToolchainContext> toolchainContexts;
  private final ConstraintSemantics<RuleContext> constraintSemantics;
  private final ImmutableSet<String> requiredConfigFragments;
  private final List<Expander> makeVariableExpanders = new ArrayList<>();
  private final ImmutableMap<String, ImmutableMap<String, String>> execProperties;

  /** Map of exec group names to ActionOwners. */
  private final Map<String, ActionOwner> actionOwners = new HashMap<>();

  private final SymbolGenerator<ActionLookupKey> actionOwnerSymbolGenerator;

  /* lazily computed cache for Make variables, computed from the above. See get... method */
  private transient ConfigurationMakeVariableContext configurationMakeVariableContext = null;

  private RuleContext(
      Builder builder,
      AttributeMap attributes,
      ListMultimap<String, ConfiguredTargetAndData> targetMap,
      ListMultimap<String, ConfiguredFilesetEntry> filesetEntryMap,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      ImmutableList<Class<? extends Fragment>> universalFragments,
      String ruleClassNameForLogging,
      ActionLookupKey actionLookupKey,
      ImmutableMap<String, Attribute> aspectAttributes,
      @Nullable ToolchainCollection<ResolvedToolchainContext> toolchainContexts,
      ConstraintSemantics<RuleContext> constraintSemantics,
      ImmutableSet<String> requiredConfigFragments)
      throws InvalidExecGroupException {
    super(
        builder.env,
        builder.target.getAssociatedRule(),
        builder.configuration,
        builder.prerequisiteMap.get(null),
        builder.visibility);
    this.rule = builder.target.getAssociatedRule();
    this.aspects = builder.aspects;
    this.aspectDescriptors =
        builder
            .aspects
            .stream()
            .map(a -> a.getDescriptor())
            .collect(ImmutableList.toImmutableList());
    this.configurationFragmentPolicy = builder.configurationFragmentPolicy;
    this.universalFragments = universalFragments;
    this.targetMap = targetMap;
    this.filesetEntryMap = filesetEntryMap;
    this.configConditions = configConditions;
    this.attributes = new AspectAwareAttributeMapper(attributes, aspectAttributes);
    Set<String> allEnabledFeatures = new HashSet<>();
    Set<String> allDisabledFeatures = new HashSet<>();
    getAllFeatures(allEnabledFeatures, allDisabledFeatures);
    this.enabledFeatures = ImmutableSortedSet.copyOf(allEnabledFeatures);
    this.disabledFeatures = ImmutableSortedSet.copyOf(allDisabledFeatures);
    this.ruleClassNameForLogging = ruleClassNameForLogging;
    this.hostConfiguration = builder.hostConfiguration;
    this.actionOwnerSymbolGenerator = new SymbolGenerator<>(actionLookupKey);
    reporter = builder.reporter;
    this.toolchainContexts = toolchainContexts;
    this.execProperties = parseExecProperties(builder.rawExecProperties);
    this.constraintSemantics = constraintSemantics;
    this.requiredConfigFragments = requiredConfigFragments;
  }

  private void getAllFeatures(Set<String> allEnabledFeatures, Set<String> allDisabledFeatures) {
    Set<String> globallyEnabled = new HashSet<>();
    Set<String> globallyDisabled = new HashSet<>();
    parseFeatures(getConfiguration().getDefaultFeatures(), globallyEnabled, globallyDisabled);
    if (getConfiguration().getFatApkSplitSanitizer().feature != null) {
      globallyEnabled.add(getConfiguration().getFatApkSplitSanitizer().feature);
    }
    Set<String> packageEnabled = new HashSet<>();
    Set<String> packageDisabled = new HashSet<>();
    parseFeatures(getRule().getPackage().getFeatures(), packageEnabled, packageDisabled);
    Set<String> ruleEnabled = new HashSet<>();
    Set<String> ruleDisabled = new HashSet<>();
    if (attributes().has("features", Type.STRING_LIST)) {
      parseFeatures(attributes().get("features", Type.STRING_LIST), ruleEnabled, ruleDisabled);
    }

    Set<String> ruleDisabledFeatures =
        Sets.union(ruleDisabled, Sets.difference(packageDisabled, ruleEnabled));
    allDisabledFeatures.addAll(Sets.union(ruleDisabledFeatures, globallyDisabled));

    Set<String> packageFeatures =
        Sets.difference(Sets.union(globallyEnabled, packageEnabled), packageDisabled);
    Set<String> ruleFeatures =
        Sets.difference(Sets.union(packageFeatures, ruleEnabled), ruleDisabled);
    allEnabledFeatures.addAll(Sets.difference(ruleFeatures, globallyDisabled));
  }

  private void parseFeatures(Iterable<String> features, Set<String> enabled, Set<String> disabled) {
    for (String feature : features) {
      if (feature.startsWith("-")) {
        disabled.add(feature.substring(1));
      } else if (feature.equals("no_layering_check")) {
        // TODO(bazel-team): Remove once we do not have BUILD files left that contain
        // 'no_layering_check'.
        disabled.add(feature.substring(3));
      } else {
        enabled.add(feature);
      }
    }
  }

  public RepositoryName getRepository() {
    return rule.getRepository();
  }

  @Override
  public ArtifactRoot getBinDirectory() {
    return getConfiguration().getBinDirectory(getLabel().getRepository());
  }

  public ArtifactRoot getIncludeDirectory() {
    return getConfiguration().getIncludeDirectory(getLabel().getRepository());
  }

  public ArtifactRoot getGenfilesDirectory() {
    return getConfiguration().getGenfilesDirectory(getLabel().getRepository());
  }

  public ArtifactRoot getCoverageMetadataDirectory() {
    return getConfiguration().getCoverageMetadataDirectory(getLabel().getRepository());
  }

  public ArtifactRoot getTestLogsDirectory() {
    return getConfiguration().getTestLogsDirectory(getLabel().getRepository());
  }

  public PathFragment getBinFragment() {
    return getConfiguration().getBinFragment(getLabel().getRepository());
  }

  public PathFragment getGenfilesFragment() {
    return getConfiguration().getGenfilesFragment(getLabel().getRepository());
  }

  @Override
  public ArtifactRoot getMiddlemanDirectory() {
    return getConfiguration().getMiddlemanDirectory(getLabel().getRepository());
  }

  public Rule getRule() {
    return rule;
  }

  public ImmutableList<Aspect> getAspects() {
    return aspects;
  }

  /**
   * If this target's configuration suppresses analysis failures, this returns a list
   * of strings, where each string corresponds to a description of an error that occurred during
   * processing this target.
   *
   * @throws IllegalStateException if this target's configuration does not suppress analysis
   *     failures (if {@code getConfiguration().allowAnalysisFailures()} is false)
   */
  public List<String> getSuppressedErrorMessages() {
    Preconditions.checkState(getConfiguration().allowAnalysisFailures(),
        "Error messages can only be retrieved via RuleContext if allow_analysis_failures is true");
    Preconditions.checkState(reporter instanceof SuppressingErrorReporter,
        "Unexpected error reporter");
    return ((SuppressingErrorReporter) reporter).getErrorMessages();
  }

  /**
   * If this <code>RuleContext</code> is for an aspect implementation, returns that aspect.
   * (it is the last aspect in the list of aspects applied to a target; all other aspects
   * are the ones main aspect sees as specified by its "required_aspect_providers")
   * Otherwise returns <code>null</code>.
   */
  @Nullable
  public Aspect getMainAspect() {
    return aspects.isEmpty() ? null : aspects.get(aspects.size() - 1);
  }

  /**
   * Returns a rule class name suitable for log messages, including an aspect name if applicable.
   */
  public String getRuleClassNameForLogging() {
    return ruleClassNameForLogging;
  }

  /**
   * Returns the workspace name for the rule.
   */
  public String getWorkspaceName() {
    return rule.getRepository().strippedName();
  }

  /**
   * The configuration conditions that trigger this rule's configurable attributes.
   */
  public ImmutableMap<Label, ConfigMatchingProvider> getConfigConditions() {
    return configConditions;
  }

  /**
   * Returns the host configuration for this rule.
   */
  public BuildConfiguration getHostConfiguration() {
    return hostConfiguration;
  }

  /**
   * All aspects applied to the rule.
   */
  public ImmutableList<AspectDescriptor> getAspectDescriptors() {
    return aspectDescriptors;
  }

  /**
   * Accessor for the attributes of the rule and its aspects.
   *
   * <p>The rule's native attributes can be queried both on their structure / existence and values
   * Aspect attributes can only be queried on their structure.
   *
   * <p>This should be the sole interface for reading rule/aspect attributes in {@link RuleContext}.
   * Don't expose other access points through new public methods.
   */
  public AttributeMap attributes() {
    return attributes;
  }

  @Override
  public boolean hasErrors() {
    return reporter.hasErrors();
  }

  /**
   * Returns an immutable map from attribute name to list of configured targets for that attribute.
   */
  public ListMultimap<String, ? extends TransitiveInfoCollection> getConfiguredTargetMap() {
    return Multimaps.transformValues(targetMap, ConfiguredTargetAndData::getConfiguredTarget);
  }

  /**
   * Returns an immutable map from attribute name to list of {@link ConfiguredTargetAndData} objects
   * for that attribute.
   */
  public ListMultimap<String, ConfiguredTargetAndData> getConfiguredTargetAndDataMap() {
    return targetMap;
  }

  /** Returns the {@link ConfiguredTargetAndData} the given attribute. */
  public List<ConfiguredTargetAndData> getPrerequisiteConfiguredTargets(String attributeName) {
    return targetMap.get(attributeName);
  }

  /**
   * Returns an immutable map from attribute name to list of fileset entries.
   */
  public ListMultimap<String, ConfiguredFilesetEntry> getFilesetEntryMap() {
    return filesetEntryMap;
  }

  @Override
  public ActionOwner getActionOwner() {
    return getActionOwner(DEFAULT_EXEC_GROUP_NAME);
  }

  @Override
  @Nullable
  public ActionOwner getActionOwner(String execGroup) {
    if (actionOwners.containsKey(execGroup)) {
      return actionOwners.get(execGroup);
    }
    if (toolchainContexts != null && !toolchainContexts.hasToolchainContext(execGroup)) {
      return null;
    }
    ActionOwner actionOwner =
        createActionOwner(
            rule,
            aspectDescriptors,
            getConfiguration(),
            getExecProperties(execGroup, execProperties),
            getExecutionPlatform(execGroup));
    actionOwners.put(execGroup, actionOwner);
    return actionOwner;
  }

  /**
   * We have to re-implement this method here because it is declared in the interface {@link
   * ActionConstructionContext}. This class inherits from {@link TargetContext} which doesn't
   * implement the {@link ActionConstructionContext} interface.
   */
  @Override
  public ActionKeyContext getActionKeyContext() {
    return super.getActionKeyContext();
  }

  /**
   * An opaque symbol generator to be used when identifying objects by their action owner/index of
   * creation. Only needed if an object needs to know whether it was created by the same action
   * owner in the same order as another object. Each symbol must call {@link
   * SymbolGenerator#generate} separately to obtain a unique object.
   */
  public SymbolGenerator<?> getSymbolGenerator() {
    return actionOwnerSymbolGenerator;
  }

  /**
   * Returns a configuration fragment for this this target.
   */
  @Nullable
  public <T extends Fragment> T getFragment(Class<T> fragment, ConfigurationTransition transition) {
    return getFragment(fragment, fragment.getSimpleName(), "", transition);
  }

  @Nullable
  <T extends Fragment> T getFragment(
      Class<T> fragment,
      String name,
      String additionalErrorMessage,
      ConfigurationTransition transition) {
    // TODO(bazel-team): The fragments can also be accessed directly through BuildConfiguration.
    // Can we lock that down somehow?
    Preconditions.checkArgument(isLegalFragment(fragment, transition),
        "%s has to declare '%s' as a required fragment "
        + "in %s configuration in order to access it.%s",
        getRuleClassNameForLogging(), name, FragmentCollection.getConfigurationName(transition),
        additionalErrorMessage);
    return getConfiguration(transition).getFragment(fragment);
  }

  @Nullable
  public <T extends Fragment> T getFragment(Class<T> fragment) {
    // No transition means target configuration.
    return getFragment(fragment, NoTransition.INSTANCE);
  }

  @Nullable
  public Fragment getStarlarkFragment(String name, ConfigurationTransition transition)
      throws EvalException {
    Class<? extends Fragment> fragmentClass =
        getConfiguration(transition).getStarlarkFragmentByName(name);
    if (fragmentClass == null) {
      return null;
    }
    try {
      return getFragment(
          fragmentClass,
          name,
          String.format(
              " Please update the '%1$sfragments' argument of the rule definition "
                  + "(for example: %1$sfragments = [\"%2$s\"])",
              transition.isHostTransition() ? "host_" : "", name),
          transition);
    } catch (IllegalArgumentException ex) { // fishy
      throw new EvalException(ex.getMessage());
    }
  }

  public ImmutableCollection<String> getStarlarkFragmentNames(ConfigurationTransition transition) {
    return getConfiguration(transition).getStarlarkFragmentNames();
  }

  public <T extends Fragment> boolean isLegalFragment(
      Class<T> fragment, ConfigurationTransition transition) {
    return universalFragments.contains(fragment)
        || configurationFragmentPolicy.isLegalConfigurationFragment(fragment, transition);
  }

  public <T extends Fragment> boolean isLegalFragment(Class<T> fragment) {
    // No transition means target configuration.
    return isLegalFragment(fragment, NoTransition.INSTANCE);
  }

  private BuildConfiguration getConfiguration(ConfigurationTransition transition) {
    return transition.isHostTransition() ? hostConfiguration : getConfiguration();
  }

  @Override
  public ActionLookupKey getOwner() {
    return getAnalysisEnvironment().getOwner();
  }

  public ImmutableList<Artifact> getBuildInfo(BuildInfoKey key) throws InterruptedException {
    return getAnalysisEnvironment()
        .getBuildInfo(
            AnalysisUtils.isStampingEnabled(this, getConfiguration()), key, getConfiguration());
  }

  private static ImmutableMap<String, String> computeExecProperties(
      Map<String, String> targetExecProperties, @Nullable PlatformInfo executionPlatform) {
    Map<String, String> execProperties = new HashMap<>();

    if (executionPlatform != null) {
      execProperties.putAll(executionPlatform.execProperties());
    }

    // If the same key occurs both in the platform and in target-specific properties, the
    // value is taken from target-specific properties (effectively overriding the platform
    // properties).
    execProperties.putAll(targetExecProperties);
    return ImmutableMap.copyOf(execProperties);
  }

  @VisibleForTesting
  public static ActionOwner createActionOwner(
      Rule rule,
      ImmutableList<AspectDescriptor> aspectDescriptors,
      BuildConfiguration configuration,
      Map<String, String> targetExecProperties,
      @Nullable PlatformInfo executionPlatform) {
    return ActionOwner.create(
        rule.getLabel(),
        aspectDescriptors,
        rule.getLocation(),
        configuration.getMnemonic(),
        rule.getTargetKind(),
        configuration.checksum(),
        configuration.toBuildEvent(),
        configuration.isHostConfiguration() ? HOST_CONFIGURATION_PROGRESS_TAG : null,
        computeExecProperties(targetExecProperties, executionPlatform),
        executionPlatform);
  }

  @Override
  public void registerAction(ActionAnalysisMetadata... action) {
    getAnalysisEnvironment().registerAction(action);
  }

  /**
   * Convenience function for subclasses to report non-attribute-specific
   * errors in the current rule.
   */
  @Override
  public void ruleError(String message) {
    reporter.ruleError(message);
  }

  /**
   * Convenience function for subclasses to report non-attribute-specific
   * warnings in the current rule.
   */
  @Override
  public void ruleWarning(String message) {
    reporter.ruleWarning(message);
  }

  /**
   * Convenience function for subclasses to report attribute-specific errors in
   * the current rule.
   *
   * <p>If the name of the attribute starts with <code>$</code>
   * it is replaced with a string <code>(an implicit dependency)</code>.
   */
  @Override
  public void attributeError(String attrName, String message) {
    reporter.attributeError(attrName, message);
  }

  /**
   * Like attributeError, but does not mark the configured target as errored.
   *
   * <p>If the name of the attribute starts with <code>$</code>
   * it is replaced with a string <code>(an implicit dependency)</code>.
   */
  @Override
  public void attributeWarning(String attrName, String message) {
    reporter.attributeWarning(attrName, message);
  }

  /**
   * Returns an artifact beneath the root of either the "bin" or "genfiles"
   * tree, whose path is based on the name of this target and the current
   * configuration.  The choice of which tree to use is based on the rule with
   * which this target (which must be an OutputFile or a Rule) is associated.
   */
  public Artifact createOutputArtifact() {
    Target target = getTarget();
    PathFragment rootRelativePath = getPackageDirectory()
        .getRelative(PathFragment.create(target.getName()));

    return internalCreateOutputArtifact(rootRelativePath, target, OutputFile.Kind.FILE);
  }

  /**
   * Returns an artifact beneath the root of either the "bin" or "genfiles"
   * tree, whose path is based on the name of this target and the current
   * configuration, with a script suffix appropriate for the current host platform. ({@code .cmd}
   * for Windows, otherwise {@code .sh}). The choice of which tree to use is based on the rule with
   * which this target (which must be an OutputFile or a Rule) is associated.
   */
  public Artifact createOutputArtifactScript() {
    Target target = getTarget();
    // TODO(laszlocsomor): Use the execution platform, not the host platform.
    boolean isExecutedOnWindows = OS.getCurrent() == OS.WINDOWS;

    String fileExtension = isExecutedOnWindows ? ".cmd" : ".sh";

    PathFragment rootRelativePath = getPackageDirectory()
        .getRelative(PathFragment.create(target.getName() + fileExtension));

    return internalCreateOutputArtifact(rootRelativePath, target, OutputFile.Kind.FILE);
  }

  /**
   * Returns the output artifact of an {@link OutputFile} of this target.
   *
   * @see #createOutputArtifact()
   */
  public Artifact createOutputArtifact(OutputFile out) {
    PathFragment packageRelativePath = getPackageDirectory()
        .getRelative(PathFragment.create(out.getName()));
    return internalCreateOutputArtifact(packageRelativePath, out, out.getKind());
  }

  /**
   * Implementation for {@link #createOutputArtifact()} and
   * {@link #createOutputArtifact(OutputFile)}. This is private so that
   * {@link #createOutputArtifact(OutputFile)} can have a more specific
   * signature.
   */
  private Artifact internalCreateOutputArtifact(PathFragment rootRelativePath,
      Target target, OutputFile.Kind outputFileKind) {
    Preconditions.checkState(
        target.getLabel().getPackageIdentifier().equals(getLabel().getPackageIdentifier()),
        "Creating output artifact for target '%s' in different package than the rule '%s' "
            + "being analyzed", target.getLabel(), getLabel());
    ArtifactRoot root = getBinOrGenfilesDirectory();

    switch (outputFileKind) {
      case FILE:
        return getDerivedArtifact(rootRelativePath, root);
      case FILESET:
        return getAnalysisEnvironment().getFilesetArtifact(rootRelativePath, root);
      default:
        throw new IllegalStateException();
    }
  }

  /**
   * Returns the root of either the "bin" or "genfiles" tree, based on this target and the current
   * configuration. The choice of which tree to use is based on the rule with which this target
   * (which must be an OutputFile or a Rule) is associated.
   */
  @Override
  public ArtifactRoot getBinOrGenfilesDirectory() {
    return rule.hasBinaryOutput()
        ? getConfiguration().getBinDirectory(getLabel().getRepository())
        : getConfiguration().getGenfilesDirectory(getLabel().getRepository());
  }

  /**
   * Creates an artifact in a directory that is unique to the package that contains the rule, thus
   * guaranteeing that it never clashes with artifacts created by rules in other packages.
   */
  public Artifact getBinArtifact(String relative) {
    return getBinArtifact(PathFragment.create(relative));
  }

  public Artifact getBinArtifact(PathFragment relative) {
    return getPackageRelativeArtifact(
        relative, getConfiguration().getBinDirectory(getLabel().getRepository()));
  }

  /**
   * Creates an artifact in a directory that is unique to the package that contains the rule, thus
   * guaranteeing that it never clashes with artifacts created by rules in other packages.
   */
  public Artifact getGenfilesArtifact(String relative) {
    return getGenfilesArtifact(PathFragment.create(relative));
  }

  public Artifact getGenfilesArtifact(PathFragment relative) {
    return getPackageRelativeArtifact(
        relative, getConfiguration().getGenfilesDirectory(getLabel().getRepository()));
  }

  @Override
  public Artifact getShareableArtifact(PathFragment rootRelativePath, ArtifactRoot root) {
    return getAnalysisEnvironment().getDerivedArtifact(rootRelativePath, root);
  }

  @Override
  public Artifact.DerivedArtifact getPackageRelativeArtifact(
      PathFragment relative, ArtifactRoot root) {
    return getPackageRelativeArtifact(relative, root, /*contentBasedPath=*/ false);
  }

  /**
   * Same as {@link #getPackageRelativeArtifact(PathFragment, ArtifactRoot)} but includes the option
   * option to use a content-based path for this artifact (see {@link
   * BuildConfiguration#useContentBasedOutputPaths()}).
   */
  private Artifact.DerivedArtifact getPackageRelativeArtifact(
      PathFragment relative, ArtifactRoot root, boolean contentBasedPath) {
    return getDerivedArtifact(getPackageDirectory().getRelative(relative), root, contentBasedPath);
  }

  /**
   * Creates an artifact in a directory that is unique to the package that contains the rule, thus
   * guaranteeing that it never clashes with artifacts created by rules in other packages.
   */
  public Artifact getPackageRelativeArtifact(String relative, ArtifactRoot root) {
    return getPackageRelativeArtifact(relative, root, /*contentBasedPath=*/ false);
  }

  /**
   * Same as {@link #getPackageRelativeArtifact(String, ArtifactRoot)} but includes the option to
   * use a content-based path for this artifact (see {@link
   * BuildConfiguration#useContentBasedOutputPaths()}).
   */
  private Artifact getPackageRelativeArtifact(
      String relative, ArtifactRoot root, boolean contentBasedPath) {
    return getPackageRelativeArtifact(PathFragment.create(relative), root, contentBasedPath);
  }

  @Override
  public PathFragment getPackageDirectory() {
    return getLabel()
        .getPackageIdentifier()
        .getPackagePath(getConfiguration().isSiblingRepositoryLayout());
  }

  /**
   * Creates an artifact under a given root with the given root-relative path.
   *
   * <p>Verifies that it is in the root-relative directory corresponding to the package of the rule,
   * thus ensuring that it doesn't clash with other artifacts generated by other rules using this
   * method.
   */
  @Override
  public Artifact.DerivedArtifact getDerivedArtifact(
      PathFragment rootRelativePath, ArtifactRoot root) {
    return getDerivedArtifact(rootRelativePath, root, /*contentBasedPath=*/ false);
  }

  /**
   * Same as {@link #getDerivedArtifact(PathFragment, ArtifactRoot)} but includes the option to use
   * a content-based path for this artifact (see {@link
   * BuildConfiguration#useContentBasedOutputPaths()}).
   */
  public Artifact.DerivedArtifact getDerivedArtifact(
      PathFragment rootRelativePath, ArtifactRoot root, boolean contentBasedPath) {
    Preconditions.checkState(rootRelativePath.startsWith(getPackageDirectory()),
        "Output artifact '%s' not under package directory '%s' for target '%s'",
        rootRelativePath, getPackageDirectory(), getLabel());
    return getAnalysisEnvironment().getDerivedArtifact(rootRelativePath, root, contentBasedPath);
  }

  @Override
  public SpecialArtifact getTreeArtifact(PathFragment rootRelativePath, ArtifactRoot root) {
    Preconditions.checkState(rootRelativePath.startsWith(getPackageDirectory()),
        "Output artifact '%s' not under package directory '%s' for target '%s'",
        rootRelativePath, getPackageDirectory(), getLabel());
    return getAnalysisEnvironment().getTreeArtifact(rootRelativePath, root);
  }

  /**
   * Creates a tree artifact in a directory that is unique to the package that contains the rule,
   * thus guaranteeing that it never clashes with artifacts created by rules in other packages.
   */
  public Artifact getPackageRelativeTreeArtifact(PathFragment relative, ArtifactRoot root) {
    return getTreeArtifact(getPackageDirectory().getRelative(relative), root);
  }

  /**
   * Creates an artifact in a directory that is unique to the rule, thus guaranteeing that it never
   * clashes with artifacts created by other rules.
   */
  public Artifact getUniqueDirectoryArtifact(
      String uniqueDirectory, String relative, ArtifactRoot root) {
    return getUniqueDirectoryArtifact(uniqueDirectory, PathFragment.create(relative), root);
  }

  @Override
  public Artifact getUniqueDirectoryArtifact(String uniqueDirectorySuffix, String relative) {
    return getUniqueDirectoryArtifact(uniqueDirectorySuffix, relative, getBinOrGenfilesDirectory());
  }

  @Override
  public Artifact getUniqueDirectoryArtifact(String uniqueDirectorySuffix, PathFragment relative) {
    return getUniqueDirectoryArtifact(uniqueDirectorySuffix, relative, getBinOrGenfilesDirectory());
  }

  @Override
  public Artifact getUniqueDirectoryArtifact(
      String uniqueDirectory, PathFragment relative, ArtifactRoot root) {
    return getDerivedArtifact(getUniqueDirectory(uniqueDirectory).getRelative(relative), root);
  }

  /**
   * Returns true iff the rule, or any attached aspect, has an attribute with the given name and
   * type.
   */
  public boolean isAttrDefined(String attrName, Type<?> type) {
    return attributes().has(attrName, type);
  }

  /**
   * Returns the dependencies through a {@code LABEL_DICT_UNARY} attribute as a map from
   * a string to a {@link TransitiveInfoCollection}.
   */
  public Map<String, TransitiveInfoCollection> getPrerequisiteMap(String attributeName) {
    Preconditions.checkState(attributes().has(attributeName, BuildType.LABEL_DICT_UNARY));

    ImmutableMap.Builder<String, TransitiveInfoCollection> result = ImmutableMap.builder();
    Map<String, Label> dict = attributes().get(attributeName, BuildType.LABEL_DICT_UNARY);
    Map<Label, ConfiguredTarget> labelToDep = new HashMap<>();
    for (ConfiguredTargetAndData dep : targetMap.get(attributeName)) {
      labelToDep.put(dep.getTarget().getLabel(), dep.getConfiguredTarget());
    }

    for (Map.Entry<String, Label> entry : dict.entrySet()) {
      result.put(entry.getKey(), Preconditions.checkNotNull(labelToDep.get(entry.getValue())));
    }

    return result.build();
  }

  /**
   * Returns the prerequisites keyed by their configuration transition keys. If the split transition
   * is not active (e.g. split() returned an empty list), the key is an empty Optional.
   */
  public Map<Optional<String>, ? extends List<? extends TransitiveInfoCollection>>
      getSplitPrerequisites(String attributeName) {
    return Maps.transformValues(
        getSplitPrerequisiteConfiguredTargetAndTargets(attributeName),
        (ctatList) -> Lists.transform(ctatList, ConfiguredTargetAndData::getConfiguredTarget));
  }

  /**
   * Returns the prerequisites keyed by their transition keys. If the split transition is not active
   * (e.g. split() returned an empty list), the key is an empty Optional.
   */
  public Map<Optional<String>, List<ConfiguredTargetAndData>>
      getSplitPrerequisiteConfiguredTargetAndTargets(String attributeName) {
    checkAttributeIsDependency(attributeName);
    // Use an ImmutableListMultimap.Builder here to preserve ordering.
    ImmutableListMultimap.Builder<Optional<String>, ConfiguredTargetAndData> result =
        ImmutableListMultimap.builder();
    List<ConfiguredTargetAndData> deps = getPrerequisiteConfiguredTargets(attributeName);
    for (ConfiguredTargetAndData t : deps) {
      ImmutableList<String> transitionKeys = t.getTransitionKeys();
      if (transitionKeys.isEmpty()) {
        // The split transition is not active, i.e. does not change build configurations.
        // TODO(jungjw): Investigate if we need to do a check here.
        return ImmutableMap.of(Optional.absent(), deps);
      }
      for (String key : transitionKeys) {
        result.put(Optional.of(key), t);
      }
    }
    return Multimaps.asMap(result.build());
  }

  /**
   * Returns the specified provider of the prerequisite referenced by the attribute in the argument.
   * If the attribute is empty or it does not support the specified provider, returns null.
   */
  @Nullable
  public <C extends TransitiveInfoProvider> C getPrerequisite(
      String attributeName, Class<C> provider) {
    TransitiveInfoCollection prerequisite = getPrerequisite(attributeName);
    return prerequisite == null ? null : prerequisite.getProvider(provider);
  }

  /**
   * Returns the transitive info collection that feeds into this target through the specified
   * attribute. Returns null if the attribute is empty.
   */
  @Nullable
  public TransitiveInfoCollection getPrerequisite(String attributeName) {
    ConfiguredTargetAndData result = getPrerequisiteConfiguredTargetAndData(attributeName);
    return result == null ? null : result.getConfiguredTarget();
  }

  /**
   * Returns the {@link ConfiguredTargetAndData} that feeds ino this target through the specified
   * attribute. Returns null if the attribute is empty.
   */
  @Nullable
  public ConfiguredTargetAndData getPrerequisiteConfiguredTargetAndData(String attributeName) {
    checkAttributeIsDependency(attributeName);
    List<ConfiguredTargetAndData> elements = getPrerequisiteConfiguredTargets(attributeName);
    if (elements.size() > 1) {
      throw new IllegalStateException(getRuleClassNameForLogging() + " attribute " + attributeName
          + " produces more than one prerequisite");
    }
    return elements.isEmpty() ? null : elements.get(0);
  }

  /**
   * For a given attribute, returns all the ConfiguredTargetAndTargets of that attribute. Each
   * ConfiguredTargetAndData is keyed by the {@link BuildConfiguration} that created it.
   */
  public ImmutableListMultimap<BuildConfiguration, ConfiguredTargetAndData>
      getPrerequisiteCofiguredTargetAndTargetsByConfiguration(String attributeName) {
    checkAttributeIsDependency(attributeName);
    List<ConfiguredTargetAndData> ctatCollection = getPrerequisiteConfiguredTargets(attributeName);
    ImmutableListMultimap.Builder<BuildConfiguration, ConfiguredTargetAndData> result =
        ImmutableListMultimap.builder();
    for (ConfiguredTargetAndData ctad : ctatCollection) {
      result.put(ctad.getConfiguration(), ctad);
    }
    return result.build();
  }

  /**
   * For a given attribute, returns all declared provider provided by targets of that attribute.
   * Each declared provider is keyed by the {@link BuildConfiguration} under which the provider was
   * created.
   */
  public <C extends Info>
      ImmutableListMultimap<BuildConfiguration, C> getPrerequisitesByConfiguration(
          String attributeName, BuiltinProvider<C> provider) {
    checkAttributeIsDependency(attributeName);
    List<ConfiguredTargetAndData> ctatCollection = getPrerequisiteConfiguredTargets(attributeName);
    ImmutableListMultimap.Builder<BuildConfiguration, C> result =
        ImmutableListMultimap.builder();
    for (ConfiguredTargetAndData prerequisite : ctatCollection) {
      C prerequisiteProvider = prerequisite.getConfiguredTarget().get(provider);
      if (prerequisiteProvider != null) {
        result.put(prerequisite.getConfiguration(), prerequisiteProvider);
      }
    }
    return result.build();
  }

  /**
   * For a given attribute, returns all {@link TransitiveInfoCollection}s provided by targets of
   * that attribute. Each {@link TransitiveInfoCollection} is keyed by the {@link
   * BuildConfiguration} under which the collection was created.
   */
  public ImmutableListMultimap<BuildConfiguration, TransitiveInfoCollection>
      getPrerequisitesByConfiguration(String attributeName) {
    checkAttributeIsDependency(attributeName);
    List<ConfiguredTargetAndData> ctatCollection = getPrerequisiteConfiguredTargets(attributeName);
    ImmutableListMultimap.Builder<BuildConfiguration, TransitiveInfoCollection> result =
        ImmutableListMultimap.builder();
    for (ConfiguredTargetAndData prerequisite : ctatCollection) {
      result.put(prerequisite.getConfiguration(), prerequisite.getConfiguredTarget());
    }
    return result.build();
  }

  /**
   * Returns the list of transitive info collections that feed into this target through the
   * specified attribute.
   */
  public List<? extends TransitiveInfoCollection> getPrerequisites(String attributeName) {
    if (!attributes().has(attributeName)) {
      return ImmutableList.of();
    }

    List<ConfiguredTargetAndData> prerequisiteConfiguredTargets;
    // android_binary and android_test override deps to use a split transition.
    if ((getRule().getRuleClass().equals("android_binary")
            || getRule().getRuleClass().equals("android_test"))
        && attributeName.equals("deps")
        && attributes().getAttributeDefinition(attributeName).getTransitionFactory().isSplit()) {
      // TODO(b/168038145): Restore legacy behavior of returning the prerequisites from the first
      // portion of the split transition.
      // Callers should be identified, cleaned up, and this check removed.
      Map<Optional<String>, List<ConfiguredTargetAndData>> map =
          getSplitPrerequisiteConfiguredTargetAndTargets(attributeName);
      prerequisiteConfiguredTargets =
          map.isEmpty() ? ImmutableList.of() : map.entrySet().iterator().next().getValue();
    } else {
      prerequisiteConfiguredTargets = getPrerequisiteConfiguredTargets(attributeName);
    }

    return Lists.transform(
        prerequisiteConfiguredTargets, ConfiguredTargetAndData::getConfiguredTarget);
  }

  /**
   * Returns all the providers of the specified type that are listed under the specified attribute
   * of this target in the BUILD file.
   */
  public <C extends TransitiveInfoProvider> List<C> getPrerequisites(
      String attributeName, Class<C> classType) {
    AnalysisUtils.checkProvider(classType);
    return AnalysisUtils.getProviders(getPrerequisites(attributeName), classType);
  }

  /**
   * Returns all the declared providers (native and Starlark) for the specified constructor under
   * the specified attribute of this target in the BUILD file.
   */
  public <T extends Info> List<T> getPrerequisites(
      String attributeName, NativeProvider<T> starlarkKey) {
    return AnalysisUtils.getProviders(getPrerequisites(attributeName), starlarkKey);
  }

  /**
   * Returns all the declared providers (native and Starlark) for the specified constructor under
   * the specified attribute of this target in the BUILD file.
   */
  public <T extends Info> List<T> getPrerequisites(
      String attributeName, BuiltinProvider<T> starlarkKey) {
    return AnalysisUtils.getProviders(getPrerequisites(attributeName), starlarkKey);
  }

  /**
   * Returns the declared provider (native and Starlark) for the specified constructor under the
   * specified attribute of this target in the BUILD file. May return null if there is no
   * TransitiveInfoCollection under the specified attribute.
   */
  @Nullable
  public <T extends Info> T getPrerequisite(String attributeName, NativeProvider<T> starlarkKey) {
    TransitiveInfoCollection prerequisite = getPrerequisite(attributeName);
    return prerequisite == null ? null : prerequisite.get(starlarkKey);
  }

  /**
   * Returns the declared provider (native and Starlark) for the specified constructor under the
   * specified attribute of this target in the BUILD file. May return null if there is no
   * TransitiveInfoCollection under the specified attribute.
   */
  @Nullable
  public <T extends Info> T getPrerequisite(String attributeName, BuiltinProvider<T> starlarkKey) {
    TransitiveInfoCollection prerequisite = getPrerequisite(attributeName);
    return prerequisite == null ? null : prerequisite.get(starlarkKey);
  }

  /**
   * Returns all the providers of the specified type that are listed under the specified attribute
   * of this target in the BUILD file, and that contain the specified provider.
   */
  public <C extends TransitiveInfoProvider>
      Iterable<? extends TransitiveInfoCollection> getPrerequisitesIf(
          String attributeName, Class<C> classType) {
    AnalysisUtils.checkProvider(classType);
    return AnalysisUtils.filterByProvider(getPrerequisites(attributeName), classType);
  }

  /**
   * Returns all the providers of the specified type that are listed under the specified attribute
   * of this target in the BUILD file, and that contain the specified provider.
   */
  public <C extends Info> Iterable<? extends TransitiveInfoCollection> getPrerequisitesIf(
      String attributeName, NativeProvider<C> classType) {
    return AnalysisUtils.filterByProvider(getPrerequisites(attributeName), classType);
  }

  /**
   * Returns all the providers of the specified type that are listed under the specified attribute
   * of this target in the BUILD file, and that contain the specified provider.
   */
  public <C extends Info> Iterable<? extends TransitiveInfoCollection> getPrerequisitesIf(
      String attributeName, BuiltinProvider<C> classType) {
    return AnalysisUtils.filterByProvider(getPrerequisites(attributeName), classType);
  }

  /**
   * Returns the prerequisite referred to by the specified attribute. Also checks whether the
   * attribute is marked as executable and that the target referred to can actually be executed.
   *
   * @param attributeName the name of the attribute
   * @return the {@link FilesToRunProvider} interface of the prerequisite.
   */
  @Nullable
  public FilesToRunProvider getExecutablePrerequisite(String attributeName) {
    Attribute ruleDefinition = attributes().getAttributeDefinition(attributeName);

    if (ruleDefinition == null) {
      throw new IllegalStateException(getRuleClassNameForLogging() + " attribute " + attributeName
          + " is not defined");
    }
    if (!ruleDefinition.isExecutable()) {
      throw new IllegalStateException(getRuleClassNameForLogging() + " attribute " + attributeName
          + " is not configured to be executable");
    }

    TransitiveInfoCollection prerequisite = getPrerequisite(attributeName);
    if (prerequisite == null) {
      return null;
    }

    FilesToRunProvider result = prerequisite.getProvider(FilesToRunProvider.class);
    if (result == null || result.getExecutable() == null) {
      attributeError(
          attributeName, prerequisite.getLabel() + " does not refer to a valid executable target");
    }
    return result;
  }

  public void initConfigurationMakeVariableContext(
      Iterable<? extends MakeVariableSupplier> makeVariableSuppliers) {
    Preconditions.checkState(configurationMakeVariableContext == null);
    configurationMakeVariableContext =
        new ConfigurationMakeVariableContext(
            this, getRule().getPackage(), getConfiguration(), makeVariableSuppliers);
  }

  public void initConfigurationMakeVariableContext(MakeVariableSupplier... makeVariableSuppliers) {
    initConfigurationMakeVariableContext(ImmutableList.copyOf(makeVariableSuppliers));
  }

  public Expander getExpander(TemplateContext templateContext) {
    Expander expander = new Expander(this, templateContext);
    makeVariableExpanders.add(expander);
    return expander;
  }

  public Expander getExpander() {
    Expander expander = new Expander(this, getConfigurationMakeVariableContext());
    makeVariableExpanders.add(expander);
    return expander;
  }

  public Expander getExpander(ImmutableMap<Label, ImmutableCollection<Artifact>> labelMap) {
    Expander expander = new Expander(this, getConfigurationMakeVariableContext(), labelMap);
    makeVariableExpanders.add(expander);
    return expander;
  }

  /**
   * Returns a cached context that maps Make variable names (string) to values (string) without any
   * extra {@link MakeVariableSupplier}.
   */
  public ConfigurationMakeVariableContext getConfigurationMakeVariableContext() {
    if (configurationMakeVariableContext == null) {
      initConfigurationMakeVariableContext(ImmutableList.<MakeVariableSupplier>of());
    }
    return configurationMakeVariableContext;
  }

  @Nullable
  public ResolvedToolchainContext getToolchainContext() {
    return toolchainContexts == null ? null : toolchainContexts.getDefaultToolchainContext();
  }

  @Nullable
  private ResolvedToolchainContext getToolchainContext(String execGroup) {
    return toolchainContexts == null ? null : toolchainContexts.getToolchainContext(execGroup);
  }

  public boolean hasToolchainContext(String execGroup) {
    return toolchainContexts != null && toolchainContexts.hasToolchainContext(execGroup);
  }

  @Nullable
  public ToolchainCollection<ResolvedToolchainContext> getToolchainContexts() {
    return toolchainContexts;
  }

  public boolean targetPlatformHasConstraint(ConstraintValueInfo constraintValue) {
    if (toolchainContexts == null || toolchainContexts.getTargetPlatform() == null) {
      return false;
    }
    // All toolchain contexts should have the same target platform so we access via the default.
    return toolchainContexts.getTargetPlatform().constraints().hasConstraintValue(constraintValue);
  }

  public ConstraintSemantics<RuleContext> getConstraintSemantics() {
    return constraintSemantics;
  }

  /**
   * Returns the configuration fragments this rule uses.
   *
   * <p>Returned results are alphabetically ordered.
   */
  public ImmutableSortedSet<String> getRequiredConfigFragments() {
    ImmutableSortedSet.Builder<String> ans = ImmutableSortedSet.naturalOrder();
    ans.addAll(requiredConfigFragments);
    for (Expander makeVariableExpander : makeVariableExpanders) {
      for (String makeVariable : makeVariableExpander.lookedUpVariables()) {
        // User-defined make values may be set either in "--define foo=bar" or in a vardef in the
        // rule's package. Both are equivalent for these purposes, since in both cases setting
        // "--define foo=bar" impacts the rule's output.
        if (getRule().getPackage().getMakeEnvironment().containsKey(makeVariable)
            || getConfiguration().getCommandLineBuildVariables().containsKey(makeVariable)) {
          ans.add("--define:" + makeVariable);
        }
      }
    }
    return ans.build();
  }

  private ImmutableMap<String, ImmutableMap<String, String>> parseExecProperties(
      Map<String, String> execProperties) throws InvalidExecGroupException {
    if (execProperties.isEmpty()) {
      return ImmutableMap.of(DEFAULT_EXEC_GROUP_NAME, ImmutableMap.of());
    } else {
      return parseExecProperties(
          execProperties,
          toolchainContexts == null ? ImmutableSet.of() : toolchainContexts.getExecGroups());
    }
  }

  /**
   * Parse raw exec properties attribute value into a map of exec group names to their properties.
   * The raw map can have keys of two forms: (1) 'property' and (2) 'exec_group_name.property'. The
   * former get parsed into the target's default exec group, the latter get parsed into their
   * relevant exec groups.
   */
  private static ImmutableMap<String, ImmutableMap<String, String>> parseExecProperties(
      Map<String, String> rawExecProperties, Set<String> execGroups)
      throws InvalidExecGroupException {
    Map<String, Map<String, String>> consolidatedProperties = new HashMap<>();
    consolidatedProperties.put(DEFAULT_EXEC_GROUP_NAME, new HashMap<>());
    for (Map.Entry<String, String> execProperty : rawExecProperties.entrySet()) {
      String rawProperty = execProperty.getKey();
      int delimiterIndex = rawProperty.indexOf('.');
      if (delimiterIndex == -1) {
        consolidatedProperties
            .get(DEFAULT_EXEC_GROUP_NAME)
            .put(rawProperty, execProperty.getValue());
      } else {
        String execGroup = rawProperty.substring(0, delimiterIndex);
        String property = rawProperty.substring(delimiterIndex + 1);
        if (!execGroups.contains(execGroup)) {
          throw new InvalidExecGroupException(
              String.format(
                  "Tried to set exec property '%s' for non-existent exec group '%s'.",
                  property, execGroup));
        }
        consolidatedProperties.putIfAbsent(execGroup, new HashMap<>());
        consolidatedProperties.get(execGroup).put(property, execProperty.getValue());
      }
    }

    // Copy everything to immutable maps.
    ImmutableMap.Builder<String, ImmutableMap<String, String>> execProperties =
        new ImmutableMap.Builder<>();
    for (Map.Entry<String, Map<String, String>> execGroupMap : consolidatedProperties.entrySet()) {
      execProperties.put(execGroupMap.getKey(), ImmutableMap.copyOf(execGroupMap.getValue()));
    }

    return execProperties.build();
  }

  /**
   * Gets the combined exec properties of the given exec group and the target's exec properties. If
   * a property is set in both, the exec group properties take precedence. If a non-existent exec
   * group is passed in, just returns the target's exec properties.
   *
   * @param execGroup group whose properties to retrieve
   * @param execProperties Map of exec group name to map of properties and values
   */
  private static ImmutableMap<String, String> getExecProperties(
      String execGroup, Map<String, ImmutableMap<String, String>> execProperties) {
    if (!execProperties.containsKey(execGroup) || execGroup.equals(DEFAULT_EXEC_GROUP_NAME)) {
      return execProperties.get(DEFAULT_EXEC_GROUP_NAME);
    }

    // Use a HashMap to build here because we expect duplicate keys to happen
    // (and rewrite previous entries).
    Map<String, String> targetAndGroupProperties =
        new HashMap<>(execProperties.get(DEFAULT_EXEC_GROUP_NAME));
    targetAndGroupProperties.putAll(execProperties.get(execGroup));
    return ImmutableMap.copyOf(targetAndGroupProperties);
  }

  /** An error for when the user tries to access an non-existent exec group */
  public static final class InvalidExecGroupException extends Exception
      implements SaneAnalysisException {
    InvalidExecGroupException(String message) {
      super(message);
    }

    @Override
    public DetailedExitCode getDetailedExitCode() {
      return DetailedExitCode.of(
          FailureDetail.newBuilder()
              .setMessage(getMessage())
              .setAnalysis(Analysis.newBuilder().setCode(Code.EXEC_GROUP_MISSING))
              .build());
    }
  }

  @VisibleForTesting
  public ImmutableMap<String, ImmutableMap<String, String>> getExecPropertiesForTesting() {
    return execProperties;
  }

  private void checkAttributeIsDependency(String attributeName) {
    Attribute attributeDefinition = attributes.getAttributeDefinition(attributeName);
    if (attributeDefinition == null) {
      throw new IllegalStateException(getRule().getLocation() + ": " + getRuleClassNameForLogging()
        + " attribute " + attributeName + " is not defined");
    }
    if (attributeDefinition.getType().getLabelClass() != LabelClass.DEPENDENCY) {
      throw new IllegalStateException(getRuleClassNameForLogging() + " attribute " + attributeName
        + " is not a label type attribute");
    }
  }

  @Override
  @Nullable
  public PlatformInfo getExecutionPlatform() {
    if (getToolchainContext() == null) {
      return null;
    }
    return getToolchainContext().executionPlatform();
  }

  @Override
  @Nullable
  public PlatformInfo getExecutionPlatform(String execGroup) {
    if (getToolchainContexts() == null) {
      return null;
    }
    ResolvedToolchainContext toolchainContext = getToolchainContext(execGroup);
    return toolchainContext == null ? null : toolchainContext.executionPlatform();
  }

  /**
   * For the specified attribute "attributeName" (which must be of type list(label)), resolve all
   * the labels into ConfiguredTargets (for the configuration appropriate to the attribute) and
   * return their build artifacts as a {@link PrerequisiteArtifacts} instance.
   *
   * @param attributeName the name of the attribute to traverse
   */
  public PrerequisiteArtifacts getPrerequisiteArtifacts(String attributeName) {
    return PrerequisiteArtifacts.get(this, attributeName);
  }

  /**
   * For the specified attribute "attributeName" (which must be of type label), resolves the
   * ConfiguredTarget and returns its single build artifact.
   *
   * <p>If the attribute is optional, has no default and was not specified, then null will be
   * returned. Note also that null is returned (and an attribute error is raised) if there wasn't
   * exactly one build artifact for the target.
   */
  public Artifact getPrerequisiteArtifact(String attributeName) {
    TransitiveInfoCollection target = getPrerequisite(attributeName);
    return transitiveInfoCollectionToArtifact(attributeName, target);
  }

  /**
   * Equivalent to getPrerequisiteArtifact(), but also asserts that host-configuration is
   * appropriate for the specified attribute.
   */
  // TODO(b/165916637): Fix callers to this method to use getPrerequisiteArtifact instead.
  public Artifact getHostPrerequisiteArtifact(String attributeName) {
    return getPrerequisiteArtifact(attributeName);
  }

  private Artifact transitiveInfoCollectionToArtifact(
      String attributeName, TransitiveInfoCollection target) {
    if (target != null) {
      NestedSet<Artifact> artifacts = target.getProvider(FileProvider.class).getFilesToBuild();
      if (artifacts.isSingleton()) {
        return artifacts.getSingleton();
      } else {
        attributeError(attributeName, target.getLabel() + " expected a single artifact");
      }
    }
    return null;
  }

  /**
   * Returns the sole file in the "srcs" attribute. Reports an error and
   * (possibly) returns null if "srcs" does not identify a single file of the
   * expected type.
   */
  public Artifact getSingleSource(String fileTypeName) {
    List<Artifact> srcs = PrerequisiteArtifacts.get(this, "srcs").list();
    switch (srcs.size()) {
      case 0 : // error already issued by getSrc()
        return null;
      case 1 : // ok
        return Iterables.getOnlyElement(srcs);
      default :
        attributeError("srcs", "only a single " + fileTypeName + " is allowed here");
        return srcs.get(0);
    }
  }

  public Artifact getSingleSource() {
    return getSingleSource(getRuleClassNameForLogging() + " source file");
  }

  /**
   * Returns a path fragment qualified by the rule name and unique fragment to
   * disambiguate artifacts produced from the source file appearing in
   * multiple rules.
   *
   * <p>For example "pkg/dir/name" -> "pkg/&lt;fragment>/rule/dir/name.
   */
  public final PathFragment getUniqueDirectory(String fragment) {
    return getUniqueDirectory(PathFragment.create(fragment));
  }

  /**
   * Returns a path fragment qualified by the rule name and unique fragment to disambiguate
   * artifacts produced from the source file appearing in multiple rules.
   *
   * <p>For example "pkg/dir/name" -> "pkg/&lt;fragment>/rule/dir/name.
   */
  @Override
  public final PathFragment getUniqueDirectory(PathFragment fragment) {
    return AnalysisUtils.getUniqueDirectory(
        getLabel(), fragment, getConfiguration().isSiblingRepositoryLayout());
  }

  /**
   * Check that all targets that were specified as sources are from the same
   * package as this rule. Output a warning or an error for every target that is
   * imported from a different package.
   */
  public void checkSrcsSamePackage(boolean onlyWarn) {
    PathFragment packageName = getLabel().getPackageFragment();
    for (Artifact srcItem : PrerequisiteArtifacts.get(this, "srcs").list()) {
      if (!srcItem.isSourceArtifact()) {
        // In theory, we should not do this check. However, in practice, we
        // have a couple of rules that do not obey the "srcs must contain
        // files and only files" rule. Thus, we are stuck with this hack here :(
        continue;
      }
      Label associatedLabel = srcItem.getOwner();
      PathFragment itemPackageName = associatedLabel.getPackageFragment();
      if (!itemPackageName.equals(packageName)) {
        String message = "please do not import '" + associatedLabel + "' directly. "
            + "You should either move the file to this package or depend on "
            + "an appropriate rule there";
        if (onlyWarn) {
          attributeWarning("srcs", message);
        } else {
          attributeError("srcs", message);
        }
      }
    }
  }


  /**
   * Returns the label to which the {@code NODEP_LABEL} attribute
   * {@code attrName} refers, checking that it is a valid label, and that it is
   * referring to a local target. Reports a warning otherwise.
   */
  public Label getLocalNodepLabelAttribute(String attrName) {
    Label label = attributes().get(attrName, BuildType.NODEP_LABEL);
    if (label == null) {
      return null;
    }

    if (!getTarget().getLabel().getPackageFragment().equals(label.getPackageFragment())) {
      attributeWarning(attrName, "does not reference a local rule");
    }

    return label;
  }

  @Override
  public Artifact getImplicitOutputArtifact(ImplicitOutputsFunction function)
      throws InterruptedException {
    return getImplicitOutputArtifact(function, /*contentBasedPath=*/ false);
  }

  /**
   * Same as {@link #getImplicitOutputArtifact(ImplicitOutputsFunction)} but includes the option to
   * use a content-based path for this artifact (see {@link
   * BuildConfiguration#useContentBasedOutputPaths()}).
   */
  public Artifact getImplicitOutputArtifact(
      ImplicitOutputsFunction function, boolean contentBasedPath) throws InterruptedException {
    Iterable<String> result;
    try {
      result =
          function.getImplicitOutputs(
              getAnalysisEnvironment().getEventHandler(), RawAttributeMapper.of(rule));
    } catch (EvalException e) {
      // It's ok as long as we don't use this method from Starlark.
      throw new IllegalStateException(e);
    }
    return getImplicitOutputArtifact(Iterables.getOnlyElement(result), contentBasedPath);
  }

  /** Only use from Starlark. Returns the implicit output artifact for a given output path. */
  public Artifact getImplicitOutputArtifact(String path) {
    return getImplicitOutputArtifact(path, /*contentBasedPath=*/ false);
  }

  /**
   * Same as {@link #getImplicitOutputArtifact(String)} but includes the option to use a a
   * content-based path for this artifact (see {@link
   * BuildConfiguration#useContentBasedOutputPaths()}).
   */
  private Artifact getImplicitOutputArtifact(String path, boolean contentBasedPath) {
    return getPackageRelativeArtifact(path, getBinOrGenfilesDirectory(), contentBasedPath);
  }

  /**
   * Convenience method to return a configured target for the "compiler" attribute. Allows caller to
   * decide whether a warning should be printed if the "compiler" attribute is not set to the
   * default value.
   *
   * @param warnIfNotDefault if true, print a warning if the value for the "compiler" attribute is
   *     set to something other than the default
   * @return a ConfiguredTarget for the "compiler" attribute
   */
  public final FilesToRunProvider getCompiler(boolean warnIfNotDefault) {
    Label label = attributes().get("compiler", BuildType.LABEL);
    if (warnIfNotDefault && !label.equals(getRule().getAttrDefaultValue("compiler"))) {
      attributeWarning("compiler", "setting the compiler is strongly discouraged");
    }
    return getExecutablePrerequisite("compiler");
  }

  /**
   * Returns the (unmodifiable, ordered) list of artifacts which are the outputs
   * of this target.
   *
   * <p>Each element in this list is associated with a single output, either
   * declared implicitly (via setImplicitOutputsFunction()) or explicitly
   * (listed in the 'outs' attribute of our rule).
   */
  public final ImmutableList<Artifact> getOutputArtifacts() {
    ImmutableList.Builder<Artifact> artifacts = ImmutableList.builder();
    for (OutputFile out : getRule().getOutputFiles()) {
      artifacts.add(createOutputArtifact(out));
    }
    return artifacts.build();
  }

  /**
   * Like {@link #getOutputArtifacts()} but for a singular output item.
   * Reports an error if the "out" attribute is not a singleton.
   *
   * @return null if the output list is empty, the artifact for the first item
   *         of the output list otherwise
   */
  public Artifact getOutputArtifact() {
    List<Artifact> outs = getOutputArtifacts();
    if (outs.size() != 1) {
      attributeError("out", "exactly one output file required");
      if (outs.isEmpty()) {
        return null;
      }
    }
    return outs.get(0);
  }

  @Override
  public final Artifact.DerivedArtifact getRelatedArtifact(
      PathFragment pathFragment, String extension) {
    PathFragment file = FileSystemUtils.replaceExtension(pathFragment, extension);
    return getDerivedArtifact(file, getConfiguration().getBinDirectory(getLabel().getRepository()));
  }

  /**
   * Returns true if the target for this context is a test target.
   */
  public boolean isTestTarget() {
    return TargetUtils.isTestRule(getTarget());
  }

  /** Returns true if the testonly attribute is set on this context. */
  public boolean isTestOnlyTarget() {
    return attributes().has("testonly", Type.BOOLEAN) && attributes().get("testonly", Type.BOOLEAN);
  }

  /**
   * Returns true if {@code label} is visible from {@code prerequisite}.
   *
   * <p>This only computes the logic as implemented by the visibility system. The final decision
   * whether a dependency is allowed is made by {@link PrerequisiteValidator}.
   */
  public static boolean isVisible(Label label, TransitiveInfoCollection prerequisite) {
    // Check visibility attribute
    for (PackageGroupContents specification :
        prerequisite.getProvider(VisibilityProvider.class).getVisibility().toList()) {
      if (specification.containsPackage(label.getPackageIdentifier())) {
        return true;
      }
    }

    return false;
  }

  /**
   * Returns true if {@code rule} is visible from {@code prerequisite}.
   *
   * <p>This only computes the logic as implemented by the visibility system. The final decision
   * whether a dependency is allowed is made by {@link PrerequisiteValidator}.
   */
  public static boolean isVisible(Rule rule, TransitiveInfoCollection prerequisite) {
    return isVisible(rule.getLabel(), prerequisite);
  }

  /** @return the set of features applicable for the current rule. */
  public ImmutableSet<String> getFeatures() {
    return enabledFeatures;
  }

  /** @return the set of features that are disabled for the current rule. */
  public ImmutableSet<String> getDisabledFeatures() {
    return disabledFeatures;
  }

  @Override
  public RuleErrorConsumer getRuleErrorConsumer() {
    return this;
  }

  /**
   * Returns {@code true} if a {@link RequiredConfigFragmentsProvider} should be included for this
   * rule.
   */
  public boolean shouldIncludeRequiredConfigFragmentsProvider() {
    IncludeConfigFragmentsEnum setting =
        getConfiguration()
            .getOptions()
            .get(CoreOptions.class)
            .includeRequiredConfigFragmentsProvider;
    switch (setting) {
      case OFF:
        return false;
      case DIRECT_HOST_ONLY:
        return getConfiguration().isHostConfiguration();
      case DIRECT:
      case TRANSITIVE:
        return true;
    }
    throw new IllegalStateException("Unknown setting: " + setting);
  }

  @Override
  public String toString() {
    return "RuleContext(" + getLabel() + ", " + getConfiguration() + ")";
  }

  /**
   * Builder class for a RuleContext.
   */
  public static final class Builder implements RuleErrorConsumer  {
    private final AnalysisEnvironment env;
    private final Target target;
    private final ConfigurationFragmentPolicy configurationFragmentPolicy;
    private ImmutableList<Class<? extends Fragment>> universalFragments;
    private final BuildConfiguration configuration;
    private final BuildConfiguration hostConfiguration;
    private final ActionLookupKey actionOwnerSymbol;
    private final PrerequisiteValidator prerequisiteValidator;
    private final RuleErrorConsumer reporter;
    private OrderedSetMultimap<Attribute, ConfiguredTargetAndData> prerequisiteMap;
    private ImmutableMap<Label, ConfigMatchingProvider> configConditions;
    private NestedSet<PackageGroupContents> visibility;
    private ImmutableMap<String, Attribute> aspectAttributes;
    private ImmutableList<Aspect> aspects;
    private ToolchainCollection<ResolvedToolchainContext> toolchainContexts;
    private ImmutableMap<String, String> rawExecProperties;
    private ConstraintSemantics<RuleContext> constraintSemantics;
    private ImmutableSet<String> requiredConfigFragments = ImmutableSet.of();

    @VisibleForTesting
    public Builder(
        AnalysisEnvironment env,
        Target target,
        ImmutableList<Aspect> aspects,
        BuildConfiguration configuration,
        BuildConfiguration hostConfiguration,
        PrerequisiteValidator prerequisiteValidator,
        ConfigurationFragmentPolicy configurationFragmentPolicy,
        ActionLookupKey actionOwnerSymbol) {
      this.env = Preconditions.checkNotNull(env);
      this.target = Preconditions.checkNotNull(target);
      this.aspects = aspects;
      this.configurationFragmentPolicy = Preconditions.checkNotNull(configurationFragmentPolicy);
      this.configuration = Preconditions.checkNotNull(configuration);
      this.hostConfiguration = Preconditions.checkNotNull(hostConfiguration);
      this.prerequisiteValidator = prerequisiteValidator;
      this.actionOwnerSymbol = Preconditions.checkNotNull(actionOwnerSymbol);
      if (configuration.allowAnalysisFailures()) {
        reporter = new SuppressingErrorReporter();
      } else {
        reporter =
            new ErrorReporter(
                env, target.getAssociatedRule(), configuration, getRuleClassNameForLogging());
      }
    }

    @VisibleForTesting
    public RuleContext build() throws InvalidExecGroupException {
      Preconditions.checkNotNull(prerequisiteMap);
      Preconditions.checkNotNull(configConditions);
      Preconditions.checkNotNull(visibility);
      Preconditions.checkNotNull(constraintSemantics);
      AttributeMap attributes =
          ConfiguredAttributeMapper.of(target.getAssociatedRule(), configConditions);
      checkAttributesNonEmpty(attributes);
      ListMultimap<String, ConfiguredTargetAndData> targetMap = createTargetMap();
      ListMultimap<String, ConfiguredFilesetEntry> filesetEntryMap =
          createFilesetEntryMap(target.getAssociatedRule(), configConditions);
      if (rawExecProperties == null) {
        if (!attributes.has(RuleClass.EXEC_PROPERTIES, Type.STRING_DICT)) {
          rawExecProperties = ImmutableMap.of();
        } else {
          rawExecProperties =
              ImmutableMap.copyOf(attributes.get(RuleClass.EXEC_PROPERTIES, Type.STRING_DICT));
        }
      }
      return new RuleContext(
          this,
          attributes,
          targetMap,
          filesetEntryMap,
          configConditions,
          universalFragments,
          getRuleClassNameForLogging(),
          actionOwnerSymbol,
          aspectAttributes != null ? aspectAttributes : ImmutableMap.<String, Attribute>of(),
          toolchainContexts,
          constraintSemantics,
          requiredConfigFragments);
    }

    private void checkAttributesNonEmpty(AttributeMap attributes) {
      for (String attributeName : attributes.getAttributeNames()) {
        Attribute attr = attributes.getAttributeDefinition(attributeName);
        if (!attr.isNonEmpty()) {
          continue;
        }
        Object attributeValue = attributes.get(attributeName, attr.getType());

        // TODO(adonovan): define in terms of Starlark.len?
        boolean isEmpty = false;
        if (attributeValue instanceof List<?>) {
          isEmpty = ((List) attributeValue).isEmpty();
        } else if (attributeValue instanceof Map<?, ?>) {
          isEmpty = ((Map) attributeValue).isEmpty();
        }
        if (isEmpty) {
          reporter.attributeError(attr.getName(), "attribute must be non empty");
        }
      }
    }

    public Builder setVisibility(NestedSet<PackageGroupContents> visibility) {
      this.visibility = visibility;
      return this;
    }

    /**
     * Sets the prerequisites and checks their visibility. It also generates appropriate error or
     * warning messages and sets the error flag as appropriate.
     */
    public Builder setPrerequisites(
        OrderedSetMultimap<Attribute, ConfiguredTargetAndData> prerequisiteMap) {
      this.prerequisiteMap = Preconditions.checkNotNull(prerequisiteMap);
      return this;
    }

    /**
     * Adds attributes which are defined by an Aspect (and not by RuleClass).
     */
    public Builder setAspectAttributes(Map<String, Attribute> aspectAttributes) {
      this.aspectAttributes = ImmutableMap.copyOf(aspectAttributes);
      return this;
    }

    /**
     * Sets the configuration conditions needed to determine which paths to follow for this
     * rule's configurable attributes.
     */
    public Builder setConfigConditions(
        ImmutableMap<Label, ConfigMatchingProvider> configConditions) {
      this.configConditions = Preconditions.checkNotNull(configConditions);
      return this;
    }

    /** Sets the fragment that can be legally accessed even when not explicitly declared. */
    public Builder setUniversalFragments(ImmutableList<Class<? extends Fragment>> fragments) {
      // TODO(bazel-team): Add this directly to ConfigurationFragmentPolicy, so we
      // don't need separate logic specifically for checking this fragment. The challenge is
      // that we need RuleClassProvider to figure out what this fragment is, and not every
      // call state that creates ConfigurationFragmentPolicy has access to that.
      this.universalFragments = fragments;
      return this;
    }

    /** Sets the {@link ResolvedToolchainContext} used to access toolchains used by this rule. */
    public Builder setToolchainContext(ResolvedToolchainContext toolchainContext) {
      Preconditions.checkState(
          this.toolchainContexts == null,
          "toolchainContexts has already been set for this Builder");
      this.toolchainContexts =
          ToolchainCollection.<ResolvedToolchainContext>builder()
              .addDefaultContext(toolchainContext)
              .build();
      return this;
    }

    /** Sets the collection of {@link ResolvedToolchainContext}s available to this rule. */
    @VisibleForTesting
    public Builder setToolchainContexts(
        ToolchainCollection<ResolvedToolchainContext> toolchainContexts) {
      Preconditions.checkState(
          this.toolchainContexts == null,
          "toolchainContexts has already been set for this Builder");
      this.toolchainContexts = toolchainContexts;
      return this;
    }

    /**
     * Warning: if you set the exec properties using this method any exec_properties attribute value
     * will be ignored in favor of this value.
     */
    public Builder setExecProperties(ImmutableMap<String, String> execProperties) {
      this.rawExecProperties = execProperties;
      return this;
    }

    public Builder setConstraintSemantics(ConstraintSemantics<RuleContext> constraintSemantics) {
      this.constraintSemantics = constraintSemantics;
      return this;
    }

    public Builder setRequiredConfigFragments(ImmutableSet<String> requiredConfigFragments) {
      this.requiredConfigFragments = requiredConfigFragments;
      return this;
    }

    private boolean validateFilesetEntry(FilesetEntry filesetEntry, ConfiguredTargetAndData src) {
      NestedSet<Artifact> filesToBuild =
          src.getConfiguredTarget().getProvider(FileProvider.class).getFilesToBuild();
      if (filesToBuild.isSingleton() && filesToBuild.getSingleton().isFileset()) {
        return true;
      }

      if (filesetEntry.isSourceFileset()) {
        return true;
      }

      Target srcTarget = src.getTarget();
      if (!(srcTarget instanceof FileTarget)) {
        attributeError("entries", String.format(
            "Invalid 'srcdir' target '%s'. Must be another Fileset or package",
            srcTarget.getLabel()));
        return false;
      }

      if (srcTarget instanceof OutputFile) {
        attributeWarning("entries", String.format("'srcdir' target '%s' is not an input file. "
            + "This forces the Fileset to be executed unconditionally",
            srcTarget.getLabel()));
      }

      return true;
    }

    /**
     * Determines and returns a map from attribute name to list of configured fileset entries, based
     * on a PrerequisiteMap instance.
     */
    private ListMultimap<String, ConfiguredFilesetEntry> createFilesetEntryMap(
        final Rule rule, ImmutableMap<Label, ConfigMatchingProvider> configConditions) {
      if (!target.getTargetKind().equals("Fileset rule")) {
        return ImmutableSortedKeyListMultimap.<String, ConfiguredFilesetEntry>builder().build();
      }

      final ImmutableSortedKeyListMultimap.Builder<String, ConfiguredFilesetEntry> mapBuilder =
          ImmutableSortedKeyListMultimap.builder();
      for (Attribute attr : rule.getAttributes()) {
        if (attr.getType() != BuildType.FILESET_ENTRY_LIST) {
          continue;
        }
        String attributeName = attr.getName();
        Map<Label, ConfiguredTargetAndData> ctMap = new HashMap<>();
        for (ConfiguredTargetAndData prerequisite : prerequisiteMap.get(attr)) {
          ctMap.put(
              AliasProvider.getDependencyLabel(prerequisite.getConfiguredTarget()), prerequisite);
        }
        List<FilesetEntry> entries = ConfiguredAttributeMapper.of(rule, configConditions)
            .get(attributeName, BuildType.FILESET_ENTRY_LIST);
        for (FilesetEntry entry : entries) {
          if (entry.getFiles() == null) {
            Label label = entry.getSrcLabel();
            ConfiguredTargetAndData src = ctMap.get(label);
            if (!validateFilesetEntry(entry, src)) {
              continue;
            }

            mapBuilder.put(
                attributeName, new ConfiguredFilesetEntry(entry, src.getConfiguredTarget()));
          } else {
            ImmutableList.Builder<TransitiveInfoCollection> files = ImmutableList.builder();
            for (Label file : entry.getFiles()) {
              files.add(ctMap.get(file).getConfiguredTarget());
            }
            mapBuilder.put(attributeName, new ConfiguredFilesetEntry(entry, files.build()));
          }
        }
      }
      return mapBuilder.build();
    }

    /** Determines and returns a map from attribute name to list of configured targets. */
    private ImmutableSortedKeyListMultimap<String, ConfiguredTargetAndData> createTargetMap() {
      ImmutableSortedKeyListMultimap.Builder<String, ConfiguredTargetAndData> mapBuilder =
          ImmutableSortedKeyListMultimap.builder();

      for (Map.Entry<Attribute, Collection<ConfiguredTargetAndData>> entry :
          prerequisiteMap.asMap().entrySet()) {
        Attribute attribute = entry.getKey();
        if (attribute == null) {
          continue;
        }

        if (attribute.isSingleArtifact() && entry.getValue().size() > 1) {
          attributeError(attribute.getName(), "must contain a single dependency");
          continue;
        }

        if (attribute.isSilentRuleClassFilter()) {
          Predicate<RuleClass> filter = attribute.getAllowedRuleClassesPredicate();
          for (ConfiguredTargetAndData configuredTarget : entry.getValue()) {
            Target prerequisiteTarget = configuredTarget.getTarget();
            if ((prerequisiteTarget instanceof Rule)
                && filter.apply(((Rule) prerequisiteTarget).getRuleClassObject())) {
              validateDirectPrerequisite(attribute, configuredTarget);
              mapBuilder.put(attribute.getName(), configuredTarget);
            }
          }
        } else {
          for (ConfiguredTargetAndData configuredTarget : entry.getValue()) {
            validateDirectPrerequisite(attribute, configuredTarget);
            mapBuilder.put(attribute.getName(), configuredTarget);
          }
        }
      }
      return mapBuilder.build();
    }

    /**
     * Post a raw event to the analysis environment's event handler. This circumvents normal
     * error and warning reporting functionality to post events, and should only be used
     * in rare cases where a custom event needs to be handled.
     */
    public void post(Postable event) {
      env.getEventHandler().post(event);
    }

    @Override
    public void ruleError(String message) {
      reporter.ruleError(message);
    }

    @Override
    public void attributeError(String attrName, String message) {
      reporter.attributeError(attrName, message);
    }

    @Override
    public void ruleWarning(String message) {
      reporter.ruleWarning(message);
    }

    @Override
    public void attributeWarning(String attrName, String message) {
      reporter.attributeWarning(attrName, message);
    }

    @Override
    public boolean hasErrors() {
      return reporter.hasErrors();
    }

    private String badPrerequisiteMessage(
        ConfiguredTargetAndData prerequisite, String reason, boolean isWarning) {
      String msgReason = reason != null ? " (" + reason + ")" : "";
      if (isWarning) {
        return String.format(
            "%s is unexpected here%s; continuing anyway",
            AliasProvider.describeTargetWithAliases(prerequisite, TargetMode.WITH_KIND),
            msgReason);
      }
      return String.format("%s is misplaced here%s",
          AliasProvider.describeTargetWithAliases(prerequisite, TargetMode.WITH_KIND), msgReason);
    }

    private void reportBadPrerequisite(
        Attribute attribute,
        ConfiguredTargetAndData prerequisite,
        String reason,
        boolean isWarning) {
      String message = badPrerequisiteMessage(prerequisite, reason, isWarning);
      if (isWarning) {
        attributeWarning(attribute.getName(), message);
      } else {
        attributeError(attribute.getName(), message);
      }
    }

    private void validateDirectPrerequisiteType(
        ConfiguredTargetAndData prerequisite, Attribute attribute) {
      Target prerequisiteTarget = prerequisite.getTarget();
      Label prerequisiteLabel = prerequisiteTarget.getLabel();

      if (prerequisiteTarget instanceof Rule) {
        Rule prerequisiteRule = (Rule) prerequisiteTarget;

        String reason =
            attribute
                .getValidityPredicate()
                .checkValid(target.getAssociatedRule(), prerequisiteRule);
        if (reason != null) {
          reportBadPrerequisite(attribute, prerequisite, reason, false);
        }
      }

      if (prerequisiteTarget instanceof Rule) {
        validateRuleDependency(prerequisite, attribute);
      } else if (prerequisiteTarget instanceof FileTarget) {
        if (attribute.isStrictLabelCheckingEnabled()) {
          if (!attribute.getAllowedFileTypesPredicate()
              .apply(((FileTarget) prerequisiteTarget).getFilename())) {
            if (prerequisiteTarget instanceof InputFile
                && !((InputFile) prerequisiteTarget).getPath().exists()) {
              // Misplaced labels, no corresponding target exists
              if (attribute.getAllowedFileTypesPredicate().isNone()
                  && !((InputFile) prerequisiteTarget).getFilename().contains(".")) {
                // There are no allowed files in the attribute but it's not a valid rule,
                // and the filename doesn't contain a dot --> probably a misspelled rule
                attributeError(attribute.getName(),
                    "rule '" + prerequisiteLabel + "' does not exist");
              } else {
                attributeError(attribute.getName(),
                    "target '" + prerequisiteLabel + "' does not exist");
              }
            } else {
              // The file exists but has a bad extension
              reportBadPrerequisite(attribute, prerequisite,
                  "expected " + attribute.getAllowedFileTypesPredicate(), false);
            }
          }
        }
      }
    }

    /** Returns whether the context being constructed is for the evaluation of an aspect. */
    public boolean forAspect() {
      return !aspects.isEmpty();
    }

    public Rule getRule() {
      return target.getAssociatedRule();
    }

    /**
     * Expose the Starlark semantics that governs the building of this rule (and the rest of the
     * build)
     */
    public StarlarkSemantics getStarlarkSemantics() throws InterruptedException {
      return env.getStarlarkSemantics();
    }

    /**
     * Returns a rule class name suitable for log messages, including an aspect name if applicable.
     */
    public String getRuleClassNameForLogging() {
      if (aspects.isEmpty()) {
        return target.getAssociatedRule().getRuleClass();
      }

      return Joiner.on(",")
              .join(aspects.stream().map(a -> a.getDescriptor()).collect(Collectors.toList()))
          + " aspect on "
          + target.getAssociatedRule().getRuleClass();
    }

    public BuildConfiguration getConfiguration() {
      return configuration;
    }

    /**
     * @return true if {@code rule} is visible from {@code prerequisite}.
     *     <p>This only computes the logic as implemented by the visibility system. The final
     *     decision whether a dependency is allowed is made by {@link PrerequisiteValidator}, who is
     *     supposed to call this method to determine whether a dependency is allowed as per
     *     visibility rules.
     */
    public boolean isVisible(TransitiveInfoCollection prerequisite) {
      return RuleContext.isVisible(target.getAssociatedRule(), prerequisite);
    }

    private void validateDirectPrerequisiteFileTypes(
        ConfiguredTargetAndData prerequisite, Attribute attribute) {
      if (attribute.isSkipAnalysisTimeFileTypeCheck()) {
        return;
      }
      FileTypeSet allowedFileTypes = attribute.getAllowedFileTypesPredicate();
      if (allowedFileTypes == null) {
        // It's not a label or label_list attribute.
        return;
      }
      if (allowedFileTypes == FileTypeSet.ANY_FILE && !attribute.isNonEmpty()
          && !attribute.isSingleArtifact()) {
        return;
      }

      // If we allow any file we still need to check if there are actually files generated
      // Note that this check only runs for ANY_FILE predicates if the attribute is NON_EMPTY
      // or SINGLE_ARTIFACT
      // If we performed this check when allowedFileTypes == NO_FILE this would
      // always throw an error in those cases
      if (allowedFileTypes != FileTypeSet.NO_FILE) {
        NestedSet<Artifact> artifacts =
            prerequisite.getConfiguredTarget().getProvider(FileProvider.class).getFilesToBuild();
        if (attribute.isSingleArtifact() && !artifacts.isSingleton()) {
          attributeError(
              attribute.getName(),
              "'" + prerequisite.getTarget().getLabel() + "' must produce a single file");
          return;
        }
        for (Artifact sourceArtifact : artifacts.toList()) {
          if (allowedFileTypes.apply(sourceArtifact.getFilename())) {
            return;
          }
          if (sourceArtifact.isTreeArtifact()) {
            return;
          }
        }
        attributeError(
            attribute.getName(),
            "'"
                + prerequisite.getTarget().getLabel()
                + "' does not produce any "
                + getRuleClassNameForLogging()
                + " "
                + attribute.getName()
                + " files (expected "
                + allowedFileTypes
                + ")");
      }
    }

    /**
     * Because some rules still have to use allowedRuleClasses to do rule dependency validation. A
     * dependency is valid if it is from a rule in allowedRuledClasses, OR if all of the providers
     * in requiredProviders are provided by the target.
     */
    private void validateRuleDependency(ConfiguredTargetAndData prerequisite, Attribute attribute) {

      Set<String> unfulfilledRequirements = new LinkedHashSet<>();
      if (checkRuleDependencyClass(prerequisite, attribute, unfulfilledRequirements)) {
        return;
      }

      if (checkRuleDependencyClassWarnings(prerequisite, attribute)) {
        return;
      }

      if (checkRuleDependencyMandatoryProviders(prerequisite, attribute, unfulfilledRequirements)) {
        return;
      }

      // not allowed rule class and some mandatory providers missing => reject.
      if (!unfulfilledRequirements.isEmpty()) {
        attributeError(
            attribute.getName(), StringUtil.joinEnglishList(unfulfilledRequirements, "and"));
      }
    }

    /** Check if prerequisite should be allowed based on its rule class. */
    private boolean checkRuleDependencyClass(
        ConfiguredTargetAndData prerequisite,
        Attribute attribute,
        Set<String> unfulfilledRequirements) {
      if (attribute.getAllowedRuleClassesPredicate() != Predicates.<RuleClass>alwaysTrue()) {
        if (attribute
            .getAllowedRuleClassesPredicate()
            .apply(((Rule) prerequisite.getTarget()).getRuleClassObject())) {
          // prerequisite has an allowed rule class => accept.
          return true;
        }
        // remember that the rule class that was not allowed;
        // but maybe prerequisite provides required providers? do not reject yet.
        unfulfilledRequirements.add(
            badPrerequisiteMessage(
                prerequisite,
                "expected " + attribute.getAllowedRuleClassesPredicate(),
                false));
      }
      return false;
    }

    /**
     * Check if prerequisite should be allowed with warning based on its rule class.
     *
     * <p>If yes, also issues said warning.
     */
    private boolean checkRuleDependencyClassWarnings(
        ConfiguredTargetAndData prerequisite, Attribute attribute) {
      if (attribute
          .getAllowedRuleClassesWarningPredicate()
          .apply(((Rule) prerequisite.getTarget()).getRuleClassObject())) {
        Predicate<RuleClass> allowedRuleClasses = attribute.getAllowedRuleClassesPredicate();
        reportBadPrerequisite(
            attribute,
            prerequisite,
            allowedRuleClasses == Predicates.<RuleClass>alwaysTrue()
                ? null
                : "expected " + allowedRuleClasses,
            true);
        // prerequisite has a rule class allowed with a warning => accept, emitting a warning.
        return true;
      }
      return false;
    }

    /** Check if prerequisite should be allowed based on required providers on the attribute. */
    private boolean checkRuleDependencyMandatoryProviders(
        ConfiguredTargetAndData prerequisite,
        Attribute attribute,
        Set<String> unfulfilledRequirements) {
      RequiredProviders requiredProviders = attribute.getRequiredProviders();

      if (requiredProviders.acceptsAny()) {
        // If no required providers specified, we do not know if we should accept.
        return false;
      }

      if (prerequisite.getConfiguredTarget().satisfies(requiredProviders)) {
        return true;
      }

      unfulfilledRequirements.add(
          String.format(
              "'%s' does not have mandatory providers: %s",
              prerequisite.getTarget().getLabel(),
              prerequisite
                  .getConfiguredTarget()
                  .missingProviders(requiredProviders)
                  .getDescription()));

      return false;
    }

    private void validateDirectPrerequisite(
        Attribute attribute, ConfiguredTargetAndData prerequisite) {
      validateDirectPrerequisiteType(prerequisite, attribute);
      validateDirectPrerequisiteFileTypes(prerequisite, attribute);
      if (attribute.performPrereqValidatorCheck()) {
        prerequisiteValidator.validate(this, prerequisite, attribute);
      }
    }
  }

  /** Helper class for reporting errors and warnings. */
  private static final class ErrorReporter extends EventHandlingErrorReporter
      implements RuleErrorConsumer {
    private final Rule rule;
    private final BuildConfiguration configuration;

    ErrorReporter(
        AnalysisEnvironment env,
        Rule rule,
        BuildConfiguration configuration,
        String ruleClassNameForLogging) {
      super(ruleClassNameForLogging, env);
      this.rule = rule;
      this.configuration = configuration;
    }

    @Override
    protected String getMacroMessageAppendix(String unusedAttrName) {
      // TODO(b/141234726):  Historically this reported the location
      // of the rule attribute in the macro call (assuming no **kwargs),
      // but we no longer locations for individual attributes.
      // We should record the instantiation call stack in each rule
      // and report the position of its topmost frame here.
      return rule.wasCreatedByMacro()
          ? String.format(
              ". Since this rule was created by the macro '%s', the error might have been "
                  + "caused by the macro implementation",
              getGeneratorFunction())
          : "";
    }

    private String getGeneratorFunction() {
      return (String) rule.getAttr("generator_function");
    }

    @Override
    protected Label getLabel() {
      return rule.getLabel();
    }

    @Override
    protected BuildConfiguration getConfiguration() {
      return configuration;
    }

    @Override
    protected Location getRuleLocation() {
      return rule.getLocation();
    }
  }

  /**
   * Implementation of an error consumer which does not post any events, saves rule and attribute
   * errors for future consumption, and drops warnings.
   */
  public static final class SuppressingErrorReporter implements RuleErrorConsumer {
    private final List<String> errorMessages = Lists.newArrayList();

    @Override
    public void ruleWarning(String message) {}

    @Override
    public void ruleError(String message) {
      errorMessages.add(message);
    }

    @Override
    public void attributeWarning(String attrName, String message) {}

    @Override
    public void attributeError(String attrName, String message) {
      errorMessages.add(message);
    }

    @Override
    public boolean hasErrors() {
      return !errorMessages.isEmpty();
    }

    /**
     * Returns the error message strings reported to this error consumer.
     */
    public List<String> getErrorMessages() {
      return errorMessages;
    }
  }
}
