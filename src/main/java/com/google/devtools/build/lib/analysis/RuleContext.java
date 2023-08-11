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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.devtools.build.lib.analysis.constraints.ConstraintConstants.OS_TO_CONSTRAINTS;
import static com.google.devtools.build.lib.analysis.test.ExecutionInfo.DEFAULT_TEST_RUNNER_EXEC_GROUP;
import static com.google.devtools.build.lib.packages.ExecGroup.DEFAULT_EXEC_GROUP_NAME;

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
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimaps;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionRegistry;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.AliasProvider.TargetMode;
import com.google.devtools.build.lib.analysis.ExecGroupCollection.InvalidExecGroupException;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoKey;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.ConfigConditions;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.FeatureSet;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.analysis.stringtemplate.TemplateContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.ImmutableSortedKeyListMultimap;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BazelStarlarkContext;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.ConfigurationFragmentPolicy;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Package.ConfigSettingVisibilityPolicy;
import com.google.devtools.build.lib.packages.PackageSpecification.PackageGroupContents;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.RequiredProviders;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkProviderWrapper;
import com.google.devtools.build.lib.packages.SymbolGenerator;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.Type.LabelClass;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Identifier;
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
    implements ActionConstructionContext, ActionRegistry, RuleErrorConsumer, AutoCloseable {

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

  public static final String TOOLCHAIN_ATTR_NAME = "$toolchain";

  /** A fake attribute to use for toolchain-related validation errors. */
  private static final Attribute TOOLCHAIN_ATTRIBUTE =
      new Attribute.Builder<>(TOOLCHAIN_ATTR_NAME, BuildType.LABEL_LIST).build();

  private final Rule rule;
  /**
   * A list of all aspects applied to the target. If this {@code RuleContext} is for a rule
   * implementation, {@code aspects} is an empty list.
   *
   * <p>Otherwise, the last aspect in the list is the one that this {@code RuleContext} is for.
   */
  private final ImmutableList<Aspect> aspects;

  private final ImmutableList<AspectDescriptor> aspectDescriptors;
  private final ListMultimap<String, ConfiguredTargetAndData> targetMap;
  private final ImmutableMap<Label, ConfigMatchingProvider> configConditions;
  private final AspectAwareAttributeMapper attributes;
  private final FeatureSet features;
  private final String ruleClassNameForLogging;
  private final ConfigurationFragmentPolicy configurationFragmentPolicy;
  private final ConfiguredRuleClassProvider ruleClassProvider;
  private final RuleErrorConsumer reporter;
  @Nullable private final ToolchainCollection<ResolvedToolchainContext> toolchainContexts;
  private final ExecGroupCollection execGroupCollection;
  @Nullable private final RequiredConfigFragmentsProvider requiredConfigFragments;
  @Nullable private final NestedSet<Package> transitivePackagesForRunfileRepoMappingManifest;
  private final List<Expander> makeVariableExpanders = new ArrayList<>();

  /** Map of exec group names to ActionOwners. */
  private final Map<String, ActionOwner> actionOwners = new HashMap<>();

  private final SymbolGenerator<ActionLookupKey> actionOwnerSymbolGenerator;

  /* lazily computed cache for Make variables, computed from the above. See get... method */
  private transient ConfigurationMakeVariableContext configurationMakeVariableContext = null;

  /**
   * Thread used for any Starlark evaluation during analysis, e.g. rule implementation function for
   * a Starlark-defined rule, or Starlarkified helper logic for native rules that have been
   * partially migrated to {@code @_builtins}.
   */
  private final StarlarkThread starlarkThread;
  /**
   * The {@code ctx} object passed to a Starlark-defined rule's or aspect's implementation function.
   * This object may outlive the analysis phase, e.g. if it is returned in a provider.
   *
   * <p>Initialized explicitly by calling {@link #initStarlarkRuleContext}. Native rules that do not
   * pass this object to {@code @_builtins} might avoid the cost of initializing this object, but
   * for everyone else it's mandatory.
   */
  @Nullable private StarlarkRuleContext starlarkRuleContext;

  private RuleContext(
      Builder builder,
      AttributeMap attributes,
      ListMultimap<String, ConfiguredTargetAndData> targetMap,
      ExecGroupCollection execGroupCollection) {
    super(
        builder.env,
        builder.target.getAssociatedRule(),
        builder.configuration,
        builder.prerequisiteMap.get(null),
        builder.visibility);
    this.rule = builder.target.getAssociatedRule();
    this.aspects = builder.aspects;
    this.aspectDescriptors = aspects.stream().map(Aspect::getDescriptor).collect(toImmutableList());
    this.configurationFragmentPolicy = builder.configurationFragmentPolicy;
    this.ruleClassProvider = builder.ruleClassProvider;
    this.targetMap = targetMap;
    this.configConditions = builder.configConditions.asProviders();
    this.attributes = new AspectAwareAttributeMapper(attributes, builder.aspectAttributes);
    this.features = computeFeatures();
    this.ruleClassNameForLogging = builder.getRuleClassNameForLogging();
    this.actionOwnerSymbolGenerator = new SymbolGenerator<>(builder.actionOwnerSymbol);
    this.reporter = builder.reporter;
    this.toolchainContexts = builder.toolchainContexts;
    this.execGroupCollection = execGroupCollection;
    this.requiredConfigFragments = builder.requiredConfigFragments;
    this.transitivePackagesForRunfileRepoMappingManifest =
        builder.transitivePackagesForRunfileRepoMappingManifest;
    this.starlarkThread = createStarlarkThread(builder.mutability); // uses above state
  }

  private FeatureSet computeFeatures() {
    FeatureSet pkg = rule.getPackage().getPackageArgs().features();
    FeatureSet rule =
        attributes().has("features", Type.STRING_LIST)
            ? FeatureSet.parse(attributes().get("features", Type.STRING_LIST))
            : FeatureSet.EMPTY;
    return FeatureSet.mergeWithGlobalFeatures(
        FeatureSet.merge(pkg, rule), getConfiguration().getDefaultFeatures());
  }

  public boolean isAllowTagsPropagation() {
    return getAnalysisEnvironment()
        .getStarlarkSemantics()
        .getBool(BuildLanguageOptions.INCOMPATIBLE_ALLOW_TAGS_PROPAGATION);
  }

  public RepositoryName getRepository() {
    return rule.getRepository();
  }

  @Override
  public ArtifactRoot getBinDirectory() {
    return getConfiguration().getBinDirectory(getLabel().getRepository());
  }

  @SuppressWarnings("Unused") // Goal is to migrate here.
  public ArtifactRoot getBuildInfoDirectory() {
    return getConfiguration().getBuildInfoDirectory(getLabel().getRepository());
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
   * If this target's configuration suppresses analysis failures, this returns a list of strings,
   * where each string corresponds to a description of an error that occurred during processing this
   * target.
   *
   * @throws IllegalStateException if this target's configuration does not suppress analysis
   *     failures (if {@code getConfiguration().allowAnalysisFailures()} is false)
   */
  public List<String> getSuppressedErrorMessages() {
    Preconditions.checkState(
        getConfiguration().allowAnalysisFailures(),
        "Error messages can only be retrieved via RuleContext if allow_analysis_failures is true");
    Preconditions.checkState(
        reporter instanceof SuppressingErrorReporter, "Unexpected error reporter");
    return ((SuppressingErrorReporter) reporter).getErrorMessages();
  }

  /**
   * If this <code>RuleContext</code> is for an aspect implementation, returns that aspect. (it is
   * the last aspect in the list of aspects applied to a target; all other aspects are the ones main
   * aspect sees as specified by its "required_aspect_providers") Otherwise returns <code>null
   * </code>.
   */
  @Nullable
  public Aspect getMainAspect() {
    return Streams.findLast(aspects.stream()).orElse(null);
  }

  /**
   * Returns a rule class name suitable for log messages, including an aspect name if applicable.
   */
  public String getRuleClassNameForLogging() {
    return ruleClassNameForLogging;
  }

  /** Returns the workspace name for the rule. */
  public String getWorkspaceName() {
    return rule.getPackage().getWorkspaceName();
  }

  /** The configuration conditions that trigger this rule's configurable attributes. */
  public ImmutableMap<Label, ConfigMatchingProvider> getConfigConditions() {
    return configConditions;
  }

  /** All aspects applied to the rule. */
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

  @Override
  public ActionOwner getActionOwner() {
    return getActionOwner(DEFAULT_EXEC_GROUP_NAME);
  }

  /**
   * Returns a special action owner for test actions. Test actions should run on the target platform
   * rather than the host platform. Note that the value is not cached (on the assumption that this
   * method is only called once).
   */
  public ActionOwner getTestActionOwner() {
    PlatformInfo testExecutionPlatform;
    ImmutableMap<String, String> testExecProperties;

    // If we have a toolchain, pull the target platform out of it.
    if (toolchainContexts != null) {
      // TODO(https://github.com/bazelbuild/bazel/issues/17466): This doesn't respect execution
      // properties coming from the target's `exec_properties` attribute.
      // src/test/java/com/google/devtools/build/lib/analysis/test/TestActionBuilderTest.java has a
      // test to test for it when it gets figured out.
      testExecutionPlatform = toolchainContexts.getTargetPlatform();
      testExecProperties = testExecutionPlatform.execProperties();
    } else {
      testExecutionPlatform = null;
      testExecProperties = getExecGroups().getExecProperties(DEFAULT_TEST_RUNNER_EXEC_GROUP);
    }

    ActionOwner actionOwner =
        createActionOwner(
            rule, aspectDescriptors, getConfiguration(), testExecProperties, testExecutionPlatform);

    if (actionOwner == null) {
      actionOwner = getActionOwner();
    }
    return actionOwner;
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
            execGroupCollection.getExecProperties(execGroup),
            getExecutionPlatform(execGroup));
    actionOwners.put(execGroup, actionOwner);
    return actionOwner;
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

  /** Returns a configuration fragment for this this target. */
  @Nullable
  public <T extends Fragment> T getFragment(Class<T> fragment) {
    return getFragment(fragment, fragment.getSimpleName(), "");
  }

  @Nullable
  private <T extends Fragment> T getFragment(
      Class<T> fragment, String name, String additionalErrorMessage) {
    // TODO(bazel-team): The fragments can also be accessed directly through
    // BuildConfigurationValue. Can we lock that down somehow?
    Preconditions.checkArgument(
        isLegalFragment(fragment),
        "%s has to declare '%s' as a required fragment in order to access it.%s",
        ruleClassNameForLogging,
        name,
        additionalErrorMessage);
    return getConfiguration().getFragment(fragment);
  }

  @Nullable
  public Fragment getStarlarkFragment(String name) throws EvalException {
    Class<? extends Fragment> fragmentClass = getConfiguration().getStarlarkFragmentByName(name);
    if (fragmentClass == null) {
      return null;
    }
    try {
      Preconditions.checkArgument(
          isLegalFragment(fragmentClass),
          "%s has to declare '%s' as a required fragment in order to access it."
              + " Please update the 'fragments' argument of the rule definition "
              + "(for example: fragments = [\"%s\"])",
          ruleClassNameForLogging,
          name,
          name);
      return getConfiguration().getFragment(fragmentClass);
    } catch (IllegalArgumentException ex) { // fishy
      throw new EvalException(ex.getMessage());
    }
  }

  public ImmutableCollection<String> getStarlarkFragmentNames() {
    return getConfiguration().getStarlarkFragmentNames();
  }

  public <T extends Fragment> boolean isLegalFragment(Class<T> fragment) {
    return ruleClassProvider.getFragmentRegistry().getUniversalFragments().contains(fragment)
        || configurationFragmentPolicy.isLegalConfigurationFragment(fragment);
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

  @VisibleForTesting
  public static ActionOwner createActionOwner(
      Rule rule,
      ImmutableList<AspectDescriptor> aspectDescriptors,
      BuildConfigurationValue buildConfigurationValue,
      ImmutableMap<String, String> execProperties,
      @Nullable PlatformInfo executionPlatform) {
    return ActionOwner.create(
        rule.getLabel(),
        rule.getLocation(),
        rule.getTargetKind(),
        buildConfigurationValue,
        executionPlatform,
        aspectDescriptors,
        execProperties);
  }

  @Override
  public void registerAction(ActionAnalysisMetadata action) {
    getAnalysisEnvironment().registerAction(action);
  }

  /**
   * Convenience function for subclasses to report non-attribute-specific errors in the current
   * rule.
   */
  @Override
  public void ruleError(String message) {
    reporter.ruleError(message);
  }

  /**
   * Convenience function for subclasses to report non-attribute-specific warnings in the current
   * rule.
   */
  @Override
  public void ruleWarning(String message) {
    reporter.ruleWarning(message);
  }

  /**
   * Convenience function for subclasses to report attribute-specific errors in the current rule.
   *
   * <p>If the name of the attribute starts with <code>$</code> it is replaced with a string <code>
   * (an implicit dependency)</code>.
   */
  @Override
  public void attributeError(String attrName, String message) {
    reporter.attributeError(attrName, message);
  }

  /**
   * Like attributeError, but does not mark the configured target as errored.
   *
   * <p>If the name of the attribute starts with <code>$</code> it is replaced with a string <code>
   * (an implicit dependency)</code>.
   */
  @Override
  public void attributeWarning(String attrName, String message) {
    reporter.attributeWarning(attrName, message);
  }

  /**
   * Returns an artifact beneath the root of either the "bin" or "genfiles" tree, whose path is
   * based on the name of this target and the current configuration. The choice of which tree to use
   * is based on the rule with which this target (which must be an OutputFile or a Rule) is
   * associated.
   */
  public Artifact createOutputArtifact() {
    Target target = getTarget();
    PathFragment rootRelativePath =
        getPackageDirectory().getRelative(PathFragment.create(target.getName()));

    return internalCreateOutputArtifact(rootRelativePath, target, OutputFile.Kind.FILE);
  }

  /**
   * Returns the output artifact of an {@link OutputFile} of this target.
   *
   * @see #createOutputArtifact()
   */
  public Artifact createOutputArtifact(OutputFile out) {
    PathFragment packageRelativePath =
        getPackageDirectory().getRelative(PathFragment.create(out.getName()));
    return internalCreateOutputArtifact(packageRelativePath, out, out.getKind());
  }

  /**
   * Returns an artifact beneath the root of either the "bin" or "genfiles" tree, whose path is
   * based on the name of this target and the current configuration, with a script suffix
   * appropriate for the current host platform. ({@code .cmd} for Windows, otherwise {@code .sh}).
   * The choice of which tree to use is based on the rule with which this target (which must be an
   * OutputFile or a Rule) is associated.
   */
  public Artifact createOutputArtifactScript() {
    Target target = getTarget();

    String fileExtension = isExecutedOnWindows() ? ".cmd" : ".sh";

    PathFragment rootRelativePath =
        getPackageDirectory().getRelative(PathFragment.create(target.getName() + fileExtension));

    return internalCreateOutputArtifact(rootRelativePath, target, OutputFile.Kind.FILE);
  }

  /**
   * Implementation for {@link #createOutputArtifact()} and {@link
   * #createOutputArtifact(OutputFile)}. This is private so that {@link
   * #createOutputArtifact(OutputFile)} can have a more specific signature.
   */
  private Artifact internalCreateOutputArtifact(
      PathFragment rootRelativePath, Target target, OutputFile.Kind outputFileKind) {
    Preconditions.checkState(
        target.getLabel().getPackageIdentifier().equals(getLabel().getPackageIdentifier()),
        "Creating output artifact for target '%s' in different package than the rule '%s' "
            + "being analyzed",
        target.getLabel(),
        getLabel());
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
    return rule.outputsToBindir()
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
    return getPackageRelativeArtifact(relative, root, /* contentBasedPath= */ false);
  }

  /**
   * Same as {@link #getPackageRelativeArtifact(PathFragment, ArtifactRoot)} but includes the option
   * option to use a content-based path for this artifact (see {@link
   * BuildConfigurationValue#useContentBasedOutputPaths()}).
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
    return getPackageRelativeArtifact(relative, root, /* contentBasedPath= */ false);
  }

  /**
   * Same as {@link #getPackageRelativeArtifact(String, ArtifactRoot)} but includes the option to
   * use a content-based path for this artifact (see {@link
   * BuildConfigurationValue#useContentBasedOutputPaths()}).
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
    return getDerivedArtifact(rootRelativePath, root, /* contentBasedPath= */ false);
  }

  /**
   * Same as {@link #getDerivedArtifact(PathFragment, ArtifactRoot)} but includes the option to use
   * a content-based path for this artifact (see {@link
   * BuildConfigurationValue#useContentBasedOutputPaths()}).
   */
  public Artifact.DerivedArtifact getDerivedArtifact(
      PathFragment rootRelativePath, ArtifactRoot root, boolean contentBasedPath) {
    Preconditions.checkState(
        rootRelativePath.startsWith(getPackageDirectory()),
        "Output artifact '%s' not under package directory '%s' for target '%s'",
        rootRelativePath,
        getPackageDirectory(),
        getLabel());
    return getAnalysisEnvironment().getDerivedArtifact(rootRelativePath, root, contentBasedPath);
  }

  @Override
  public SpecialArtifact getTreeArtifact(PathFragment rootRelativePath, ArtifactRoot root) {
    Preconditions.checkState(
        rootRelativePath.startsWith(getPackageDirectory()),
        "Output artifact '%s' not under package directory '%s' for target '%s'",
        rootRelativePath,
        getPackageDirectory(),
        getLabel());
    return getAnalysisEnvironment().getTreeArtifact(rootRelativePath, root);
  }

  /**
   * Creates a tree artifact in a directory that is unique to the package that contains the rule,
   * thus guaranteeing that it never clashes with artifacts created by rules in other packages.
   */
  public Artifact getPackageRelativeTreeArtifact(PathFragment relative, ArtifactRoot root) {
    return getTreeArtifact(getPackageDirectory().getRelative(relative), root);
  }

  public Artifact getPackageRelativeTreeArtifact(String relative, ArtifactRoot root) {
    return getPackageRelativeTreeArtifact(PathFragment.create(relative), root);
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
   * Returns the dependencies through a {@code LABEL_DICT_UNARY} attribute as a map from a string to
   * a {@link TransitiveInfoCollection}.
   */
  public Map<String, TransitiveInfoCollection> getPrerequisiteMap(String attributeName) {
    Preconditions.checkState(attributes().has(attributeName, BuildType.LABEL_DICT_UNARY));

    ImmutableMap.Builder<String, TransitiveInfoCollection> result = ImmutableMap.builder();
    Map<String, Label> dict = attributes().get(attributeName, BuildType.LABEL_DICT_UNARY);
    Map<Label, ConfiguredTarget> labelToDep = new HashMap<>();
    for (ConfiguredTargetAndData dep : targetMap.get(attributeName)) {
      labelToDep.put(dep.getTargetLabel(), dep.getConfiguredTarget());
    }

    for (Map.Entry<String, Label> entry : dict.entrySet()) {
      result.put(entry.getKey(), Preconditions.checkNotNull(labelToDep.get(entry.getValue())));
    }

    return result.buildOrThrow();
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
   * Returns the {@link ConfiguredTargetAndData} that feeds ino this target through the specified
   * attribute. Returns null if the attribute is empty.
   */
  @Nullable
  public ConfiguredTargetAndData getPrerequisiteConfiguredTargetAndData(String attributeName) {
    checkAttributeIsDependency(attributeName);
    List<ConfiguredTargetAndData> elements = getPrerequisiteConfiguredTargets(attributeName);
    Preconditions.checkState(
        elements.size() <= 1,
        "%s attribute %s produces more than one prerequisite",
        ruleClassNameForLogging,
        attributeName);
    return elements.isEmpty() ? null : elements.get(0);
  }

  /**
   * For a given attribute, returns all declared provider provided by targets of that attribute.
   * Each declared provider is keyed by the {@link BuildConfigurationValue} under which the provider
   * was created.
   */
  public <C extends Info>
      ImmutableListMultimap<BuildConfigurationValue, C> getPrerequisitesByConfiguration(
          String attributeName, BuiltinProvider<C> provider) {
    checkAttributeIsDependency(attributeName);
    List<ConfiguredTargetAndData> ctatCollection = getPrerequisiteConfiguredTargets(attributeName);
    ImmutableListMultimap.Builder<BuildConfigurationValue, C> result =
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
   * Returns the list of transitive info collections that feed into this target through the
   * specified attribute.
   */
  public List<? extends TransitiveInfoCollection> getPrerequisites(String attributeName) {
    if (!attributes().has(attributeName)) {
      return ImmutableList.of();
    }

    List<ConfiguredTargetAndData> prerequisiteConfiguredTargets;
    // android_binary, android_test, and android_binary_internal override deps to use a split
    // transition.
    if ((rule.getRuleClass().equals("android_binary")
            || rule.getRuleClass().equals("android_test")
            || rule.getRuleClass().equals("android_binary_internal"))
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
   * Returns all the declared Starlark wrapped providers for the specified constructor under the
   * specified attribute of this target in the BUILD file.
   */
  public <T extends Info> ImmutableList<T> getPrerequisites(
      String attributeName, StarlarkProviderWrapper<T> starlarkKey) throws RuleErrorException {
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

    Preconditions.checkNotNull(
        ruleDefinition, "%s attribute %s is not defined", ruleClassNameForLogging, attributeName);
    Preconditions.checkState(
        ruleDefinition.isExecutable(),
        "%s attribute %s is not configured to be executable",
        ruleClassNameForLogging,
        attributeName);

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
    Preconditions.checkState(
        configurationMakeVariableContext == null,
        "Attempted to init an already initialized Make var context (did you call"
            + " initConfigurationMakeVariableContext() after accessing ctx.var?)");
    configurationMakeVariableContext =
        new ConfigurationMakeVariableContext(
            this, rule.getPackage(), getConfiguration(), makeVariableSuppliers);
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
   *
   * <p>CAUTION: If there's no context, this will initialize the context with no
   * MakeVariableSuppliers. Call {@link #initConfigurationMakeVariableContext} first if you want to
   * register suppliers.
   */
  public ConfigurationMakeVariableContext getConfigurationMakeVariableContext() {
    if (configurationMakeVariableContext == null) {
      initConfigurationMakeVariableContext(ImmutableList.of());
    }
    return configurationMakeVariableContext;
  }

  private StarlarkThread createStarlarkThread(Mutability mutability) {
    AnalysisEnvironment env = getAnalysisEnvironment();
    StarlarkThread thread = new StarlarkThread(mutability, env.getStarlarkSemantics());
    thread.setPrintHandler(Event.makeDebugPrintHandler(env.getEventHandler()));
    new BazelStarlarkContext(
            BazelStarlarkContext.Phase.ANALYSIS,
            getSymbolGenerator(),
            /* analysisRuleLabel= */ getLabel())
        .storeInThread(thread);
    return thread;
  }

  public StarlarkThread getStarlarkThread() {
    return starlarkThread;
  }

  /**
   * Initializes the StarlarkRuleContext for use and returns it. No-op if already initialized.
   *
   * <p>Throws RuleErrorException on failure.
   */
  public StarlarkRuleContext initStarlarkRuleContext() throws RuleErrorException {
    if (starlarkRuleContext == null) {
      AspectDescriptor descriptor =
          aspects.isEmpty() ? null : Iterables.getLast(aspects).getDescriptor();
      this.starlarkRuleContext = new StarlarkRuleContext(this, descriptor);
    }
    return starlarkRuleContext;
  }

  public StarlarkRuleContext getStarlarkRuleContext() {
    Preconditions.checkNotNull(starlarkRuleContext, "Must call initStarlarkRuleContext() first");
    return starlarkRuleContext;
  }

  /**
   * Retrieves the {@code @_builtins}-defined Starlark object registered in the {@code
   * exported_to_java} mapping under the given name.
   *
   * <p>Reports and raises a rule error if no symbol by that name is defined.
   */
  public Object getStarlarkDefinedBuiltin(String name)
      throws RuleErrorException, InterruptedException {
    Object result = getAnalysisEnvironment().getStarlarkDefinedBuiltins().get(name);
    if (result == null) {
      throwWithRuleError(
          String.format(
              "(Internal error) No symbol named '%s' defined in the @_builtins exported_to_java"
                  + " dict",
              name));
    }
    return result;
  }

  /**
   * Calls a Starlark function in this rule's Starlark thread with the given positional and keyword
   * arguments. On failure, calls {@link #throwWithRuleError} with the Starlark stack trace.
   *
   * <p>This convenience method avoids the need to catch EvalException when the failure would just
   * immediately terminate rule analysis anyway.
   */
  public Object callStarlarkOrThrowRuleError(
      Object func, List<Object> args, Map<String, Object> kwargs)
      throws RuleErrorException, InterruptedException {
    try {
      return Starlark.call(starlarkThread, func, args, kwargs);
    } catch (EvalException e) {
      throw throwWithRuleError(e.getMessageWithStack());
    }
  }

  /**
   * Prepares Starlark objects created during this target's analysis for use by others. Freezes
   * mutability, clears expensive references.
   */
  @Override
  public void close() {
    starlarkThread.mutability().freeze();
    if (starlarkRuleContext != null) {
      starlarkRuleContext.nullify();
      starlarkRuleContext = null;
    }
  }

  @Nullable
  public Label targetPlatform() {
    if (toolchainContexts == null) {
      return null;
    }
    PlatformInfo targetPlatform = toolchainContexts.getTargetPlatform();
    if (targetPlatform == null) {
      return null;
    }
    return targetPlatform.label();
  }

  private boolean useAutoExecGroupsForRule() {
    if (attributes().has("$use_auto_exec_groups")) {
      return (boolean) attributes().get("$use_auto_exec_groups", Type.BOOLEAN);
    } else {
      return getConfiguration().useAutoExecGroups();
    }
  }

  private boolean useAutoExecGroupsForAspect(Aspect aspect) {
    ImmutableMap<String, Attribute> aspectAttributes = aspect.getDefinition().getAttributes();

    if (aspectAttributes.containsKey("$use_auto_exec_groups")) {
      return (boolean) aspectAttributes.get("$use_auto_exec_groups").getDefaultValueUnchecked();
    } else {
      return getConfiguration().useAutoExecGroups();
    }
  }

  public boolean useAutoExecGroups() {
    Aspect aspect = getMainAspect();

    if (aspect == null) {
      return useAutoExecGroupsForRule();
    } else {
      return useAutoExecGroupsForAspect(aspect);
    }
  }

  /**
   * Returns the toolchain context from the default exec group. Important note: In case automatic
   * exec groups are enabled, use `getToolchainInfo(Label toolchainType)` function.
   */
  @Nullable
  public ResolvedToolchainContext getToolchainContext() {
    return toolchainContexts == null ? null : toolchainContexts.getDefaultToolchainContext();
  }

  @Nullable
  private ResolvedToolchainContext getToolchainContext(String execGroup) {
    return toolchainContexts == null ? null : toolchainContexts.getToolchainContext(execGroup);
  }

  private boolean isAutomaticExecGroup(String execGroupName) {
    return !Identifier.isValid(execGroupName) && !execGroupName.equals(DEFAULT_EXEC_GROUP_NAME);
  }

  @Nullable
  private ResolvedToolchainContext getToolchainContextForToolchainType(Label toolchainType) {
    ResolvedToolchainContext toolchainContext =
        toolchainContexts.getToolchainContext(toolchainType.toString());
    if (toolchainContext != null && toolchainContext.forToolchainType(toolchainType) != null) {
      // Return early if name of the Automatic Exec Group (AEG) and toolchain type matches.
      return toolchainContext;
    }

    // Alias can be used for toolchains, in which case name of AEG will not match with the toolchain
    // type in its ResolvedToolchainContext (AEGs are created before toolchain context is resolved).
    String aliasName =
        toolchainContexts.getExecGroupNames().stream()
            .filter(this::isAutomaticExecGroup)
            .filter(
                name -> {
                  ResolvedToolchainContext context = toolchainContexts.getToolchainContext(name);
                  return (context != null
                      && context
                          .requestedToolchainTypeLabels()
                          .containsKey(Label.parseCanonicalUnchecked(name)));
                })
            .findFirst()
            .orElse(null);
    return aliasName == null ? null : toolchainContexts.getToolchainContext(aliasName);
  }

  /**
   * Returns the toolchain info from the default exec group in case automatic exec groups are not
   * enabled. If they are enabled, retrieves toolchain info from the corresponding automatic exec
   * group.
   */
  @Nullable
  public ToolchainInfo getToolchainInfo(Label toolchainType) {
    ResolvedToolchainContext toolchainContext;
    if (useAutoExecGroups()) {
      toolchainContext = getToolchainContextForToolchainType(toolchainType);
    } else {
      toolchainContext = getToolchainContext();
    }
    return toolchainContext == null ? null : toolchainContext.forToolchainType(toolchainType);
  }

  public boolean hasToolchainContext(String execGroup) {
    return toolchainContexts != null && toolchainContexts.hasToolchainContext(execGroup);
  }

  @Nullable
  public ToolchainCollection<ResolvedToolchainContext> getToolchainContexts() {
    return toolchainContexts;
  }

  public ExecGroupCollection getExecGroups() {
    return execGroupCollection;
  }

  public boolean targetPlatformHasConstraint(ConstraintValueInfo constraintValue) {
    if (toolchainContexts == null || toolchainContexts.getTargetPlatform() == null) {
      return false;
    }
    // All toolchain contexts should have the same target platform so we access via the default.
    return toolchainContexts.getTargetPlatform().constraints().hasConstraintValue(constraintValue);
  }

  public ConfiguredRuleClassProvider getRuleClassProvider() {
    return ruleClassProvider;
  }

  /**
   * Returns the configuration fragments this rule uses if it should be included for this rule.
   * Otherwise it returns null.
   */
  @Nullable
  public RequiredConfigFragmentsProvider getRequiredConfigFragments() {
    if (requiredConfigFragments == null) {
      return null;
    }

    RequiredConfigFragmentsProvider.Builder merged = null;

    // Add variables accessed through ctx.var, if this is a Starlark rule.
    if (starlarkRuleContext != null) {
      for (String makeVariable : starlarkRuleContext.lookedUpVariables()) {
        if (isUserDefinedMakeVariable(makeVariable)) {
          if (merged == null) {
            merged = RequiredConfigFragmentsProvider.builder().merge(requiredConfigFragments);
          }
          merged.addDefine(makeVariable);
        }
      }
    }

    // Add variables accessed through Make variable substitution.
    for (Expander makeVariableExpander : makeVariableExpanders) {
      for (String makeVariable : makeVariableExpander.lookedUpVariables()) {
        if (isUserDefinedMakeVariable(makeVariable)) {
          if (merged == null) {
            merged = RequiredConfigFragmentsProvider.builder().merge(requiredConfigFragments);
          }
          merged.addDefine(makeVariable);
        }
      }
    }

    return merged == null ? requiredConfigFragments : merged.build();
  }

  /**
   * Returns the set of transitive packages. This is only intended to be used to create the repo
   * mapping manifest for the runfiles tree. Can be null if transitive packages are not tracked (see
   * {@link
   * com.google.devtools.build.lib.skyframe.SkyframeExecutor#shouldStoreTransitivePackagesInLoadingAndAnalysis}).
   */
  @Nullable
  public NestedSet<Package> getTransitivePackagesForRunfileRepoMappingManifest() {
    return transitivePackagesForRunfileRepoMappingManifest;
  }

  private boolean isUserDefinedMakeVariable(String makeVariable) {
    // User-defined make values may be set either in "--define foo=bar" or in a vardef in the rule's
    // package. Both are equivalent for these purposes, since in both cases setting
    // "--define foo=bar" impacts the rule's output.
    return rule.getPackage().getMakeEnvironment().containsKey(makeVariable)
        || getConfiguration().getCommandLineBuildVariables().containsKey(makeVariable);
  }

  private void checkAttributeIsDependency(String attributeName) {
    Attribute attributeDefinition = attributes.getAttributeDefinition(attributeName);
    Preconditions.checkNotNull(
        attributeDefinition,
        "%s: %s attribute %s is not defined",
        rule.getLocation(),
        ruleClassNameForLogging,
        attributeName);
    Preconditions.checkState(
        attributeDefinition.getType().getLabelClass() == LabelClass.DEPENDENCY,
        "%s attribute %s is not a label type attribute",
        ruleClassNameForLogging,
        attributeName);
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
    if (toolchainContexts == null) {
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

  @Nullable
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
   * Returns the sole file in the "srcs" attribute. Reports an error and (possibly) returns null if
   * "srcs" does not identify a single file of the expected type.
   */
  @Nullable
  public Artifact getSingleSource(String fileTypeName) {
    List<Artifact> srcs = PrerequisiteArtifacts.get(this, "srcs").list();
    switch (srcs.size()) {
      case 0: // error already issued by getSrc()
        return null;
      case 1: // ok
        return Iterables.getOnlyElement(srcs);
      default:
        attributeError("srcs", "only a single " + fileTypeName + " is allowed here");
        return srcs.get(0);
    }
  }

  public Artifact getSingleSource() {
    return getSingleSource(ruleClassNameForLogging + " source file");
  }

  /**
   * Returns a path fragment qualified by the rule name and unique fragment to disambiguate
   * artifacts produced from the source file appearing in multiple rules.
   *
   * <p>For example "pkg/dir/name" -> "pkg/&lt;fragment>/rule/dir/name.
   */
  public PathFragment getUniqueDirectory(String fragment) {
    return getUniqueDirectory(PathFragment.create(fragment));
  }

  /**
   * Returns a path fragment qualified by the rule name and unique fragment to disambiguate
   * artifacts produced from the source file appearing in multiple rules.
   *
   * <p>For example "pkg/dir/name" -> "pkg/&lt;fragment>/rule/dir/name.
   */
  @Override
  public PathFragment getUniqueDirectory(PathFragment fragment) {
    return AnalysisUtils.getUniqueDirectory(
        getLabel(), fragment, getConfiguration().isSiblingRepositoryLayout());
  }

  /**
   * Check that all targets that were specified as sources are from the same package as this rule.
   * Output a warning or an error for every target that is imported from a different package.
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
        String message =
            "please do not import '"
                + associatedLabel
                + "' directly. "
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
   * Returns the label to which the {@code NODEP_LABEL} attribute {@code attrName} refers, checking
   * that it is a valid label, and that it is referring to a local target. Reports a warning
   * otherwise.
   */
  @Nullable
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
    return getImplicitOutputArtifact(function, /* contentBasedPath= */ false);
  }

  /**
   * Same as {@link #getImplicitOutputArtifact(ImplicitOutputsFunction)} but includes the option to
   * use a content-based path for this artifact (see {@link
   * BuildConfigurationValue#useContentBasedOutputPaths()}).
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
    return getImplicitOutputArtifact(path, /* contentBasedPath= */ false);
  }

  /**
   * Same as {@link #getImplicitOutputArtifact(String)} but includes the option to use a a
   * content-based path for this artifact (see {@link
   * BuildConfigurationValue#useContentBasedOutputPaths()}).
   */
  // TODO(bazel-team): Consider removing contentBasedPath stuff, which is unused as of 18 months
  // after its introduction in cl/252148134.
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
  public FilesToRunProvider getCompiler(boolean warnIfNotDefault) {
    Label label = attributes().get("compiler", BuildType.LABEL);
    if (warnIfNotDefault && !label.equals(rule.getAttrDefaultValue("compiler"))) {
      attributeWarning("compiler", "setting the compiler is strongly discouraged");
    }
    return getExecutablePrerequisite("compiler");
  }

  /**
   * Returns the (unmodifiable, ordered) list of artifacts which are the outputs of this target.
   *
   * <p>Each element in this list is associated with a single output, either declared implicitly
   * (via setImplicitOutputsFunction()) or explicitly (listed in the 'outs' attribute of our rule).
   */
  public ImmutableList<Artifact> getOutputArtifacts() {
    ImmutableList.Builder<Artifact> artifacts = ImmutableList.builder();
    for (OutputFile out : rule.getOutputFiles()) {
      artifacts.add(createOutputArtifact(out));
    }
    return artifacts.build();
  }

  /**
   * Like {@link #getOutputArtifacts()} but for a singular output item. Reports an error if the
   * "out" attribute is not a singleton.
   *
   * @return null if the output list is empty, the artifact for the first item of the output list
   *     otherwise
   */
  @Nullable
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
  public Artifact.DerivedArtifact getRelatedArtifact(PathFragment pathFragment, String extension) {
    PathFragment file = FileSystemUtils.replaceExtension(pathFragment, extension);
    return getDerivedArtifact(file, getConfiguration().getBinDirectory(getLabel().getRepository()));
  }

  /** Returns true if the target for this context is a test target. */
  public boolean isTestTarget() {
    return TargetUtils.isTestRule(getTarget());
  }

  /** Returns true if the testonly attribute is set on this context. */
  public boolean isTestOnlyTarget() {
    return attributes().has("testonly", Type.BOOLEAN) && attributes().get("testonly", Type.BOOLEAN);
  }

  /** Returns true if the execution platform is Windows. */
  public boolean isExecutedOnWindows() {
    return getExecutionPlatform()
        .constraints()
        .hasConstraintValue(OS_TO_CONSTRAINTS.get(OS.WINDOWS));
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

  /**
   * @return the set of features applicable for the current rule.
   */
  public ImmutableSet<String> getFeatures() {
    return features.on();
  }

  /**
   * @return the set of features that are disabled for the current rule.
   */
  public ImmutableSet<String> getDisabledFeatures() {
    return features.off();
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
    return requiredConfigFragments != null;
  }

  @Override
  public String toString() {
    return "RuleContext(" + getLabel() + ", " + getConfiguration() + ")";
  }

  /** Builder class for a RuleContext. */
  public static final class Builder implements RuleErrorConsumer {
    private final AnalysisEnvironment env;
    private final Target target;
    private final ImmutableList<Aspect> aspects;
    private final BuildConfigurationValue configuration;
    private final RuleErrorConsumer reporter;
    private ConfiguredRuleClassProvider ruleClassProvider;
    private ConfigurationFragmentPolicy configurationFragmentPolicy;
    private ActionLookupKey actionOwnerSymbol;
    private OrderedSetMultimap<Attribute, ConfiguredTargetAndData> prerequisiteMap;
    private ConfigConditions configConditions;
    private Mutability mutability;
    private NestedSet<PackageGroupContents> visibility;
    private ImmutableMap<String, Attribute> aspectAttributes = ImmutableMap.of();
    private ToolchainCollection<ResolvedToolchainContext> toolchainContexts;
    private ExecGroupCollection.Builder execGroupCollectionBuilder;
    private ImmutableMap<String, String> rawExecProperties;
    @Nullable private RequiredConfigFragmentsProvider requiredConfigFragments;
    @Nullable private NestedSet<Package> transitivePackagesForRunfileRepoMappingManifest;

    @VisibleForTesting
    public Builder(
        AnalysisEnvironment env,
        Target target,
        ImmutableList<Aspect> aspects,
        BuildConfigurationValue configuration) {
      this.env = Preconditions.checkNotNull(env);
      this.target = Preconditions.checkNotNull(target);
      this.aspects = Preconditions.checkNotNull(aspects);
      this.configuration = Preconditions.checkNotNull(configuration);
      if (configuration.allowAnalysisFailures()) {
        reporter = new SuppressingErrorReporter();
      } else {
        reporter =
            new ErrorReporter(
                env, target.getAssociatedRule(), configuration, getRuleClassNameForLogging());
      }
    }

    /**
     * Same as {@link #build}, except without some attribute checks.
     *
     * <p>Don't use this function outside of testing. The use should be limited to cases where
     * specifying ConfigConditions.EMPTY, which can cause a noMatchError when accessing attributes
     * within attribute checking.
     */
    @VisibleForTesting
    public RuleContext unsafeBuild() throws InvalidExecGroupException {
      return build(false);
    }

    @VisibleForTesting
    public RuleContext build() throws InvalidExecGroupException {
      return build(true);
    }

    private RuleContext build(boolean attributeChecks) throws InvalidExecGroupException {
      Preconditions.checkNotNull(ruleClassProvider);
      Preconditions.checkNotNull(configurationFragmentPolicy);
      Preconditions.checkNotNull(actionOwnerSymbol);
      Preconditions.checkNotNull(prerequisiteMap);
      Preconditions.checkNotNull(configConditions);
      Preconditions.checkNotNull(mutability);
      Preconditions.checkNotNull(visibility);
      ConfiguredAttributeMapper attributes =
          ConfiguredAttributeMapper.of(
              target.getAssociatedRule(), configConditions.asProviders(), configuration);
      ListMultimap<String, ConfiguredTargetAndData> targetMap = createTargetMap();
      validateExtraPrerequisites(attributeChecks, attributes);

      return new RuleContext(
          this,
          attributes,
          targetMap,
          createExecGroupCollection(execGroupCollectionBuilder, attributes));
    }

    private ExecGroupCollection createExecGroupCollection(
        ExecGroupCollection.Builder execGroupCollectionBuilder, AttributeMap attributes)
        throws InvalidExecGroupException {
      if (rawExecProperties == null) {
        if (!attributes.has(RuleClass.EXEC_PROPERTIES_ATTR, Type.STRING_DICT)) {
          rawExecProperties = ImmutableMap.of();
        } else {
          rawExecProperties =
              ImmutableMap.copyOf(attributes.get(RuleClass.EXEC_PROPERTIES_ATTR, Type.STRING_DICT));
        }
      }

      return execGroupCollectionBuilder.build(toolchainContexts, rawExecProperties);
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
        if (attributeValue instanceof List) {
          isEmpty = ((List<?>) attributeValue).isEmpty();
        } else if (attributeValue instanceof Map) {
          isEmpty = ((Map<?, ?>) attributeValue).isEmpty();
        }
        if (isEmpty) {
          reporter.attributeError(attr.getName(), "attribute must be non empty");
        }
      }
    }

    private void checkAttributesForDuplicateLabels(ConfiguredAttributeMapper attributes) {
      for (String attributeName : attributes.getAttributeNames()) {
        Attribute attr = attributes.getAttributeDefinition(attributeName);
        if (attr.getType() != BuildType.LABEL_LIST) {
          continue;
        }

        Set<Label> duplicates = attributes.checkForDuplicateLabels(attr);
        for (Label label : duplicates) {
          reporter.attributeError(attr.getName(), String.format("Label '%s' is duplicated", label));
        }
      }
    }

    @CanIgnoreReturnValue
    public Builder setRuleClassProvider(ConfiguredRuleClassProvider ruleClassProvider) {
      this.ruleClassProvider = ruleClassProvider;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setConfigurationFragmentPolicy(ConfigurationFragmentPolicy policy) {
      this.configurationFragmentPolicy = policy;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setActionOwnerSymbol(ActionLookupKey actionOwnerSymbol) {
      this.actionOwnerSymbol = actionOwnerSymbol;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setMutability(Mutability mutability) {
      this.mutability = mutability;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setVisibility(NestedSet<PackageGroupContents> visibility) {
      this.visibility = visibility;
      return this;
    }

    /**
     * Sets the prerequisites and checks their visibility. It also generates appropriate error or
     * warning messages and sets the error flag as appropriate.
     */
    @CanIgnoreReturnValue
    public Builder setPrerequisites(
        OrderedSetMultimap<Attribute, ConfiguredTargetAndData> prerequisiteMap) {
      this.prerequisiteMap = Preconditions.checkNotNull(prerequisiteMap);
      return this;
    }

    /** Adds attributes which are defined by an Aspect (and not by RuleClass). */
    @CanIgnoreReturnValue
    public Builder setAspectAttributes(Map<String, Attribute> aspectAttributes) {
      this.aspectAttributes = ImmutableMap.copyOf(aspectAttributes);
      return this;
    }

    /**
     * Sets the configuration conditions needed to determine which paths to follow for this rule's
     * configurable attributes.
     */
    @CanIgnoreReturnValue
    public Builder setConfigConditions(ConfigConditions configConditions) {
      this.configConditions = Preconditions.checkNotNull(configConditions);
      return this;
    }

    /** Sets the collection of {@link ResolvedToolchainContext}s available to this rule. */
    @CanIgnoreReturnValue
    @VisibleForTesting
    public Builder setToolchainContexts(
        ToolchainCollection<ResolvedToolchainContext> toolchainContexts) {
      Preconditions.checkState(
          this.toolchainContexts == null,
          "toolchainContexts has already been set for this Builder");
      this.toolchainContexts = toolchainContexts;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setExecGroupCollectionBuilder(
        ExecGroupCollection.Builder execGroupCollectionBuilder) {
      this.execGroupCollectionBuilder = execGroupCollectionBuilder;
      return this;
    }

    /**
     * Warning: if you set the exec properties using this method any exec_properties attribute value
     * will be ignored in favor of this value.
     */
    @CanIgnoreReturnValue
    public Builder setExecProperties(ImmutableMap<String, String> execProperties) {
      this.rawExecProperties = execProperties;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setRequiredConfigFragments(
        @Nullable RequiredConfigFragmentsProvider requiredConfigFragments) {
      this.requiredConfigFragments = requiredConfigFragments;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setTransitivePackagesForRunfileRepoMappingManifest(
        @Nullable NestedSet<Package> packages) {
      this.transitivePackagesForRunfileRepoMappingManifest = packages;
      return this;
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
          Predicate<String> filter = attribute.getAllowedRuleClassPredicate();
          for (ConfiguredTargetAndData configuredTarget : entry.getValue()) {
            if (filter.apply(configuredTarget.getRuleClass())) {
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

    private static String badPrerequisiteMessage(
        ConfiguredTargetAndData prerequisite, String reason, boolean isWarning) {
      String msgReason = reason != null ? " (" + reason + ")" : "";
      if (isWarning) {
        return String.format(
            "%s is unexpected here%s; continuing anyway",
            AliasProvider.describeTargetWithAliases(prerequisite, TargetMode.WITH_KIND), msgReason);
      }
      return String.format(
          "%s is misplaced here%s",
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
      String ruleClass = prerequisite.getRuleClass();
      if (!ruleClass.isEmpty()) {
        String reason =
            attribute
                .getValidityPredicate()
                .checkValid(target.getAssociatedRule(), ruleClass, prerequisite.getRuleTags());
        if (reason != null) {
          reportBadPrerequisite(attribute, prerequisite, reason, false);
        }
        validateRuleDependency(prerequisite, attribute);
        return;
      }

      if (!(prerequisite.isTargetFile() && attribute.isStrictLabelCheckingEnabled())) {
        return;
      }

      Label prerequisiteTargetLabel = prerequisite.getTargetLabel();
      if (attribute.getAllowedFileTypesPredicate().apply(prerequisiteTargetLabel.getName())) {
        return;
      }

      if (prerequisite.isTargetInputFile() && !prerequisite.getInputPath().exists()) {
        // Misplaced labels, no corresponding target exists
        if (attribute.getAllowedFileTypesPredicate().isNone()
            && !prerequisiteTargetLabel.getName().contains(".")) {
          // There are no allowed files in the attribute but it's not a valid rule,
          // and the filename doesn't contain a dot --> probably a misspelled rule
          attributeError(
              attribute.getName(), "rule '" + prerequisiteTargetLabel + "' does not exist");
        } else {
          attributeError(
              attribute.getName(), "target '" + prerequisiteTargetLabel + "' does not exist");
        }
        return;
      }
      // The file exists but has a bad extension
      reportBadPrerequisite(
          attribute, prerequisite, "expected " + attribute.getAllowedFileTypesPredicate(), false);
    }

    /** Returns whether the context being constructed is for the evaluation of an aspect. */
    public boolean forAspect() {
      return !aspects.isEmpty();
    }

    public Rule getRule() {
      return target.getAssociatedRule();
    }

    /**
     * Returns the {@link StarlarkSemantics} governs the building of this rule (and the rest of the
     * build).
     */
    public StarlarkSemantics getStarlarkSemantics() {
      return env.getStarlarkSemantics();
    }

    /**
     * Returns a rule class name suitable for log messages, including an aspect name if applicable.
     */
    private String getRuleClassNameForLogging() {
      if (aspects.isEmpty()) {
        return target.getAssociatedRule().getRuleClass();
      }

      return Joiner.on(",")
              .join(aspects.stream().map(Aspect::getDescriptor).collect(Collectors.toList()))
          + " aspect on "
          + target.getAssociatedRule().getRuleClass();
    }

    public BuildConfigurationValue getConfiguration() {
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
      if (allowedFileTypes == FileTypeSet.ANY_FILE
          && !attribute.isNonEmpty()
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
              "'" + prerequisite.getTargetLabel() + "' must produce a single file");
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
                + prerequisite.getTargetLabel()
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
    private static boolean checkRuleDependencyClass(
        ConfiguredTargetAndData prerequisite,
        Attribute attribute,
        Set<String> unfulfilledRequirements) {
      var predicate = attribute.getAllowedRuleClassPredicate();
      if (predicate == Predicates.<String>alwaysTrue()) {
        // alwaysTrue is a special sentinel value. See
        // RuleClass.Builder.RuleClassNamePredicate.unspecified.
        return false;
      }

      if (predicate.apply(prerequisite.getRuleClass())) {
        // prerequisite has an allowed rule class => accept.
        return true;
      }
      // remember that the rule class that was not allowed;
      // but maybe prerequisite provides required providers? do not reject yet.
      unfulfilledRequirements.add(
          badPrerequisiteMessage(prerequisite, "expected " + predicate, false));
      return false;
    }

    /**
     * Check if prerequisite should be allowed with warning based on its rule class.
     *
     * <p>If yes, also issues said warning.
     */
    private boolean checkRuleDependencyClassWarnings(
        ConfiguredTargetAndData prerequisite, Attribute attribute) {
      if (!attribute.getAllowedRuleClassWarningPredicate().apply(prerequisite.getRuleClass())) {
        return false;
      }

      Predicate<String> allowedRuleClasses = attribute.getAllowedRuleClassPredicate();
      reportBadPrerequisite(
          attribute,
          prerequisite,
          allowedRuleClasses == Predicates.<String>alwaysTrue()
              ? null
              : "expected " + allowedRuleClasses,
          true);
      // prerequisite has a rule class allowed with a warning => accept, emitting a warning.
      return true;
    }

    /** Check if prerequisite should be allowed based on required providers on the attribute. */
    private static boolean checkRuleDependencyMandatoryProviders(
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
              prerequisite.getTargetLabel(),
              prerequisite
                  .getConfiguredTarget()
                  .missingProviders(requiredProviders)
                  .getDescription()));

      return false;
    }

    /**
     * Perform extra validation of prerequisites. Standard attribute-based dependencies are already
     * validated as part of {@link #createTargetMap}.
     */
    private void validateExtraPrerequisites(
        boolean attributeChecks, ConfiguredAttributeMapper attributes) {
      // These checks can fail when ConfigConditions.EMPTY are empty, resulting in noMatchError
      // accessing attributes without a default condition.
      // ConfigConditions.EMPTY is always true for non-rules:
      // https://cs.opensource.google/bazel/bazel/+/master:src/main/java/com/google/devtools/build/lib/skyframe/ConfiguredTargetFunction.java;l=943;drc=720dc5fd640de692db129777c7c7c32924627c43
      // This can happen in BuildViewForTesting.getRuleContextForTesting as it specifies
      // ConfigConditions.EMPTY.
      if (attributeChecks && target instanceof Rule) {
        checkAttributesNonEmpty(attributes);
        checkAttributesForDuplicateLabels(attributes);
      }

      // This conditionally checks visibility on config_setting rules based on
      // --config_setting_visibility_policy. This should be removed as soon as it's deemed safe
      // to unconditionally check visibility. See
      // https://github.com/bazelbuild/bazel/issues/12669.
      ConfigSettingVisibilityPolicy configSettingVisibilityPolicy =
          target.getPackage().getConfigSettingVisibilityPolicy();
      if (configSettingVisibilityPolicy != ConfigSettingVisibilityPolicy.LEGACY_OFF) {

        // Validate config conditions.
        Attribute configSettingAttr = attributes.getAttributeDefinition("$config_dependencies");
        for (ConfiguredTargetAndData condition : configConditions.asConfiguredTargets().values()) {
          validateDirectPrerequisite(
              configSettingAttr,
              // Another nuance: when both --incompatible_enforce_config_setting_visibility and
              // --incompatible_config_setting_private_default_visibility are disabled, both of
              // these are ignored:
              //
              //  - visibility settings on a select() -> config_setting dep
              //  - visibility settings on a select() -> alias -> config_setting dep chain
              //
              // In that scenario, both are ignored because the logic here that checks the
              // select() -> ??? edge is completely skipped.
              //
              // When just --incompatible_enforce_config_setting_visibility is on, that means
              // "enforce config_setting visibility with public default". That's a temporary state
              // to support depot migration. In that case, we continue to ignore the alias'
              // visibility in preference for the config_setting. So skip select() -> alias as
              // before, but now enforce select() -> config_setting_the_alias_refers_to.
              //
              // When we also turn on --incompatible_config_setting_private_default_visibility, we
              // expect full standard visibility compliance. In that case we directly evaluate the
              // alias visibility, as is usual semantics. So two the following two edges are
              // checked: 1: select() -> alias and 2: alias -> config_setting.
              configSettingVisibilityPolicy == ConfigSettingVisibilityPolicy.DEFAULT_PUBLIC
                  ? condition.fromConfiguredTargetNoCheck(
                      condition.getConfiguredTarget().getActual())
                  : condition);
        }
      }

      // Validate toolchains.
      if (toolchainContexts != null) {
        for (var toolchainContext : toolchainContexts.getContextMap().values()) {
          for (var prerequisite : toolchainContext.prerequisiteTargets()) {
            validateDirectPrerequisite(TOOLCHAIN_ATTRIBUTE, prerequisite);
          }
        }
      }
    }

    private void validateDirectPrerequisite(
        Attribute attribute, ConfiguredTargetAndData prerequisite) {
      validateDirectPrerequisiteType(prerequisite, attribute);
      validateDirectPrerequisiteFileTypes(prerequisite, attribute);
      if (attribute.performPrereqValidatorCheck()) {
        ruleClassProvider.getPrerequisiteValidator().validate(this, prerequisite, attribute);
      }
    }
  }

  /** Helper class for reporting errors and warnings. */
  private static final class ErrorReporter extends EventHandlingErrorReporter
      implements RuleErrorConsumer {
    private final Rule rule;
    private final BuildConfigurationValue configuration;

    ErrorReporter(
        AnalysisEnvironment env,
        Rule rule,
        BuildConfigurationValue configuration,
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
    protected BuildConfigurationValue getConfiguration() {
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

    /** Returns the error message strings reported to this error consumer. */
    public List<String> getErrorMessages() {
      return errorMessages;
    }
  }
}
