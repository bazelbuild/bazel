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

package com.google.devtools.build.lib.analysis.config;

import static java.util.Comparator.comparing;
import static java.util.stream.Collectors.joining;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Suppliers;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ClassToInstanceMap;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Interner;
import com.google.common.collect.Multimap;
import com.google.common.collect.MutableClassToInstanceMap;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.BuildConfigurationEvent;
import com.google.devtools.build.lib.actions.CommandLines.CommandLineLimits;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.BuildConfigurationApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkInterfaceUtils;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * Instances of BuildConfiguration represent a collection of context information which may affect a
 * build (for example: the target platform for compilation, or whether or not debug tables are
 * required). In fact, all "environmental" information (e.g. from the tool's command-line, as
 * opposed to the BUILD file) that can affect the output of any build tool should be explicitly
 * represented in the BuildConfiguration instance.
 *
 * <p>A single build may require building tools to run on a variety of platforms: when compiling a
 * server application for production, we must build the build tools (like compilers) to run on the
 * host platform, but cross-compile the application for the production environment.
 *
 * <p>There is always at least one BuildConfiguration instance in any build: the one representing
 * the host platform. Additional instances may be created, in a cross-compilation build, for
 * example.
 *
 * <p>Instances of BuildConfiguration are canonical:
 *
 * <pre>c1.equals(c2) <=> c1==c2.</pre>
 */
// TODO(janakr): If overhead of fragments class names is too high, add constructor that just takes
// fragments and gets names from them.
@AutoCodec
public class BuildConfiguration implements BuildConfigurationApi {
  /**
   * Sorts fragments by class name. This produces a stable order which, e.g., facilitates consistent
   * output from buildMnemonic.
   */
  @AutoCodec
  public static final Comparator<Class<? extends Fragment>> lexicalFragmentSorter =
      Comparator.comparing(Class::getName);

  private static final Interner<ImmutableSortedMap<Class<? extends Fragment>, Fragment>>
      fragmentsInterner = BlazeInterners.newWeakInterner();

  private static final Interner<ImmutableSortedMap<String, String>> executionInfoInterner =
      BlazeInterners.newWeakInterner();

  /** Compute the default shell environment for actions from the command line options. */
  public interface ActionEnvironmentProvider {
    ActionEnvironment getActionEnvironment(BuildOptions options);
  }

  /**
   * An interface for language-specific configurations.
   *
   * <p>All implementations must be immutable and communicate this as clearly as possible (e.g.
   * declare {@link ImmutableList} signatures on their interfaces vs. {@link List}). This is because
   * fragment instances may be shared across configurations.
   */
  public abstract static class Fragment {
    /**
     * Validates the options for this Fragment. Issues warnings for the use of deprecated options,
     * and warnings or errors for any option settings that conflict.
     */
    @SuppressWarnings("unused")
    public void reportInvalidOptions(EventHandler reporter, BuildOptions buildOptions) {}

    /**
     * Returns a fragment of the output directory name for this configuration. The output directory
     * for the whole configuration contains all the short names by all fragments.
     */
    @Nullable
    public String getOutputDirectoryName() {
      return null;
    }
  }

  private final OutputDirectories outputDirectories;

  private final ImmutableSortedMap<Class<? extends Fragment>, Fragment> fragments;
  private final FragmentClassSet fragmentClassSet;

  private final ImmutableMap<String, Class<? extends Fragment>> skylarkVisibleFragments;
  private final RepositoryName mainRepositoryName;
  private final ImmutableSet<String> reservedActionMnemonics;
  private CommandLineLimits commandLineLimits;

  /**
   * The global "make variables" such as "$(TARGET_CPU)"; these get applied to all rules analyzed in
   * this configuration.
   */
  private final ImmutableMap<String, String> globalMakeEnv;

  private final ActionEnvironment actionEnv;
  private final ActionEnvironment testEnv;

  private final BuildOptions buildOptions;
  private final BuildOptions.OptionsDiffForReconstruction buildOptionsDiff;
  private final CoreOptions options;

  private final ImmutableMap<String, String> commandLineBuildVariables;

  private final String checksum;
  private final int hashCode; // We can precompute the hash code as all its inputs are immutable.

  /** Data for introspecting the options used by this configuration. */
  private final TransitiveOptionDetails transitiveOptionDetails;

  private final Supplier<BuildConfigurationEvent> buildEventSupplier;

  /**
   * Returns true if this configuration is semantically equal to the other, with the possible
   * exception that the other has fewer fragments.
   *
   * <p>This is useful for trimming: as the same configuration gets "trimmed" while going down a
   * dependency chain, it's still the same configuration but loses some of its fragments. So we need
   * a more nuanced concept of "equality" than simple reference equality.
   */
  // TODO(b/121048710): make this reflect starlark options
  public boolean equalsOrIsSupersetOf(BuildConfiguration other) {
    return this.equals(other)
        || (other != null
            // TODO(gregce): add back in output root checking. This requires a better approach to
            // configuration-safe output paths. If the parent config has a fragment the child config
            // doesn't, it may inject $(FOO) into the output roots. So the child bindir might be
            // "bazel-out/arm-linux-fastbuild/bin" while the parent bindir is
            // "bazel-out/android-arm-linux-fastbuild/bin". That's pretty awkward to check here.
            //      && outputRoots.equals(other.outputRoots)
            && fragments.values().containsAll(other.fragments.values())
            && buildOptions.getNativeOptions().containsAll(other.buildOptions.getNativeOptions()));
  }

  /**
   * Returns {@code true} if this configuration is semantically equal to the other, including
   * checking that both have the same sets of fragments and options.
   */
  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof BuildConfiguration)) {
      return false;
    }
    BuildConfiguration otherConfig = (BuildConfiguration) other;
    return fragments.values().equals(otherConfig.fragments.values())
        && buildOptions.equals(otherConfig.buildOptions);
  }

  private int computeHashCode() {
    return Objects.hash(fragments, buildOptions.getNativeOptions());
  }

  public void describe(StringBuilder sb) {
    sb.append("BuildConfiguration ").append(checksum()).append(":\n");
    // Fragments.
    sb.append("  fragments: ")
        .append(
            getFragmentsMap().keySet().stream()
                .sorted(comparing(Class::getName))
                .map(Class::getName)
                .collect(joining(",")))
        .append("\n");
    // Options.
    getOptions().getFragmentClasses().stream()
        .sorted(comparing(Class::getName))
        .map(optionsClass -> getOptions().get(optionsClass))
        .forEach(options -> options.describe(sb));
  }

  @Override
  public int hashCode() {
    return hashCode;
  }

  /** Returns map of all the fragments for this configuration. */
  public ImmutableMap<Class<? extends Fragment>, Fragment> getFragmentsMap() {
    return fragments;
  }

  /**
   * Validates the options for this BuildConfiguration. Issues warnings for the use of deprecated
   * options, and warnings or errors for any option settings that conflict.
   */
  public void reportInvalidOptions(EventHandler reporter) {
    for (Fragment fragment : fragments.values()) {
      fragment.reportInvalidOptions(reporter, this.buildOptions);
    }

    if (options.outputDirectoryName != null) {
      reporter.handle(
          Event.error(
              "The internal '--output directory name' option cannot be used on the command line"));
    }
  }

  /**
   * Compute the test environment, which, at configuration level, is a pair consisting of the
   * statically set environment variables with their values and the set of environment variables to
   * be inherited from the client environment.
   */
  private ActionEnvironment setupTestEnvironment() {
    // We make a copy first to remove duplicate entries; last one wins.
    Map<String, String> testEnv = new HashMap<>();
    for (Map.Entry<String, String> entry : options.testEnvironment) {
      testEnv.put(entry.getKey(), entry.getValue());
    }
    return ActionEnvironment.split(testEnv);
  }

  private static ImmutableSortedMap<Class<? extends Fragment>, Fragment> makeFragmentsMap(
      Map<Class<? extends Fragment>, Fragment> fragmentsMap) {
    return fragmentsInterner.intern(ImmutableSortedMap.copyOf(fragmentsMap, lexicalFragmentSorter));
  }

  /** Constructs a new BuildConfiguration instance. */
  public BuildConfiguration(
      BlazeDirectories directories,
      Map<Class<? extends Fragment>, Fragment> fragmentsMap,
      BuildOptions buildOptions,
      BuildOptions.OptionsDiffForReconstruction buildOptionsDiff,
      ImmutableSet<String> reservedActionMnemonics,
      ActionEnvironment actionEnvironment,
      String repositoryName) {
    this(
        directories,
        fragmentsMap,
        buildOptions,
        buildOptionsDiff,
        reservedActionMnemonics,
        actionEnvironment,
        RepositoryName.createFromValidStrippedName(repositoryName));
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec.Instantiator
  BuildConfiguration(
      BlazeDirectories directories,
      Map<Class<? extends Fragment>, Fragment> fragmentsMap,
      BuildOptions buildOptions,
      BuildOptions.OptionsDiffForReconstruction buildOptionsDiff,
      ImmutableSet<String> reservedActionMnemonics,
      ActionEnvironment actionEnvironment,
      RepositoryName mainRepositoryName) {
    // this.directories = directories;
    this.fragments = makeFragmentsMap(fragmentsMap);
    this.fragmentClassSet = FragmentClassSet.of(this.fragments.keySet());
    this.skylarkVisibleFragments = buildIndexOfSkylarkVisibleFragments();
    this.buildOptions = buildOptions.clone();
    this.buildOptionsDiff = buildOptionsDiff;
    this.options = buildOptions.get(CoreOptions.class);
    this.outputDirectories =
        new OutputDirectories(directories, options, fragments, mainRepositoryName);
    this.mainRepositoryName = mainRepositoryName;

    // We can't use an ImmutableMap.Builder here; we need the ability to add entries with keys that
    // are already in the map so that the same define can be specified on the command line twice,
    // and ImmutableMap.Builder does not support that.
    Map<String, String> commandLineDefinesBuilder = new TreeMap<>();
    for (Map.Entry<String, String> define : options.commandLineBuildVariables) {
      commandLineDefinesBuilder.put(define.getKey(), define.getValue());
    }
    commandLineBuildVariables = ImmutableMap.copyOf(commandLineDefinesBuilder);

    this.actionEnv = actionEnvironment;
    this.testEnv = setupTestEnvironment();
    this.transitiveOptionDetails =
        TransitiveOptionDetails.forOptions(
            buildOptions.getNativeOptions(), buildOptions.getStarlarkOptions());

    ImmutableMap.Builder<String, String> globalMakeEnvBuilder = ImmutableMap.builder();

    // TODO(configurability-team): Deprecate TARGET_CPU in favor of platforms.
    globalMakeEnvBuilder.put("TARGET_CPU", options.cpu);
    globalMakeEnvBuilder.put("COMPILATION_MODE", options.compilationMode.toString());

    /*
     * Attention! Document these in the build-encyclopedia
     */
    // the bin directory and the genfiles directory
    // These variables will be used on Windows as well, so we need to make sure
    // that paths use the correct system file-separator.
    globalMakeEnvBuilder.put("BINDIR", getBinDirectory().getExecPath().getPathString());
    globalMakeEnvBuilder.put("GENDIR", getGenfilesDirectory().getExecPath().getPathString());
    globalMakeEnv = globalMakeEnvBuilder.build();

    checksum = buildOptions.computeChecksum();
    hashCode = computeHashCode();

    this.reservedActionMnemonics = reservedActionMnemonics;
    this.buildEventSupplier = Suppliers.memoize(this::createBuildEvent);
    this.commandLineLimits = new CommandLineLimits(options.minParamFileSize);
  }

  /**
   * Returns a copy of this configuration only including the given fragments (which the current
   * configuration is assumed to have).
   */
  public BuildConfiguration clone(
      FragmentClassSet fragmentClasses,
      RuleClassProvider ruleClassProvider,
      BuildOptions defaultBuildOptions) {

    ClassToInstanceMap<Fragment> fragmentsMap = MutableClassToInstanceMap.create();
    for (Fragment fragment : fragments.values()) {
      if (fragmentClasses.fragmentClasses().contains(fragment.getClass())) {
        fragmentsMap.put(fragment.getClass(), fragment);
      }
    }
    BuildOptions options =
        buildOptions.trim(getOptionsClasses(fragmentsMap.keySet(), ruleClassProvider));
    BuildConfiguration newConfig =
        new BuildConfiguration(
            getDirectories(),
            fragmentsMap,
            options,
            BuildOptions.diffForReconstruction(defaultBuildOptions, options),
            reservedActionMnemonics,
            actionEnv,
            mainRepositoryName.strippedName());
    return newConfig;
  }

  /** Returns the config fragment options classes used by the given fragment types. */
  public static Set<Class<? extends FragmentOptions>> getOptionsClasses(
      Iterable<Class<? extends Fragment>> fragmentClasses, RuleClassProvider ruleClassProvider) {

    Multimap<Class<? extends BuildConfiguration.Fragment>, Class<? extends FragmentOptions>>
        fragmentToRequiredOptions = ArrayListMultimap.create();
    for (ConfigurationFragmentFactory fragmentLoader :
        ((ConfiguredRuleClassProvider) ruleClassProvider).getConfigurationFragments()) {
      fragmentToRequiredOptions.putAll(fragmentLoader.creates(), fragmentLoader.requiredOptions());
    }
    Set<Class<? extends FragmentOptions>> options = new HashSet<>();
    for (Class<? extends BuildConfiguration.Fragment> fragmentClass : fragmentClasses) {
      options.addAll(fragmentToRequiredOptions.get(fragmentClass));
    }
    return options;
  }

  private ImmutableMap<String, Class<? extends Fragment>> buildIndexOfSkylarkVisibleFragments() {
    ImmutableMap.Builder<String, Class<? extends Fragment>> builder = ImmutableMap.builder();

    for (Class<? extends Fragment> fragmentClass : fragments.keySet()) {
      SkylarkModule module = SkylarkInterfaceUtils.getSkylarkModule(fragmentClass);
      if (module != null) {
        builder.put(module.name(), fragmentClass);
      }
    }
    return builder.build();
  }

  /**
   * Retrieves the {@link TransitiveOptionDetails} containing data on this configuration's options.
   *
   * @see BuildConfigurationOptionDetails
   */
  TransitiveOptionDetails getTransitiveOptionDetails() {
    return transitiveOptionDetails;
  }

  /** Returns the output directory for this build configuration. */
  public ArtifactRoot getOutputDirectory(RepositoryName repositoryName) {
    return outputDirectories.getOutputDirectory();
  }

  @Override
  public ArtifactRoot getBinDir() {
    return outputDirectories.getBinDirectory();
  }

  /** Returns the bin directory for this build configuration. */
  public ArtifactRoot getBinDirectory() {
    return outputDirectories.getBinDirectory();
  }

  /**
   * TODO(kchodorow): This (and the other get*Directory functions) won't work with external
   * repositories without changes to how ArtifactFactory resolves derived roots. This is not an
   * issue right now because it only effects Blaze's include scanning (internal) and Bazel's
   * repositories (external) but will need to be fixed.
   */
  public ArtifactRoot getBinDirectory(RepositoryName repositoryName) {
    return outputDirectories.getBinDirectory();
  }

  /** Returns a relative path to the bin directory at execution time. */
  public PathFragment getBinFragment() {
    return outputDirectories.getBinDirectory().getExecPath();
  }

  /** Returns the include directory for this build configuration. */
  public ArtifactRoot getIncludeDirectory(RepositoryName repositoryName) {
    return outputDirectories.getIncludeDirectory();
  }

  @Override
  public ArtifactRoot getGenfilesDir() {
    return outputDirectories.getGenfilesDirectory();
  }

  /** Returns the genfiles directory for this build configuration. */
  public ArtifactRoot getGenfilesDirectory() {
    return outputDirectories.getGenfilesDirectory();
  }

  public ArtifactRoot getGenfilesDirectory(RepositoryName repositoryName) {
    return outputDirectories.getGenfilesDirectory();
  }

  public boolean hasSeparateGenfilesDirectory() {
    return !outputDirectories.mergeGenfilesDirectory();
  }

  /**
   * Returns the directory where coverage-related artifacts and metadata files should be stored.
   * This includes for example uninstrumented class files needed for Jacoco's coverage reporting
   * tools.
   */
  public ArtifactRoot getCoverageMetadataDirectory(RepositoryName repositoryName) {
    return outputDirectories.getCoverageMetadataDirectory();
  }

  /** Returns the testlogs directory for this build configuration. */
  public ArtifactRoot getTestLogsDirectory(RepositoryName repositoryName) {
    return outputDirectories.getTestLogsDirectory();
  }

  /** Returns a relative path to the genfiles directory at execution time. */
  public PathFragment getGenfilesFragment() {
    return outputDirectories.getGenfilesFragment();
  }

  /**
   * Returns the path separator for the host platform. This is basically the same as {@link
   * java.io.File#pathSeparator}, except that that returns the value for this JVM, which may or may
   * not match the host platform. You should only use this when invoking tools that are known to use
   * the native path separator, i.e., the path separator for the machine that they run on.
   */
  @Override
  public String getHostPathSeparator() {
    return outputDirectories.getHostPathSeparator();
  }

  /** Returns the internal directory (used for middlemen) for this build configuration. */
  public ArtifactRoot getMiddlemanDirectory(RepositoryName repositoryName) {
    return outputDirectories.getMiddlemanDirectory();
  }

  public boolean isStrictFilesets() {
    return options.strictFilesets;
  }

  public boolean isStrictFilesetOutput() {
    return options.strictFilesetOutput;
  }

  public String getMainRepositoryName() {
    return mainRepositoryName.strippedName();
  }

  /**
   * Returns the configuration-dependent string for this configuration. This is also the name of the
   * configuration's base output directory unless {@link CoreOptions#outputDirectoryName} overrides
   * it.
   */
  public String getMnemonic() {
    return outputDirectories.getMnemonic();
  }

  @Override
  public String toString() {
    return checksum();
  }

  public ActionEnvironment getActionEnvironment() {
    return actionEnv;
  }

  /**
   * Return the "fixed" part of the actions' environment variables.
   *
   * <p>An action's full set of environment variables consist of a "fixed" part and of a "variable"
   * part. The "fixed" variables are independent of the Bazel client's own environment, and are
   * returned by this function. The "variable" ones are inherited from the Bazel client's own
   * environment, and are returned by {@link #getVariableShellEnvironment}.
   *
   * <p>Since values of the "fixed" variables are already known at analysis phase, it is returned
   * here as a map.
   */
  @Override
  public ImmutableMap<String, String> getLocalShellEnvironment() {
    return actionEnv.getFixedEnv().toMap();
  }

  /**
   * Return the "variable" part of the actions' environment variables.
   *
   * <p>An action's full set of environment variables consist of a "fixed" part and of a "variable"
   * part. The "fixed" variables are independent of the Bazel client's own environment, and are
   * returned by {@link #getLocalShellEnvironment}. The "variable" ones are inherited from the Bazel
   * client's own environment, and are returned by this function.
   *
   * <p>The values of the "variable" variables are tracked in Skyframe via the {@link
   * com.google.devtools.build.lib.skyframe.SkyFunctions#CLIENT_ENVIRONMENT_VARIABLE} skyfunction.
   * This method only returns the names of those variables to be inherited, if set in the client's
   * environment. (Variables where the name is not returned in this set should not be taken from the
   * client environment.)
   */
  @Deprecated // Use getActionEnvironment instead.
  public Iterable<String> getVariableShellEnvironment() {
    return actionEnv.getInheritedEnv();
  }

  /**
   * Returns a regex-based instrumentation filter instance that used to match label names to
   * identify targets to be instrumented in the coverage mode.
   */
  public RegexFilter getInstrumentationFilter() {
    return options.instrumentationFilter;
  }

  /**
   * Returns a boolean of whether to include targets created by *_test rules in the set of targets
   * matched by --instrumentation_filter. If this is false, all test targets are excluded from
   * instrumentation.
   */
  public boolean shouldInstrumentTestTargets() {
    return options.instrumentTestTargets;
  }

  /**
   * Returns a new, unordered mapping of names to values of "Make" variables defined by this
   * configuration.
   *
   * <p>This does *not* include package-defined overrides (e.g. vardef) and so should not be used by
   * the build logic. This is used only for the 'info' command.
   *
   * <p>Command-line definitions of make environments override variables defined by {@code
   * Fragment.addGlobalMakeVariables()}.
   */
  public Map<String, String> getMakeEnvironment() {
    Map<String, String> makeEnvironment = new HashMap<>();
    makeEnvironment.putAll(globalMakeEnv);
    makeEnvironment.putAll(commandLineBuildVariables);
    return ImmutableMap.copyOf(makeEnvironment);
  }

  /**
   * Returns a new, unordered mapping of names that are set through the command lines. (Fragments,
   * in particular the Google C++ support, can set variables through the command line.)
   */
  public ImmutableMap<String, String> getCommandLineBuildVariables() {
    return commandLineBuildVariables;
  }

  /** Returns the global defaults for this configuration for the Make environment. */
  public ImmutableMap<String, String> getGlobalMakeEnvironment() {
    return globalMakeEnv;
  }

  /**
   * Returns the default value for the specified "Make" variable for this configuration. Returns
   * null if no value was found.
   */
  public String getMakeVariableDefault(String var) {
    return globalMakeEnv.get(var);
  }

  /** Returns a configuration fragment instances of the given class. */
  public <T extends Fragment> T getFragment(Class<T> clazz) {
    return clazz.cast(fragments.get(clazz));
  }

  /** Returns true if the requested configuration fragment is present. */
  public <T extends Fragment> boolean hasFragment(Class<T> clazz) {
    return getFragment(clazz) != null;
  }

  /** Returns true if all requested configuration fragment are present (this may be slow). */
  public boolean hasAllFragments(Set<Class<?>> fragmentClasses) {
    for (Class<?> fragmentClass : fragmentClasses) {
      if (!hasFragment(fragmentClass.asSubclass(Fragment.class))) {
        return false;
      }
    }
    return true;
  }

  public BlazeDirectories getDirectories() {
    return outputDirectories.getDirectories();
  }

  /** Which fragments does this configuration contain? */
  public FragmentClassSet fragmentClasses() {
    return fragmentClassSet;
  }

  /** Returns true if non-functional build stamps are enabled. */
  public boolean stampBinaries() {
    return options.stampBinaries;
  }

  /** Returns true if extended sanity checks should be enabled. */
  public boolean extendedSanityChecks() {
    return options.extendedSanityChecks;
  }

  /** Returns true if we are building runfiles manifests for this configuration. */
  public boolean buildRunfilesManifests() {
    return options.buildRunfilesManifests;
  }

  /** Returns true if we are building runfile links for this configuration. */
  public boolean buildRunfileLinks() {
    return options.buildRunfilesManifests && options.buildRunfiles;
  }

  /** Returns if we are building external runfiles symlinks using the old-style structure. */
  public boolean legacyExternalRunfiles() {
    return options.legacyExternalRunfiles;
  }

  /**
   * Returns user-specified test environment variables and their values, as set by the --test_env
   * options.
   */
  @Override
  public ImmutableMap<String, String> getTestEnv() {
    return testEnv.getFixedEnv().toMap();
  }

  /**
   * Returns user-specified test environment variables and their values, as set by the {@code
   * --test_env} options. It is incomplete in that it is not a superset of the {@link
   * #getActionEnvironment}, but both have to be applied, with this one being applied after the
   * other, such that {@code --test_env} settings can override {@code --action_env} settings.
   */
  // TODO(ulfjack): Just return the merged action and test action environment here?
  public ActionEnvironment getTestActionEnvironment() {
    return testEnv;
  }

  public CommandLineLimits getCommandLineLimits() {
    return commandLineLimits;
  }

  @Override
  public boolean isCodeCoverageEnabled() {
    return options.collectCodeCoverage;
  }

  public RunUnder getRunUnder() {
    return options.runUnder;
  }

  /** Returns true if this is a host configuration. */
  public boolean isHostConfiguration() {
    return options.isHost;
  }

  /** Returns true if this is an execution configuration. */
  public boolean isExecConfiguration() {
    return options.isExec;
  }

  /** Returns true if this is an tool-related configuration. */
  public boolean isToolConfiguration() {
    return isExecConfiguration() || isHostConfiguration();
  }

  public boolean checkVisibility() {
    return options.checkVisibility;
  }

  public boolean checkLicenses() {
    return options.checkLicenses;
  }

  public boolean enforceConstraints() {
    return options.enforceConstraints;
  }

  public boolean allowAnalysisFailures() {
    return options.allowAnalysisFailures;
  }

  public boolean evaluatingForAnalysisTest() {
    return options.evaluatingForAnalysisTest;
  }

  public int analysisTestingDepsLimit() {
    return options.analysisTestingDepsLimit;
  }

  public List<Label> getActionListeners() {
    return options.actionListeners;
  }

  public boolean inmemoryUnusedInputsList() {
    return options.inmemoryUnusedInputsList;
  }

  /**
   * Returns whether FileWriteAction may transparently compress its contents in the analysis phase
   * to save memory. Semantics are not affected.
   */
  public FileWriteAction.Compression transparentCompression() {
    return FileWriteAction.Compression.fromBoolean(options.transparentCompression);
  }

  /**
   * Returns whether we should trim configurations to only include the fragments needed to correctly
   * analyze a rule.
   */
  public boolean trimConfigurations() {
    return options.configsMode == CoreOptions.ConfigsMode.ON;
  }

  /**
   * Returns whether we should trim configurations to only include the fragments needed to correctly
   * analyze a rule.
   */
  public boolean trimConfigurationsRetroactively() {
    return options.configsMode == CoreOptions.ConfigsMode.RETROACTIVE;
  }

  /**
   * <b>>Experimental feature:</b> if true, qualifying outputs use path prefixes based on their
   * content instead of the traditional <code>blaze-out/$CPU-$COMPILATION_MODE</code>.
   *
   * <p>This promises both more intrinsic correctness (outputs with different contents can't write
   * to the same path) and efficiency (outputs with the <i>same</i> contents share the same path and
   * therefore permit better action caching). But it's highly experimental and should not be relied
   * on in any serious way any time soon.
   *
   * <p>See <a href="https://github.com/bazelbuild/bazel/issues/6526">#6526</a> for details.
   */
  public boolean useContentBasedOutputPaths() {
    return options.outputPathsMode == CoreOptions.OutputPathsMode.CONTENT;
  }

  public boolean allowUnresolvedSymlinks() {
    return options.allowUnresolvedSymlinks;
  }

  /** Returns compilation mode. */
  public CompilationMode getCompilationMode() {
    return options.compilationMode;
  }

  /** Returns the cache key of the build options used to create this configuration. */
  public String checksum() {
    return checksum;
  }

  /** Returns a copy of the build configuration options for this configuration. */
  public BuildOptions cloneOptions() {
    BuildOptions clone = buildOptions.clone();
    return clone;
  }

  /**
   * Returns the actual options reference used by this configuration.
   *
   * <p><b>Be very careful using this method.</b> Options classes are mutable - no caller should
   * ever call this method if there's any change the reference might be written to. This method only
   * exists because {@link #cloneOptions} can be expensive when applied to every edge in a
   * dependency graph.
   *
   * <p>Do not use this method without careful review with other Bazel developers.
   */
  public BuildOptions getOptions() {
    return buildOptions;
  }

  public BuildOptions.OptionsDiffForReconstruction getBuildOptionsDiff() {
    return buildOptionsDiff;
  }

  public String getCpu() {
    return options.cpu;
  }

  @VisibleForTesting
  public String getHostCpu() {
    return options.hostCpu;
  }

  // TODO(buchgr): Revisit naming and functionality of this flag. See #9248 for details.
  public static boolean runfilesEnabled(CoreOptions options) {
    switch (options.enableRunfiles) {
      case YES:
        return true;
      case NO:
        return false;
      default:
        return OS.getCurrent() != OS.WINDOWS;
    }
  }

  public boolean runfilesEnabled() {
    return runfilesEnabled(this.options);
  }

  public boolean inprocessSymlinkCreation() {
    return options.inprocessSymlinkCreation;
  }

  public boolean skipRunfilesManifests() {
    return options.skipRunfilesManifests;
  }

  /**
   * Returns a modified copy of {@code executionInfo} if any {@code executionInfoModifiers} apply to
   * the given {@code mnemonic}. Otherwise returns {@code executionInfo} unchanged.
   */
  public ImmutableMap<String, String> modifiedExecutionInfo(
      ImmutableMap<String, String> executionInfo, String mnemonic) {
    if (!options.executionInfoModifier.matches(mnemonic)) {
      return executionInfo;
    }
    Map<String, String> mutableCopy = new HashMap<>(executionInfo);
    modifyExecutionInfo(mutableCopy, mnemonic);
    return executionInfoInterner.intern(ImmutableSortedMap.copyOf(mutableCopy));
  }

  /** Applies {@code executionInfoModifiers} to the given {@code executionInfo}. */
  public void modifyExecutionInfo(Map<String, String> executionInfo, String mnemonic) {
    options.executionInfoModifier.apply(mnemonic, executionInfo);
  }

  /** @return the list of default features used for all packages. */
  public List<String> getDefaultFeatures() {
    return options.defaultFeatures;
  }

  /**
   * Returns the "top-level" environment space, i.e. the set of environments all top-level targets
   * must be compatible with. An empty value implies no restrictions.
   */
  public List<Label> getTargetEnvironments() {
    return options.targetEnvironments;
  }

  /**
   * Returns the {@link Label} of the {@code environment_group} target that will be used to find the
   * target environment during auto-population.
   */
  public Label getAutoCpuEnvironmentGroup() {
    return options.autoCpuEnvironmentGroup;
  }

  public Class<? extends Fragment> getSkylarkFragmentByName(String name) {
    return skylarkVisibleFragments.get(name);
  }

  public ImmutableCollection<String> getSkylarkFragmentNames() {
    return skylarkVisibleFragments.keySet();
  }

  public BuildEventId getEventId() {
    return BuildEventId.configurationId(checksum());
  }

  public BuildConfigurationEvent toBuildEvent() {
    return buildEventSupplier.get();
  }

  private BuildConfigurationEvent createBuildEvent() {
    BuildEventId eventId = getEventId();
    BuildEventStreamProtos.BuildEvent.Builder builder =
        BuildEventStreamProtos.BuildEvent.newBuilder();
    builder
        .setId(eventId.asStreamProto())
        .setConfiguration(
            BuildEventStreamProtos.Configuration.newBuilder()
                .setMnemonic(getMnemonic())
                .setPlatformName(getCpu())
                .putAllMakeVariable(getMakeEnvironment())
                .setCpu(getCpu())
                .build());
    return new BuildConfigurationEvent(eventId, builder.build());
  }

  public ImmutableSet<String> getReservedActionMnemonics() {
    return reservedActionMnemonics;
  }
}
