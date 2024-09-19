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

package com.google.devtools.build.lib.packages;

import static com.google.common.base.MoreObjects.firstNonNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.BiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Interner;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.BazelModuleContext.LoadGraphVisitor;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.StarlarkThreadContext;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Package.Builder.PackageSettings;
import com.google.devtools.build.lib.packages.TargetDefinitionContext.MacroNamespaceViolationException;
import com.google.devtools.build.lib.packages.WorkspaceFileValue.WorkspaceFileKey;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading.Code;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.OptionalLong;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.Semaphore;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.SymbolGenerator;
import net.starlark.java.syntax.Location;

/**
 * A package, which is a container of {@link Rule}s, each of which contains a dictionary of named
 * attributes.
 *
 * <p>Package instances are intended to be immutable and for all practical purposes can be treated
 * as such. Note, however, that some member variables exposed via the public interface are not
 * strictly immutable, so until their types are guaranteed immutable we're not applying the
 * {@code @Immutable} annotation here.
 *
 * <p>This class should not be extended - it's only non-final for mocking!
 *
 * <p>When changing this class, make sure to make corresponding changes to serialization!
 */
@SuppressWarnings("JavaLangClash")
public class Package {

  // TODO(bazel-team): This class and its builder are ginormous. Future refactoring work might
  // attempt to separate the concerns of:
  //   - instantiating targets/macros, adding them to the package, and accessing/indexing them
  //     afterwards
  //   - utility logical like validating names, checking for conflicts, etc.
  //   - tracking and enforcement of limits
  //   - grouping metadata based on whether it's known prior to BUILD file evaluation, prior to
  //     symbolic macro evalutaion, or at the time of final Package construction
  //   - machinery specific to external package / WORKSPACE / bzlmod

  // ==== Static fields and enums ====

  /**
   * How to enforce config_setting visibility settings.
   *
   * <p>This is a temporary setting in service of https://github.com/bazelbuild/bazel/issues/12669.
   * After enough depot cleanup, config_setting will have the same visibility enforcement as all
   * other rules.
   */
  public enum ConfigSettingVisibilityPolicy {
    /** Don't enforce visibility for any config_setting. */
    LEGACY_OFF,
    /** Honor explicit visibility settings on config_setting, else use //visibility:public. */
    DEFAULT_PUBLIC,
    /** Enforce config_setting visibility exactly the same as all other rules. */
    DEFAULT_STANDARD
  }

  /**
   * The "workspace name" of packages generated by Bzlmod to contain repo rules.
   *
   * <p>Normally, packages containing repo rules are differentiated from packages containing build
   * rules by the {@link PackageIdentifier}: The singular repo-rule-containing package is {@code
   * //external}. However, in Bzlmod, packages containing repo rules need to have meaningful {@link
   * PackageIdentifier}s, so there needs to be some other way to distinguish them from
   * build-rule-containing packages. We use the following magic string as the "workspace name" for
   * repo-rule-containing packages generated by Bzlmod.
   *
   * @see #isRepoRulePackage()
   */
  private static final String DUMMY_WORKSPACE_NAME_FOR_BZLMOD_PACKAGES = "__dummy_workspace_bzlmod";

  /** Sentinel value for package overhead being empty. */
  private static final long PACKAGE_OVERHEAD_UNSET = -1;

  /** Used for constructing macro namespace violation error messages. */
  private static final String MACRO_NAMING_RULES =
      "Name must be the same as the macro's name, or the macro's name followed by '_'"
          + " (recommended), '-', or '.', and a non-empty string.";

  // ==== General package metadata fields ====

  private final Metadata metadata;

  private final Optional<Root> sourceRoot;

  // For BUILD files, this is initialized immediately. For WORKSPACE files, it is known only after
  // Starlark evaluation of the WORKSPACE file has finished.
  private String workspaceName;

  // Can be changed during BUILD file evaluation due to exports_files() modifying its visibility.
  private InputFile buildFile;

  // Mutated during BUILD file evaluation (but not by symbolic macro evaluation).
  private PackageArgs packageArgs = PackageArgs.DEFAULT;

  // Mutated during BUILD file evaluation (but not by symbolic macro evaluation).
  private ImmutableMap<String, String> makeEnv;

  // These two fields are mutually exclusive. Which one is set depends on
  // PackageSettings#precomputeTransitiveLoads. See Package.Builder#setLoads.
  @Nullable private ImmutableList<Module> directLoads;
  @Nullable private ImmutableList<Label> transitiveLoads;

  /**
   * True iff this package's BUILD files contained lexical or grammatical errors, or experienced
   * errors during evaluation, or semantic errors during the construction of any rule.
   *
   * <p>Note: A package containing errors does not necessarily prevent a build; if all the rules
   * needed for a given build were constructed prior to the first error, the build may proceed.
   */
  private boolean containsErrors;

  /**
   * The first detailed error encountered during this package's construction and evaluation, or
   * {@code null} if there were no such errors or all its errors lacked details.
   */
  @Nullable private FailureDetail failureDetail;

  private long computationSteps;

  /**
   * A rough approximation of the memory and general accounting costs associated with a loaded
   * package. A value of -1 means it is unset. Stored as a long to take up less memory per pkg.
   */
  private long packageOverhead = PACKAGE_OVERHEAD_UNSET;

  // ==== Fields specific to external package / WORKSPACE logic ====

  /**
   * The map from each repository to that repository's remappings map. This is only used in the
   * //external package, it is an empty map for all other packages. For example, an entry of {"@foo"
   * : {"@x", "@y"}} indicates that, within repository foo, "@x" should be remapped to "@y".
   */
  private ImmutableMap<RepositoryName, ImmutableMap<String, RepositoryName>>
      externalPackageRepositoryMappings;

  private ImmutableList<TargetPattern> registeredExecutionPlatforms;
  private ImmutableList<TargetPattern> registeredToolchains;
  private OptionalInt firstWorkspaceSuffixRegisteredToolchain;

  // ==== Target and macro fields ====

  /** The collection of all targets defined in this package, indexed by name. */
  // TODO(bazel-team): Clarify what this map contains when a rule and its output both share the same
  // name.
  private ImmutableSortedMap<String, Target> targets;

  /**
   * The collection of all symbolic macro instances defined in this package, indexed by their {@link
   * MacroInstance#getId id} (not name).
   */
  // TODO(#19922): Consider enforcing that macro namespaces are "exclusive", meaning that target
  // names may only suffix a macro name when the target is created (transitively) within the macro.
  // This would be a major change that would break the (common) use case where a BUILD file
  // declares both "foo" and "foo_test".
  private ImmutableSortedMap<String, MacroInstance> macros;

  /**
   * A map from names of targets declared in a symbolic macro which violate macro naming rules, such
   * as "lib%{name}-src.jar" implicit outputs in java rules, to the name of the macro instance where
   * they were declared.
   *
   * <p>Initialized by the builder in {@link #finishInit}.
   */
  @Nullable private ImmutableMap<String, String> macroNamespaceViolatingTargets;

  /**
   * A map from names of targets declared in a symbolic macro to the (innermost) macro instance
   * where they were declared.
   */
  // TODO: #19922 - If this field were made serializable (currently it's not), it would subsume
  // macroNamespaceViolatingTargets, since we can just map the target to its macro and then check
  // whether it is in the macro's namespace.
  //
  // TODO: #19922 - Don't maintain this extra map of all macro-instantiated targets. We have a
  // couple options:
  //   1) Have Target store a reference to its declaring MacroInstance directly. To avoid adding a
  //      field to that class (a not insignificant cost), we can merge it with the reference to its
  //      package: If we're not in a macro, we point to the package, and if we are, we point to the
  //      innermost macro, and hop to the MacroInstance to get a reference to the Package (or parent
  //      macro).
  //   2) To support lazy macro evaluation, we'll probably need a prefix trie in Package to find the
  //      macros whose namespaces contain the requested target name. For targets that respect their
  //      macro's namespace, we could just look them up in the trie. This assumes we already know
  //      whether the target is well-named, which we wouldn't if we got rid of
  //      macroNamespaceViolatingTargets.
  private ImmutableMap<String, MacroInstance> targetsToDeclaringMacros;

  // ==== Constructor ====

  /**
   * Constructs a new (incomplete) Package instance. Intended only for use by {@link
   * Package.Builder}.
   *
   * <p>Packages and Targets refer to one another. Therefore, the builder needs to have a Package
   * instance on-hand before it can associate any targets with the package. Certain Metadata fields
   * like the package's name must be known before that point, while other fields are filled in when
   * the builder calls {@link Package#finishInit}.
   */
  // TODO(#19922): Better separate fields that must be known a priori from those determined through
  // BUILD evaluation.
  private Package(Metadata metadata) {
    this.metadata = metadata;
    this.sourceRoot = computeSourceRoot(metadata);
  }

  // ==== General package metadata accessors ====

  public Metadata getMetadata() {
    return metadata;
  }

  /**
   * Returns this package's identifier.
   *
   * <p>This is a suffix of {@code getFilename().getParentDirectory()}.
   */
  public PackageIdentifier getPackageIdentifier() {
    return metadata.packageIdentifier();
  }

  /**
   * Returns the name of this package. If this build is using external repositories then this name
   * may not be unique!
   */
  public String getName() {
    return metadata.getName();
  }

  /** Like {@link #getName}, but has type {@code PathFragment}. */
  public PathFragment getNameFragment() {
    return getPackageIdentifier().getPackageFragment();
  }

  /**
   * Returns the filename of the BUILD file which defines this package. The parent directory of the
   * BUILD file is the package directory.
   */
  public RootedPath getFilename() {
    return metadata.buildFilename();
  }

  /** Returns the directory containing the package's BUILD file. */
  public Path getPackageDirectory() {
    return metadata.getPackageDirectory();
  }

  /**
   * Whether this package should contain only repo rules (returns {@code true}) or only build rules
   * (returns {@code false}).
   */
  private boolean isRepoRulePackage() {
    return metadata.isRepoRulePackage;
  }

  /**
   * Returns the map of repository reassignments for BUILD packages. This will be empty for packages
   * within the main workspace.
   */
  public RepositoryMapping getRepositoryMapping() {
    return metadata.repositoryMapping();
  }

  /**
   * How to enforce visibility on <code>config_setting</code> See {@link
   * ConfigSettingVisibilityPolicy} for details.
   *
   * <p>Null for repo rule packages.
   */
  @Nullable
  public ConfigSettingVisibilityPolicy getConfigSettingVisibilityPolicy() {
    return metadata.configSettingVisibilityPolicy();
  }

  /**
   * Returns the name of the workspace this package is in. Used as a prefix for the runfiles
   * directory. This can be set in the WORKSPACE file. This must be a valid target name.
   */
  public String getWorkspaceName() {
    return workspaceName;
  }

  /** Returns the InputFile target for this package's BUILD file. */
  public InputFile getBuildFile() {
    return buildFile;
  }

  /**
   * Returns the label of this package's BUILD file.
   *
   * <p>Typically <code>getBuildFileLabel().getName().equals("BUILD")</code> -- though not
   * necessarily: data in a subdirectory of a test package may use a different filename to avoid
   * inadvertently creating a new package.
   */
  public Label getBuildFileLabel() {
    return buildFile.getLabel();
  }

  /**
   * Returns the collection of package-level attributes set by the {@code package()} callable and
   * similar methods.
   */
  public PackageArgs getPackageArgs() {
    return packageArgs;
  }

  /**
   * Returns the "Make" environment of this package, containing package-local definitions of "Make"
   * variables.
   */
  public ImmutableMap<String, String> getMakeEnvironment() {
    return makeEnv;
  }

  /**
   * Returns the root of the source tree beneath which this package's BUILD file was found, or
   * {@link Optional#empty} if this package was derived from a WORKSPACE file.
   *
   * <p>Assumes invariant: If non-empty, {@code
   * getSourceRoot().get().getRelative(packageId.getSourceRoot()).equals(getPackageDirectory())}
   */
  public Optional<Root> getSourceRoot() {
    return sourceRoot;
  }

  /**
   * Returns a list of Starlark files transitively loaded by this package.
   *
   * <p>If transitive loads are not {@linkplain PackageSettings#precomputeTransitiveLoads
   * precomputed}, performs a traversal over the load graph to compute them.
   *
   * <p>If only the count of transitively loaded files is needed, use {@link
   * #countTransitivelyLoadedStarlarkFiles}. For a customized online visitation, use {@link
   * #visitLoadGraph}.
   */
  public ImmutableList<Label> getOrComputeTransitivelyLoadedStarlarkFiles() {
    return transitiveLoads != null ? transitiveLoads : computeTransitiveLoads(directLoads);
  }

  /**
   * Counts the number Starlark files transitively loaded by this package.
   *
   * <p>If transitive loads are not {@linkplain PackageSettings#precomputeTransitiveLoads
   * precomputed}, performs a traversal over the load graph to count them.
   */
  public int countTransitivelyLoadedStarlarkFiles() {
    if (transitiveLoads != null) {
      return transitiveLoads.size();
    }
    Set<Label> loads = new HashSet<>();
    visitLoadGraph(loads::add);
    return loads.size();
  }

  /**
   * Performs an online visitation of the load graph rooted at this package.
   *
   * <p>If transitive loads were {@linkplain PackageSettings#precomputeTransitiveLoads precomputed},
   * each file is passed to {@link LoadGraphVisitor#visit} once regardless of its return value.
   */
  public <E1 extends Exception, E2 extends Exception> void visitLoadGraph(
      LoadGraphVisitor<E1, E2> visitor) throws E1, E2 {
    if (transitiveLoads != null) {
      for (Label load : transitiveLoads) {
        visitor.visit(load);
      }
    } else {
      BazelModuleContext.visitLoadGraphRecursively(directLoads, visitor);
    }
  }

  private static ImmutableList<Label> computeTransitiveLoads(Iterable<Module> directLoads) {
    Set<Label> loads = new LinkedHashSet<>();
    BazelModuleContext.visitLoadGraphRecursively(directLoads, loads::add);
    return ImmutableList.copyOf(loads);
  }

  /**
   * Returns true if errors were encountered during evaluation of this package. (The package may be
   * incomplete and its contents should not be relied upon for critical operations. However, any
   * Rules belonging to the package are guaranteed to be intact, unless their <code>containsErrors()
   * </code> flag is set.)
   */
  public boolean containsErrors() {
    return containsErrors;
  }

  /**
   * Returns the first {@link FailureDetail} describing one of the package's errors, or {@code null}
   * if it has no errors or all its errors lack details.
   */
  @Nullable
  public FailureDetail getFailureDetail() {
    return failureDetail;
  }

  /** Returns the number of Starlark computation steps executed by this BUILD file. */
  public long getComputationSteps() {
    return computationSteps;
  }

  /** Returns package overhead as configured by the configured {@link PackageOverheadEstimator}. */
  public OptionalLong getPackageOverhead() {
    return packageOverhead == PACKAGE_OVERHEAD_UNSET
        ? OptionalLong.empty()
        : OptionalLong.of(packageOverhead);
  }

  // ==== Accessors specific to external package / WORKSPACE logic ====

  /**
   * Returns the repository mapping for the requested external repository.
   *
   * @throws UnsupportedOperationException if called from a package other than the //external
   *     package
   */
  public ImmutableMap<String, RepositoryName> getExternalPackageRepositoryMapping(
      RepositoryName repository) {
    if (!isRepoRulePackage()) {
      throw new UnsupportedOperationException(
          "Can only access the external package repository"
              + "mappings from the //external package");
    }
    return externalPackageRepositoryMappings.getOrDefault(repository, ImmutableMap.of());
  }

  /**
   * Returns the full map of repository mappings collected so far.
   *
   * @throws UnsupportedOperationException if called from a package other than the //external
   *     package
   */
  ImmutableMap<RepositoryName, ImmutableMap<String, RepositoryName>>
      getExternalPackageRepositoryMappings() {
    if (!isRepoRulePackage()) {
      throw new UnsupportedOperationException(
          "Can only access the external package repository"
              + "mappings from the //external package");
    }
    return this.externalPackageRepositoryMappings;
  }

  public ImmutableList<TargetPattern> getRegisteredExecutionPlatforms() {
    return registeredExecutionPlatforms;
  }

  public ImmutableList<TargetPattern> getRegisteredToolchains() {
    return registeredToolchains;
  }

  public ImmutableList<TargetPattern> getUserRegisteredToolchains() {
    return getRegisteredToolchains()
        .subList(
            0, firstWorkspaceSuffixRegisteredToolchain.orElse(getRegisteredToolchains().size()));
  }

  public ImmutableList<TargetPattern> getWorkspaceSuffixRegisteredToolchains() {
    return getRegisteredToolchains()
        .subList(
            firstWorkspaceSuffixRegisteredToolchain.orElse(getRegisteredToolchains().size()),
            getRegisteredToolchains().size());
  }

  OptionalInt getFirstWorkspaceSuffixRegisteredToolchain() {
    return firstWorkspaceSuffixRegisteredToolchain;
  }

  // ==== Target and macro accessors ====

  /** Returns an (immutable, ordered) view of all the targets belonging to this package. */
  public ImmutableSortedMap<String, Target> getTargets() {
    return targets;
  }

  /** Common getTargets implementation, accessible by {@link Package.Builder}. */
  private static Set<Target> getTargets(BiMap<String, Target> targetMap) {
    return targetMap.values();
  }

  /**
   * Returns a (read-only, ordered) iterable of all the targets belonging to this package which are
   * instances of the specified class.
   */
  public <T extends Target> Iterable<T> getTargets(Class<T> targetClass) {
    return getTargets(targets, targetClass);
  }

  /**
   * Common getTargets implementation, accessible by both {@link Package} and {@link
   * Package.Builder}.
   */
  private static <T extends Target> Iterable<T> getTargets(
      Map<String, Target> targetMap, Class<T> targetClass) {
    return Iterables.filter(targetMap.values(), targetClass);
  }

  /**
   * Returns the rule that corresponds to a particular BUILD target name. Useful for walking through
   * the dependency graph of a target. Fails if the target is not a Rule.
   */
  public Rule getRule(String targetName) {
    return (Rule) targets.get(targetName);
  }

  /**
   * Returns a map from names of targets declared in a symbolic macro which violate macro naming
   * rules, such as "lib%{name}-src.jar" implicit outputs in java rules, to the name of the macro
   * instance where they were declared.
   */
  ImmutableMap<String, String> getMacroNamespaceViolatingTargets() {
    Preconditions.checkNotNull(
        macroNamespaceViolatingTargets,
        "This method is only available after the package has been loaded.");
    return macroNamespaceViolatingTargets;
  }

  /**
   * Throws {@link MacroNamespaceViolationException} if the given target (which must be a member of
   * this package) violates macro naming rules.
   */
  public void checkMacroNamespaceCompliance(Target target) throws MacroNamespaceViolationException {
    Preconditions.checkArgument(
        this.equals(target.getPackage()), "Target must belong to this package");
    @Nullable
    String macroNamespaceViolated = getMacroNamespaceViolatingTargets().get(target.getName());
    if (macroNamespaceViolated != null) {
      throw new MacroNamespaceViolationException(
          String.format(
              "Target %s declared in symbolic macro '%s' violates macro naming rules and cannot be"
                  + " built. %s",
              target.getLabel(), macroNamespaceViolated, MACRO_NAMING_RULES));
    }
  }

  /**
   * Returns the target (a member of this package) whose name is "targetName". First rules are
   * searched, then output files, then input files. The target name must be valid, as defined by
   * {@code LabelValidator#validateTargetName}.
   *
   * @throws NoSuchTargetException if the specified target was not found.
   */
  public Target getTarget(String targetName) throws NoSuchTargetException {
    Target target = targets.get(targetName);
    if (target != null) {
      return target;
    }

    Label label;
    try {
      label = Label.create(metadata.packageIdentifier(), targetName);
    } catch (LabelSyntaxException e) {
      throw new IllegalArgumentException(targetName, e);
    }

    if (metadata.succinctTargetNotFoundErrors()) {
      throw new NoSuchTargetException(
          label, String.format("target '%s' not declared in package '%s'", targetName, getName()));
    } else {
      String alternateTargetSuggestion = getAlternateTargetSuggestion(targetName);
      throw new NoSuchTargetException(
          label,
          String.format(
              "target '%s' not declared in package '%s' defined by %s%s",
              targetName,
              getName(),
              metadata.buildFilename().asPath().getPathString(),
              alternateTargetSuggestion));
    }
  }

  private String getAlternateTargetSuggestion(String targetName) {
    // If there's a file on the disk that's not mentioned in the BUILD file,
    // produce a more informative error.  NOTE! this code path is only executed
    // on failure, which is (relatively) very rare.  In the common case no
    // stat(2) is executed.
    Path filename = metadata.getPackageDirectory().getRelative(targetName);
    if (!PathFragment.isNormalized(targetName) || "*".equals(targetName)) {
      // Don't check for file existence if the target name is not normalized
      // because the error message would be confusing and wrong. If the
      // targetName is "foo/bar/.", and there is a directory "foo/bar", it
      // doesn't mean that "//pkg:foo/bar/." is a valid label.
      // Also don't check if the target name is a single * character since
      // it's invalid on Windows.
      return "";
    } else if (filename.isDirectory()) {
      return "; however, a source directory of this name exists.  (Perhaps add "
          + "'exports_files([\""
          + targetName
          + "\"])' to "
          + getName()
          + "/BUILD, or define a "
          + "filegroup?)";
    } else if (filename.exists()) {
      return "; however, a source file of this name exists.  (Perhaps add "
          + "'exports_files([\""
          + targetName
          + "\"])' to "
          + getName()
          + "/BUILD?)";
    } else {
      return TargetSuggester.suggestTargets(targetName, targets.keySet());
    }
  }

  /**
   * Returns all symbolic macros defined in the package, indexed by {@link MacroInstance#getId id}.
   *
   * <p>Note that {@code MacroInstance}s hold just the information known at the time a macro was
   * declared, even though by the time the {@code Package} is fully constructed we already have
   * fully evaluated these macros.
   */
  public ImmutableMap<String, MacroInstance> getMacrosById() {
    return macros;
  }

  /**
   * Returns the (innermost) symbolic macro instance that declared the given target, or null if the
   * target was not created in a symbolic macro or no such target by the given name exists.
   */
  @Nullable
  public MacroInstance getDeclaringMacroForTarget(String target) {
    return targetsToDeclaringMacros.get(target);
  }

  // ==== Initialization ====

  private static Optional<Root> computeSourceRoot(Metadata metadata) {
    if (metadata.isRepoRulePackage()) {
      return Optional.empty();
    }

    RootedPath buildFileRootedPath = metadata.buildFilename();
    Root buildFileRoot = buildFileRootedPath.getRoot();
    PathFragment pkgIdFragment = metadata.packageIdentifier().getSourceRoot();
    PathFragment pkgDirFragment = buildFileRootedPath.getRootRelativePath().getParentDirectory();

    Root sourceRoot;
    if (pkgIdFragment.equals(pkgDirFragment)) {
      // Fast path: BUILD file path and package name are the same, don't create an extra root.
      sourceRoot = buildFileRoot;
    } else {
      // TODO(bazel-team): Can this expr be simplified to just pkgDirFragment?
      PathFragment current = buildFileRootedPath.asPath().asFragment().getParentDirectory();
      for (int i = 0, len = pkgIdFragment.segmentCount(); i < len && current != null; i++) {
        current = current.getParentDirectory();
      }
      if (current == null || current.isEmpty()) {
        // This is never really expected to work. The below check should fail.
        sourceRoot = buildFileRoot;
      } else {
        // Note that current is an absolute path.
        sourceRoot = Root.fromPath(buildFileRoot.getRelative(current));
      }
    }

    Preconditions.checkArgument(
        sourceRoot.asPath() != null
            && sourceRoot.getRelative(pkgIdFragment).equals(metadata.getPackageDirectory()),
        "Invalid BUILD file name for package '%s': %s (in source %s with packageDirectory %s and"
            + " package identifier source root %s)",
        metadata.packageIdentifier(),
        metadata.buildFilename(),
        sourceRoot,
        metadata.getPackageDirectory(),
        metadata.packageIdentifier().getSourceRoot());

    return Optional.of(sourceRoot);
  }

  /**
   * Completes the initialization of this package. Only after this method may a package by shared
   * publicly.
   */
  private void finishInit(Builder builder) {
    this.containsErrors |= builder.containsErrors;
    if (directLoads == null && transitiveLoads == null) {
      Preconditions.checkState(containsErrors, "Loads not set for error-free package");
      builder.setLoads(ImmutableList.of());
    }

    this.workspaceName = builder.workspaceName;

    this.makeEnv = ImmutableMap.copyOf(builder.makeEnv);
    this.targets = ImmutableSortedMap.copyOf(builder.targets);
    this.macros = ImmutableSortedMap.copyOf(builder.macros);
    this.macroNamespaceViolatingTargets =
        builder.macroNamespaceViolatingTargets != null
            ? ImmutableMap.copyOf(builder.macroNamespaceViolatingTargets)
            : ImmutableMap.of();
    this.targetsToDeclaringMacros = ImmutableSortedMap.copyOf(builder.targetsToDeclaringMacros);
    this.failureDetail = builder.getFailureDetail();
    this.registeredExecutionPlatforms = ImmutableList.copyOf(builder.registeredExecutionPlatforms);
    this.registeredToolchains = ImmutableList.copyOf(builder.registeredToolchains);
    this.firstWorkspaceSuffixRegisteredToolchain = builder.firstWorkspaceSuffixRegisteredToolchain;
    ImmutableMap.Builder<RepositoryName, ImmutableMap<String, RepositoryName>>
        repositoryMappingsBuilder = ImmutableMap.builder();
    if (!builder.externalPackageRepositoryMappings.isEmpty() && !builder.isRepoRulePackage()) {
      // 'repo_mapping' should only be used in the //external package, i.e. should only appear
      // in WORKSPACE files. Currently, if someone tries to use 'repo_mapping' in a BUILD rule, they
      // will get a "no such attribute" error. This check is to protect against a 'repo_mapping'
      // attribute being added to a rule in the future.
      throw new IllegalArgumentException(
          "'repo_mapping' may only be used in the //external package");
    }
    builder.externalPackageRepositoryMappings.forEach(
        (k, v) -> repositoryMappingsBuilder.put(k, ImmutableMap.copyOf(v)));
    this.externalPackageRepositoryMappings = repositoryMappingsBuilder.buildOrThrow();
    OptionalLong overheadEstimate = builder.packageOverheadEstimator.estimatePackageOverhead(this);
    this.packageOverhead = overheadEstimate.orElse(PACKAGE_OVERHEAD_UNSET);
  }

  // TODO(bazel-team): This is a mutation, but Package is supposed to be immutable. In practice it
  // seems like this is only called during Package construction (including deserialization),
  // principally via Rule#reportError. I would bet that all of the callers have access to a
  // Package.Builder, in which case they should report the bit using the builder instead. But maybe
  // it's easier to just checkState() that the Package hasn't already finished constructing.
  void setContainsErrors() {
    containsErrors = true;
  }

  // ==== Stringification / debugging ====

  @Override
  public String toString() {
    return "Package("
        + getName()
        + ")="
        + (targets != null ? getTargets(Rule.class) : "initializing...");
  }

  /**
   * Dumps the package for debugging. Do not depend on the exact format/contents of this debugging
   * output.
   */
  public void dump(PrintStream out) {
    out.println("  Package " + getName() + " (" + metadata.buildFilename().asPath() + ")");

    // Rules:
    out.println("    Rules");
    for (Rule rule : getTargets(Rule.class)) {
      out.println("      " + rule.getTargetKind() + " " + rule.getLabel());
      for (Attribute attr : rule.getAttributes()) {
        for (Object possibleValue :
            AggregatingAttributeMapper.of(rule).visitAttribute(attr.getName(), attr.getType())) {
          out.println("        " + attr.getName() + " = " + possibleValue);
        }
      }
    }

    // Files:
    out.println("    Files");
    for (FileTarget file : getTargets(FileTarget.class)) {
      out.print("      " + file.getTargetKind() + " " + file.getLabel());
      if (file instanceof OutputFile) {
        out.println(" (generated by " + ((OutputFile) file).getGeneratingRule().getLabel() + ")");
      } else {
        out.println();
      }
    }
  }

  // ==== Error reporting ====

  /**
   * Returns an error {@link Event} with {@link Location} and {@link DetailedExitCode} properties.
   */
  public static Event error(Location location, String message, Code code) {
    Event error = Event.error(location, message);
    return error.withProperty(
        DetailedExitCode.class,
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setMessage(message)
                .setPackageLoading(PackageLoading.newBuilder().setCode(code))
                .build()));
  }

  /**
   * If {@code pkg.containsErrors()}, sends an errorful "package contains errors" {@link Event}
   * (augmented with {@code pkg.getFailureDetail()}, if present) to the given {@link EventHandler}.
   */
  public static void maybeAddPackageContainsErrorsEventToHandler(
      Package pkg, EventHandler eventHandler) {
    if (pkg.containsErrors()) {
      eventHandler.handle(
          Event.error(
              String.format(
                  "package contains errors: %s%s",
                  pkg.getNameFragment(),
                  pkg.getFailureDetail() != null
                      ? ": " + pkg.getFailureDetail().getMessage()
                      : "")));
    }
  }

  /**
   * Given a {@link FailureDetail} and target, returns a modified {@code FailureDetail} that
   * attributes its error to the target.
   *
   * <p>If the given detail is null, then a generic {@link Code#TARGET_MISSING} detail identifying
   * the target is returned.
   */
  public static FailureDetail contextualizeFailureDetailForTarget(
      @Nullable FailureDetail failureDetail, Target target) {
    String prefix =
        "Target '" + target.getLabel() + "' contains an error and its package is in error";
    if (failureDetail == null) {
      return FailureDetail.newBuilder()
          .setMessage(prefix)
          .setPackageLoading(PackageLoading.newBuilder().setCode(Code.TARGET_MISSING))
          .build();
    }
    return failureDetail.toBuilder().setMessage(prefix + ": " + failureDetail.getMessage()).build();
  }

  // ==== Builders ====

  /**
   * Returns a new {@link Builder} suitable for constructing an ordinary package (i.e. not one for
   * WORKSPACE or bzlmod).
   */
  public static Builder newPackageBuilder(
      PackageSettings packageSettings,
      PackageIdentifier id,
      RootedPath filename,
      String workspaceName,
      Optional<String> associatedModuleName,
      Optional<String> associatedModuleVersion,
      boolean noImplicitFileExport,
      RepositoryMapping repositoryMapping,
      RepositoryMapping mainRepositoryMapping,
      @Nullable Semaphore cpuBoundSemaphore,
      PackageOverheadEstimator packageOverheadEstimator,
      @Nullable ImmutableMap<Location, String> generatorMap,
      // TODO(bazel-team): See Builder() constructor comment about use of null for this param.
      @Nullable ConfigSettingVisibilityPolicy configSettingVisibilityPolicy,
      @Nullable Globber globber) {
    // Determine whether this is for a repo rule package. We shouldn't actually have to do this
    // because newPackageBuilder() is supposed to only be called for normal packages. Unfortunately
    // serialization still uses the same code path for deserializing BUILD and WORKSPACE files,
    // violating this method's contract.
    boolean isRepoRulePackage =
        id.equals(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER)
            // For bzlmod packages, setWorkspaceName() is not called, so this expression doesn't
            // change during package evaluation.
            || workspaceName.equals(DUMMY_WORKSPACE_NAME_FOR_BZLMOD_PACKAGES);

    return new Builder(
        new Metadata(
            /* packageIdentifier= */ id,
            /* buildFilename= */ filename,
            /* isRepoRulePackage= */ isRepoRulePackage,
            /* repositoryMapping= */ repositoryMapping,
            /* associatedModuleName= */ associatedModuleName,
            /* associatedModuleVersion= */ associatedModuleVersion,
            /* configSettingVisibilityPolicy= */ configSettingVisibilityPolicy,
            /* succinctTargetNotFoundErrors= */ packageSettings.succinctTargetNotFoundErrors()),
        SymbolGenerator.create(id),
        packageSettings.precomputeTransitiveLoads(),
        noImplicitFileExport,
        workspaceName,
        mainRepositoryMapping,
        cpuBoundSemaphore,
        packageOverheadEstimator,
        generatorMap,
        globber);
  }

  public static Builder newExternalPackageBuilder(
      PackageSettings packageSettings,
      WorkspaceFileKey workspaceFileKey,
      String workspaceName,
      RepositoryMapping mainRepoMapping,
      boolean noImplicitFileExport,
      PackageOverheadEstimator packageOverheadEstimator) {
    return new Builder(
        new Metadata(
            /* packageIdentifier= */ LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER,
            /* buildFilename= */ workspaceFileKey.getPath(),
            /* isRepoRulePackage= */ true,
            /* repositoryMapping= */ mainRepoMapping,
            /* associatedModuleName= */ Optional.empty(),
            /* associatedModuleVersion= */ Optional.empty(),
            /* configSettingVisibilityPolicy= */ null,
            /* succinctTargetNotFoundErrors= */ packageSettings.succinctTargetNotFoundErrors()),
        // The SymbolGenerator is based on workspaceFileKey rather than a package id or path,
        // in order to distinguish different chunks of the same WORKSPACE file.
        SymbolGenerator.create(workspaceFileKey),
        packageSettings.precomputeTransitiveLoads(),
        noImplicitFileExport,
        workspaceName,
        mainRepoMapping,
        /* cpuBoundSemaphore= */ null,
        packageOverheadEstimator,
        /* generatorMap= */ null,
        /* globber= */ null);
  }

  public static Builder newExternalPackageBuilderForBzlmod(
      RootedPath moduleFilePath,
      boolean noImplicitFileExport,
      PackageIdentifier basePackageId,
      RepositoryMapping repoMapping) {
    return new Builder(
            new Metadata(
                /* packageIdentifier= */ basePackageId,
                /* buildFilename= */ moduleFilePath,
                /* isRepoRulePackage= */ true,
                /* repositoryMapping= */ repoMapping,
                /* associatedModuleName= */ Optional.empty(),
                /* associatedModuleVersion= */ Optional.empty(),
                /* configSettingVisibilityPolicy= */ null,
                /* succinctTargetNotFoundErrors= */ PackageSettings.DEFAULTS
                    .succinctTargetNotFoundErrors()),
            SymbolGenerator.create(basePackageId),
            PackageSettings.DEFAULTS.precomputeTransitiveLoads(),
            noImplicitFileExport,
            /* workspaceName= */ DUMMY_WORKSPACE_NAME_FOR_BZLMOD_PACKAGES,
            /* mainRepositoryMapping= */ null,
            /* cpuBoundSemaphore= */ null,
            PackageOverheadEstimator.NOOP_ESTIMATOR,
            /* generatorMap= */ null,
            /* globber= */ null)
        .setLoads(ImmutableList.of());
  }

  // ==== Non-trivial nested classes ====

  /**
   * A builder for {@link Package} objects. Only intended to be used by {@link PackageFactory} and
   * {@link com.google.devtools.build.lib.skyframe.PackageFunction}.
   */
  public static class Builder extends TargetDefinitionContext {

    /**
     * A bundle of options affecting package construction, that is not specific to any particular
     * package.
     */
    public interface PackageSettings {
      /**
       * Returns whether or not extra detail should be added to {@link NoSuchTargetException}s
       * thrown from {@link #getTarget}. Useful for toning down verbosity in situations where it can
       * be less helpful.
       */
      // TODO(bazel-team): Arguably, this could be replaced by a boolean param to getTarget(), or
      // some separate action taken by the caller. But there's a lot of call sites that would need
      // updating.
      default boolean succinctTargetNotFoundErrors() {
        return false;
      }

      /**
       * Determines whether to precompute a list of transitively loaded starlark files while
       * building packages.
       *
       * <p>Typically, direct loads are stored as a {@code ImmutableList<Module>}. This is
       * sufficient to reconstruct the full load graph by recursively traversing {@link
       * BazelModuleContext#loads}. If the package is going to be serialized, however, it may make
       * more sense to precompute a flat list containing the labels of all transitively loaded bzl
       * files since {@link Module} is costly to serialize.
       *
       * <p>If this returns {@code true}, transitive loads are stored as an {@code
       * ImmutableList<Label>} and direct loads are not stored.
       */
      default boolean precomputeTransitiveLoads() {
        return false;
      }

      PackageSettings DEFAULTS = new PackageSettings() {};
    }

    private final SymbolGenerator<?> symbolGenerator;

    // Same as pkg.metadata.
    private final Metadata metadata;

    /**
     * The output instance for this builder. Needs to be instantiated and available with name info
     * throughout initialization. All other settings are applied during {@link #build}. See {@link
     * Package#Package} and {@link Package#finishInit} for details.
     */
    private final Package pkg;

    // Initialized from outside but also potentially set by `workspace()` function in WORKSPACE
    // file.
    private String workspaceName;

    private final Label buildFileLabel;

    private final boolean precomputeTransitiveLoads;
    private final boolean noImplicitFileExport;

    // The map from each repository to that repository's remappings map.
    // This is only used in the //external package, it is an empty map for all other packages.
    private final HashMap<RepositoryName, HashMap<String, RepositoryName>>
        externalPackageRepositoryMappings = new HashMap<>();

    /** Converts label literals to Label objects within this package. */
    private final LabelConverter labelConverter;

    /** Estimates the package overhead of this package. */
    private final PackageOverheadEstimator packageOverheadEstimator;

    /**
     * Semaphore held by the Skyframe thread when performing CPU work.
     *
     * <p>This should be released when performing I/O.
     */
    @Nullable // Only non-null when inside PackageFunction.compute and the semaphore is enabled.
    private final Semaphore cpuBoundSemaphore;

    // TreeMap so that the iteration order of variables is consistent regardless of insertion order
    // (which may change due to serialization). This is useful so that the serialized representation
    // is deterministic.
    private final TreeMap<String, String> makeEnv = new TreeMap<>();

    private final StoredEventHandler localEventHandler = new StoredEventHandler();

    @Nullable private String ioExceptionMessage = null;
    @Nullable private IOException ioException = null;
    @Nullable private DetailedExitCode ioExceptionDetailedExitCode = null;
    // TODO(#19922): Consider having separate containsErrors fields on Metadata and Package. In that
    // case, this field is replaced by the one on Metadata.
    private boolean containsErrors = false;
    // A package's FailureDetail field derives from the events on its Builder's event handler.
    // During package deserialization, those events are unavailable, because those events aren't
    // serialized [*]. Its FailureDetail value is serialized, however. During deserialization, that
    // value is assigned here, so that it can be assigned to the deserialized package.
    //
    // Likewise, during workspace part assembly, errors from parent parts should propagate to their
    // children.
    //
    // [*] Not in the context of the package, anyway. Skyframe values containing a package may
    // serialize events emitted during its construction/evaluation.
    @Nullable private FailureDetail failureDetailOverride = null;

    // Used by glob(). Null for contexts where glob() is disallowed, including WORKSPACE files and
    // some tests.
    @Nullable private final Globber globber;

    private final Map<Label, EnvironmentGroup> environmentGroups = new HashMap<>();

    // All targets added to the package.
    //
    // We use SnapshottableBiMap to help track insertion order of Rule targets, for use by
    // native.existing_rules().
    //
    // Use addOrReplaceTarget() to add new entries.
    private BiMap<String, Target> targets =
        new SnapshottableBiMap<>(target -> target instanceof Rule);

    // The snapshot of {@link #targets} for use in rule finalizer macros. Contains all
    // non-finalizer-instantiated rule targets (i.e. all rule targets except for those instantiated
    // in a finalizer or in a macro called from a finalizer).
    //
    // Initialized by expandAllRemainingMacros() and reset to null by beforeBuild().
    @Nullable private Map<String, Rule> rulesSnapshotViewForFinalizers;

    // All instances of symbolic macros created during package construction, indexed by id (not
    // name).
    private final Map<String, MacroInstance> macros = new LinkedHashMap<>();

    /**
     * Ids of all symbolic macros that have been declared but not yet evaluated.
     *
     * <p>These are listed in the order they were declared. (This probably doesn't matter, but let's
     * be protective against possible non-determinism.)
     *
     * <p>Generally, ordinary symbolic macros are evaluated eagerly and not added to this set, while
     * finalizers, as well as any macros called by finalizers, always use deferred evaluation and
     * end up in here.
     */
    private final Set<String> unexpandedMacros = new LinkedHashSet<>();

    /**
     * Represents the innermost currently executing symbolic macro, or null if none are running.
     *
     * <p>Logically, this is the top entry of a stack of frames where each frame corresponds to a
     * nested symbolic macro invocation. In actuality, symbolic macros do not necessarily run
     * eagerly when they are invoked, so this is not really a call stack per se. We leave it to the
     * pkgbuilder client to set the current frame, so that the choice of whether to push and pop, or
     * process a worklist of queued evaluations, is up to them.
     *
     * <p>The state of this field is used to determine what Starlark APIs are available (see user
     * documentation on {@code macro()} at {@link StarlarkRuleFunctionsApi#macro}), and to help
     * enforce naming requirements on targets and macros.
     */
    private MacroFrame currentMacroFrame = null;

    /**
     * Represents the state of a running symbolic macro (see {@link #currentMacroFrame}).
     * Semi-opaque.
     */
    static class MacroFrame {
      final MacroInstance macroInstance;
      // Most name conflicts are caught by checking the keys of the `targets` and `macros` maps.
      // It is not a conflict for a target or macro to have the same name as the macro it is
      // declared in, yet such a target or macro may still conflict with siblings in the same macro.
      // We use this bool to track whether or not a newly introduced macro, M, having the same name
      // as its parent (the current macro), would clash with an already defined sibling of M.
      private boolean mainSubmacroHasBeenDefined = false;

      MacroFrame(MacroInstance macroInstance) {
        this.macroInstance = macroInstance;
      }
    }

    private enum NameConflictCheckingPolicy {
      UNKNOWN,
      NOT_GUARANTEED,
      ENABLED;
    }

    /**
     * Whether to do all validation checks for name clashes among targets, macros, and output file
     * prefixes.
     *
     * <p>The {@code NOT_GUARANTEED} value should only be used when the package data has already
     * been validated, e.g. in package deserialization.
     *
     * <p>Setting it to {@code NOT_GUARANTEED} does not necessarily turn off *all* checking, just
     * some of the more expensive ones. Do not rely on being able to violate these checks.
     */
    private NameConflictCheckingPolicy nameConflictCheckingPolicy =
        NameConflictCheckingPolicy.UNKNOWN;

    /**
     * Stores labels for each rule so that we don't have to call the costly {@link Rule#getLabels}
     * twice (once for {@link #checkForInputOutputConflicts} and once for {@link #beforeBuild}).
     *
     * <p>This field is null if name conflict checking is disabled. It is also null after the
     * package is built.
     */
    // TODO(#19922): Technically we don't need to store entries for rules that were created by
    // macros; see rulesCreatedInMacros, below.
    @Nullable private Map<Rule, List<Label>> ruleLabels = new HashMap<>();

    /**
     * Stores labels of rule targets that were created in symbolic macros. We don't implicitly
     * create input files on behalf of such targets (though they may still be created on behalf of
     * other targets not in macros).
     *
     * <p>This field is null if name conflict checking is disabled. It is also null after the
     * package is built.
     */
    // TODO(#19922): This can be eliminated once we have Targets directly store a reference to the
    // MacroInstance that instantiated them. (This is a little nontrivial because we'd like to avoid
    // simply adding a new field to Target subclasses, and instead want to combine it with the
    // existing Package-typed field.)
    @Nullable private Set<Rule> rulesCreatedInMacros = new HashSet<>();

    /**
     * A map from names of targets declared in a symbolic macro which violate macro naming rules,
     * such as "lib%{name}-src.jar" implicit outputs in java rules, to the name of the macro
     * instance where they were declared.
     *
     * <p>This field is null if name conflict checking is disabled. The content of the map is
     * manipulated only in {@link #checkRuleAndOutputs}.
     */
    @Nullable
    private LinkedHashMap<String, String> macroNamespaceViolatingTargets = new LinkedHashMap<>();

    /**
     * A map from target name to the (innermost) macro instance that declared it. See {@link
     * Package#targetsToDeclaringMacros}.
     */
    private LinkedHashMap<String, MacroInstance> targetsToDeclaringMacros = new LinkedHashMap<>();

    /**
     * The collection of the prefixes of every output file. Maps each prefix to an arbitrary output
     * file having that prefix. Used for error reporting.
     *
     * <p>This field is null if name conflict checking is disabled. It is also null after the
     * package is built. The content of the map is manipulated only in {@link #checkRuleAndOutputs}.
     */
    @Nullable private Map<String, OutputFile> outputFilePrefixes = new HashMap<>();

    private final List<TargetPattern> registeredExecutionPlatforms = new ArrayList<>();
    private final List<TargetPattern> registeredToolchains = new ArrayList<>();

    /**
     * Tracks the index within {@link #registeredToolchains} of the first toolchain registered from
     * the WORKSPACE suffixes rather than the WORKSPACE file (if any).
     *
     * <p>This is needed to distinguish between these toolchains during resolution: toolchains
     * registered in WORKSPACE have precedence over those defined in non-root Bazel modules, which
     * in turn have precedence over those from the WORKSPACE suffixes.
     */
    private OptionalInt firstWorkspaceSuffixRegisteredToolchain = OptionalInt.empty();

    /** True iff the "package" function has already been called in this package. */
    private boolean packageFunctionUsed;

    private final Interner<ImmutableList<?>> listInterner = new ThreadCompatibleInterner<>();

    private final ImmutableMap<Location, String> generatorMap;

    private final TestSuiteImplicitTestsAccumulator testSuiteImplicitTestsAccumulator =
        new TestSuiteImplicitTestsAccumulator();

    /** Returns the "generator_name" to use for a given call site location in a BUILD file. */
    @Nullable
    String getGeneratorNameByLocation(Location loc) {
      return generatorMap.get(loc);
    }

    /**
     * Returns the value to use for {@code test_suite}s' {@code $implicit_tests} attribute, as-is,
     * when the {@code test_suite} doesn't specify an explicit, non-empty {@code tests} value. The
     * returned list is mutated by the package-building process - it may be observed to be empty or
     * incomplete before package loading is complete. When package loading is complete it will
     * contain the label of each non-manual test matching the provided tags in the package, in label
     * order.
     *
     * <p>This method <b>MUST</b> be called before the package is built - otherwise the requested
     * implicit tests won't be accumulated.
     */
    List<Label> getTestSuiteImplicitTestsRef(List<String> tags) {
      return testSuiteImplicitTestsAccumulator.getTestSuiteImplicitTestsRefForTags(tags);
    }

    @ThreadCompatible
    private static final class ThreadCompatibleInterner<T> implements Interner<T> {
      private final Map<T, T> interns = new HashMap<>();

      @Override
      public T intern(T sample) {
        T existing = interns.putIfAbsent(sample, sample);
        return firstNonNull(existing, sample);
      }
    }

    private boolean alreadyBuilt = false;

    private Builder(
        Metadata metadata,
        SymbolGenerator<?> symbolGenerator,
        boolean precomputeTransitiveLoads,
        boolean noImplicitFileExport,
        String workspaceName,
        RepositoryMapping mainRepositoryMapping,
        @Nullable Semaphore cpuBoundSemaphore,
        PackageOverheadEstimator packageOverheadEstimator,
        @Nullable ImmutableMap<Location, String> generatorMap,
        @Nullable Globber globber) {
      super(mainRepositoryMapping);
      this.metadata = metadata;
      this.pkg = new Package(metadata);
      this.symbolGenerator = symbolGenerator;
      this.workspaceName = Preconditions.checkNotNull(workspaceName);

      try {
        this.buildFileLabel =
            Label.create(
                metadata.packageIdentifier(),
                metadata.buildFilename().getRootRelativePath().getBaseName());
      } catch (LabelSyntaxException e) {
        // This can't actually happen.
        throw new AssertionError(
            "Package BUILD file has an illegal name: " + metadata.buildFilename(), e);
      }

      this.precomputeTransitiveLoads = precomputeTransitiveLoads;
      this.noImplicitFileExport = noImplicitFileExport;
      this.labelConverter =
          new LabelConverter(metadata.packageIdentifier(), metadata.repositoryMapping());
      if (metadata.getName().startsWith("javatests/")) {
        mergePackageArgsFrom(PackageArgs.builder().setDefaultTestOnly(true));
      }
      this.cpuBoundSemaphore = cpuBoundSemaphore;
      this.packageOverheadEstimator = packageOverheadEstimator;
      this.generatorMap = (generatorMap == null) ? ImmutableMap.of() : generatorMap;
      this.globber = globber;

      // Add target for the BUILD file itself.
      // (This may be overridden by an exports_file declaration.)
      addInputFile(
          new InputFile(
              pkg,
              buildFileLabel,
              Location.fromFile(metadata.buildFilename().asPath().toString())));
    }

    SymbolGenerator<?> getSymbolGenerator() {
      return symbolGenerator;
    }

    /** Retrieves this object from a Starlark thread. Returns null if not present. */
    @Nullable
    public static Builder fromOrNull(StarlarkThread thread) {
      StarlarkThreadContext ctx = thread.getThreadLocal(StarlarkThreadContext.class);
      return (ctx instanceof Builder) ? (Builder) ctx : null;
    }

    /**
     * Retrieves this object from a Starlark thread. If not present, throws {@code EvalException}
     * with an error message indicating that {@code what} can't be used in this Starlark
     * environment.
     *
     * <p>If {@code allowBuild} is false, this method also throws if we're currently executing a
     * BUILD file (or legacy macro called from a BUILD file).
     *
     * <p>If {@code allowFinalizers} is false, this method also throws if we're currently executing
     * a rule finalizer implementation (or a legacy macro called from within such an
     * implementation).
     *
     * <p>If {@code allowNonFinalizerSymbolicMacros} is false, this method also throws if we're
     * currently executing the implementation of a symbolic macro implementation which is not a rule
     * finalizer (or a legacy macro called from within such an implementation).
     *
     * <p>If {@code allowWorkspace} is false, this method also throws if we're currently executing a
     * WORKSPACE file (or a legacy macro called from a WORKSPACE file).
     *
     * <p>It is not allowed for all three bool params to be false.
     */
    @CanIgnoreReturnValue
    public static Builder fromOrFail(
        StarlarkThread thread,
        String what,
        boolean allowBuild,
        boolean allowFinalizers,
        boolean allowNonFinalizerSymbolicMacros,
        boolean allowWorkspace)
        throws EvalException {
      Preconditions.checkArgument(
          allowBuild || allowFinalizers || allowNonFinalizerSymbolicMacros || allowWorkspace);

      @Nullable StarlarkThreadContext ctx = thread.getThreadLocal(StarlarkThreadContext.class);
      boolean bad = false;
      if (ctx instanceof Builder builder) {
        bad |= !allowBuild && !builder.isRepoRulePackage();
        bad |= !allowFinalizers && builder.currentlyInFinalizer();
        bad |= !allowNonFinalizerSymbolicMacros && builder.currentlyInNonFinalizerMacro();
        bad |= !allowWorkspace && builder.isRepoRulePackage();
        if (!bad) {
          return builder;
        }
      }

      boolean symbolicMacrosEnabled =
          thread
              .getSemantics()
              .getBool(BuildLanguageOptions.EXPERIMENTAL_ENABLE_FIRST_CLASS_MACROS);
      ArrayList<String> allowedUses = new ArrayList<>();
      if (allowBuild) {
        // Only disambiguate as "legacy" if the alternative, symbolic macros, are enabled.
        allowedUses.add(
            String.format("a BUILD file (or %smacro)", symbolicMacrosEnabled ? "legacy " : ""));
      }
      // Even if symbolic macros are allowed, don't mention them in the error message unless they
      // are enabled.
      if (symbolicMacrosEnabled) {
        if (allowFinalizers && allowNonFinalizerSymbolicMacros) {
          allowedUses.add("a symbolic macro");
        } else if (allowFinalizers) {
          allowedUses.add("a rule finalizer");
        } else if (allowNonFinalizerSymbolicMacros) {
          allowedUses.add("a non-finalizer symbolic macro");
        }
      }
      if (allowWorkspace) {
        allowedUses.add("a WORKSPACE file");
      }
      throw Starlark.errorf(
          "%s can only be used while evaluating %s", what, StringUtil.joinEnglishList(allowedUses));
    }

    /** Convenience method for {@link #fromOrFail} that permits any context with a Builder. */
    @CanIgnoreReturnValue
    public static Builder fromOrFail(StarlarkThread thread, String what) throws EvalException {
      return fromOrFail(
          thread,
          what,
          /* allowBuild= */ true,
          /* allowFinalizers= */ true,
          /* allowNonFinalizerSymbolicMacros= */ true,
          /* allowWorkspace= */ true);
    }

    /**
     * Convenience method for {@link #fromOrFail} that permits only BUILD contexts (without symbolic
     * macros).
     */
    @CanIgnoreReturnValue
    public static Builder fromOrFailAllowBuildOnly(StarlarkThread thread, String what)
        throws EvalException {
      return fromOrFail(
          thread,
          what,
          /* allowBuild= */ true,
          /* allowFinalizers= */ false,
          /* allowNonFinalizerSymbolicMacros= */ false,
          /* allowWorkspace= */ false);
    }

    /** Convenience method for {@link #fromOrFail} that permits only WORKSPACE contexts. */
    @CanIgnoreReturnValue
    public static Builder fromOrFailAllowWorkspaceOnly(StarlarkThread thread, String what)
        throws EvalException {
      return fromOrFail(
          thread,
          what,
          /* allowBuild= */ false,
          /* allowFinalizers= */ false,
          /* allowNonFinalizerSymbolicMacros= */ false,
          /* allowWorkspace= */ true);
    }

    /**
     * Convenience method for {@link #fromOrFail} that permits BUILD or WORKSPACE or rule finalizer
     * contexts.
     */
    @CanIgnoreReturnValue
    public static Builder fromOrFailDisallowNonFinalizerMacros(StarlarkThread thread, String what)
        throws EvalException {
      return fromOrFail(
          thread,
          what,
          /* allowBuild= */ true,
          /* allowFinalizers= */ true,
          /* allowNonFinalizerSymbolicMacros= */ false,
          /* allowWorkspace= */ true);
    }

    /**
     * Convenience method for {@link #fromOrFail} that permits BUILD or symbolic macro contexts
     * (including rule finalizers).
     */
    @CanIgnoreReturnValue
    public static Builder fromOrFailDisallowWorkspace(StarlarkThread thread, String what)
        throws EvalException {
      return fromOrFail(
          thread,
          what,
          /* allowBuild= */ true,
          /* allowFinalizers= */ true,
          /* allowNonFinalizerSymbolicMacros= */ true,
          /* allowWorkspace= */ false);
    }

    PackageIdentifier getPackageIdentifier() {
      return metadata.packageIdentifier();
    }

    /**
     * Determine whether this package should contain build rules (returns {@code false}) or repo
     * rules (returns {@code true}).
     */
    public boolean isRepoRulePackage() {
      return metadata.isRepoRulePackage();
    }

    /**
     * Returns the name of the workspace this package is in. Used as a prefix for the runfiles
     * directory. This can be set in the WORKSPACE file. This must be a valid target name.
     */
    String getWorkspaceName() {
      // Current value is stored in the builder field, final value is copied to the Package in
      // finishInit().
      return workspaceName;
    }

    /**
     * Returns the name of the Bzlmod module associated with the repo this package is in. If this
     * package is not from a Bzlmod repo, this is empty. For repos generated by module extensions,
     * this is the name of the module hosting the extension.
     */
    Optional<String> getAssociatedModuleName() {
      return metadata.associatedModuleName();
    }

    /**
     * Returns the version of the Bzlmod module associated with the repo this package is in. If this
     * package is not from a Bzlmod repo, this is empty. For repos generated by module extensions,
     * this is the version of the module hosting the extension.
     */
    Optional<String> getAssociatedModuleVersion() {
      return metadata.associatedModuleVersion();
    }

    /**
     * Updates the externalPackageRepositoryMappings entry for {@code repoWithin}. Adds new entry
     * from {@code localName} to {@code mappedName} in {@code repoWithin}'s map.
     *
     * @param repoWithin the RepositoryName within which the mapping should apply
     * @param localName the name that actually appears in the WORKSPACE and BUILD files in the
     *     {@code repoWithin} repository
     * @param mappedName the RepositoryName by which localName should be referenced
     */
    @CanIgnoreReturnValue
    Builder addRepositoryMappingEntry(
        RepositoryName repoWithin, String localName, RepositoryName mappedName) {
      HashMap<String, RepositoryName> mapping =
          externalPackageRepositoryMappings.computeIfAbsent(
              repoWithin, (RepositoryName k) -> new HashMap<>());
      mapping.put(localName, mappedName);
      return this;
    }

    /** Adds all the mappings from a given {@link Package}. */
    @CanIgnoreReturnValue
    Builder addRepositoryMappings(Package aPackage) {
      ImmutableMap<RepositoryName, ImmutableMap<String, RepositoryName>> repositoryMappings =
          aPackage.externalPackageRepositoryMappings;
      for (Map.Entry<RepositoryName, ImmutableMap<String, RepositoryName>> repositoryName :
          repositoryMappings.entrySet()) {
        for (Map.Entry<String, RepositoryName> repositoryNameRepositoryNameEntry :
            repositoryName.getValue().entrySet()) {
          addRepositoryMappingEntry(
              repositoryName.getKey(),
              repositoryNameRepositoryNameEntry.getKey(),
              repositoryNameRepositoryNameEntry.getValue());
        }
      }
      return this;
    }

    public LabelConverter getLabelConverter() {
      return labelConverter;
    }

    Interner<ImmutableList<?>> getListInterner() {
      return listInterner;
    }

    public Label getBuildFileLabel() {
      return buildFileLabel;
    }

    /**
     * Return a read-only copy of the name mapping of external repositories for a given repository.
     * Reading that mapping directly from the builder allows to also take mappings into account that
     * are only discovered while constructing the external package (e.g., the mapping of the name of
     * the main workspace to the canonical main name '@').
     */
    RepositoryMapping getRepositoryMappingFor(RepositoryName name) {
      Map<String, RepositoryName> mapping = externalPackageRepositoryMappings.get(name);
      if (mapping == null) {
        return RepositoryMapping.ALWAYS_FALLBACK;
      } else {
        return RepositoryMapping.createAllowingFallback(mapping);
      }
    }

    RootedPath getFilename() {
      return metadata.buildFilename();
    }

    /** Returns the {@link StoredEventHandler} associated with this builder. */
    public StoredEventHandler getLocalEventHandler() {
      return localEventHandler;
    }

    @CanIgnoreReturnValue
    public Builder setMakeVariable(String name, String value) {
      makeEnv.put(name, value);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder mergePackageArgsFrom(PackageArgs packageArgs) {
      pkg.packageArgs = pkg.packageArgs.mergeWith(packageArgs);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder mergePackageArgsFrom(PackageArgs.Builder builder) {
      return mergePackageArgsFrom(builder.build());
    }

    /** Called partial b/c in builder and thus subject to mutation and updates */
    public PackageArgs getPartialPackageArgs() {
      return pkg.packageArgs;
    }

    /** Uses the workspace name from {@code //external} to set this package's workspace name. */
    @CanIgnoreReturnValue
    @VisibleForTesting
    public Builder setWorkspaceName(String workspaceName) {
      this.workspaceName = workspaceName;
      return this;
    }

    /** Returns whether the "package" function has been called yet */
    boolean isPackageFunctionUsed() {
      return packageFunctionUsed;
    }

    void setPackageFunctionUsed() {
      packageFunctionUsed = true;
    }

    /** Sets the number of Starlark computation steps executed by this BUILD file. */
    void setComputationSteps(long n) {
      pkg.computationSteps = n;
    }

    Builder setIOException(IOException e, String message, DetailedExitCode detailedExitCode) {
      this.ioException = e;
      this.ioExceptionMessage = message;
      this.ioExceptionDetailedExitCode = detailedExitCode;
      return setContainsErrors();
    }

    /**
     * Declares that errors were encountering while loading this package. If called, {@link
     * #addEvent} or {@link #addEvents} should already have been called with an {@link Event} of
     * type {@link EventKind#ERROR} that includes a {@link FailureDetail}.
     */
    // TODO(bazel-team): For simplicity it would be nice to replace this with
    // getLocalEventHandler().hasErrors(), since that would prevent the kind of inconsistency where
    // we have reported an ERROR event but not called setContainsErrors(), or vice versa.
    @CanIgnoreReturnValue
    public Builder setContainsErrors() {
      // TODO(bazel-team): Maybe do Preconditions.checkState(localEventHandler.hasErrors()).
      // Maybe even assert that it has a FailureDetail, though that's a linear scan unless we
      // customize the event handler.
      containsErrors = true;
      return this;
    }

    public boolean containsErrors() {
      return containsErrors;
    }

    void setFailureDetailOverride(FailureDetail failureDetail) {
      failureDetailOverride = failureDetail;
    }

    @Nullable
    FailureDetail getFailureDetail() {
      if (failureDetailOverride != null) {
        return failureDetailOverride;
      }

      List<Event> undetailedEvents = null;
      for (Event event : localEventHandler.getEvents()) {
        if (event.getKind() != EventKind.ERROR) {
          continue;
        }
        DetailedExitCode detailedExitCode = event.getProperty(DetailedExitCode.class);
        if (detailedExitCode != null && detailedExitCode.getFailureDetail() != null) {
          return detailedExitCode.getFailureDetail();
        }
        if (containsErrors) {
          if (undetailedEvents == null) {
            undetailedEvents = new ArrayList<>();
          }
          undetailedEvents.add(event);
        }
      }
      if (undetailedEvents != null) {
        BugReport.sendNonFatalBugReport(
            new IllegalStateException("Package has undetailed error from " + undetailedEvents));
      }
      return null;
    }

    // TODO(#19922): Require this to be set before BUILD evaluation.
    @CanIgnoreReturnValue
    public Builder setLoads(Iterable<Module> directLoads) {
      checkLoadsNotSet();
      if (precomputeTransitiveLoads) {
        pkg.transitiveLoads = computeTransitiveLoads(directLoads);
      } else {
        pkg.directLoads = ImmutableList.copyOf(directLoads);
      }
      return this;
    }

    @CanIgnoreReturnValue
    Builder setTransitiveLoadsForDeserialization(ImmutableList<Label> transitiveLoads) {
      checkLoadsNotSet();
      pkg.transitiveLoads = Preconditions.checkNotNull(transitiveLoads);
      return this;
    }

    private void checkLoadsNotSet() {
      Preconditions.checkState(
          pkg.directLoads == null, "Direct loads already set: %s", pkg.directLoads);
      Preconditions.checkState(
          pkg.transitiveLoads == null, "Transitive loads already set: %s", pkg.transitiveLoads);
    }

    /**
     * Returns the {@link Globber} used to implement {@code glob()} functionality during BUILD
     * evaluation. Null for contexts where globbing is not possible, including WORKSPACE files and
     * some tests.
     */
    @Nullable
    public Globber getGlobber() {
      return globber;
    }

    /**
     * Creates a new {@link Rule} {@code r} where {@code r.getPackage()} is the {@link Package}
     * associated with this {@link Builder}.
     *
     * <p>The created {@link Rule} will have no output files and therefore will be in an invalid
     * state.
     */
    Rule createRule(
        Label label, RuleClass ruleClass, List<StarlarkThread.CallStackEntry> callstack) {
      return createRule(
          label,
          ruleClass,
          callstack.isEmpty() ? Location.BUILTIN : callstack.get(0).location,
          CallStack.compactInterior(callstack));
    }

    Rule createRule(
        Label label,
        RuleClass ruleClass,
        Location location,
        @Nullable CallStack.Node interiorCallStack) {
      return new Rule(pkg, label, ruleClass, location, interiorCallStack);
    }

    /**
     * Creates a new {@link MacroInstance} {@code m} where {@code m.getPackage()} is the {@link
     * Package} associated with this {@link Builder}.
     */
    MacroInstance createMacro(
        MacroClass macroClass, Map<String, Object> attrValues, int sameNameDepth) {
      MacroInstance parent = currentMacroFrame == null ? null : currentMacroFrame.macroInstance;
      return new MacroInstance(pkg, parent, macroClass, attrValues, sameNameDepth);
    }

    /**
     * Inserts a target into the targets map. Returns the previous target if one was present, or
     * null.
     */
    @CanIgnoreReturnValue
    @Nullable
    private Target addOrReplaceTarget(Target target) {
      Target existing = targets.put(target.getName(), target);
      if (currentMacroFrame != null) {
        targetsToDeclaringMacros.put(target.getName(), currentMacroFrame.macroInstance);
      }
      return existing;
    }

    @Nullable
    Target getTarget(String name) {
      return targets.get(name);
    }

    /**
     * Replaces a target in the {@link Package} under construction with a new target with the same
     * name and belonging to the same package.
     *
     * <p>Requires that {@link #disableNameConflictChecking} was not called.
     *
     * <p>A hack needed for {@link WorkspaceFactoryHelper}.
     */
    void replaceTarget(Target newTarget) {
      ensureNameConflictChecking();

      Preconditions.checkArgument(
          targets.containsKey(newTarget.getName()),
          "No existing target with name '%s' in the targets map",
          newTarget.getName());
      Preconditions.checkArgument(
          newTarget.getPackage() == pkg, // pointer comparison since we're constructing `pkg`
          "Replacement target belongs to package '%s', expected '%s'",
          newTarget.getPackage(),
          pkg);
      Target oldTarget = addOrReplaceTarget(newTarget);
      if (newTarget instanceof Rule) {
        List<Label> ruleLabelsForOldTarget = ruleLabels.remove(oldTarget);
        if (ruleLabelsForOldTarget != null) {
          // TODO(brandjon): Can the new target have different labels than the old? If so, we
          // probably need newTarget.getLabels() here instead. Moot if we can delete this along with
          // WORKSPACE logic.
          ruleLabels.put((Rule) newTarget, ruleLabelsForOldTarget);
        }
      }
    }

    public Set<Target> getTargets() {
      return Package.getTargets(targets);
    }

    /**
     * Returns a lightweight snapshot view of the names of all rule targets belonging to this
     * package at the time of this call; in finalizer expansion stage, returns a lightweight
     * snapshot view of only the non-finalizer-instantiated rule targets.
     *
     * @throws IllegalStateException if this method is called after {@link #beforeBuild} has been
     *     called.
     */
    Map<String, Rule> getRulesSnapshotView() {
      if (rulesSnapshotViewForFinalizers != null) {
        return rulesSnapshotViewForFinalizers;
      } else if (targets instanceof SnapshottableBiMap<?, ?>) {
        return Maps.transformValues(
            ((SnapshottableBiMap<String, Target>) targets).getTrackedSnapshot(),
            target -> (Rule) target);
      } else {
        throw new IllegalStateException(
            "getRulesSnapshotView() cannot be used after beforeBuild() has been called");
      }
    }

    /**
     * Returns an {@link Iterable} of all the rule instance targets belonging to this package.
     *
     * <p>The returned {@link Iterable} will be deterministically ordered, in the order the rule
     * instance targets were instantiated.
     */
    private Iterable<Rule> getRules() {
      return Package.getTargets(targets, Rule.class);
    }

    /**
     * Creates an input file target in this package with the specified name, if it does not yet
     * exist.
     *
     * <p>This operation is idempotent.
     *
     * @param targetName name of the input file. This must be a valid target name as defined by
     *     {@link com.google.devtools.build.lib.cmdline.LabelValidator#validateTargetName}.
     * @return the newly-created {@code InputFile}, or the old one if it already existed.
     * @throws NameConflictException if the name was already taken by another target that is not an
     *     input file
     * @throws IllegalArgumentException if the name is not a valid label
     */
    InputFile createInputFile(String targetName, Location location) throws NameConflictException {
      Target existing = targets.get(targetName);

      if (existing instanceof InputFile) {
        return (InputFile) existing; // idempotent
      }

      InputFile inputFile;
      try {
        inputFile = new InputFile(pkg, createLabel(targetName), location);
      } catch (LabelSyntaxException e) {
        throw new IllegalArgumentException(
            "FileTarget in package " + metadata.getName() + " has illegal name: " + targetName, e);
      }

      checkTargetName(inputFile);
      addInputFile(inputFile);
      return inputFile;
    }

    /**
     * Sets the visibility and license for an input file. The input file must already exist as a
     * member of this package.
     *
     * @throws IllegalArgumentException if the input file doesn't exist in this package's target
     *     map.
     */
    // TODO: #19922 - Don't allow exports_files() to modify visibility of targets that the current
    // symbolic macro did not create. Fun pathological example: exports_files() modifying the
    // visibility of :BUILD inside a symbolic macro.
    void setVisibilityAndLicense(InputFile inputFile, RuleVisibility visibility, License license) {
      String filename = inputFile.getName();
      Target cacheInstance = targets.get(filename);
      if (!(cacheInstance instanceof InputFile)) {
        throw new IllegalArgumentException(
            "Can't set visibility for nonexistent FileTarget "
                + filename
                + " in package "
                + metadata.getName()
                + ".");
      }
      if (!((InputFile) cacheInstance).isVisibilitySpecified()
          || cacheInstance.getVisibility() != visibility
          || !Objects.equals(cacheInstance.getLicense(), license)) {
        addOrReplaceTarget(
            new VisibilityLicenseSpecifiedInputFile(
                pkg, cacheInstance.getLabel(), cacheInstance.getLocation(), visibility, license));
      }
    }

    /**
     * Creates a label for a target inside this package.
     *
     * @throws LabelSyntaxException if the {@code targetName} is invalid
     */
    Label createLabel(String targetName) throws LabelSyntaxException {
      return Label.create(metadata.packageIdentifier(), targetName);
    }

    /** Adds a package group to the package. */
    void addPackageGroup(
        String name,
        Collection<String> packages,
        Collection<Label> includes,
        boolean allowPublicPrivate,
        boolean repoRootMeansCurrentRepo,
        EventHandler eventHandler,
        Location location)
        throws NameConflictException, LabelSyntaxException {
      PackageGroup group =
          new PackageGroup(
              createLabel(name),
              pkg,
              packages,
              includes,
              allowPublicPrivate,
              repoRootMeansCurrentRepo,
              eventHandler,
              location);
      checkTargetName(group);
      addOrReplaceTarget(group);

      if (group.containsErrors()) {
        setContainsErrors();
      }
    }

    /**
     * Returns true if any labels in the given list appear multiple times, reporting an appropriate
     * error message if so.
     *
     * <p>TODO(bazel-team): apply this to all build functions (maybe automatically?), possibly
     * integrate with RuleClass.checkForDuplicateLabels.
     */
    private static boolean hasDuplicateLabels(
        List<Label> labels,
        String owner,
        String attrName,
        Location location,
        EventHandler eventHandler) {
      Set<Label> dupes = CollectionUtils.duplicatedElementsOf(labels);
      for (Label dupe : dupes) {
        eventHandler.handle(
            error(
                location,
                String.format(
                    "label '%s' is duplicated in the '%s' list of '%s'", dupe, attrName, owner),
                Code.DUPLICATE_LABEL));
      }
      return !dupes.isEmpty();
    }

    /** Adds an environment group to the package. Not valid within symbolic macros. */
    void addEnvironmentGroup(
        String name,
        List<Label> environments,
        List<Label> defaults,
        EventHandler eventHandler,
        Location location)
        throws NameConflictException, LabelSyntaxException {
      Preconditions.checkState(currentMacroFrame == null);

      if (hasDuplicateLabels(environments, name, "environments", location, eventHandler)
          || hasDuplicateLabels(defaults, name, "defaults", location, eventHandler)) {
        setContainsErrors();
        return;
      }

      EnvironmentGroup group =
          new EnvironmentGroup(createLabel(name), pkg, environments, defaults, location);
      checkTargetName(group);
      addOrReplaceTarget(group);

      // Invariant: once group is inserted into targets, it must also:
      // (a) be inserted into environmentGroups, or
      // (b) have its group.processMemberEnvironments called.
      // Otherwise it will remain uninitialized,
      // causing crashes when it is later toString-ed.

      for (Event error : group.validateMembership()) {
        eventHandler.handle(error);
        setContainsErrors();
      }

      // For each declared environment, make sure it doesn't also belong to some other group.
      for (Label environment : group.getEnvironments()) {
        EnvironmentGroup otherGroup = environmentGroups.get(environment);
        if (otherGroup != null) {
          eventHandler.handle(
              error(
                  location,
                  String.format(
                      "environment %s belongs to both %s and %s",
                      environment, group.getLabel(), otherGroup.getLabel()),
                  Code.ENVIRONMENT_IN_MULTIPLE_GROUPS));
          setContainsErrors();
          // Ensure the orphan gets (trivially) initialized.
          group.processMemberEnvironments(ImmutableMap.of());
        } else {
          environmentGroups.put(environment, group);
        }
      }
    }

    /**
     * Turns off (some) conflict checking for name clashes between targets, macros, and output file
     * prefixes. (It is not guaranteed to disable all checks, since it is intended as an
     * optimization and not for semantic effect.)
     *
     * <p>This should only be done for data that has already been validated, e.g. during package
     * deserialization. Do not call this unless you know what you're doing.
     *
     * <p>This method must be called prior to {@link #addRuleUnchecked}. It may not be called,
     * neither before nor after, a call to {@link #addRule} or {@link #replaceTarget}.
     */
    @CanIgnoreReturnValue
    Builder disableNameConflictChecking() {
      Preconditions.checkState(nameConflictCheckingPolicy == NameConflictCheckingPolicy.UNKNOWN);
      this.nameConflictCheckingPolicy = NameConflictCheckingPolicy.NOT_GUARANTEED;
      this.ruleLabels = null;
      this.rulesCreatedInMacros = null;
      this.macroNamespaceViolatingTargets = null;
      this.outputFilePrefixes = null;
      return this;
    }

    private void ensureNameConflictChecking() {
      Preconditions.checkState(
          nameConflictCheckingPolicy != NameConflictCheckingPolicy.NOT_GUARANTEED);
      this.nameConflictCheckingPolicy = NameConflictCheckingPolicy.ENABLED;
    }

    /**
     * Adds a rule and its outputs to the targets map, and propagates the error bit from the rule to
     * the package.
     */
    private void addRuleInternal(Rule rule) {
      Preconditions.checkArgument(rule.getPackage() == pkg);
      for (OutputFile outputFile : rule.getOutputFiles()) {
        addOrReplaceTarget(outputFile);
      }
      addOrReplaceTarget(rule);
      if (rule.containsErrors()) {
        this.setContainsErrors();
      }
    }

    /**
     * Adds a rule without certain validation checks. Requires that {@link
     * #disableNameConflictChecking} was already called.
     */
    void addRuleUnchecked(Rule rule) {
      Preconditions.checkState(
          nameConflictCheckingPolicy == NameConflictCheckingPolicy.NOT_GUARANTEED);
      addRuleInternal(rule);
    }

    /**
     * Adds a rule, subject to the usual validation checks. Requires that {@link
     * #disableNameConflictChecking} was not called.
     */
    void addRule(Rule rule) throws NameConflictException {
      ensureNameConflictChecking();

      List<Label> labels = rule.getLabels();
      checkRuleAndOutputs(rule, labels);
      addRuleInternal(rule);
      ruleLabels.put(rule, labels);
      if (currentMacroFrame != null) {
        rulesCreatedInMacros.add(rule);
      }
    }

    /** Adds a symbolic macro instance to the package. */
    public void addMacro(MacroInstance macro) throws NameConflictException {
      Preconditions.checkState(
          !isRepoRulePackage(), "Cannot instantiate symbolic macros in this context");

      checkMacroName(macro);
      Object prev = macros.put(macro.getId(), macro);
      Preconditions.checkState(prev == null);
      unexpandedMacros.add(macro.getId());

      // Track whether a main submacro has been seen yet. Conflict checking for this is done in
      // checkMacroName().
      if (currentMacroFrame != null) {
        if (macro.getName().equals(currentMacroFrame.macroInstance.getName())) {
          currentMacroFrame.mainSubmacroHasBeenDefined = true;
        }
      }
    }

    /**
     * Marks a symbolic macro as having finished evaluating.
     *
     * <p>This will prevent the macro from being run by {@link #expandAllRemainingMacros}.
     *
     * <p>The macro must not have previously been marked complete.
     */
    public void markMacroComplete(MacroInstance macro) {
      String id = macro.getId();
      if (!unexpandedMacros.remove(id)) {
        throw new IllegalArgumentException(
            String.format("Macro id '%s' unknown or already marked complete", id));
      }
    }

    /** Returns the current macro frame, or null if there is no currently running symbolic macro. */
    @Nullable
    MacroFrame getCurrentMacroFrame() {
      return currentMacroFrame;
    }

    /**
     * Returns true if a symbolic macro is running and the current macro frame is not a rule
     * finalizer.
     *
     * <p>Note that this function examines only the current macro frame, not any parent frames; and
     * thus returns true even if the current non-finalizer macro was called within a finalizer
     * macro.
     */
    public boolean currentlyInNonFinalizerMacro() {
      return currentMacroFrame != null
          ? !currentMacroFrame.macroInstance.getMacroClass().isFinalizer()
          : false;
    }

    /**
     * Returns true if a symbolic macro is running and the current macro frame is a rule finalizer.
     */
    public boolean currentlyInFinalizer() {
      return currentMacroFrame != null
          ? currentMacroFrame.macroInstance.getMacroClass().isFinalizer()
          : false;
    }

    /**
     * Sets the current macro frame and returns the old one.
     *
     * <p>Either the new or old frame may be null, indicating no currently running symbolic macro.
     */
    @Nullable
    MacroFrame setCurrentMacroFrame(@Nullable MacroFrame frame) {
      MacroFrame prev = currentMacroFrame;
      currentMacroFrame = frame;
      return prev;
    }

    /**
     * If we are currently executing a symbolic macro, returns the result of unioning the given
     * visibility with the location of the innermost macro's code. Otherwise, returns the given
     * visibility unmodified.
     *
     * <p>The location of the macro's code is considered to be the package containing the .bzl file
     * from which the macro's {@code MacroClass} was exported.
     */
    RuleVisibility copyAppendingCurrentMacroLocation(RuleVisibility visibility) {
      if (currentMacroFrame == null) {
        return visibility;
      }
      MacroClass macroClass = currentMacroFrame.macroInstance.getMacroClass();
      PackageIdentifier macroLocation = macroClass.getDefiningBzlLabel().getPackageIdentifier();
      Label newVisibilityItem = Label.createUnvalidated(macroLocation, "__pkg__");

      if (visibility.equals(RuleVisibility.PRIVATE)) {
        // Private is dropped.
        return PackageGroupsRuleVisibility.create(ImmutableList.of(newVisibilityItem));
      } else if (visibility.equals(RuleVisibility.PUBLIC)) {
        // Public is idempotent.
        return visibility;
      } else {
        ImmutableList.Builder<Label> items = new ImmutableList.Builder<>();
        items.addAll(visibility.getDeclaredLabels());
        items.add(newVisibilityItem);
        return PackageGroupsRuleVisibility.create(items.build());
      }
    }

    void addRegisteredExecutionPlatforms(List<TargetPattern> platforms) {
      this.registeredExecutionPlatforms.addAll(platforms);
    }

    void addRegisteredToolchains(List<TargetPattern> toolchains, boolean forWorkspaceSuffix) {
      if (forWorkspaceSuffix && firstWorkspaceSuffixRegisteredToolchain.isEmpty()) {
        firstWorkspaceSuffixRegisteredToolchain = OptionalInt.of(registeredToolchains.size());
      }
      this.registeredToolchains.addAll(toolchains);
    }

    void setFirstWorkspaceSuffixRegisteredToolchain(
        OptionalInt firstWorkspaceSuffixRegisteredToolchain) {
      this.firstWorkspaceSuffixRegisteredToolchain = firstWorkspaceSuffixRegisteredToolchain;
    }

    /**
     * Ensures that all symbolic macros in the package have expanded.
     *
     * <p>This does not run any macro that has already been evaluated. It *does* run macros that are
     * newly discovered during the operation of this method.
     */
    public void expandAllRemainingMacros(StarlarkSemantics semantics) throws InterruptedException {
      // TODO: #19922 - Protect against unreasonable macro stack depth and large numbers of symbolic
      // macros overall, for both the eager and deferred evaluation strategies.

      // Note that this operation is idempotent for symmetry with build()/buildPartial(). Though
      // it's not entirely clear that this is necessary.

      // TODO: #19922 - Once compatibility with native.existing_rules() in legacy macros is no
      // longer a concern, we will want to support delayed expansion of non-finalizer macros before
      // the finalizer expansion step.

      // Finalizer expansion step.
      if (!unexpandedMacros.isEmpty()) {
        Preconditions.checkState(
            unexpandedMacros.stream().allMatch(id -> macros.get(id).getMacroClass().isFinalizer()),
            "At the beginning of finalizer expansion, unexpandedMacros must contain only"
                + " finalizers");

        // Save a snapshot of rule targets for use by native.existing_rules() inside all finalizers.
        // We must take this snapshot before calling any finalizer because the snapshot must not
        // include any rule instantiated by a finalizer or macro called from a finalizer.
        if (rulesSnapshotViewForFinalizers == null) {
          Preconditions.checkState(
              targets instanceof SnapshottableBiMap<?, ?>,
              "Cannot call expandAllRemainingMacros() after beforeBuild() has been called");
          rulesSnapshotViewForFinalizers = getRulesSnapshotView();
        }

        while (!unexpandedMacros.isEmpty()) { // NB: collection mutated by body
          String id = unexpandedMacros.iterator().next();
          MacroInstance macro = macros.get(id);
          MacroClass.executeMacroImplementation(macro, this, semantics);
        }
      }
    }

    @CanIgnoreReturnValue
    private Builder beforeBuild(boolean discoverAssumedInputFiles) throws NoSuchPackageException {
      // For correct semantics, we refuse to build a package that has declared symbolic macros that
      // have not yet been expanded. (Currently finalizers are the only use case where this happens,
      // but the Package logic is agnostic to that detail.)
      //
      // Production code should be calling expandAllRemainingMacros() to guarantee that nothing is
      // left unexpanded. Tests that do not declare any symbolic macros need not make the call.
      // Package deserialization doesn't have to do it either, since we shouldn't be evaluating
      // symbolic macros on the deserialized result of an already evaluated package.
      Preconditions.checkState(
          unexpandedMacros.isEmpty(),
          "Cannot build a package with unexpanded symbolic macros; call"
              + " expandAllRemainingMacros()");

      if (ioException != null) {
        throw new NoSuchPackageException(
            getPackageIdentifier(), ioExceptionMessage, ioException, ioExceptionDetailedExitCode);
      }

      // SnapshottableBiMap does not allow removing targets (in order to efficiently track rule
      // insertion order). However, we *do* need to support removal of targets in
      // PackageFunction.handleLabelsCrossingSubpackagesAndPropagateInconsistentFilesystemExceptions
      // which is called *between* calls to beforeBuild and finishBuild. We thus repoint the targets
      // map to the SnapshottableBiMap's underlying bimap and thus stop tracking insertion order.
      // After this point, snapshots of targets should no longer be used, and any further
      // getRulesSnapshotView calls will throw.
      if (targets instanceof SnapshottableBiMap<?, ?>) {
        targets = ((SnapshottableBiMap<String, Target>) targets).getUnderlyingBiMap();
        rulesSnapshotViewForFinalizers = null;
      }

      // We create an InputFile corresponding to the BUILD file in Builder's constructor. However,
      // the visibility of this target may be overridden with an exports_files directive, so we wait
      // until now to obtain the current instance from the targets map.
      pkg.buildFile = (InputFile) Preconditions.checkNotNull(targets.get(buildFileLabel.getName()));

      // TODO(bazel-team): We run testSuiteImplicitTestsAccumulator here in beforeBuild(), but what
      // if one of the accumulated tests is later removed in PackageFunction, between the call to
      // buildPartial() and finishBuild(), due to a label-crossing-subpackage-boundary error? Seems
      // like that would mean a test_suite is referencing a Target that's been deleted from its
      // Package.

      // Clear tests before discovering them again in order to keep this method idempotent -
      // otherwise we may double-count tests if we're called twice due to a skyframe restart, etc.
      testSuiteImplicitTestsAccumulator.clearAccumulatedTests();

      Map<String, InputFile> newInputFiles = new HashMap<>();
      for (Rule rule : getRules()) {
        if (discoverAssumedInputFiles) {
          // Labels mentioned by a rule that refer to an unknown target in the current package are
          // assumed to be InputFiles, unless they overlap a namespace owned by a macro. Create
          // these InputFiles now. But don't do this for rules created within a symbolic macro,
          // since we don't want the evaluation of the macro to affect the semantics of whether or
          // not this target was created (i.e. all implicitly created files are knowable without
          // necessarily evaluating symbolic macros).
          if (rulesCreatedInMacros.contains(rule)) {
            continue;
          }
          // We use a temporary map, newInputFiles, to avoid concurrent modification to this.targets
          // while iterating (via getRules() above).
          List<Label> labels = (ruleLabels != null) ? ruleLabels.get(rule) : rule.getLabels();
          for (Label label : labels) {
            String name = label.getName();
            if (label.getPackageIdentifier().equals(metadata.packageIdentifier())
                && !targets.containsKey(name)
                && !newInputFiles.containsKey(name)) {
              // Check for collision with a macro namespace. Currently this is a linear loop over
              // all symbolic macros in the package.
              // TODO(#19922): This is quadratic complexity, optimize with a trie or similar if
              // needed.
              boolean macroConflictsFound = false;
              for (MacroInstance macro : macros.values()) {
                macroConflictsFound |= nameIsWithinMacroNamespace(name, macro.getName());
              }
              if (!macroConflictsFound) {
                Location loc = rule.getLocation();
                newInputFiles.put(
                    name,
                    // Targets added this way are not in any macro, so
                    // copyAppendingCurrentMacroLocation() munging isn't applicable.
                    noImplicitFileExport
                        ? new PrivateVisibilityInputFile(pkg, label, loc)
                        : new InputFile(pkg, label, loc));
              }
            }
          }
        }

        testSuiteImplicitTestsAccumulator.processRule(rule);
      }

      // Make sure all accumulated values are sorted for determinism.
      testSuiteImplicitTestsAccumulator.sortTests();

      for (InputFile file : newInputFiles.values()) {
        addInputFile(file);
      }

      return this;
    }

    /** Intended for use by {@link com.google.devtools.build.lib.skyframe.PackageFunction} only. */
    // TODO(bazel-team): It seems like the motivation for this method (added in cl/74794332) is to
    // allow PackageFunction to delete targets that are found to violate the
    // label-crossing-subpackage-boundaries check. Is there a simpler way to express this idea that
    // doesn't make package-building a multi-stage process?
    @CanIgnoreReturnValue
    public Builder buildPartial() throws NoSuchPackageException {
      if (alreadyBuilt) {
        return this;
      }
      return beforeBuild(/* discoverAssumedInputFiles= */ true);
    }

    /** Intended for use by {@link com.google.devtools.build.lib.skyframe.PackageFunction} only. */
    public Package finishBuild() {
      if (alreadyBuilt) {
        return pkg;
      }

      // Freeze targets and distributions.
      for (Rule rule : getRules()) {
        rule.freeze();
      }
      ruleLabels = null;
      rulesCreatedInMacros = null;
      outputFilePrefixes = null;
      targets = Maps.unmodifiableBiMap(targets);

      // Now all targets have been loaded, so we validate the group's member environments.
      for (EnvironmentGroup envGroup : ImmutableSet.copyOf(environmentGroups.values())) {
        List<Event> errors = envGroup.processMemberEnvironments(targets);
        if (!errors.isEmpty()) {
          Event.replayEventsOn(localEventHandler, errors);
          // TODO(bazel-team): Can't we automatically infer containsError from the presence of
          // ERRORs on our handler?
          setContainsErrors();
        }
      }

      // Build the package.
      pkg.finishInit(this);
      alreadyBuilt = true;
      return pkg;
    }

    /** Completes package construction. Idempotent. */
    // TODO(brandjon): Do we actually care about idempotence?
    public Package build() throws NoSuchPackageException {
      return build(/* discoverAssumedInputFiles= */ true);
    }

    /**
     * Constructs the package (or does nothing if it's already built) and returns it.
     *
     * @param discoverAssumedInputFiles whether to automatically add input file targets to this
     *     package for "dangling labels", i.e. labels mentioned in this package that point to an
     *     up-until-now non-existent target in this package
     */
    Package build(boolean discoverAssumedInputFiles) throws NoSuchPackageException {
      if (alreadyBuilt) {
        return pkg;
      }
      beforeBuild(discoverAssumedInputFiles);
      return finishBuild();
    }

    /**
     * Adds an input file to this package.
     *
     * <p>There must not already be a target with the same name (i.e., this is not idempotent).
     */
    private void addInputFile(InputFile inputFile) {
      Target prev = addOrReplaceTarget(inputFile);
      Preconditions.checkState(prev == null);
    }

    /**
     * Precondition check for {@link #addRule} (to be called before the rule and its outputs are in
     * the targets map). Verifies that:
     *
     * <ul>
     *   <li>The added rule's name, and the names of its output files, are not the same as the name
     *       of any target already declared in the package.
     *   <li>The added rule's output files list does not contain the same name twice.
     *   <li>The added rule does not have an input file and an output file that share the same name.
     *   <li>For each of the added rule's output files, no directory prefix of that file matches the
     *       name of another output file in the package; and conversely, the file is not itself a
     *       prefix for another output file. (This check statefully mutates the {@code
     *       outputFilePrefixes} field.)
     * </ul>
     */
    // TODO(bazel-team): We verify that all prefixes of output files are distinct from other output
    // file names, but not that they're distinct from other target names in the package. What
    // happens if you define an input file "abc" and output file "abc/xyz"?
    private void checkRuleAndOutputs(Rule rule, List<Label> labels) throws NameConflictException {
      Preconditions.checkNotNull(outputFilePrefixes); // ensured by addRule's precondition

      // Check the name of the new rule itself.
      String ruleName = rule.getName();
      checkTargetName(rule);

      ImmutableList<OutputFile> outputFiles = rule.getOutputFiles();
      Map<String, OutputFile> outputFilesByName =
          Maps.newHashMapWithExpectedSize(outputFiles.size());

      // Check the new rule's output files, both for direct conflicts and prefix conflicts.
      for (OutputFile outputFile : outputFiles) {
        String outputFileName = outputFile.getName();
        // Check for duplicate within a single rule. (Can't use checkTargetName since this rule's
        // outputs aren't in the target map yet.)
        if (outputFilesByName.put(outputFileName, outputFile) != null) {
          throw new NameConflictException(
              String.format(
                  "rule '%s' has more than one generated file named '%s'",
                  ruleName, outputFileName));
        }
        // Check for conflict with any other already added target.
        checkTargetName(outputFile);
        // TODO(bazel-team): We also need to check for a conflict between an output file and its own
        // rule, which is not yet in the targets map.

        // Check if this output file is the prefix of an already existing one.
        if (outputFilePrefixes.containsKey(outputFileName)) {
          throw overlappingOutputFilePrefixes(outputFile, outputFilePrefixes.get(outputFileName));
        }

        // Check if a prefix of this output file matches an already existing one.
        PathFragment outputFileFragment = PathFragment.create(outputFileName);
        int segmentCount = outputFileFragment.segmentCount();
        for (int i = 1; i < segmentCount; i++) {
          String prefix = outputFileFragment.subFragment(0, i).toString();
          if (outputFilesByName.containsKey(prefix)) {
            throw overlappingOutputFilePrefixes(outputFile, outputFilesByName.get(prefix));
          }
          if (targets.get(prefix) instanceof OutputFile) {
            throw overlappingOutputFilePrefixes(outputFile, (OutputFile) targets.get(prefix));
          }

          // Store in persistent map, for checking when adding future rules.
          outputFilePrefixes.putIfAbsent(prefix, outputFile);
        }
      }

      // Check for the same file appearing as both an input and output of the new rule.
      PackageIdentifier packageIdentifier = rule.getLabel().getPackageIdentifier();
      for (Label inputLabel : labels) {
        if (packageIdentifier.equals(inputLabel.getPackageIdentifier())
            && outputFilesByName.containsKey(inputLabel.getName())) {
          throw new NameConflictException(
              String.format(
                  "rule '%s' has file '%s' as both an input and an output",
                  ruleName, inputLabel.getName()));
        }
      }
    }

    /**
     * Returns whether a given {@code name} is within the namespace that would be owned by a macro
     * called {@code macroName}.
     *
     * <p>This is purely a string operation and does not reference actual targets and macros.
     *
     * <p>A macro named "foo" owns the namespace consisting of "foo" and all "foo_${BAR}",
     * "foo-${BAR}", or "foo.${BAR}", where ${BAR} is a non-empty string. ("_" is the recommended
     * separator; "." is required for file extensions.) This criteria is transitive; a submacro's
     * namespace is a subset of the parent macro's namespace. Therefore, if a name is valid w.r.t.
     * the macro that declares it, it is also valid for all ancestor macros.
     *
     * <p>Note that just because a name is within a macro's namespace does not necessarily mean the
     * corresponding target or macro was declared within this macro.
     */
    private boolean nameIsWithinMacroNamespace(String name, String macroName) {
      if (name.equals(macroName)) {
        return true;
      } else if (name.startsWith(macroName)) {
        String suffix = name.substring(macroName.length());
        // 0-length suffix handled above.
        if (suffix.length() >= 2
            && (suffix.startsWith("_") || suffix.startsWith(".") || suffix.startsWith("-"))) {
          return true;
        }
      }
      return false;
    }

    /**
     * Throws {@link NameConflictException} if the given target's name can't be added because of a
     * conflict. If the given target's name violates symbolic macro naming rules, this method
     * doesn't throw but instead records that the target's name is in violation, so that an attempt
     * to use the target will fail during the analysis phase.
     *
     * <p>The given target must *not* have already been added (via {@link #addOrReplaceTarget}).
     *
     * <p>We defer enforcement of symbolic macro naming rules for targets to the analysis phase
     * because otherwise, we could not use java rules (which declare lib%{name}-src.jar implicit
     * outputs) transitively in any symbolic macro.
     */
    // TODO(#19922): Provide a way to allow targets which violate naming rules to be configured
    // (either only as a dep to other targets declared in the current macro, or also externally).
    // TODO(#19922): Ensure `bazel build //pkg:all` (or //pkg:*) ignores violating targets.
    private void checkTargetName(Target target) throws NameConflictException {
      checkForExistingTargetName(target);

      checkForExistingMacroName(target.getName(), "target");

      if (currentMacroFrame != null
          && !nameIsWithinMacroNamespace(
              target.getName(), currentMacroFrame.macroInstance.getName())) {
        macroNamespaceViolatingTargets.put(
            target.getName(), currentMacroFrame.macroInstance.getName());
      }
    }

    /**
     * Add all given map entries to the builder's map from names of targets declared in a symbolic
     * macro which violate macro naming rules to the name of the macro instance where they were
     * declared.
     *
     * <p>Intended to be used for package deserialization.
     */
    void putAllMacroNamespaceViolatingTargets(Map<String, String> macroNamespaceViolatingTargets) {
      if (this.macroNamespaceViolatingTargets == null) {
        this.macroNamespaceViolatingTargets = new LinkedHashMap<>();
      }
      this.macroNamespaceViolatingTargets.putAll(macroNamespaceViolatingTargets);
    }

    /**
     * Throws {@link NameConflictException} if the given target's name matches that of an existing
     * target in the package, or an existing macro in the package that is not its ancestor.
     *
     * <p>The given target must *not* have already been added (via {@link #addOrReplaceTarget}).
     */
    private void checkForExistingTargetName(Target target) throws NameConflictException {
      Target existing = targets.get(target.getName());
      if (existing == null) {
        return;
      }

      String subject = String.format("%s '%s'", target.getTargetKind(), target.getName());
      if (target instanceof OutputFile givenOutput) {
        subject += String.format(" in rule '%s'", givenOutput.getGeneratingRule().getName());
      }

      String object =
          existing instanceof OutputFile existingOutput
              ? String.format(
                  "generated file from rule '%s'", existingOutput.getGeneratingRule().getName())
              : existing.getTargetKind();
      object += ", defined at " + existing.getLocation();

      throw new NameConflictException(
          String.format("%s conflicts with existing %s", subject, object));
    }

    /**
     * Throws {@link NameConflictException} if the given macro's name can't be added, either because
     * of a conflict or because of a violation of symbolic macro naming rules (if applicable).
     *
     * <p>The given macro must *not* have already been added (via {@link #addMacro}).
     */
    private void checkMacroName(MacroInstance macro) throws NameConflictException {
      String name = macro.getName();

      // A macro can share names with its main target but no other target. Since the macro hasn't
      // even been added yet, it hasn't run, and its main target is not yet defined. Therefore, any
      // match in the targets map represents a real conflict.
      Target existingTarget = targets.get(name);
      if (existingTarget != null) {
        throw new NameConflictException(
            String.format("macro '%s' conflicts with an existing target.", name));
      }

      checkForExistingMacroName(name, "macro");

      if (currentMacroFrame != null
          && !nameIsWithinMacroNamespace(name, currentMacroFrame.macroInstance.getName())) {
        throw new MacroNamespaceViolationException(
            String.format(
                "macro '%s' cannot declare submacro named '%s'. %s",
                currentMacroFrame.macroInstance.getName(), name, MACRO_NAMING_RULES));
      }
    }

    /**
     * Throws {@link NameConflictException} if the given name (of a hypothetical target or macro)
     * matches the name of an existing macro in the package, and the existing macro is not currently
     * executing (i.e. on the macro stack).
     *
     * <p>{@code what} must be either "macro" or "target".
     */
    private void checkForExistingMacroName(String name, String what) throws NameConflictException {
      // Macros are indexed by id, not name, so we can't just use macros.get() directly.
      // Instead, we reason that if at least one macro by the given name exists, then there is one
      // with an id suffix of ":1".
      MacroInstance existing = macros.get(name + ":1");
      if (existing == null) {
        return;
      }

      // A conflict is still ok if it's only with enclosing macros. It's enough to check that 1) we
      // have the same name as the immediately enclosing macro (relying inductively on the check
      // that was done when that macro was added), and 2) there is no sibling macro of the same name
      // already defined in the current frame.
      if (currentMacroFrame != null) {
        if (name.equals(currentMacroFrame.macroInstance.getName())
            && !currentMacroFrame.mainSubmacroHasBeenDefined) {
          return;
        }
      }

      // TODO(#19922): Add definition location info for the existing object, like we have in
      // checkForExistingTargetName. Complicated by the fact that there may be more than one macro
      // of that name.
      throw new NameConflictException(
          String.format(
              "%s '%s' conflicts with an existing macro (and was not created by it)", what, name));
    }

    /**
     * Returns a {@link NameConflictException} about two output files clashing (i.e., due to one
     * being a prefix of the other)
     */
    private static NameConflictException overlappingOutputFilePrefixes(
        OutputFile added, OutputFile existing) {
      if (added.getGeneratingRule() == existing.getGeneratingRule()) {
        return new NameConflictException(
            String.format(
                "rule '%s' has conflicting output files '%s' and '%s'",
                added.getGeneratingRule().getName(), added.getName(), existing.getName()));
      } else {
        return new NameConflictException(
            String.format(
                "output file '%s' of rule '%s' conflicts with output file '%s' of rule '%s'",
                added.getName(),
                added.getGeneratingRule().getName(),
                existing.getName(),
                existing.getGeneratingRule().getName()));
      }
    }

    @Nullable
    public Semaphore getCpuBoundSemaphore() {
      return cpuBoundSemaphore;
    }
  }

  /** A collection of data that is known before BUILD file evaluation even begins. */
  public record Metadata(
      PackageIdentifier packageIdentifier,
      RootedPath buildFilename,
      boolean isRepoRulePackage,
      RepositoryMapping repositoryMapping,
      Optional<String> associatedModuleName,
      Optional<String> associatedModuleVersion,
      @Nullable ConfigSettingVisibilityPolicy configSettingVisibilityPolicy,
      boolean succinctTargetNotFoundErrors) {

    public Metadata {
      Preconditions.checkNotNull(packageIdentifier);
      Preconditions.checkNotNull(buildFilename);
      Preconditions.checkNotNull(repositoryMapping);
      Preconditions.checkNotNull(associatedModuleName);
      Preconditions.checkNotNull(associatedModuleVersion);

      // Check for consistency between isRepoRulePackage and whether the buildFilename is a
      // WORKSPACE / MODULE.bazel file.
      String baseName = buildFilename.asPath().getBaseName();
      boolean isWorkspaceFile =
          baseName.equals(LabelConstants.WORKSPACE_DOT_BAZEL_FILE_NAME.getPathString())
              || baseName.equals(LabelConstants.WORKSPACE_FILE_NAME.getPathString());
      boolean isModuleDotBazelFile =
          baseName.equals(LabelConstants.MODULE_DOT_BAZEL_FILE_NAME.getPathString());
      Preconditions.checkArgument(isRepoRulePackage == (isWorkspaceFile || isModuleDotBazelFile));
    }

    /** Returns the name of this package (sans repository), e.g. "foo/bar". */
    public String getName() {
      return packageIdentifier.getPackageFragment().getPathString();
    }

    /**
     * Returns the directory in which this package's BUILD file resides.
     *
     * <p>All InputFile members of the packages are located relative to this directory.
     */
    public Path getPackageDirectory() {
      return buildFilename.asPath().getParentDirectory();
    }
  }

  /** Package codec implementation. */
  @VisibleForTesting
  static final class PackageCodec implements ObjectCodec<Package> {
    @Override
    public Class<Package> getEncodedClass() {
      return Package.class;
    }

    @Override
    public void serialize(SerializationContext context, Package input, CodedOutputStream codedOut)
        throws IOException, SerializationException {
      context.checkClassExplicitlyAllowed(Package.class, input);
      PackageCodecDependencies codecDeps = context.getDependency(PackageCodecDependencies.class);
      codecDeps.getPackageSerializer().serialize(context, input, codedOut);
    }

    @Override
    public Package deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      PackageCodecDependencies codecDeps = context.getDependency(PackageCodecDependencies.class);
      return codecDeps.getPackageSerializer().deserialize(context, codedIn);
    }
  }
}
