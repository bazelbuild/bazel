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

  private final Metadata metadata;

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

  /**
   * The collection of all targets defined in this package, indexed by name.
   *
   * <p>Note that a target and a macro may share the same name.
   */
  private ImmutableSortedMap<String, Target> targets;

  /**
   * The collection of all symbolic macro instances defined in this package, indexed by name.
   *
   * <p>Note that a target and a macro may share the same name.
   */
  // TODO(#19922): Enforce that macro namespaces are "exclusive", meaning that target names may only
  // suffix a macro name when the target is created (transitively) within the macro.
  private ImmutableSortedMap<String, MacroInstance> macros;

  public PackageArgs getPackageArgs() {
    return metadata.packageArgs;
  }

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

  /**
   * The map from each repository to that repository's remappings map. This is only used in the
   * //external package, it is an empty map for all other packages. For example, an entry of {"@foo"
   * : {"@x", "@y"}} indicates that, within repository foo, "@x" should be remapped to "@y".
   */
  private ImmutableMap<RepositoryName, ImmutableMap<String, RepositoryName>>
      externalPackageRepositoryMappings;

  /**
   * A rough approximation of the memory and general accounting costs associated with a loaded
   * package. A value of -1 means it is unset. Stored as a long to take up less memory per pkg.
   */
  private long packageOverhead = PACKAGE_OVERHEAD_UNSET;

  /** Returns package overhead as configured by the configured {@link PackageOverheadEstimator}. */
  public OptionalLong getPackageOverhead() {
    return packageOverhead == PACKAGE_OVERHEAD_UNSET
        ? OptionalLong.empty()
        : OptionalLong.of(packageOverhead);
  }

  /** Sets package overhead as configured by the configured {@link PackageOverheadEstimator}. */
  void setPackageOverhead(OptionalLong packageOverhead) {
    this.packageOverhead =
        packageOverhead.isPresent() ? packageOverhead.getAsLong() : PACKAGE_OVERHEAD_UNSET;
  }

  private ImmutableList<TargetPattern> registeredExecutionPlatforms;
  private ImmutableList<TargetPattern> registeredToolchains;
  private OptionalInt firstWorkspaceSuffixRegisteredToolchain;
  private long computationSteps;

  /** Returns the number of Starlark computation steps executed by this BUILD file. */
  public long getComputationSteps() {
    return computationSteps;
  }

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
  }

  /** Returns this package's identifier. */
  public PackageIdentifier getPackageIdentifier() {
    return metadata.packageIdentifier;
  }

  /**
   * Whether this package should contain only repo rules (returns {@code true}) or only build rules
   * (returns {@code false}).
   */
  private boolean isRepoRulePackage() {
    return metadata.packageIdentifier.equals(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER)
        || metadata.workspaceName.equals(DUMMY_WORKSPACE_NAME_FOR_BZLMOD_PACKAGES);
  }

  /**
   * Returns the repository mapping for the requested external repository.
   *
   * @throws UnsupportedOperationException if called from a package other than the //external
   *     package
   */
  public ImmutableMap<String, RepositoryName> getRepositoryMapping(RepositoryName repository) {
    if (!isRepoRulePackage()) {
      throw new UnsupportedOperationException(
          "Can only access the external package repository"
              + "mappings from the //external package");
    }
    return externalPackageRepositoryMappings.getOrDefault(repository, ImmutableMap.of());
  }

  /** Get the repository mapping for this package. */
  public RepositoryMapping getRepositoryMapping() {
    return metadata.repositoryMapping;
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

  /**
   * Returns the source root (a directory) beneath which this package's BUILD file was found, or
   * {@link Optional#empty} if this package was derived from a workspace file.
   *
   * <p>Assumes invariant: If non-empty, {@code
   * getSourceRoot().get().getRelative(packageId.getSourceRoot()).equals(getPackageDirectory())}
   */
  public Optional<Root> getSourceRoot() {
    return metadata.sourceRoot;
  }

  private static Root getSourceRoot(RootedPath buildFileRootedPath, PathFragment packageFragment) {
    PathFragment packageDirectory = buildFileRootedPath.getRootRelativePath().getParentDirectory();
    if (packageFragment.equals(packageDirectory)) {
      // Fast path: BUILD file path and package name are the same, don't create an extra root.
      return buildFileRootedPath.getRoot();
    }
    PathFragment current = buildFileRootedPath.asPath().asFragment().getParentDirectory();
    for (int i = 0, len = packageFragment.segmentCount(); i < len && current != null; i++) {
      current = current.getParentDirectory();
    }
    if (current == null || current.isEmpty()) {
      // This is never really expected to work. The check below in #finishInit should fail.
      return buildFileRootedPath.getRoot();
    }
    // Note that current is an absolute path.
    return Root.fromPath(buildFileRootedPath.getRoot().getRelative(current));
  }

  /**
   * Completes the initialization of this package. Only after this method may a package by shared
   * publicly.
   */
  private void finishInit(Builder builder) {
    String baseName = metadata.filename.getRootRelativePath().getBaseName();

    this.containsErrors |= builder.containsErrors;
    if (metadata.directLoads == null && metadata.transitiveLoads == null) {
      Preconditions.checkState(containsErrors, "Loads not set for error-free package");
      builder.setLoads(ImmutableList.of());
    }

    if (isWorkspaceFile(baseName) || isModuleDotBazelFile(baseName)) {
      Preconditions.checkState(isRepoRulePackage());
      this.metadata.sourceRoot = Optional.empty();
    } else {
      Root sourceRoot =
          getSourceRoot(metadata.filename, metadata.packageIdentifier.getSourceRoot());
      if (sourceRoot.asPath() == null
          || !sourceRoot
              .getRelative(metadata.packageIdentifier.getSourceRoot())
              .equals(metadata.packageDirectory)) {
        throw new IllegalArgumentException(
            "Invalid BUILD file name for package '"
                + metadata.packageIdentifier
                + "': "
                + metadata.filename
                + " (in source "
                + sourceRoot
                + " with packageDirectory "
                + metadata.packageDirectory
                + " and package identifier source root "
                + metadata.packageIdentifier.getSourceRoot()
                + ")");
      }
      this.metadata.sourceRoot = Optional.of(sourceRoot);
    }

    this.metadata.makeEnv = ImmutableMap.copyOf(builder.makeEnv);
    this.targets = ImmutableSortedMap.copyOf(builder.targets);
    this.macros = ImmutableSortedMap.copyOf(builder.macros);
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
    setPackageOverhead(builder.packageOverheadEstimator.estimatePackageOverhead(this));
  }

  private static boolean isWorkspaceFile(String baseFileName) {
    return baseFileName.equals(LabelConstants.WORKSPACE_DOT_BAZEL_FILE_NAME.getPathString())
        || baseFileName.equals(LabelConstants.WORKSPACE_FILE_NAME.getPathString());
  }

  private static boolean isModuleDotBazelFile(String baseFileName) {
    return baseFileName.equals(LabelConstants.MODULE_DOT_BAZEL_FILE_NAME.getPathString());
  }

  public Metadata getMetadata() {
    return metadata;
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
    // TODO(bazel-team): Seems like a code smell that Metadata fields are being mutated here,
    // possibly after package construction is complete.
    return metadata.transitiveLoads != null
        ? metadata.transitiveLoads
        : computeTransitiveLoads(metadata.directLoads);
  }

  /**
   * Counts the number Starlark files transitively loaded by this package.
   *
   * <p>If transitive loads are not {@linkplain PackageSettings#precomputeTransitiveLoads
   * precomputed}, performs a traversal over the load graph to count them.
   */
  public int countTransitivelyLoadedStarlarkFiles() {
    if (metadata.transitiveLoads != null) {
      return metadata.transitiveLoads.size();
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
    if (metadata.transitiveLoads != null) {
      for (Label load : metadata.transitiveLoads) {
        visitor.visit(load);
      }
    } else {
      BazelModuleContext.visitLoadGraphRecursively(metadata.directLoads, visitor);
    }
  }

  private static ImmutableList<Label> computeTransitiveLoads(Iterable<Module> directLoads) {
    Set<Label> loads = new LinkedHashSet<>();
    BazelModuleContext.visitLoadGraphRecursively(directLoads, loads::add);
    return ImmutableList.copyOf(loads);
  }

  /**
   * Returns the filename of the BUILD file which defines this package. The parent directory of the
   * BUILD file is the package directory.
   */
  public RootedPath getFilename() {
    return metadata.filename;
  }

  /** Returns the directory containing the package's BUILD file. */
  public Path getPackageDirectory() {
    return metadata.packageDirectory;
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
    return metadata.getNameFragment();
  }

  /** Returns all make variables for a given platform. */
  public ImmutableMap<String, String> getMakeEnvironment() {
    return metadata.makeEnv;
  }

  /**
   * Returns the label of this package's BUILD file.
   *
   * <p>Typically <code>getBuildFileLabel().getName().equals("BUILD")</code> -- though not
   * necessarily: data in a subdirectory of a test package may use a different filename to avoid
   * inadvertently creating a new package.
   */
  public Label getBuildFileLabel() {
    return metadata.buildFile.getLabel();
  }

  /** Returns the InputFile target for this package's BUILD file. */
  public InputFile getBuildFile() {
    return metadata.buildFile;
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

  // TODO(bazel-team): Seems like we shouldn't permit this mutation on an already-initialized
  // Package. Is it possible for this to be called today after initialization?
  void setContainsErrors() {
    containsErrors = true;
  }

  /**
   * Returns the first {@link FailureDetail} describing one of the package's errors, or {@code null}
   * if it has no errors or all its errors lack details.
   */
  @Nullable
  public FailureDetail getFailureDetail() {
    return failureDetail;
  }

  /**
   * Returns a {@link FailureDetail} attributing a target error to the package's {@link
   * FailureDetail}, or a generic {@link Code#TARGET_MISSING} failure detail if the package has
   * none.
   *
   * <p>May only be called when {@link #containsErrors()} is true and with a target whose package is
   * this one.
   */
  public FailureDetail contextualizeFailureDetailForTarget(Target target) {
    Preconditions.checkState(
        target.getPackage().metadata.packageIdentifier.equals(metadata.packageIdentifier),
        "contextualizeFailureDetailForTarget called for target not in package. target=%s,"
            + " package=%s",
        target,
        this);
    Preconditions.checkState(
        containsErrors,
        "contextualizeFailureDetailForTarget called for package not in error. target=%s",
        target);
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

  /** Returns this package's workspace name. */
  public String getWorkspaceName() {
    return metadata.workspaceName;
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
      label = Label.create(metadata.packageIdentifier, targetName);
    } catch (LabelSyntaxException e) {
      throw new IllegalArgumentException(targetName, e);
    }

    if (metadata.succinctTargetNotFoundErrors) {
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
              metadata.filename.asPath().getPathString(),
              alternateTargetSuggestion));
    }
  }

  private String getAlternateTargetSuggestion(String targetName) {
    // If there's a file on the disk that's not mentioned in the BUILD file,
    // produce a more informative error.  NOTE! this code path is only executed
    // on failure, which is (relatively) very rare.  In the common case no
    // stat(2) is executed.
    Path filename = metadata.packageDirectory.getRelative(targetName);
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

  /** Returns all symbolic macros defined in the package. */
  // TODO(#19922): Clarify this comment to indicate whether the macros have already been expanded
  // by the point the Package has been built. The answer's probably "yes". In that case, this
  // accessor is still useful for introspecting e.g. by `bazel query`.
  public ImmutableMap<String, MacroInstance> getMacros() {
    return macros;
  }

  /**
   * How to enforce visibility on <code>config_setting</code> See {@link
   * ConfigSettingVisibilityPolicy} for details.
   */
  public ConfigSettingVisibilityPolicy getConfigSettingVisibilityPolicy() {
    return metadata.configSettingVisibilityPolicy;
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
    out.println("  Package " + getName() + " (" + metadata.filename.asPath() + ")");

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
    return new Builder(
        SymbolGenerator.create(id),
        packageSettings,
        id,
        filename,
        workspaceName,
        associatedModuleName,
        associatedModuleVersion,
        noImplicitFileExport,
        repositoryMapping,
        mainRepositoryMapping,
        cpuBoundSemaphore,
        packageOverheadEstimator,
        generatorMap,
        configSettingVisibilityPolicy,
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
        // The SymbolGenerator is based on workspaceFileKey rather than a package id or path,
        // in order to distinguish different chunks of the same WORKSPACE file.
        SymbolGenerator.create(workspaceFileKey),
        packageSettings,
        LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER,
        /* filename= */ workspaceFileKey.getPath(),
        workspaceName,
        /* associatedModuleName= */ Optional.empty(),
        /* associatedModuleVersion= */ Optional.empty(),
        noImplicitFileExport,
        /* repositoryMapping= */ mainRepoMapping,
        /* mainRepositoryMapping= */ mainRepoMapping,
        /* cpuBoundSemaphore= */ null,
        packageOverheadEstimator,
        /* generatorMap= */ null,
        /* configSettingVisibilityPolicy= */ null,
        /* globber= */ null);
  }

  public static Builder newExternalPackageBuilderForBzlmod(
      RootedPath moduleFilePath,
      boolean noImplicitFileExport,
      PackageIdentifier basePackageId,
      RepositoryMapping repoMapping) {
    return new Builder(
            SymbolGenerator.create(basePackageId),
            PackageSettings.DEFAULTS,
            basePackageId,
            /* filename= */ moduleFilePath,
            DUMMY_WORKSPACE_NAME_FOR_BZLMOD_PACKAGES,
            /* associatedModuleName= */ Optional.empty(),
            /* associatedModuleVersion= */ Optional.empty(),
            noImplicitFileExport,
            repoMapping,
            /* mainRepositoryMapping= */ null,
            /* cpuBoundSemaphore= */ null,
            PackageOverheadEstimator.NOOP_ESTIMATOR,
            /* generatorMap= */ null,
            /* configSettingVisibilityPolicy= */ null,
            /* globber= */ null)
        .setLoads(ImmutableList.of());
  }

  /**
   * Returns an error {@link Event} with {@link Location} and {@link DetailedExitCode} properties.
   */
  public static Event error(Location location, String message, Code code) {
    Event error = Event.error(location, message);
    return error.withProperty(DetailedExitCode.class, createDetailedCode(message, code));
  }

  private static DetailedExitCode createDetailedCode(String errorMessage, Code code) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(errorMessage)
            .setPackageLoading(PackageLoading.newBuilder().setCode(code))
            .build());
  }

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

    /**
     * The output instance for this builder. Needs to be instantiated and available with name info
     * throughout initialization. All other settings are applied during {@link #build}. See {@link
     * Package#Package} and {@link Package#finishInit} for details.
     */
    private final Package pkg;

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

    // All targets added to the package. We use SnapshottableBiMap to help track insertion order of
    // Rule targets, for use by native.existing_rules().
    private BiMap<String, Target> targets =
        new SnapshottableBiMap<>(target -> target instanceof Rule);

    // All instances of symbolic macros created during package construction.
    private final Map<String, MacroInstance> macros = new LinkedHashMap<>();

    /**
     * A stack of currently executing symbolic macros, outermost first.
     *
     * <p>Certain APIs are only available when this stack is empty (i.e. not in any symbolic macro).
     * See user documentation on {@code macro()} ({@link StarlarkRuleFunctionsApi#macro}).
     */
    private final List<MacroInstance> macroStack = new ArrayList<>();

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
    @Nullable private Map<Rule, List<Label>> ruleLabels = new HashMap<>();

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
        SymbolGenerator<?> symbolGenerator,
        PackageSettings packageSettings,
        PackageIdentifier id,
        RootedPath filename,
        String workspaceName,
        Optional<String> associatedModuleName,
        Optional<String> associatedModuleVersion,
        boolean noImplicitFileExport,
        RepositoryMapping repositoryMapping,
        @Nullable RepositoryMapping mainRepositoryMapping,
        @Nullable Semaphore cpuBoundSemaphore,
        PackageOverheadEstimator packageOverheadEstimator,
        @Nullable ImmutableMap<Location, String> generatorMap,
        // TODO(bazel-team): Config policy is an enum, what is null supposed to mean?
        // Maybe convert null -> LEGACY_OFF, assuming that's the correct default.
        @Nullable ConfigSettingVisibilityPolicy configSettingVisibilityPolicy,
        @Nullable Globber globber) {
      super(mainRepositoryMapping);
      this.symbolGenerator = symbolGenerator;

      Metadata metadata = new Metadata();
      metadata.packageIdentifier = Preconditions.checkNotNull(id);

      metadata.filename = filename;
      metadata.packageDirectory = filename.asPath().getParentDirectory();
      try {
        metadata.buildFileLabel = Label.create(id, filename.getRootRelativePath().getBaseName());
      } catch (LabelSyntaxException e) {
        // This can't actually happen.
        throw new AssertionError("Package BUILD file has an illegal name: " + filename, e);
      }

      metadata.workspaceName = Preconditions.checkNotNull(workspaceName);
      metadata.repositoryMapping = Preconditions.checkNotNull(repositoryMapping);
      metadata.associatedModuleName = Preconditions.checkNotNull(associatedModuleName);
      metadata.associatedModuleVersion = Preconditions.checkNotNull(associatedModuleVersion);
      metadata.succinctTargetNotFoundErrors = packageSettings.succinctTargetNotFoundErrors();
      metadata.configSettingVisibilityPolicy = configSettingVisibilityPolicy;

      this.pkg = new Package(metadata);

      this.precomputeTransitiveLoads = packageSettings.precomputeTransitiveLoads();
      this.noImplicitFileExport = noImplicitFileExport;
      this.labelConverter = new LabelConverter(id, repositoryMapping);
      if (pkg.getName().startsWith("javatests/")) {
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
              pkg, metadata.buildFileLabel, Location.fromFile(filename.asPath().toString())));
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
     * <p>If {@code allowSymbolicMacros} is false, this method also throws if we're currently
     * executing a symbolic macro implementation. (Legacy macros that are not called from within a
     * symbolic macro are fine.)
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
        boolean allowSymbolicMacros,
        boolean allowWorkspace)
        throws EvalException {
      Preconditions.checkArgument(allowBuild || allowSymbolicMacros || allowWorkspace);

      @Nullable StarlarkThreadContext ctx = thread.getThreadLocal(StarlarkThreadContext.class);
      boolean bad = false;
      if (ctx instanceof Builder builder) {
        bad |= !allowBuild && !builder.isRepoRulePackage();
        bad |= !allowSymbolicMacros && !builder.macroStack.isEmpty();
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
      if (allowSymbolicMacros && symbolicMacrosEnabled) {
        allowedUses.add("a symbolic macro");
      }
      if (allowWorkspace) {
        allowedUses.add("a WORKSPACE file");
      }
      throw Starlark.errorf(
          "%s can only be used while evaluating %s",
          what, StringUtil.joinEnglishList(allowedUses, "or"));
    }

    /** Convenience method for {@link #fromOrFail} that permits any context with a Builder. */
    @CanIgnoreReturnValue
    public static Builder fromOrFail(StarlarkThread thread, String what) throws EvalException {
      return fromOrFail(
          thread,
          what,
          /* allowBuild= */ true,
          /* allowSymbolicMacros= */ true,
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
          /* allowSymbolicMacros= */ false,
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
          /* allowSymbolicMacros= */ false,
          /* allowWorkspace= */ true);
    }

    /**
     * Convenience method for {@link #fromOrFail} that permits BUILD or WORKSPACE contexts (without
     * symbolic macros).
     */
    @CanIgnoreReturnValue
    public static Builder fromOrFailDisallowSymbolicMacros(StarlarkThread thread, String what)
        throws EvalException {
      return fromOrFail(
          thread,
          what,
          /* allowBuild= */ true,
          /* allowSymbolicMacros= */ false,
          /* allowWorkspace= */ true);
    }

    /** Convenience method for {@link #fromOrFail} that permits BUILD or symbolic macro contexts. */
    @CanIgnoreReturnValue
    public static Builder fromOrFailDisallowWorkspace(StarlarkThread thread, String what)
        throws EvalException {
      return fromOrFail(
          thread,
          what,
          /* allowBuild= */ true,
          /* allowSymbolicMacros= */ true,
          /* allowWorkspace= */ false);
    }

    PackageIdentifier getPackageIdentifier() {
      return pkg.getPackageIdentifier();
    }

    /**
     * Determine whether this package should contain build rules (returns {@code false}) or repo
     * rules (returns {@code true}).
     */
    public boolean isRepoRulePackage() {
      return pkg.isRepoRulePackage();
    }

    String getPackageWorkspaceName() {
      return pkg.getWorkspaceName();
    }

    /**
     * Returns the name of the Bzlmod module associated with the repo this package is in. If this
     * package is not from a Bzlmod repo, this is empty. For repos generated by module extensions,
     * this is the name of the module hosting the extension.
     */
    Optional<String> getAssociatedModuleName() {
      return pkg.metadata.associatedModuleName;
    }

    /**
     * Returns the version of the Bzlmod module associated with the repo this package is in. If this
     * package is not from a Bzlmod repo, this is empty. For repos generated by module extensions,
     * this is the version of the module hosting the extension.
     */
    Optional<String> getAssociatedModuleVersion() {
      return pkg.metadata.associatedModuleVersion;
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
      return pkg.metadata.buildFileLabel;
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
      return pkg.metadata.filename;
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
      pkg.metadata.packageArgs = pkg.metadata.packageArgs.mergeWith(packageArgs);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder mergePackageArgsFrom(PackageArgs.Builder builder) {
      return mergePackageArgsFrom(builder.build());
    }

    /** Called partial b/c in builder and thus subject to mutation and updates */
    public PackageArgs getPartialPackageArgs() {
      return pkg.metadata.packageArgs;
    }

    /** Uses the workspace name from {@code //external} to set this package's workspace name. */
    // TODO(#19922): Seems like this is only used for WORKSPACE logic (`workspace()` callable), but
    // it clashes with the notion that, for BUILD files, the workspace name should be supplied to
    // the Builder constructor and not mutated during evaluation. Either put up with this until we
    // delete WORKSPACE logic, or else separate the `workspace()` callable's mutation from this
    // metadata field.
    @CanIgnoreReturnValue
    @VisibleForTesting
    public Builder setWorkspaceName(String workspaceName) {
      pkg.metadata.workspaceName = workspaceName;
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
        pkg.metadata.transitiveLoads = computeTransitiveLoads(directLoads);
      } else {
        pkg.metadata.directLoads = ImmutableList.copyOf(directLoads);
      }
      return this;
    }

    @CanIgnoreReturnValue
    Builder setTransitiveLoadsForDeserialization(ImmutableList<Label> transitiveLoads) {
      checkLoadsNotSet();
      pkg.metadata.transitiveLoads = Preconditions.checkNotNull(transitiveLoads);
      return this;
    }

    private void checkLoadsNotSet() {
      Preconditions.checkState(
          pkg.metadata.directLoads == null,
          "Direct loads already set: %s",
          pkg.metadata.directLoads);
      Preconditions.checkState(
          pkg.metadata.transitiveLoads == null,
          "Transitive loads already set: %s",
          pkg.metadata.transitiveLoads);
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
      Target oldTarget = targets.put(newTarget.getName(), newTarget);
      if (newTarget instanceof Rule) {
        List<Label> ruleLabelsForOldTarget = ruleLabels.remove(oldTarget);
        if (ruleLabelsForOldTarget != null) {
          ruleLabels.put((Rule) newTarget, ruleLabelsForOldTarget);
        }
      }
    }

    public Set<Target> getTargets() {
      return Package.getTargets(targets);
    }

    /**
     * Returns a lightweight snapshot view of the names of all rule targets belonging to this
     * package at the time of this call.
     *
     * @throws IllegalStateException if this method is called after {@link #beforeBuild} has been
     *     called.
     */
    Map<String, Rule> getRulesSnapshotView() {
      if (targets instanceof SnapshottableBiMap<?, ?>) {
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
            "FileTarget in package " + pkg.getName() + " has illegal name: " + targetName, e);
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
    void setVisibilityAndLicense(InputFile inputFile, RuleVisibility visibility, License license) {
      String filename = inputFile.getName();
      Target cacheInstance = targets.get(filename);
      if (!(cacheInstance instanceof InputFile)) {
        throw new IllegalArgumentException(
            "Can't set visibility for nonexistent FileTarget "
                + filename
                + " in package "
                + pkg.getName()
                + ".");
      }
      if (!((InputFile) cacheInstance).isVisibilitySpecified()
          || cacheInstance.getVisibility() != visibility
          || !Objects.equals(cacheInstance.getLicense(), license)) {
        targets.put(
            filename,
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
      return Label.create(pkg.getPackageIdentifier(), targetName);
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
      targets.put(group.getName(), group);

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
      Preconditions.checkState(macroStack.isEmpty());

      if (hasDuplicateLabels(environments, name, "environments", location, eventHandler)
          || hasDuplicateLabels(defaults, name, "defaults", location, eventHandler)) {
        setContainsErrors();
        return;
      }

      EnvironmentGroup group =
          new EnvironmentGroup(createLabel(name), pkg, environments, defaults, location);
      checkTargetName(group);
      targets.put(group.getName(), group);

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
        targets.put(outputFile.getName(), outputFile);
      }
      targets.put(rule.getName(), rule);
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
    }

    /** Adds a symbolic macro instance to the package. */
    public void addMacro(MacroInstance macro) throws NameConflictException {
      checkMacroName(macro);
      macros.put(macro.getName(), macro);
    }

    /** Pushes a macro instance onto the stack of currently executing symbolic macros. */
    public void pushMacro(MacroInstance macro) {
      macroStack.add(macro);
    }

    /** Pops the stack of currently executing symbolic macros. */
    public MacroInstance popMacro() {
      return macroStack.remove(macroStack.size() - 1);
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

    @CanIgnoreReturnValue
    private Builder beforeBuild(boolean discoverAssumedInputFiles) throws NoSuchPackageException {
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
      }

      // We create an InputFile corresponding to the BUILD file in Builder's constructor. However,
      // the visibility of this target may be overridden with an exports_files directive, so we wait
      // until now to obtain the current instance from the targets map.
      pkg.metadata.buildFile =
          (InputFile)
              Preconditions.checkNotNull(targets.get(pkg.metadata.buildFileLabel.getName()));

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
          // All labels mentioned by a rule that refer to an unknown target in the current package
          // are assumed to be InputFiles, so let's create them. We add them to a temporary map
          // to avoid concurrent modification to this.targets while iterating (via getRules()).
          List<Label> labels = (ruleLabels != null) ? ruleLabels.get(rule) : rule.getLabels();
          for (Label label : labels) {
            if (label.getPackageIdentifier().equals(pkg.getPackageIdentifier())
                && !targets.containsKey(label.getName())
                // The existence of a macro by the same name blocks implicit creation of an input
                // file. This is because we plan on allowing macros to be passed as inputs to other
                // macros, and don't want this usage to be implicitly conflated with an unrelated
                // input file by the same name (e.g., if the macro's label makes its way into a
                // target definition by mistake, we want that to be treated as an unknown target
                // rather than a missing input file).
                // TODO(#19922): Update this comment when said behavior is implemented.
                && !macros.containsKey(label.getName())
                && !newInputFiles.containsKey(label.getName())) {
              Location loc = rule.getLocation();
              newInputFiles.put(
                  label.getName(),
                  noImplicitFileExport
                      ? new PrivateVisibilityInputFile(pkg, label, loc)
                      : new InputFile(pkg, label, loc));
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

    public Package build() throws NoSuchPackageException {
      return build(/* discoverAssumedInputFiles= */ true);
    }

    /**
     * Build the package, optionally adding any labels in the package not already associated with a
     * target as an input file.
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
      Target prev = targets.put(inputFile.getLabel().getName(), inputFile);
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
     * Throws {@link NameConflictException} if the given name of a declared object inside a symbolic
     * macro (i.e., a target or a submacro) does not follow the required prefix-based naming
     * convention.
     *
     * <p>A macro "foo" may define targets and submacros that have the name "foo" (the macro's "main
     * target") or "foo_BAR" where BAR is a non-empty string. The macro may not define the name
     * "foo_", or names that do not have "foo" as a prefix.
     */
    private void checkDeclaredNameValidForMacro(
        String what, String declaredName, String enclosingMacroName) throws NameConflictException {
      if (declaredName.equals(enclosingMacroName)) {
        return;
      } else if (declaredName.startsWith(enclosingMacroName)) {
        String suffix = declaredName.substring(enclosingMacroName.length());
        // 0-length suffix handled above.
        if (suffix.length() > 2 && suffix.startsWith("_")) {
          return;
        }
      }

      throw new NameConflictException(
          String.format(
              """
              macro '%s' cannot declare %s named '%s'. Name must be the same as the \
              macro's name or a suffix of the macro's name plus '_'.""",
              enclosingMacroName, what, declaredName));
    }

    /**
     * Throws {@link NameConflictException} if the given target's name can't be added, either
     * because of a conflict or because of a violation of symbolic macro naming rules (if
     * applicable).
     */
    private void checkTargetName(Target added) throws NameConflictException {
      checkForExistingTargetName(added);

      if (!macroStack.isEmpty()) {
        String enclosingMacroName = Iterables.getLast(macroStack).getName();
        checkDeclaredNameValidForMacro("target", added.getName(), enclosingMacroName);
      }
    }

    /**
     * Throws {@link NameConflictException} if the given target's name matches that of an existing
     * target in the package.
     */
    private void checkForExistingTargetName(Target added) throws NameConflictException {
      Target existing = targets.get(added.getName());
      if (existing == null) {
        return;
      }

      String subject = String.format("%s '%s'", added.getTargetKind(), added.getName());
      if (added instanceof OutputFile addedOutput) {
        subject += String.format(" in rule '%s'", addedOutput.getGeneratingRule().getName());
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
     */
    private void checkMacroName(MacroInstance added) throws NameConflictException {
      checkForExistingMacroName(added);

      if (!macroStack.isEmpty()) {
        String enclosingMacroName = Iterables.getLast(macroStack).getName();
        checkDeclaredNameValidForMacro("submacro", added.getName(), enclosingMacroName);
      }
    }

    /**
     * Throws {@link NameConflictException} if the given macro's name matches that of an existing
     * macro in the package.
     */
    private void checkForExistingMacroName(MacroInstance added) throws NameConflictException {
      MacroInstance existing = macros.get(added.getName());
      if (existing == null) {
        return;
      }

      // TODO(#19922): Add definition location info for the existing object, like we have in the
      // case for rules.
      throw new NameConflictException(
          String.format("macro '%s' conflicts with existing macro", added.getName()));
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

  /**
   * A collection of data about a package that does not require evaluating the whole package.
   *
   * <p>In particular, this does not contain any target information. It does contain data known
   * prior to BUILD file evaluation, data mutated by BUILD file evaluation, and data computed
   * immediately after BUILD file evaluation.
   *
   * <p>This object is supplied to symbolic macro expansion.
   */
  public static final class Metadata {

    private Metadata() {}

    // Fields that are known before the beginning of BUILD file execution

    private PackageIdentifier packageIdentifier;

    /**
     * Returns the package identifier for this package.
     *
     * <p>This is a suffix of {@code getFilename().getParentDirectory()}.
     */
    public PackageIdentifier getPackageIdentifier() {
      return packageIdentifier;
    }

    /**
     * Returns the name of this package. If this build is using external repositories then this name
     * may not be unique!
     */
    public String getName() {
      return packageIdentifier.getPackageFragment().getPathString();
    }

    /** Like {@link #getName}, but has type {@code PathFragment}. */
    public PathFragment getNameFragment() {
      return packageIdentifier.getPackageFragment();
    }

    private RootedPath filename;

    /** Returns the filename of this package's BUILD file. */
    public RootedPath getFilename() {
      return filename;
    }

    private Path packageDirectory;

    /**
     * Returns the directory in which this package's BUILD file resides. All InputFile members of
     * the packages are located relative to this directory.
     */
    public Path getPackageDirectory() {
      return packageDirectory;
    }

    private Label buildFileLabel;

    /** Returns the label of this package's BUILD file. */
    public Label getBuildFileLabel() {
      return buildFileLabel;
    }

    private String workspaceName;

    /**
     * Returns the name of the workspace this package is in. Used as a prefix for the runfiles
     * directory. This can be set in the WORKSPACE file. This must be a valid target name.
     */
    public String getWorkspaceName() {
      return workspaceName;
    }

    private RepositoryMapping repositoryMapping;

    /**
     * Returns the map of repository reassignments for BUILD packages. This will be empty for
     * packages within the main workspace.
     */
    public RepositoryMapping getRepositoryMapping() {
      return repositoryMapping;
    }

    /**
     * The name of the Bzlmod module associated with the repo this package is in. If this package is
     * not from a Bzlmod repo, this is empty. For repos generated by module extensions, this is the
     * name of the module hosting the extension.
     */
    private Optional<String> associatedModuleName;

    /**
     * The version of the Bzlmod module associated with the repo this package is in. If this package
     * is not from a Bzlmod repo, this is empty. For repos generated by module extensions, this is
     * the version of the module hosting the extension.
     */
    private Optional<String> associatedModuleVersion;

    private ConfigSettingVisibilityPolicy configSettingVisibilityPolicy;

    /** Returns the visibility enforcement policy for {@code config_setting}. */
    public ConfigSettingVisibilityPolicy getConfigSettingVisibilityPolicy() {
      return configSettingVisibilityPolicy;
    }

    // These two fields are mutually exclusive. Which one is set depends on
    // PackageSettings#precomputeTransitiveLoads.
    @Nullable private ImmutableList<Module> directLoads;
    @Nullable private ImmutableList<Label> transitiveLoads;

    /** Governs the error message behavior of {@link Package#getTarget}. */
    // TODO(bazel-team): Arguably, this could be replaced by a boolean param to getTarget(), or some
    // separate action taken by the caller. But there's a lot of call sites that would need
    // updating.
    private boolean succinctTargetNotFoundErrors;

    // Fields that are updated during BUILD file execution.

    private PackageArgs packageArgs = PackageArgs.DEFAULT;

    /**
     * Returns the collection of package-level attributes set by the {@code package()} callable and
     * similar methods. May be modified during BUILD file execution.
     */
    public PackageArgs getPackageArgs() {
      return packageArgs;
    }

    // Fields that are only set after BUILD file execution (but before symbolic macro expansion).

    private InputFile buildFile;

    /**
     * Returns the InputFile target corresponding to this package's BUILD file.
     *
     * <p>This may change during BUILD file execution as a result of exports_files changing the
     * BUILD file's visibility.
     */
    public InputFile getBuildFile() {
      return buildFile;
    }

    private Optional<Root> sourceRoot;

    /**
     * Returns the root of the source tree in which this package was found. It is an invariant that
     * {@code sourceRoot.getRelative(packageId.getSourceRoot()).equals(packageDirectory)}. Returns
     * {@link Optional#empty} if this {@link Package} is derived from a WORKSPACE file.
     */
    public Optional<Root> getSourceRoot() {
      return sourceRoot;
    }

    private ImmutableMap<String, String> makeEnv;

    /**
     * Returns the "Make" environment of this package, containing package-local definitions of
     * "Make" variables.
     */
    public ImmutableMap<String, String> getMakeEnvironment() {
      return makeEnv;
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
