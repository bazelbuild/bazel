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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Interner;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.collect.ImmutableSortedKeyMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.packages.RuleClass.Builder.ThirdPartyLicenseExistencePolicy;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.syntax.Location;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.starlark.spelling.SpellChecker;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.TreeMap;
import javax.annotation.Nullable;

/**
 * A package, which is a container of {@link Rule}s, each of which contains a dictionary of named
 * attributes.
 *
 * <p>Package instances are intended to be immutable and for all practical purposes can be treated
 * as such. Note, however, that some member variables exposed via the public interface are not
 * strictly immutable, so until their types are guaranteed immutable we're not applying the
 * {@code @Immutable} annotation here.
 *
 * <p>When changing this class, make sure to make corresponding changes to serialization!
 */
@SuppressWarnings("JavaLangClash")
public class Package {
  /**
   * Common superclass for all name-conflict exceptions.
   */
  public static class NameConflictException extends Exception {
    private NameConflictException(String message) {
      super(message);
    }
  }

  /**
   * The repository identifier for this package.
   */
  private final PackageIdentifier packageIdentifier;

  /** The filename of this package's BUILD file. */
  private RootedPath filename;

  /**
   * The directory in which this package's BUILD file resides.  All InputFile
   * members of the packages are located relative to this directory.
   */
  private Path packageDirectory;

  /**
   * The name of the workspace this package is in. Used as a prefix for the runfiles directory.
   * This can be set in the WORKSPACE file. This must be a valid target name.
   */
  private String workspaceName;

  /**
   * The root of the source tree in which this package was found. It is an invariant that {@code
   * sourceRoot.getRelative(packageId.getSourceRoot()).equals(packageDirectory)}. Returns {@link
   * Optional#empty} if this {@link Package} is derived from a WORKSPACE file.
   */
  private Optional<Root> sourceRoot;

  /**
   * The "Make" environment of this package, containing package-local
   * definitions of "Make" variables.
   */
  private ImmutableMap<String, String> makeEnv;

  /** The collection of all targets defined in this package, indexed by name. */
  private ImmutableSortedKeyMap<String, Target> targets;

  /**
   * Default visibility for rules that do not specify it.
   */
  private RuleVisibility defaultVisibility;
  private boolean defaultVisibilitySet;

  /**
   * Default package-level 'testonly' value for rules that do not specify it.
   */
  private boolean defaultTestOnly = false;

  /**
   * Default package-level 'deprecation' value for rules that do not specify it.
   */
  private String defaultDeprecation;

  /**
   * Default header strictness checking for rules that do not specify it.
   */
  private String defaultHdrsCheck;

  /** Default copts for cc_* rules. The rules' individual copts will append to this value. */
  private ImmutableList<String> defaultCopts;

  /**
   * The InputFile target corresponding to this package's BUILD file.
   */
  private InputFile buildFile;

  /**
   * True iff this package's BUILD files contained lexical or grammatical
   * errors, or experienced errors during evaluation, or semantic errors during
   * the construction of any rule.
   *
   * <p>Note: A package containing errors does not necessarily prevent a build;
   * if all the rules needed for a given build were constructed prior to the
   * first error, the build may proceed.
   */
  private boolean containsErrors;

  /** The list of transitive closure of the Starlark file dependencies. */
  private ImmutableList<Label> starlarkFileDependencies;

  /** The package's default "applicable_licenses" attribute. */
  private Set<Label> defaultApplicableLicenses = ImmutableSet.of();

  /**
   * The package's default "licenses" and "distribs" attributes, as specified
   * in calls to licenses() and distribs() in the BUILD file.
   */
  // These sets contain the values specified by the most recent licenses() or
  // distribs() declarations encountered during package parsing:
  private License defaultLicense;
  private Set<License.DistributionType> defaultDistributionSet;

  /**
   * The map from each repository to that repository's remappings map.
   * This is only used in the //external package, it is an empty map for all other packages.
   * For example, an entry of {"@foo" : {"@x", "@y"}} indicates that, within repository foo,
   * "@x" should be remapped to "@y".
   */
  private ImmutableMap<RepositoryName, ImmutableMap<RepositoryName, RepositoryName>>
      externalPackageRepositoryMappings;

  /**
   * The map of repository reassignments for BUILD packages. This will be empty for packages
   * within the main workspace.
   */
  private ImmutableMap<RepositoryName, RepositoryName> repositoryMapping;

  private Set<Label> defaultCompatibleWith = ImmutableSet.of();
  private Set<Label> defaultRestrictedTo = ImmutableSet.of();

  private ImmutableSet<String> features;

  private ImmutableList<Event> events;
  private ImmutableList<Postable> posts;

  private ImmutableList<String> registeredExecutionPlatforms;
  private ImmutableList<String> registeredToolchains;

  private long computationSteps;

  /** Returns the number of Starlark computation steps executed by this BUILD file. */
  public long getComputationSteps() {
    return computationSteps;
  }

  /**
   * Package initialization, part 1 of 3: instantiates a new package with the
   * given name.
   *
   * <p>As part of initialization, {@link Builder} constructs {@link InputFile}
   * and {@link PackageGroup} instances that require a valid Package instance where
   * {@link Package#getNameFragment()} is accessible. That's why these settings are
   * applied here at the start.
   *
   * @precondition {@code name} must be a suffix of
   * {@code filename.getParentDirectory())}.
   */
  protected Package(PackageIdentifier packageId, String runfilesPrefix) {
    this.packageIdentifier = packageId;
    this.workspaceName = runfilesPrefix;
  }

  /** Returns this packages' identifier. */
  public PackageIdentifier getPackageIdentifier() {
    return packageIdentifier;
  }

  /**
   * Returns the repository mapping for the requested external repository.
   *
   * @throws UnsupportedOperationException if called from a package other than
   *     the //external package
   */
  public ImmutableMap<RepositoryName, RepositoryName> getRepositoryMapping(
      RepositoryName repository) {
    if (!packageIdentifier.equals(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER)) {
      throw new UnsupportedOperationException("Can only access the external package repository"
          + "mappings from the //external package");
    }

    // We are passed a repository name as seen from the main repository, not necessarily
    // a canonical repository name. So, we first have to find the canonical name for the
    // repository in question before we can look up the mapping for it.
    RepositoryName actualRepositoryName =
        externalPackageRepositoryMappings
            .getOrDefault(RepositoryName.MAIN, ImmutableMap.of())
            .getOrDefault(repository, repository);

    return externalPackageRepositoryMappings.getOrDefault(actualRepositoryName, ImmutableMap.of());
  }

  /** Get the repository mapping for this package. */
  public ImmutableMap<RepositoryName, RepositoryName> getRepositoryMapping() {
    return repositoryMapping;
  }

  /**
   * Returns the full map of repository mappings collected so far.
   *
   * @throws UnsupportedOperationException if called from a package other than the //external
   *     package
   */
  public ImmutableMap<RepositoryName, ImmutableMap<RepositoryName, RepositoryName>>
      getExternalPackageRepositoryMappings() {
    if (!packageIdentifier.equals(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER)) {
      throw new UnsupportedOperationException(
          "Can only access the external package repository"
              + "mappings from the //external package");
    }
    return this.externalPackageRepositoryMappings;
  }

  /**
   * Package initialization: part 2 of 3: sets this package's default header
   * strictness checking.
   *
   * <p>This is needed to support C++-related rule classes
   * which accesses {@link #getDefaultHdrsCheck} from the still-under-construction
   * package.
   */
  private void setDefaultHdrsCheck(String defaultHdrsCheck) {
    this.defaultHdrsCheck = defaultHdrsCheck;
  }

  /**
   * Set the default 'testonly' value for this package.
   */
  private void setDefaultTestOnly(boolean testOnly) {
    defaultTestOnly = testOnly;
  }

  /**
   * Set the default 'deprecation' value for this package.
   */
  private void setDefaultDeprecation(String deprecation) {
    defaultDeprecation = deprecation;
  }

  /**
   * Sets the default 'applicable_licenses" value for this package attribute when not explicitly
   * specified by the rule.
   */
  private void setDefaultApplicableLicenses(Set<Label> licenses) {
    defaultApplicableLicenses = licenses;
  }

  /**
   * Sets the default value to use for a rule's {@link RuleClass#COMPATIBLE_ENVIRONMENT_ATTR}
   * attribute when not explicitly specified by the rule.
   */
  private void setDefaultCompatibleWith(Set<Label> environments) {
    defaultCompatibleWith = environments;
  }

  /**
   * Sets the default value to use for a rule's {@link RuleClass#RESTRICTED_ENVIRONMENT_ATTR}
   * attribute when not explicitly specified by the rule.
   */
  private void setDefaultRestrictedTo(Set<Label> environments) {
    defaultRestrictedTo = environments;
  }

  /**
   * Returns the source root (a directory) beneath which this package's BUILD file was found, or
   * {@link Optional#empty} if this package was derived from a workspace file.
   *
   * <p>Assumes invariant: If non-empty, {@code
   * getSourceRoot().get().getRelative(packageId.getSourceRoot()).equals(getPackageDirectory())}
   */
  public Optional<Root> getSourceRoot() {
    return sourceRoot;
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
   * Package initialization: part 3 of 3: applies all other settings and completes
   * initialization of the package.
   *
   * <p>Only after this method is called can this package be considered "complete"
   * and be shared publicly.
   */
  private void finishInit(Builder builder) {
    // If any error occurred during evaluation of this package, consider all
    // rules in the package to be "in error" also (even if they were evaluated
    // prior to the error).  This behaviour is arguably stricter than need be,
    // but stopping a build only for some errors but not others creates user
    // confusion.
    if (builder.containsErrors) {
      for (Rule rule : builder.getTargets(Rule.class)) {
        rule.setContainsErrors();
      }
    }
    this.filename = builder.getFilename();
    this.packageDirectory = filename.asPath().getParentDirectory();
    String baseName = filename.getRootRelativePath().getBaseName();

    if (isWorkspaceFile(baseName)) {
      Preconditions.checkState(
          packageIdentifier.equals(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER));
      this.sourceRoot = Optional.empty();
    } else {
      Root sourceRoot = getSourceRoot(filename, packageIdentifier.getSourceRoot());
      if (sourceRoot.asPath() == null
          || !sourceRoot.getRelative(packageIdentifier.getSourceRoot()).equals(packageDirectory)) {
        throw new IllegalArgumentException(
            "Invalid BUILD file name for package '"
                + packageIdentifier
                + "': "
                + filename
                + " (in source "
                + sourceRoot
                + " with packageDirectory "
                + packageDirectory
                + " and package identifier source root "
                + packageIdentifier.getSourceRoot()
                + ")");
      }
      this.sourceRoot = Optional.of(sourceRoot);
    }

    this.makeEnv = ImmutableMap.copyOf(builder.makeEnv);
    this.targets = ImmutableSortedKeyMap.copyOf(builder.targets);
    this.defaultVisibility = builder.defaultVisibility;
    this.defaultVisibilitySet = builder.defaultVisibilitySet;
    if (builder.defaultCopts == null) {
      this.defaultCopts = ImmutableList.of();
    } else {
      this.defaultCopts = ImmutableList.copyOf(builder.defaultCopts);
    }
    this.buildFile = builder.buildFile;
    this.containsErrors = builder.containsErrors;
    this.starlarkFileDependencies = builder.starlarkFileDependencies;
    this.defaultLicense = builder.defaultLicense;
    this.defaultDistributionSet = builder.defaultDistributionSet;
    this.features = ImmutableSortedSet.copyOf(builder.features);
    this.events = ImmutableList.copyOf(builder.events);
    this.posts = ImmutableList.copyOf(builder.posts);
    this.registeredExecutionPlatforms = ImmutableList.copyOf(builder.registeredExecutionPlatforms);
    this.registeredToolchains = ImmutableList.copyOf(builder.registeredToolchains);
    this.repositoryMapping = Preconditions.checkNotNull(builder.repositoryMapping);
    ImmutableMap.Builder<RepositoryName, ImmutableMap<RepositoryName, RepositoryName>>
        repositoryMappingsBuilder = ImmutableMap.builder();
    if (!builder.externalPackageRepositoryMappings.isEmpty() && !builder.isWorkspace()) {
      // 'repo_mapping' should only be used in the //external package, i.e. should only appear
      // in WORKSPACE files. Currently, if someone tries to use 'repo_mapping' in a BUILD rule, they
      // will get a "no such attribute" error. This check is to protect against a 'repo_mapping'
      // attribute being added to a rule in the future.
      throw new IllegalArgumentException(
          "'repo_mapping' may only be used in the //external package");
    }
    builder.externalPackageRepositoryMappings.forEach((k, v) ->
        repositoryMappingsBuilder.put(k, ImmutableMap.copyOf(v)));
    this.externalPackageRepositoryMappings = repositoryMappingsBuilder.build();
  }

  private static boolean isWorkspaceFile(String baseFileName) {
    return baseFileName.equals(LabelConstants.WORKSPACE_DOT_BAZEL_FILE_NAME.getPathString())
        || baseFileName.equals(LabelConstants.WORKSPACE_FILE_NAME.getPathString());
  }

  /** Returns the list of transitive closure of the Starlark file dependencies of this package. */
  public ImmutableList<Label> getStarlarkFileDependencies() {
    return starlarkFileDependencies;
  }

  /**
   * Returns the filename of the BUILD file which defines this package. The parent directory of the
   * BUILD file is the package directory.
   */
  public RootedPath getFilename() {
    return filename;
  }

  /**
   * Returns the directory containing the package's BUILD file.
   */
  public Path getPackageDirectory() {
    return packageDirectory;
  }

  /**
   * Returns the name of this package. If this build is using external repositories then this name
   * may not be unique!
   */
  public String getName() {
    return packageIdentifier.getPackageFragment().getPathString();
  }

  /**
   * Like {@link #getName}, but has type {@code PathFragment}.
   */
  public PathFragment getNameFragment() {
    return packageIdentifier.getPackageFragment();
  }

  /**
   * Returns all make variables for a given platform.
   */
  public ImmutableMap<String, String> getMakeEnvironment() {
    return makeEnv;
  }

  /**
   * Returns the label of this package's BUILD file.
   *
   * <p> Typically <code>getBuildFileLabel().getName().equals("BUILD")</code> --
   * though not necessarily: data in a subdirectory of a test package may use a
   * different filename to avoid inadvertently creating a new package.
   */
  public Label getBuildFileLabel() {
    return buildFile.getLabel();
  }

  /**
   * Returns the InputFile target for this package's BUILD file.
   */
  public InputFile getBuildFile() {
    return buildFile;
  }

  /**
   * Returns true if errors were encountered during evaluation of this package.
   * (The package may be incomplete and its contents should not be relied upon
   * for critical operations. However, any Rules belonging to the package are
   * guaranteed to be intact, unless their <code>containsErrors()</code> flag
   * is set.)
   */
  public boolean containsErrors() {
    return containsErrors;
  }

  public List<Postable> getPosts() {
    return posts;
  }

  public List<Event> getEvents() {
    return events;
  }

  /** Returns an (immutable, ordered) view of all the targets belonging to this package. */
  public ImmutableSortedKeyMap<String, Target> getTargets() {
    return targets;
  }

  /** Common getTargets implementation, accessible by {@link Package.Builder}. */
  private static Set<Target> getTargets(BiMap<String, Target> targetMap) {
    return targetMap.values();
  }

  /**
   * Returns a (read-only, ordered) iterable of all the targets belonging
   * to this package which are instances of the specified class.
   */
  public <T extends Target> Iterable<T> getTargets(Class<T> targetClass) {
    return getTargets(targets, targetClass);
  }

  /**
   * Common getTargets implementation, accessible by both {@link Package} and
   * {@link Package.Builder}.
   */
  private static <T extends Target> Iterable<T> getTargets(Map<String, Target> targetMap,
      Class<T> targetClass) {
    return Iterables.filter(targetMap.values(), targetClass);
  }

  /**
   * Returns the rule that corresponds to a particular BUILD target name. Useful
   * for walking through the dependency graph of a target.
   * Fails if the target is not a Rule.
   */
  public Rule getRule(String targetName) {
    return (Rule) targets.get(targetName);
  }

  /**
   * Returns this package's workspace name.
   */
  public String getWorkspaceName() {
    return workspaceName;
  }

  /**
   * Returns the features specified in the <code>package()</code> declaration.
   */
  public ImmutableSet<String> getFeatures() {
    return features;
  }

  /**
   * Returns the target (a member of this package) whose name is "targetName".
   * First rules are searched, then output files, then input files.  The target
   * name must be valid, as defined by {@code LabelValidator#validateTargetName}.
   *
   * @throws NoSuchTargetException if the specified target was not found.
   */
  public Target getTarget(String targetName) throws NoSuchTargetException {
    Target target = targets.get(targetName);
    if (target != null) {
      return target;
    }

    // No such target.

    // If there's a file on the disk that's not mentioned in the BUILD file,
    // produce a more informative error.  NOTE! this code path is only executed
    // on failure, which is (relatively) very rare.  In the common case no
    // stat(2) is executed.
    Path filename = getPackageDirectory().getRelative(targetName);
    String suffix;
    if (!PathFragment.isNormalized(targetName) || "*".equals(targetName)) {
      // Don't check for file existence if the target name is not normalized
      // because the error message would be confusing and wrong. If the
      // targetName is "foo/bar/.", and there is a directory "foo/bar", it
      // doesn't mean that "//pkg:foo/bar/." is a valid label.
      // Also don't check if the target name is a single * character since
      // it's invalid on Windows.
      suffix = "";
    } else if (filename.isDirectory()) {
      suffix =
          "; however, a source directory of this name exists.  (Perhaps add "
              + "'exports_files([\""
              + targetName
              + "\"])' to "
              + getName()
              + "/BUILD, or define a "
              + "filegroup?)";
    } else if (filename.exists()) {
      suffix =
          "; however, a source file of this name exists.  (Perhaps add "
              + "'exports_files([\""
              + targetName
              + "\"])' to "
              + getName()
              + "/BUILD?)";
    } else {
      suffix = SpellChecker.didYouMean(targetName, targets.keySet());
    }

    throw makeNoSuchTargetException(targetName, suffix);
  }

  protected NoSuchTargetException makeNoSuchTargetException(String targetName, String suffix) {
    Label label;
    try {
      label = Label.create(packageIdentifier, targetName);
    } catch (LabelSyntaxException e) {
      throw new IllegalArgumentException(targetName);
    }
    String msg =
        String.format(
            "target '%s' not declared in package '%s'%s defined by %s",
            targetName, getName(), suffix, filename.asPath().getPathString());
    return new NoSuchTargetException(label, msg);
  }

  /**
   * Returns the default visibility for this package.
   */
  public RuleVisibility getDefaultVisibility() {
    return defaultVisibility;
  }

  /**
   * Returns the default testonly value.
   */
  public Boolean getDefaultTestOnly() {
    return defaultTestOnly;
  }

  /**
   * Returns the default deprecation value.
   */
  public String getDefaultDeprecation() {
    return defaultDeprecation;
  }

  /** Gets the default header checking mode. */
  public String getDefaultHdrsCheck() {
    return defaultHdrsCheck != null ? defaultHdrsCheck : "strict";
  }

  /**
   * Returns the default copts value, to which rules should append their
   * specific copts.
   */
  public ImmutableList<String> getDefaultCopts() {
    return defaultCopts;
  }

  /**
   * Returns whether the default header checking mode has been set or it is the
   * default value.
   */
  public boolean isDefaultHdrsCheckSet() {
    return defaultHdrsCheck != null;
  }

  public boolean isDefaultVisibilitySet() {
    return defaultVisibilitySet;
  }

  /** Gets the licenses list for the default applicable_licenses declared by this package. */
  public Set<Label> getDefaultApplicableLicenses() {
    return defaultApplicableLicenses;
  }

  /**
   * Gets the parsed license object for the default license
   * declared by this package.
   */
  public License getDefaultLicense() {
    return defaultLicense;
  }

  /**
   * Returns the parsed set of distributions declared as the default for this
   * package.
   */
  public Set<License.DistributionType> getDefaultDistribs() {
    return defaultDistributionSet;
  }

  /**
   * Returns the default value to use for a rule's {@link RuleClass#COMPATIBLE_ENVIRONMENT_ATTR}
   * attribute when not explicitly specified by the rule.
   */
  public Set<Label> getDefaultCompatibleWith() {
    return defaultCompatibleWith;
  }

  /**
   * Returns the default value to use for a rule's {@link RuleClass#RESTRICTED_ENVIRONMENT_ATTR}
   * attribute when not explicitly specified by the rule.
   */
  public Set<Label> getDefaultRestrictedTo() {
    return defaultRestrictedTo;
  }

  public ImmutableList<String> getRegisteredExecutionPlatforms() {
    return registeredExecutionPlatforms;
  }

  public ImmutableList<String> getRegisteredToolchains() {
    return registeredToolchains;
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
    out.println("  Package " + getName() + " (" + getFilename().asPath() + ")");

    // Rules:
    out.println("    Rules");
    for (Rule rule : getTargets(Rule.class)) {
      out.println("      " + rule.getTargetKind() + " " + rule.getLabel());
      for (Attribute attr : rule.getAttributes()) {
        for (Object possibleValue : AggregatingAttributeMapper.of(rule)
            .visitAttribute(attr.getName(), attr.getType())) {
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

  public static Builder newExternalPackageBuilder(
      Builder.Helper helper,
      RootedPath workspacePath,
      String runfilesPrefix,
      StarlarkSemantics starlarkSemantics) {
    return new Builder(
            helper,
            LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER,
            runfilesPrefix,
            starlarkSemantics.incompatibleNoImplicitFileExport(),
            Builder.EMPTY_REPOSITORY_MAPPING)
        .setFilename(workspacePath);
  }

  /**
   * A builder for {@link Package} objects. Only intended to be used by {@link PackageFactory} and
   * {@link com.google.devtools.build.lib.skyframe.PackageFunction}.
   */
  public static class Builder {

    public static final ImmutableMap<RepositoryName, RepositoryName> EMPTY_REPOSITORY_MAPPING =
        ImmutableMap.of();

    public interface Helper {
      /**
       * Returns a fresh {@link Package} instance that a {@link Builder} will internally mutate
       * during package loading. Called by {@link PackageFactory}.
       */
      Package createFreshPackage(PackageIdentifier packageId, String runfilesPrefix);
    }

    /** {@link Helper} that simply calls the {@link Package} constructor. */
    public static class DefaultHelper implements Helper {
      public static final DefaultHelper INSTANCE = new DefaultHelper();

      private DefaultHelper() {
      }

      @Override
      public Package createFreshPackage(PackageIdentifier packageId, String runfilesPrefix) {
        return new Package(packageId, runfilesPrefix);
      }
    }

    /**
     * The output instance for this builder. Needs to be instantiated and
     * available with name info throughout initialization. All other settings
     * are applied during {@link #build}. See {@link Package#Package}
     * and {@link Package#finishInit} for details.
     */
    private final Package pkg;

    private final boolean noImplicitFileExport;
    private final CallStack.Builder callStackBuilder = new CallStack.Builder();

    // The map from each repository to that repository's remappings map.
    // This is only used in the //external package, it is an empty map for all other packages.
    public final HashMap<RepositoryName, HashMap<RepositoryName, RepositoryName>>
        externalPackageRepositoryMappings = new HashMap<>();
    /**
     * The map of repository reassignments for BUILD packages loaded within external repositories.
     * It contains an entry from "@<main workspace name>" to "@" for packages within the main
     * workspace.
     */
    private final ImmutableMap<RepositoryName, RepositoryName> repositoryMapping;

    private RootedPath filename = null;
    private Label buildFileLabel = null;
    private InputFile buildFile = null;
    // TreeMap so that the iteration order of variables is predictable. This is useful so that the
    // serialized representation is deterministic.
    private TreeMap<String, String> makeEnv = new TreeMap<>();
    private RuleVisibility defaultVisibility = ConstantRuleVisibility.PRIVATE;
    private boolean defaultVisibilitySet;
    private List<String> defaultCopts = null;
    private List<String> features = new ArrayList<>();
    private List<Event> events = Lists.newArrayList();
    private List<Postable> posts = Lists.newArrayList();
    @Nullable private String ioExceptionMessage = null;
    @Nullable private IOException ioException = null;
    private boolean containsErrors = false;

    private ImmutableList<Label> defaultApplicableLicenses = ImmutableList.of();
    private License defaultLicense = License.NO_LICENSE;
    private Set<License.DistributionType> defaultDistributionSet = License.DEFAULT_DISTRIB;

    private BiMap<String, Target> targets = HashBiMap.create();
    private final Map<Label, EnvironmentGroup> environmentGroups = new HashMap<>();

    private ImmutableList<Label> starlarkFileDependencies = ImmutableList.of();

    private final List<String> registeredExecutionPlatforms = new ArrayList<>();
    private final List<String> registeredToolchains = new ArrayList<>();

    private ThirdPartyLicenseExistencePolicy thirdPartyLicenceExistencePolicy =
        ThirdPartyLicenseExistencePolicy.USER_CONTROLLABLE;

    /**
     * True iff the "package" function has already been called in this package.
     */
    private boolean packageFunctionUsed;

    /**
     * The collection of the prefixes of every output file. Maps every prefix
     * to an output file whose prefix it is.
     *
     * <p>This is needed to make the output file prefix conflict check be
     * reasonably fast. However, since it can potentially take a lot of memory and
     * is useless after the package has been loaded, it isn't passed to the
     * package itself.
     */
    private Map<String, OutputFile> outputFilePrefixes = new HashMap<>();

    private final Interner<ImmutableList<?>> listInterner = new ThreadCompatibleInterner<>();

    private final Map<Location, String> generatorNameByLocation = new HashMap<>();

    Map<Location, String> getGeneratorNameByLocation() {
      return generatorNameByLocation;
    }

    @ThreadCompatible
    private static class ThreadCompatibleInterner<T> implements Interner<T> {
      private final Map<T, T> interns = new HashMap<>();

      @Override
      public T intern(T sample) {
        T t = interns.get(sample);
        if (t != null) {
          return t;
        }
        interns.put(sample, sample);
        return sample;
      }
    }

    private boolean alreadyBuilt = false;

    Builder(
        Helper helper,
        PackageIdentifier id,
        String runfilesPrefix,
        boolean noImplicitFileExport,
        ImmutableMap<RepositoryName, RepositoryName> repositoryMapping) {
      this.pkg = helper.createFreshPackage(id, runfilesPrefix);
      this.noImplicitFileExport = noImplicitFileExport;
      this.repositoryMapping = repositoryMapping;
      if (pkg.getName().startsWith("javatests/")) {
        setDefaultTestonly(true);
      }
    }

    PackageIdentifier getPackageIdentifier() {
      return pkg.getPackageIdentifier();
    }

    /** Determine if we are in the WORKSPACE file or not */
    boolean isWorkspace() {
      return pkg.getPackageIdentifier().equals(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER);
    }

    String getPackageWorkspaceName() {
      return pkg.getWorkspaceName();
    }

    /**
     * Updates the externalPackageRepositoryMappings entry for {@code repoWithin}. Adds new
     * entry from {@code localName} to {@code mappedName} in {@code repoWithin}'s map.
     *
     * @param repoWithin the RepositoryName within which the mapping should apply
     * @param localName the RepositoryName that actually appears in the WORKSPACE and BUILD files
     *    in the {@code repoWithin} repository
     * @param mappedName the RepositoryName by which localName should be referenced
     */
    Builder addRepositoryMappingEntry(
        RepositoryName repoWithin, RepositoryName localName, RepositoryName mappedName) {
      HashMap<RepositoryName, RepositoryName> mapping =
          externalPackageRepositoryMappings
              .computeIfAbsent(repoWithin, (RepositoryName k) -> new HashMap<>());
      mapping.put(localName, mappedName);
      return this;
    }

    /** Adds all the mappings from a given {@link Package}. */
    Builder addRepositoryMappings(Package aPackage) {
      ImmutableMap<RepositoryName, ImmutableMap<RepositoryName, RepositoryName>>
          repositoryMappings = aPackage.externalPackageRepositoryMappings;
      for (Map.Entry<RepositoryName, ImmutableMap<RepositoryName, RepositoryName>> repositoryName :
          repositoryMappings.entrySet()) {
        for (Map.Entry<RepositoryName, RepositoryName> repositoryNameRepositoryNameEntry :
            repositoryName.getValue().entrySet()) {
          addRepositoryMappingEntry(
              repositoryName.getKey(),
              repositoryNameRepositoryNameEntry.getKey(),
              repositoryNameRepositoryNameEntry.getValue());
        }
      }
      return this;
    }

    /** Get the repository mapping for this package */
    ImmutableMap<RepositoryName, RepositoryName> getRepositoryMapping() {
      return this.repositoryMapping;
    }

    Interner<ImmutableList<?>> getListInterner() {
      return listInterner;
    }

    /** Sets the name of this package's BUILD file. */
    Builder setFilename(RootedPath filename) {
      this.filename = filename;
      try {
        buildFileLabel = createLabel(filename.getRootRelativePath().getBaseName());
        addInputFile(buildFileLabel, Location.fromFile(filename.asPath().toString()));
      } catch (LabelSyntaxException e) {
        // This can't actually happen.
        throw new AssertionError("Package BUILD file has an illegal name: " + filename);
      }
      return this;
    }

    Label getBuildFileLabel() {
      return buildFileLabel;
    }

    /**
     * Return a read-only copy of the name mapping of external repositories for a given repository.
     * Reading that mapping directly from the builder allows to also take mappings into account that
     * are only discovered while constructing the external package (e.g., the mapping of the name of
     * the main workspace to the canonical main name '@').
     */
    ImmutableMap<RepositoryName, RepositoryName> getRepositoryMappingFor(RepositoryName name) {
      Map<RepositoryName, RepositoryName> mapping = externalPackageRepositoryMappings.get(name);
      if (mapping == null) {
        return ImmutableMap.<RepositoryName, RepositoryName>of();
      } else {
        return ImmutableMap.<RepositoryName, RepositoryName>copyOf(mapping);
      }
    }

    RootedPath getFilename() {
      return filename;
    }

    public List<Postable> getPosts() {
      return posts;
    }

    public List<Event> getEvents() {
      return events;
    }

    Builder setMakeVariable(String name, String value) {
      this.makeEnv.put(name, value);
      return this;
    }

    /**
     * Sets the default visibility for this package. Called at most once per
     * package from PackageFactory.
     */
    Builder setDefaultVisibility(RuleVisibility visibility) {
      this.defaultVisibility = visibility;
      this.defaultVisibilitySet = true;
      return this;
    }

    /**
     * Sets whether the default visibility is set in the BUILD file.
     */
    Builder setDefaultVisibilitySet(boolean defaultVisibilitySet) {
      this.defaultVisibilitySet = defaultVisibilitySet;
      return this;
    }

    /** Sets the default value of 'testonly'. Rule-level 'testonly' will override this. */
    Builder setDefaultTestonly(boolean defaultTestonly) {
      pkg.setDefaultTestOnly(defaultTestonly);
      return this;
    }

    /**
     * Sets the default value of 'deprecation'. Rule-level 'deprecation' will append to this.
     */
    Builder setDefaultDeprecation(String defaultDeprecation) {
      pkg.setDefaultDeprecation(defaultDeprecation);
      return this;
    }

    /**
     * Uses the workspace name from {@code //external} to set this package's workspace name.
     */
    @VisibleForTesting
    public Builder setWorkspaceName(String workspaceName) {
      pkg.workspaceName = workspaceName;
      return this;
    }

    public Builder setThirdPartyLicenceExistencePolicy(ThirdPartyLicenseExistencePolicy policy) {
      this.thirdPartyLicenceExistencePolicy = policy;
      return this;
    }

    public ThirdPartyLicenseExistencePolicy getThirdPartyLicenseExistencePolicy() {
      return thirdPartyLicenceExistencePolicy;
    }

    /**
     * Returns whether the "package" function has been called yet
     */
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

    /**
     * Sets the default header checking mode.
     */
    public Builder setDefaultHdrsCheck(String hdrsCheck) {
      // Note that this setting is propagated directly to the package because
      // other code needs the ability to read this info directly from the
      // under-construction package. See {@link Package#setDefaultHdrsCheck}.
      pkg.setDefaultHdrsCheck(hdrsCheck);
      return this;
    }

    /** Sets the default value of copts. Rule-level copts will append to this. */
    public Builder setDefaultCopts(List<String> defaultCopts) {
      this.defaultCopts = defaultCopts;
      return this;
    }

    public Builder addFeatures(Iterable<String> features) {
      Iterables.addAll(this.features, features);
      return this;
    }

    Builder setIOExceptionAndMessage(IOException e, String message) {
      this.ioException = e;
      this.ioExceptionMessage = message;
      return setContainsErrors();
    }

    /**
     * Declares that errors were encountering while loading this package.
     */
    public Builder setContainsErrors() {
      containsErrors = true;
      return this;
    }

    public boolean containsErrors() {
      return containsErrors;
    }

    Builder addPosts(Iterable<Postable> posts) {
      for (Postable post : posts) {
        this.posts.add(post);
      }
      return this;
    }

    Builder addEvents(Iterable<Event> events) {
      for (Event event : events) {
        addEvent(event);
      }
      return this;
    }

    public Builder addEvent(Event event) {
      this.events.add(event);
      return this;
    }

    Builder setStarlarkFileDependencies(ImmutableList<Label> starlarkFileDependencies) {
      this.starlarkFileDependencies = starlarkFileDependencies;
      return this;
    }

    /**
     * Sets the default value to use for a rule's {@link RuleClass#APPLICABLE_LICENSES_ATTR}
     * attribute when not explicitly specified by the rule. Records a package error if any labels
     * are duplicated.
     */
    void setDefaultApplicableLicenses(List<Label> licenses, String attrName, Location location) {
      if (hasDuplicateLabels(
          licenses, "package " + pkg.getName(), attrName, location, this::addEvent)) {
        setContainsErrors();
      }
      pkg.setDefaultApplicableLicenses(ImmutableSet.copyOf(licenses));
    }

    ImmutableList<Label> getDefaultApplicableLicenses() {
      return defaultApplicableLicenses;
    }

    /**
     * Sets the default license for this package.
     */
    void setDefaultLicense(License license) {
      this.defaultLicense = license;
    }

    /**
     * Returns the <b>current</b> default {@link License}. This should be used with caution - its
     * value may change during package loading, so it might not reflect the package's final default
     * value.
     */
    License getDefaultLicense() {
      return defaultLicense;
    }

    /**
     * Initializes the default set of distributions for targets in this package.
     *
     * <p> TODO(bazel-team): (2011) consider moving the license & distribs info into Metadata--maybe
     * even in the Build language.
     */
    void setDefaultDistribs(Set<DistributionType> dists) {
      this.defaultDistributionSet = dists;
    }

    /**
     * Sets the default value to use for a rule's {@link RuleClass#COMPATIBLE_ENVIRONMENT_ATTR}
     * attribute when not explicitly specified by the rule. Records a package error if
     * any labels are duplicated.
     */
    void setDefaultCompatibleWith(List<Label> environments, String attrName, Location location) {
      if (hasDuplicateLabels(
          environments, "package " + pkg.getName(), attrName, location, this::addEvent)) {
        setContainsErrors();
      }
      pkg.setDefaultCompatibleWith(ImmutableSet.copyOf(environments));
    }

    /**
     * Sets the default value to use for a rule's {@link RuleClass#RESTRICTED_ENVIRONMENT_ATTR}
     * attribute when not explicitly specified by the rule. Records a package error if
     * any labels are duplicated.
     */
    void setDefaultRestrictedTo(List<Label> environments, String attrName, Location location) {
      if (hasDuplicateLabels(
          environments, "package " + pkg.getName(), attrName, location, this::addEvent)) {
        setContainsErrors();
      }

      pkg.setDefaultRestrictedTo(ImmutableSet.copyOf(environments));
    }

    /**
     * Creates a new {@link Rule} {@code r} where {@code r.getPackage()} is the {@link Package}
     * associated with this {@link Builder}.
     *
     * <p>The created {@link Rule} will have no attribute values, no output files, and therefore
     * will be in an invalid state.
     */
    Rule createRule(
        Label label,
        RuleClass ruleClass,
        Location location,
        List<StarlarkThread.CallStackEntry> callstack,
        AttributeContainer attributeContainer) {
      return new Rule(
          pkg, label, ruleClass, location, callStackBuilder.of(callstack), attributeContainer);
    }

    /**
     * Same as {@link #createRule(Label, RuleClass, Location, List<StarlarkThread.CallStackEntry>,
     * AttributeContainer)}, except allows specifying an {@link ImplicitOutputsFunction} override.
     * Only use if you know what you're doing.
     */
    Rule createRule(
        Label label,
        RuleClass ruleClass,
        Location location,
        List<StarlarkThread.CallStackEntry> callstack,
        AttributeContainer attributeContainer,
        ImplicitOutputsFunction implicitOutputsFunction) {
      return new Rule(
          pkg,
          label,
          ruleClass,
          location,
          callStackBuilder.of(callstack),
          attributeContainer,
          implicitOutputsFunction);
    }

    @Nullable
    Target getTarget(String name) {
      return targets.get(name);
    }

    /**
     * Removes a target from the {@link Package} under construction. Intended to be used only by
     * {@link com.google.devtools.build.lib.skyframe.PackageFunction} to remove targets whose
     * labels cross subpackage boundaries.
     */
    public void removeTarget(Target target) {
      if (target.getPackage() == pkg) {
        this.targets.remove(target.getName());
      }
    }

    public Set<Target> getTargets() {
      return Package.getTargets(targets);
    }

    /**
     * Returns an (immutable, unordered) view of all the targets belonging to
     * this package which are instances of the specified class.
     */
    private <T extends Target> Iterable<T> getTargets(Class<T> targetClass) {
      return Package.getTargets(targets, targetClass);
    }

    /**
     * An input file name conflicts with an existing package member.
     */
    static class GeneratedLabelConflict extends NameConflictException {
      private GeneratedLabelConflict(String message) {
        super(message);
      }
    }

    /**
     * Creates an input file target in this package with the specified name.
     *
     * @param targetName name of the input file.  This must be a valid target
     *   name as defined by {@link
     *   com.google.devtools.build.lib.cmdline.LabelValidator#validateTargetName}.
     * @return the newly-created InputFile, or the old one if it already existed.
     * @throws GeneratedLabelConflict if the name was already taken by a Rule or
     *     an OutputFile target.
     * @throws IllegalArgumentException if the name is not a valid label
     */
    InputFile createInputFile(String targetName, Location location)
        throws GeneratedLabelConflict {
      Target existing = targets.get(targetName);
      if (existing == null) {
        try {
          return addInputFile(createLabel(targetName), location);
        } catch (LabelSyntaxException e) {
          throw new IllegalArgumentException("FileTarget in package " + pkg.getName()
                                             + " has illegal name: " + targetName);
        }
      } else if (existing instanceof InputFile) {
        return (InputFile) existing; // idempotent
      } else {
        throw new GeneratedLabelConflict("generated label '//" + pkg.getName() + ":"
            + targetName + "' conflicts with existing "
            + existing.getTargetKind());
      }
    }

    /**
     * Sets the visibility and license for an input file. The input file must already exist as
     * a member of this package.
     * @throws IllegalArgumentException if the input file doesn't exist in this
     *     package's target map.
     */
    void setVisibilityAndLicense(InputFile inputFile, RuleVisibility visibility, License license) {
      String filename = inputFile.getName();
      Target cacheInstance = targets.get(filename);
      if (!(cacheInstance instanceof InputFile)) {
        throw new IllegalArgumentException("Can't set visibility for nonexistent FileTarget "
                                           + filename + " in package " + pkg.getName() + ".");
      }
      if (!((InputFile) cacheInstance).isVisibilitySpecified()
          || cacheInstance.getVisibility() != visibility
          || !Objects.equals(cacheInstance.getLicense(), license)) {
        targets.put(filename, new InputFile(
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

    /**
     * Adds a package group to the package.
     */
    void addPackageGroup(String name, Collection<String> packages, Collection<Label> includes,
        EventHandler eventHandler, Location location)
        throws NameConflictException, LabelSyntaxException {
      PackageGroup group =
          new PackageGroup(createLabel(name), pkg, packages, includes, eventHandler, location);
      Target existing = targets.get(group.getName());
      if (existing != null) {
        throw nameConflict(group, existing);
      }

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
        Collection<Label> labels,
        String owner,
        String attrName,
        Location location,
        EventHandler eventHandler) {
      Set<Label> dupes = CollectionUtils.duplicatedElementsOf(labels);
      for (Label dupe : dupes) {
        eventHandler.handle(Event.error(location, String.format(
            "label '%s' is duplicated in the '%s' list of '%s'", dupe, attrName, owner)));
      }
      return !dupes.isEmpty();
    }

    /**
     * Adds an environment group to the package.
     */
    void addEnvironmentGroup(String name, List<Label> environments, List<Label> defaults,
        EventHandler eventHandler, Location location)
        throws NameConflictException, LabelSyntaxException {

      if (hasDuplicateLabels(environments, name, "environments", location, eventHandler)
          || hasDuplicateLabels(defaults, name, "defaults", location, eventHandler)) {
        setContainsErrors();
        return;
      }

      EnvironmentGroup group = new EnvironmentGroup(createLabel(name), pkg, environments,
          defaults, location);
      Target existing = targets.get(group.getName());
      if (existing != null) {
        throw nameConflict(group, existing);
      }

      targets.put(group.getName(), group);
      Collection<Event> membershipErrors = group.validateMembership();
      if (!membershipErrors.isEmpty()) {
        for (Event error : membershipErrors) {
          eventHandler.handle(error);
        }
        setContainsErrors();
        return;
      }

      // For each declared environment, make sure it doesn't also belong to some other group.
      for (Label environment : group.getEnvironments()) {
        EnvironmentGroup otherGroup = environmentGroups.get(environment);
        if (otherGroup != null) {
          eventHandler.handle(Event.error(location, "environment " + environment + " belongs to"
              + " both " + group.getLabel() + " and " + otherGroup.getLabel()));
          setContainsErrors();
        } else {
          environmentGroups.put(environment, group);
        }
      }
    }

    /**
     * Same as {@link #addRule}, except with no name conflict checks.
     *
     * <p>Don't call this function unless you know what you're doing.
     */
    void addRuleUnchecked(Rule rule) {
      Preconditions.checkArgument(rule.getPackage() == pkg);
      // Now, modify the package:
      for (OutputFile outputFile : rule.getOutputFiles()) {
        targets.put(outputFile.getName(), outputFile);
        PathFragment outputFileFragment = PathFragment.create(outputFile.getName());
        int segmentCount = outputFileFragment.segmentCount();
        for (int i = 1; i < segmentCount; i++) {
          String prefix = outputFileFragment.subFragment(0, i).toString();
          outputFilePrefixes.putIfAbsent(prefix, outputFile);
        }
      }
      targets.put(rule.getName(), rule);
      if (rule.containsErrors()) {
        this.setContainsErrors();
      }
    }

    void addRule(Rule rule) throws NameConflictException {
      checkForConflicts(rule);
      addRuleUnchecked(rule);
    }

    void addRegisteredExecutionPlatforms(List<String> platforms) {
      this.registeredExecutionPlatforms.addAll(platforms);
    }

    void addRegisteredToolchains(List<String> toolchains) {
      this.registeredToolchains.addAll(toolchains);
    }

    private Builder beforeBuild(boolean discoverAssumedInputFiles) throws NoSuchPackageException {
      Preconditions.checkNotNull(pkg);
      Preconditions.checkNotNull(filename);
      Preconditions.checkNotNull(buildFileLabel);
      Preconditions.checkNotNull(makeEnv);
      if (ioException != null) {
        throw new NoSuchPackageException(getPackageIdentifier(), ioExceptionMessage, ioException);
      }

      // We create the original BUILD InputFile when the package filename is set; however, the
      // visibility may be overridden with an exports_files directive, so we need to obtain the
      // current instance here.
      buildFile = (InputFile) Preconditions.checkNotNull(targets.get(buildFileLabel.getName()));

      List<Label> labelsOfTestTargets = new ArrayList<>();
      List<Rule> implicitTestSuiteRuleInstances = new ArrayList<>();
      Map<Label, InputFile> newInputFiles = new HashMap<>();
      for (final Rule rule : getTargets(Rule.class)) {
        if (discoverAssumedInputFiles) {
          // All labels mentioned in a rule that refer to an unknown target in the
          // current package are assumed to be InputFiles, so let's create them:
          for (AttributeMap.DepEdge depEdge : AggregatingAttributeMapper.of(rule).visitLabels()) {
            InputFile inputFile = createInputFileMaybe(depEdge.getLabel(), rule.getLocation());
            if (inputFile != null && !newInputFiles.containsKey(depEdge.getLabel())) {
              newInputFiles.put(depEdge.getLabel(), inputFile);
            }
          }
        }

        // "test_suite" rules have the idiosyncratic semantics of implicitly
        // depending on all tests in the package, iff tests=[] and suites=[].
        // Note, we implement this here when the Package is fully constructed,
        // since clearly this information isn't available at Rule construction
        // time, as forward references are permitted.
        if (TargetUtils.isTestRule(rule) && !TargetUtils.hasManualTag(rule)) {
          labelsOfTestTargets.add(rule.getLabel());
        }

        AttributeMap attributes = NonconfigurableAttributeMapper.of(rule);
        if (rule.getRuleClass().equals("test_suite")
            && attributes.get("tests", BuildType.LABEL_LIST).isEmpty()) {
          implicitTestSuiteRuleInstances.add(rule);
        }
      }

      for (InputFile inputFile : newInputFiles.values()) {
        addInputFile(inputFile);
      }

      if (!implicitTestSuiteRuleInstances.isEmpty()) {
        Collections.sort(labelsOfTestTargets);
        for (Rule rule : implicitTestSuiteRuleInstances) {
          rule.setAttributeValueByName("$implicit_tests", labelsOfTestTargets);
        }
      }
      return this;
    }

    /** Intended for use by {@link com.google.devtools.build.lib.skyframe.PackageFunction} only. */
    public Builder buildPartial() throws NoSuchPackageException {
      if (alreadyBuilt) {
        return this;
      }
      return beforeBuild(/*discoverAssumedInputFiles=*/ true);
    }

    /** Intended for use by {@link com.google.devtools.build.lib.skyframe.PackageFunction} only. */
    public Package finishBuild() {
      if (alreadyBuilt) {
        return pkg;
      }

      // Freeze targets and distributions.
      targets = Maps.unmodifiableBiMap(targets);
      defaultDistributionSet =
          Collections.unmodifiableSet(defaultDistributionSet);

      // Now all targets have been loaded, so we validate the group's member environments.
      for (EnvironmentGroup envGroup : ImmutableSet.copyOf(environmentGroups.values())) {
        Collection<Event> errors = envGroup.processMemberEnvironments(targets);
        if (!errors.isEmpty()) {
          addEvents(errors);
          setContainsErrors();
        }
      }

      // Build the package.
      pkg.finishInit(this);
      alreadyBuilt = true;
      return pkg;
    }

    public Package build() throws NoSuchPackageException {
      return build(/*discoverAssumedInputFiles=*/ true);
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
     * If "label" refers to a non-existent target in the current package, create an InputFile
     * target.
     */
    private InputFile createInputFileMaybe(Label label, Location location) {
      if (label != null && label.getPackageIdentifier().equals(pkg.getPackageIdentifier())) {
        if (!targets.containsKey(label.getName())) {
          return noImplicitFileExport
              ? new InputFile(
                  pkg, label, location, ConstantRuleVisibility.PRIVATE, License.NO_LICENSE)
              : new InputFile(pkg, label, location);
        }
      }
      return null;
    }

    private InputFile addInputFile(Label label, Location location) {
      return addInputFile(new InputFile(pkg, label, location));
    }

    private InputFile addInputFile(InputFile inputFile) {
      Target prev = targets.put(inputFile.getLabel().getName(), inputFile);
      Preconditions.checkState(prev == null);
      return inputFile;
    }

    /**
     * Precondition check for addRule. We must maintain these invariants of the package:
     *
     * <ul>
     * <li>Each name refers to at most one target.
     * <li>No rule with errors is inserted into the package.
     * <li>The generating rule of every output file in the package must itself be in the package.
     * </ul>
     */
    private void checkForConflicts(Rule rule) throws NameConflictException {
      String name = rule.getName();
      Target existing = targets.get(name);
      if (existing != null) {
        throw nameConflict(rule, existing);
      }
      Map<String, OutputFile> outputFiles = new HashMap<>();

      for (OutputFile outputFile : rule.getOutputFiles()) {
        String outputFileName = outputFile.getName();
        if (outputFiles.put(outputFileName, outputFile) != null) { // dups within a single rule:
          throw duplicateOutputFile(outputFile, outputFile);
        }
        existing = targets.get(outputFileName);
        if (existing != null) {
          throw duplicateOutputFile(outputFile, existing);
        }

        // Check if this output file is the prefix of an already existing one
        if (outputFilePrefixes.containsKey(outputFileName)) {
          throw conflictingOutputFile(outputFile, outputFilePrefixes.get(outputFileName));
        }

        // Check if a prefix of this output file matches an already existing one
        PathFragment outputFileFragment = PathFragment.create(outputFileName);
        int segmentCount = outputFileFragment.segmentCount();
        for (int i = 1; i < segmentCount; i++) {
          String prefix = outputFileFragment.subFragment(0, i).toString();
          if (outputFiles.containsKey(prefix)) {
            throw conflictingOutputFile(outputFile, outputFiles.get(prefix));
          }
          if (targets.containsKey(prefix)
              && targets.get(prefix) instanceof OutputFile) {
            throw conflictingOutputFile(outputFile, (OutputFile) targets.get(prefix));
          }

          outputFilePrefixes.putIfAbsent(prefix, outputFile);
        }
      }

      checkForInputOutputConflicts(rule, outputFiles.keySet());
    }

    /**
     * A utility method that checks for conflicts between input file names and output file names for
     * a rule from a build file.
     *
     * @param rule the rule whose inputs and outputs are to be checked for conflicts.
     * @param outputFiles a set containing the names of output files to be generated by the rule.
     * @throws NameConflictException if a conflict is found.
     */
    private void checkForInputOutputConflicts(Rule rule, Set<String> outputFiles)
        throws NameConflictException {
      PackageIdentifier packageIdentifier = rule.getLabel().getPackageIdentifier();
      for (Label inputLabel : rule.getLabels()) {
        if (packageIdentifier.equals(inputLabel.getPackageIdentifier())
            && outputFiles.contains(inputLabel.getName())) {
          throw inputOutputNameConflict(rule, inputLabel.getName());
        }
      }
    }

    /** An output file conflicts with another output file or the BUILD file. */
    private NameConflictException duplicateOutputFile(OutputFile duplicate, Target existing) {
      return new NameConflictException(duplicate.getTargetKind() + " '" + duplicate.getName()
          + "' in rule '" + duplicate.getGeneratingRule().getName() + "' "
          + conflictsWith(existing));
    }

    /** The package contains two targets with the same name. */
    private NameConflictException nameConflict(Target duplicate, Target existing) {
      return new NameConflictException(duplicate.getTargetKind() + " '" + duplicate.getName()
          + "' in package '" + duplicate.getLabel().getPackageName() + "' "
          + conflictsWith(existing));
    }

    /** A a rule has a input/output name conflict. */
    private NameConflictException inputOutputNameConflict(Rule rule, String conflictingName) {
      return new NameConflictException("rule '" + rule.getName() + "' has file '"
          + conflictingName + "' as both an input and an output");
    }

    private static NameConflictException conflictingOutputFile(
        OutputFile added, OutputFile existing) {
      if (added.getGeneratingRule() == existing.getGeneratingRule()) {
        return new NameConflictException(String.format(
            "rule '%s' has conflicting output files '%s' and '%s'", added.getGeneratingRule()
                .getName(), added.getName(), existing.getName()));
      } else {
        return new NameConflictException(String.format(
            "output file '%s' of rule '%s' conflicts with output file '%s' of rule '%s'", added
                .getName(), added.getGeneratingRule().getName(), existing.getName(), existing
                .getGeneratingRule().getName()));
      }
    }

    /**
     * Utility function for generating exception messages.
     */
    private static String conflictsWith(Target target) {
      String message = "conflicts with existing ";
      if (target instanceof OutputFile) {
        message +=
            "generated file from rule '"
                + ((OutputFile) target).getGeneratingRule().getName()
                + "'";
      } else {
        message += target.getTargetKind();
      }
      return message + ", defined at " + target.getLocation();
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
    public void serialize(
        SerializationContext context,
        Package input,
        CodedOutputStream codedOut)
        throws IOException, SerializationException {
      context.checkClassExplicitlyAllowed(Package.class, input);
      PackageCodecDependencies codecDeps = context.getDependency(PackageCodecDependencies.class);
      codecDeps.getPackageSerializer().serialize(context, input, codedOut);
    }

    @Override
    public Package deserialize(
        DeserializationContext context,
        CodedInputStream codedIn)
        throws SerializationException, IOException {
      PackageCodecDependencies codecDeps = context.getDependency(PackageCodecDependencies.class);
      try {
        return codecDeps.getPackageSerializer().deserialize(context, codedIn);
      } catch (InterruptedException e) {
        throw new IllegalStateException(
            "Unexpected InterruptedException during Package deserialization", e);
      }
    }
  }
}
