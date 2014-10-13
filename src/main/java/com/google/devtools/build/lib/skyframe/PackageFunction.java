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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.CachingPackageLocator;
import com.google.devtools.build.lib.packages.InvalidPackageNameException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageIdentifier;
import com.google.devtools.build.lib.packages.PackageLoadedEvent;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.GlobValue.InvalidGlobPatternException;
import com.google.devtools.build.lib.skyframe.SkylarkImportLookupFunction.SkylarkImportNotFoundException;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.syntax.SkylarkEnvironment;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.JavaClock;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;

import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import javax.annotation.Nullable;

/**
 * A SkyFunction for {@link PackageValue}s.
 */
public class PackageFunction implements SkyFunction {

  private final EventHandler reporter;
  private final PackageFactory packageFactory;
  private final CachingPackageLocator packageLocator;
  private final ConcurrentMap<String, Package.LegacyBuilder> packageFunctionCache;
  private final AtomicBoolean showLoadingProgress;
  private final AtomicReference<EventBus> eventBus;
  private final AtomicInteger numPackagesLoaded;

  private static final PathFragment PRELUDE_FILE_FRAGMENT =
      new PathFragment("devtools/blaze/rules/prelude.bzl");

  static final String DEFAULTS_PACKAGE_NAME = "tools/defaults";
  static final String EXTERNAL_PACKAGE_NAME = "external";

  public PackageFunction(Reporter reporter, PackageFactory packageFactory,
      CachingPackageLocator pkgLocator, AtomicBoolean showLoadingProgress,
      ConcurrentMap<String, Package.LegacyBuilder> packageFunctionCache,
      AtomicReference<EventBus> eventBus, AtomicInteger numPackagesLoaded) {
    this.reporter = reporter;

    this.packageFactory = packageFactory;
    this.packageLocator = pkgLocator;
    this.showLoadingProgress = showLoadingProgress;
    this.packageFunctionCache = packageFunctionCache;
    this.eventBus = eventBus;
    this.numPackagesLoaded = numPackagesLoaded;
  }

  private static void maybeThrowFilesystemInconsistency(Exception skyframeException,
      boolean packageWasInError) throws InconsistentFilesystemException {
    if (!packageWasInError) {
      throw new InconsistentFilesystemException("Encountered error '"
          + skyframeException.getMessage() + "' but didn't encounter it when doing the same thing "
          + "earlier in the build");
    }
  }

  /**
   * Marks the given dependencies, and returns those already present. Ignores any exception
   * thrown while building the dependency, except for filesystem inconsistencies.
   *
   * <p>We need to mark dependencies implicitly used by the legacy package loading code, but we
   * don't care about any skyframe errors since the package knows whether it's in error or not.
   */
  private static Pair<? extends Map<PathFragment, PackageLookupValue>, Boolean>
  getPackageLookupDepsAndPropagateInconsistentFilesystemExceptions(Iterable<SkyKey> depKeys,
      Environment env, boolean packageWasInError) throws InconsistentFilesystemException {
    Preconditions.checkState(
        Iterables.all(depKeys, SkyFunctions.isSkyFunction(SkyFunctions.PACKAGE_LOOKUP)), depKeys);
    boolean packageShouldBeInError = packageWasInError;
    ImmutableMap.Builder<PathFragment, PackageLookupValue> builder = ImmutableMap.builder();
    for (Map.Entry<SkyKey, ValueOrException<Exception>> entry :
        env.getValuesOrThrow(depKeys, Exception.class).entrySet()) {
      PathFragment pkgName = ((PackageIdentifier) entry.getKey().argument()).getPackageFragment();
      try {
        PackageLookupValue value = (PackageLookupValue) entry.getValue().get();
        if (value != null) {
          builder.put(pkgName, value);
        }
      } catch (BuildFileNotFoundException e) {
        maybeThrowFilesystemInconsistency(e, packageWasInError);
      } catch (InconsistentFilesystemException e) {
        throw e;
      } catch (FileSymlinkCycleException e) {
        // Legacy doesn't detect symlink cycles.
        packageShouldBeInError = true;
      } catch (Exception e) {
        throw new IllegalStateException("Unexpected Exception type from PackageLookupValue.", e);
      }
    }
    return Pair.of(builder.build(), packageShouldBeInError);
  }

  private static boolean markFileDepsAndPropagateInconsistentFilesystemExceptions(
      Iterable<SkyKey> depKeys, Environment env, boolean packageWasInError)
          throws InconsistentFilesystemException {
    Preconditions.checkState(
        Iterables.all(depKeys, SkyFunctions.isSkyFunction(SkyFunctions.FILE)), depKeys);
    boolean packageShouldBeInError = packageWasInError;
    for (Map.Entry<SkyKey, ValueOrException<Exception>> entry :
        env.getValuesOrThrow(depKeys, Exception.class).entrySet()) {
      try {
        entry.getValue().get();
      } catch (IOException e) {
        maybeThrowFilesystemInconsistency(e, packageWasInError);
      } catch (FileSymlinkCycleException e) {
        // Legacy doesn't detect symlink cycles.
        packageShouldBeInError = true;
      } catch (InconsistentFilesystemException e) {
        throw e;
      } catch (Exception e) {
        throw new IllegalStateException("Unexpected Exception type from FileValue.", e);
      }
    }
    return packageShouldBeInError;
  }

  private static boolean markGlobDepsAndPropagateInconsistentFilesystemExceptions(
      Iterable<SkyKey> depKeys, Environment env, boolean packageWasInError)
          throws InconsistentFilesystemException {
    Preconditions.checkState(
        Iterables.all(depKeys, SkyFunctions.isSkyFunction(SkyFunctions.GLOB)), depKeys);
    boolean packageShouldBeInError = packageWasInError;
    for (Map.Entry<SkyKey, ValueOrException<Exception>> entry :
        env.getValuesOrThrow(depKeys, Exception.class).entrySet()) {
      try {
        entry.getValue().get();
      } catch (IOException | BuildFileNotFoundException e) {
        maybeThrowFilesystemInconsistency(e, packageWasInError);
      } catch (FileSymlinkCycleException e) {
        // Legacy doesn't detect symlink cycles.
        packageShouldBeInError = true;
      } catch (InconsistentFilesystemException e) {
        throw e;
      } catch (Exception e) {
        throw new IllegalStateException("Unexpected Exception type from GlobValue.", e);
      }
    }
    return packageShouldBeInError;
  }

  /**
   * Marks dependencies implicitly used by legacy package loading code, after the fact. Note that
   * the given package might already be in error.
   *
   * <p>Any skyframe exceptions encountered here are ignored, as similar errors should have
   * already been encountered by legacy package loading (if not, then the filesystem is
   * inconsistent).
   */
  private static boolean markDependenciesAndPropagateInconsistentFilesystemExceptions(
      Package pkg, Environment env, Collection<Pair<String, Boolean>> globPatterns)
          throws InconsistentFilesystemException {
    boolean packageShouldBeInError = pkg.containsErrors();

    // TODO(bazel-team): This means that many packages will have to be preprocessed twice. Ouch!
    // We need a better continuation mechanism to avoid repeating work. [skyframe-loading]

    // TODO(bazel-team): It would be preferable to perform I/O from the package preprocessor via
    // Skyframe rather than add (potentially incomplete) dependencies after the fact.
    // [skyframe-loading]

    Set<SkyKey> subincludePackageLookupDepKeys = Sets.newHashSet();
    for (Label label : pkg.getSubincludes().keySet()) {
      // Declare a dependency on the package lookup for the package giving access to the label.
      subincludePackageLookupDepKeys.add(PackageLookupValue.key(label.getPackageFragment()));
    }
    Pair<? extends Map<PathFragment, PackageLookupValue>, Boolean> subincludePackageLookupResult =
        getPackageLookupDepsAndPropagateInconsistentFilesystemExceptions(
            subincludePackageLookupDepKeys, env, pkg.containsErrors());
    Map<PathFragment, PackageLookupValue> subincludePackageLookupDeps =
        subincludePackageLookupResult.getFirst();
    packageShouldBeInError = subincludePackageLookupResult.getSecond();
    List<SkyKey> subincludeFileDepKeys = Lists.newArrayList();
    for (Entry<Label, Path> subincludeEntry : pkg.getSubincludes().entrySet()) {
      // Ideally, we would have a direct dependency on the target with the given label, but then
      // subincluding a file from the same package will cause a dependency cycle, since targets
      // depend on their containing packages.
      Label label = subincludeEntry.getKey();
      PackageLookupValue subincludePackageLookupValue =
          subincludePackageLookupDeps.get(label.getPackageFragment());
      if (subincludePackageLookupValue != null) {
        // Declare a dependency on the actual file that was subincluded.
        Path subincludeFilePath = subincludeEntry.getValue();
        if (subincludeFilePath != null) {
          if (!subincludePackageLookupValue.packageExists()) {
            // Legacy blaze puts a non-null path when only when the package does indeed exist.
            throw new InconsistentFilesystemException(String.format(
                "Unexpected package in %s. Was it modified during the build?", subincludeFilePath));
          }
          // Sanity check for consistency of Skyframe and legacy blaze.
          Path subincludeFilePathSkyframe =
              subincludePackageLookupValue.getRoot().getRelative(label.toPathFragment());
          if (!subincludeFilePathSkyframe.equals(subincludeFilePath)) {
            throw new InconsistentFilesystemException(String.format(
                "Inconsistent package location for %s: '%s' vs '%s'. "
                + "Was the source tree modified during the build?",
                label.getPackageFragment(), subincludeFilePathSkyframe, subincludeFilePath));
          }
          // The actual file may be under a different package root than the package being
          // constructed.
          SkyKey subincludeSkyKey =
              FileValue.key(RootedPath.toRootedPath(subincludePackageLookupValue.getRoot(),
                  subincludeFilePath));
          subincludeFileDepKeys.add(subincludeSkyKey);
        }
      }
    }
    packageShouldBeInError = markFileDepsAndPropagateInconsistentFilesystemExceptions(
        subincludeFileDepKeys, env, pkg.containsErrors());
    // Another concern is a subpackage cutting off the subinclude label, but this is already
    // handled by the legacy package loading code which calls into our SkyframePackageLocator.

    // TODO(bazel-team): In the long term, we want to actually resolve the glob patterns within
    // Skyframe. For now, just logging the glob requests provides correct incrementality and
    // adequate performance.
    PathFragment packagePathFragment = pkg.getNameFragment();
    List<SkyKey> globDepKeys = Lists.newArrayList();
    for (Pair<String, Boolean> globPattern : globPatterns) {
      String pattern = globPattern.getFirst();
      boolean excludeDirs = globPattern.getSecond();
      SkyKey globSkyKey;
      try {
        globSkyKey = GlobValue.key(packagePathFragment, pattern, excludeDirs);
      } catch (InvalidGlobPatternException e) {
        // Globs that make it to pkg.getGlobPatterns() should already be filtered for errors.
        throw new IllegalStateException(e);
      }
      globDepKeys.add(globSkyKey);
    }
    packageShouldBeInError = markGlobDepsAndPropagateInconsistentFilesystemExceptions(globDepKeys,
        env, pkg.containsErrors());
    return packageShouldBeInError;
  }

  /**
   * Adds a dependency on the WORKSPACE file, representing it as a special type of package.
   * @throws PackageFunctionException if there is an error computing the workspace file or adding
   * its rules to the //external package.
   */
  private SkyValue getExternalPackage(SkyKey key, Environment env,
      Path packageLookupPath) throws PackageFunctionException {
    RootedPath workspacePath = RootedPath.toRootedPath(
        packageLookupPath, new PathFragment("WORKSPACE"));
    SkyKey workspaceKey = WorkspaceFileValue.key(workspacePath);
    WorkspaceFileValue workspace = null;
    try {
      workspace = (WorkspaceFileValue) env.getValueOrThrow(workspaceKey, Exception.class);
    } catch (IOException | SkyFunctionException | FileSymlinkCycleException |
        InconsistentFilesystemException e) {
      throw new PackageFunctionException(key, new BadWorkspaceFileException(e.getMessage()));
    } catch (Exception e) {
      throw new IllegalStateException("Unexpected Exception type from WorkspaceFileValue.", e);
    }
    if (workspace == null) {
      return null;
    }

    Package pkg = workspace.getPackage();
    if (pkg.containsErrors()) {
      throw new PackageFunctionException(key, new BuildFileContainsErrorsException("external",
          "Package 'external' contains errors"));
    }

    return new PackageValue(pkg);
  }

  @Override
  public SkyValue compute(SkyKey key, Environment env) throws PackageFunctionException,
      InterruptedException {
    PackageIdentifier packageKey = (PackageIdentifier) key.argument();
    PathFragment packageNameFragment = packageKey.getPackageFragment();
    String packageName = packageNameFragment.getPathString();

    SkyKey packageLookupKey = PackageLookupValue.key(packageKey);
    PackageLookupValue packageLookupValue;
    try {
      packageLookupValue = (PackageLookupValue)
          env.getValueOrThrow(packageLookupKey, Exception.class);
    } catch (BuildFileNotFoundException e) {
      throw new PackageFunctionException(key, e);
    } catch (InconsistentFilesystemException e) {
      throw new PackageFunctionException(key,
          new InternalInconsistentFilesystemException(packageName, e));
    } catch (Exception e) {
      throw new IllegalStateException("Unexpected Exception type from PackageLookupValue.", e);
    }
    if (packageLookupValue == null) {
      return null;
    }

    if (!packageLookupValue.packageExists()) {
      switch (packageLookupValue.getErrorReason()) {
        case NO_BUILD_FILE:
        case DELETED_PACKAGE:
          throw new PackageFunctionException(key, new BuildFileNotFoundException(packageName,
              packageLookupValue.getErrorMsg()));
        case INVALID_PACKAGE_NAME:
          throw new PackageFunctionException(key, new InvalidPackageNameException(packageName,
              packageLookupValue.getErrorMsg()));
        default:
          // We should never get here.
          Preconditions.checkState(false);
      }
    }

    if (packageName.equals(EXTERNAL_PACKAGE_NAME)) {
      return getExternalPackage(key, env, packageLookupValue.getRoot());
    }

    RootedPath buildFileRootedPath = RootedPath.toRootedPath(packageLookupValue.getRoot(),
        packageNameFragment.getChild("BUILD"));
    FileValue buildFileValue;
    try {
      buildFileValue = (FileValue) env.getValueOrThrow(FileValue.key(buildFileRootedPath),
          Exception.class);
    } catch (IOException | FileSymlinkCycleException | InconsistentFilesystemException e) {
      throw new IllegalStateException("Package lookup succeeded but encountered error when "
          + "getting FileValue for BUILD file directly.", e);
    } catch (Exception e) {
      throw new IllegalStateException("Unexpected Exception type from FileValue.", e);
    }
    Preconditions.checkState(buildFileValue.exists(),
        "Package lookup succeeded but BUILD file doesn't exist");
    Path buildFilePath = buildFileRootedPath.asPath();

    String replacementContents = null;
    if (packageName.equals(DEFAULTS_PACKAGE_NAME)) {
      replacementContents = BuildVariableValue.DEFAULTS_PACKAGE_CONTENTS.get(env);
      if (replacementContents == null) {
        return null;
      }
    }

    RuleVisibility defaultVisibility = BuildVariableValue.DEFAULT_VISIBILITY.get(env);
    if (defaultVisibility == null) {
      return null;
    }

    SkyKey astLookupKey = ASTLookupValue.key(PRELUDE_FILE_FRAGMENT);
    ASTLookupValue astLookupValue;
    astLookupValue = (ASTLookupValue) env.getValue(astLookupKey);
    if (astLookupValue == null) {
      return null;
    }
    List<Statement> preludeStatements = astLookupValue == ASTLookupValue.NO_FILE
        ? ImmutableList.<Statement>of() : astLookupValue.getAST().getStatements();

    // Load the BUILD file AST and handle Skylark dependencies. This way BUILD files are
    // only loaded twice if there are unavailable Skylark or package dependencies or an
    // IOException occurs. Note that the BUILD files are still parsed two times.
    ParserInputSource inputSource;
    try {
      if (showLoadingProgress.get() && !packageFunctionCache.containsKey(packageName)) {
        // TODO(bazel-team): don't duplicate the loading message if there are unavailable
        // Skylark dependencies.
        reporter.handle(Event.progress("Loading package: " + packageName));
      }
      inputSource = ParserInputSource.create(buildFilePath);
    } catch (IOException e) {
      env.getListener().handle(Event.error(Location.fromFile(buildFilePath), e.getMessage()));
      throw new PackageFunctionException(key, new BuildFileContainsErrorsException(
          packageName, e.getMessage()));
    }
    Map<PathFragment, SkylarkEnvironment> imports =
        fetchImportsFromBuildFile(buildFilePath, preludeStatements, inputSource, key,
            packageName, env);
    if (imports == null) {
      return null;
    }

    Package.LegacyBuilder legacyPkgBuilder = loadPackage(inputSource, replacementContents,
        packageName, buildFilePath, defaultVisibility, preludeStatements, imports);
    legacyPkgBuilder.buildPartial();
    handleLabelsCrossingSubpackages(packageLookupValue.getRoot(), packageNameFragment,
        legacyPkgBuilder, env);
    if (env.valuesMissing()) {
      // The package we just loaded will be in the {@code packageFunctionCache} next when this
      // SkyFunction is called again.
      return null;
    }
    Collection<Pair<String, Boolean>> globPatterns = legacyPkgBuilder.getGlobPatterns();
    Package pkg = legacyPkgBuilder.finishBuild();
    Event.replayEventsOn(env.getListener(), pkg.getEvents());
    boolean packageShouldBeConsideredInError = pkg.containsErrors();
    try {
      packageShouldBeConsideredInError =
          markDependenciesAndPropagateInconsistentFilesystemExceptions(pkg, env,
              globPatterns);
    } catch (InconsistentFilesystemException e) {
      packageFunctionCache.remove(packageName);
      throw new PackageFunctionException(key,
          new InternalInconsistentFilesystemException(packageName, e));
    }

    if (env.valuesMissing()) {
      return null;
    }
    // We know this SkyFunction will not be called again, so we can remove the cache entry.
    packageFunctionCache.remove(packageName);

    if (packageShouldBeConsideredInError) {
      throw new PackageFunctionException(key, new BuildFileContainsErrorsException(pkg,
          "Package '" + packageName + "' contains errors"));
    }
    return new PackageValue(pkg);
  }

  private Map<PathFragment, SkylarkEnvironment> fetchImportsFromBuildFile(Path buildFilePath,
      List<Statement> preludeStatements, ParserInputSource inputSource,
      SkyKey key, String packageName, Environment env) throws PackageFunctionException {
    StoredEventHandler eventHandler = new StoredEventHandler();
    BuildFileAST buildFileAST = BuildFileAST.parseBuildFile(
          inputSource, preludeStatements, eventHandler, null, true);

    if (eventHandler.hasErrors()) {
      // In case of Python preprocessing, errors have already been reported (see checkSyntax).
      // In other cases, errors will be reported later.
      // TODO(bazel-team): maybe we could get rid of checkSyntax and always report errors here?
      return ImmutableMap.<PathFragment, SkylarkEnvironment>of();
    }

    ImmutableCollection<PathFragment> imports = buildFileAST.getImports();
    Map<PathFragment, SkylarkEnvironment> importMap = new HashMap<>();
    try {
      for (PathFragment importFile : imports) {
        SkyKey importsLookupKey = SkylarkImportLookupValue.key(importFile);
        SkylarkImportLookupValue importLookupValue = (SkylarkImportLookupValue)
            env.getValueOrThrow(importsLookupKey, SkylarkImportNotFoundException.class);
        if (importLookupValue != null) {
          importMap.put(importFile, importLookupValue.getImportedEnvironment());
        }
      }
    } catch (SkylarkImportNotFoundException e) {
      env.getListener().handle(Event.error(Location.fromFile(buildFilePath), e.getMessage()));
      throw new PackageFunctionException(key,
          new BuildFileContainsErrorsException(packageName, e.getMessage()));
    }
    if (env.valuesMissing()) {
      // There are unavailable Skylark dependencies.
      return null;
    }
    return importMap;
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private static void handleLabelsCrossingSubpackages(Path pkgRoot, PathFragment pkgNameFragment,
      Package.LegacyBuilder pkgBuilder, Environment env) {
    Set<SkyKey> containingPkgLookupKeys = Sets.newHashSet();
    for (Target target : pkgBuilder.getTargets()) {
      PathFragment dir = target.getLabel().toPathFragment().getParentDirectory();
      if (dir.equals(pkgNameFragment)) {
        continue;
      }
      containingPkgLookupKeys.add(ContainingPackageLookupValue.key(dir));
    }
    for (Label subincludeLabel : pkgBuilder.getSubincludeLabels()) {
      PathFragment dir = subincludeLabel.toPathFragment().getParentDirectory();
      if (dir.equals(pkgNameFragment)) {
        continue;
      }
      containingPkgLookupKeys.add(ContainingPackageLookupValue.key(dir));
    }
    Map<SkyKey, SkyValue> containingPkgLookupValues = env.getValues(containingPkgLookupKeys);
    for (Target target : ImmutableSet.copyOf(pkgBuilder.getTargets())) {
      if (maybeAddEventAboutLabelCrossingSubpackage(pkgBuilder, pkgRoot, target.getLabel(),
          target.getLocation(), containingPkgLookupValues)) {
        pkgBuilder.removeTarget(target);
        pkgBuilder.setContainsErrors();
      }
    }
    for (Label subincludeLabel : pkgBuilder.getSubincludeLabels()) {
      if (maybeAddEventAboutLabelCrossingSubpackage(pkgBuilder, pkgRoot, subincludeLabel,
          /*location=*/null, containingPkgLookupValues)) {
        pkgBuilder.setContainsErrors();
      }
    }
  }

  private static boolean maybeAddEventAboutLabelCrossingSubpackage(
      Package.LegacyBuilder pkgBuilder, Path pkgRoot, Label label, @Nullable Location location,
      Map<SkyKey, SkyValue> containingPkgLookupValues) {
    PathFragment dir = label.toPathFragment().getParentDirectory();
    ContainingPackageLookupValue containingPkgLookupValue =
        (ContainingPackageLookupValue) containingPkgLookupValues.get(
            ContainingPackageLookupValue.key(dir));
    if (containingPkgLookupValue == null) {
      return false;
    }
    if (!containingPkgLookupValue.hasContainingPackage()) {
      // The missing package here is a problem, but it's not an error from the perspective of
      // PackageFunction.
      return false;
    }
    PathFragment containingPkg = containingPkgLookupValue.getContainingPackageName();
    if (containingPkg.equals(label.getPackageFragment())) {
      // The label does not cross a subpackage boundary.
      return false;
    }
    if (!containingPkg.startsWith(label.getPackageFragment())) {
      // This label is referencing an imaginary package, because the containing package should
      // extend the label's package: if the label is //a/b:c/d, the containing package could be
      // //a/b/c or //a/b, but should never be //a. Usually such errors will be caught earlier, but
      // in some exceptional cases (such as a Python-aware BUILD file catching its own io
      // exceptions), it reaches here, and we tolerate it.
      return false;
    }
    PathFragment labelNameFragment = new PathFragment(label.getName());
    String message = String.format("Label '%s' crosses boundary of subpackage '%s'",
        label, containingPkg);
    Path containingRoot = containingPkgLookupValue.getContainingPackageRoot();
    if (pkgRoot.equals(containingRoot)) {
      PathFragment labelNameInContainingPackage = labelNameFragment.subFragment(
          containingPkg.segmentCount() - label.getPackageFragment().segmentCount(),
          labelNameFragment.segmentCount());
      message += " (perhaps you meant to put the colon here: "
          + "'//" + containingPkg + ":" + labelNameInContainingPackage + "'?)";
    } else {
      message += " (have you deleted " + containingPkg + "/BUILD? "
          + "If so, use the --deleted_packages=" + containingPkg + " option)";
    }
    pkgBuilder.addEvent(Event.error(location, message));
    return true;
  }

  /**
   * Constructs a {@link Package} object for the given package using legacy package loading.
   * Note that the returned package may be in error.
   */
  private Package.LegacyBuilder loadPackage(ParserInputSource inputSource,
      @Nullable String replacementContents,
      String packageName, Path buildFilePath, RuleVisibility defaultVisibility,
      List<Statement> preludeStatements, Map<PathFragment, SkylarkEnvironment> imports)
          throws InterruptedException {
    ParserInputSource replacementSource = replacementContents == null ? null
        : ParserInputSource.create(replacementContents, buildFilePath);
    Package.LegacyBuilder pkgBuilder = packageFunctionCache.get(packageName);
    if (pkgBuilder == null) {
      Clock clock = new JavaClock();
      long startTime = clock.nanoTime();
      pkgBuilder = packageFactory.createPackage(packageName, buildFilePath, preludeStatements,
          inputSource, imports, packageLocator, replacementSource, defaultVisibility);

      if (eventBus.get() != null) {
        eventBus.get().post(new PackageLoadedEvent(packageName,
            (clock.nanoTime() - startTime) / (1000 * 1000),
            // It's impossible to tell if the package was loaded before, so we always pass false.
            /*reloading=*/false,
            // This isn't completely correct since we may encounter errors later (e.g. filesystem
            // inconsistencies)
            !pkgBuilder.containsErrors()));
      }

      numPackagesLoaded.incrementAndGet();
      packageFunctionCache.put(packageName, pkgBuilder);
    }
    return pkgBuilder;
  }

  static class InternalInconsistentFilesystemException extends NoSuchPackageException {
    public InternalInconsistentFilesystemException(String packageName,
        InconsistentFilesystemException e) {
      super(packageName, e.getMessage(), e);
    }
  }

  private static class BadWorkspaceFileException extends NoSuchPackageException {
    private BadWorkspaceFileException(String message) {
      super("external", "Error encountered while dealing with the WORKSPACE file: " + message);
    }
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link PackageFunction#compute}.
   */
  private static class PackageFunctionException extends SkyFunctionException {
    public PackageFunctionException(SkyKey key, NoSuchPackageException e) {
      super(key, e);
    }
  }
}
