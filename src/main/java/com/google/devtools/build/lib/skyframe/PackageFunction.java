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
import com.google.devtools.build.lib.Constants;
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
import com.google.devtools.build.lib.packages.PackageFactory.Globber;
import com.google.devtools.build.lib.packages.PackageIdentifier;
import com.google.devtools.build.lib.packages.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.packages.PackageLoadedEvent;
import com.google.devtools.build.lib.packages.Preprocessor;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.skyframe.ASTFileLookupValue.ASTLookupInputException;
import com.google.devtools.build.lib.skyframe.GlobValue.InvalidGlobPatternException;
import com.google.devtools.build.lib.skyframe.SkylarkImportLookupFunction.SkylarkImportFailedException;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.EvalException;
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
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException3;
import com.google.devtools.build.skyframe.ValueOrException4;

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
  private final ConcurrentMap<PackageIdentifier, Package.LegacyBuilder> packageFunctionCache;
  private final AtomicBoolean showLoadingProgress;
  private final AtomicReference<EventBus> eventBus;
  private final AtomicInteger numPackagesLoaded;
  private final Profiler profiler = Profiler.instance();

  private static final PathFragment PRELUDE_FILE_FRAGMENT =
      new PathFragment(Constants.PRELUDE_FILE_DEPOT_RELATIVE_PATH);

  static final String DEFAULTS_PACKAGE_NAME = "tools/defaults";
  public static final String EXTERNAL_PACKAGE_NAME = "external";

  static {
    Preconditions.checkArgument(!PRELUDE_FILE_FRAGMENT.isAbsolute());
  }

  public PackageFunction(Reporter reporter, PackageFactory packageFactory,
      CachingPackageLocator pkgLocator, AtomicBoolean showLoadingProgress,
      ConcurrentMap<PackageIdentifier, Package.LegacyBuilder> packageFunctionCache,
      AtomicReference<EventBus> eventBus, AtomicInteger numPackagesLoaded) {
    this.reporter = reporter;

    this.packageFactory = packageFactory;
    this.packageLocator = pkgLocator;
    this.showLoadingProgress = showLoadingProgress;
    this.packageFunctionCache = packageFunctionCache;
    this.eventBus = eventBus;
    this.numPackagesLoaded = numPackagesLoaded;
  }

  private static void maybeThrowFilesystemInconsistency(String packageName,
      Exception skyframeException, boolean packageWasInError)
          throws InternalInconsistentFilesystemException {
    if (!packageWasInError) {
      throw new InternalInconsistentFilesystemException(packageName, "Encountered error '"
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
  getPackageLookupDepsAndPropagateInconsistentFilesystemExceptions(String packageName,
      Iterable<SkyKey> depKeys, Environment env, boolean packageWasInError)
          throws InternalInconsistentFilesystemException {
    Preconditions.checkState(
        Iterables.all(depKeys, SkyFunctions.isSkyFunction(SkyFunctions.PACKAGE_LOOKUP)), depKeys);
    boolean packageShouldBeInError = packageWasInError;
    ImmutableMap.Builder<PathFragment, PackageLookupValue> builder = ImmutableMap.builder();
    for (Map.Entry<SkyKey, ValueOrException3<BuildFileNotFoundException,
        InconsistentFilesystemException, FileSymlinkCycleException>> entry :
            env.getValuesOrThrow(depKeys, BuildFileNotFoundException.class,
                InconsistentFilesystemException.class,
                FileSymlinkCycleException.class).entrySet()) {
      PathFragment pkgName = ((PackageIdentifier) entry.getKey().argument()).getPackageFragment();
      try {
        PackageLookupValue value = (PackageLookupValue) entry.getValue().get();
        if (value != null) {
          builder.put(pkgName, value);
        }
      } catch (BuildFileNotFoundException e) {
        maybeThrowFilesystemInconsistency(packageName, e, packageWasInError);
      } catch (InconsistentFilesystemException e) {
        throw new InternalInconsistentFilesystemException(packageName, e);
      } catch (FileSymlinkCycleException e) {
        // Legacy doesn't detect symlink cycles.
        packageShouldBeInError = true;
      }
    }
    return Pair.of(builder.build(), packageShouldBeInError);
  }

  private static boolean markFileDepsAndPropagateInconsistentFilesystemExceptions(
      String packageName, Iterable<SkyKey> depKeys, Environment env, boolean packageWasInError)
          throws InternalInconsistentFilesystemException {
    Preconditions.checkState(
        Iterables.all(depKeys, SkyFunctions.isSkyFunction(SkyFunctions.FILE)), depKeys);
    boolean packageShouldBeInError = packageWasInError;
    for (Map.Entry<SkyKey, ValueOrException3<IOException, FileSymlinkCycleException,
        InconsistentFilesystemException>> entry : env.getValuesOrThrow(depKeys, IOException.class,
            FileSymlinkCycleException.class, InconsistentFilesystemException.class).entrySet()) {
      try {
        entry.getValue().get();
      } catch (IOException e) {
        maybeThrowFilesystemInconsistency(packageName, e, packageWasInError);
      } catch (FileSymlinkCycleException e) {
        // Legacy doesn't detect symlink cycles.
        packageShouldBeInError = true;
      } catch (InconsistentFilesystemException e) {
        throw new InternalInconsistentFilesystemException(packageName, e);
      }
    }
    return packageShouldBeInError;
  }

  private static boolean markGlobDepsAndPropagateInconsistentFilesystemExceptions(
      String packageName, Iterable<SkyKey> depKeys, Environment env, boolean packageWasInError)
          throws InternalInconsistentFilesystemException {
    Preconditions.checkState(
        Iterables.all(depKeys, SkyFunctions.isSkyFunction(SkyFunctions.GLOB)), depKeys);
    boolean packageShouldBeInError = packageWasInError;
    for (Map.Entry<SkyKey, ValueOrException4<IOException, BuildFileNotFoundException,
        FileSymlinkCycleException, InconsistentFilesystemException>> entry :
        env.getValuesOrThrow(depKeys, IOException.class, BuildFileNotFoundException.class,
            FileSymlinkCycleException.class, InconsistentFilesystemException.class).entrySet()) {
      try {
        entry.getValue().get();
      } catch (IOException | BuildFileNotFoundException e) {
        maybeThrowFilesystemInconsistency(packageName, e, packageWasInError);
      } catch (FileSymlinkCycleException e) {
        // Legacy doesn't detect symlink cycles.
        packageShouldBeInError = true;
      } catch (InconsistentFilesystemException e) {
        throw new InternalInconsistentFilesystemException(packageName, e);
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
      Package pkg, Environment env, Collection<Pair<String, Boolean>> globPatterns,
      Map<Label, Path> subincludes) throws InternalInconsistentFilesystemException {
    boolean packageShouldBeInError = pkg.containsErrors();

    // TODO(bazel-team): This means that many packages will have to be preprocessed twice. Ouch!
    // We need a better continuation mechanism to avoid repeating work. [skyframe-loading]

    // TODO(bazel-team): It would be preferable to perform I/O from the package preprocessor via
    // Skyframe rather than add (potentially incomplete) dependencies after the fact.
    // [skyframe-loading]

    Set<SkyKey> subincludePackageLookupDepKeys = Sets.newHashSet();
    for (Label label : pkg.getSubincludeLabels()) {
      // Declare a dependency on the package lookup for the package giving access to the label.
      subincludePackageLookupDepKeys.add(PackageLookupValue.key(label.getPackageFragment()));
    }
    Pair<? extends Map<PathFragment, PackageLookupValue>, Boolean> subincludePackageLookupResult =
        getPackageLookupDepsAndPropagateInconsistentFilesystemExceptions(pkg.getName(),
            subincludePackageLookupDepKeys, env, pkg.containsErrors());
    Map<PathFragment, PackageLookupValue> subincludePackageLookupDeps =
        subincludePackageLookupResult.getFirst();
    packageShouldBeInError = subincludePackageLookupResult.getSecond();
    List<SkyKey> subincludeFileDepKeys = Lists.newArrayList();
    for (Entry<Label, Path> subincludeEntry : subincludes.entrySet()) {
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
            throw new InternalInconsistentFilesystemException(pkg.getName(), String.format(
                "Unexpected package in %s. Was it modified during the build?", subincludeFilePath));
          }
          // Sanity check for consistency of Skyframe and legacy blaze.
          Path subincludeFilePathSkyframe =
              subincludePackageLookupValue.getRoot().getRelative(label.toPathFragment());
          if (!subincludeFilePathSkyframe.equals(subincludeFilePath)) {
            throw new InternalInconsistentFilesystemException(pkg.getName(), String.format(
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
        pkg.getName(), subincludeFileDepKeys, env, pkg.containsErrors());

    // TODO(bazel-team): In the long term, we want to actually resolve the glob patterns within
    // Skyframe. For now, just logging the glob requests provides correct incrementality and
    // adequate performance.
    PackageIdentifier packageId = pkg.getPackageIdentifier();
    List<SkyKey> globDepKeys = Lists.newArrayList();
    for (Pair<String, Boolean> globPattern : globPatterns) {
      String pattern = globPattern.getFirst();
      boolean excludeDirs = globPattern.getSecond();
      SkyKey globSkyKey;
      try {
        globSkyKey = GlobValue.key(packageId, pattern, excludeDirs);
      } catch (InvalidGlobPatternException e) {
        // Globs that make it to pkg.getGlobPatterns() should already be filtered for errors.
        throw new IllegalStateException(e);
      }
      globDepKeys.add(globSkyKey);
    }
    packageShouldBeInError = markGlobDepsAndPropagateInconsistentFilesystemExceptions(
        pkg.getName(), globDepKeys, env, pkg.containsErrors());
    return packageShouldBeInError;
  }

  /**
   * Adds a dependency on the WORKSPACE file, representing it as a special type of package.
   * @throws PackageFunctionException if there is an error computing the workspace file or adding
   * its rules to the //external package.
   */
  private SkyValue getExternalPackage(Environment env, Path packageLookupPath)
      throws PackageFunctionException {
    RootedPath workspacePath = RootedPath.toRootedPath(
        packageLookupPath, new PathFragment("WORKSPACE"));
    SkyKey workspaceKey = PackageValue.workspaceKey(workspacePath);
    PackageValue workspace = null;
    try {
      workspace = (PackageValue) env.getValueOrThrow(workspaceKey, IOException.class,
          FileSymlinkCycleException.class, InconsistentFilesystemException.class,
          EvalException.class);
    } catch (IOException | FileSymlinkCycleException | InconsistentFilesystemException
        | EvalException e) {
      throw new PackageFunctionException(new BadWorkspaceFileException(e.getMessage()),
          Transience.PERSISTENT);
    }
    if (workspace == null) {
      return null;
    }

    Package pkg = workspace.getPackage();
    Event.replayEventsOn(env.getListener(), pkg.getEvents());
    if (pkg.containsErrors()) {
      throw new PackageFunctionException(new BuildFileContainsErrorsException("external",
          "Package 'external' contains errors"),
          pkg.containsTemporaryErrors() ? Transience.TRANSIENT : Transience.PERSISTENT);
    }

    return new PackageValue(pkg);
  }

  @Override
  public SkyValue compute(SkyKey key, Environment env) throws PackageFunctionException,
      InterruptedException {
    PackageIdentifier packageId = (PackageIdentifier) key.argument();
    PathFragment packageNameFragment = packageId.getPackageFragment();
    String packageName = packageNameFragment.getPathString();

    SkyKey packageLookupKey = PackageLookupValue.key(packageId);
    PackageLookupValue packageLookupValue;
    try {
      packageLookupValue = (PackageLookupValue)
          env.getValueOrThrow(packageLookupKey, BuildFileNotFoundException.class,
              InconsistentFilesystemException.class);
    } catch (BuildFileNotFoundException e) {
      throw new PackageFunctionException(e, Transience.PERSISTENT);
    } catch (InconsistentFilesystemException e) {
      // This error is not transient from the perspective of the PackageFunction.
      throw new PackageFunctionException(
          new InternalInconsistentFilesystemException(packageName, e), Transience.PERSISTENT);
    }
    if (packageLookupValue == null) {
      return null;
    }

    if (!packageLookupValue.packageExists()) {
      switch (packageLookupValue.getErrorReason()) {
        case NO_BUILD_FILE:
        case DELETED_PACKAGE:
        case NO_EXTERNAL_PACKAGE:
          throw new PackageFunctionException(new BuildFileNotFoundException(packageName,
              packageLookupValue.getErrorMsg()), Transience.PERSISTENT);
        case INVALID_PACKAGE_NAME:
          throw new PackageFunctionException(new InvalidPackageNameException(packageName,
              packageLookupValue.getErrorMsg()), Transience.PERSISTENT);
        default:
          // We should never get here.
          throw new IllegalStateException();
      }
    }

    if (packageName.equals(EXTERNAL_PACKAGE_NAME)) {
      return getExternalPackage(env, packageLookupValue.getRoot());
    }
    PackageValue externalPackage = (PackageValue) env.getValue(
        PackageValue.key(PackageIdentifier.createInDefaultRepo(EXTERNAL_PACKAGE_NAME)));
    if (externalPackage == null) {
      return null;
    }
    Package externalPkg = externalPackage.getPackage();

    PathFragment buildFileFragment = packageNameFragment.getChild("BUILD");
    RootedPath buildFileRootedPath = RootedPath.toRootedPath(packageLookupValue.getRoot(),
        buildFileFragment);
    FileValue buildFileValue;
    try {
      buildFileValue = (FileValue) env.getValueOrThrow(FileValue.key(buildFileRootedPath),
          IOException.class, FileSymlinkCycleException.class,
          InconsistentFilesystemException.class);
    } catch (IOException | FileSymlinkCycleException | InconsistentFilesystemException e) {
      throw new IllegalStateException("Package lookup succeeded but encountered error when "
          + "getting FileValue for BUILD file directly.", e);
    }
    if (buildFileValue == null) {
      return null;
    }
    Preconditions.checkState(buildFileValue.exists(),
        "Package lookup succeeded but BUILD file doesn't exist");
    Path buildFilePath = buildFileRootedPath.asPath();

    String replacementContents = null;
    if (packageName.equals(DEFAULTS_PACKAGE_NAME)) {
      replacementContents = PrecomputedValue.DEFAULTS_PACKAGE_CONTENTS.get(env);
      if (replacementContents == null) {
        return null;
      }
    }

    RuleVisibility defaultVisibility = PrecomputedValue.DEFAULT_VISIBILITY.get(env);
    if (defaultVisibility == null) {
      return null;
    }

    ASTFileLookupValue astLookupValue = null;
    SkyKey astLookupKey = null;
    try {
      astLookupKey = ASTFileLookupValue.key(PRELUDE_FILE_FRAGMENT);
    } catch (ASTLookupInputException e) {
      // There's a static check ensuring that PRELUDE_FILE_FRAGMENT is relative.
      throw new IllegalStateException(e);
    }
    try {
      astLookupValue = (ASTFileLookupValue) env.getValueOrThrow(astLookupKey,
          ErrorReadingSkylarkExtensionException.class, InconsistentFilesystemException.class);
    } catch (ErrorReadingSkylarkExtensionException | InconsistentFilesystemException e) {
      throw new PackageFunctionException(new BadPreludeFileException(packageName, e.getMessage()),
          Transience.PERSISTENT);
    }
    if (astLookupValue == null) {
      return null;
    }
    List<Statement> preludeStatements = astLookupValue.getAST() == null
        ? ImmutableList.<Statement>of() : astLookupValue.getAST().getStatements();

    // Load the BUILD file AST and handle Skylark dependencies. This way BUILD files are
    // only loaded twice if there are unavailable Skylark or package dependencies or an
    // IOException occurs. Note that the BUILD files are still parsed two times.
    ParserInputSource inputSource;
    try {
      if (showLoadingProgress.get() && !packageFunctionCache.containsKey(packageId)) {
        // TODO(bazel-team): don't duplicate the loading message if there are unavailable
        // Skylark dependencies.
        reporter.handle(Event.progress("Loading package: " + packageName));
      }
      inputSource = ParserInputSource.create(buildFilePath);
    } catch (IOException e) {
      env.getListener().handle(Event.error(Location.fromFile(buildFilePath), e.getMessage()));
      // Note that we did this work, so we should conservatively report this error as transient.
      throw new PackageFunctionException(new BuildFileContainsErrorsException(
          packageName, e.getMessage()), Transience.TRANSIENT);
    }
    SkylarkImportResult importResult = fetchImportsFromBuildFile(buildFilePath, buildFileFragment,
        packageId.getRepository(), preludeStatements, inputSource, packageName, env);
    if (importResult == null) {
      return null;
    }

    Package.LegacyBuilder legacyPkgBuilder = loadPackage(externalPkg, inputSource,
        replacementContents, packageId, buildFilePath, defaultVisibility, preludeStatements,
        importResult);
    legacyPkgBuilder.buildPartial();
    try {
      handleLabelsCrossingSubpackagesAndPropagateInconsistentFilesystemExceptions(
          packageLookupValue.getRoot(), packageId, legacyPkgBuilder, env);
    } catch (InternalInconsistentFilesystemException e) {
      packageFunctionCache.remove(packageId);
      throw new PackageFunctionException(e,
          e.isTransient() ? Transience.TRANSIENT : Transience.PERSISTENT);
    }
    if (env.valuesMissing()) {
      // The package we just loaded will be in the {@code packageFunctionCache} next when this
      // SkyFunction is called again.
      return null;
    }
    Collection<Pair<String, Boolean>> globPatterns = legacyPkgBuilder.getGlobPatterns();
    Map<Label, Path> subincludes = legacyPkgBuilder.getSubincludes();
    Package pkg = legacyPkgBuilder.finishBuild();
    Event.replayEventsOn(env.getListener(), pkg.getEvents());
    boolean packageShouldBeConsideredInError = pkg.containsErrors();
    try {
      packageShouldBeConsideredInError =
          markDependenciesAndPropagateInconsistentFilesystemExceptions(pkg, env,
              globPatterns, subincludes);
    } catch (InternalInconsistentFilesystemException e) {
      packageFunctionCache.remove(packageId);
      throw new PackageFunctionException(e,
          e.isTransient() ? Transience.TRANSIENT : Transience.PERSISTENT);
    }

    if (env.valuesMissing()) {
      return null;
    }
    // We know this SkyFunction will not be called again, so we can remove the cache entry.
    packageFunctionCache.remove(packageId);

    if (packageShouldBeConsideredInError) {
      throw new PackageFunctionException(new BuildFileContainsErrorsException(pkg,
          "Package '" + packageName + "' contains errors"),
          pkg.containsTemporaryErrors() ? Transience.TRANSIENT : Transience.PERSISTENT);
    }
    return new PackageValue(pkg);
  }

  private SkylarkImportResult fetchImportsFromBuildFile(Path buildFilePath,
      PathFragment buildFileFragment, RepositoryName repo,
      List<Statement> preludeStatements, ParserInputSource inputSource,
      String packageName, Environment env) throws PackageFunctionException {
    StoredEventHandler eventHandler = new StoredEventHandler();
    BuildFileAST buildFileAST = BuildFileAST.parseBuildFile(
          inputSource, preludeStatements, eventHandler, null, true);

    if (eventHandler.hasErrors()) {
      // In case of Python preprocessing, errors have already been reported (see checkSyntax).
      // In other cases, errors will be reported later.
      // TODO(bazel-team): maybe we could get rid of checkSyntax and always report errors here?
      return new SkylarkImportResult(
          ImmutableMap.<PathFragment, SkylarkEnvironment>of(),
          ImmutableList.<Label>of());
    }

    ImmutableCollection<PathFragment> imports = buildFileAST.getImports();
    Map<PathFragment, SkylarkEnvironment> importMap = new HashMap<>();
    ImmutableList.Builder<SkylarkFileDependency> fileDependencies = ImmutableList.builder();
    try {
      for (PathFragment importFile : imports) {
        SkyKey importsLookupKey =
            SkylarkImportLookupValue.key(repo, buildFileFragment, importFile);
        SkylarkImportLookupValue importLookupValue = (SkylarkImportLookupValue)
            env.getValueOrThrow(importsLookupKey, SkylarkImportFailedException.class,
                InconsistentFilesystemException.class, ASTLookupInputException.class,
                BuildFileNotFoundException.class);
        if (importLookupValue != null) {
          importMap.put(importFile, importLookupValue.getImportedEnvironment());
          fileDependencies.add(importLookupValue.getDependency());
        }
      }
    } catch (SkylarkImportFailedException e) {
      env.getListener().handle(Event.error(Location.fromFile(buildFilePath), e.getMessage()));
      throw new PackageFunctionException(new BuildFileContainsErrorsException(packageName,
          e.getMessage()), Transience.PERSISTENT);
    } catch (InconsistentFilesystemException e) {
      throw new PackageFunctionException(new InternalInconsistentFilesystemException(packageName,
          e), Transience.PERSISTENT);
    } catch (ASTLookupInputException e) {
      // The load syntax is bad in the BUILD file so BuildFileContainsErrorsException is OK.
      throw new PackageFunctionException(new BuildFileContainsErrorsException(packageName,
          e.getMessage()), Transience.PERSISTENT);
    } catch (BuildFileNotFoundException e) {
      throw new PackageFunctionException(e, Transience.PERSISTENT);
    }
    if (env.valuesMissing()) {
      // There are unavailable Skylark dependencies.
      return null;
    }
    return new SkylarkImportResult(importMap, transitiveClosureOfLabels(fileDependencies.build()));
  }

  private ImmutableList<Label> transitiveClosureOfLabels(
      ImmutableList<SkylarkFileDependency> immediateDeps) {
    Set<Label> transitiveClosure = Sets.newHashSet();
    transitiveClosureOfLabels(immediateDeps, transitiveClosure);
    return ImmutableList.copyOf(transitiveClosure);
  }

  private void transitiveClosureOfLabels(
      ImmutableList<SkylarkFileDependency> immediateDeps, Set<Label> transitiveClosure) {
    for (SkylarkFileDependency dep : immediateDeps) {
      if (transitiveClosure.add(dep.getLabel())) {
        transitiveClosureOfLabels(dep.getDependencies(), transitiveClosure);
      }
    }
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private static void handleLabelsCrossingSubpackagesAndPropagateInconsistentFilesystemExceptions(
      Path pkgRoot, PackageIdentifier pkgId, Package.LegacyBuilder pkgBuilder, Environment env)
          throws InternalInconsistentFilesystemException {
    Set<SkyKey> containingPkgLookupKeys = Sets.newHashSet();
    Map<Target, SkyKey> targetToKey = new HashMap<>();
    for (Target target : pkgBuilder.getTargets()) {
      PathFragment dir = target.getLabel().toPathFragment().getParentDirectory();
      PackageIdentifier dirId = new PackageIdentifier(pkgId.getRepository(), dir);
      if (dir.equals(pkgId.getPackageFragment())) {
        continue;
      }
      SkyKey key = ContainingPackageLookupValue.key(dirId);
      targetToKey.put(target, key);
      containingPkgLookupKeys.add(key);
    }
    Map<Label, SkyKey> subincludeToKey = new HashMap<>();
    for (Label subincludeLabel : pkgBuilder.getSubincludeLabels()) {
      PathFragment dir = subincludeLabel.toPathFragment().getParentDirectory();
      PackageIdentifier dirId = new PackageIdentifier(pkgId.getRepository(), dir);
      if (dir.equals(pkgId.getPackageFragment())) {
        continue;
      }
      SkyKey key = ContainingPackageLookupValue.key(dirId);
      subincludeToKey.put(subincludeLabel, key);
      containingPkgLookupKeys.add(ContainingPackageLookupValue.key(dirId));
    }
    Map<SkyKey, ValueOrException3<BuildFileNotFoundException, InconsistentFilesystemException,
        FileSymlinkCycleException>> containingPkgLookupValues = env.getValuesOrThrow(
            containingPkgLookupKeys, BuildFileNotFoundException.class,
            InconsistentFilesystemException.class, FileSymlinkCycleException.class);
    if (env.valuesMissing()) {
      return;
    }
    for (Target target : ImmutableSet.copyOf(pkgBuilder.getTargets())) {
      SkyKey key = targetToKey.get(target);
      if (!containingPkgLookupValues.containsKey(key)) {
        continue;
      }
      ContainingPackageLookupValue containingPackageLookupValue =
          getContainingPkgLookupValueAndPropagateInconsistentFilesystemExceptions(
              pkgId.getPackageFragment().getPathString(), containingPkgLookupValues.get(key), env);
      if (maybeAddEventAboutLabelCrossingSubpackage(pkgBuilder, pkgRoot, target.getLabel(),
          target.getLocation(), containingPackageLookupValue)) {
        pkgBuilder.removeTarget(target);
        pkgBuilder.setContainsErrors();
      }
    }
    for (Label subincludeLabel : pkgBuilder.getSubincludeLabels()) {
      SkyKey key = subincludeToKey.get(subincludeLabel);
      if (!containingPkgLookupValues.containsKey(key)) {
        continue;
      }
      ContainingPackageLookupValue containingPackageLookupValue =
          getContainingPkgLookupValueAndPropagateInconsistentFilesystemExceptions(
              pkgId.getPackageFragment().getPathString(), containingPkgLookupValues.get(key), env);
      if (maybeAddEventAboutLabelCrossingSubpackage(pkgBuilder, pkgRoot, subincludeLabel,
          /*location=*/null, containingPackageLookupValue)) {
        pkgBuilder.setContainsErrors();
      }
    }
  }

  @Nullable
  private static ContainingPackageLookupValue
  getContainingPkgLookupValueAndPropagateInconsistentFilesystemExceptions(String packageName,
      ValueOrException3<BuildFileNotFoundException, InconsistentFilesystemException,
      FileSymlinkCycleException> containingPkgLookupValueOrException, Environment env)
          throws InternalInconsistentFilesystemException {
    try {
      return (ContainingPackageLookupValue) containingPkgLookupValueOrException.get();
    } catch (BuildFileNotFoundException | FileSymlinkCycleException e) {
      env.getListener().handle(Event.error(null, e.getMessage()));
      return null;
    } catch (InconsistentFilesystemException e) {
      throw new InternalInconsistentFilesystemException(packageName, e);
    }
  }

  private static boolean maybeAddEventAboutLabelCrossingSubpackage(
      Package.LegacyBuilder pkgBuilder, Path pkgRoot, Label label, @Nullable Location location,
      @Nullable ContainingPackageLookupValue containingPkgLookupValue) {
    if (containingPkgLookupValue == null) {
      return true;
    }
    if (!containingPkgLookupValue.hasContainingPackage()) {
      // The missing package here is a problem, but it's not an error from the perspective of
      // PackageFunction.
      return false;
    }
    PackageIdentifier containingPkg = containingPkgLookupValue.getContainingPackageName();
    if (containingPkg.equals(label.getPackageIdentifier())) {
      // The label does not cross a subpackage boundary.
      return false;
    }
    if (!containingPkg.getPackageFragment().startsWith(label.getPackageFragment())) {
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
          containingPkg.getPackageFragment().segmentCount()
              - label.getPackageFragment().segmentCount(),
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
  private Package.LegacyBuilder loadPackage(Package externalPkg,
      ParserInputSource inputSource, @Nullable String replacementContents,
      PackageIdentifier packageId, Path buildFilePath, RuleVisibility defaultVisibility,
      List<Statement> preludeStatements, SkylarkImportResult importResult)
          throws InterruptedException {
    ParserInputSource replacementSource = replacementContents == null ? null
        : ParserInputSource.create(replacementContents, buildFilePath);
    Package.LegacyBuilder pkgBuilder = packageFunctionCache.get(packageId);
    if (pkgBuilder == null) {
      Clock clock = new JavaClock();
      long startTime = clock.nanoTime();
      profiler.startTask(ProfilerTask.CREATE_PACKAGE, packageId.toString());
      try {
        Globber globber = packageFactory.createLegacyGlobber(buildFilePath.getParentDirectory(),
            packageId, packageLocator);
        StoredEventHandler localReporter = new StoredEventHandler();
        Preprocessor.Result preprocessingResult = replacementSource == null
            ? packageFactory.preprocess(packageId, buildFilePath, inputSource, globber,
                localReporter)
                : Preprocessor.Result.noPreprocessing(replacementSource);
        pkgBuilder = packageFactory.createPackageFromPreprocessingResult(externalPkg, packageId,
            buildFilePath, preprocessingResult, localReporter.getEvents(), preludeStatements,
            importResult.importMap, importResult.fileDependencies, packageLocator,
            defaultVisibility, globber);
        if (eventBus.get() != null) {
          eventBus.get().post(new PackageLoadedEvent(packageId.toString(),
              (clock.nanoTime() - startTime) / (1000 * 1000),
              // It's impossible to tell if the package was loaded before, so we always pass false.
              /*reloading=*/false,
              // This isn't completely correct since we may encounter errors later (e.g. filesystem
              // inconsistencies)
              !pkgBuilder.containsErrors()));
        }
        numPackagesLoaded.incrementAndGet();
        packageFunctionCache.put(packageId, pkgBuilder);
      } finally {
        profiler.completeTask(ProfilerTask.CREATE_PACKAGE);
      }
    }
    return pkgBuilder;
  }

  private static class InternalInconsistentFilesystemException extends NoSuchPackageException {
    private boolean isTransient;

    /**
     * Used to represent a filesystem inconsistency discovered outside the
     * {@link PackageFunction}.
     */
    public InternalInconsistentFilesystemException(String packageName,
        InconsistentFilesystemException e) {
      super(packageName, e.getMessage(), e);
      // This is not a transient error from the perspective of the PackageFunction.
      this.isTransient = false;
    }

    /** Used to represent a filesystem inconsistency discovered by the {@link PackageFunction}. */
    public InternalInconsistentFilesystemException(String packageName,
        String inconsistencyMessage) {
      this(packageName, new InconsistentFilesystemException(inconsistencyMessage));
      this.isTransient = true;
    }

    public boolean isTransient() {
      return isTransient;
    }
  }

  private static class BadWorkspaceFileException extends NoSuchPackageException {
    private BadWorkspaceFileException(String message) {
      super("external", "Error encountered while dealing with the WORKSPACE file: " + message);
    }
  }

  private static class BadPreludeFileException extends NoSuchPackageException {
    private BadPreludeFileException(String packageName, String message) {
      super(packageName, "Error encountered while reading the prelude file: " + message);
    }
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link PackageFunction#compute}.
   */
  private static class PackageFunctionException extends SkyFunctionException {
    public PackageFunctionException(NoSuchPackageException e, Transience transience) {
      super(e, transience);
    }
  }

  /** A simple value class to store the result of the Skylark imports.*/
  private static final class SkylarkImportResult {
    private final Map<PathFragment, SkylarkEnvironment> importMap;
    private final ImmutableList<Label> fileDependencies;
    private SkylarkImportResult(Map<PathFragment, SkylarkEnvironment> importMap,
        ImmutableList<Label> fileDependencies) {
      this.importMap = importMap;
      this.fileDependencies = fileDependencies;
    }
  }
}
