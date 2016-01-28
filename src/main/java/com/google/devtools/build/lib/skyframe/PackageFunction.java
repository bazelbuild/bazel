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
package com.google.devtools.build.lib.skyframe;

import com.google.common.cache.Cache;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.CachingPackageLocator;
import com.google.devtools.build.lib.packages.InvalidPackageNameException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Package.LegacyBuilder;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageFactory.Globber;
import com.google.devtools.build.lib.packages.Preprocessor;
import com.google.devtools.build.lib.packages.Preprocessor.AstAfterPreprocessing;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.skyframe.GlobValue.InvalidGlobPatternException;
import com.google.devtools.build.lib.skyframe.SkylarkImportLookupFunction.SkylarkImportFailedException;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Environment.Extension;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.syntax.SkylarkImport;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException2;
import com.google.devtools.build.skyframe.ValueOrException3;
import com.google.devtools.build.skyframe.ValueOrException4;

import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import javax.annotation.Nullable;

/**
 * A SkyFunction for {@link PackageValue}s.
 */
public class PackageFunction implements SkyFunction {

  private final PackageFactory packageFactory;
  private final CachingPackageLocator packageLocator;
  private final Cache<PackageIdentifier, Package.LegacyBuilder> packageFunctionCache;
  private final Cache<PackageIdentifier, Preprocessor.AstAfterPreprocessing> astCache;
  private final AtomicBoolean showLoadingProgress;
  private final AtomicInteger numPackagesLoaded;
  private final Profiler profiler = Profiler.instance();
  private final Label preludeLabel;

  // Not final only for testing.
  @Nullable private SkylarkImportLookupFunction skylarkImportLookupFunctionForInlining;

  static final PathFragment DEFAULTS_PACKAGE_NAME = new PathFragment("tools/defaults");

  public PackageFunction(
      PackageFactory packageFactory,
      CachingPackageLocator pkgLocator,
      AtomicBoolean showLoadingProgress,
      Cache<PackageIdentifier, LegacyBuilder> packageFunctionCache,
      Cache<PackageIdentifier, AstAfterPreprocessing> astCache,
      AtomicInteger numPackagesLoaded,
      @Nullable SkylarkImportLookupFunction skylarkImportLookupFunctionForInlining) {
    this.skylarkImportLookupFunctionForInlining = skylarkImportLookupFunctionForInlining;
    // Can be null in tests.
    this.preludeLabel = packageFactory == null
        ? null
        : packageFactory.getRuleClassProvider().getPreludeLabel();
    this.packageFactory = packageFactory;
    this.packageLocator = pkgLocator;
    this.showLoadingProgress = showLoadingProgress;
    this.packageFunctionCache = packageFunctionCache;
    this.astCache = astCache;
    this.numPackagesLoaded = numPackagesLoaded;
  }

  public void setSkylarkImportLookupFunctionForInliningForTesting(
      SkylarkImportLookupFunction skylarkImportLookupFunctionForInlining) {
    this.skylarkImportLookupFunctionForInlining = skylarkImportLookupFunctionForInlining;
  }

  private static void maybeThrowFilesystemInconsistency(PackageIdentifier packageIdentifier,
      Exception skyframeException, boolean packageWasInError)
          throws InternalInconsistentFilesystemException {
    if (!packageWasInError) {
      throw new InternalInconsistentFilesystemException(packageIdentifier, "Encountered error '"
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
  getPackageLookupDepsAndPropagateInconsistentFilesystemExceptions(
      PackageIdentifier packageIdentifier,
      Iterable<SkyKey> depKeys, Environment env, boolean packageWasInError)
          throws InternalInconsistentFilesystemException {
    Preconditions.checkState(
        Iterables.all(depKeys, SkyFunctions.isSkyFunction(SkyFunctions.PACKAGE_LOOKUP)), depKeys);
    boolean packageShouldBeInError = packageWasInError;
    ImmutableMap.Builder<PathFragment, PackageLookupValue> builder = ImmutableMap.builder();
    for (Map.Entry<SkyKey, ValueOrException3<BuildFileNotFoundException,
        InconsistentFilesystemException, FileSymlinkException>> entry :
            env.getValuesOrThrow(depKeys, BuildFileNotFoundException.class,
                InconsistentFilesystemException.class,
                FileSymlinkException.class).entrySet()) {
      PathFragment pkgName = ((PackageIdentifier) entry.getKey().argument()).getPackageFragment();
      try {
        PackageLookupValue value = (PackageLookupValue) entry.getValue().get();
        if (value != null) {
          builder.put(pkgName, value);
        }
      } catch (BuildFileNotFoundException e) {
        maybeThrowFilesystemInconsistency(packageIdentifier, e, packageWasInError);
      } catch (InconsistentFilesystemException e) {
        throw new InternalInconsistentFilesystemException(packageIdentifier, e);
      } catch (FileSymlinkException e) {
        // Legacy doesn't detect symlink cycles.
        packageShouldBeInError = true;
      }
    }
    return Pair.of(builder.build(), packageShouldBeInError);
  }

  private static boolean markFileDepsAndPropagateInconsistentFilesystemExceptions(
      PackageIdentifier packageIdentifier, Iterable<SkyKey> depKeys, Environment env,
      boolean packageWasInError) throws InternalInconsistentFilesystemException {
    Preconditions.checkState(
        Iterables.all(depKeys, SkyFunctions.isSkyFunction(SkyFunctions.FILE)), depKeys);
    boolean packageShouldBeInError = packageWasInError;
    for (Map.Entry<SkyKey, ValueOrException3<IOException, FileSymlinkException,
        InconsistentFilesystemException>> entry : env.getValuesOrThrow(depKeys, IOException.class,
            FileSymlinkException.class, InconsistentFilesystemException.class).entrySet()) {
      try {
        entry.getValue().get();
      } catch (IOException e) {
        maybeThrowFilesystemInconsistency(packageIdentifier, e, packageWasInError);
      } catch (FileSymlinkException e) {
        // Legacy doesn't detect symlink cycles.
        packageShouldBeInError = true;
      } catch (InconsistentFilesystemException e) {
        throw new InternalInconsistentFilesystemException(packageIdentifier, e);
      }
    }
    return packageShouldBeInError;
  }

  private static boolean markGlobDepsAndPropagateInconsistentFilesystemExceptions(
      PackageIdentifier packageIdentifier, Iterable<SkyKey> depKeys, Environment env,
      boolean packageWasInError) throws InternalInconsistentFilesystemException {
    Preconditions.checkState(
        Iterables.all(depKeys, SkyFunctions.isSkyFunction(SkyFunctions.GLOB)), depKeys);
    boolean packageShouldBeInError = packageWasInError;
    for (Map.Entry<SkyKey, ValueOrException4<IOException, BuildFileNotFoundException,
        FileSymlinkException, InconsistentFilesystemException>> entry :
        env.getValuesOrThrow(depKeys, IOException.class, BuildFileNotFoundException.class,
            FileSymlinkException.class, InconsistentFilesystemException.class).entrySet()) {
      try {
        entry.getValue().get();
      } catch (IOException | BuildFileNotFoundException e) {
        maybeThrowFilesystemInconsistency(packageIdentifier, e, packageWasInError);
      } catch (FileSymlinkException e) {
        // Legacy doesn't detect symlink cycles.
        packageShouldBeInError = true;
      } catch (InconsistentFilesystemException e) {
        throw new InternalInconsistentFilesystemException(packageIdentifier, e);
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
      Environment env,
      Collection<Pair<String, Boolean>> globPatterns,
      Map<Label, Path> subincludes,
      PackageIdentifier packageIdentifier,
      boolean containsErrors)
      throws InternalInconsistentFilesystemException {
    boolean packageShouldBeInError = containsErrors;

    // TODO(bazel-team): This means that many packages will have to be preprocessed twice. Ouch!
    // We need a better continuation mechanism to avoid repeating work. [skyframe-loading]

    // TODO(bazel-team): It would be preferable to perform I/O from the package preprocessor via
    // Skyframe rather than add (potentially incomplete) dependencies after the fact.
    // [skyframe-loading]

    Set<SkyKey> subincludePackageLookupDepKeys = Sets.newHashSet();
    for (Label label : subincludes.keySet()) {
      // Declare a dependency on the package lookup for the package giving access to the label.
      subincludePackageLookupDepKeys.add(PackageLookupValue.key(label.getPackageIdentifier()));
    }
    Pair<? extends Map<PathFragment, PackageLookupValue>, Boolean> subincludePackageLookupResult =
        getPackageLookupDepsAndPropagateInconsistentFilesystemExceptions(
            packageIdentifier, subincludePackageLookupDepKeys, env, containsErrors);
    Map<PathFragment, PackageLookupValue> subincludePackageLookupDeps =
        subincludePackageLookupResult.getFirst();
    packageShouldBeInError |= subincludePackageLookupResult.getSecond();
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
        if (subincludeFilePath != null && !subincludePackageLookupValue.packageExists()) {
          // Legacy blaze puts a non-null path when only when the package does indeed exist.
          throw new InternalInconsistentFilesystemException(
              packageIdentifier,
              String.format(
                  "Unexpected package in %s. Was it modified during the build?",
                  subincludeFilePath));
        }
        if (subincludePackageLookupValue.packageExists()) {
          // Sanity check for consistency of Skyframe and legacy blaze.
          Path subincludeFilePathSkyframe =
              subincludePackageLookupValue.getRoot().getRelative(label.toPathFragment());
          if (subincludeFilePath != null
              && !subincludeFilePathSkyframe.equals(subincludeFilePath)) {
            throw new InternalInconsistentFilesystemException(
                packageIdentifier,
                String.format(
                    "Inconsistent package location for %s: '%s' vs '%s'. "
                        + "Was the source tree modified during the build?",
                    label.getPackageFragment(),
                    subincludeFilePathSkyframe,
                    subincludeFilePath));
          }
          // The actual file may be under a different package root than the package being
          // constructed.
          SkyKey subincludeSkyKey =
              FileValue.key(
                  RootedPath.toRootedPath(
                      subincludePackageLookupValue.getRoot(),
                      label.getPackageFragment().getRelative(label.getName())));
          subincludeFileDepKeys.add(subincludeSkyKey);
        }
      }
    }
    packageShouldBeInError |=
        markFileDepsAndPropagateInconsistentFilesystemExceptions(
            packageIdentifier, subincludeFileDepKeys, env, containsErrors);

    // TODO(bazel-team): In the long term, we want to actually resolve the glob patterns within
    // Skyframe. For now, just logging the glob requests provides correct incrementality and
    // adequate performance.
    List<SkyKey> globDepKeys = Lists.newArrayList();
    for (Pair<String, Boolean> globPattern : globPatterns) {
      String pattern = globPattern.getFirst();
      boolean excludeDirs = globPattern.getSecond();
      SkyKey globSkyKey;
      try {
        globSkyKey =
            GlobValue.key(packageIdentifier, pattern, excludeDirs, PathFragment.EMPTY_FRAGMENT);
      } catch (InvalidGlobPatternException e) {
        // Globs that make it to pkg.getGlobPatterns() should already be filtered for errors.
        throw new IllegalStateException(e);
      }
      globDepKeys.add(globSkyKey);
    }
    packageShouldBeInError |=
        markGlobDepsAndPropagateInconsistentFilesystemExceptions(
            packageIdentifier, globDepKeys, env, containsErrors);
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
      FileSymlinkException.class, InconsistentFilesystemException.class,
      EvalException.class, SkylarkImportFailedException.class);
    } catch (IOException | FileSymlinkException | InconsistentFilesystemException
          | EvalException | SkylarkImportFailedException e) {
      throw new PackageFunctionException(new BadWorkspaceFileException(e.getMessage()),
          Transience.PERSISTENT);
    }
    if (workspace == null) {
      return null;
    }

    Package pkg = workspace.getPackage();
    Event.replayEventsOn(env.getListener(), pkg.getEvents());

    return new PackageValue(pkg);
  }

  @Override
  public SkyValue compute(SkyKey key, Environment env) throws PackageFunctionException,
      InterruptedException {
    PackageIdentifier packageId = (PackageIdentifier) key.argument();
    PathFragment packageNameFragment = packageId.getPackageFragment();

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
          new InternalInconsistentFilesystemException(packageId, e), Transience.PERSISTENT);
    }
    if (packageLookupValue == null) {
      return null;
    }

    if (!packageLookupValue.packageExists()) {
      switch (packageLookupValue.getErrorReason()) {
        case NO_BUILD_FILE:
        case DELETED_PACKAGE:
          throw new PackageFunctionException(new BuildFileNotFoundException(packageId,
              packageLookupValue.getErrorMsg()), Transience.PERSISTENT);
        case INVALID_PACKAGE_NAME:
          throw new PackageFunctionException(new InvalidPackageNameException(packageId,
              packageLookupValue.getErrorMsg()), Transience.PERSISTENT);
        default:
          // We should never get here.
          throw new IllegalStateException();
      }
    }

    if (packageId.equals(Label.EXTERNAL_PACKAGE_IDENTIFIER)) {
      return getExternalPackage(env, packageLookupValue.getRoot());
    }
    SkyKey externalPackageKey = PackageValue.key(Label.EXTERNAL_PACKAGE_IDENTIFIER);
    PackageValue externalPackage = (PackageValue) env.getValue(externalPackageKey);
    if (externalPackage == null) {
      return null;
    }
    Package externalPkg = externalPackage.getPackage();
    if (externalPkg.containsErrors()) {
      throw new PackageFunctionException(
          new BuildFileContainsErrorsException(Label.EXTERNAL_PACKAGE_IDENTIFIER),
          Transience.PERSISTENT);
    }

    PathFragment buildFileFragment = packageNameFragment.getChild("BUILD");
    RootedPath buildFileRootedPath = RootedPath.toRootedPath(packageLookupValue.getRoot(),
        buildFileFragment);
    FileValue buildFileValue = null;
    Path buildFilePath = buildFileRootedPath.asPath();
    String replacementContents = null;

    if (!isDefaultsPackage(packageId)) {
      buildFileValue = getBuildFileValue(env, buildFileRootedPath);
      if (buildFileValue == null) {
        return null;
      }
    } else {
      replacementContents = PrecomputedValue.DEFAULTS_PACKAGE_CONTENTS.get(env);
      if (replacementContents == null) {
        return null;
      }
    }

    RuleVisibility defaultVisibility = PrecomputedValue.DEFAULT_VISIBILITY.get(env);
    if (defaultVisibility == null) {
      return null;
    }

    SkyKey astLookupKey = ASTFileLookupValue.key(preludeLabel);
    ASTFileLookupValue astLookupValue = null;
    try {
      astLookupValue = (ASTFileLookupValue) env.getValueOrThrow(astLookupKey,
          ErrorReadingSkylarkExtensionException.class, InconsistentFilesystemException.class);
    } catch (ErrorReadingSkylarkExtensionException | InconsistentFilesystemException e) {
      throw new PackageFunctionException(
          new BadPreludeFileException(packageId, e.getMessage()), Transience.PERSISTENT);
    }
    if (astLookupValue == null) {
      return null;
    }
    // The prelude file doesn't have to exist. If not, we substitute an empty statement list.
    List<Statement> preludeStatements =
        astLookupValue.lookupSuccessful()
            ? astLookupValue.getAST().getStatements() : ImmutableList.<Statement>of();
    Package.LegacyBuilder legacyPkgBuilder =
        loadPackage(
            externalPkg,
            replacementContents,
            packageId,
            buildFilePath,
            buildFileValue,
            defaultVisibility,
            preludeStatements,
            env);
    if (legacyPkgBuilder == null) {
      return null;
    }
    legacyPkgBuilder.buildPartial();
    try {
      // Since the Skyframe dependencies we request below in
      // markDependenciesAndPropagateInconsistentFilesystemExceptions are requested independently of
      // the ones requested here in
      // handleLabelsCrossingSubpackagesAndPropagateInconsistentFilesystemExceptions, we don't
      // bother checking for missing values and instead piggyback on the env.missingValues() call
      // for the former. This avoids a Skyframe restart.
      handleLabelsCrossingSubpackagesAndPropagateInconsistentFilesystemExceptions(
          packageLookupValue.getRoot(), packageId, legacyPkgBuilder, env);
    } catch (InternalInconsistentFilesystemException e) {
      packageFunctionCache.invalidate(packageId);
      throw new PackageFunctionException(e,
          e.isTransient() ? Transience.TRANSIENT : Transience.PERSISTENT);
    }
    Collection<Pair<String, Boolean>> globPatterns = legacyPkgBuilder.getGlobPatterns();
    Map<Label, Path> subincludes = legacyPkgBuilder.getSubincludes();
    boolean packageShouldBeConsideredInError;
    try {
      packageShouldBeConsideredInError =
          markDependenciesAndPropagateInconsistentFilesystemExceptions(
              env, globPatterns, subincludes, packageId, legacyPkgBuilder.containsErrors());
    } catch (InternalInconsistentFilesystemException e) {
      packageFunctionCache.invalidate(packageId);
      throw new PackageFunctionException(e,
          e.isTransient() ? Transience.TRANSIENT : Transience.PERSISTENT);
    }
    if (env.valuesMissing()) {
      return null;
    }

    Event.replayEventsOn(env.getListener(), legacyPkgBuilder.getEvents());

    if (packageShouldBeConsideredInError) {
      legacyPkgBuilder.setContainsErrors();
    }
    Package pkg = legacyPkgBuilder.finishBuild();

    // We know this SkyFunction will not be called again, so we can remove the cache entry.
    packageFunctionCache.invalidate(packageId);

    return new PackageValue(pkg);
  }

  private FileValue getBuildFileValue(Environment env, RootedPath buildFileRootedPath) {
    FileValue buildFileValue;
    try {
      buildFileValue = (FileValue) env.getValueOrThrow(FileValue.key(buildFileRootedPath),
          IOException.class, FileSymlinkException.class,
          InconsistentFilesystemException.class);
    } catch (IOException | FileSymlinkException | InconsistentFilesystemException e) {
      throw new IllegalStateException("Package lookup succeeded but encountered error when "
          + "getting FileValue for BUILD file directly.", e);
    }
    if (buildFileValue == null) {
      return null;
    }
    Preconditions.checkState(buildFileValue.exists(),
        "Package lookup succeeded but BUILD file doesn't exist");
    return buildFileValue;
  }

  @Nullable
  private SkylarkImportResult discoverSkylarkImports(
      Path buildFilePath,
      PackageIdentifier packageId,
      AstAfterPreprocessing astAfterPreprocessing,
      Environment env)
      throws PackageFunctionException, InterruptedException {
    SkylarkImportResult importResult;
    if (astAfterPreprocessing.containsAstParsingErrors) {
      importResult =
          new SkylarkImportResult(
              ImmutableMap.<String, Extension>of(),
              ImmutableList.<Label>of());
    } else {
      importResult =
          fetchImportsFromBuildFile(
              buildFilePath,
              packageId,
              astAfterPreprocessing.ast,
              env,
              skylarkImportLookupFunctionForInlining);
    }

    return importResult;
  }

  /**
   * Fetch the skylark loads for this BUILD file. If any of them haven't been computed yet,
   * returns null.
   */
  @Nullable
  static SkylarkImportResult fetchImportsFromBuildFile(
      Path buildFilePath,
      PackageIdentifier packageId,
      BuildFileAST buildFileAST,
      Environment env,
      SkylarkImportLookupFunction skylarkImportLookupFunctionForInlining)
      throws PackageFunctionException, InterruptedException {
    ImmutableList<SkylarkImport> imports = buildFileAST.getImports();
    Map<String, Extension> importMap = Maps.newHashMapWithExpectedSize(imports.size());
    ImmutableList.Builder<SkylarkFileDependency> fileDependencies = ImmutableList.builder();
    ImmutableMap<String, Label> importPathMap;

    // Find the labels corresponding to the load statements.
    Label labelForCurrBuildFile;
    try {
      labelForCurrBuildFile = Label.create(packageId, "BUILD");
    } catch (LabelSyntaxException e) {
      // Shouldn't happen; the Label is well-formed by construction.
      throw new IllegalStateException(e);
    }
    try {
      importPathMap = SkylarkImportLookupFunction.findLabelsForLoadStatements(
          imports, labelForCurrBuildFile, env);
      if (importPathMap == null) {
        return null;
      }
    } catch (SkylarkImportFailedException e) {
      throw new PackageFunctionException(
          new BuildFileContainsErrorsException(packageId, e.getMessage()), Transience.PERSISTENT);
    }

    // Look up and load the imports.
    ImmutableCollection<Label> importLabels = importPathMap.values();
    List<SkyKey> importLookupKeys = Lists.newArrayListWithExpectedSize(importLabels.size());
    boolean inWorkspace = buildFilePath.getBaseName().endsWith("WORKSPACE");
    for (Label importLabel : importLabels) {
      importLookupKeys.add(SkylarkImportLookupValue.key(importLabel, inWorkspace));
    }
    Map<SkyKey, SkyValue> skylarkImportMap = Maps.newHashMapWithExpectedSize(importPathMap.size());
    boolean valuesMissing = false;

    try {
      if (skylarkImportLookupFunctionForInlining == null) {
        // Not inlining
        Map<SkyKey,
            ValueOrException2<
                SkylarkImportFailedException,
                InconsistentFilesystemException>> skylarkLookupResults = env.getValuesOrThrow(
                    importLookupKeys,
                    SkylarkImportFailedException.class,
                    InconsistentFilesystemException.class);
        valuesMissing = env.valuesMissing();
        for (Map.Entry<
              SkyKey,
              ValueOrException2<
                  SkylarkImportFailedException,
                  InconsistentFilesystemException>> entry : skylarkLookupResults.entrySet()) {
          // Fetching the value will raise any deferred exceptions
          skylarkImportMap.put(entry.getKey(), entry.getValue().get());
        }
      } else {
        // Inlining calls to SkylarkImportLookupFunction
        for (SkyKey importLookupKey : importLookupKeys) {
          SkyValue skyValue = skylarkImportLookupFunctionForInlining.computeWithInlineCalls(
              importLookupKey, env);
          if (skyValue == null) {
            Preconditions.checkState(
                env.valuesMissing(), "no skylark import value for %s", importLookupKey);
            // We continue making inline calls even if some requested values are missing, to
            // maximize the number of dependent (non-inlined) SkyFunctions that are requested, thus
            // avoiding a quadratic number of restarts.
            valuesMissing = true;
          } else {
            skylarkImportMap.put(importLookupKey, skyValue);
          }
        }

      }
    } catch (SkylarkImportFailedException e) {
      throw new PackageFunctionException(
          new BuildFileContainsErrorsException(packageId, e.getMessage()), Transience.PERSISTENT);
    } catch (InconsistentFilesystemException e) {
      throw new PackageFunctionException(
          new InternalInconsistentFilesystemException(packageId, e), Transience.PERSISTENT);
    }

    if (valuesMissing) {
      // Some imports are unavailable.
      return null;
    }

    // Process the loaded imports.
    for (Entry<String, Label> importEntry : importPathMap.entrySet()) {
      String importString = importEntry.getKey();
      Label importLabel = importEntry.getValue();
      SkyKey keyForLabel = SkylarkImportLookupValue.key(importLabel, inWorkspace);
      SkylarkImportLookupValue importLookupValue =
          (SkylarkImportLookupValue) skylarkImportMap.get(keyForLabel);
      importMap.put(importString, importLookupValue.getEnvironmentExtension());
      fileDependencies.add(importLookupValue.getDependency());
    }

    return new SkylarkImportResult(importMap, transitiveClosureOfLabels(fileDependencies.build()));
  }

  private static ImmutableList<Label> transitiveClosureOfLabels(
      ImmutableList<SkylarkFileDependency> immediateDeps) {
    Set<Label> transitiveClosure = Sets.newHashSet();
    transitiveClosureOfLabels(immediateDeps, transitiveClosure);
    return ImmutableList.copyOf(transitiveClosure);
  }

  private static void transitiveClosureOfLabels(
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
      PackageIdentifier dirId = PackageIdentifier.create(pkgId.getRepository(), dir);
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
      PackageIdentifier dirId = PackageIdentifier.create(pkgId.getRepository(), dir);
      if (dir.equals(pkgId.getPackageFragment())) {
        continue;
      }
      SkyKey key = ContainingPackageLookupValue.key(dirId);
      subincludeToKey.put(subincludeLabel, key);
      containingPkgLookupKeys.add(ContainingPackageLookupValue.key(dirId));
    }
    Map<SkyKey, ValueOrException3<BuildFileNotFoundException, InconsistentFilesystemException,
        FileSymlinkException>> containingPkgLookupValues = env.getValuesOrThrow(
            containingPkgLookupKeys, BuildFileNotFoundException.class,
            InconsistentFilesystemException.class, FileSymlinkException.class);
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
              pkgId, containingPkgLookupValues.get(key), env);
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
              pkgId, containingPkgLookupValues.get(key), env);
      if (maybeAddEventAboutLabelCrossingSubpackage(pkgBuilder, pkgRoot, subincludeLabel,
          /*location=*/null, containingPackageLookupValue)) {
        pkgBuilder.setContainsErrors();
      }
    }
  }

  @Nullable
  private static ContainingPackageLookupValue
  getContainingPkgLookupValueAndPropagateInconsistentFilesystemExceptions(
      PackageIdentifier packageIdentifier,
      ValueOrException3<BuildFileNotFoundException, InconsistentFilesystemException,
      FileSymlinkException> containingPkgLookupValueOrException, Environment env)
          throws InternalInconsistentFilesystemException {
    try {
      return (ContainingPackageLookupValue) containingPkgLookupValueOrException.get();
    } catch (BuildFileNotFoundException | FileSymlinkException e) {
      env.getListener().handle(Event.error(null, e.getMessage()));
      return null;
    } catch (InconsistentFilesystemException e) {
      throw new InternalInconsistentFilesystemException(packageIdentifier, e);
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
    if (!containingPkg.getPathFragment().startsWith(
        label.getPackageIdentifier().getPathFragment())) {
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
   *
   * <p>May return null if the computation has to be restarted.
   *
   * <p>Exactly one of {@code replacementContents} and {@code buildFileValue} will be
   * non-{@code null}. The former indicates that we have a faux BUILD file with the given contents
   * and the latter indicates that we have a legitimate BUILD file and should actually do
   * preprocessing.
   */
  @Nullable
  private Package.LegacyBuilder loadPackage(
      Package externalPkg,
      @Nullable String replacementContents,
      PackageIdentifier packageId,
      Path buildFilePath,
      @Nullable FileValue buildFileValue,
      RuleVisibility defaultVisibility,
      List<Statement> preludeStatements,
      Environment env)
      throws InterruptedException, PackageFunctionException {
    Package.LegacyBuilder pkgBuilder = packageFunctionCache.getIfPresent(packageId);
    if (pkgBuilder == null) {
      profiler.startTask(ProfilerTask.CREATE_PACKAGE, packageId.toString());
      try {
        AstAfterPreprocessing astAfterPreprocessing = astCache.getIfPresent(packageId);
        if (astAfterPreprocessing == null) {
          if (showLoadingProgress.get()) {
            env.getListener().handle(Event.progress("Loading package: " + packageId));
          }
          Globber globber = packageFactory.createLegacyGlobber(buildFilePath.getParentDirectory(),
              packageId, packageLocator);
          Preprocessor.Result preprocessingResult;
          if (replacementContents == null) {
            Preconditions.checkNotNull(buildFileValue, packageId);
            byte[] buildFileBytes;
            try {
              buildFileBytes = buildFileValue.isSpecialFile()
                  ? FileSystemUtils.readContent(buildFilePath)
                  : FileSystemUtils.readWithKnownFileSize(buildFilePath, buildFileValue.getSize());
            } catch (IOException e) {
              // Note that we did this work, so we should conservatively report this error as
              // transient.
              throw new PackageFunctionException(new BuildFileContainsErrorsException(
                  packageId, e.getMessage()), Transience.TRANSIENT);
            }
            try {
              preprocessingResult = packageFactory.preprocess(buildFilePath, packageId,
                  buildFileBytes, globber);
            } catch (IOException e) {
              throw new PackageFunctionException(
                  new BuildFileContainsErrorsException(
                      packageId, "preprocessing failed" + e.getMessage(), e), Transience.TRANSIENT);
            }
          } else {
            ParserInputSource replacementSource =
                ParserInputSource.create(replacementContents, buildFilePath.asFragment());
            preprocessingResult = Preprocessor.Result.noPreprocessing(replacementSource);
          }
          StoredEventHandler astParsingEventHandler = new StoredEventHandler();
          BuildFileAST ast = PackageFactory.parseBuildFile(packageId, preprocessingResult.result,
              preludeStatements, astParsingEventHandler);
          // If no globs were fetched during preprocessing, then there's no need to reuse *this
          // globber instance* during BUILD file evaluation; the correctness and performance
          // arguments below do not apply.
          Globber globberToStore = globber.getGlobPatterns().isEmpty() ? null : globber;
          astAfterPreprocessing = new AstAfterPreprocessing(preprocessingResult, ast,
              astParsingEventHandler, globberToStore);
          astCache.put(packageId, astAfterPreprocessing);
        }
        SkylarkImportResult importResult;
        try {
          importResult = discoverSkylarkImports(
              buildFilePath,
              packageId,
              astAfterPreprocessing,
              env);
        } catch (PackageFunctionException | InterruptedException e) {
          astCache.invalidate(packageId);
          throw e;
        }
        if (importResult == null) {
          return null;
        }
        astCache.invalidate(packageId);
        // If the globber was used to evaluate globs during preprocessing, it's important that we
        // reuse that globber during BUILD file evaluation for two reasons: (i) correctness, since
        // Skyframe deps are added after the fact (ii) performance, in the case that globs were
        // fetched lazily during preprocessing. See Preprocessor.Factory#considersGlobs.
        Globber globber = astAfterPreprocessing.globber != null
            ? astAfterPreprocessing.globber
            : packageFactory.createLegacyGlobber(buildFilePath.getParentDirectory(),
                packageId, packageLocator);
        pkgBuilder = packageFactory.createPackageFromPreprocessingAst(externalPkg, packageId,
            buildFilePath, astAfterPreprocessing, importResult.importMap,
            importResult.fileDependencies, defaultVisibility, globber);
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
    public InternalInconsistentFilesystemException(PackageIdentifier packageIdentifier,
        InconsistentFilesystemException e) {
      super(packageIdentifier, e.getMessage(), e);
      // This is not a transient error from the perspective of the PackageFunction.
      this.isTransient = false;
    }

    /** Used to represent a filesystem inconsistency discovered by the {@link PackageFunction}. */
    public InternalInconsistentFilesystemException(PackageIdentifier packageIdentifier,
        String inconsistencyMessage) {
      this(packageIdentifier, new InconsistentFilesystemException(inconsistencyMessage));
      this.isTransient = true;
    }

    public boolean isTransient() {
      return isTransient;
    }
  }

  private static class BadWorkspaceFileException extends NoSuchPackageException {
    private BadWorkspaceFileException(String message) {
      super(
          Label.EXTERNAL_PACKAGE_IDENTIFIER,
          "Error encountered while dealing with the WORKSPACE file: " + message);
    }
  }

  private static class BadPreludeFileException extends NoSuchPackageException {
    private BadPreludeFileException(PackageIdentifier packageIdentifier, String message) {
      super(packageIdentifier, "Error encountered while reading the prelude file: " + message);
    }
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link PackageFunction#compute}.
   */
  static class PackageFunctionException extends SkyFunctionException {
    public PackageFunctionException(NoSuchPackageException e, Transience transience) {
      super(e, transience);
    }
  }

  /** A simple value class to store the result of the Skylark imports.*/
  static final class SkylarkImportResult {
    final Map<String, Extension> importMap;
    final ImmutableList<Label> fileDependencies;
    private SkylarkImportResult(
        Map<String, Extension> importMap,
        ImmutableList<Label> fileDependencies) {
      this.importMap = importMap;
      this.fileDependencies = fileDependencies;
    }
  }

  static boolean isDefaultsPackage(PackageIdentifier packageIdentifier) {
    return packageIdentifier.getRepository().isDefault()
        && packageIdentifier.getPackageFragment().equals(DEFAULTS_PACKAGE_NAME);
  }
}
