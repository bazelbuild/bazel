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
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.CachingPackageLocator;
import com.google.devtools.build.lib.packages.Globber;
import com.google.devtools.build.lib.packages.InvalidPackageNameException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageFactory.LegacyGlobber;
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
import com.google.devtools.build.lib.syntax.SkylarkSemanticsOptions;
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
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
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
  private final Cache<PackageIdentifier, CacheEntryWithGlobDeps<Package.Builder>>
      packageFunctionCache;
  private final Cache<PackageIdentifier, CacheEntryWithGlobDeps<AstAfterPreprocessing>> astCache;
  private final AtomicBoolean showLoadingProgress;
  private final AtomicInteger numPackagesLoaded;
  @Nullable private final PackageProgressReceiver packageProgress;
  private final Profiler profiler = Profiler.instance();
  private final Label preludeLabel;

  // Not final only for testing.
  @Nullable private SkylarkImportLookupFunction skylarkImportLookupFunctionForInlining;

  private final ActionOnIOExceptionReadingBuildFile actionOnIOExceptionReadingBuildFile;

  static final PathFragment DEFAULTS_PACKAGE_NAME = PathFragment.create("tools/defaults");

  public PackageFunction(
      PackageFactory packageFactory,
      CachingPackageLocator pkgLocator,
      AtomicBoolean showLoadingProgress,
      Cache<PackageIdentifier, CacheEntryWithGlobDeps<Package.Builder>> packageFunctionCache,
      Cache<PackageIdentifier, CacheEntryWithGlobDeps<AstAfterPreprocessing>> astCache,
      AtomicInteger numPackagesLoaded,
      @Nullable SkylarkImportLookupFunction skylarkImportLookupFunctionForInlining,
      @Nullable PackageProgressReceiver packageProgress,
      ActionOnIOExceptionReadingBuildFile actionOnIOExceptionReadingBuildFile) {
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
    this.packageProgress = packageProgress;
    this.actionOnIOExceptionReadingBuildFile = actionOnIOExceptionReadingBuildFile;
  }

  public PackageFunction(
      PackageFactory packageFactory,
      CachingPackageLocator pkgLocator,
      AtomicBoolean showLoadingProgress,
      Cache<PackageIdentifier, CacheEntryWithGlobDeps<Package.Builder>> packageFunctionCache,
      Cache<PackageIdentifier, CacheEntryWithGlobDeps<AstAfterPreprocessing>> astCache,
      AtomicInteger numPackagesLoaded,
      @Nullable SkylarkImportLookupFunction skylarkImportLookupFunctionForInlining) {
    this(
        packageFactory,
        pkgLocator,
        showLoadingProgress,
        packageFunctionCache,
        astCache,
        numPackagesLoaded,
        skylarkImportLookupFunctionForInlining,
        null,
        ActionOnIOExceptionReadingBuildFile.UseOriginalIOException.INSTANCE);
  }

  public void setSkylarkImportLookupFunctionForInliningForTesting(
      SkylarkImportLookupFunction skylarkImportLookupFunctionForInlining) {
    this.skylarkImportLookupFunctionForInlining = skylarkImportLookupFunctionForInlining;
  }

  /**
   * What to do when encountering an {@link IOException} trying to read the contents of a BUILD
   * file.
   *
   * <p>Any choice besides
   * {@link ActionOnIOExceptionReadingBuildFile.UseOriginalIOException#INSTANCE} is potentially
   * incrementally unsound: if the initial {@link IOException} is transient, then Blaze will
   * "incorrectly" not attempt to redo package loading for this BUILD file on incremental builds.
   *
   * <p>The fact that this behavior is configurable and potentially unsound is a concession to
   * certain desired use cases with fancy filesystems.
   */
  public interface ActionOnIOExceptionReadingBuildFile {
    /**
     * Given the {@link IOException} encountered when reading the contents of a BUILD file,
     * returns the contents that should be used, or {@code null} if the original {@link IOException}
     * should be respected (that is, we should error-out with a package loading error).
     */
    @Nullable
    byte[] maybeGetBuildFileContentsToUse(IOException originalExn);

    /**
     * A {@link ActionOnIOExceptionReadingBuildFile} whose {@link #maybeGetBuildFileContentsToUse}
     * has the sensible behavior of always respecting the initial {@link IOException}.
     */
    public static class UseOriginalIOException implements ActionOnIOExceptionReadingBuildFile {
      public static final UseOriginalIOException INSTANCE = new UseOriginalIOException();

      private UseOriginalIOException() {
      }

      @Override
      @Nullable
      public byte[] maybeGetBuildFileContentsToUse(IOException originalExn) {
        return null;
      }
    }
  }

  /** An entry in {@link PackageFunction}'s internal caches. */
  public static class CacheEntryWithGlobDeps<T> {
    private final T value;
    private final Set<SkyKey> globDepKeys;
    @Nullable
    private final LegacyGlobber legacyGlobber;

    private CacheEntryWithGlobDeps(T value, Set<SkyKey> globDepKeys,
        @Nullable LegacyGlobber legacyGlobber) {
      this.value = value;
      this.globDepKeys = globDepKeys;
      this.legacyGlobber = legacyGlobber;
    }
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
   * Marks the given dependencies, and returns those already present. Ignores any exception thrown
   * while building the dependency, except for filesystem inconsistencies.
   *
   * <p>We need to mark dependencies implicitly used by the legacy package loading code, but we
   * don't care about any skyframe errors since the package knows whether it's in error or not.
   */
  private static Pair<? extends Map<PathFragment, PackageLookupValue>, Boolean>
      getPackageLookupDepsAndPropagateInconsistentFilesystemExceptions(
          PackageIdentifier packageIdentifier,
          Iterable<SkyKey> depKeys,
          Environment env,
          boolean packageWasInError)
          throws InternalInconsistentFilesystemException, InterruptedException {
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

  private static boolean markFileDepsAndPropagateFilesystemExceptions(
      PackageIdentifier packageIdentifier,
      Iterable<SkyKey> depKeys,
      Environment env,
      boolean packageWasInError)
      throws InternalInconsistentFilesystemException, InterruptedException {
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

  /**
   * These deps have already been marked (see {@link SkyframeHybridGlobber}) but we need to properly
   * handle some errors that legacy package loading can't handle gracefully.
   */
  private static boolean handleGlobDepsAndPropagateFilesystemExceptions(
      PackageIdentifier packageIdentifier,
      Iterable<SkyKey> depKeys,
      Environment env,
      boolean packageWasInError)
      throws InternalInconsistentFilesystemException, InterruptedException {
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
   * <p>Most skyframe exceptions encountered here are ignored, as similar errors should have already
   * been encountered by legacy package loading (if not, then the filesystem is inconsistent). Some
   * exceptions that Skyframe is stricter about (disallowed access to files outside package roots)
   * are propagated.
   */
  private static boolean markDependenciesAndPropagateFilesystemExceptions(
      Environment env,
      Set<SkyKey> globDepKeys,
      Map<Label, Path> subincludes,
      PackageIdentifier packageIdentifier,
      boolean containsErrors)
      throws InternalInconsistentFilesystemException, InterruptedException {
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
        markFileDepsAndPropagateFilesystemExceptions(
            packageIdentifier, subincludeFileDepKeys, env, containsErrors);

    packageShouldBeInError |=
        handleGlobDepsAndPropagateFilesystemExceptions(
            packageIdentifier, globDepKeys, env, containsErrors);

    return packageShouldBeInError;
  }

  /**
   * Adds a dependency on the WORKSPACE file, representing it as a special type of package.
   *
   * @throws PackageFunctionException if there is an error computing the workspace file or adding
   *     its rules to the //external package.
   */
  private SkyValue getExternalPackage(Environment env, Path packageLookupPath)
      throws PackageFunctionException, InterruptedException {
    RootedPath workspacePath = RootedPath.toRootedPath(
        packageLookupPath, Label.EXTERNAL_PACKAGE_FILE_NAME);
    SkyKey workspaceKey = ExternalPackageFunction.key(workspacePath);
    PackageValue workspace = null;
    try {
      // This may throw a NoSuchPackageException if the WORKSPACE file was malformed or had other
      // problems. Since this function can't add much context, we silently bubble it up.
      workspace =
          (PackageValue)
              env.getValueOrThrow(
                  workspaceKey,
                  IOException.class,
                  FileSymlinkException.class,
                  InconsistentFilesystemException.class,
                  EvalException.class,
                  SkylarkImportFailedException.class);
    } catch (IOException | FileSymlinkException | InconsistentFilesystemException
          | EvalException | SkylarkImportFailedException e) {
      throw new PackageFunctionException(
          new NoSuchPackageException(
              Label.EXTERNAL_PACKAGE_IDENTIFIER,
              "Error encountered while dealing with the WORKSPACE file: " + e.getMessage()),
          Transience.PERSISTENT);
    }
    if (workspace == null) {
      return null;
    }

    Package pkg = workspace.getPackage();
    Event.replayEventsOn(env.getListener(), pkg.getEvents());

    packageFactory.afterDoneLoadingPackage(pkg);
    return new PackageValue(pkg);
  }

  @Override
  public SkyValue compute(SkyKey key, Environment env) throws PackageFunctionException,
      InterruptedException {
    PackageIdentifier packageId = (PackageIdentifier) key.argument();

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
          new NoSuchPackageException(packageId, e.getMessage(), e), Transience.PERSISTENT);
    }
    if (packageLookupValue == null) {
      return null;
    }

    if (!packageLookupValue.packageExists()) {
      switch (packageLookupValue.getErrorReason()) {
        case NO_BUILD_FILE:
        case DELETED_PACKAGE:
        case REPOSITORY_NOT_FOUND:
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
    WorkspaceNameValue workspaceNameValue =
        (WorkspaceNameValue) env.getValue(WorkspaceNameValue.key());
    if (workspaceNameValue == null) {
      return null;
    }
    String workspaceName = workspaceNameValue.getName();
    RootedPath buildFileRootedPath = packageLookupValue.getRootedPath(packageId);
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

    SkylarkSemanticsOptions skylarkSemantics = PrecomputedValue.SKYLARK_SEMANTICS.get(env);
    if (skylarkSemantics == null) {
      return null;
    }

    SkyKey astLookupKey = ASTFileLookupValue.key(preludeLabel);
    ASTFileLookupValue astLookupValue = null;
    try {
      astLookupValue = (ASTFileLookupValue) env.getValueOrThrow(astLookupKey,
          ErrorReadingSkylarkExtensionException.class, InconsistentFilesystemException.class);
    } catch (ErrorReadingSkylarkExtensionException | InconsistentFilesystemException e) {
      throw new PackageFunctionException(
          new NoSuchPackageException(
              packageId, "Error encountered while reading the prelude file: " + e.getMessage()),
          Transience.PERSISTENT);
    }
    if (astLookupValue == null) {
      return null;
    }
    // The prelude file doesn't have to exist. If not, we substitute an empty statement list.
    List<Statement> preludeStatements =
        astLookupValue.lookupSuccessful()
            ? astLookupValue.getAST().getStatements() : ImmutableList.<Statement>of();
    CacheEntryWithGlobDeps<Package.Builder> packageBuilderAndGlobDeps =
        loadPackage(
            workspaceName,
            replacementContents,
            packageId,
            buildFilePath,
            buildFileValue,
            defaultVisibility,
            skylarkSemantics,
            preludeStatements,
            packageLookupValue.getRoot(),
            env);
    if (packageBuilderAndGlobDeps == null) {
      return null;
    }
    Package.Builder pkgBuilder = packageBuilderAndGlobDeps.value;
    pkgBuilder.buildPartial();
    try {
      // Since the Skyframe dependencies we request below in
      // markDependenciesAndPropagateFilesystemExceptions are requested independently of
      // the ones requested here in
      // handleLabelsCrossingSubpackagesAndPropagateInconsistentFilesystemExceptions, we don't
      // bother checking for missing values and instead piggyback on the env.missingValues() call
      // for the former. This avoids a Skyframe restart.
      handleLabelsCrossingSubpackagesAndPropagateInconsistentFilesystemExceptions(
          packageLookupValue.getRoot(), packageId, pkgBuilder, env);
    } catch (InternalInconsistentFilesystemException e) {
      packageFunctionCache.invalidate(packageId);
      throw new PackageFunctionException(
          e.toNoSuchPackageException(),
          e.isTransient() ? Transience.TRANSIENT : Transience.PERSISTENT);
    }
    Set<SkyKey> globKeys = packageBuilderAndGlobDeps.globDepKeys;
    Map<Label, Path> subincludes = pkgBuilder.getSubincludes();
    boolean packageShouldBeConsideredInError;
    try {
      packageShouldBeConsideredInError =
          markDependenciesAndPropagateFilesystemExceptions(
              env, globKeys, subincludes, packageId, pkgBuilder.containsErrors());
    } catch (InternalInconsistentFilesystemException e) {
      packageFunctionCache.invalidate(packageId);
      throw new PackageFunctionException(
          e.toNoSuchPackageException(),
          e.isTransient() ? Transience.TRANSIENT : Transience.PERSISTENT);
    }
    if (env.valuesMissing()) {
      return null;
    }

    Event.replayEventsOn(env.getListener(), pkgBuilder.getEvents());

    if (packageShouldBeConsideredInError) {
      pkgBuilder.setContainsErrors();
    }
    Package pkg = pkgBuilder.finishBuild();

    // We know this SkyFunction will not be called again, so we can remove the cache entry.
    packageFunctionCache.invalidate(packageId);

    packageFactory.afterDoneLoadingPackage(pkg);
    return new PackageValue(pkg);
  }

  private static FileValue getBuildFileValue(Environment env, RootedPath buildFileRootedPath)
      throws InterruptedException {
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
      throws NoSuchPackageException, InterruptedException {
    Preconditions.checkArgument(!packageId.getRepository().isDefault());

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
      throw new BuildFileContainsErrorsException(packageId, e.getMessage());
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
        LinkedHashMap<Label, SkylarkImportLookupValue> alreadyVisitedImports =
            Maps.newLinkedHashMapWithExpectedSize(importLookupKeys.size());
        for (SkyKey importLookupKey : importLookupKeys) {
          SkyValue skyValue =
              skylarkImportLookupFunctionForInlining.computeWithInlineCalls(
                  importLookupKey, env, alreadyVisitedImports);
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
      throw new BuildFileContainsErrorsException(packageId, e.getMessage());
    } catch (InconsistentFilesystemException e) {
      throw new NoSuchPackageException(packageId, e.getMessage(), e);
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
      Path pkgRoot, PackageIdentifier pkgId, Package.Builder pkgBuilder, Environment env)
      throws InternalInconsistentFilesystemException, InterruptedException {
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
      Package.Builder pkgBuilder, Path pkgRoot, Label label, @Nullable Location location,
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
    if (!containingPkg.getSourceRoot().startsWith(
        label.getPackageIdentifier().getSourceRoot())) {
      // This label is referencing an imaginary package, because the containing package should
      // extend the label's package: if the label is //a/b:c/d, the containing package could be
      // //a/b/c or //a/b, but should never be //a. Usually such errors will be caught earlier, but
      // in some exceptional cases (such as a Python-aware BUILD file catching its own io
      // exceptions), it reaches here, and we tolerate it.
      return false;
    }
    PathFragment labelNameFragment = PathFragment.create(label.getName());
    String message = String.format("Label '%s' crosses boundary of subpackage '%s'",
        label, containingPkg);
    Path containingRoot = containingPkgLookupValue.getContainingPackageRoot();
    if (pkgRoot.equals(containingRoot)) {
      PathFragment labelNameInContainingPackage = labelNameFragment.subFragment(
          containingPkg.getPackageFragment().segmentCount()
              - label.getPackageFragment().segmentCount(),
          labelNameFragment.segmentCount());
      message += " (perhaps you meant to put the colon here: '";
      if (containingPkg.getRepository().isDefault() || containingPkg.getRepository().isMain()) {
        message += "//";
      }
      message += containingPkg + ":" + labelNameInContainingPackage + "'?)";
    } else {
      message += " (have you deleted " + containingPkg + "/BUILD? "
          + "If so, use the --deleted_packages=" + containingPkg + " option)";
    }
    pkgBuilder.addEvent(Event.error(location, message));
    return true;
  }

  /**
   * A {@link Globber} implemented on top of skyframe that falls back to a
   * {@link PackageFactory.LegacyGlobber} on a skyframe cache-miss. This way we don't require a
   * skyframe restart after a call to {@link Globber#runAsync} and before/during a call to
   * {@link Globber#fetch}.
   *
   * <p>There are three advantages to this hybrid approach over the more obvious approach of solely
   * using a {@link PackageFactory.LegacyGlobber}:
   * <ul>
   * <li>We trivially have the proper Skyframe {@link GlobValue} deps, whereas we would need to
   * request them after-the-fact if we solely used a {@link PackageFactory.LegacyGlobber}.
   * <li>We don't need to re-evaluate globs whose expression hasn't changed (e.g. in the common case
   * of a BUILD file edit that doesn't change a glob expression), whereas legacy package loading
   * with a {@link PackageFactory.LegacyGlobber} would naively re-evaluate globs when re-evaluating
   * the BUILD file.
   * <li>We don't need to re-evaluate invalidated globs *twice* (the single re-evaluation via our
   * GlobValue deps is sufficient and optimal). See above for why the second evaluation would
   * happen.
   * </ul>
   */
  private static class SkyframeHybridGlobber implements Globber {
    private final PackageIdentifier packageId;
    private final Path packageRoot;
    private final Environment env;
    private final LegacyGlobber legacyGlobber;
    private final Set<SkyKey> globDepsRequested = Sets.newConcurrentHashSet();

    private SkyframeHybridGlobber(PackageIdentifier packageId, Path packageRoot, Environment env,
        LegacyGlobber legacyGlobber) {
      this.packageId = packageId;
      this.packageRoot = packageRoot;
      this.env = env;
      this.legacyGlobber = legacyGlobber;
    }

    private Set<SkyKey> getGlobDepsRequested() {
      return ImmutableSet.copyOf(globDepsRequested);
    }

    private SkyKey getGlobKey(String pattern, boolean excludeDirs) throws BadGlobException {
      try {
        return GlobValue.key(packageId, packageRoot, pattern, excludeDirs,
              PathFragment.EMPTY_FRAGMENT);
      } catch (InvalidGlobPatternException e) {
        throw new BadGlobException(e.getMessage());
      }
    }

    @Override
    public Token runAsync(List<String> includes, List<String> excludes, boolean excludeDirs)
        throws BadGlobException, InterruptedException {
      List<SkyKey> globKeys = new ArrayList<>(includes.size() + excludes.size());
      LinkedHashSet<SkyKey> includesKeys = Sets.newLinkedHashSetWithExpectedSize(includes.size());
      LinkedHashSet<SkyKey> excludesKeys = Sets.newLinkedHashSetWithExpectedSize(excludes.size());
      Map<SkyKey, String> globKeyToIncludeStringMap =
          Maps.newHashMapWithExpectedSize(includes.size());
      Map<SkyKey, String> globKeyToExcludeStringMap =
          Maps.newHashMapWithExpectedSize(excludes.size());

      for (String pattern : includes) {
        SkyKey globKey = getGlobKey(pattern, excludeDirs);
        globKeys.add(globKey);
        includesKeys.add(globKey);
        globKeyToIncludeStringMap.put(globKey, pattern);
      }
      for (String pattern : excludes) {
        SkyKey globKey = getGlobKey(pattern, excludeDirs);
        globKeys.add(globKey);
        excludesKeys.add(globKey);
        globKeyToExcludeStringMap.put(globKey, pattern);
      }
      globDepsRequested.addAll(globKeys);

      Map<SkyKey, ValueOrException4<IOException, BuildFileNotFoundException,
          FileSymlinkCycleException, InconsistentFilesystemException>> globValueMap =
          env.getValuesOrThrow(globKeys, IOException.class, BuildFileNotFoundException.class,
              FileSymlinkCycleException.class, InconsistentFilesystemException.class);

      // For each missing glob, evaluate it asychronously via the delegate.
      //
      // TODO(bazel-team): Consider not delegating missing globs during glob prefetching - a
      // single skyframe restart after the prefetch step is probably tolerable.
      Collection<SkyKey> missingKeys = getMissingKeys(globKeys, globValueMap);
      List<String> includesToDelegate = new ArrayList<>(missingKeys.size());
      List<String> excludesToDelegate = new ArrayList<>(missingKeys.size());
      for (SkyKey missingKey : missingKeys) {
        String missingIncludePattern = globKeyToIncludeStringMap.get(missingKey);
        if (missingIncludePattern != null) {
          includesToDelegate.add(missingIncludePattern);
          includesKeys.remove(missingKey);
        }
        String missingExcludePattern = globKeyToExcludeStringMap.get(missingKey);
        if (missingExcludePattern != null) {
          excludesToDelegate.add(missingExcludePattern);
          excludesKeys.remove(missingKey);
        }
      }
      Token legacyIncludesToken =
          legacyGlobber.runAsync(includesToDelegate, ImmutableList.<String>of(), excludeDirs);
      // See the HybridToken class-comment for why we pass excludesToDelegate as the includes
      // parameter here.
      Token legacyExcludesToken =
          legacyGlobber.runAsync(excludesToDelegate, ImmutableList.<String>of(), excludeDirs);

      return new HybridToken(globValueMap, includesKeys, excludesKeys,
          legacyIncludesToken, legacyExcludesToken);
    }

    private Collection<SkyKey> getMissingKeys(Collection<SkyKey> globKeys,
        Map<SkyKey, ValueOrException4<IOException, BuildFileNotFoundException,
            FileSymlinkCycleException, InconsistentFilesystemException>> globValueMap) {
      List<SkyKey> missingKeys = new ArrayList<>(globKeys.size());
      for (SkyKey globKey : globKeys) {
        ValueOrException4<IOException, BuildFileNotFoundException, FileSymlinkCycleException,
            InconsistentFilesystemException> valueOrException = globValueMap.get(globKey);
        if (valueOrException == null) {
          missingKeys.add(globKey);
        }
        try {
          if (valueOrException.get() == null) {
            missingKeys.add(globKey);
          }
        } catch (IOException | BuildFileNotFoundException | FileSymlinkCycleException
            | InconsistentFilesystemException doesntMatter) {
          continue;
        }
      }
      return missingKeys;
    }

    @Override
    public List<String> fetch(Token token) throws IOException, InterruptedException {
      HybridToken hybridToken = (HybridToken) token;
      return hybridToken.resolve(legacyGlobber);
    }

    @Override
    public void onInterrupt() {
      legacyGlobber.onInterrupt();
    }

    @Override
    public void onCompletion() {
      legacyGlobber.onCompletion();
    }

    /**
     * A {@link Globber.Token} that encapsulates the result of a single {@link Globber#runAsync}
     * call via the fetching of some globs from skyframe, and some other globs via a
     * {@link PackageFactory.LegacyGlobber}. We take care to properly handle 'includes' vs
     * 'excludes'.
     *
     * <p>That is, we evaluate {@code glob(includes, excludes)} by partitioning {@code includes} and
     * {@code excludes}.
     *
     * <pre>
     * {@code
     * includes = includes_sky U includes_leg
     * excludes = excludes_sky U excludes_leg
     * }
     * </pre>
     *
     * <p>and then noting
     *
     * <pre>
     * {@code
     * glob(includes, excludes) =
     *     (glob(includes_sky, []) U glob(includes_leg, []))
     *   - (glob(excludes_sky, []) U glob(excludes_leg, []))
     * }
     * </pre>
     *
     * <p>Importantly, we pass excludes=[] in all cases; otherwise we'd be incorrectly not
     * subtracting excluded glob matches from the overall list of matches. In other words, we
     * implement the subtractive nature of excludes ourselves in {@link #resolve}.
     */
    private static class HybridToken extends Globber.Token {
      // The result of the Skyframe lookup for all the needed glob patterns.
      private final Map<SkyKey, ValueOrException4<IOException, BuildFileNotFoundException,
          FileSymlinkCycleException, InconsistentFilesystemException>> globValueMap;
      // The skyframe keys corresponding to the 'includes' patterns fetched from Skyframe
      // (this is includes_sky above).
      private final Iterable<SkyKey> includesGlobKeys;
      // The skyframe keys corresponding to the 'excludes' patterns fetched from Skyframe
      // (this is excludes_sky above).
      private final Iterable<SkyKey> excludesGlobKeys;
      // A token for computing includes_leg.
      private final Token legacyIncludesToken;
      // A token for computing excludes_leg.
      private final Token legacyExcludesToken;

      private HybridToken(Map<SkyKey, ValueOrException4<IOException, BuildFileNotFoundException,
          FileSymlinkCycleException, InconsistentFilesystemException>> globValueMap,
          Iterable<SkyKey> includesGlobKeys, Iterable<SkyKey> excludesGlobKeys,
          Token delegateIncludesToken, Token delegateExcludesToken) {
        this.globValueMap = globValueMap;
        this.includesGlobKeys = includesGlobKeys;
        this.excludesGlobKeys = excludesGlobKeys;
        this.legacyIncludesToken = delegateIncludesToken;
        this.legacyExcludesToken = delegateExcludesToken;
      }

      private List<String> resolve(Globber delegate) throws IOException, InterruptedException {
        HashSet<String> matches = new HashSet<>();
        for (SkyKey includeGlobKey : includesGlobKeys) {
          // TODO(bazel-team): NestedSet expansion here is suboptimal.
          for (PathFragment match : getGlobMatches(includeGlobKey, globValueMap)) {
            matches.add(match.getPathString());
          }
        }
        matches.addAll(delegate.fetch(legacyIncludesToken));
        for (SkyKey excludeGlobKey : excludesGlobKeys) {
          for (PathFragment match : getGlobMatches(excludeGlobKey, globValueMap)) {
            matches.remove(match.getPathString());
          }
        }
        for (String delegateExcludeMatch : delegate.fetch(legacyExcludesToken)) {
          matches.remove(delegateExcludeMatch);
        }
        List<String> result = new ArrayList<>(matches);
        // Skyframe glob results are unsorted. And we used a LegacyGlobber that doesn't sort.
        // Therefore, we want to unconditionally sort here.
        Collections.sort(result);
        return result;
      }

      private static NestedSet<PathFragment> getGlobMatches(
          SkyKey globKey,
          Map<
                  SkyKey,
                  ValueOrException4<
                      IOException, BuildFileNotFoundException, FileSymlinkCycleException,
                      InconsistentFilesystemException>>
              globValueMap)
          throws IOException {
        ValueOrException4<IOException, BuildFileNotFoundException, FileSymlinkCycleException,
            InconsistentFilesystemException> valueOrException =
                Preconditions.checkNotNull(globValueMap.get(globKey), "%s should not be missing",
                    globKey);
        try {
          return Preconditions.checkNotNull((GlobValue) valueOrException.get(),
              "%s should not be missing", globKey).getMatches();
        } catch (BuildFileNotFoundException | FileSymlinkCycleException
            | InconsistentFilesystemException e) {
          // Legacy package loading is only able to handle an IOException, so a rethrow here is the
          // best we can do. But after legacy package loading, PackageFunction will go through all
          // the skyframe deps and properly handle InconsistentFilesystemExceptions.
          throw new IOException(e.getMessage());
        }
      }
    }
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
  private CacheEntryWithGlobDeps<Package.Builder> loadPackage(
      String workspaceName,
      @Nullable String replacementContents,
      PackageIdentifier packageId,
      Path buildFilePath,
      @Nullable FileValue buildFileValue,
      RuleVisibility defaultVisibility,
      SkylarkSemanticsOptions skylarkSemantics,
      List<Statement> preludeStatements,
      Path packageRoot,
      Environment env)
      throws InterruptedException, PackageFunctionException {
    CacheEntryWithGlobDeps<Package.Builder> packageFunctionCacheEntry =
        packageFunctionCache.getIfPresent(packageId);
    if (packageFunctionCacheEntry == null) {
      profiler.startTask(ProfilerTask.CREATE_PACKAGE, packageId.toString());
      if (packageProgress != null) {
        packageProgress.startReadPackage(packageId);
      }
      try {
        CacheEntryWithGlobDeps<AstAfterPreprocessing> astCacheEntry =
            astCache.getIfPresent(packageId);
        if (astCacheEntry == null) {
          if (showLoadingProgress.get()) {
            env.getListener().handle(Event.progress("Loading package: " + packageId));
          }
          // We use a LegacyGlobber that doesn't sort the matches for each individual glob pattern,
          // since we want to sort the final result anyway.
          LegacyGlobber legacyGlobber = packageFactory.createLegacyGlobberThatDoesntSort(
              buildFilePath.getParentDirectory(), packageId, packageLocator);
          SkyframeHybridGlobber skyframeGlobber = new SkyframeHybridGlobber(packageId, packageRoot,
              env, legacyGlobber);
          ParserInputSource input;
          if (replacementContents == null) {
            Preconditions.checkNotNull(buildFileValue, packageId);
            byte[] buildFileBytes = null;
            try {
              buildFileBytes =
                  buildFileValue.isSpecialFile()
                      ? FileSystemUtils.readContent(buildFilePath)
                      : FileSystemUtils.readWithKnownFileSize(
                          buildFilePath, buildFileValue.getSize());
            } catch (IOException e) {
              buildFileBytes =
                  actionOnIOExceptionReadingBuildFile.maybeGetBuildFileContentsToUse(e);
              if (buildFileBytes == null) {
                // Note that we did the work that led to this IOException, so we should
                // conservatively report this error as transient.
                throw new PackageFunctionException(new BuildFileContainsErrorsException(
                    packageId, e.getMessage(), e), Transience.TRANSIENT);
              }
              // If control flow reaches here, we're in territory that is deliberately unsound.
              // See the javadoc for ActionOnIOExceptionReadingBuildFile.
            }
            input =
                ParserInputSource.create(
                    FileSystemUtils.convertFromLatin1(buildFileBytes),
                    buildFilePath.asFragment());
          } else {
            input = ParserInputSource.create(replacementContents, buildFilePath.asFragment());
          }
          StoredEventHandler astParsingEventHandler = new StoredEventHandler();
          BuildFileAST ast =
              PackageFactory.parseBuildFile(
                  packageId, input, preludeStatements, astParsingEventHandler);
          // If no globs were fetched during preprocessing, then there's no need to reuse the
          // legacy globber instance during BUILD file evaluation since the performance argument
          // below does not apply.
          Set<SkyKey> globDepsRequested = skyframeGlobber.getGlobDepsRequested();
          LegacyGlobber legacyGlobberToStore = globDepsRequested.isEmpty() ? null : legacyGlobber;
          astCacheEntry =
              new CacheEntryWithGlobDeps<>(
                  new AstAfterPreprocessing(ast, astParsingEventHandler),
                  globDepsRequested,
                  legacyGlobberToStore);
          astCache.put(packageId, astCacheEntry);
        }
        AstAfterPreprocessing astAfterPreprocessing = astCacheEntry.value;
        Set<SkyKey> globDepsRequestedDuringPreprocessing = astCacheEntry.globDepKeys;
        SkylarkImportResult importResult;
        try {
          importResult =
              fetchImportsFromBuildFile(
                  buildFilePath,
                  packageId,
                  astAfterPreprocessing.ast,
                  env,
                  skylarkImportLookupFunctionForInlining);
        } catch (NoSuchPackageException e) {
          throw new PackageFunctionException(e, Transience.PERSISTENT);
        } catch (InterruptedException e) {
          astCache.invalidate(packageId);
          throw e;
        }
        if (importResult == null) {
          return null;
        }
        astCache.invalidate(packageId);
        // If a legacy globber was used to evaluate globs during preprocessing, it's important that
        // we reuse that globber during BUILD file evaluation for performance, in the case that
        // globs were fetched lazily during preprocessing. See Preprocessor.Factory#considersGlobs.
        LegacyGlobber legacyGlobber = astCacheEntry.legacyGlobber != null
            ? astCacheEntry.legacyGlobber
            : packageFactory.createLegacyGlobber(
                buildFilePath.getParentDirectory(), packageId, packageLocator);
        SkyframeHybridGlobber skyframeGlobber = new SkyframeHybridGlobber(packageId, packageRoot,
            env, legacyGlobber);
        Package.Builder pkgBuilder = packageFactory.createPackageFromPreprocessingAst(
            workspaceName,
            packageId,
            buildFilePath,
            astAfterPreprocessing,
            importResult.importMap,
            importResult.fileDependencies,
            defaultVisibility,
            skylarkSemantics,
            skyframeGlobber);
        Set<SkyKey> globDepsRequested = ImmutableSet.<SkyKey>builder()
            .addAll(globDepsRequestedDuringPreprocessing)
            .addAll(skyframeGlobber.getGlobDepsRequested())
            .build();
        packageFunctionCacheEntry =
            new CacheEntryWithGlobDeps<>(pkgBuilder, globDepsRequested, null);
        numPackagesLoaded.incrementAndGet();
        if (packageProgress != null) {
          packageProgress.doneReadPackage(packageId);
        }
        packageFunctionCache.put(packageId, packageFunctionCacheEntry);
      } finally {
        profiler.completeTask(ProfilerTask.CREATE_PACKAGE);
      }
    }
    return packageFunctionCacheEntry;
  }

  private static class InternalInconsistentFilesystemException extends Exception {
    private boolean isTransient;

    private PackageIdentifier packageIdentifier;

    /**
     * Used to represent a filesystem inconsistency discovered outside the
     * {@link PackageFunction}.
     */
    public InternalInconsistentFilesystemException(PackageIdentifier packageIdentifier,
        InconsistentFilesystemException e) {
      super(e.getMessage(), e);
      this.packageIdentifier = packageIdentifier;
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

    private NoSuchPackageException toNoSuchPackageException() {
      return new NoSuchPackageException(
          packageIdentifier, this.getMessage(), (Exception) this.getCause());
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

  public static boolean isDefaultsPackage(PackageIdentifier packageIdentifier) {
    return packageIdentifier.getRepository().isMain()
        && packageIdentifier.getPackageFragment().equals(DEFAULTS_PACKAGE_NAME);
  }
}
