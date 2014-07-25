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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.BulkPackageLocatorForCrossingSubpackageBoundaries;
import com.google.devtools.build.lib.packages.CachingPackageLocator;
import com.google.devtools.build.lib.packages.InvalidPackageNameException;
import com.google.devtools.build.lib.packages.LegacyPackage;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageLoadedEvent;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.build.lib.skyframe.GlobValue.InvalidGlobPatternException;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.ParserInputSource;
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

  private final ErrorEventListener reporter;
  private final PackageFactory packageFactory;
  private final CachingPackageLocator packageLocator;
  private final ConcurrentMap<String, LegacyPackage> packageFunctionCache;
  private final AtomicBoolean showLoadingProgress;
  private final AtomicReference<EventBus> eventBus;
  private final AtomicInteger numPackagesLoaded;

  /**
   * A package locator that delegates to the {@link PackageFunction}'s
   * CachingPackageLocator and also tracks the implicit dependencies on existing packages needed by
   * {@link BulkPackageLocatorForCrossingSubpackageBoundaries} for Skyframe's consumption.
   */
  private class SkyframePackageLocator implements CachingPackageLocator,
      BulkPackageLocatorForCrossingSubpackageBoundaries {

    private final Set<SkyKey> deps = Sets.newConcurrentHashSet();

    private SkyframePackageLocator() {}

    /**
     * Returns an immutable copy of the package lookup dependencies recorded since this
     * SkyframePackageLocator was constructed.
     */
    private ImmutableSet<SkyKey> getValues() {
      return ImmutableSet.copyOf(deps);
    }

    @Override
    @ThreadSafe
    public Path getBuildFileForPackage(String packageName) {
      return packageLocator.getBuildFileForPackage(packageName);
    }

    /**
     * Same specification as
     * {@link BulkPackageLocatorForCrossingSubpackageBoundaries#getExistingPackages},
     * but since the state of the package depends on this call, we need to record dependencies.
     */
    @Override
    public Map<PathFragment, Path> getExistingPackages(Set<PathFragment> candidates)
        throws InterruptedException {
      ImmutableMap.Builder<PathFragment, Path> result = ImmutableMap.builder();
      for (PathFragment pkgName : candidates) {
        // We first add a dependency on the PackageLookupValue for this package. This is important
        // in case the value of --deleted_packages changes in the future.
        deps.add(PackageLookupValue.key(pkgName));
        // Note the call to our wrapper implementation, which handles deleted packages.
        Path path = getBuildFileForPackage(pkgName.getPathString());
        if (path != null) {
          result.put(pkgName, path);
        }
      }
      return result.build();
    }
  }

  static final String DEFAULTS_PACKAGE_NAME = "tools/defaults";

  public PackageFunction(Reporter reporter, PackageFactory packageFactory,
      CachingPackageLocator pkgLocator, AtomicBoolean showLoadingProgress,
      ConcurrentMap<String, LegacyPackage> packageFunctionCache,
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
      PathFragment pkgName = (PathFragment) entry.getKey().argument();
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
      LegacyPackage pkg, Environment env, SkyframePackageLocator skyframePackageLocator)
          throws InconsistentFilesystemException {
    boolean packageShouldBeInError = pkg.containsErrors();
    // Tell the environment about our dependencies on potential subpackages.
    getPackageLookupDepsAndPropagateInconsistentFilesystemExceptions(
        skyframePackageLocator.getValues(), env, pkg.containsErrors());

    // TODO(bazel-team): Mark Skylark imports as dependencies.

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
    for (Pair<String, Boolean> globPattern : pkg.getGlobPatterns()) {
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

  @Override
  public SkyValue compute(SkyKey key, Environment env) throws PackageFunctionException,
      InterruptedException {
    PathFragment packagePathFragment = (PathFragment) key.argument();
    String packageName = packagePathFragment.getPathString();

    SkyKey packageLookupKey = PackageLookupValue.key(packagePathFragment);
    PackageLookupValue packageLookupValue;
    try {
      packageLookupValue = (PackageLookupValue)
          env.getValueOrThrow(packageLookupKey, Exception.class);
    } catch (BuildFileNotFoundException e) {
      throw new PackageFunctionException(key, e);
    } catch (InconsistentFilesystemException e) {
      throw new PackageFunctionException(key,
          new InconsistentFilesystemDuringPackageLoadingException(packageName, e));
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

    // If any calls to our implementation of
    // BulkPackageLocatorForCrossingSubpackageBoundaries#getExistingPackages are needed to
    // construct this package object, then we should record the dependencies.
    SkyframePackageLocator skyframePackageLocator = new SkyframePackageLocator();
    Path buildFilePath =
        packageLookupValue.getRoot().getRelative(packagePathFragment).getChild("BUILD");

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

    LegacyPackage legacyPkg = loadPackage(replacementContents, packageName, buildFilePath,
        env.getListener(), skyframePackageLocator, defaultVisibility);
    boolean packageShouldBeConsideredInError = legacyPkg.containsErrors();
    InconsistentFilesystemException inconsistentFilesystemException = null;
    try {
      packageShouldBeConsideredInError =
          markDependenciesAndPropagateInconsistentFilesystemExceptions(legacyPkg, env,
              skyframePackageLocator);
    } catch (InconsistentFilesystemException e) {
      inconsistentFilesystemException = e;
    }

    if (env.valuesMissing()) {
      // The package we just loaded will be in the {@code packageFunctionCache} next when this
      // SkyFunction is called again.
      return null;
    }
    // We know this SkyFunction will not be called again, so we can remove the cache entry.
    packageFunctionCache.remove(packageName);

    // This is safe since the LegacyPackage instance is only used at the Package type from here on
    // out.
    legacyPkg.dropLegacyData();
    Package pkg = legacyPkg;

    if (inconsistentFilesystemException != null) {
      throw new PackageFunctionException(key,
          new InconsistentFilesystemDuringPackageLoadingException(packageName,
              inconsistentFilesystemException));
    }

    if (packageShouldBeConsideredInError) {
      throw new PackageFunctionException(key, new BuildFileContainsErrorsException(pkg,
          "Package '" + packageName + "' contains errors"));
    }
    return new PackageValue(pkg);
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Constructs a {@link Package} object for the given package using legacy package loading.
   * Note that the returned package may be in error.
   */
  private LegacyPackage loadPackage(@Nullable String replacementContents, String packageName,
      Path buildFilePath, ErrorEventListener listener,
      SkyframePackageLocator skyframePackageLocator,
      RuleVisibility defaultVisibility) throws InterruptedException {
    ParserInputSource replacementSource = replacementContents == null ? null
        : ParserInputSource.create(replacementContents, buildFilePath);

    LegacyPackage pkg = packageFunctionCache.get(packageName);
    if (pkg == null) {
      if (showLoadingProgress.get()) {
        reporter.progress(null, "Loading package: " + packageName);
      }

      Clock clock = new JavaClock();
      long startTime = clock.nanoTime();
      pkg = packageFactory.createPackage(packageName, buildFilePath, skyframePackageLocator,
          replacementSource, defaultVisibility, skyframePackageLocator);

      if (eventBus.get() != null) {
        eventBus.get().post(new PackageLoadedEvent(pkg.getName(),
            (clock.nanoTime() - startTime) / (1000 * 1000),
            // It's impossible to tell if the package was loaded before, so we always pass false.
            /*reloading=*/false,
            !pkg.containsErrors()));
      }

      numPackagesLoaded.incrementAndGet();
      packageFunctionCache.put(packageName, pkg);
    }
    Event.replayEventsOn(listener, pkg.getEvents());
    return pkg;
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link PackageFunction#build}.
   */
  private static class PackageFunctionException extends SkyFunctionException {
    public PackageFunctionException(SkyKey key, NoSuchPackageException e) {
      super(key, e);
    }
  }
}
