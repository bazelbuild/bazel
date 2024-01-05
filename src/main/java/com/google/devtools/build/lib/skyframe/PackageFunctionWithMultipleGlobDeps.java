// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.ThreadStateReceiver;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.io.FileSymlinkException;
import com.google.devtools.build.lib.io.InconsistentFilesystemException;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.CachingPackageLocator;
import com.google.devtools.build.lib.packages.Globber;
import com.google.devtools.build.lib.packages.GlobberUtils;
import com.google.devtools.build.lib.packages.NonSkyframeGlobber;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;
import javax.annotation.Nullable;

/**
 * Computes the {@link PackageValue} which depends on multiple GLOB nodes.
 *
 * <p>Every glob pattern defined in the package's {@code BUILD} file is represented as a single
 * GLOB node in the dependency graph.
 *
 * <p>{@link PackageFunctionWithMultipleGlobDeps} subclass supports both {@code SKYFRAME_HYBRID}
 * and {@code NON_SKYFRAME} globbing strategy. Incremental evaluation adopts {@code SKYFRAME_HYBRID}
 * globbing strategy, and it can just use the unchanged {@link GlobValue} stored in Skyframe when
 * incrementally reloading the package. On the other hand, {@code NON_SKYFRAME} globbing strategy is
 * used for non-incremental evaluations with no GLOB node queried and stored in Skyframe.
 */
final class PackageFunctionWithMultipleGlobDeps extends PackageFunction {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  PackageFunctionWithMultipleGlobDeps(
      PackageFactory packageFactory,
      CachingPackageLocator pkgLocator,
      AtomicBoolean showLoadingProgress,
      AtomicInteger numPackagesSuccessfullyLoaded,
      @Nullable BzlLoadFunction bzlLoadFunctionForInlining,
      @Nullable PackageProgressReceiver packageProgress,
      ActionOnIOExceptionReadingBuildFile actionOnIOExceptionReadingBuildFile,
      boolean shouldUseRepoDotBazel,
      GlobbingStrategy globbingStrategy,
      Function<SkyKey, ThreadStateReceiver> threadStateReceiverFactoryForMetrics,
      AtomicReference<Semaphore> cpuBoundSemaphore) {
    super(
        packageFactory,
        pkgLocator,
        showLoadingProgress,
        numPackagesSuccessfullyLoaded,
        bzlLoadFunctionForInlining,
        packageProgress,
        actionOnIOExceptionReadingBuildFile,
        shouldUseRepoDotBazel,
        globbingStrategy,
        threadStateReceiverFactoryForMetrics,
        cpuBoundSemaphore);
  }

  /**
   * If globbing strategy is {@code GlobbingStrategy#SKYFRAME_HYBRID}, these deps should have
   * already been marked by the {@link Globber} but we need to properly handle symlink issues that
   * {@link NonSkyframeGlobber} can't handle gracefully.
   */
  @Override
  protected void handleGlobDepsAndPropagateFilesystemExceptions(
      PackageIdentifier packageIdentifier,
      LoadedPackage loadedPackage,
      Environment env,
      boolean packageWasInError)
      throws InternalInconsistentFilesystemException, FileSymlinkException, InterruptedException {
    Set<SkyKey> depKeys = ((LoadedPackageWithGlobDeps) loadedPackage).globDepKeys;
    checkState(Iterables.all(depKeys, SkyFunctions.isSkyFunction(SkyFunctions.GLOB)), depKeys);
    FileSymlinkException arbitraryFse = null;
    SkyframeLookupResult result = env.getValuesAndExceptions(depKeys);
    for (SkyKey key : depKeys) {
      try {
        result.getOrThrow(key, IOException.class, BuildFileNotFoundException.class);
      } catch (InconsistentFilesystemException e) {
        throw new InternalInconsistentFilesystemException(packageIdentifier, e);
      } catch (FileSymlinkException e) {
        // Non-Skyframe globbing doesn't explicitly detect symlink issues, but certain filesystems
        // might detect some symlink issues. For example, many filesystems have a hardcoded bound on
        // the number of symlink hops they will follow when resolving paths (e.g. Unix's ELOOP).
        // Since Skyframe globbing does explicitly detect symlink issues, we are able to:
        //   (1) Provide a more informative error message.
        //   (2) Confidently act as though the symlink issue is non-transient.
        arbitraryFse = e;
      } catch (IOException | BuildFileNotFoundException e) {
        maybeThrowFilesystemInconsistency(packageIdentifier, e, packageWasInError);
      }
    }
    if (arbitraryFse != null) {
      // If there was at least one symlink issue and no inconsistent filesystem issues, arbitrarily
      // rethrow one of the symlink issues.
      throw arbitraryFse;
    }
  }

  private static class LoadedPackageWithGlobDeps extends LoadedPackage {
    private final Set<SkyKey> globDepKeys;

    private LoadedPackageWithGlobDeps(
        Package.Builder builder, Set<SkyKey> globDepKeys, long loadTimeNanos) {
      super(builder, loadTimeNanos);
      this.globDepKeys = globDepKeys;
    }
  }

  private interface GlobberWithSkyframeGlobDeps extends Globber {
    Set<SkyKey> getGlobDepsRequested();
  }

  private static class NonSkyframeGlobberWithNoGlobDeps implements GlobberWithSkyframeGlobDeps {
    private final NonSkyframeGlobber delegate;

    private NonSkyframeGlobberWithNoGlobDeps(NonSkyframeGlobber delegate) {
      this.delegate = delegate;
    }

    @Override
    public Set<SkyKey> getGlobDepsRequested() {
      return ImmutableSet.of();
    }

    @Override
    public Token runAsync(
        List<String> includes,
        List<String> excludes,
        Globber.Operation globberOperation,
        boolean allowEmpty)
        throws BadGlobException, InterruptedException {
      return delegate.runAsync(includes, excludes, globberOperation, allowEmpty);
    }

    @Override
    public List<String> fetchUnsorted(Token token)
        throws BadGlobException, IOException, InterruptedException {
      return delegate.fetchUnsorted(token);
    }

    @Override
    public void onInterrupt() {
      delegate.onInterrupt();
    }

    @Override
    public void onCompletion() {
      delegate.onCompletion();
    }
  }

  /**
   * A {@link Globber} implemented on top of Skyframe that falls back to a {@link
   * NonSkyframeGlobber} on a Skyframe cache-miss. This way we don't require a Skyframe restart
   * after a call to {@link Globber#runAsync} and before/during a call to {@link
   * Globber#fetchUnsorted}.
   *
   * <p>There are three advantages to this hybrid approach over the more obvious approach of solely
   * using a {@link NonSkyframeGlobber}:
   *
   * <ul>
   *   <li>We trivially have the proper Skyframe {@link GlobValue} deps, whereas we would need to
   *       request them after-the-fact if we solely used a {@link NonSkyframeGlobber}.
   *   <li>We don't need to re-evaluate globs whose expression hasn't changed (e.g. in the common
   *       case of a BUILD file edit that doesn't change a glob expression), whereas invoking the
   *       package loading machinery in {@link PackageFactory} with a {@link NonSkyframeGlobber}
   *       would naively re-evaluate globs when re-evaluating the BUILD file.
   *   <li>We don't need to re-evaluate invalidated globs *twice* (the single re-evaluation via our
   *       GlobValue deps is sufficient and optimal). See above for why the second evaluation would
   *       happen.
   * </ul>
   *
   * <p>One general disadvantage of the hybrid approach is that we do the logical globbing work
   * twice on clean builds. A part of this is that we do double the number of 'stat' filesystem
   * operations: non-Skyframe globbing does `stat` operations following symlinks, and Skyframe's
   * {@link FileStateFunction} does those operations not following symlinks (since {@link
   * FileFunction} handles symlink chains manually). We used to have a similar concern for `readdir`
   * operations, but we mitigated it by restructuring the non-Skyframe globbing code so that it
   * doesn't follow symlinks for these operations, allowing the results to be cached and used by
   * {@link DirectoryListingStateFunction}.
   *
   * <p>This theoretical inefficiency isn't a big deal in practice, and historical attempts to
   * completely remove it by solely using Skyframe's {@link GlobFunction} have been unsuccessful due
   * to the consequences of Skyframe restarts on package loading performance. If we knew the full
   * set of `glob` calls that would be performed during BUILD file evaluation, then we could
   * precompute those {@link GlobValue} nodes and not have any Skyframe restarts during. But that's
   * a big "if"; consider glob calls with non-static arguments:
   *
   * <pre>
   * P = f(42)
   * g(glob(P))
   * </pre>
   *
   * Also consider dependent glob calls:
   *
   * <pre>
   * L = glob(["foo.*"])
   * g(glob([f(x) for x in L])
   * </pre>
   *
   * One historical attempt at addressing this issue was to do a first pass of BUILD file evaluation
   * where we tried to encounter as many concrete glob calls as possible but without doing full
   * Starlark evaluation, and then do the real pass of BUILD file evaluation. This approach was a
   * net performance regression, due to the first pass both having non-trivial Starlark evaluation
   * cost (consider a very expensive function 'f' in the first example above) and also not
   * encountering all glob calls (meaning the real pass can still have the core problem with
   * Skyframe restarts).
   */
  private static class SkyframeHybridGlobber implements GlobberWithSkyframeGlobDeps {
    private final PackageIdentifier packageId;
    private final Root packageRoot;
    private final Environment env;
    private final NonSkyframeGlobber nonSkyframeGlobber;
    private final Set<SkyKey> globDepsRequested = Sets.newConcurrentHashSet();

    private SkyframeHybridGlobber(
        PackageIdentifier packageId,
        Root packageRoot,
        Environment env,
        NonSkyframeGlobber nonSkyframeGlobber) {
      this.packageId = packageId;
      this.packageRoot = packageRoot;
      this.env = env;
      this.nonSkyframeGlobber = nonSkyframeGlobber;
    }

    @Override
    public Set<SkyKey> getGlobDepsRequested() {
      return ImmutableSet.copyOf(globDepsRequested);
    }

    private SkyKey getGlobKey(String pattern, Globber.Operation globberOperation)
        throws BadGlobException {
      try {
        return GlobValue.key(
            packageId, packageRoot, pattern, globberOperation, PathFragment.EMPTY_FRAGMENT);
      } catch (InvalidGlobPatternException e) {
        throw new BadGlobException(e.getMessage());
      }
    }

    @Override
    public Token runAsync(
        List<String> includes,
        List<String> excludes,
        Globber.Operation globberOperation,
        boolean allowEmpty)
        throws BadGlobException, InterruptedException {
      LinkedHashSet<SkyKey> globKeys = Sets.newLinkedHashSetWithExpectedSize(includes.size());
      Map<SkyKey, String> globKeyToPatternMap = Maps.newHashMapWithExpectedSize(includes.size());

      for (String pattern : includes) {
        SkyKey globKey = getGlobKey(pattern, globberOperation);
        globKeys.add(globKey);
        globKeyToPatternMap.put(globKey, pattern);
      }

      globDepsRequested.addAll(globKeys);

      SkyframeLookupResult globValueMap = env.getValuesAndExceptions(globKeys);

      // For each missing glob, evaluate it asynchronously via the delegate.
      List<SkyKey> missingKeys = new ArrayList<>(globKeys.size());
      for (SkyKey globKey : globKeys) {
        try {
          SkyValue value =
              globValueMap.getOrThrow(globKey, IOException.class, BuildFileNotFoundException.class);
          if (value == null) {
            missingKeys.add(globKey);
          }
        } catch (IOException | BuildFileNotFoundException e) {
          logger.atWarning().withCause(e).log("Exception processing %s", globKey);
        }
      }
      List<String> globsToDelegate = new ArrayList<>(missingKeys.size());
      for (SkyKey missingKey : missingKeys) {
        String missingPattern = globKeyToPatternMap.get(missingKey);
        if (missingPattern != null) {
          globsToDelegate.add(missingPattern);
          globKeys.remove(missingKey);
        }
      }
      NonSkyframeGlobber.Token nonSkyframeIncludesToken =
          globsToDelegate.isEmpty()
              ? null
              : nonSkyframeGlobber.runAsync(
                  globsToDelegate, ImmutableList.of(), globberOperation, allowEmpty);
      return new HybridToken(
          globValueMap, globKeys, nonSkyframeIncludesToken, excludes, globberOperation, allowEmpty);
    }

    @Override
    public List<String> fetchUnsorted(Token token)
        throws BadGlobException, IOException, InterruptedException {
      HybridToken hybridToken = (HybridToken) token;
      return hybridToken.resolve(nonSkyframeGlobber);
    }

    @Override
    public void onInterrupt() {
      nonSkyframeGlobber.onInterrupt();
    }

    @Override
    public void onCompletion() {
      nonSkyframeGlobber.onCompletion();
    }

    /**
     * A {@link Globber.Token} that encapsulates the result of a single {@link Globber#runAsync}
     * call via the fetching of some globs from skyframe, and some other globs via a {@link
     * NonSkyframeGlobber}. 'exclude' patterns are evaluated using {@link UnixGlob#removeExcludes}
     * after merging the glob results in {@link #resolve}.
     */
    private static class HybridToken extends Globber.Token {
      // The result of the Skyframe lookup for all the needed glob patterns.
      private final SkyframeLookupResult globValueMap;
      // The skyframe keys corresponding to the 'includes' patterns fetched from Skyframe
      // (this is includes_sky above).
      private final Iterable<SkyKey> includesGlobKeys;
      @Nullable private final NonSkyframeGlobber.Token nonSkyframeGlobberIncludesToken;

      private final List<String> excludes;

      private final Globber.Operation globberOperation;

      private final boolean allowEmpty;

      private HybridToken(
          SkyframeLookupResult globValueMap,
          Iterable<SkyKey> includesGlobKeys,
          @Nullable NonSkyframeGlobber.Token nonSkyframeGlobberIncludesToken,
          List<String> excludes,
          Globber.Operation globberOperation,
          boolean allowEmpty) {
        this.globValueMap = globValueMap;
        this.includesGlobKeys = includesGlobKeys;
        this.nonSkyframeGlobberIncludesToken = nonSkyframeGlobberIncludesToken;
        this.excludes = excludes;
        this.globberOperation = globberOperation;
        this.allowEmpty = allowEmpty;
      }

      private List<String> resolve(NonSkyframeGlobber nonSkyframeGlobber)
          throws BadGlobException, IOException, InterruptedException {
        HashSet<String> matches = new HashSet<>();
        for (SkyKey includeGlobKey : includesGlobKeys) {
          // TODO(bazel-team): NestedSet expansion here is suboptimal.
          boolean foundMatch = false;
          for (PathFragment match : getGlobMatches(includeGlobKey, globValueMap)) {
            matches.add(match.getPathString());
            foundMatch = true;
          }
          if (!allowEmpty && !foundMatch) {
            GlobberUtils.throwBadGlobExceptionEmptyResult(
                ((GlobDescriptor) includeGlobKey.argument()).getPattern(), globberOperation);
          }
        }
        if (nonSkyframeGlobberIncludesToken != null) {
          matches.addAll(nonSkyframeGlobber.fetchUnsorted(nonSkyframeGlobberIncludesToken));
        }
        try {
          UnixGlob.removeExcludes(matches, excludes);
        } catch (UnixGlob.BadPattern ex) {
          throw new BadGlobException(ex.getMessage());
        }
        List<String> result = new ArrayList<>(matches);

        if (!allowEmpty && result.isEmpty()) {
          GlobberUtils.throwBadGlobExceptionAllExcluded(globberOperation);
        }
        return result;
      }

      private static ImmutableSet<PathFragment> getGlobMatches(
          SkyKey globKey, SkyframeLookupResult globValueMap) throws IOException {
        try {
          return checkNotNull(
              (GlobValue)
                  globValueMap.getOrThrow(
                      globKey, BuildFileNotFoundException.class, IOException.class),
              "%s should not be missing",
              globKey)
              .getMatches();
        } catch (BuildFileNotFoundException e) {
          throw new SkyframeGlobbingIOException(e);
        }
      }
    }
  }

  static class SkyframeGlobbingIOException extends IOException {
    SkyframeGlobbingIOException(BuildFileNotFoundException cause) {
      super(cause.getMessage(), cause);
    }
  }

  @Override
  protected GlobberWithSkyframeGlobDeps makeGlobber(
      NonSkyframeGlobber nonSkyframeGlobber,
      PackageIdentifier packageId,
      Root packageRoot,
      Environment env) {
    switch (globbingStrategy) {
      case SKYFRAME_HYBRID:
        return new SkyframeHybridGlobber(packageId, packageRoot, env, nonSkyframeGlobber);
      case NON_SKYFRAME:
        // Skyframe globbing is only useful for incremental correctness and performance. The
        // first time Bazel loads a package ever, Skyframe globbing is actually pure overhead
        // (SkyframeHybridGlobber will make full use of NonSkyframeGlobber).
        return new NonSkyframeGlobberWithNoGlobDeps(nonSkyframeGlobber);
    }
    throw new AssertionError(globbingStrategy);
  }

  @Override
  protected LoadedPackage newLoadedPackage(
      Package.Builder packageBuilder, @Nullable Globber globber, long loadTimeNanos) {
    Set<SkyKey> globDepKeys = ImmutableSet.of();
    if (globber != null) {
      globDepKeys = ((GlobberWithSkyframeGlobDeps) globber).getGlobDepsRequested();
    }
    return new LoadedPackageWithGlobDeps(packageBuilder, globDepKeys, loadTimeNanos);
  }
}
