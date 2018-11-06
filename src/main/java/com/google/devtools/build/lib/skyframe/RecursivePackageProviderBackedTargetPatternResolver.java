// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPatternResolver;
import com.google.devtools.build.lib.concurrent.MultisetSemaphore;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.pkgcache.RecursivePackageProvider;
import com.google.devtools.build.lib.pkgcache.TargetPatternResolverUtil;
import com.google.devtools.build.lib.util.BatchCallback;
import com.google.devtools.build.lib.util.ThreadSafeBatchCallback;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

/**
 * A {@link TargetPatternResolver} backed by a {@link RecursivePackageProvider}.
 */
@ThreadCompatible
public class RecursivePackageProviderBackedTargetPatternResolver
    extends TargetPatternResolver<Target> {

  // TODO(janakr): Move this to a more generic place and unify with SkyQueryEnvironment's value?
  private static final int MAX_PACKAGES_BULK_GET = 1000;

  protected final FilteringPolicy policy;
  private final RecursivePackageProvider recursivePackageProvider;
  private final ExtendedEventHandler eventHandler;
  private final MultisetSemaphore<PackageIdentifier> packageSemaphore;

  public RecursivePackageProviderBackedTargetPatternResolver(
      RecursivePackageProvider recursivePackageProvider,
      ExtendedEventHandler eventHandler,
      FilteringPolicy policy,
      MultisetSemaphore<PackageIdentifier> packageSemaphore) {
    this.recursivePackageProvider = recursivePackageProvider;
    this.eventHandler = eventHandler;
    this.policy = policy;
    this.packageSemaphore = packageSemaphore;
  }

  @Override
  public void warn(String msg) {
    eventHandler.handle(Event.warn(msg));
  }

  /**
   * Gets a {@link Package} from the {@link RecursivePackageProvider}. May return a {@link Package}
   * that has errors.
   */
  private Package getPackage(PackageIdentifier pkgIdentifier)
      throws NoSuchPackageException, InterruptedException {
    return recursivePackageProvider.getPackage(eventHandler, pkgIdentifier);
  }

  private Map<PackageIdentifier, Package> bulkGetPackages(Iterable<PackageIdentifier> pkgIds)
          throws NoSuchPackageException, InterruptedException {
    return recursivePackageProvider.bulkGetPackages(pkgIds);
  }

  @Override
  public Target getTargetOrNull(Label label) throws InterruptedException {
    try {
      if (!isPackage(label.getPackageIdentifier())) {
        return null;
      }
      return recursivePackageProvider.getTarget(eventHandler, label);
    } catch (NoSuchThingException e) {
      return null;
    }
  }

  @Override
  public ResolvedTargets<Target> getExplicitTarget(Label label)
      throws TargetParsingException, InterruptedException {
    try {
      Target target = recursivePackageProvider.getTarget(eventHandler, label);
      return policy.shouldRetain(target, true)
          ? ResolvedTargets.of(target)
          : ResolvedTargets.<Target>empty();
    } catch (NoSuchThingException e) {
      throw new TargetParsingException(e.getMessage(), e);
    }
  }

  @Override
  public ResolvedTargets<Target> getTargetsInPackage(
      String originalPattern, PackageIdentifier packageIdentifier, boolean rulesOnly)
      throws TargetParsingException, InterruptedException {
    FilteringPolicy actualPolicy = rulesOnly
        ? FilteringPolicies.and(FilteringPolicies.RULES_ONLY, policy)
        : policy;
    try {
      Package pkg = getPackage(packageIdentifier);
      return TargetPatternResolverUtil.resolvePackageTargets(pkg, actualPolicy);
    } catch (NoSuchThingException e) {
      String message = TargetPatternResolverUtil.getParsingErrorMessage(
          e.getMessage(), originalPattern);
      throw new TargetParsingException(message, e);
    }
  }

  private Map<PackageIdentifier, ResolvedTargets<Target>> bulkGetTargetsInPackage(
          String originalPattern,
          Iterable<PackageIdentifier> pkgIds, FilteringPolicy policy)
          throws InterruptedException {
    try {
      Map<PackageIdentifier, Package> pkgs = bulkGetPackages(pkgIds);
      if (pkgs.size() != Iterables.size(pkgIds)) {
        throw new IllegalStateException("Bulk package retrieval missing results: "
            + Sets.difference(ImmutableSet.copyOf(pkgIds), pkgs.keySet()));
      }
      ImmutableMap.Builder<PackageIdentifier, ResolvedTargets<Target>> result =
              ImmutableMap.builder();
      for (PackageIdentifier pkgId : pkgIds) {
        Package pkg = pkgs.get(pkgId);
        result.put(pkgId,  TargetPatternResolverUtil.resolvePackageTargets(pkg, policy));
      }
      return result.build();
    } catch (NoSuchThingException e) {
      String message = TargetPatternResolverUtil.getParsingErrorMessage(
              e.getMessage(), originalPattern);
      throw new IllegalStateException(
          "Mismatch: Expected given pkgIds to correspond to valid Packages. " + message, e);
    }
  }

  @Override
  public boolean isPackage(PackageIdentifier packageIdentifier) throws InterruptedException {
    return recursivePackageProvider.isPackage(eventHandler, packageIdentifier);
  }

  @Override
  public String getTargetKind(Target target) {
    return target.getTargetKind();
  }

  /**
   * A {@link ThreadSafeBatchCallback} that trivially delegates to a {@link BatchCallback} in a
   * synchronized manner.
   */
  private static class SynchronizedBatchCallback<T, E extends Exception>
      implements ThreadSafeBatchCallback<T, E> {
    private final BatchCallback<T, E> delegate;

    public SynchronizedBatchCallback(BatchCallback<T, E> delegate) {
      this.delegate = delegate;
    }

    @Override
    public synchronized void process(Iterable<T> partialResult) throws E, InterruptedException {
      delegate.process(partialResult);
    }
  }

  @Override
  public <E extends Exception> void findTargetsBeneathDirectory(
      final RepositoryName repository,
      final String originalPattern,
      String directory,
      boolean rulesOnly,
      ImmutableSet<PathFragment> blacklistedSubdirectories,
      ImmutableSet<PathFragment> excludedSubdirectories,
      BatchCallback<Target, E> callback,
      Class<E> exceptionClass)
      throws TargetParsingException, E, InterruptedException {
    try {
      findTargetsBeneathDirectoryAsyncImpl(
          repository,
          originalPattern,
          directory,
          rulesOnly,
          blacklistedSubdirectories,
          excludedSubdirectories,
          new SynchronizedBatchCallback<Target, E>(callback),
          MoreExecutors.newDirectExecutorService()).get();
    } catch (ExecutionException e) {
      Throwable cause = e.getCause();
      Throwables.propagateIfPossible(cause, TargetParsingException.class, exceptionClass);
      throw new IllegalStateException(e.getCause());
    }
  }

  @Override
  public <E extends Exception> ListenableFuture<Void> findTargetsBeneathDirectoryAsync(
      RepositoryName repository,
      String originalPattern,
      String directory,
      boolean rulesOnly,
      ImmutableSet<PathFragment> blacklistedSubdirectories,
      ImmutableSet<PathFragment> excludedSubdirectories,
      ThreadSafeBatchCallback<Target, E> callback,
      Class<E> exceptionClass,
      ListeningExecutorService executor) {
    return findTargetsBeneathDirectoryAsyncImpl(
        repository,
        originalPattern,
        directory,
        rulesOnly,
        blacklistedSubdirectories,
        excludedSubdirectories,
        callback,
        executor);
  }

  private <E extends Exception> ListenableFuture<Void> findTargetsBeneathDirectoryAsyncImpl(
      final RepositoryName repository,
      final String originalPattern,
      String directory,
      boolean rulesOnly,
      ImmutableSet<PathFragment> blacklistedSubdirectories,
      ImmutableSet<PathFragment> excludedSubdirectories,
      final ThreadSafeBatchCallback<Target, E> callback,
      ListeningExecutorService executor) {
    final FilteringPolicy actualPolicy = rulesOnly
        ? FilteringPolicies.and(FilteringPolicies.RULES_ONLY, policy)
        : policy;
    final PathFragment pathFragment;
    Iterable<PathFragment> packagesUnderDirectory;
    try {
      pathFragment = TargetPatternResolverUtil.getPathFragment(directory);
      packagesUnderDirectory =
          recursivePackageProvider.getPackagesUnderDirectory(
              eventHandler,
              repository,
              pathFragment,
              blacklistedSubdirectories,
              excludedSubdirectories);
    } catch (TargetParsingException e) {
      return Futures.immediateFailedFuture(e);
    } catch (InterruptedException e) {
      return Futures.immediateCancelledFuture();
    }

    if (Iterables.isEmpty(packagesUnderDirectory)) {
      return Futures.immediateFailedFuture(
          new TargetParsingException("no targets found beneath '" + pathFragment + "'"));
    }

    Iterable<PackageIdentifier> pkgIds =
        Iterables.transform(
            packagesUnderDirectory, path -> PackageIdentifier.create(repository, path));

    // For very large sets of packages, we may not want to process all of them at once, so we split
    // into batches.
    List<List<PackageIdentifier>> partitions =
        ImmutableList.copyOf(Iterables.partition(pkgIds, MAX_PACKAGES_BULK_GET));
    ArrayList<ListenableFuture<Void>> futures = new ArrayList<>(partitions.size());
    for (final Iterable<PackageIdentifier> pkgIdBatch : partitions) {
      futures.add(
          executor.submit(
              () -> {
                ImmutableSet<PackageIdentifier> pkgIdBatchSet = ImmutableSet.copyOf(pkgIdBatch);
                packageSemaphore.acquireAll(pkgIdBatchSet);
                try {
                  Iterable<ResolvedTargets<Target>> resolvedTargets =
                      bulkGetTargetsInPackage(originalPattern, pkgIdBatch, actualPolicy).values();
                  List<Target> filteredTargets = new ArrayList<>(calculateSize(resolvedTargets));
                  for (ResolvedTargets<Target> targets : resolvedTargets) {
                    filteredTargets.addAll(targets.getTargets());
                  }
                  // TODO(bazel-core): Invoking the callback while holding onto the package
                  // semaphore can lead to deadlocks. Also, if the semaphore has a small count,
                  // acquireAll can also lead to problems if we don't batch appropriately.
                  // Although we default to an unbounded semaphore for SkyQuery and this is an
                  // unreported issue, consider refactoring so that the code is strictly correct.
                  callback.process(filteredTargets);
                } finally {
                  packageSemaphore.releaseAll(pkgIdBatchSet);
                }
                return null;
              }));
    }
    return Futures.whenAllSucceed(futures).call(() -> null, directExecutor());
  }

  private static <T> int calculateSize(Iterable<ResolvedTargets<T>> resolvedTargets) {
    int size = 0;
    for (ResolvedTargets<T> targets : resolvedTargets) {
      size += targets.getTargets().size();
    }
    return size;
  }
}

