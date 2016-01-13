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

import static com.google.devtools.build.lib.pkgcache.FilteringPolicies.NO_FILTER;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPatternResolver;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.pkgcache.RecursivePackageProvider;
import com.google.devtools.build.lib.pkgcache.TargetPatternResolverUtil;
import com.google.devtools.build.lib.util.BatchCallback;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A {@link TargetPatternResolver} backed by a {@link RecursivePackageProvider}.
 */
public class RecursivePackageProviderBackedTargetPatternResolver
    implements TargetPatternResolver<Target> {

  // TODO(janakr): Move this to a more generic place and unify with SkyQueryEnvironment's value?
  private static final int MAX_PACKAGES_BULK_GET = 10000;

  private final RecursivePackageProvider recursivePackageProvider;
  private final EventHandler eventHandler;
  private final FilteringPolicy policy;

  public RecursivePackageProviderBackedTargetPatternResolver(
      RecursivePackageProvider recursivePackageProvider,
      EventHandler eventHandler,
      FilteringPolicy policy) {
    this.recursivePackageProvider = recursivePackageProvider;
    this.eventHandler = eventHandler;
    this.policy = policy;
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
    return recursivePackageProvider.bulkGetPackages(eventHandler, pkgIds);
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
    return getTargetsInPackage(originalPattern, packageIdentifier, actualPolicy);
  }

  private ResolvedTargets<Target> getTargetsInPackage(String originalPattern,
      PackageIdentifier packageIdentifier, FilteringPolicy policy)
      throws TargetParsingException, InterruptedException {
    try {
      Package pkg = getPackage(packageIdentifier);
      return TargetPatternResolverUtil.resolvePackageTargets(pkg, policy);
    } catch (NoSuchThingException e) {
      String message = TargetPatternResolverUtil.getParsingErrorMessage(
          e.getMessage(), originalPattern);
      throw new TargetParsingException(message, e);
    }
  }

  private Map<PackageIdentifier, ResolvedTargets<Target>> bulkGetTargetsInPackage(
          String originalPattern,
          Iterable<PackageIdentifier> pkgIds, FilteringPolicy policy)
          throws TargetParsingException, InterruptedException {
    try {
      Map<PackageIdentifier, Package> pkgs = bulkGetPackages(pkgIds);
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
      throw new TargetParsingException(message, e);
    }
  }

  @Override
  public boolean isPackage(PackageIdentifier packageIdentifier) {
    return recursivePackageProvider.isPackage(eventHandler, packageIdentifier);
  }

  @Override
  public String getTargetKind(Target target) {
    return target.getTargetKind();
  }

  @Override
  public <E extends Exception> void findTargetsBeneathDirectory(
      final RepositoryName repository,
      String originalPattern,
      String directory,
      boolean rulesOnly,
      ImmutableSet<PathFragment> excludedSubdirectories,
      BatchCallback<Target, E> callback)
      throws TargetParsingException, E, InterruptedException {
    FilteringPolicy actualPolicy = rulesOnly
        ? FilteringPolicies.and(FilteringPolicies.RULES_ONLY, policy)
        : policy;
    PathFragment pathFragment = TargetPatternResolverUtil.getPathFragment(directory);
    Iterable<PathFragment> packagesUnderDirectory =
        recursivePackageProvider.getPackagesUnderDirectory(
            repository, pathFragment, excludedSubdirectories);

    Iterable<PackageIdentifier> pkgIds = Iterables.transform(packagesUnderDirectory,
            new Function<PathFragment, PackageIdentifier>() {
              @Override
              public PackageIdentifier apply(PathFragment path) {
                return PackageIdentifier.create(repository, path);
              }
            });
    boolean foundTarget = false;
    // For very large sets of packages, we may not want to process all of them at once, so we split
    // into batches.
    for (Iterable<PackageIdentifier> pkgIdBatch :
        Iterables.partition(pkgIds, MAX_PACKAGES_BULK_GET)) {

      Iterable<ResolvedTargets<Target>> resolvedTargets =
          bulkGetTargetsInPackage(originalPattern, pkgIdBatch, NO_FILTER).values();
      List<Target> filteredTargets = new ArrayList<>(calculateSize(resolvedTargets));
      for (ResolvedTargets<Target> targets : resolvedTargets) {
        for (Target target : targets.getTargets()) {
          // Perform the no-targets-found check before applying the filtering policy so we only
          // return the error if the input directory's subtree really contains no targets.
          foundTarget = true;
          if (actualPolicy.shouldRetain(target, false)) {
            filteredTargets.add(target);
          }
        }
      }
      callback.process(filteredTargets);
    }

    if (!foundTarget) {
      throw new TargetParsingException("no targets found beneath '" + pathFragment + "'");
    }
  }

  private static <T> int calculateSize(Iterable<ResolvedTargets<T>> resolvedTargets) {
    int size = 0;
    for (ResolvedTargets<T> targets : resolvedTargets) {
      size += targets.getTargets().size();
    }
    return size;
  }
}

