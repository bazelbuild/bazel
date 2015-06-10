// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPatternResolver;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageIdentifier;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.pkgcache.RecursivePackageProvider;
import com.google.devtools.build.lib.pkgcache.TargetPatternResolverUtil;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;

/**
 * A {@link TargetPatternResolver} backed by a {@link RecursivePackageProvider}.
 */
public class RecursivePackageProviderBackedTargetPatternResolver
    implements TargetPatternResolver<Target> {

  private final RecursivePackageProvider recursivePackageProvider;
  private final EventHandler eventHandler;
  private final FilteringPolicy policy;
  private final PathPackageLocator pkgPath;

  public RecursivePackageProviderBackedTargetPatternResolver(
      final RecursivePackageProvider recursivePackageProvider,
      EventHandler eventHandler,
      FilteringPolicy policy,
      PathPackageLocator pkgPath) {
    this.recursivePackageProvider = recursivePackageProvider;
    this.eventHandler = eventHandler;
    this.policy = policy;
    this.pkgPath = pkgPath;
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
    Package pkg;
    try {
      pkg = recursivePackageProvider.getPackage(eventHandler, pkgIdentifier);
    } catch (NoSuchPackageException e) {
      pkg = e.getPackage();
      if (pkg == null) {
        throw e;
      }
    }
    return pkg;
  }

  @Override
  public Target getTargetOrNull(String targetName) throws InterruptedException {
    try {
      Label label = Label.parseAbsolute(targetName);
      if (!isPackage(label.getPackageName())) {
        return null;
      }
      return recursivePackageProvider.getTarget(eventHandler, label);
    } catch (Label.SyntaxException | NoSuchThingException e) {
      return null;
    }
  }

  @Override
  public ResolvedTargets<Target> getExplicitTarget(String targetName)
      throws TargetParsingException, InterruptedException {
    Label label = TargetPatternResolverUtil.label(targetName);
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
  public ResolvedTargets<Target> getTargetsInPackage(String originalPattern, String packageName,
                                                     boolean rulesOnly)
      throws TargetParsingException, InterruptedException {
    FilteringPolicy actualPolicy = rulesOnly
        ? FilteringPolicies.and(FilteringPolicies.RULES_ONLY, policy)
        : policy;
    return getTargetsInPackage(originalPattern, new PathFragment(packageName), actualPolicy);
  }

  private ResolvedTargets<Target> getTargetsInPackage(String originalPattern,
      PathFragment packageNameFragment, FilteringPolicy policy)
      throws TargetParsingException, InterruptedException {
    String packageName = packageNameFragment.toString();

    // It's possible for this check to pass, but for
    // Label.validatePackageNameFull to report an error because the
    // package name is illegal.  That's a little weird, but we can live with
    // that for now--see test case: testBadPackageNameButGoodEnoughForALabel.
    if (LabelValidator.validatePackageName(packageName) != null) {
      throw new TargetParsingException("'" + packageName + "' is not a valid package name");
    }
    if (!isPackage(packageName)) {
      throw new TargetParsingException(
          TargetPatternResolverUtil.getParsingErrorMessage(
              "no such package '" + packageName + "': BUILD file not found on package path",
              originalPattern));
    }

    try {
      Package pkg = getPackage(PackageIdentifier.createInDefaultRepo(packageNameFragment));
      return TargetPatternResolverUtil.resolvePackageTargets(pkg, policy);
    } catch (NoSuchThingException e) {
      String message = TargetPatternResolverUtil.getParsingErrorMessage(
          "package contains errors", originalPattern);
      throw new TargetParsingException(message, e);
    }
  }

  @Override
  public boolean isPackage(String packageName) {
    return recursivePackageProvider.isPackage(eventHandler, packageName);
  }

  @Override
  public String getTargetKind(Target target) {
    return target.getTargetKind();
  }

  @Override
  public ResolvedTargets<Target> findTargetsBeneathDirectory(String originalPattern,
      String directory, boolean rulesOnly, ImmutableSet<String> excludedSubdirectories)
      throws TargetParsingException, InterruptedException {
    FilteringPolicy actualPolicy = rulesOnly
        ? FilteringPolicies.and(FilteringPolicies.RULES_ONLY, policy)
        : policy;

    PathFragment pathFragment = getPathFragment(directory);

    ImmutableSet.Builder<PathFragment> excludedPathFragmentsBuilder = ImmutableSet.builder();
    for (String excludedDirectory : excludedSubdirectories) {
      excludedPathFragmentsBuilder.add(getPathFragment(excludedDirectory));
    }
    ImmutableSet<PathFragment> excludedPathFragments = excludedPathFragmentsBuilder.build();

    ResolvedTargets.Builder<Target> targetBuilder = ResolvedTargets.builder();
    for (Path root : pkgPath.getPathEntries()) {
      RootedPath rootedPath = RootedPath.toRootedPath(root, pathFragment);
      Iterable<PathFragment> packagesUnderDirectory =
          recursivePackageProvider.getPackagesUnderDirectory(rootedPath, excludedPathFragments);
      for (PathFragment pkg : packagesUnderDirectory) {
        targetBuilder.merge(getTargetsInPackage(originalPattern, pkg, FilteringPolicies.NO_FILTER));
      }
    }

    // Perform the no-targets-found check before applying the filtering policy so we only return the
    // error if the input directory's subtree really contains no targets.
    if (targetBuilder.isEmpty()) {
      throw new TargetParsingException("no targets found beneath '" + pathFragment + "'");
    }
    ResolvedTargets<Target> prefilteredTargets = targetBuilder.build();
     
    ResolvedTargets.Builder<Target> filteredBuilder = ResolvedTargets.builder();
    if (prefilteredTargets.hasError()) {
      filteredBuilder.setError();
    }
    for (Target target : prefilteredTargets.getTargets()) {
      if (actualPolicy.shouldRetain(target, false)) {
        filteredBuilder.add(target);
      }
    }
    return filteredBuilder.build();
  }

  private static PathFragment getPathFragment(String pathPrefix) throws TargetParsingException {
    PathFragment directory = new PathFragment(pathPrefix);
    if (directory.containsUplevelReferences()) {
      throw new TargetParsingException("up-level references are not permitted: '"
          + directory.getPathString() + "'");
    }
    if (!pathPrefix.isEmpty() && (LabelValidator.validatePackageName(pathPrefix) != null)) {
      throw new TargetParsingException("'" + pathPrefix + "' is not a valid package name");
    }
    return directory;
  }
}

