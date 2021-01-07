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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.cmdline.TargetPatternResolver;
import com.google.devtools.build.lib.concurrent.BatchCallback;
import com.google.devtools.build.lib.concurrent.BatchCallback.NullCallback;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.AbstractRecursivePackageProvider.MissingDepException;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.pkgcache.TargetPatternResolverUtil;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;

/**
 * PrepareDepsOfPatternFunction ensures the graph loads targets matching the pattern and its
 * transitive dependencies.
 */
public class PrepareDepsOfPatternFunction implements SkyFunction {
  private final AtomicReference<PathPackageLocator> pkgPath;
  private final boolean traverseTestSuites;

  public PrepareDepsOfPatternFunction(
      AtomicReference<PathPackageLocator> pkgPath, boolean traverseTestSuites) {
    this.pkgPath = pkgPath;
    this.traverseTestSuites = traverseTestSuites;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey key, Environment env)
      throws SkyFunctionException, InterruptedException {
    TargetPatternValue.TargetPatternKey patternKey =
        ((TargetPatternValue.TargetPatternKey) key.argument());

    // DepsOfPatternPreparer below expects to be able to ignore the filtering policy from the
    // TargetPatternKey, which should be valid because PrepareDepsOfPatternValue.keys
    // unconditionally creates TargetPatternKeys with the NO_FILTER filtering policy. (Compare
    // with SkyframeTargetPatternEvaluator, which can create TargetPatternKeys with other
    // filtering policies like FILTER_TESTS or FILTER_MANUAL.) This check makes sure that the
    // key's filtering policy is NO_FILTER as expected.
    Preconditions.checkState(patternKey.getPolicy().equals(FilteringPolicies.NO_FILTER),
        patternKey.getPolicy());

    TargetPattern parsedPattern = patternKey.getParsedPattern();

    IgnoredPackagePrefixesValue repositoryIgnoredPrefixes =
        (IgnoredPackagePrefixesValue)
            env.getValue(IgnoredPackagePrefixesValue.key(parsedPattern.getRepository()));
    if (repositoryIgnoredPrefixes == null) {
      return null;
    }
    ImmutableSet<PathFragment> repositoryIgnoredPatterns = repositoryIgnoredPrefixes.getPatterns();

    // This SkyFunction is used to load the universe, so we want both the
    // ignored directories from the global list of exclusions (set with
    // .bazelignore in Bazel and set staticly in other binaries) and the
    // excluded directories from the TargetPatternKey itself to be embedded in
    // the SkyKeys created and used by the DepsOfPatternPreparer. The
    // DepsOfPatternPreparer ignores excludedSubdirectories and embeds
    // repositoryIgnoredSubdirectories in the SkyKeys it creates and uses.
    ImmutableSet<PathFragment> repositoryIgnoredSubdirectories =
        patternKey.getAllSubdirectoriesToExclude(repositoryIgnoredPatterns);
    ImmutableSet<PathFragment> excludedSubdirectories = ImmutableSet.of();

    DepsOfPatternPreparer preparer =
        new DepsOfPatternPreparer(env, pkgPath.get(), traverseTestSuites);

    try {
      parsedPattern.eval(
          preparer,
          repositoryIgnoredSubdirectories,
          excludedSubdirectories,
          NullCallback.<Void>instance(),
          RuntimeException.class);
    } catch (TargetParsingException e) {
      throw new PrepareDepsOfPatternFunctionException(e);
    } catch (MissingDepException e) {
      // The DepsOfPatternPreparer constructed above might throw MissingDepException to signal
      // when it has a dependency on a missing Environment value.
      return null;
    }
    return PrepareDepsOfPatternValue.INSTANCE;
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by {@link
   * PrepareDepsOfPatternFunction#compute}.
   */
  private static final class PrepareDepsOfPatternFunctionException extends SkyFunctionException {

    PrepareDepsOfPatternFunctionException(TargetParsingException e) {
      super(e, Transience.PERSISTENT);
    }
  }

  /**
   * A {@link TargetPatternResolver} backed by an {@link
   * com.google.devtools.build.skyframe.SkyFunction.Environment} whose methods do not actually
   * return resolved targets, but that ensures the graph loads the matching targets <b>and</b> their
   * transitive dependencies. Its methods may throw {@link MissingDepException} if the package
   * values this depends on haven't been calculated and added to its environment.
   */
  static class DepsOfPatternPreparer extends TargetPatternResolver<Void> {

    // Because PrepareDepsOfPatternFunction's only goal is to ensure the proper Skyframe nodes and
    // edges are in the graph, we don't need to worry about
    // EnvironmentBackedRecursivePackageProvider#encounteredPackageErrors.
    private final EnvironmentBackedRecursivePackageProvider packageProvider;
    private final Environment env;
    private final ImmutableList<Root> pkgRoots;
    private final boolean traverseTestSuites;

    DepsOfPatternPreparer(Environment env, PathPackageLocator pkgPath, boolean traverseTestSuites) {
      this.env = env;
      this.packageProvider = new EnvironmentBackedRecursivePackageProvider(env);
      this.pkgRoots = pkgPath.getPathEntries();
      this.traverseTestSuites = traverseTestSuites;
    }

    @Override
    public void warn(String msg) {
      env.getListener().handle(Event.warn(msg));
    }

    @Override
    public Void getTargetOrNull(Label label) throws InterruptedException {
      // Note:
      // This method is used in just one place, TargetPattern.TargetsInPackage#getWildcardConflict.
      // Returning null tells #getWildcardConflict that there is not a target with a name like
      // "all" or "all-targets", which means that TargetPattern.TargetsInPackage will end up
      // calling DepsOfTargetPreparer#getTargetsInPackage.
      // TODO (bazel-team): Consider replacing this with an isTarget method on the interface.
      return null;
    }

    @Override
    public ResolvedTargets<Void> getExplicitTarget(Label label)
        throws TargetParsingException, InterruptedException {
      try {
        Target target = packageProvider.getTarget(env.getListener(), label);
        SkyKey key = TransitiveTraversalValue.key(target.getLabel());
        SkyValue token =
            env.getValueOrThrow(key, NoSuchPackageException.class, NoSuchTargetException.class);
        if (token == null) {
          throw new MissingDepException();
        }
        return ResolvedTargets.empty();
      } catch (NoSuchThingException e) {
        throw new TargetParsingException(e.getMessage(), e, e.getDetailedExitCode());
      }
    }

    @Override
    public Collection<Void> getTargetsInPackage(
        String originalPattern, PackageIdentifier packageIdentifier, boolean rulesOnly)
        throws TargetParsingException, InterruptedException {
      FilteringPolicy policy =
          rulesOnly ? FilteringPolicies.RULES_ONLY : FilteringPolicies.NO_FILTER;
      return getTargetsInPackage(originalPattern, packageIdentifier, policy);
    }

    private Collection<Void> getTargetsInPackage(
        String originalPattern, PackageIdentifier packageIdentifier, FilteringPolicy policy)
        throws TargetParsingException, InterruptedException {
      try {
        Package pkg = packageProvider.getPackage(env.getListener(), packageIdentifier);
        Collection<Target> packageTargets =
            TargetPatternResolverUtil.resolvePackageTargets(pkg, policy);
        ImmutableList.Builder<SkyKey> builder = ImmutableList.builder();
        for (Target target : packageTargets) {
          builder.add(TransitiveTraversalValue.key(target.getLabel()));
        }
        ImmutableList<SkyKey> skyKeys = builder.build();
        env.getValuesOrThrow(skyKeys, NoSuchPackageException.class, NoSuchTargetException.class);
        if (env.valuesMissing()) {
          throw new MissingDepException();
        }
        return ImmutableSet.of();
      } catch (NoSuchThingException e) {
        String message = TargetPatternResolverUtil.getParsingErrorMessage(
            "package contains errors", originalPattern);
        throw new TargetParsingException(message, e, e.getDetailedExitCode());
      }
    }

    @Override
    public boolean isPackage(PackageIdentifier packageIdentifier) throws InterruptedException {
      return packageProvider.isPackage(env.getListener(), packageIdentifier);
    }

    @Override
    public String getTargetKind(Void target) {
      // Note:
      // This method is used in just one place, TargetPattern.TargetsInPackage#getWildcardConflict.
      // Because DepsOfPatternPreparer#getTargetOrNull always returns null, this method is never
      // called.
      throw new UnsupportedOperationException();
    }

    @Override
    public <E extends Exception> void findTargetsBeneathDirectory(
        RepositoryName repository,
        String originalPattern,
        String directory,
        boolean rulesOnly,
        ImmutableSet<PathFragment> repositoryIgnoredSubdirectories,
        ImmutableSet<PathFragment> excludedSubdirectories,
        BatchCallback<Void, E> callback,
        Class<E> exceptionClass)
        throws TargetParsingException, E, InterruptedException {
      PathFragment directoryPathFragment = TargetPatternResolverUtil.getPathFragment(directory);
      if (repositoryIgnoredSubdirectories.contains(directoryPathFragment)) {
        return;
      }
      Preconditions.checkArgument(excludedSubdirectories.isEmpty(), excludedSubdirectories);
      FilteringPolicy policy =
          rulesOnly ? FilteringPolicies.RULES_ONLY : FilteringPolicies.NO_FILTER;
      List<Root> roots = new ArrayList<>();
      if (repository.isMain()) {
        roots.addAll(pkgRoots);
      } else {
        RepositoryDirectoryValue repositoryValue =
            (RepositoryDirectoryValue) env.getValue(RepositoryDirectoryValue.key(repository));
        if (repositoryValue == null) {
          throw new MissingDepException();
        }

        if (!repositoryValue.repositoryExists()) {
          // This shouldn't be possible; we're given a repository, so we assume that the caller has
          // already checked for its existence.
          throw new IllegalStateException(String.format("No such repository '%s'", repository));
        }
        roots.add(Root.fromPath(repositoryValue.getPath()));
      }

      for (Root root : roots) {
        RootedPath rootedPath = RootedPath.toRootedPath(root, directoryPathFragment);
        env.getValues(getDeps(repository, repositoryIgnoredSubdirectories, policy, rootedPath));
        if (env.valuesMissing()) {
          throw new MissingDepException();
        }
      }
    }

    private ImmutableList<SkyKey> getDeps(
        RepositoryName repository,
        ImmutableSet<PathFragment> repositoryIgnoredSubdirectories,
        FilteringPolicy policy,
        RootedPath rootedPath) {
      List<SkyKey> keys = new ArrayList<>();
      keys.add(
          PrepareDepsOfTargetsUnderDirectoryValue.key(
              repository, rootedPath, repositoryIgnoredSubdirectories, policy));
      keys.add(
          CollectPackagesUnderDirectoryValue.key(
              repository, rootedPath, repositoryIgnoredSubdirectories));
      if (traverseTestSuites) {
        keys.add(
            PrepareTestSuitesUnderDirectoryValue.key(
                repository, rootedPath, repositoryIgnoredSubdirectories));
      }
      return ImmutableList.copyOf(keys);
    }
  }
}
