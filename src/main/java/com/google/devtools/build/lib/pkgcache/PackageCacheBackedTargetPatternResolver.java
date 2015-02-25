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

package com.google.devtools.build.lib.pkgcache;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPatternResolver;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageIdentifier;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.concurrent.ThreadPoolExecutor;

/**
 * An implementation of the {@link TargetPatternResolver} that uses the {@link
 * RecursivePackageProvider} as the backing implementation.
 */
final class PackageCacheBackedTargetPatternResolver implements TargetPatternResolver<Target> {

  private final RecursivePackageProvider packageProvider;
  private final EventHandler eventHandler;
  private final boolean keepGoing;
  private final FilteringPolicy policy;
  private final ThreadPoolExecutor packageVisitorPool;

  PackageCacheBackedTargetPatternResolver(RecursivePackageProvider packageProvider,
      EventHandler eventHandler, boolean keepGoing, FilteringPolicy policy,
      ThreadPoolExecutor packageVisitorPool) {
    this.packageProvider = packageProvider;
    this.eventHandler = eventHandler;
    this.keepGoing = keepGoing;
    this.policy = policy;
    this.packageVisitorPool = packageVisitorPool;
  }

  @Override
  public void warn(String msg) {
    eventHandler.handle(Event.warn(msg));
  }

  @Override
  public Target getTargetOrNull(String targetName) throws InterruptedException {
    try {
      return packageProvider.getTarget(eventHandler, Label.parseAbsolute(targetName));
    } catch (NoSuchPackageException | NoSuchTargetException | Label.SyntaxException e) {
      return null;
    }
  }

  @Override
  public ResolvedTargets<Target> getExplicitTarget(String targetName)
      throws TargetParsingException, InterruptedException {
    Label label = TargetPatternResolverUtil.label(targetName);
    return getExplicitTarget(label, targetName);
  }

  private ResolvedTargets<Target> getExplicitTarget(Label label, String originalLabel)
      throws TargetParsingException, InterruptedException {
    try {
      Target target = packageProvider.getTarget(eventHandler, label);
      if (policy.shouldRetain(target, true)) {
        return ResolvedTargets.of(target);
      }
      return ResolvedTargets.<Target>empty();
    } catch (BuildFileContainsErrorsException e) {
      // We don't need to report an error here because errors
      // would have already been reported in this case.
      return handleParsingError(eventHandler, originalLabel,
          new TargetParsingException(e.getMessage(), e), keepGoing);
    } catch (NoSuchThingException e) {
      return handleParsingError(eventHandler, originalLabel,
          new TargetParsingException(e.getMessage(), e), keepGoing);
    }
  }

  /**
   * Handles an error differently based on the value of keepGoing.
   *
   * @param badPattern The pattern we were unable to parse.
   * @param e The underlying exception.
   * @param keepGoing It true, report a warning and return.
   *     If false, throw the exception.
   * @return the empty set.
   * @throws TargetParsingException if !keepGoing.
   */
  private ResolvedTargets<Target> handleParsingError(EventHandler eventHandler, String badPattern,
      TargetParsingException e, boolean keepGoing) throws TargetParsingException {
    if (eventHandler instanceof ParseFailureListener) {
      ((ParseFailureListener) eventHandler).parsingError(badPattern, e.getMessage());
    }
    if (keepGoing) {
      eventHandler.handle(Event.error("Skipping '" + badPattern + "': " + e.getMessage()));
      return ResolvedTargets.<Target>failed();
    } else {
      throw e;
    }
  }

  @Override
  public ResolvedTargets<Target> getTargetsInPackage(String originalPattern, String packageName,
      boolean rulesOnly) throws TargetParsingException, InterruptedException {
    FilteringPolicy actualPolicy = rulesOnly
        ? FilteringPolicies.and(FilteringPolicies.RULES_ONLY, policy)
        : policy;
    return getTargetsInPackage(originalPattern, packageName, actualPolicy);
  }

  private ResolvedTargets<Target> getTargetsInPackage(String originalPattern, String packageName,
      FilteringPolicy policy) throws TargetParsingException, InterruptedException {
    // Normalise, e.g "foo//bar" -> "foo/bar"; "foo/" -> "foo":
    packageName = new PathFragment(packageName).toString();

    // it's possible for this check to pass, but for Label.validatePackageNameFull to report an
    // error because the package name is illegal.  That's a little weird, but we can live with
    // that for now--see test case: testBadPackageNameButGoodEnoughForALabel. (BTW I tried
    // duplicating that validation logic in Label but it was extremely tricky.)
    if (LabelValidator.validatePackageName(packageName) != null) {
      return handleParsingError(eventHandler, originalPattern,
                                new TargetParsingException(
                                  "'" + packageName + "' is not a valid package name"), keepGoing);
    }
    Package pkg;
    try {
      pkg = packageProvider.getPackage(
          eventHandler, PackageIdentifier.createInDefaultRepo(packageName));
    } catch (NoSuchPackageException e) {
      return handleParsingError(eventHandler, originalPattern, new TargetParsingException(
          TargetPatternResolverUtil.getParsingErrorMessage(
              e.getMessage(), originalPattern)), keepGoing);
    }

    if (pkg.containsErrors()) {
      // Report an error, but continue (and return partial results) if keepGoing is specified.
      handleParsingError(eventHandler, originalPattern, new TargetParsingException(
          TargetPatternResolverUtil.getParsingErrorMessage(
              "package contains errors", originalPattern)), keepGoing);
    }

    return TargetPatternResolverUtil.resolvePackageTargets(pkg, policy);
  }

  @Override
  public ResolvedTargets<Target> findTargetsBeneathDirectory(String originalPattern,
      String pathPrefix, boolean rulesOnly) throws TargetParsingException, InterruptedException {
    FilteringPolicy actualPolicy = rulesOnly
        ? FilteringPolicies.and(FilteringPolicies.RULES_ONLY, policy)
        : policy;
    return findTargetsBeneathDirectory(eventHandler, originalPattern, pathPrefix, actualPolicy,
        keepGoing, pathPrefix.isEmpty());
  }

  private ResolvedTargets<Target> findTargetsBeneathDirectory(final EventHandler eventHandler,
      final String originalPattern, String pathPrefix, final FilteringPolicy policy,
      final boolean keepGoing, boolean useTopLevelExcludes)
      throws TargetParsingException, InterruptedException {
    PathFragment directory = new PathFragment(pathPrefix);
    if (directory.containsUplevelReferences()) {
      throw new TargetParsingException("up-level references are not permitted: '"
          + pathPrefix + "'");
    }
    if (!pathPrefix.isEmpty() && (LabelValidator.validatePackageName(pathPrefix) != null)) {
      return handleParsingError(eventHandler, pathPrefix, new TargetParsingException(
          "'" + pathPrefix + "' is not a valid package name"), keepGoing);
    }

    final ResolvedTargets.Builder<Target> builder = ResolvedTargets.concurrentBuilder();
    try {
      packageProvider.visitPackageNamesRecursively(eventHandler, directory,
          useTopLevelExcludes, packageVisitorPool,
          new PathPackageLocator.AcceptsPathFragment() {
            @Override
            public void accept(PathFragment packageName) {
              String pkgName = packageName.getPathString();
              try {
                // Get the targets without transforming. We'll do that later below.
                builder.merge(getTargetsInPackage(originalPattern, pkgName,
                    FilteringPolicies.NO_FILTER));
              } catch (InterruptedException e) {
                throw new RuntimeParsingException(new TargetParsingException("interrupted"));
              } catch (TargetParsingException e) {
                // We'd like to make visitPackageNamesRecursively() generic
                // over some checked exception type (TargetParsingException in
                // this case). To do so, we'd have to make AbstractQueueVisitor
                // generic over the same exception type. That won't work due to
                // type erasure. As a workaround, we wrap the exception here,
                // and unwrap it below.
                throw new RuntimeParsingException(e);
              }
            }
          });
    } catch (RuntimeParsingException e) {
      throw e.unwrap();
    } catch (UnsupportedOperationException e) {
      throw new TargetParsingException("recursive target patterns are not permitted: '"
          + originalPattern + "'");
    }

    if (builder.isEmpty()) {
      return handleParsingError(eventHandler, originalPattern,
          new TargetParsingException("no targets found beneath '" + directory + "'"),
          keepGoing);
    }

    // Apply the transform after the check so we only return the
    // error if the tree really contains no targets.
    ResolvedTargets<Target> intermediateResult = builder.build();
    ResolvedTargets.Builder<Target> filteredBuilder = ResolvedTargets.builder();
    if (intermediateResult.hasError()) {
      filteredBuilder.setError();
    }
    for (Target target : intermediateResult.getTargets()) {
      if (policy.shouldRetain(target, false)) {
        filteredBuilder.add(target);
      }
    }
    return filteredBuilder.build();
  }

  @Override
  public boolean isPackage(String packageName) {
    return packageProvider.isPackage(packageName);
  }

  @Override
  public String getTargetKind(Target target) {
    return target.getTargetKind();
  }

  private static final class RuntimeParsingException extends RuntimeException {
    private TargetParsingException parsingException;

    public RuntimeParsingException(TargetParsingException cause) {
      super(cause);
      this.parsingException = Preconditions.checkNotNull(cause);
    }

    public TargetParsingException unwrap() {
      return parsingException;
    }
  }
}
