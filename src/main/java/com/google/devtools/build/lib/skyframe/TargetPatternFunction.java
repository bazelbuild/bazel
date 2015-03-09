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

import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.cmdline.TargetPatternResolver;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageIdentifier;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.pkgcache.TargetPatternResolverUtil;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import javax.annotation.Nullable;

/**
 * TargetPatternFunction translates a target pattern (eg, "foo/...") into a set of resolved
 * Targets.
 */
public class TargetPatternFunction implements SkyFunction {

  private final AtomicReference<PathPackageLocator> pkgPath;

  public TargetPatternFunction(AtomicReference<PathPackageLocator> pkgPath) {
    this.pkgPath = pkgPath;
  }

  @Override
  public SkyValue compute(SkyKey key, Environment env) throws TargetPatternFunctionException,
      InterruptedException {
    TargetPatternValue.TargetPattern patternKey =
        ((TargetPatternValue.TargetPattern) key.argument());

    TargetPattern.Parser parser = new TargetPattern.Parser(patternKey.getOffset());
    try {
      Resolver resolver = new Resolver(env, patternKey.getPolicy(), pkgPath);
      TargetPattern resolvedPattern = parser.parse(patternKey.getPattern());
      return new TargetPatternValue(resolvedPattern.eval(resolver));
    } catch (TargetParsingException e) {
      throw new TargetPatternFunctionException(e);
    } catch (TargetPatternResolver.MissingDepException e) {
      return null;
    }
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private static class Resolver implements TargetPatternResolver<Target> {
    private final Environment env;
    private final FilteringPolicy policy;
    private final AtomicReference<PathPackageLocator> pkgPath;

    public Resolver(Environment env, FilteringPolicy policy,
                    AtomicReference<PathPackageLocator> pkgPath) {
      this.policy = policy;
      this.env = env;
      this.pkgPath = pkgPath;
    }

    @Override
    public void warn(String msg) {
      env.getListener().handle(Event.warn(msg));
    }

    /**
     * Gets a Package via the Skyframe env. May return a Package that has errors.
     */
    private Package getPackage(PackageIdentifier pkgIdentifier)
        throws MissingDepException, NoSuchThingException {
      SkyKey pkgKey = PackageValue.key(pkgIdentifier);
      Package pkg;
      try {
        PackageValue pkgValue =
            (PackageValue) env.getValueOrThrow(pkgKey, NoSuchPackageException.class);
        if (pkgValue == null) {
          throw new MissingDepException();
        }
        pkg = pkgValue.getPackage();
      } catch (NoSuchPackageException e) {
        pkg = e.getPackage();
        if (pkg == null) {
          throw e;
        }
      }
      return pkg;
    }

    @Override
    public Target getTargetOrNull(String targetName) throws InterruptedException,
        MissingDepException {
      try {
        Label label = Label.parseAbsolute(targetName);
        if (!isPackage(label.getPackageName())) {
          return null;
        }
        Package pkg = getPackage(label.getPackageIdentifier());
        return pkg.getTarget(label.getName());
      } catch (Label.SyntaxException | NoSuchThingException e) {
        return null;
      }
    }

    @Override
    public ResolvedTargets<Target> getExplicitTarget(String targetName)
        throws TargetParsingException, InterruptedException, MissingDepException {
      Label label = TargetPatternResolverUtil.label(targetName);
      try {
        Package pkg = getPackage(label.getPackageIdentifier());
        Target target = pkg.getTarget(label.getName());
        return  policy.shouldRetain(target, true)
            ? ResolvedTargets.of(target)
            : ResolvedTargets.<Target>empty();
      } catch (NoSuchThingException e) {
        throw new TargetParsingException(e.getMessage(), e);
      }
    }

    @Override
    public ResolvedTargets<Target> getTargetsInPackage(String originalPattern, String packageName,
                                                       boolean rulesOnly)
        throws TargetParsingException, InterruptedException, MissingDepException {
      FilteringPolicy actualPolicy = rulesOnly
          ? FilteringPolicies.and(FilteringPolicies.RULES_ONLY, policy)
          : policy;
      return getTargetsInPackage(originalPattern, packageName, actualPolicy);
    }

    private ResolvedTargets<Target> getTargetsInPackage(String originalPattern, String packageName,
                                                        FilteringPolicy policy)
        throws TargetParsingException, MissingDepException {
      // Normalise, e.g "foo//bar" -> "foo/bar"; "foo/" -> "foo":
      PathFragment packageNameFragment = new PathFragment(packageName);
      packageName = packageNameFragment.toString();

      // It's possible for this check to pass, but for
      // Label.validatePackageNameFull to report an error because the
      // package name is illegal.  That's a little weird, but we can live with
      // that for now--see test case: testBadPackageNameButGoodEnoughForALabel.
      // (BTW I tried duplicating that validation logic in Label but it was
      // extremely tricky.)
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
        Package pkg = getPackage(
            PackageIdentifier.createInDefaultRepo(packageNameFragment.toString()));
        return TargetPatternResolverUtil.resolvePackageTargets(pkg, policy);
      } catch (NoSuchThingException e) {
        String message = TargetPatternResolverUtil.getParsingErrorMessage(
            "package contains errors", originalPattern);
        throw new TargetParsingException(message, e);
      }
    }

    @Override
    public boolean isPackage(String packageName) throws MissingDepException {
      SkyKey packageLookupKey;
      packageLookupKey = PackageLookupValue.key(new PathFragment(packageName));
      PackageLookupValue packageLookupValue = (PackageLookupValue) env.getValue(packageLookupKey);
      if (packageLookupValue == null) {
        throw new MissingDepException();
      }
      return packageLookupValue.packageExists();
    }

    @Override
    public String getTargetKind(Target target) {
      return target.getTargetKind();
    }

    @Override
    public ResolvedTargets<Target> findTargetsBeneathDirectory(
        String originalPattern, String pathPrefix, boolean rulesOnly)
        throws TargetParsingException, MissingDepException {
      FilteringPolicy actualPolicy = rulesOnly
          ? FilteringPolicies.and(FilteringPolicies.RULES_ONLY, policy)
          : policy;

      PathFragment directory = new PathFragment(pathPrefix);
      if (directory.containsUplevelReferences()) {
        throw new TargetParsingException("up-level references are not permitted: '"
            + directory.getPathString() + "'");
      }
      if (!pathPrefix.isEmpty() && (LabelValidator.validatePackageName(pathPrefix) != null)) {
        throw new TargetParsingException("'" + pathPrefix + "' is not a valid package name");
      }

      ResolvedTargets.Builder<Target> builder = ResolvedTargets.builder();

      List<RecursivePkgValue> lookupValues = new ArrayList<>();
      for (Path root : pkgPath.get().getPathEntries()) {
        SkyKey key = RecursivePkgValue.key(RootedPath.toRootedPath(root, directory));
        RecursivePkgValue lookup = (RecursivePkgValue) env.getValue(key);
        if (lookup != null) {
          lookupValues.add(lookup);
        }
      }
      if (env.valuesMissing()) {
        throw new MissingDepException();
      }

      for (RecursivePkgValue value : lookupValues) {
        for (String pkg : value.getPackages()) {
          builder.merge(getTargetsInPackage(originalPattern, pkg, FilteringPolicies.NO_FILTER));
        }
      }

      if (builder.isEmpty()) {
        throw new TargetParsingException("no targets found beneath '" + directory + "'");
      }

      // Apply the transform after the check so we only return the
      // error if the tree really contains no targets.
      ResolvedTargets<Target> intermediateResult = builder.build();
      ResolvedTargets.Builder<Target> filteredBuilder = ResolvedTargets.builder();
      if (intermediateResult.hasError()) {
        filteredBuilder.setError();
      }
      for (Target target : intermediateResult.getTargets()) {
        if (actualPolicy.shouldRetain(target, false)) {
          filteredBuilder.add(target);
        }
      }
      return filteredBuilder.build();
    }
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link TargetPatternFunction#compute}.
   */
  private static final class TargetPatternFunctionException extends SkyFunctionException {
    public TargetPatternFunctionException(TargetParsingException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
