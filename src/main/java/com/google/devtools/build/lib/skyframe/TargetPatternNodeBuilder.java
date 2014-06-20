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
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.pkgcache.TargetPatternResolverUtil;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeBuilder;
import com.google.devtools.build.skyframe.NodeBuilderException;
import com.google.devtools.build.skyframe.NodeKey;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import javax.annotation.Nullable;

/**
 * TargetPatternNodeBuilder translates a target pattern (eg, "foo/...") into a set of resolved
 * Targets.
 */
public class TargetPatternNodeBuilder implements NodeBuilder {

  private final AtomicReference<PathPackageLocator> pkgPath;

  public TargetPatternNodeBuilder(AtomicReference<PathPackageLocator> pkgPath) {
    this.pkgPath = pkgPath;
  }

  @Override
  public Node build(NodeKey key, Environment env) throws TargetPatternNodeBuilderException,
      InterruptedException {
    TargetPatternNode.TargetPattern patternKey =
        ((TargetPatternNode.TargetPattern) key.getNodeName());

    TargetPattern.Parser parser = new TargetPattern.Parser(patternKey.getOffset());
    try {
      Resolver resolver = new Resolver(env, patternKey.getPolicy(), pkgPath);
      TargetPattern resolvedPattern = parser.parse(patternKey.getPattern());
      return new TargetPatternNode(resolvedPattern.eval(resolver));
    } catch (TargetParsingException e) {
      throw new TargetPatternNodeBuilderException(key, e);
    } catch (TargetPatternResolver.MissingDepException e) {
      return null;
    }
  }

  @Nullable
  @Override
  public String extractTag(NodeKey nodeKey) {
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
      env.getListener().warn(null, msg);
    }

    /**
     * Gets a Package via the Skyframe env. May return a Package that has errors.
     */
    private Package getPackage(PathFragment pkgName)
        throws MissingDepException, NoSuchThingException {
      NodeKey pkgKey = PackageNode.key(pkgName);
      Package pkg;
      try {
        PackageNode pkgNode =
            (PackageNode) env.getDepOrThrow(pkgKey, NoSuchThingException.class);
        if (pkgNode == null) {
          throw new MissingDepException();
        }
        pkg = pkgNode.getPackage();
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
        Package pkg = getPackage(label.getPackageFragment());
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
        Package pkg = getPackage(label.getPackageFragment());
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
        Package pkg = getPackage(packageNameFragment);
        return TargetPatternResolverUtil.resolvePackageTargets(pkg, policy);
      } catch (NoSuchThingException e) {
        String message = TargetPatternResolverUtil.getParsingErrorMessage(
            "package contains errors", originalPattern);
        throw new TargetParsingException(message, e);
      }
    }

    @Override
    public boolean isPackage(String packageName) throws MissingDepException {
      NodeKey packageLookupKey = PackageLookupNode.key(new PathFragment(packageName));
      PackageLookupNode packageLookupNode = (PackageLookupNode) env.getDep(packageLookupKey);
      if (packageLookupNode == null) {
        throw new MissingDepException();
      }
      return packageLookupNode.packageExists();
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

      List<RecursivePkgNode> lookupNodes = new ArrayList<>();
      for (Path root : pkgPath.get().getPathEntries()) {
        NodeKey key = RecursivePkgNode.key(RootedPath.toRootedPath(root, directory));
        RecursivePkgNode lookup = (RecursivePkgNode) env.getDep(key);
        if (lookup != null) {
          lookupNodes.add(lookup);
        }
      }
      if (env.depsMissing()) {
        throw new MissingDepException();
      }

      for (RecursivePkgNode node : lookupNodes) {
        for (Package pkg : node.getPackages()) {
          // TODO(bazel-team): Use the packages we just got from the RecursivePkgNodes instead of
          // throwing them away.
          builder.merge(
              getTargetsInPackage(originalPattern, pkg.getName(), FilteringPolicies.NO_FILTER));
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
   * {@link TargetPatternNodeBuilder#build}.
   */
  private static final class TargetPatternNodeBuilderException extends NodeBuilderException {
    public TargetPatternNodeBuilderException(NodeKey key, TargetParsingException e) {
      super(key, e);
    }
  }
}
