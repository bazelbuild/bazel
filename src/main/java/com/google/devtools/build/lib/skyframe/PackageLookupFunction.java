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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicReference;

import javax.annotation.Nullable;

/**
 * SkyFunction for {@link PackageLookupValue}s.
 */
public class PackageLookupFunction implements SkyFunction {

  private final AtomicReference<ImmutableSet<PackageIdentifier>> deletedPackages;

  public PackageLookupFunction(AtomicReference<ImmutableSet<PackageIdentifier>> deletedPackages) {
    this.deletedPackages = deletedPackages;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws PackageLookupFunctionException {
    PathPackageLocator pkgLocator = PrecomputedValue.PATH_PACKAGE_LOCATOR.get(env);
    PackageIdentifier packageKey = (PackageIdentifier) skyKey.argument();
    if (PackageFunction.isDefaultsPackage(packageKey)) {
      return PackageLookupValue.success(pkgLocator.getPathEntries().get(0));
    }

    if (!packageKey.getRepository().isMain()) {
      return computeExternalPackageLookupValue(skyKey, env, packageKey);
    } else if (packageKey.equals(Label.EXTERNAL_PACKAGE_IDENTIFIER)) {
      return computeWorkspaceLookupValue(env, packageKey);
    }

    String packageNameErrorMsg = LabelValidator.validatePackageName(
        packageKey.getPackageFragment().getPathString());
    if (packageNameErrorMsg != null) {
      return PackageLookupValue.invalidPackageName("Invalid package name '" + packageKey + "': "
          + packageNameErrorMsg);
    }

    if (deletedPackages.get().contains(packageKey)) {
      return PackageLookupValue.DELETED_PACKAGE_VALUE;
    }

    BlacklistedPackagePrefixesValue blacklistedPatternsValue =
        (BlacklistedPackagePrefixesValue) env.getValue(BlacklistedPackagePrefixesValue.key());
    if (blacklistedPatternsValue == null) {
      return null;
    }

    PathFragment buildFileFragment = packageKey.getPackageFragment();
    for (PathFragment pattern : blacklistedPatternsValue.getPatterns()) {
      if (buildFileFragment.startsWith(pattern)) {
        return PackageLookupValue.DELETED_PACKAGE_VALUE;
      }
    }

    // TODO(bazel-team): The following is O(n^2) on the number of elements on the package path due
    // to having restart the SkyFunction after every new dependency. However, if we try to batch
    // the missing value keys, more dependencies than necessary will be declared. This wart can be
    // fixed once we have nicer continuation support [skyframe-loading]
    for (Path packagePathEntry : pkgLocator.getPathEntries()) {
      PackageLookupValue value = getPackageLookupValue(env, packagePathEntry, packageKey);
      if (value == null || value.packageExists()) {
        return value;
      }
    }
    return PackageLookupValue.NO_BUILD_FILE_VALUE;
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  @Nullable
  private FileValue getFileValue(
      RootedPath buildFileRootedPath, Environment env, PackageIdentifier packageIdentifier)
      throws PackageLookupFunctionException {
    String basename = buildFileRootedPath.asPath().getBaseName();
    SkyKey fileSkyKey = FileValue.key(buildFileRootedPath);
    FileValue fileValue = null;
    try {
      fileValue = (FileValue) env.getValueOrThrow(fileSkyKey, IOException.class,
          FileSymlinkException.class, InconsistentFilesystemException.class);
    } catch (IOException e) {
      // TODO(bazel-team): throw an IOException here and let PackageFunction wrap that into a
      // BuildFileNotFoundException.
      throw new PackageLookupFunctionException(new BuildFileNotFoundException(packageIdentifier,
          "IO errors while looking for " + basename + " file reading "
              + buildFileRootedPath.asPath() + ": " + e.getMessage(), e),
          Transience.PERSISTENT);
    } catch (FileSymlinkException e) {
      throw new PackageLookupFunctionException(new BuildFileNotFoundException(packageIdentifier,
          "Symlink cycle detected while trying to find " + basename + " file "
              + buildFileRootedPath.asPath()),
          Transience.PERSISTENT);
    } catch (InconsistentFilesystemException e) {
      // This error is not transient from the perspective of the PackageLookupFunction.
      throw new PackageLookupFunctionException(e, Transience.PERSISTENT);
    }
    return fileValue;
  }

  private PackageLookupValue getPackageLookupValue(Environment env, Path packagePathEntry,
      PackageIdentifier packageIdentifier) throws PackageLookupFunctionException {
    PathFragment buildFileFragment = packageIdentifier.getPackageFragment().getChild("BUILD");
    RootedPath buildFileRootedPath = RootedPath.toRootedPath(packagePathEntry,
        buildFileFragment);
    FileValue fileValue = getFileValue(buildFileRootedPath, env, packageIdentifier);
    if (fileValue == null) {
      return null;
    }
    if (fileValue.isFile()) {
      return PackageLookupValue.success(buildFileRootedPath.getRoot());
    }
    return PackageLookupValue.NO_BUILD_FILE_VALUE;
  }

  /**
   * Gets a PackageLookupValue from a different Bazel repository.
   *
   * <p>To do this, it looks up the "external" package and finds a path mapping for the repository
   * name.</p>
   */
  private PackageLookupValue computeExternalPackageLookupValue(
      SkyKey skyKey, Environment env, PackageIdentifier packageIdentifier)
      throws PackageLookupFunctionException {
    PackageIdentifier id = (PackageIdentifier) skyKey.argument();
    SkyKey repositoryKey = RepositoryValue.key(id.getRepository());
    RepositoryValue repositoryValue;
    try {
      repositoryValue = (RepositoryValue) env.getValueOrThrow(
          repositoryKey, NoSuchPackageException.class, IOException.class, EvalException.class);
      if (repositoryValue == null) {
        return null;
      }
    } catch (NoSuchPackageException | IOException | EvalException e) {
      throw new PackageLookupFunctionException(new BuildFileNotFoundException(id, e.getMessage()),
          Transience.PERSISTENT);
    }
    PathFragment buildFileFragment = id.getPackageFragment().getChild("BUILD");
    RootedPath buildFileRootedPath = RootedPath.toRootedPath(repositoryValue.getPath(),
        buildFileFragment);
    FileValue fileValue = getFileValue(buildFileRootedPath, env, packageIdentifier);
    if (fileValue == null) {
      return null;
    }

    if (fileValue.isFile()) {
        return PackageLookupValue.success(repositoryValue.getPath());
    }

    return PackageLookupValue.NO_BUILD_FILE_VALUE;
  }

  /**
   * Look for a WORKSPACE file on each package path.  If none is found, use the last package path
   * and pretend it was found there.
   */
  private SkyValue computeWorkspaceLookupValue(
      Environment env, PackageIdentifier packageIdentifier)
      throws PackageLookupFunctionException {
    PathPackageLocator pkgLocator = PrecomputedValue.PATH_PACKAGE_LOCATOR.get(env);
    Path lastPackagePath = null;
    for (Path packagePathEntry : pkgLocator.getPathEntries()) {
      lastPackagePath = packagePathEntry;
      RootedPath workspacePath = RootedPath.toRootedPath(
          packagePathEntry, Label.EXTERNAL_PACKAGE_FILE_NAME);
      FileValue value = getFileValue(workspacePath, env, packageIdentifier);
      if (value == null) {
        return null;
      }
      if (value.exists()) {
        return PackageLookupValue.workspace(packagePathEntry);
      }
    }

    // Fall back on the last package path entry if there were any and nothing else worked.
    return PackageLookupValue.workspace(lastPackagePath);
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link PackageLookupFunction#compute}.
   */
  private static final class PackageLookupFunctionException extends SkyFunctionException {
    public PackageLookupFunctionException(BuildFileNotFoundException e, Transience transience) {
      super(e, transience);
    }

    public PackageLookupFunctionException(InconsistentFilesystemException e,
        Transience transience) {
      super(e, transience);
    }
  }
}
