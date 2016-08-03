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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.PackageLookupValue.BuildFileName;
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
      return PackageLookupValue.success(pkgLocator.getPathEntries().get(0), BuildFileName.BUILD);
    }

    if (!packageKey.getRepository().isMain()) {
      return computeExternalPackageLookupValue(skyKey, env, packageKey);
    } else if (packageKey.equals(Label.EXTERNAL_PACKAGE_IDENTIFIER)) {
      return computeWorkspacePackageLookupValue(env, pkgLocator.getPathEntries());
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

    return getPackageLookupValue(env, pkgLocator.getPathEntries(), packageKey, BuildFileName.BUILD);
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  @Nullable
  private FileValue getFileValue(
      RootedPath fileRootedPath, Environment env, PackageIdentifier packageIdentifier)
      throws PackageLookupFunctionException {
    String basename = fileRootedPath.asPath().getBaseName();
    SkyKey fileSkyKey = FileValue.key(fileRootedPath);
    FileValue fileValue = null;
    try {
      fileValue = (FileValue) env.getValueOrThrow(fileSkyKey, IOException.class,
          FileSymlinkException.class, InconsistentFilesystemException.class);
    } catch (IOException e) {
      // TODO(bazel-team): throw an IOException here and let PackageFunction wrap that into a
      // BuildFileNotFoundException.
      throw new PackageLookupFunctionException(new BuildFileNotFoundException(packageIdentifier,
          "IO errors while looking for " + basename + " file reading "
              + fileRootedPath.asPath() + ": " + e.getMessage(), e),
          Transience.PERSISTENT);
    } catch (FileSymlinkException e) {
      throw new PackageLookupFunctionException(new BuildFileNotFoundException(packageIdentifier,
          "Symlink cycle detected while trying to find " + basename + " file "
              + fileRootedPath.asPath()),
          Transience.PERSISTENT);
    } catch (InconsistentFilesystemException e) {
      // This error is not transient from the perspective of the PackageLookupFunction.
      throw new PackageLookupFunctionException(e, Transience.PERSISTENT);
    }
    return fileValue;
  }

  private PackageLookupValue getPackageLookupValue(
      Environment env,
      ImmutableList<Path> packagePathEntries,
      PackageIdentifier packageIdentifier,
      BuildFileName buildFileName)
      throws PackageLookupFunctionException {
    // TODO(bazel-team): The following is O(n^2) on the number of elements on the package path due
    // to having restart the SkyFunction after every new dependency. However, if we try to batch
    // the missing value keys, more dependencies than necessary will be declared. This wart can be
    // fixed once we have nicer continuation support [skyframe-loading]
    for (Path packagePathEntry : packagePathEntries) {
      PathFragment buildFileFragment = buildFileName.getBuildFileFragment(packageIdentifier);
      RootedPath buildFileRootedPath = RootedPath.toRootedPath(packagePathEntry,
          buildFileFragment);
      FileValue fileValue = getFileValue(buildFileRootedPath, env, packageIdentifier);
      if (fileValue == null) {
        return null;
      }
      if (fileValue.isFile()) {
        return PackageLookupValue.success(buildFileRootedPath.getRoot(), buildFileName);
      }
    }
    return PackageLookupValue.NO_BUILD_FILE_VALUE;
  }

  private PackageLookupValue computeWorkspacePackageLookupValue(
      Environment env, ImmutableList<Path> packagePathEntries)
      throws PackageLookupFunctionException {
    PackageLookupValue result =
        getPackageLookupValue(
            env, packagePathEntries, Label.EXTERNAL_PACKAGE_IDENTIFIER, BuildFileName.WORKSPACE);
    if (result == null) {
      return null;
    }
    if (result.packageExists()) {
      return result;
    }
    // Fall back on the last package path entry if there were any and nothing else worked.
    // TODO(kchodorow): get rid of this, the semantics are wrong (successful package lookup should
    // mean the package exists). a bunch of tests need to be rewritten first though.
    if (packagePathEntries.isEmpty()) {
      return PackageLookupValue.NO_BUILD_FILE_VALUE;
    }
    Path lastPackagePath = packagePathEntries.get(packagePathEntries.size() - 1);
    FileValue lastPackagePackagePathFileValue = getFileValue(
        RootedPath.toRootedPath(lastPackagePath, PathFragment.EMPTY_FRAGMENT),
        env,
        Label.EXTERNAL_PACKAGE_IDENTIFIER);
    if (lastPackagePackagePathFileValue == null) {
      return null;
    }
    return lastPackagePackagePathFileValue.exists()
        ? PackageLookupValue.success(lastPackagePath, BuildFileName.WORKSPACE)
        : PackageLookupValue.NO_BUILD_FILE_VALUE;
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
    BuildFileName buildFileName = BuildFileName.BUILD;
    PathFragment buildFileFragment = id.getPackageFragment().getChild(buildFileName.getFilename());
    RootedPath buildFileRootedPath = RootedPath.toRootedPath(repositoryValue.getPath(),
        buildFileFragment);
    FileValue fileValue = getFileValue(buildFileRootedPath, env, packageIdentifier);
    if (fileValue == null) {
      return null;
    }

    if (fileValue.isFile()) {
      return PackageLookupValue.success(repositoryValue.getPath(), buildFileName);
    }

    return PackageLookupValue.NO_BUILD_FILE_VALUE;
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
