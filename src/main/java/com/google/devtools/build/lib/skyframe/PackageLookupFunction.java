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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.PackageIdentifier;
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
class PackageLookupFunction implements SkyFunction {

  private final AtomicReference<ImmutableSet<String>> deletedPackages;

  PackageLookupFunction(AtomicReference<ImmutableSet<String>> deletedPackages) {
    this.deletedPackages = deletedPackages;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws PackageLookupFunctionException {
    PathPackageLocator pkgLocator = PrecomputedValue.PATH_PACKAGE_LOCATOR.get(env);
    PackageIdentifier packageKey = (PackageIdentifier) skyKey.argument();
    if (!packageKey.getRepository().isDefault()) {
      return computeExternalPackageLookupValue(skyKey, env);
    }
    PathFragment pkg = packageKey.getPackageFragment();
    String pkgName = pkg.getPathString();
    String packageNameErrorMsg = LabelValidator.validatePackageName(pkgName);
    if (packageNameErrorMsg != null) {
      return PackageLookupValue.invalidPackageName("Invalid package name '" + pkgName + "': "
          + packageNameErrorMsg);
    }

    if (deletedPackages.get().contains(pkg.getPathString())) {
      return PackageLookupValue.deletedPackage();
    }

    // TODO(bazel-team): The following is O(n^2) on the number of elements on the package path due
    // to having restart the SkyFunction after every new dependency. However, if we try to batch
    // the missing value keys, more dependencies than necessary will be declared. This wart can be
    // fixed once we have nicer continuation support [skyframe-loading]
    for (Path packagePathEntry : pkgLocator.getPathEntries()) {
      PackageLookupValue value = getPackageLookupValue(env, packagePathEntry, pkg);
      if (value == null || value.packageExists()) {
        return value;
      }
    }
    return PackageLookupValue.noBuildFile();
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private PackageLookupValue getPackageLookupValue(Environment env, Path packagePathEntry,
      PathFragment pkgFragment) throws PackageLookupFunctionException {
    PathFragment buildFileFragment;
    boolean isWorkspace = false;
    if (pkgFragment.getPathString().equals(PackageFunction.EXTERNAL_PACKAGE_NAME)) {
      buildFileFragment = new PathFragment("WORKSPACE");
      isWorkspace = true;
    } else {
      buildFileFragment = pkgFragment.getChild("BUILD");
    }
    RootedPath buildFileRootedPath = RootedPath.toRootedPath(packagePathEntry,
        buildFileFragment);
    String basename = buildFileRootedPath.asPath().getBaseName();
    SkyKey fileSkyKey = FileValue.key(buildFileRootedPath);
    FileValue fileValue = null;
    try {
      fileValue = (FileValue) env.getValueOrThrow(fileSkyKey, IOException.class,
          FileSymlinkCycleException.class, InconsistentFilesystemException.class);
    } catch (IOException e) {
      String pkgName = pkgFragment.getPathString();
      // TODO(bazel-team): throw an IOException here and let PackageFunction wrap that into a
      // BuildFileNotFoundException.
      throw new PackageLookupFunctionException(new BuildFileNotFoundException(pkgName,
          "IO errors while looking for " + basename + " file reading "
              + buildFileRootedPath.asPath() + ": " + e.getMessage(), e),
           Transience.PERSISTENT);
    } catch (FileSymlinkCycleException e) {
      String pkgName = buildFileRootedPath.asPath().getPathString();
      throw new PackageLookupFunctionException(new BuildFileNotFoundException(pkgName,
          "Symlink cycle detected while trying to find " + basename + " file "
              + buildFileRootedPath.asPath()),
          Transience.PERSISTENT);
    } catch (InconsistentFilesystemException e) {
      // This error is not transient from the perspective of the PackageLookupFunction.
      throw new PackageLookupFunctionException(e, Transience.PERSISTENT);
    }
    if (fileValue == null) {
      return null;
    }
    if (fileValue.isFile() || isWorkspace) {
      return PackageLookupValue.success(buildFileRootedPath.getRoot());
    }
    return PackageLookupValue.noBuildFile();
  }

  /**
   * Gets a PackageLookupValue from a different Bazel repository.
   *
   * To do this, it looks up the "external" package and finds a path mapping for the repository
   * name.
   */
  private PackageLookupValue computeExternalPackageLookupValue(
      SkyKey skyKey, Environment env) throws PackageLookupFunctionException {
    PackageIdentifier id = (PackageIdentifier) skyKey.argument();
    SkyKey repositoryKey = RepositoryValue.key(id.getRepository());
    RepositoryValue repositoryValue = null;
    try {
      repositoryValue = (RepositoryValue) env.getValueOrThrow(
          repositoryKey, NoSuchPackageException.class, IOException.class, EvalException.class);
      if (repositoryValue == null) {
        return null;
      }
    } catch (NoSuchPackageException e) {
      throw new PackageLookupFunctionException(e, Transience.PERSISTENT);
    } catch (IOException | EvalException e) {
      throw new PackageLookupFunctionException(new BuildFileContainsErrorsException(
          PackageFunction.EXTERNAL_PACKAGE_NAME, e.getMessage()), Transience.PERSISTENT);
    }

    return getPackageLookupValue(env, repositoryValue.getPath(), id.getPackageFragment());
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link PackageLookupFunction#compute}.
   */
  private static final class PackageLookupFunctionException extends SkyFunctionException {
    public PackageLookupFunctionException(NoSuchPackageException e, Transience transience) {
      super(e, transience);
    }

    public PackageLookupFunctionException(InconsistentFilesystemException e,
        Transience transience) {
      super(e, transience);
    }
  }
}
