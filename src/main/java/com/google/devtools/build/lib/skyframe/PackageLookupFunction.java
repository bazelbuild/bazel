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
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicReference;

import javax.annotation.Nullable;

/**
 * SkyFunction for {@link PackageLookupValue}s.
 */
class PackageLookupFunction implements SkyFunction {

  private final AtomicReference<PathPackageLocator> pkgLocator;
  private final AtomicReference<ImmutableSet<String>> deletedPackages;

  PackageLookupFunction(AtomicReference<PathPackageLocator> pkgLocator,
      AtomicReference<ImmutableSet<String>> deletedPackages) {
    this.pkgLocator = pkgLocator;
    this.deletedPackages = deletedPackages;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws PackageLookupFunctionException {
    PathFragment pkg = (PathFragment) skyKey.argument();

    // This represents a package lookup at the package root.
    if (pkg.equals(PathFragment.EMPTY_FRAGMENT)) {
      return PackageLookupValue.invalidPackageName("The empty package name is invalid");
    }

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
    for (Path packagePathEntry : pkgLocator.get().getPathEntries()) {
      PathFragment buildFileFragment = null;
      if (pkgName.equals("external")) {
        buildFileFragment = new PathFragment("WORKSPACE");
      } else {
        buildFileFragment = pkg.getChild("BUILD");
      }
      RootedPath buildFileRootedPath = RootedPath.toRootedPath(packagePathEntry,
          buildFileFragment);
      String basename = buildFileRootedPath.asPath().getBaseName();
      SkyKey fileSkyKey = FileValue.key(buildFileRootedPath);
      FileValue fileValue = null;
      try {
        fileValue = (FileValue) env.getValueOrThrow(fileSkyKey, Exception.class);
      } catch (IOException e) {
        // TODO(bazel-team): throw an IOException here and let PackageFunction wrap that into a
        // BuildFileNotFoundException.
        throw new PackageLookupFunctionException(skyKey, new BuildFileNotFoundException(pkgName,
            "IO errors while looking for " + basename + " file reading "
                + buildFileRootedPath.asPath() + ": " + e.getMessage(), e));
      } catch (FileSymlinkCycleException e) {
        throw new PackageLookupFunctionException(skyKey,
            new BuildFileNotFoundException(pkgName, "Symlink cycle detected while trying to find "
                + basename + " file " + buildFileRootedPath.asPath()));
      } catch (InconsistentFilesystemException e) {
        throw new PackageLookupFunctionException(skyKey, e);
      } catch (Exception e) {
        throw new IllegalStateException("Not IOException of InconsistentFilesystemException", e);
      }
      if (fileValue == null) {
        return null;
      }
      if (fileValue.isFile()) {
        // Result does not depend on package path entries beyond the first match.
        return PackageLookupValue.success(packagePathEntry);
      }
    }
    return PackageLookupValue.noBuildFile();
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link PackageLookupFunction#compute}.
   */
  private static final class PackageLookupFunctionException extends SkyFunctionException {
    public PackageLookupFunctionException(SkyKey key, BuildFileNotFoundException e) {
      super(key, e);
    }

    public PackageLookupFunctionException(SkyKey key, InconsistentFilesystemException e) {
      super(key, e, /*isTransient=*/true);
    }
  }
}
