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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;

import java.util.concurrent.atomic.AtomicReference;

/** Common utilities for dealing with files outside the package roots. */
public class ExternalFilesHelper {

  private final AtomicReference<PathPackageLocator> pkgLocator;
  private final ExternalFileAction externalFileAction;

  // This variable is set to true from multiple threads, but only read once, in the main thread.
  // So volatility or an AtomicBoolean is not needed.
  private boolean externalFileSeen = false;

  /**
   * @param pkgLocator an {@link AtomicReference} to a {@link PathPackageLocator} used to
   *    determine what files are internal.
   * @param errorOnExternalFiles If files outside of package paths should be allowed.
   */
  public ExternalFilesHelper(
      AtomicReference<PathPackageLocator> pkgLocator, boolean errorOnExternalFiles) {
    this.pkgLocator = pkgLocator;
    if (errorOnExternalFiles) {
      this.externalFileAction = ExternalFileAction.ERROR_OUT;
    } else {
      this.externalFileAction = ExternalFileAction.DEPEND_ON_EXTERNAL_PKG;
    }
  }

  private enum ExternalFileAction {
    // Re-check the files when the WORKSPACE file changes.
    DEPEND_ON_EXTERNAL_PKG,

    // Throw an exception if there is an external file.
    ERROR_OUT,
  }

  boolean isExternalFileSeen() {
    return externalFileSeen;
  }

  static boolean isInternal(RootedPath rootedPath, PathPackageLocator packageLocator) {
    // TODO(bazel-team): This is inefficient when there are a lot of package roots or there are a
    // lot of external directories. Consider either explicitly preventing this case or using a more
    // efficient approach here (e.g. use a trie for determining if a file is under an external
    // directory).
    return packageLocator.getPathEntries().contains(rootedPath.getRoot());
  }

  /**
   * If this instance is configured with DEPEND_ON_EXTERNAL_PKG and rootedPath is a file that isn't
   * under a package root then this adds a dependency on the //external package. If the action is
   * ERROR_OUT, it will throw an error instead.
   */
  public void maybeHandleExternalFile(RootedPath rootedPath, SkyFunction.Environment env)
      throws FileOutsidePackageRootsException {
    if (isInternal(rootedPath, pkgLocator.get())) {
      return;
    }

    externalFileSeen = true;
    if (externalFileAction == ExternalFileAction.DEPEND_ON_EXTERNAL_PKG) {
      // For files outside the package roots, add a dependency on the //external package so that if
      // the WORKSPACE file changes, the File/DirectoryStateValue will be re-evaluated.
      //
      // Note that:
      // - We don't add a dependency on the parent directory at the package root boundary, so
      // the only transitive dependencies from files inside the package roots to external files
      // are through symlinks. So the upwards transitive closure of external files is small.
      // - The only way other than external repositories for external source files to get into the
      // skyframe graph in the first place is through symlinks outside the package roots, which we
      // neither want to encourage nor optimize for since it is not common. So the set of external
      // files is small.
      // TODO(kchodorow): check that the path is under output_base/external before adding the dep.
      PackageValue pkgValue = (PackageValue) env.getValue(PackageValue.key(
              PackageIdentifier.createInDefaultRepo(PackageIdentifier.EXTERNAL_PREFIX)));
      if (pkgValue == null) {
        return;
      }
      Preconditions.checkState(!pkgValue.getPackage().containsErrors());
    } else {
      throw new FileOutsidePackageRootsException(rootedPath);
    }
  }
}
