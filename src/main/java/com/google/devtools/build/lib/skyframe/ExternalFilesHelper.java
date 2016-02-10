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

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
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
    if (externalFileAction == ExternalFileAction.ERROR_OUT) {
      throw new FileOutsidePackageRootsException(rootedPath);
    }

    // The outputBase may be null if we're not actually running a build.
    Path outputBase = pkgLocator.get().getOutputBase();
    Path relativeExternal = pkgLocator
        .get().getOutputBase().getRelative(Label.EXTERNAL_PATH_PREFIX);
    if (outputBase != null && !rootedPath.asPath().startsWith(relativeExternal)) {
      return;
    }

    // For files that are under $OUTPUT_BASE/external, add a dependency on the corresponding rule
    // so that if the WORKSPACE file changes, the File/DirectoryStateValue will be re-evaluated.
    //
    // Note that:
    // - We don't add a dependency on the parent directory at the package root boundary, so
    // the only transitive dependencies from files inside the package roots to external files
    // are through symlinks. So the upwards transitive closure of external files is small.
    // - The only way other than external repositories for external source files to get into the
    // skyframe graph in the first place is through symlinks outside the package roots, which we
    // neither want to encourage nor optimize for since it is not common. So the set of external
    // files is small.

    PathFragment repositoryPath = rootedPath.asPath().relativeTo(relativeExternal);
    if (repositoryPath.segmentCount() == 0) {
      // We are the top of the repository path (<outputBase>/external), not in an actual external
      // repository path.
      return;
    }
    String repositoryName = repositoryPath.getSegment(0);

    try {
      RepositoryFunction.getRule(repositoryName, env);
    } catch (RepositoryFunction.RepositoryNotFoundException ex) {
      // The repository we are looking for does not exist so we should depend on the whole
      // WORKSPACE file. In that case, the call to RepositoryFunction#getRule(String, Environment)
      // already requested all repository functions from the WORKSPACE file from Skyframe as part of
      // the resolution. Therefore we are safe to ignore that Exception.
    } catch (RepositoryFunction.RepositoryFunctionException ex) {
      // This should never happen.
      throw new IllegalStateException(
          "Repository " + repositoryName + " cannot be resolved for path " + rootedPath, ex);
    }
  }
}
