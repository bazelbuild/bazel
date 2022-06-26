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

package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.io.InconsistentFilesystemException;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Represents logic that evaluates the root of the package containing an exec path, for constructing
 * {@link Artifact} objects out of the action cache.
 */
public interface PackageRootResolver {

  /**
   * Returns mapping from execPath to Root. Root will be null if the path has no containing package.
   *
   * @param execPaths the paths to find {@link Root}s for. The search for a containing package will
   *     start with the path's parent directory, since the path is assumed to be a file.
   * @return mappings from {@code execPath} to {@link Root}, or null if for some reason we cannot
   *     determine the result at this time (such as when used within a SkyFunction)
   */
  @Nullable
  Map<PathFragment, Root> findPackageRootsForFiles(Iterable<PathFragment> execPaths)
      throws PackageRootException, InterruptedException;

  /**
   * Exception encapsulating a failure to find a package root in {@link
   * com.google.devtools.build.lib.skyframe.PackageLookupFunction} (via {@link
   * com.google.devtools.build.lib.skyframe.ContainingPackageLookupFunction}). Contains a {@link
   * FailureDetails.IncludeScanning} error for use in a {@link DetailedExitCode}.
   */
  class PackageRootException extends Exception {
    private final FailureDetails.IncludeScanning error;

    private PackageRootException(
        PathFragment execPath, FailureDetails.IncludeScanning error, Exception e) {
      super(
          "Unable to resolve " + execPath.getPathString() + " as an artifact: " + e.getMessage(),
          e);
      this.error = error;
    }

    public static PackageRootException create(PathFragment execPath, BuildFileNotFoundException e) {
      FailureDetails.FailureDetail failureDetail = e.getDetailedExitCode().getFailureDetail();
      FailureDetails.IncludeScanning.Code code =
          e.hasExplicitDetailedExitCode()
              ? DetailedExitCode.getExitCode(failureDetail).isInfrastructureFailure()
                  ? FailureDetails.IncludeScanning.Code.SYSTEM_PACKAGE_LOAD_FAILURE
                  : FailureDetails.IncludeScanning.Code.USER_PACKAGE_LOAD_FAILURE
              : FailureDetails.IncludeScanning.Code.UNDIFFERENTIATED_PACKAGE_LOAD_FAILURE;
      FailureDetails.PackageLoading.Code packageLoadingCode =
          failureDetail.getPackageLoading().getCode();
      if (packageLoadingCode == FailureDetails.PackageLoading.Code.PACKAGE_LOADING_UNKNOWN) {
        BugReport.sendBugReport(
            new IllegalStateException(
                "Exception for " + execPath + " had no PackageLoading.Code: " + failureDetail, e));
      }
      return new PackageRootException(
          execPath,
          FailureDetails.IncludeScanning.newBuilder()
              .setCode(code)
              .setPackageLoadingCode(packageLoadingCode)
              .build(),
          e);
    }

    public static PackageRootException create(
        PathFragment execPath, InconsistentFilesystemException e) {
      return new PackageRootException(
          execPath,
          FailureDetails.IncludeScanning.newBuilder()
              .setCode(FailureDetails.IncludeScanning.Code.SYSTEM_PACKAGE_LOAD_FAILURE)
              .setPackageLoadingCode(
                  FailureDetails.PackageLoading.Code.PERSISTENT_INCONSISTENT_FILESYSTEM_ERROR)
              .build(),
          e);
    }

    public FailureDetails.IncludeScanning getError() {
      return error;
    }
  }
}
