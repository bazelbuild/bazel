// Copyright 2017 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.packages.BuildFileName;
import com.google.devtools.build.lib.repository.ExternalPackageHelper;
import com.google.devtools.build.lib.skyframe.PackageFunction.ActionOnIOExceptionReadingBuildFile;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;

/** Hardcoded constants describing bazel-on-skyframe behavior. */
public class BazelSkyframeExecutorConstants {
  private BazelSkyframeExecutorConstants() {}

  public static final ImmutableSet<PathFragment> HARDCODED_IGNORED_PACKAGE_PREFIXES =
      ImmutableSet.of();

  /**
   * The file .bazelignore can be used to specify directories to be ignored by bazel
   *
   * <p>This is intended for directories containing non-bazel sources (either generated, or
   * versioned sources built by other tools) that happen to contain a file called BUILD.
   *
   * <p>For the time being, this ignore functionality is limited by the fact that it is applied only
   * after pattern expansion. So if a pattern expansion fails (e.g., due to symlink-cycles) and
   * therefore fails the build, this ignore functionality currently has no chance to kick in.
   */
  public static final SkyFunction IGNORED_PACKAGE_PREFIXES_FUNCTION =
      new IgnoredPackagePrefixesFunction(PathFragment.create(".bazelignore"));

  public static final CrossRepositoryLabelViolationStrategy
      CROSS_REPOSITORY_LABEL_VIOLATION_STRATEGY = CrossRepositoryLabelViolationStrategy.ERROR;

  public static final ImmutableList<BuildFileName> BUILD_FILES_BY_PRIORITY =
      ImmutableList.of(BuildFileName.BUILD_DOT_BAZEL, BuildFileName.BUILD);

  private static final ImmutableList<BuildFileName> WORKSPACE_FILES_BY_PRIORITY =
      ImmutableList.of(BuildFileName.WORKSPACE_DOT_BAZEL, BuildFileName.WORKSPACE);

  public static final ExternalPackageHelper EXTERNAL_PACKAGE_HELPER =
      new ExternalPackageHelper(WORKSPACE_FILES_BY_PRIORITY);

  public static final ActionOnIOExceptionReadingBuildFile
      ACTION_ON_IO_EXCEPTION_READING_BUILD_FILE =
          ActionOnIOExceptionReadingBuildFile.UseOriginalIOException.INSTANCE;

  public static final boolean USE_REPO_DOT_BAZEL = true;

  public static SequencedSkyframeExecutor.Builder newBazelSkyframeExecutorBuilder() {
    return SequencedSkyframeExecutor.builder()
        .setIgnoredPackagePrefixesFunction(IGNORED_PACKAGE_PREFIXES_FUNCTION)
        .setActionOnIOExceptionReadingBuildFile(ACTION_ON_IO_EXCEPTION_READING_BUILD_FILE)
        .setShouldUseRepoDotBazel(USE_REPO_DOT_BAZEL)
        .setCrossRepositoryLabelViolationStrategy(CROSS_REPOSITORY_LABEL_VIOLATION_STRATEGY)
        .setBuildFilesByPriority(BUILD_FILES_BY_PRIORITY)
        .setExternalPackageHelper(EXTERNAL_PACKAGE_HELPER);
  }
}
