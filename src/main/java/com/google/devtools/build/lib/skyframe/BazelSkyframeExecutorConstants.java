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
import com.google.devtools.build.lib.packages.BuildFileName;
import com.google.devtools.build.lib.skyframe.PackageFunction.ActionOnIOExceptionReadingBuildFile;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;

/** Hardcoded constants describing bazel-on-skyframe behavior. */
public class BazelSkyframeExecutorConstants {
  private BazelSkyframeExecutorConstants() {
  }

  public static final CrossRepositoryLabelViolationStrategy
      CROSS_REPOSITORY_LABEL_VIOLATION_STRATEGY = CrossRepositoryLabelViolationStrategy.ERROR;

  public static final ImmutableList<BuildFileName> BUILD_FILES_BY_PRIORITY =
      ImmutableList.of(BuildFileName.BUILD_DOT_BAZEL, BuildFileName.BUILD);

  public static final ActionOnIOExceptionReadingBuildFile
      ACTION_ON_IO_EXCEPTION_READING_BUILD_FILE =
          ActionOnIOExceptionReadingBuildFile.UseOriginalIOException.INSTANCE;
}
