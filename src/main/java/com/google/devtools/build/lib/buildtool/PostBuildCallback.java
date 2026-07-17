// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import java.util.Collection;
import javax.annotation.Nullable;

/** Called by BuildTool after a successful invocation. */
public interface PostBuildCallback {

  /**
   * Hook for subclasses to execute after building has succeeded.
   *
   * @param successfulTargets The configured targets that have been built successfully.
   * @return on error returns the failure detail, on success null.
   */
  @Nullable
  FailureDetail process(Collection<ConfiguredTarget> successfulTargets) throws InterruptedException;
}
