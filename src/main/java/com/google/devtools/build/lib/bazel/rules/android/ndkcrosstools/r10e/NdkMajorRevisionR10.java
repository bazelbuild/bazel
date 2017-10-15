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

package com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.r10e;

import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.ApiLevel;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.NdkMajorRevision;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.NdkPaths;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.StlImpl;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CrosstoolRelease;

/** Logic specific to Android NDK R12. */
public class NdkMajorRevisionR10 implements NdkMajorRevision {
  @Override
  public CrosstoolRelease crosstoolRelease(
      NdkPaths ndkPaths, StlImpl stlImpl, String hostPlatform) {
    return AndroidNdkCrosstoolsR10e.create(ndkPaths, stlImpl, hostPlatform);
  }

  @Override
  public ApiLevel apiLevel(EventHandler eventHandler, String name, String apiLevel) {
    return new ApiLevelR10e(eventHandler, name, apiLevel);
  }
}
