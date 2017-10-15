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

package com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools;

import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CrosstoolRelease;

/** An interface to encapsulate logic that is specific to the Ndk major revision. */
public interface NdkMajorRevision {
  /**
   * Creates a CrosstoolRelease proto for the Android NDK. The crosstools are generated through code
   * rather than checked in as a flat file to reduce the amount of templateing needed (for
   * parameters like the release name and certain paths), to reduce duplication, and to make it
   * easier to support future versions of the NDK.
   */
  CrosstoolRelease crosstoolRelease(NdkPaths ndkPaths, StlImpl stlImpl, String hostPlatform);

  /**
   * Creates an {@code ApiLevel} that contains information about NDK-specific supported api levels
   * and api level to architecture mappings.
   */
  ApiLevel apiLevel(EventHandler eventHandler, String name, String apiLevel);
}
