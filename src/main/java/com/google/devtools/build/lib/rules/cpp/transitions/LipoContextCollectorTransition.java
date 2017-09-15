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

package com.google.devtools.build.lib.rules.cpp.transitions;

import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.PatchTransition;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import com.google.devtools.build.lib.rules.cpp.FdoSupport;

/**
 * Configuration transition that enters "LIPO context collector" mode on a
 * "LIPO optimization"-enabled input configuration.
 *
 * <p>See {@link FdoSupport} for details.
 */
public class LipoContextCollectorTransition implements PatchTransition {
  public static final LipoContextCollectorTransition INSTANCE =
      new LipoContextCollectorTransition();

  private LipoContextCollectorTransition() {}

  @Override
  public BuildOptions apply(BuildOptions options) {
    // If this target and its transitive closure don't have C++ options, there's no
    // LIPO context to change.
    if (!options.contains(CppOptions.class)) {
      return options;
    }
    CppOptions cppOptions = options.get(CppOptions.class);
    if (!cppOptions.isLipoOptimization()) {
      return options;
    }
    BuildOptions collectorOptions = options.clone();
    collectorOptions.get(CppOptions.class).lipoConfigurationState =
        CppOptions.LipoConfigurationState.LIPO_CONTEXT_COLLECTOR;
    return collectorOptions;
  }
}
